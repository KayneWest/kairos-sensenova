#!/usr/bin/env python3
"""Decision-quality audit for the latent imagination planner.

Measures whether "thinking in latents" improves action selection, using
external proxies (future-MSE to the real future, persistence baselines)
instead of the scorer's own outputs. See
docs/ACTION_CONDITIONED_IMAGINATION_HANDOFF.md ("Next Agent: Highest-Value
Implementation") for the spec this implements.

Candidate families per context:
  controls:  true / zero / shuffle(roll) / time_shift / time_shift2 /
             time_perm / time_reverse action variants + zero plan token
  sampled:   randn    ~ N(0, 1)                (what training eval used)
             matched  ~ N(mu, sigma) fit to empirical true-plan stats
             bank     = true plans encoded from other contexts

Headline gates:
  1. true plan ranks above wrong-action controls (score space AND future-MSE space)
  2. argmax-score selection beats a random candidate under future-MSE
  3. selection quality improves with K
  4. effects hold on a held-out source (bridge, manifest weight 0)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DREAMER4_ROOT = REPO_ROOT / "dreamer4" / "dreamer4"
for item in (str(DREAMER4_ROOT), str(PROJECT_ROOT / "scripts")):
    if item not in sys.path:
        sys.path.insert(0, item)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_dynamics import load_frozen_tokenizer_from_pt_ckpt  # noqa: E402
from wm_dataset import WMDataset, collate_batch  # noqa: E402
from train_latent_imagination_planner import (  # noqa: E402
    LatentImaginationPlanner,
    PlannerConfig,
    encode_batch,
    discounted_returns,
    make_action_variant,
    masked_mse,
    resolve_path,
    seed_everything,
)

CONTROL_MODES = ["zero", "shuffle", "time_shift", "time_shift2", "time_perm", "time_reverse"]
SAMPLED_FAMILIES = ["randn", "matched", "bank"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decision-quality audit for the latent imagination planner.")
    p.add_argument("--ckpt", required=True, help="Planner checkpoint (.pt with planner/config/step).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--manifest-json", default="", help="Override manifest (default: from ckpt config).")
    p.add_argument("--tokenizer-ckpt", default="", help="Override tokenizer (default: from ckpt config).")
    p.add_argument("--source-names", default="", help="Comma list of manifest sources to audit (default: all with data on disk).")
    p.add_argument("--num-contexts", type=int, default=256, help="Held-out contexts per source.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-sampled", type=int, default=64, help="Sampled candidates per family (max K).")
    p.add_argument("--k-sweep", default="1,4,8,16,32,64")
    p.add_argument("--horizons", default="4,8,16")
    p.add_argument("--eval-chunk", type=int, default=16, help="Contexts per forward chunk in the audit pass.")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260706)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(resolve_path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = PlannerConfig(**ckpt["config"])
    step = int(ckpt.get("step", -1))
    horizons = sorted(int(h) for h in args.horizons.split(",") if h.strip())
    k_sweep = sorted(int(k) for k in args.k_sweep.split(",") if k.strip())
    max_h = max(horizons)
    if cfg.ctx_len + max_h > cfg.seq_len:
        raise ValueError(f"ctx_len + max horizon exceeds seq_len: {cfg.ctx_len}+{max_h} > {cfg.seq_len}")
    if max(k_sweep) > args.num_sampled:
        raise ValueError("max K in --k-sweep exceeds --num-sampled")

    tokenizer_ckpt = args.tokenizer_ckpt or cfg.tokenizer_ckpt
    encoder, _decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(resolve_path(tokenizer_ckpt)), device=device)
    patch = int(tok_args.get("patch", 8))
    packing_factor = int(tok_args.get("packing_factor", 2))
    n_latents = int(tok_args.get("n_latents", 16))
    n_spatial = n_latents // packing_factor
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    z_dim = n_spatial * d_bottleneck * packing_factor

    model = LatentImaginationPlanner(
        z_dim=z_dim,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.hidden_dim,
        plan_dim=cfg.plan_dim,
        horizon=cfg.horizon,
        plan_unit_norm=getattr(cfg, "plan_unit_norm", False),
        plan_step_conditioning=getattr(cfg, "plan_step_conditioning", False),
    ).to(device)
    model.load_state_dict(ckpt["planner"], strict=True)
    model.eval()

    manifest_path = resolve_path(args.manifest_json or cfg.manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested = [s.strip() for s in args.source_names.split(",") if s.strip()]
    sources = []
    for source in manifest.get("sources", []):
        name = source.get("name", "")
        if requested and name not in requested:
            continue
        raw = Path(str(source["raw"]))
        frames = Path(str(source["frames"]))
        if not raw.exists() or not frames.exists():
            print(json.dumps({"phase": "skip_source", "source": name, "reason": "missing data dirs"}), flush=True)
            continue
        sources.append({"name": name, "raw": str(raw), "frames": str(frames), "weight": int(source.get("weight", 1))})
    if not sources:
        raise ValueError("No auditable sources found.")

    meta = {
        "phase": "decision_quality_audit",
        "ckpt": str(args.ckpt),
        "ckpt_step": step,
        "z_dim": z_dim,
        "sources": [s["name"] for s in sources],
        "num_contexts": args.num_contexts,
        "num_sampled": args.num_sampled,
        "k_sweep": k_sweep,
        "horizons": horizons,
        "seed": args.seed,
    }
    print(json.dumps(meta, indent=2), flush=True)
    (out_dir / "audit_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    summaries = {}
    for source in sources:
        started = time.time()
        try:
            result = audit_source(
                source=source,
                model=model,
                encoder=encoder,
                patch=patch,
                n_spatial=n_spatial,
                packing_factor=packing_factor,
                cfg=cfg,
                args=args,
                horizons=horizons,
                k_sweep=k_sweep,
                device=device,
                out_dir=out_dir,
            )
        except Exception as exc:  # keep auditing remaining sources
            print(json.dumps({"phase": "source_error", "source": source["name"], "error": str(exc)}), flush=True)
            continue
        result["elapsed_s"] = time.time() - started
        summaries[source["name"]] = result
        print(json.dumps({"phase": "source_done", "source": source["name"], **headline(result)}, indent=2), flush=True)

    payload = {"meta": meta, "per_source": summaries, "gates": compute_gates(summaries)}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"phase": "audit_complete", "out_dir": str(out_dir), "gates": payload["gates"]}, indent=2), flush=True)
    return 0


@torch.no_grad()
def audit_source(*, source, model, encoder, patch, n_spatial, packing_factor, cfg, args, horizons, k_sweep, device, out_dir):
    name = source["name"]
    dataset = WMDataset(
        data_dir=[source["raw"]],
        frames_dir=[source["frames"]],
        seq_len=cfg.seq_len,
        img_size=cfg.img_size,
        action_dim=cfg.action_dim,
        raw_action_dim=cfg.raw_action_dim,
        tasks_json=str(resolve_path(cfg.tasks_json)) if cfg.tasks_json else "",
        tasks=None,
        strict_tasks=False,
        action_features=cfg.action_features,
        require_non_noop=cfg.require_non_noop,
        no_op_threshold=cfg.no_op_threshold,
        min_non_noop_steps=cfg.min_non_noop_steps,
        reward_filter_mode=cfg.reward_filter_mode,
        reward_signal_threshold=cfg.reward_signal_threshold,
        min_reward_signal_steps=cfg.min_reward_signal_steps,
        require_visual_delta=cfg.require_visual_delta,
        visual_delta_threshold=cfg.visual_delta_threshold,
        min_visual_delta_steps=cfg.min_visual_delta_steps,
        visual_delta_stride=cfg.visual_delta_stride,
        verbose=False,
    )
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_batch,
        generator=gen,
    )

    max_h = max(horizons)
    ctx = int(cfg.ctx_len)
    store = {"ctx_z": [], "ctx_actions": [], "future_z": [], "future_actions": [], "future_mask": [], "rewards": [], "emb_id": []}
    collected = 0
    for raw_batch in loader:
        batch = encode_batch(
            raw_batch=raw_batch,
            encoder=encoder,
            patch=patch,
            n_spatial=n_spatial,
            packing_factor=packing_factor,
            action_frame_offset=cfg.action_frame_offset,
            device=device,
        )
        z_flat = batch["z"].flatten(2)
        store["ctx_z"].append(z_flat[:, :ctx].cpu())
        store["future_z"].append(z_flat[:, ctx : ctx + max_h].cpu())
        store["ctx_actions"].append(batch["actions"][:, :ctx].cpu())
        store["future_actions"].append(batch["transition_actions"][:, ctx : ctx + cfg.horizon].cpu())
        store["future_mask"].append(batch["transition_mask"][:, ctx : ctx + cfg.horizon].cpu())
        store["rewards"].append(batch["rewards"][:, ctx : ctx + max_h].cpu())
        store["emb_id"].append(raw_batch["emb_id"].cpu())
        collected += z_flat.shape[0]
        if collected >= args.num_contexts:
            break
    data = {k: torch.cat(v, dim=0)[: args.num_contexts] for k, v in store.items()}
    n = data["ctx_z"].shape[0]
    if n < 8:
        raise ValueError(f"only {n} contexts collected for {name}")
    print(json.dumps({"phase": "contexts_collected", "source": name, "n": n, "dataset_windows": len(dataset)}), flush=True)

    # Pass 1: context encodings + true plans for the whole eval set (plan stats + bank).
    ctx_h_all, true_plan_all = [], []
    for i in range(0, n, args.eval_chunk):
        cz = data["ctx_z"][i : i + args.eval_chunk].to(device)
        ca = data["ctx_actions"][i : i + args.eval_chunk].to(device)
        fa = data["future_actions"][i : i + args.eval_chunk].to(device)
        h = model.encode_context(cz, ca)
        ctx_h_all.append(h.cpu())
        true_plan_all.append(model.encode_plan(h, fa).cpu())
    ctx_h_all = torch.cat(ctx_h_all, dim=0)
    true_plan_all = torch.cat(true_plan_all, dim=0)
    plan_mu = true_plan_all.mean(dim=0)
    plan_sigma = true_plan_all.std(dim=0).clamp_min(1e-6)
    plan_stats = {
        "plan_l2_mean": float(true_plan_all.pow(2).mean()),
        "plan_norm_mean": float(true_plan_all.norm(dim=-1).mean()),
        "randn_norm_expected": float(np.sqrt(model.plan_dim)),
        "plan_sigma_mean": float(plan_sigma.mean()),
    }

    K = int(args.num_sampled)
    rows = []
    jsonl_path = out_dir / f"contexts_{name}.jsonl"
    jsonl_file = jsonl_path.open("w", encoding="utf-8")
    rng = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    for i in range(0, n, args.eval_chunk):
        sl = slice(i, min(i + args.eval_chunk, n))
        B = sl.stop - sl.start
        ctx_z = data["ctx_z"][sl].to(device)
        ctx_h = ctx_h_all[sl].to(device)
        future_z = data["future_z"][sl].to(device)  # (B, max_h, z)
        future_actions = data["future_actions"][sl].to(device)
        future_mask = data["future_mask"][sl].to(device)
        rewards = data["rewards"][sl].to(device)
        true_plan = true_plan_all[sl].to(device)

        # --- control candidates (plan from action variants, exactly as in training) ---
        control_plans = {"true": true_plan, "zero_plan_token": torch.zeros_like(true_plan)}
        for mode in CONTROL_MODES:
            if mode == "shuffle":
                # roll guarantees a *different* context's actions (randperm can map to self)
                neg_actions, neg_mask = future_actions.roll(1, dims=0), future_mask.roll(1, dims=0)
            else:
                neg_actions, neg_mask = make_action_variant(future_actions, future_mask, mode)
            control_plans[mode] = model.encode_plan(ctx_h, neg_actions * neg_mask)

        # --- sampled candidate families ---
        sampled_plans = {}
        sampled_plans["randn"] = torch.randn((B, K, model.plan_dim), generator=rng).to(device)
        sampled_plans["matched"] = (plan_mu[None, None] + plan_sigma[None, None] * torch.randn((B, K, model.plan_dim), generator=rng)).to(device)
        bank_idx = torch.randint(0, n - 1, (B, K), generator=rng)
        row_ids = torch.arange(sl.start, sl.stop)[:, None]
        bank_idx = torch.where(bank_idx >= row_ids, bank_idx + 1, bank_idx)  # exclude self
        sampled_plans["bank"] = true_plan_all[bank_idx.clamp(0, n - 1)].to(device)

        # --- roll out every candidate once to max_h, score on trained horizon ---
        def rollout(plans_flat, reps):
            czr = ctx_z[:, None].expand(B, reps, *ctx_z.shape[1:]).reshape(B * reps, *ctx_z.shape[1:])
            chr_ = ctx_h[:, None].expand(B, reps, ctx_h.shape[-1]).reshape(B * reps, -1)
            fut = model.propose_future(czr, chr_, plans_flat, horizon=max_h)
            score = model.score_future(chr_, fut[:, : cfg.horizon], plans_flat)
            return fut.view(B, reps, max_h, -1), score.view(B, reps)

        ctrl_names = list(control_plans.keys())
        ctrl_stack = torch.stack([control_plans[c] for c in ctrl_names], dim=1)  # (B, C, plan)
        ctrl_fut, ctrl_score = rollout(ctrl_stack.reshape(B * len(ctrl_names), -1), len(ctrl_names))
        fam_fut, fam_score = {}, {}
        for fam in SAMPLED_FAMILIES:
            fam_fut[fam], fam_score[fam] = rollout(sampled_plans[fam].reshape(B * K, -1), K)

        def mse_at(fut, h):  # fut: (B, R, max_h, z) -> (B, R)
            return (fut[:, :, :h] - future_z[:, None, :h]).pow(2).mean(dim=(2, 3))

        persist = ctx_z[:, -1][:, None, None, :].expand(B, 1, max_h, future_z.shape[-1])
        persist_mse = {h: mse_at(persist, h)[:, 0] for h in horizons}
        ctrl_mse = {h: mse_at(ctrl_fut, h) for h in horizons}
        fam_mse = {fam: {h: mse_at(fam_fut[fam], h) for h in horizons} for fam in SAMPLED_FAMILIES}

        h0 = cfg.horizon  # selection horizon
        true_idx = ctrl_names.index("true")
        true_mse = {h: ctrl_mse[h][:, true_idx] for h in horizons}
        true_score = ctrl_score[:, true_idx]
        returns = discounted_returns(rewards[:, :h0], gamma=cfg.gamma)

        # inverse round-trip on the real future with the true plan
        inv_pred = model.inverse_actions(ctx_h, future_z[:, :h0], true_plan)
        inv_mse = ((inv_pred - future_actions).pow(2) * future_mask).sum(dim=(1, 2)) / future_mask.sum(dim=(1, 2)).clamp_min(1.0)

        for b in range(B):
            row = {
                "context_id": int(sl.start + b),
                "source": name,
                "emb_id": int(data["emb_id"][sl.start + b]),
                "return_h": float(returns[b]),
                "true_score": float(true_score[b]),
                "inverse_mse": float(inv_mse[b]),
                "action_norm": float(future_actions[b].norm(dim=-1).mean()),
            }
            for h in horizons:
                row[f"true_mse_h{h}"] = float(true_mse[h][b])
                row[f"persist_mse_h{h}"] = float(persist_mse[h][b])
            for ci, cname in enumerate(ctrl_names):
                if cname == "true":
                    continue
                row[f"score_margin_{cname}"] = float(true_score[b] - ctrl_score[b, ci])
                row[f"mse_ratio_{cname}"] = float(ctrl_mse[h0][b, ci] / max(float(true_mse[h0][b]), 1e-12))
            # rank of true plan (by score) among true + controls + matched sampled
            pool_scores = torch.cat([ctrl_score[b], fam_score["matched"][b]])
            row["oracle_rank_pct"] = float((pool_scores < true_score[b]).float().mean())
            row["oracle_top1"] = bool((true_score[b] >= pool_scores.max()).item())
            for fam in SAMPLED_FAMILIES:
                s, m = fam_score[fam][b], fam_mse[fam][h0][b]
                c = corr(s, -m)
                row[f"fidelity_corr_{fam}"] = c
                for k in k_sweep:
                    sel = int(s[:k].argmax())
                    row[f"sel_mse_{fam}_k{k}"] = float(m[sel])
                    row[f"rand_mse_{fam}_k{k}"] = float(m[:k].mean())
                    row[f"best_mse_{fam}_k{k}"] = float(m[:k].min())
                for h in horizons:
                    sel = int(fam_score[fam][b].argmax())
                    row[f"sel_mse_{fam}_h{h}"] = float(fam_mse[fam][h][b, sel])
                    row[f"rand_mse_{fam}_h{h}"] = float(fam_mse[fam][h][b].mean())
            rows.append(row)
            jsonl_file.write(json.dumps(row, sort_keys=True) + "\n")
    jsonl_file.close()

    return summarize_source(rows, horizons=horizons, k_sweep=k_sweep, h0=cfg.horizon, plan_stats=plan_stats, bootstrap=args.bootstrap, seed=args.seed)


def corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten() - x.float().mean()
    y = y.float().flatten() - y.float().mean()
    denom = float(x.norm() * y.norm())
    if denom <= 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def boot_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_source(rows, *, horizons, k_sweep, h0, plan_stats, bootstrap, seed):
    arr = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0] if isinstance(rows[0][k], (int, float, bool)) and k not in ("context_id", "emb_id")}
    out = {"n_contexts": len(rows), "plan_stats": plan_stats}

    def stat(key, values=None):
        v = arr[key] if values is None else values
        lo, hi = boot_ci(v, bootstrap, seed)
        return {"mean": float(v.mean()), "ci_lo": lo, "ci_hi": hi}

    out["oracle_top1_rate"] = stat("oracle_top1")
    out["oracle_rank_pct"] = stat("oracle_rank_pct")
    out["score_return_corr"] = float(np.corrcoef(arr["true_score"], arr["return_h"])[0, 1]) if arr["return_h"].std() > 1e-9 else 0.0
    out["inverse_mse"] = stat("inverse_mse")
    out["controls"] = {}
    for key in sorted(arr):
        if key.startswith("score_margin_"):
            cname = key[len("score_margin_"):]
            out["controls"][cname] = {
                "score_margin": stat(key),
                "mse_ratio": stat(f"mse_ratio_{cname}"),
            }
    out["horizon"] = {}
    for h in horizons:
        out["horizon"][f"h{h}"] = {
            "true_over_persistence": float((arr[f"true_mse_h{h}"] / np.maximum(arr[f"persist_mse_h{h}"], 1e-12)).mean()),
            "true_mse": float(arr[f"true_mse_h{h}"].mean()),
            "persist_mse": float(arr[f"persist_mse_h{h}"].mean()),
        }
    out["selection"] = {}
    for fam in SAMPLED_FAMILIES:
        fam_out = {"fidelity_corr": stat(f"fidelity_corr_{fam}")}
        for k in k_sweep:
            delta = arr[f"sel_mse_{fam}_k{k}"] - arr[f"rand_mse_{fam}_k{k}"]
            denom = np.maximum(arr[f"rand_mse_{fam}_k{k}"] - arr[f"best_mse_{fam}_k{k}"], 1e-12)
            regret = (arr[f"sel_mse_{fam}_k{k}"] - arr[f"best_mse_{fam}_k{k}"]) / denom
            fam_out[f"k{k}"] = {
                "sel_minus_rand_mse": stat(None, values=delta),
                "sel_beats_rand_frac": float((delta < 0).mean()),
                "regret": float(regret.mean()) if k > 1 else 1.0,
            }
        for h in horizons:
            delta_h = arr[f"sel_mse_{fam}_h{h}"] - arr[f"rand_mse_{fam}_h{h}"]
            fam_out[f"h{h}_sel_minus_rand_mse"] = float(delta_h.mean())
        out["selection"][fam] = fam_out
    return out


def headline(result) -> dict:
    kmax = max(int(k[1:]) for k in result["selection"]["bank"] if k.startswith("k"))
    return {
        "oracle_top1_rate": result["oracle_top1_rate"]["mean"],
        "oracle_rank_pct": result["oracle_rank_pct"]["mean"],
        "score_return_corr": result["score_return_corr"],
        "shuffle_mse_ratio": result["controls"].get("shuffle", {}).get("mse_ratio", {}).get("mean"),
        "zero_mse_ratio": result["controls"].get("zero", {}).get("mse_ratio", {}).get("mean"),
        "time_shift_mse_ratio": result["controls"].get("time_shift", {}).get("mse_ratio", {}).get("mean"),
        f"bank_sel_minus_rand_k{kmax}": result["selection"]["bank"][f"k{kmax}"]["sel_minus_rand_mse"]["mean"],
        f"bank_sel_beats_rand_frac_k{kmax}": result["selection"]["bank"][f"k{kmax}"]["sel_beats_rand_frac"],
        "plan_norm_mean": result["plan_stats"]["plan_norm_mean"],
    }


def compute_gates(summaries) -> dict:
    """Pass/fail rollup of the handoff's decision-quality gates."""
    gates = {}
    for name, res in summaries.items():
        sel = res["selection"]
        ks = sorted(int(k[1:]) for k in sel["bank"] if k.startswith("k"))
        kmax = ks[-1]
        k_deltas = [sel["bank"][f"k{k}"]["sel_minus_rand_mse"]["mean"] for k in ks]
        gates[name] = {
            "true_beats_zero": res["controls"].get("zero", {}).get("mse_ratio", {}).get("mean", 0) > 1.0
            and res["controls"].get("zero", {}).get("score_margin", {}).get("ci_lo", -1) > 0,
            "true_beats_shuffle": res["controls"].get("shuffle", {}).get("mse_ratio", {}).get("mean", 0) > 1.0
            and res["controls"].get("shuffle", {}).get("score_margin", {}).get("ci_lo", -1) > 0,
            "true_beats_time_shift": res["controls"].get("time_shift", {}).get("mse_ratio", {}).get("mean", 0) > 1.0,
            "selection_beats_random_ci": sel["bank"][f"k{kmax}"]["sel_minus_rand_mse"]["ci_hi"] < 0,
            "selection_improves_with_k": k_deltas[-1] < k_deltas[0],
            "scorer_fidelity_positive": sel["bank"]["fidelity_corr"]["ci_lo"] > 0,
        }
        gates[name]["all"] = all(gates[name].values())
    return gates


if __name__ == "__main__":
    raise SystemExit(main())
