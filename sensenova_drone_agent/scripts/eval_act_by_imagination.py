#!/usr/bin/env python3
"""Act-by-imagination evaluation: do actions emitted from selected imagined
futures beat actions emitted without thinking?

See docs/ACT_BY_IMAGINATION_HARNESS.md for the protocol, controls, and gates.
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
from torch.utils.data import DataLoader

from train_dynamics import load_frozen_tokenizer_from_pt_ckpt  # noqa: E402
from wm_dataset import WMDataset, collate_batch  # noqa: E402
from train_latent_imagination_planner import (  # noqa: E402
    LatentImaginationPlanner,
    PlannerConfig,
    encode_batch,
    discounted_returns,
    make_action_variant,
    resolve_path,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Act-by-imagination evaluation.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--manifest-json", default="")
    p.add_argument("--tokenizer-ckpt", default="")
    p.add_argument("--source-names", default="soar_native_v2,dreamer4_hf_expert,hf_robot_bridge_orig_lerobot_dreamer4")
    p.add_argument("--num-contexts", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-bank", type=int, default=64)
    p.add_argument("--k-sweep", default="1,4,8,16,32,64")
    p.add_argument("--eval-chunk", type=int, default=16)
    p.add_argument("--inverse-plan-mode", default="candidate", choices=["candidate", "zero"],
                   help="Plan input for inverse dynamics at act time. 'zero' = plan-free decoding (for checkpoints trained with inverse plan dropout).")
    p.add_argument("--score-plan-mode", default="plan", choices=["plan", "zero"],
                   help="Plan input when scoring candidates. 'zero' for checkpoints trained with score plan dropout.")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260708)
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
    k_sweep = sorted(int(k) for k in args.k_sweep.split(",") if k.strip())
    if max(k_sweep) > args.num_bank:
        raise ValueError("max K exceeds --num-bank")

    tokenizer_ckpt = args.tokenizer_ckpt or cfg.tokenizer_ckpt
    encoder, _decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(resolve_path(tokenizer_ckpt)), device=device)
    patch = int(tok_args.get("patch", 8))
    packing_factor = int(tok_args.get("packing_factor", 2))
    n_spatial = int(tok_args.get("n_latents", 16)) // packing_factor
    z_dim = n_spatial * int(tok_args.get("d_bottleneck", 32)) * packing_factor

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

    manifest = json.loads(resolve_path(args.manifest_json or cfg.manifest_json).read_text(encoding="utf-8"))
    requested = [s.strip() for s in args.source_names.split(",") if s.strip()]
    meta = {
        "phase": "act_by_imagination",
        "ckpt": str(args.ckpt),
        "ckpt_step": int(ckpt.get("step", -1)),
        "num_contexts": args.num_contexts,
        "num_bank": args.num_bank,
        "k_sweep": k_sweep,
        "seed": args.seed,
        "inverse_plan_mode": args.inverse_plan_mode,
    }
    print(json.dumps(meta, indent=2), flush=True)
    (out_dir / "act_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    summaries = {}
    for source in manifest.get("sources", []):
        name = source.get("name", "")
        if name not in requested:
            continue
        if not Path(str(source["raw"])).exists():
            print(json.dumps({"phase": "skip_source", "source": name}), flush=True)
            continue
        started = time.time()
        result = eval_source(
            source=source, model=model, encoder=encoder, patch=patch, n_spatial=n_spatial,
            packing_factor=packing_factor, cfg=cfg, args=args, k_sweep=k_sweep, device=device, out_dir=out_dir,
        )
        result["elapsed_s"] = time.time() - started
        summaries[name] = result
        print(json.dumps({"phase": "source_done", "source": name, **headline(result, k_sweep)}, indent=2), flush=True)

    payload = {"meta": meta, "per_source": summaries, "gates": compute_gates(summaries, k_sweep)}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"phase": "act_eval_complete", "out_dir": str(out_dir), "gates": payload["gates"]}, indent=2), flush=True)
    return 0


@torch.no_grad()
def eval_source(*, source, model, encoder, patch, n_spatial, packing_factor, cfg, args, k_sweep, device, out_dir):
    name = source["name"]
    dataset = WMDataset(
        data_dir=[str(source["raw"])],
        frames_dir=[str(source["frames"])],
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
        verbose=False,
    )
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        drop_last=True, collate_fn=collate_batch, generator=gen)

    ctx = int(cfg.ctx_len)
    h = int(cfg.horizon)
    store = {"ctx_z": [], "ctx_actions": [], "future_z": [], "future_actions": [], "future_mask": [], "rewards": [], "last_act": []}
    collected = 0
    for raw_batch in loader:
        batch = encode_batch(
            raw_batch=raw_batch, encoder=encoder, patch=patch, n_spatial=n_spatial,
            packing_factor=packing_factor, action_frame_offset=cfg.action_frame_offset, device=device,
        )
        z_flat = batch["z"].flatten(2)
        store["ctx_z"].append(z_flat[:, :ctx].cpu())
        store["future_z"].append(z_flat[:, ctx : ctx + h].cpu())
        store["ctx_actions"].append(batch["actions"][:, :ctx].cpu())
        store["future_actions"].append(batch["transition_actions"][:, ctx : ctx + h].cpu())
        store["future_mask"].append(batch["transition_mask"][:, ctx : ctx + h].cpu())
        store["rewards"].append(batch["rewards"][:, ctx : ctx + h].cpu())
        store["last_act"].append(batch["transition_actions"][:, ctx - 1].cpu())
        collected += z_flat.shape[0]
        if collected >= args.num_contexts:
            break
    data = {k: torch.cat(v, dim=0)[: args.num_contexts] for k, v in store.items()}
    n = data["ctx_z"].shape[0]
    if n < 8:
        raise ValueError(f"only {n} contexts for {name}")
    print(json.dumps({"phase": "contexts_collected", "source": name, "n": n}), flush=True)

    # dataset-level blind priors
    mask_all = data["future_mask"]
    mean_action = (data["future_actions"] * mask_all).sum(dim=0) / mask_all.sum(dim=0).clamp_min(1.0)  # (h, A)

    # pass 1: context encodings and true plans (bank source)
    ctx_h_all, true_plan_all = [], []
    for i in range(0, n, args.eval_chunk):
        cz = data["ctx_z"][i : i + args.eval_chunk].to(device)
        ca = data["ctx_actions"][i : i + args.eval_chunk].to(device)
        fa = data["future_actions"][i : i + args.eval_chunk].to(device)
        hh = model.encode_context(cz, ca)
        ctx_h_all.append(hh.cpu())
        true_plan_all.append(model.encode_plan(hh, fa).cpu())
    ctx_h_all = torch.cat(ctx_h_all, dim=0)
    true_plan_all = torch.cat(true_plan_all, dim=0)

    K = int(args.num_bank)
    rng = torch.Generator(device="cpu").manual_seed(args.seed + 3)
    rows = []
    jsonl = (out_dir / f"act_contexts_{name}.jsonl").open("w", encoding="utf-8")

    def amse(pred, target, mask):  # (..., h, A) -> (...)
        return ((pred - target).pow(2) * mask).sum(dim=(-1, -2)) / mask.sum(dim=(-1, -2)).clamp_min(1.0)

    def acos(pred, target, mask):
        p = (pred * mask).flatten(-2)
        t = (target * mask).flatten(-2)
        denom = (p.norm(dim=-1) * t.norm(dim=-1)).clamp_min(1e-9)
        return (p * t).sum(dim=-1) / denom

    for i in range(0, n, args.eval_chunk):
        sl = slice(i, min(i + args.eval_chunk, n))
        B = sl.stop - sl.start
        ctx_z = data["ctx_z"][sl].to(device)
        ctx_h = ctx_h_all[sl].to(device)
        future_z = data["future_z"][sl].to(device)
        a_true = data["future_actions"][sl].to(device)
        a_mask = data["future_mask"][sl].to(device)
        rewards = data["rewards"][sl].to(device)
        true_plan = true_plan_all[sl].to(device)

        bank_idx = torch.randint(0, n - 1, (B, K), generator=rng)
        row_ids = torch.arange(sl.start, sl.stop)[:, None]
        bank_idx = torch.where(bank_idx >= row_ids, bank_idx + 1, bank_idx).clamp(0, n - 1)
        bank_plans = true_plan_all[bank_idx].to(device)  # (B, K, plan)

        zero_a, zero_m = make_action_variant(a_true, a_mask, "zero")
        named_plans = torch.stack([true_plan, model.encode_plan(ctx_h, zero_a * zero_m)], dim=1)  # true, zero

        def rollout_and_act(plans_flat, reps):
            czr = ctx_z[:, None].expand(B, reps, *ctx_z.shape[1:]).reshape(B * reps, *ctx_z.shape[1:])
            chr_ = ctx_h[:, None].expand(B, reps, ctx_h.shape[-1]).reshape(B * reps, -1)
            fut = model.propose_future(czr, chr_, plans_flat, horizon=h)
            score_plan = torch.zeros_like(plans_flat) if args.score_plan_mode == "zero" else plans_flat
            score = model.score_future(chr_, fut, score_plan)
            inv_plan = torch.zeros_like(plans_flat) if args.inverse_plan_mode == "zero" else plans_flat
            act = model.inverse_actions(chr_, fut, inv_plan)
            return fut.view(B, reps, h, -1), score.view(B, reps), act.view(B, reps, h, -1)

        _bank_fut, bank_score, bank_act = rollout_and_act(bank_plans.reshape(B * K, -1), K)
        _named_fut, _named_score, named_act = rollout_and_act(named_plans.reshape(B * 2, -1), 2)
        oracle_act = model.inverse_actions(ctx_h, future_z, true_plan)  # real future, ceiling

        bank_amse = amse(bank_act, a_true[:, None], a_mask[:, None])  # (B, K)
        returns = discounted_returns(rewards, gamma=cfg.gamma)
        mean_a = mean_action.to(device)[None].expand(B, -1, -1)
        last_a = data["last_act"][sl].to(device)[:, None, :].expand(B, h, -1)

        rand_pick = torch.randint(0, K, (B,), generator=rng)
        for b in range(B):
            sel_full = int(bank_score[b].argmax())
            row = {
                "context_id": int(sl.start + b),
                "source": name,
                "return_h": float(returns[b]),
                "amse_true_plan": float(amse(named_act[b, 0], a_true[b], a_mask[b])),
                "amse_zero_plan": float(amse(named_act[b, 1], a_true[b], a_mask[b])),
                "amse_oracle_real_future": float(amse(oracle_act[b], a_true[b], a_mask[b])),
                "amse_mean_action": float(amse(mean_a[b], a_true[b], a_mask[b])),
                "amse_repeat_last": float(amse(last_a[b], a_true[b], a_mask[b])),
                "amse_worst": float(bank_amse[b, int(bank_score[b].argmin())]),
                "acos_selected": float(acos(bank_act[b, sel_full], a_true[b], a_mask[b])),
                "acos_random": float(acos(bank_act[b, int(rand_pick[b])], a_true[b], a_mask[b])),
            }
            for k in k_sweep:
                sel = int(bank_score[b, :k].argmax())
                row[f"amse_sel_k{k}"] = float(bank_amse[b, sel])
                row[f"amse_rand_k{k}"] = float(bank_amse[b, :k].mean())
                row[f"amse_best_k{k}"] = float(bank_amse[b, :k].min())
            rows.append(row)
            jsonl.write(json.dumps(row, sort_keys=True) + "\n")
    jsonl.close()
    return summarize(rows, k_sweep=k_sweep, bootstrap=args.bootstrap, seed=args.seed)


def boot_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows, *, k_sweep, bootstrap, seed):
    arr = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0] if k not in ("source", "context_id")}
    out = {"n_contexts": len(rows)}

    def stat(values):
        lo, hi = boot_ci(values, bootstrap, seed)
        return {"mean": float(values.mean()), "ci_lo": lo, "ci_hi": hi}

    kmax = max(k_sweep)
    for key in ("amse_true_plan", "amse_zero_plan", "amse_oracle_real_future", "amse_mean_action",
                "amse_repeat_last", "amse_worst", "acos_selected", "acos_random"):
        out[key] = stat(arr[key])
    out["k_sweep"] = {}
    for k in k_sweep:
        delta = arr[f"amse_sel_k{k}"] - arr[f"amse_rand_k{k}"]
        out["k_sweep"][f"k{k}"] = {
            "amse_selected": stat(arr[f"amse_sel_k{k}"]),
            "sel_minus_rand": stat(delta),
            "sel_beats_rand_frac": float((delta < 0).mean()),
        }
    sel = arr[f"amse_sel_k{kmax}"]
    out["sel_vs_zero_plan_delta"] = stat(sel - arr["amse_zero_plan"])
    out["sel_vs_mean_action_delta"] = stat(sel - arr["amse_mean_action"])
    out["sel_vs_repeat_last_delta"] = stat(sel - arr["amse_repeat_last"])
    pos = arr["return_h"] > 1e-6
    if pos.sum() >= 16:
        d = (arr[f"amse_sel_k{kmax}"] - arr[f"amse_rand_k{kmax}"])[pos]
        out["positive_return_subset"] = {
            "n": int(pos.sum()),
            "sel_minus_rand": stat(d),
            "sel_beats_rand_frac": float((d < 0).mean()),
        }
    return out


def headline(result, k_sweep):
    kmax = max(k_sweep)
    kk = result["k_sweep"][f"k{kmax}"]
    return {
        "amse_selected": kk["amse_selected"]["mean"],
        "sel_minus_rand": kk["sel_minus_rand"]["mean"],
        "sel_beats_rand_frac": kk["sel_beats_rand_frac"],
        "amse_zero_plan": result["amse_zero_plan"]["mean"],
        "amse_mean_action": result["amse_mean_action"]["mean"],
        "amse_oracle_real_future": result["amse_oracle_real_future"]["mean"],
        "acos_selected": result["acos_selected"]["mean"],
    }


def compute_gates(summaries, k_sweep):
    kmax = max(k_sweep)
    gates = {}
    for name, res in summaries.items():
        ks = res["k_sweep"]
        deltas = [ks[f"k{k}"]["sel_minus_rand"]["mean"] for k in k_sweep]
        gates[name] = {
            "act_sel_beats_rand_ci": ks[f"k{kmax}"]["sel_minus_rand"]["ci_hi"] < 0,
            "act_improves_with_k": deltas[-1] < deltas[0],
            "act_sel_beats_zero_plan": res["sel_vs_zero_plan_delta"]["ci_hi"] < 0,
            "act_beats_mean_action": res["sel_vs_mean_action_delta"]["ci_hi"] < 0,
            "act_beats_repeat_last": res["sel_vs_repeat_last_delta"]["ci_hi"] < 0,
        }
        gates[name]["all"] = all(gates[name].values())
    return gates


if __name__ == "__main__":
    raise SystemExit(main())
