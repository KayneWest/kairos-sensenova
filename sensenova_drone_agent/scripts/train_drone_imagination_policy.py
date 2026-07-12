#!/usr/bin/env python3
"""PMPO-style policy trained inside the drone planner's imagination.

Replaces act-time argmax over a seed-fragile value head (the two-seed
reversal finding) with a trained policy: sample action chunks from the
policy, imagine each future with the frozen planner, score with the
plan-free value head, and update the policy with sign-of-advantage weighted
log-probs plus a KL to the planner's built-in BC prior. Log-probs are taken
of DETACHED sampled actions (the historical PMPO score-function fix; for
discrete sampling this is inherent). The planner is fully frozen.

Saves {"head", "config", "step"} loadable by the closed-loop eval's
--policy-head flag.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
from train_drone_bc_chunk_head import BCChunkHead  # noqa: E402
from train_latent_imagination_planner import (  # noqa: E402
    LatentImaginationPlanner,
    PlannerConfig,
    encode_batch,
    resolve_path,
    seed_everything,
)

NUM_ACTIONS = 9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a PMPO imagination policy on a frozen drone planner.")
    p.add_argument("--planner-ckpt", required=True, help="Planner trained with --bc-head-weight (prior lives inside).")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--tasks-json", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-samples", type=int, default=16, help="Chunks sampled from the policy per context.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--kl-weight", type=float, default=0.3)
    p.add_argument("--neg-weight", type=float, default=0.5, help="PMPO weight on negative-advantage samples.")
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(resolve_path(args.planner_ckpt), map_location="cpu", weights_only=False)
    cfg = PlannerConfig(**ckpt["config"])
    encoder, _dec, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(resolve_path(cfg.tokenizer_ckpt)), device=device)
    patch = int(tok_args.get("patch", 8))
    pk = int(tok_args.get("packing_factor", 2))
    ns = int(tok_args.get("n_latents", 16)) // pk
    z_dim = ns * int(tok_args.get("d_bottleneck", 32)) * pk
    model = LatentImaginationPlanner(
        z_dim=z_dim, action_dim=cfg.action_dim, hidden_dim=cfg.hidden_dim, plan_dim=cfg.plan_dim,
        horizon=cfg.horizon, plan_unit_norm=cfg.plan_unit_norm,
        plan_step_conditioning=getattr(cfg, "plan_step_conditioning", False),
    )
    if getattr(cfg, "bc_head_weight", 0.0) > 0:
        model.enable_bc_head()
    model = model.to(device)
    model.load_state_dict(ckpt["planner"], strict=True)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    if model.bc_head is None:
        raise ValueError("Planner has no built-in BC head; train with --bc-head-weight first.")

    policy = BCChunkHead(ctx_dim=cfg.hidden_dim, horizon=cfg.horizon, hidden_dim=1024).to(device)
    # warm-start the policy from the BC prior's behavior via distillation
    opt = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    dataset = WMDataset(
        data_dir=[str(resolve_path(args.data_dir))], frames_dir=[str(resolve_path(args.frames_dir))],
        seq_len=cfg.seq_len, img_size=cfg.img_size, action_dim=cfg.action_dim, raw_action_dim=cfg.raw_action_dim,
        tasks_json=str(resolve_path(args.tasks_json)) if args.tasks_json else "",
        tasks=None, strict_tasks=False, action_features=cfg.action_features, verbose=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        drop_last=True, collate_fn=collate_batch, persistent_workers=args.num_workers > 0)
    print(json.dumps({"phase": "pmpo_policy", "windows": len(dataset), "steps": args.steps}), flush=True)

    metrics_path = out.with_suffix(".metrics.jsonl")
    started = time.time()
    iterator = iter(loader)
    K = int(args.num_samples)
    h = cfg.horizon
    for step in range(1, args.steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        with torch.no_grad():
            batch = encode_batch(raw_batch=raw, encoder=encoder, patch=patch, n_spatial=ns,
                                 packing_factor=pk, action_frame_offset=cfg.action_frame_offset, device=device)
            zf = batch["z"].flatten(2)
            ctx_z = zf[:, : cfg.ctx_len]
            ctx_h = model.encode_context(ctx_z, batch["actions"][:, : cfg.ctx_len])
            prior_logits = model.bc_logits(ctx_h)  # (B, h, A)
        B = ctx_h.shape[0]
        logits = policy(ctx_h)  # (B, h, A) — grad flows to policy only
        logp = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            samples = torch.multinomial(probs.reshape(B * h, NUM_ACTIONS), K, replacement=True)
            samples = samples.view(B, h, K).permute(0, 2, 1).contiguous()  # (B, K, h)
            onehot = F.one_hot(samples, NUM_ACTIONS).float().view(B * K, h, NUM_ACTIONS)
            chr_ = ctx_h[:, None].expand(B, K, -1).reshape(B * K, -1)
            czr = ctx_z[:, None].expand(B, K, *ctx_z.shape[1:]).reshape(B * K, *ctx_z.shape[1:])
            plans = model.encode_plan(chr_, onehot)
            futures = model.propose_future(czr, chr_, plans, horizon=h)
            scores = model.score_future(chr_, futures, torch.zeros_like(plans)).view(B, K)
            adv = scores - scores.mean(dim=1, keepdim=True)
            w = torch.where(adv > 0, torch.ones_like(adv), -float(args.neg_weight) * torch.ones_like(adv))
        # log-prob of each (detached) sampled chunk under the policy
        lp = logp[:, None].expand(B, K, h, NUM_ACTIONS).gather(3, samples[:, :, :, None]).squeeze(-1).sum(-1)  # (B, K)
        pmpo_loss = -(w * lp).mean()
        kl = F.kl_div(logp.reshape(-1, NUM_ACTIONS),
                      F.log_softmax(prior_logits, dim=-1).reshape(-1, NUM_ACTIONS),
                      log_target=True, reduction="batchmean")
        loss = pmpo_loss + args.kl_weight * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                greedy = logits.argmax(-1)
                g_one = F.one_hot(greedy, NUM_ACTIONS).float()
                gplan = model.encode_plan(ctx_h, g_one)
                gfut = model.propose_future(ctx_z, ctx_h, gplan, horizon=h)
                gscore = model.score_future(ctx_h, gfut, torch.zeros_like(gplan)).mean()
                pr_greedy = prior_logits.argmax(-1)
                p_one = F.one_hot(pr_greedy, NUM_ACTIONS).float()
                pplan = model.encode_plan(ctx_h, p_one)
                pfut = model.propose_future(ctx_z, ctx_h, pplan, horizon=h)
                pscore = model.score_future(ctx_h, pfut, torch.zeros_like(pplan)).mean()
            row = {"step": step, "pmpo_loss": float(pmpo_loss.detach()), "kl": float(kl.detach()),
                   "policy_greedy_value": float(gscore), "prior_greedy_value": float(pscore),
                   "value_gain_vs_prior": float(gscore - pscore), "elapsed_s": time.time() - started}
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    torch.save({"head": policy.state_dict(),
                "config": {"ctx_dim": cfg.hidden_dim, "horizon": cfg.horizon, "hidden_dim": 1024},
                "step": args.steps}, out)
    print(json.dumps({"phase": "done", "out": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
