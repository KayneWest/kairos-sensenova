#!/usr/bin/env python3
"""Train a BC action-chunk head on top of the frozen latent planner encoder.

BC-anchored thinking (docs/ACT_BY_IMAGINATION_HARNESS.md, next-directions):
at act time, K chunks sampled from this head restrict the imagination search
to behavior support, curing offline-MPC model exploitation. The head maps
the planner's frozen context encoding to per-step action logits.

Trained on expert/eps-expert-only data with cross-entropy over the horizon.
Saves {"head", "planner_ckpt", "config", "step"}.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_dynamics import load_frozen_tokenizer_from_pt_ckpt  # noqa: E402
from wm_dataset import WMDataset, collate_batch  # noqa: E402
from train_latent_imagination_planner import (  # noqa: E402
    MLP,
    LatentImaginationPlanner,
    PlannerConfig,
    encode_batch,
    resolve_path,
    seed_everything,
)

NUM_ACTIONS = 9


class BCChunkHead(nn.Module):
    def __init__(self, *, ctx_dim: int, horizon: int, hidden_dim: int = 512):
        super().__init__()
        self.horizon = int(horizon)
        self.net = MLP(ctx_dim, self.horizon * NUM_ACTIONS, hidden_dim, depth=2)

    def forward(self, ctx_h: torch.Tensor) -> torch.Tensor:  # (B, h, A) logits
        return self.net(ctx_h).view(ctx_h.shape[0], self.horizon, NUM_ACTIONS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BC chunk head on frozen planner encoder.")
    p.add_argument("--planner-ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--tasks-json", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260712)
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
    packing_factor = int(tok_args.get("packing_factor", 2))
    n_spatial = int(tok_args.get("n_latents", 16)) // packing_factor
    z_dim = n_spatial * int(tok_args.get("d_bottleneck", 32)) * packing_factor
    model = LatentImaginationPlanner(
        z_dim=z_dim, action_dim=cfg.action_dim, hidden_dim=cfg.hidden_dim, plan_dim=cfg.plan_dim,
        horizon=cfg.horizon, plan_unit_norm=cfg.plan_unit_norm,
        plan_step_conditioning=getattr(cfg, "plan_step_conditioning", False),
    ).to(device)
    model.load_state_dict(ckpt["planner"], strict=True)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    head = BCChunkHead(ctx_dim=cfg.hidden_dim, horizon=cfg.horizon, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    dataset = WMDataset(
        data_dir=[str(resolve_path(args.data_dir))],
        frames_dir=[str(resolve_path(args.frames_dir))],
        seq_len=cfg.seq_len, img_size=cfg.img_size, action_dim=cfg.action_dim,
        raw_action_dim=cfg.raw_action_dim,
        tasks_json=str(resolve_path(args.tasks_json)) if args.tasks_json else "",
        tasks=None, strict_tasks=False, action_features=cfg.action_features, verbose=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        drop_last=True, collate_fn=collate_batch, persistent_workers=args.num_workers > 0)
    print(json.dumps({"phase": "bc_head_train", "windows": len(dataset), "steps": args.steps}), flush=True)

    started = time.time()
    iterator = iter(loader)
    step, correct, total = 0, 0, 0
    while step < args.steps:
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        step += 1
        with torch.no_grad():
            batch = encode_batch(raw_batch=raw, encoder=encoder, patch=patch, n_spatial=n_spatial,
                                 packing_factor=packing_factor, action_frame_offset=cfg.action_frame_offset, device=device)
            zf = batch["z"].flatten(2)
            ctx = cfg.ctx_len
            ctx_h = model.encode_context(zf[:, :ctx], batch["actions"][:, :ctx])
            target = batch["transition_actions"][:, ctx : ctx + cfg.horizon].argmax(-1)  # (B, h)
        logits = head(ctx_h.detach())
        loss = F.cross_entropy(logits.reshape(-1, NUM_ACTIONS), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        correct += int((logits.argmax(-1) == target).sum())
        total += target.numel()
        if step % 200 == 0 or step == args.steps:
            print(json.dumps({"step": step, "loss": float(loss.detach()), "acc": correct / max(1, total),
                              "elapsed_s": time.time() - started}), flush=True)
            correct, total = 0, 0
    torch.save({"head": head.state_dict(), "planner_ckpt": str(args.planner_ckpt),
                "config": {"ctx_dim": cfg.hidden_dim, "horizon": cfg.horizon, "hidden_dim": args.hidden_dim},
                "step": args.steps}, out)
    print(json.dumps({"phase": "done", "out": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
