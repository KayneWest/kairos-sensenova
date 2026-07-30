#!/usr/bin/env python3
"""Decode latent imagination traces to pixels ("watch the model think in frames").

For each sampled context: build candidate plans (true / zero / shuffle / bank),
imagine futures with the planner, score them, and decode context + real future
+ imagined futures through the frozen tokenizer decoder into a PNG grid.

Grid rows (top to bottom):
  1. real context (last ctx frames) + real future frames
  2. tokenizer reconstruction of the real future (fidelity ceiling)
  3. true-plan imagined future
  4. selected candidate (argmax score among bank candidates)
  5. random candidate
  6. worst candidate (argmin score)
  7. zero-action-plan imagined future

A traces.json sidecar records scores and future-MSE per row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DREAMER4_ROOT = REPO_ROOT / "dreamer4" / "dreamer4"
for item in (str(DREAMER4_ROOT), str(PROJECT_ROOT / "scripts")):
    if item not in sys.path:
        sys.path.insert(0, item)

import torch
from PIL import Image
from torch.utils.data import DataLoader

from model import temporal_unpatchify, unpack_spatial_to_bottleneck  # noqa: E402
from train_dynamics import load_frozen_tokenizer_from_pt_ckpt  # noqa: E402
from wm_dataset import WMDataset, collate_batch  # noqa: E402
from train_latent_imagination_planner import (  # noqa: E402
    LatentImaginationPlanner,
    PlannerConfig,
    encode_batch,
    make_action_variant,
    resolve_path,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode imagination traces to pixel grids.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--manifest-json", default="")
    p.add_argument("--tokenizer-ckpt", default="")
    p.add_argument("--source-names", default="soar_native_v2,dreamer4_hf_expert,hf_robot_bridge_orig_lerobot_dreamer4")
    p.add_argument("--num-contexts", type=int, default=6)
    p.add_argument("--num-bank", type=int, default=32, help="Bank candidates per context (plans from other contexts).")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--ctx-show", type=int, default=4, help="Context frames shown in the grid.")
    p.add_argument("--seed", type=int, default=20260707)
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
    horizon = int(args.horizon)
    if cfg.ctx_len + horizon > cfg.seq_len:
        raise ValueError("ctx_len + horizon exceeds seq_len")

    tokenizer_ckpt = args.tokenizer_ckpt or cfg.tokenizer_ckpt
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(resolve_path(tokenizer_ckpt)), device=device)
    patch = int(tok_args.get("patch", 8))
    packing_factor = int(tok_args.get("packing_factor", 2))
    n_latents = int(tok_args.get("n_latents", 16))
    n_spatial = n_latents // packing_factor
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    z_dim = n_spatial * d_bottleneck * packing_factor
    H = int(tok_args.get("H", cfg.img_size))
    W = int(tok_args.get("W", cfg.img_size))
    C = int(tok_args.get("C", 3))

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
    all_traces = []
    for source in manifest.get("sources", []):
        name = source.get("name", "")
        if name not in requested:
            continue
        if not Path(str(source["raw"])).exists():
            print(json.dumps({"phase": "skip_source", "source": name}), flush=True)
            continue
        traces = decode_source(
            source=source,
            model=model,
            encoder=encoder,
            decoder=decoder,
            cfg=cfg,
            args=args,
            horizon=horizon,
            patch=patch,
            n_spatial=n_spatial,
            packing_factor=packing_factor,
            hwc=(H, W, C),
            device=device,
            out_dir=out_dir,
        )
        all_traces.extend(traces)
        print(json.dumps({"phase": "source_done", "source": name, "contexts": len(traces)}), flush=True)

    (out_dir / "traces.json").write_text(json.dumps(all_traces, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"phase": "done", "out_dir": str(out_dir), "contexts": len(all_traces)}), flush=True)
    return 0


@torch.no_grad()
def decode_source(*, source, model, encoder, decoder, cfg, args, horizon, patch, n_spatial, packing_factor, hwc, device, out_dir):
    name = source["name"]
    H, W, C = hwc
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
    loader = DataLoader(dataset, batch_size=args.num_contexts, shuffle=True, num_workers=0, collate_fn=collate_batch, generator=gen)
    raw_batch = next(iter(loader))
    batch = encode_batch(
        raw_batch=raw_batch,
        encoder=encoder,
        patch=patch,
        n_spatial=n_spatial,
        packing_factor=packing_factor,
        action_frame_offset=cfg.action_frame_offset,
        device=device,
    )
    ctx = int(cfg.ctx_len)
    z_flat = batch["z"].flatten(2)
    ctx_z = z_flat[:, :ctx]
    future_z = z_flat[:, ctx : ctx + horizon]
    ctx_actions = batch["actions"][:, :ctx]
    future_actions = batch["transition_actions"][:, ctx : ctx + cfg.horizon]
    future_mask = batch["transition_mask"][:, ctx : ctx + cfg.horizon]
    obs = raw_batch["obs"].to(device)  # (B, T+1, C, H, W) uint8
    B = ctx_z.shape[0]

    ctx_h = model.encode_context(ctx_z, ctx_actions)
    true_plan = model.encode_plan(ctx_h, future_actions)
    zero_actions, zero_mask = make_action_variant(future_actions, future_mask, "zero")
    zero_plan = model.encode_plan(ctx_h, zero_actions * zero_mask)

    # bank candidates: true plans of other contexts in the batch
    K = min(args.num_bank, max(1, B - 1) * 4)
    rng = torch.Generator(device="cpu").manual_seed(args.seed + 7)
    bank_idx = torch.randint(0, B - 1, (B, K), generator=rng)
    row_ids = torch.arange(B)[:, None]
    bank_idx = torch.where(bank_idx >= row_ids, bank_idx + 1, bank_idx).clamp(0, B - 1)
    bank_plans = true_plan[bank_idx]  # (B, K, plan)

    def rollout(plans_flat, reps):
        czr = ctx_z[:, None].expand(B, reps, *ctx_z.shape[1:]).reshape(B * reps, *ctx_z.shape[1:])
        chr_ = ctx_h[:, None].expand(B, reps, ctx_h.shape[-1]).reshape(B * reps, -1)
        fut = model.propose_future(czr, chr_, plans_flat, horizon=horizon)
        score = model.score_future(chr_, fut[:, : cfg.horizon], plans_flat)
        return fut.view(B, reps, horizon, -1), score.view(B, reps)

    bank_fut, bank_score = rollout(bank_plans.reshape(B * K, -1), K)
    named_plans = torch.stack([true_plan, zero_plan], dim=1)
    named_fut, named_score = rollout(named_plans.reshape(B * 2, -1), 2)

    def decode_latents(z_seq):  # (N, T, z_dim) -> uint8 (N, T, H, W, C)
        z = z_seq.view(z_seq.shape[0], z_seq.shape[1], n_spatial, -1)
        z = unpack_spatial_to_bottleneck(z, k=packing_factor)
        patches = decoder(z)
        frames = temporal_unpatchify(patches, H, W, C, patch)  # (N, T, C, H, W) in [0,1]
        return (frames.clamp(0, 1) * 255).to(torch.uint8).permute(0, 1, 3, 4, 2).cpu().numpy()

    src_dir = out_dir / name
    src_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    rand_pick = torch.randint(0, K, (B,), generator=rng)
    for b in range(B):
        sel = int(bank_score[b].argmax())
        worst = int(bank_score[b].argmin())
        rnd = int(rand_pick[b])
        rows = {
            "real": None,  # filled from obs
            "recon_future": future_z[b : b + 1],
            "true_plan": named_fut[b, 0][None],
            "selected": bank_fut[b, sel][None],
            "random": bank_fut[b, rnd][None],
            "worst": bank_fut[b, worst][None],
            "zero_plan": named_fut[b, 1][None],
        }
        mse = lambda fut: float((fut[0] - future_z[b]).pow(2).mean())
        meta = {
            "source": name,
            "context_id": int(b),
            "emb_id": int(raw_batch["emb_id"][b]),
            "scores": {
                "true_plan": float(named_score[b, 0]),
                "zero_plan": float(named_score[b, 1]),
                "selected": float(bank_score[b, sel]),
                "random": float(bank_score[b, rnd]),
                "worst": float(bank_score[b, worst]),
            },
            "future_mse": {k: mse(v) for k, v in rows.items() if k not in ("real",)},
            "row_order": list(rows.keys()),
        }
        # real row: last ctx_show context frames + real future frames
        real = obs[b, ctx - args.ctx_show : ctx + horizon].permute(0, 2, 3, 1).cpu().numpy()
        decoded_rows = [real]
        pad = np.zeros((args.ctx_show, H, W, C), dtype=np.uint8)  # imagined rows have no context
        for key in list(rows.keys())[1:]:
            imgs = decode_latents(rows[key].to(device))[0]
            decoded_rows.append(np.concatenate([pad, imgs], axis=0))
        grid = np.concatenate([np.concatenate(list(r), axis=1) for r in decoded_rows], axis=0)
        path = src_dir / f"ctx_{b:02d}_grid.png"
        Image.fromarray(grid).save(path)
        meta["grid_png"] = str(path)
        traces.append(meta)
    return traces


if __name__ == "__main__":
    raise SystemExit(main())
