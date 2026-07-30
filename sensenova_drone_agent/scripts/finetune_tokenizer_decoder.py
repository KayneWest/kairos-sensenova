#!/usr/bin/env python3
"""Decoder-only tokenizer fine-tune for sharper imagination-trace rendering.

The encoder stays frozen so the latent space (and every planner/dynamics
model trained on it) is untouched; only the decoder learns to render those
latents with more detail. The joint MAE training left the decoder blurry
(scene-average reconstructions that drop small task-relevant structure like
robot arms). Pure reconstruction loss, no masking.

Saves checkpoints in the same {"args", "model", "step"} format as the
original tokenizer ckpt so load_frozen_tokenizer_from_pt_ckpt works
unchanged.
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

from model import Tokenizer, temporal_patchify  # noqa: E402
from train_dynamics import load_frozen_tokenizer_from_pt_ckpt  # noqa: E402
from wm_dataset import WMDataset, collate_batch  # noqa: E402
from train_latent_imagination_planner import resolve_path, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decoder-only tokenizer fine-tune.")
    p.add_argument("--tokenizer-ckpt", required=True)
    p.add_argument("--manifest-json", required=True)
    p.add_argument("--source-names", default="", help="Comma list (default: all manifest sources with weight > 0).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--clip-len", type=int, default=8, help="Frames per training clip.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--motion-weight", type=float, default=0.0, help="Upweight pixels that move across the clip (action-relevant structure); weight = 1 + mw * normalized per-pixel temporal std.")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260708)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --tokenizer-ckpt may point at a previous decoder fine-tune (same format)
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(resolve_path(args.tokenizer_ckpt)), device=device)
    patch = int(tok_args.get("patch", 8))
    for p_ in decoder.parameters():
        p_.requires_grad_(True)
    decoder.train()

    manifest = json.loads(resolve_path(args.manifest_json).read_text(encoding="utf-8"))
    requested = [s.strip() for s in args.source_names.split(",") if s.strip()]
    data_dirs, frame_dirs = [], []
    for source in manifest.get("sources", []):
        name = source.get("name", "")
        if requested and name not in requested:
            continue
        if not requested and int(source.get("weight", 0)) <= 0:
            continue
        if not Path(str(source["raw"])).exists():
            continue
        data_dirs.append(str(source["raw"]))
        frame_dirs.append(str(source["frames"]))
    if not data_dirs:
        raise ValueError("No sources selected.")

    dataset = WMDataset(
        data_dir=data_dirs,
        frames_dir=frame_dirs,
        seq_len=max(args.clip_len, 2),
        img_size=int(tok_args.get("H", 128)),
        action_dim=49,
        raw_action_dim=49,
        tasks_json=str(manifest.get("tasks_json", "")),
        tasks=None,
        strict_tasks=False,
        action_features="current,prev,delta,mean4,norm",
        verbose=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        collate_fn=collate_batch,
    )
    print(json.dumps({"phase": "decoder_finetune", "dataset_windows": len(dataset), "sources": data_dirs, "steps": args.steps}), flush=True)

    opt = torch.optim.AdamW(decoder.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    metrics_path = out_dir / "metrics.jsonl"
    started = time.time()
    iterator = iter(loader)
    step = 0
    while step < args.steps:
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        step += 1
        frames = raw["obs"][:, : args.clip_len].to(device, non_blocking=True).float() / 255.0
        patches = temporal_patchify(frames, patch)
        with torch.no_grad():
            z, _ = encoder(patches)
        pred = decoder(z)
        if args.motion_weight > 0:
            motion = frames.std(dim=1)  # (B,C,H,W) per-pixel temporal std
            motion = motion / motion.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            w = 1.0 + args.motion_weight * motion
            w_patches = temporal_patchify(w[:, None].expand(-1, frames.shape[1], -1, -1, -1), patch)
            err = (pred - patches).pow(2)
            loss = (w_patches * err).sum() / w_patches.sum() + 0.2 * (w_patches * (pred - patches).abs()).sum() / w_patches.sum()
        else:
            loss = F.mse_loss(pred, patches) + 0.2 * F.l1_loss(pred, patches)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": float(loss.detach()), "elapsed_s": time.time() - started}
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_ckpt(out_dir / "decoder_finetuned_latest.pt", encoder=encoder, decoder=decoder, tok_args=tok_args, step=step)
    save_ckpt(out_dir / "decoder_finetuned_final.pt", encoder=encoder, decoder=decoder, tok_args=tok_args, step=args.steps)
    print(json.dumps({"phase": "done", "out_dir": str(out_dir)}), flush=True)
    return 0


def save_ckpt(path: Path, *, encoder, decoder, tok_args, step: int) -> None:
    tok = Tokenizer(encoder, decoder)
    tmp = path.with_suffix(".tmp")
    torch.save({"args": dict(tok_args), "model": tok.state_dict(), "step": int(step)}, tmp)
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
