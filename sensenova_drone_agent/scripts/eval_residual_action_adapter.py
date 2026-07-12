#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DREAMER4_ROOT = REPO_ROOT / "dreamer4" / "dreamer4"
if str(DREAMER4_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMER4_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_dreamer4_soar_dynamics import build_dynamics, evaluate, parse_negative_modes  # noqa: E402
from train_dynamics import load_frozen_tokenizer_from_pt_ckpt  # noqa: E402
from train_residual_action_adapter import ResidualActionAdapter, ResidualDynamicsWrapper, select_sources  # noqa: E402
from wm_dataset import WMDataset, collate_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a residual action adapter checkpoint.")
    parser.add_argument("--adapter-ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest-json", default=None)
    parser.add_argument("--tokenizer-ckpt", default=None)
    parser.add_argument("--dynamics-ckpt", default=None)
    parser.add_argument("--tasks-json", default=None)
    parser.add_argument("--source-names", default=None)
    parser.add_argument(
        "--tasks",
        default=None,
        help="Optional comma-separated task names to evaluate. Missing tasks are treated as an error.",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--eval-d", type=float, default=0.25)
    parser.add_argument("--causal-min-ratio", type=float, default=1.02)
    parser.add_argument("--negative-modes", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1053)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    adapter_ckpt_path = resolve_path(args.adapter_ckpt)
    adapter_ckpt = torch.load(adapter_ckpt_path, map_location="cpu", weights_only=False)
    saved_args = dict(adapter_ckpt.get("args", {}))

    manifest_json = resolve_path(args.manifest_json or saved_args["manifest_json"])
    tokenizer_ckpt = resolve_path(args.tokenizer_ckpt or saved_args["tokenizer_ckpt"])
    dynamics_ckpt = resolve_path(args.dynamics_ckpt or saved_args["dynamics_ckpt"])
    tasks_json = resolve_path(args.tasks_json or saved_args.get("tasks_json") or "")
    source_names = str(args.source_names or saved_args.get("source_names", ""))
    task_filter = parse_csv(args.tasks)
    seq_len = int(args.seq_len if args.seq_len is not None else saved_args.get("seq_len", 16))
    action_dim = int(saved_args.get("action_dim", 49))
    action_features = str(saved_args.get("action_features", "current,prev,delta,mean4,norm"))
    negative_modes = parse_negative_modes(args.negative_modes or saved_args.get("contrast_modes", "shuffle,zero"))

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    sources = select_sources(manifest, source_names)
    dataset = WMDataset(
        data_dir=[str(resolve_path(path)) for source in sources for path in source["raw"]],
        frames_dir=[str(resolve_path(path)) for source in sources for path in source["frames"]],
        seq_len=seq_len,
        img_size=128,
        action_dim=action_dim,
        tasks_json=str(tasks_json),
        tasks=task_filter,
        strict_tasks=bool(task_filter),
        action_features=action_features,
        require_non_noop=bool(saved_args.get("require_non_noop", False)),
        no_op_threshold=float(saved_args.get("no_op_threshold", 0.0)),
        min_non_noop_steps=int(saved_args.get("min_non_noop_steps", 1)),
        require_visual_delta=bool(saved_args.get("require_visual_delta", False)),
        visual_delta_threshold=float(saved_args.get("visual_delta_threshold", 0.0)),
        min_visual_delta_steps=int(saved_args.get("min_visual_delta_steps", 1)),
        visual_delta_stride=int(saved_args.get("visual_delta_stride", 4)),
        verbose=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_batch,
    )

    encoder, _decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(tokenizer_ckpt), device=device)
    dyn_ckpt = torch.load(dynamics_ckpt, map_location="cpu", weights_only=False)
    dyn_args = dict(dyn_ckpt.get("args", {}))
    dyn_args["action_dim"] = action_dim
    dyn_args["action_features"] = action_features
    base = build_dynamics(dyn_args, tok_args).to(device)
    base.load_state_dict(dyn_ckpt["dynamics"], strict=True)
    base.eval()
    for param in base.parameters():
        param.requires_grad_(False)

    packing_factor = int(dyn_args.get("packing_factor", 2))
    n_latents = int(tok_args.get("n_latents", 16))
    n_spatial = n_latents // packing_factor
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    d_spatial = d_bottleneck * packing_factor
    k_max = int(dyn_args.get("k_max", 8))
    adapter = ResidualActionAdapter(
        action_dim=action_dim,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        k_max=k_max,
        hidden=int(saved_args.get("hidden", 256)),
    ).to(device)
    adapter.load_state_dict(adapter_ckpt["adapter"], strict=True)
    adapter.eval()
    wrapped = ResidualDynamicsWrapper(
        base=base,
        adapter=adapter,
        scale=float(saved_args.get("residual_scale", 1.0)),
    ).to(device)
    wrapped.eval()

    metrics = evaluate(
        model=wrapped,
        encoder=encoder,
        loader=loader,
        tok_args=tok_args,
        dyn_args=dyn_args,
        device=device,
        max_batches=int(args.max_batches),
        rollout_horizon=int(args.rollout_horizon),
        ctx_len=int(args.ctx_len),
        eval_d=float(args.eval_d),
        action_frame_offset=int(saved_args.get("action_frame_offset", -1)),
        seed=int(args.seed),
        causal_min_ratio=float(args.causal_min_ratio),
        negative_modes=negative_modes,
    )
    payload: dict[str, Any] = {
        "phase": "residual_action_adapter_eval",
        "adapter_ckpt": str(adapter_ckpt_path),
        "adapter_step": int(adapter_ckpt.get("step", -1)),
        "sources": [source["name"] for source in sources],
        "config": {
            "seq_len": seq_len,
            "batch_size": int(args.batch_size),
            "max_batches": int(args.max_batches),
            "rollout_horizon": int(args.rollout_horizon),
            "ctx_len": int(args.ctx_len),
            "eval_d": float(args.eval_d),
            "causal_min_ratio": float(args.causal_min_ratio),
            "negative_modes": negative_modes,
            "task_filter": task_filter,
            "residual_scale": float(saved_args.get("residual_scale", 1.0)),
            "action_frame_offset": int(saved_args.get("action_frame_offset", -1)),
            "action_dim": action_dim,
            "action_features": action_features,
        },
        "metrics": metrics,
        "elapsed_s": time.time() - started,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or None


if __name__ == "__main__":
    raise SystemExit(main())
