#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sensenova_drone.bc_train import BCTrainConfig, train_supervised_bc

DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "bc_sft" / "manifests" / "bc_manifest.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "bc_policy_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--command-loss-weight", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--zero-goal-features", action="store_true")
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--mirror-lateral-actions", action="store_true")
    parser.add_argument("--balanced-action-sampler", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BCTrainConfig(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        command_loss_weight=args.command_loss_weight,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        use_class_weights=not args.no_class_weights,
        goal_feature_mode="zeros" if args.zero_goal_features else "recorded",
        frame_stack=args.frame_stack,
        mirror_lateral_actions=args.mirror_lateral_actions,
        balanced_action_sampler=args.balanced_action_sampler,
    )
    summary = train_supervised_bc(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
