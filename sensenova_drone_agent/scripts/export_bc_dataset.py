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

from sensenova_drone.bc_data import export_bc_manifest

DEFAULT_EPISODES_ROOT = PROJECT_ROOT / "data" / "bc_sft" / "episodes"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "bc_sft" / "manifests" / "bc_manifest.jsonl"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "bc_sft" / "manifests" / "bc_manifest_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-root", default=str(DEFAULT_EPISODES_ROOT))
    parser.add_argument("--out-jsonl", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--include-worlds", default="")
    parser.add_argument("--required-decision-family", default="")
    parser.add_argument("--allowed-decision-families", default="")
    parser.add_argument("--require-decision-rich", action="store_true")
    parser.add_argument("--allowed-actions", default="")
    parser.add_argument("--followthrough-after-families", default="")
    parser.add_argument("--followthrough-family", default="")
    parser.add_argument("--followthrough-actions", default="")
    parser.add_argument("--followthrough-steps", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = export_bc_manifest(
        episodes_root=args.episodes_root,
        out_jsonl=args.out_jsonl,
        val_ratio=args.val_ratio,
        summary_json=args.summary_json,
        include_worlds={item.strip() for item in args.include_worlds.split(",") if item.strip()} or None,
        required_decision_family=args.required_decision_family.strip() or None,
        allowed_decision_families={item.strip() for item in args.allowed_decision_families.split(",") if item.strip()} or None,
        require_decision_rich=bool(args.require_decision_rich),
        allowed_actions={item.strip() for item in args.allowed_actions.split(",") if item.strip()} or None,
        followthrough_after_families={item.strip() for item in args.followthrough_after_families.split(",") if item.strip()} or None,
        followthrough_family=args.followthrough_family.strip() or None,
        followthrough_actions={item.strip() for item in args.followthrough_actions.split(",") if item.strip()} or None,
        followthrough_steps=args.followthrough_steps,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
