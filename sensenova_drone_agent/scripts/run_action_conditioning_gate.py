#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_action_conditioned_latent_dynamics.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the action-conditioning gate: train normal, shuffled-action, and zero-action "
            "latent dynamics controls before allowing BC/RL promotion."
        )
    )
    parser.add_argument("--sequence-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--offsets", default="0", help="Comma-separated future-action offsets to test, e.g. -2,-1,0,1")
    parser.add_argument("--context-len", type=int, default=8)
    parser.add_argument("--prediction-horizon", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--split-mode", choices=["episode", "episode_task", "anchor"], default="episode_task")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-future-action-rms", type=float, default=0.0)
    parser.add_argument("--min-target-delta-rms", type=float, default=0.0)
    parser.add_argument("--min-control-ratio", type=float, default=1.05)
    parser.add_argument("--max-normal-persistence-ratio", type=float, default=0.95)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets = [int(item.strip()) for item in args.offsets.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for offset in offsets:
        for mode in ["normal", "shuffle_future_actions", "zero_future_actions"]:
            run_dir = out_dir / f"offset_{offset:+d}" / mode
            if not args.dry_run:
                run_training(args, run_dir=run_dir, mode=mode, offset=offset)
            rows.append(read_result(run_dir, mode=mode, offset=offset))

    summary = summarize(rows, args)
    write_json(out_dir / "summary.json", summary)
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary, indent=2))
    return 0 if summary["gate"]["ready_for_bc_or_imagination"] else 2


def run_training(args: argparse.Namespace, *, run_dir: Path, mode: str, offset: int) -> None:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--sequence-cache",
        str(resolve_path(args.sequence_cache)),
        "--out-dir",
        str(run_dir),
        "--context-len",
        str(args.context_len),
        "--prediction-horizon",
        str(args.prediction_horizon),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--split-mode",
        str(args.split_mode),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--control-mode",
        mode,
        "--future-action-offset",
        str(offset),
        "--min-future-action-rms",
        str(args.min_future_action_rms),
        "--min-target-delta-rms",
        str(args.min_target_delta_rms),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def read_result(run_dir: Path, *, mode: str, offset: int) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        return {"mode": mode, "offset": offset, "status": "missing", "run_dir": str(run_dir)}
    summary = json.loads(path.read_text(encoding="utf-8"))
    best = summary["best_metrics"]["best_val_z_mse"]
    val = best["val"]
    return {
        "mode": mode,
        "offset": offset,
        "status": "ok",
        "run_dir": str(run_dir),
        "best_epoch": int(best["epoch"]),
        "z_mse": float(val["z_mse"]),
        "persistence_mse": float(val["persistence_mse"]),
        "z_mse_vs_persistence_ratio": float(val["z_mse_vs_persistence_ratio"]),
        "train_anchors": int(summary["train_anchors"]),
        "val_anchors": int(summary["val_anchors"]),
        "anchor_filter": summary.get("anchor_filter", {}),
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_offset: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        by_offset.setdefault(int(row["offset"]), {})[str(row["mode"])] = row

    decisions = []
    for offset, mode_rows in sorted(by_offset.items()):
        normal = mode_rows.get("normal")
        shuffled = mode_rows.get("shuffle_future_actions")
        zeroed = mode_rows.get("zero_future_actions")
        if not normal or not shuffled or not zeroed:
            continue
        shuffle_ratio = float(shuffled["z_mse"]) / max(float(normal["z_mse"]), 1e-12)
        zero_ratio = float(zeroed["z_mse"]) / max(float(normal["z_mse"]), 1e-12)
        normal_persistence_ratio = float(normal["z_mse_vs_persistence_ratio"])
        passed = (
            normal_persistence_ratio <= float(args.max_normal_persistence_ratio)
            and shuffle_ratio >= float(args.min_control_ratio)
            and zero_ratio >= float(args.min_control_ratio)
        )
        decisions.append(
            {
                "offset": offset,
                "passed": passed,
                "normal_persistence_ratio": normal_persistence_ratio,
                "shuffle_vs_normal_ratio": shuffle_ratio,
                "zero_vs_normal_ratio": zero_ratio,
                "normal_z_mse": normal["z_mse"],
                "shuffle_z_mse": shuffled["z_mse"],
                "zero_z_mse": zeroed["z_mse"],
                "persistence_mse": normal["persistence_mse"],
            }
        )
    passed_decisions = [item for item in decisions if item["passed"]]
    best_decision = None
    if decisions:
        best_decision = max(
            decisions,
            key=lambda item: (
                min(item["shuffle_vs_normal_ratio"], item["zero_vs_normal_ratio"]),
                -item["normal_persistence_ratio"],
            ),
        )
    return {
        "phase": "action_conditioning_gate",
        "sequence_cache": str(resolve_path(args.sequence_cache)),
        "config": vars(args),
        "rows": rows,
        "decisions": decisions,
        "best_decision": best_decision,
        "gate": {
            "ready_for_bc_or_imagination": bool(passed_decisions),
            "passed_offsets": [item["offset"] for item in passed_decisions],
            "decision_rule": (
                "normal must beat persistence and shuffled/zero future-action controls "
                "before BC/RL promotion."
            ),
        },
    }


def write_report(summary: dict[str, Any], out_path: Path) -> None:
    gate = summary["gate"]
    lines = [
        "# Action-Conditioning Gate",
        "",
        f"- Sequence cache: `{summary['sequence_cache']}`",
        f"- Ready for BC/RL promotion: `{gate['ready_for_bc_or_imagination']}`",
        f"- Passed offsets: `{gate['passed_offsets']}`",
        "",
        "## Decisions",
        "",
    ]
    for item in summary["decisions"]:
        lines.append(
            "- offset `{offset}`: passed `{passed}`, normal/persistence `{normal_persistence_ratio:.6f}`, "
            "shuffle/normal `{shuffle_vs_normal_ratio:.6f}`, zero/normal `{zero_vs_normal_ratio:.6f}`".format(**item)
        )
    lines.extend(
        [
            "",
            "## Rule",
            "",
            gate["decision_rule"],
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
