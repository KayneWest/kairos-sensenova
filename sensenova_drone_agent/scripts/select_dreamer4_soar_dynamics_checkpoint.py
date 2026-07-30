#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a Dreamer4 SOAR dynamics checkpoint using held-out action-causality gates "
            "instead of training loss or latest checkpoint."
        )
    )
    parser.add_argument("--data-dir", action="append", required=True)
    parser.add_argument("--frames-dir", action="append", required=True)
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--tokenizer-ckpt", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-glob", action="append", default=["step_*.pt", "final_step_*.pt", "latest.pt"])
    parser.add_argument("--max-checkpoints", type=int, default=4)
    parser.add_argument("--horizons", default="8,16")
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=256)
    parser.add_argument("--eval-d", type=float, default=0.25)
    parser.add_argument("--action-dim", type=int, default=64)
    parser.add_argument("--action-features", default="current,prev,delta,mean4,norm")
    parser.add_argument("--action-frame-offset", type=int, default=0)
    parser.add_argument("--require-non-noop", action="store_true")
    parser.add_argument("--no-op-threshold", type=float, default=0.0)
    parser.add_argument("--min-non-noop-steps", type=int, default=1)
    parser.add_argument(
        "--reward-filter-mode",
        default="none",
        choices=["none", "positive_sum", "abs_sum", "any_positive", "any_abs"],
    )
    parser.add_argument("--reward-signal-threshold", type=float, default=0.0)
    parser.add_argument("--min-reward-signal-steps", type=int, default=1)
    parser.add_argument("--require-visual-delta", action="store_true")
    parser.add_argument("--visual-delta-threshold", type=float, default=0.0)
    parser.add_argument("--min-visual-delta-steps", type=int, default=1)
    parser.add_argument("--visual-delta-stride", type=int, default=4)
    parser.add_argument("--negative-modes", default="shuffle,zero,time_shift,far_shuffle")
    parser.add_argument("--causal-min-ratio", type=float, default=1.02)
    parser.add_argument("--max-persistence-ratio", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    out_dir = resolve_path(args.out_dir)
    eval_dir = out_dir / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    horizons = parse_int_list(args.horizons)
    modes = parse_list(args.negative_modes)
    checkpoints = collect_checkpoints(args)
    if not checkpoints:
        raise RuntimeError(f"No dynamics checkpoints found under {resolve_path(args.ckpt_dir)}")

    rows = []
    for ckpt in checkpoints:
        horizon_rows = []
        for horizon in horizons:
            eval_payload = run_or_load_eval(args, ckpt, horizon, eval_dir)
            gate = gate_payload(
                eval_payload,
                negative_modes=modes,
                causal_min_ratio=float(args.causal_min_ratio),
                max_persistence_ratio=float(args.max_persistence_ratio),
            )
            horizon_rows.append(
                {
                    "horizon": int(horizon),
                    "eval_path": str((eval_dir / eval_filename(ckpt, horizon)).resolve()),
                    "gate": gate,
                    "metrics": eval_payload.get("metrics", {}),
                    "decision": eval_payload.get("decision", {}),
                }
            )
        strict = all(row["gate"]["strict_gate_passed"] for row in horizon_rows)
        score = min(row["gate"]["score"] for row in horizon_rows)
        rows.append(
            {
                "checkpoint": str(ckpt.resolve()),
                "checkpoint_name": ckpt.name,
                "step": checkpoint_step(ckpt),
                "strict_gate_passed": bool(strict),
                "score": float(score),
                "horizons": horizon_rows,
            }
        )

    passed = [row for row in rows if row["strict_gate_passed"]]
    if passed:
        selected = max(passed, key=lambda row: (row["score"], row["step"]))
    else:
        selected = max(rows, key=lambda row: (row["score"], row["step"]))

    selected_src = Path(selected["checkpoint"])
    selected_dst = out_dir / "best_selected_dynamics.pt"
    shutil.copy2(selected_src, selected_dst)

    summary = {
        "phase": "dreamer4_soar_dynamics_checkpoint_selection",
        "claim_boundary": (
            "Dynamics selection only. This does not claim real-world control or imagination-RL success."
        ),
        "config": {
            "data_dir": [str(resolve_path(path)) for path in args.data_dir],
            "frames_dir": [str(resolve_path(path)) for path in args.frames_dir],
            "tasks_json": str(resolve_path(args.tasks_json)),
            "tokenizer_ckpt": str(resolve_path(args.tokenizer_ckpt)),
            "ckpt_dir": str(resolve_path(args.ckpt_dir)),
            "horizons": horizons,
            "ctx_len": int(args.ctx_len),
            "batch_size": int(args.batch_size),
            "max_batches": int(args.max_batches),
            "negative_modes": modes,
            "causal_min_ratio": float(args.causal_min_ratio),
            "max_persistence_ratio": float(args.max_persistence_ratio),
            "action_dim": int(args.action_dim),
            "action_features": str(args.action_features),
            "action_frame_offset": int(args.action_frame_offset),
            "require_non_noop": bool(args.require_non_noop),
            "no_op_threshold": float(args.no_op_threshold),
            "min_non_noop_steps": int(args.min_non_noop_steps),
            "reward_filter_mode": str(args.reward_filter_mode),
            "reward_signal_threshold": float(args.reward_signal_threshold),
            "min_reward_signal_steps": int(args.min_reward_signal_steps),
            "require_visual_delta": bool(args.require_visual_delta),
            "visual_delta_threshold": float(args.visual_delta_threshold),
            "min_visual_delta_steps": int(args.min_visual_delta_steps),
            "visual_delta_stride": int(args.visual_delta_stride),
        },
        "rows": rows,
        "selected": selected,
        "selected_checkpoint": str(selected_dst.resolve()),
        "strict_gate_passed": bool(selected["strict_gate_passed"]),
        "elapsed_s": float(time.time() - started),
    }
    write_json(out_dir / "selection_summary.json", summary)
    write_report(out_dir / "selection_report.md", summary)
    print(json.dumps(compact_summary(summary), indent=2))
    return 0


def collect_checkpoints(args: argparse.Namespace) -> list[Path]:
    ckpt_dir = resolve_path(args.ckpt_dir)
    found: list[Path] = [resolve_path(path) for path in args.checkpoint]
    for pattern in args.checkpoint_glob:
        found.extend(sorted(ckpt_dir.glob(pattern)))
    unique: dict[str, Path] = {}
    for path in found:
        if path.exists() and path.suffix == ".pt":
            unique[str(path.resolve())] = path.resolve()
    checkpoints = sorted(unique.values(), key=lambda path: (checkpoint_step(path), path.name))
    max_checkpoints = int(args.max_checkpoints)
    if max_checkpoints > 0 and len(checkpoints) > max_checkpoints:
        checkpoints = checkpoints[-max_checkpoints:]
    return checkpoints


def run_or_load_eval(args: argparse.Namespace, ckpt: Path, horizon: int, eval_dir: Path) -> dict[str, Any]:
    out_path = eval_dir / eval_filename(ckpt, horizon)
    if out_path.exists() and not bool(args.force):
        return json.loads(out_path.read_text(encoding="utf-8"))
    seq_len = max(int(args.ctx_len) + int(horizon), int(args.ctx_len) + 1)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_dreamer4_soar_dynamics.py"),
        "--tasks-json",
        str(resolve_path(args.tasks_json)),
        "--tokenizer-ckpt",
        str(resolve_path(args.tokenizer_ckpt)),
        "--dynamics-ckpt",
        str(ckpt),
        "--out",
        str(out_path),
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(int(args.batch_size)),
        "--max-batches",
        str(int(args.max_batches)),
        "--rollout-horizon",
        str(int(horizon)),
        "--ctx-len",
        str(int(args.ctx_len)),
        "--eval-d",
        str(float(args.eval_d)),
        "--action-dim",
        str(int(args.action_dim)),
        "--action-features",
        str(args.action_features),
        "--action-frame-offset",
        str(int(args.action_frame_offset)),
        "--no-op-threshold",
        str(float(args.no_op_threshold)),
        "--min-non-noop-steps",
        str(int(args.min_non_noop_steps)),
        "--reward-filter-mode",
        str(args.reward_filter_mode),
        "--reward-signal-threshold",
        str(float(args.reward_signal_threshold)),
        "--min-reward-signal-steps",
        str(int(args.min_reward_signal_steps)),
        "--visual-delta-threshold",
        str(float(args.visual_delta_threshold)),
        "--min-visual-delta-steps",
        str(int(args.min_visual_delta_steps)),
        "--visual-delta-stride",
        str(int(args.visual_delta_stride)),
        "--negative-modes",
        str(args.negative_modes),
        "--causal-min-ratio",
        str(float(args.causal_min_ratio)),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed) + int(horizon) * 1000 + max(0, checkpoint_step(ckpt))),
    ]
    for path in args.data_dir:
        command.extend(["--data-dir", str(resolve_path(path))])
    for path in args.frames_dir:
        command.extend(["--frames-dir", str(resolve_path(path))])
    if bool(args.require_non_noop):
        command.append("--require-non-noop")
    if bool(args.require_visual_delta):
        command.append("--require-visual-delta")
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)
    return json.loads(out_path.read_text(encoding="utf-8"))


def gate_payload(
    payload: dict[str, Any],
    *,
    negative_modes: list[str],
    causal_min_ratio: float,
    max_persistence_ratio: float,
) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    direct = metrics.get("direct", {})
    auto = metrics.get("autoregressive", {})
    direct_ratios = {mode: float(direct.get(f"{mode}_over_normal", 0.0)) for mode in negative_modes}
    auto_ratios = {mode: float(auto.get(f"{mode}_over_normal", 0.0)) for mode in negative_modes}
    persistence_ratio = float(auto.get("normal_over_persistence", float("inf")))
    direct_min = min(direct_ratios.values(), default=0.0)
    auto_min = min(auto_ratios.values(), default=0.0)
    direct_pass = all(value >= float(causal_min_ratio) for value in direct_ratios.values())
    auto_pass = all(value >= float(causal_min_ratio) for value in auto_ratios.values())
    persistence_pass = persistence_ratio <= float(max_persistence_ratio)
    score = min(
        direct_min - float(causal_min_ratio),
        auto_min - float(causal_min_ratio),
        float(max_persistence_ratio) - persistence_ratio,
    )
    return {
        "strict_gate_passed": bool(direct_pass and auto_pass and persistence_pass),
        "direct_pass": bool(direct_pass),
        "autoregressive_pass": bool(auto_pass),
        "persistence_pass": bool(persistence_pass),
        "direct_min_ratio": float(direct_min),
        "autoregressive_min_ratio": float(auto_min),
        "normal_over_persistence": float(persistence_ratio),
        "score": float(score),
        "direct_ratios": direct_ratios,
        "autoregressive_ratios": auto_ratios,
    }


def eval_filename(ckpt: Path, horizon: int) -> str:
    return f"{ckpt.stem}_h{int(horizon)}.json"


def checkpoint_step(path: Path) -> int:
    match = re.search(r"(?:step|final_step)_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    if path.stem == "latest":
        return 10**12
    return -1


def write_report(out_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dreamer4 SOAR Dynamics Checkpoint Selection",
        "",
        summary["claim_boundary"],
        "",
        "## Selected",
        "",
        f"- Checkpoint: `{summary['selected']['checkpoint_name']}`",
        f"- Strict gate passed: `{summary['strict_gate_passed']}`",
        f"- Score: `{summary['selected']['score']:+.4f}`",
        f"- Copied to: `{summary['selected_checkpoint']}`",
        "",
        "## Candidates",
        "",
        "| Checkpoint | Strict gate | Score | h | Direct min | AR min | Normal / persistence |",
        "|---|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("rows", []):
        for horizon in row.get("horizons", []):
            gate = horizon["gate"]
            lines.append(
                "| {ckpt} | {passed} | {score:+.4f} | {h} | {direct:.4f} | {auto:.4f} | {persist:.4f} |".format(
                    ckpt=row["checkpoint_name"],
                    passed="yes" if row["strict_gate_passed"] else "no",
                    score=float(gate["score"]),
                    h=int(horizon["horizon"]),
                    direct=float(gate["direct_min_ratio"]),
                    auto=float(gate["autoregressive_min_ratio"]),
                    persist=float(gate["normal_over_persistence"]),
                )
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": summary["phase"],
        "strict_gate_passed": summary["strict_gate_passed"],
        "selected_checkpoint": summary["selected_checkpoint"],
        "selected_name": summary["selected"]["checkpoint_name"],
        "selected_score": summary["selected"]["score"],
        "candidate_count": len(summary.get("rows", [])),
    }


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).replace("+", ",").split(",") if item.strip()]


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).replace("+", ",").split(",") if item.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
