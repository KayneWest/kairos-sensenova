#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sensenova_drone.midtraining import (  # noqa: E402
    build_valid_anchors,
    compute_normalizer,
    load_sequence_cache,
    normalize_np,
    split_anchors,
    split_anchors_by_episode,
    split_anchors_by_task_episode,
)


CONTROL_MODES = [
    "normal",
    "shuffle_targets",
    "shuffle_z_context",
    "zero_z_context",
    "zero_prev_actions",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SOAR phase-2 midtraining duration and control-baseline validation."
    )
    parser.add_argument(
        "--sequence-cache",
        default=(
            "sensenova_drone_agent/output/soar_sequence_cache_kairos_large/"
            "soar_kairos_flat128_128traj32_trajectory_success.npz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="sensenova_drone_agent/output/soar_midtraining_validation_v1",
    )
    parser.add_argument("--context-len", type=int, default=8)
    parser.add_argument("--mtp-horizon", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--control-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=["episode", "episode_task", "anchor"], default="episode_task")
    parser.add_argument("--bc-positive-reward-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--control-modes", default=",".join(CONTROL_MODES))
    parser.add_argument("--force", action="store_true", help="Rerun jobs even when summary.json exists.")
    parser.add_argument("--dry-run", action="store_true", help="Write the planned jobs without running them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sequence_cache = resolve_path(args.sequence_cache)
    seeds = parse_int_csv(args.seeds)
    control_modes = parse_csv(args.control_modes)
    started = time.time()

    jobs = build_jobs(args, seeds, control_modes)
    if args.dry_run:
        summary = {
            "phase": "soar_midtraining_validation_suite",
            "sequence_cache": str(sequence_cache),
            "out_dir": str(out_dir),
            "planned_jobs": jobs,
        }
        write_json(out_dir / "planned_jobs.json", summary)
        print(json.dumps(summary, indent=2))
        return 0

    run_summaries: dict[str, dict[str, Any]] = {}
    for job in jobs:
        label = job["label"]
        run_dir = out_dir / "runs" / label
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"
        if summary_path.exists() and not args.force:
            run_summaries[label] = load_json(summary_path)
            continue
        command = train_command(args, sequence_cache, run_dir, job)
        (run_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
            run_dir / "stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Training job {label} failed with exit code {result.returncode}. See {run_dir}.")
        run_summaries[label] = load_json(summary_path)

    analytical = analytical_action_baselines(
        sequence_cache,
        context_len=args.context_len,
        mtp_horizon=args.mtp_horizon,
        val_ratio=args.val_ratio,
        seed=seeds[0] if seeds else 0,
        split_mode=args.split_mode,
    )
    summary = summarize_suite(args, sequence_cache, out_dir, run_summaries, analytical, time.time() - started)
    write_json(out_dir / "summary.json", summary)
    write_report(summary, out_dir / "report.md")
    print(json.dumps(compact_suite_summary(summary), indent=2))
    return 0


def build_jobs(args: argparse.Namespace, seeds: list[int], control_modes: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in seeds:
        jobs.append(
            {
                "label": f"normal_seed{seed}_{args.epochs}e",
                "seed": seed,
                "control_mode": "normal",
                "control_seed": seed,
                "epochs": args.epochs,
                "group": "duration_seed",
            }
        )
    for mode in control_modes:
        label = f"control_{mode}_seed0_{args.control_epochs}e"
        duplicate = mode == "normal" and 0 in seeds and args.control_epochs == args.epochs
        if duplicate:
            continue
        jobs.append(
            {
                "label": label,
                "seed": 0,
                "control_mode": mode,
                "control_seed": 0,
                "epochs": args.control_epochs,
                "group": "control",
            }
        )
    return jobs


def train_command(args: argparse.Namespace, sequence_cache: Path, run_dir: Path, job: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "sensenova_drone_agent/scripts/train_behavior_cloning_midtraining.py",
        "--sequence-cache",
        str(sequence_cache),
        "--out-dir",
        str(run_dir),
        "--context-len",
        str(args.context_len),
        "--mtp-horizon",
        str(args.mtp_horizon),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(job["epochs"]),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--val-ratio",
        str(args.val_ratio),
        "--split-mode",
        args.split_mode,
        "--seed",
        str(job["seed"]),
        "--device",
        args.device,
        "--control-mode",
        job["control_mode"],
        "--control-seed",
        str(job["control_seed"]),
    ]
    if args.bc_positive_reward_only:
        command.append("--bc-positive-reward-only")
    return command


def summarize_suite(
    args: argparse.Namespace,
    sequence_cache: Path,
    out_dir: Path,
    runs: dict[str, dict[str, Any]],
    analytical: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    run_rows = {label: summarize_run(summary) for label, summary in sorted(runs.items())}
    normal_rows = [
        row
        for label, row in run_rows.items()
        if row["control_mode"] == "normal" and label.startswith("normal_seed")
    ]
    control_rows = {
        row["control_mode"]: row
        for label, row in run_rows.items()
        if label.startswith("control_") or (row["control_mode"] == "normal" and label.startswith("normal_seed0_"))
    }
    normal_ref = control_rows.get("normal") or (normal_rows[0] if normal_rows else None)
    normal_action = float(normal_ref["best_val_decision_action_mse"]) if normal_ref else float("nan")
    control_ratios = {}
    for mode, row in sorted(control_rows.items()):
        action = float(row["best_val_decision_action_mse"])
        control_ratios[mode] = action / normal_action if normal_action > 0 else float("nan")

    normal_actions = [float(row["best_val_decision_action_mse"]) for row in normal_rows]
    normal_tail_gains = [float(row["tail_relative_gain"]) for row in normal_rows if row["tail_relative_gain"] is not None]
    seed_stats = {
        "count": len(normal_actions),
        "best_val_action_mse_mean": mean(normal_actions) if normal_actions else None,
        "best_val_action_mse_std": pstdev(normal_actions) if len(normal_actions) > 1 else 0.0,
        "tail_relative_gain_mean": mean(normal_tail_gains) if normal_tail_gains else None,
    }
    decision = validation_decision(normal_ref, control_ratios, seed_stats, args)
    return {
        "phase": "soar_midtraining_validation_suite",
        "claim_boundary": (
            "This validates phase-2 BC/reward/value midtraining on frozen Kairos/Wan latents. "
            "It does not train Kairos and does not run imagination RL."
        ),
        "elapsed_s": elapsed_s,
        "sequence_cache": str(sequence_cache),
        "out_dir": str(out_dir),
        "config": {
            "context_len": args.context_len,
            "mtp_horizon": args.mtp_horizon,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "control_epochs": args.control_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "val_ratio": args.val_ratio,
            "split_mode": args.split_mode,
            "bc_positive_reward_only": args.bc_positive_reward_only,
            "device": args.device,
            "seeds": parse_int_csv(args.seeds),
            "control_modes": parse_csv(args.control_modes),
        },
        "theoretical_alignment": {
            "frozen_world_model_features": True,
            "multi_token_prediction_horizon_8": args.mtp_horizon == 8,
            "task_conditioned_agent_token": True,
            "agent_token_isolation": True,
            "action_head": True,
            "reward_head": True,
            "value_head": True,
            "episode_heldout_validation": args.split_mode in {"episode", "episode_task"},
            "task_stratified_validation": args.split_mode == "episode_task",
            "control_baselines": True,
        },
        "runs": run_rows,
        "normal_seed_stats": seed_stats,
        "control_ratios_vs_normal": control_ratios,
        "analytical_action_baselines": analytical,
        "decision": decision,
    }


def summarize_run(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary["config"]
    best = summary.get("best_metrics", {})
    best_action = best.get("best_val_action_mse", {})
    best_bc_action = best.get("best_val_bc_action_mse", {})
    best_all_action = best.get("best_val_all_action_mse", {})
    best_reward = best.get("best_val_reward_mse", {})
    best_value = best.get("best_val_value_mse", {})
    best_loss = best.get("best_val_loss", {})
    duration = summary.get("duration_analysis", {})
    decision_action_mse = best_action.get("action_mse")
    decision_action_epoch = best_action.get("epoch")
    if config.get("bc_positive_reward_only") and best_bc_action:
        decision_action_mse = best_bc_action.get("bc_action_mse")
        decision_action_epoch = best_bc_action.get("epoch")
    return {
        "control_mode": config.get("control_mode", "normal"),
        "seed": config.get("seed", 0),
        "epochs": config.get("epochs"),
        "split_mode": config.get("split_mode", "unknown"),
        "train_anchors": summary.get("train_anchors"),
        "val_anchors": summary.get("val_anchors"),
        "best_val_loss": best_loss.get("loss"),
        "best_val_loss_epoch": best_loss.get("epoch"),
        "best_val_action_mse": best_action.get("action_mse"),
        "best_val_action_epoch": best_action.get("epoch"),
        "best_val_bc_action_mse": best_bc_action.get("bc_action_mse"),
        "best_val_bc_action_epoch": best_bc_action.get("epoch"),
        "best_val_all_action_mse": best_all_action.get("all_action_mse"),
        "best_val_decision_action_mse": decision_action_mse,
        "best_val_decision_action_epoch": decision_action_epoch,
        "best_val_reward_mse": best_reward.get("reward_mse"),
        "best_val_value_mse": best_value.get("value_mse"),
        "first_val_action_mse": duration.get("first_val_action_mse"),
        "last_val_action_mse": duration.get("last_val_action_mse"),
        "tail_relative_gain": duration.get("tail_relative_gain"),
        "best_at_final_epoch": duration.get("best_at_final_epoch"),
        "elapsed_s": summary.get("elapsed_s"),
        "best_checkpoint": summary.get("best_checkpoint"),
    }


def validation_decision(
    normal_ref: dict[str, Any] | None,
    ratios: dict[str, float],
    seed_stats: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    shuffle_ratio = ratios.get("shuffle_targets", 0.0)
    zero_z_ratio = ratios.get("zero_z_context", 0.0)
    shuffle_z_ratio = ratios.get("shuffle_z_context", 0.0)
    zero_prev_ratio = ratios.get("zero_prev_actions", 0.0)
    tail_gain = seed_stats.get("tail_relative_gain_mean")
    action_signal = bool(shuffle_ratio >= 1.2)
    visual_signal = bool(max(zero_z_ratio, shuffle_z_ratio) >= 1.05)
    action_context_signal = bool(zero_prev_ratio >= 1.05)
    if tail_gain is None:
        duration_status = "unknown"
    elif tail_gain > 0.02:
        duration_status = "still_improving"
    elif tail_gain < -0.02:
        duration_status = "overfit_after_best"
    else:
        duration_status = "plateaued"
    duration_plateaued = duration_status == "plateaued"
    early_stopping_recommended = duration_status == "overfit_after_best"
    theoretical_match = bool(args.mtp_horizon == 8 and args.split_mode in {"episode", "episode_task"})
    reward_value_ok = True
    if normal_ref:
        reward_value_ok = (
            float(normal_ref.get("best_val_reward_mse") or 999.0) < 0.02
            and float(normal_ref.get("best_val_value_mse") or 999.0) < 0.05
        )
    ready = bool(action_signal and visual_signal and action_context_signal and theoretical_match and reward_value_ok)
    if duration_status == "still_improving":
        recommendation = "Run a longer normal training pass before freezing the BC prior; controls can still be trusted."
    elif duration_status == "overfit_after_best":
        recommendation = "Use early stopping and improve cache scale/regularization; more epochs on this setup are harmful."
    elif ready:
        recommendation = "Use the best normal checkpoint as the phase-2 behavioral prior for imagination RL."
    else:
        recommendation = "Improve cache scale or architecture before imagination RL; current controls do not prove enough signal."
    return {
        "MIDTRAINING_THEORETICAL_TARGET_MATCHED": theoretical_match,
        "ACTION_LABEL_SIGNAL_VALIDATED": action_signal,
        "VISUAL_LATENT_SIGNAL_VALIDATED": visual_signal,
        "PREVIOUS_ACTION_CONTEXT_VALIDATED": action_context_signal,
        "REWARD_VALUE_HEADS_VALIDATED": reward_value_ok,
        "TRAINING_DURATION_STATUS": duration_status,
        "TRAINING_DURATION_PLATEAUED": duration_plateaued,
        "EARLY_STOPPING_RECOMMENDED": early_stopping_recommended,
        "BC_PRIOR_READY_FOR_IMAGINATION_RL": ready and duration_plateaued,
        "BC_PRIOR_USEFUL_BUT_TRAIN_LONGER": ready and not duration_plateaued,
        "RECOMMENDED_NEXT_STEP": recommendation,
    }


def analytical_action_baselines(
    sequence_cache: Path,
    *,
    context_len: int,
    mtp_horizon: int,
    val_ratio: float,
    seed: int,
    split_mode: str,
) -> dict[str, Any]:
    cache = load_sequence_cache(sequence_cache)
    anchors = build_valid_anchors(cache, context_len=context_len, mtp_horizon=mtp_horizon)
    if split_mode == "episode":
        _, val_anchors = split_anchors_by_episode(cache, anchors, val_ratio=val_ratio, seed=seed)
    elif split_mode == "episode_task":
        _, val_anchors = split_anchors_by_task_episode(cache, anchors, val_ratio=val_ratio, seed=seed)
    else:
        _, val_anchors = split_anchors(anchors, val_ratio=val_ratio, seed=seed)
    normalizer = compute_normalizer(cache)
    action = normalize_np(cache.action, normalizer.action_mean, normalizer.action_std)
    mean_errors: list[float] = []
    repeat_errors: list[float] = []
    positive_mean_errors: list[float] = []
    positive_repeat_errors: list[float] = []
    positive_steps = 0
    for anchor in val_anchors:
        anchor = int(anchor)
        target = action[anchor : anchor + mtp_horizon + 1]
        reward = cache.reward[anchor : anchor + mtp_horizon + 1]
        mean_pred = np.zeros_like(target)
        repeat_pred = np.repeat(action[anchor - 1][None, :], target.shape[0], axis=0)
        mean_errors.append(float(np.mean((mean_pred - target) ** 2)))
        repeat_errors.append(float(np.mean((repeat_pred - target) ** 2)))
        positive_mask = reward > 0.0
        if np.any(positive_mask):
            positive_steps += int(np.sum(positive_mask))
            positive_mean_errors.append(float(np.mean((mean_pred[positive_mask] - target[positive_mask]) ** 2)))
            positive_repeat_errors.append(float(np.mean((repeat_pred[positive_mask] - target[positive_mask]) ** 2)))
    return {
        "val_anchors": int(val_anchors.size),
        "mean_action_mse": float(np.mean(mean_errors)) if mean_errors else None,
        "repeat_previous_action_mse": float(np.mean(repeat_errors)) if repeat_errors else None,
        "positive_reward_steps": int(positive_steps),
        "positive_mean_action_mse": float(np.mean(positive_mean_errors)) if positive_mean_errors else None,
        "positive_repeat_previous_action_mse": float(np.mean(positive_repeat_errors)) if positive_repeat_errors else None,
    }


def write_report(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# SOAR Midtraining Validation",
        "",
        "Phase-2 validation for behavior cloning, reward, and value heads on frozen Kairos/Wan latents.",
        "",
        "## Decision",
        "",
    ]
    for key, value in summary["decision"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- Cache: `{summary['sequence_cache']}`",
            f"- Context len: `{summary['config']['context_len']}`",
            f"- MTP horizon: `{summary['config']['mtp_horizon']}`",
            f"- Split mode: `{summary['config']['split_mode']}`",
            f"- Seeds: `{summary['config']['seeds']}`",
            "",
            "## Normal Seed Runs",
            "",
            "| Run | Best Decision Action MSE | Epoch | All Action MSE | Reward MSE | Value MSE | Tail Gain |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in summary["runs"].items():
        if row["control_mode"] != "normal" or not label.startswith("normal_seed"):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{label}`",
                    fmt(row["best_val_decision_action_mse"]),
                    str(row["best_val_decision_action_epoch"]),
                    fmt(row["best_val_all_action_mse"]),
                    fmt(row["best_val_reward_mse"]),
                    fmt(row["best_val_value_mse"]),
                    fmt(row["tail_relative_gain"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Control Baselines",
            "",
            "| Mode | Best Decision Action MSE | Ratio vs Normal | Best Epoch |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode, ratio in summary["control_ratios_vs_normal"].items():
        row = next((item for item in summary["runs"].values() if item["control_mode"] == mode), None)
        if not row:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{mode}`",
                    fmt(row["best_val_decision_action_mse"]),
                    fmt(ratio),
                    str(row["best_val_decision_action_epoch"]),
                ]
            )
            + " |"
        )
    analytical = summary["analytical_action_baselines"]
    lines.extend(
        [
            "",
            "## Analytical Baselines",
            "",
            f"- Mean-action MSE: `{fmt(analytical.get('mean_action_mse'))}`",
            f"- Repeat-previous-action MSE: `{fmt(analytical.get('repeat_previous_action_mse'))}`",
            f"- Positive-reward mean-action MSE: `{fmt(analytical.get('positive_mean_action_mse'))}`",
            f"- Positive-reward repeat-previous-action MSE: `{fmt(analytical.get('positive_repeat_previous_action_mse'))}`",
            "",
            "## Claim Boundary",
            "",
            summary["claim_boundary"],
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_suite_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": summary["phase"],
        "config": summary["config"],
        "normal_seed_stats": summary["normal_seed_stats"],
        "control_ratios_vs_normal": summary["control_ratios_vs_normal"],
        "analytical_action_baselines": summary["analytical_action_baselines"],
        "decision": summary["decision"],
        "report": str(Path(summary["out_dir"]) / "report.md"),
    }


def parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
