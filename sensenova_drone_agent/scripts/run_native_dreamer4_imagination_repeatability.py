#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


DEFAULT_DATA_DIRS = [
    "sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4/expert",
    "sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4/mixed-small",
    "sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4/mixed-large",
]
DEFAULT_FRAME_DIRS = [
    "sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4_shards_full/expert",
    "sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4_shards_full/mixed-small",
    "sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4_shards_full/mixed-large",
]
DEFAULT_TASKS_JSON = "dreamer4/tasks.json"
DEFAULT_TOKENIZER_CKPT = (
    "sensenova_drone_agent/output/dreamer4_hf_rich_actions_strong_contrast_v1/"
    "tokenizer_ckpts/latest.pt"
)
DEFAULT_DYNAMICS_CKPT = (
    "sensenova_drone_agent/output/dreamer4_hf_rich_actions_strong_contrast_v1/"
    "dynamics_ckpts/latest.pt"
)
DEFAULT_EXISTING_RUN = "31=sensenova_drone_agent/output/native_dreamer4_imagination_calibrated_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeatability suite for the calibrated native Dreamer4 imagination result."
    )
    parser.add_argument(
        "--out-dir",
        default="sensenova_drone_agent/output/native_dreamer4_imagination_repeatability_v1",
    )
    parser.add_argument("--seeds", default="37,43", help="Comma-separated new seeds to run.")
    parser.add_argument(
        "--existing-runs",
        default=DEFAULT_EXISTING_RUN,
        help="Comma-separated seed=run_dir entries to include without rerunning. Use '' to disable.",
    )
    parser.add_argument("--docker-image", default="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
    parser.add_argument("--gpu", default="1", help="Docker GPU selector, e.g. 0, 1, or all.")
    parser.add_argument("--local", action="store_true", help="Run with the host Python instead of Docker.")
    parser.add_argument("--force", action="store_true", help="Rerun seeds even if summary.json exists.")
    parser.add_argument("--dry-run", action="store_true", help="Write planned commands without running.")

    parser.add_argument("--data-dir", action="append", default=None)
    parser.add_argument("--frames-dir", action="append", default=None)
    parser.add_argument("--tasks-json", default=DEFAULT_TASKS_JSON)
    parser.add_argument("--tokenizer-ckpt", default=DEFAULT_TOKENIZER_CKPT)
    parser.add_argument("--dynamics-ckpt", default=DEFAULT_DYNAMICS_CKPT)

    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--imagination-horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bc-steps", type=int, default=1200)
    parser.add_argument("--imagination-updates", type=int, default=400)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--action-dim", type=int, default=64)
    parser.add_argument("--action-features", default="current,prev,delta,mean4,norm")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--imagination-learning-rate", type=float, default=3e-5)
    parser.add_argument("--target-normalization", default="per_task")
    parser.add_argument("--reward-clip", type=float, default=5.0)
    parser.add_argument("--value-clip", type=float, default=5.0)
    parser.add_argument("--eval-holdout-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20260515)
    parser.add_argument("--imagination-mode", choices=["train", "no_update"], default="train")
    parser.add_argument("--advantage-mode", default="centered_sign")
    parser.add_argument("--advantage-baseline", choices=["value", "bc_return"], default="value")
    parser.add_argument("--advantage-clip", type=float, default=2.0)
    parser.add_argument("--policy-loss-min-advantage-abs", type=float, default=0.0)
    parser.add_argument("--policy-loss-max-prior-mse", type=float, default=0.0)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--prior-hinge-weight", type=float, default=25.0)
    parser.add_argument("--prior-hinge-target", type=float, default=0.008)
    parser.add_argument("--mean-prior-weight", type=float, default=10.0)
    parser.add_argument("--mean-prior-hinge-weight", type=float, default=100.0)
    parser.add_argument("--mean-prior-hinge-target", type=float, default=0.004)
    parser.add_argument("--value-loss-weight", type=float, default=0.10)
    parser.add_argument("--entropy-weight", type=float, default=0.0005)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--min-policy-minus-bc", type=float, default=0.0)
    parser.add_argument("--min-policy-minus-zero", type=float, default=0.0)
    parser.add_argument("--min-policy-return-delta", type=float, default=0.0)
    parser.add_argument("--max-prior-mse-after", type=float, default=0.006)
    parser.add_argument("--max-prior-mse-delta", type=float, default=0.002)
    parser.add_argument("--min-pass-fraction", type=float, default=2.0 / 3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_csv(args.seeds)
    existing_runs = parse_existing_runs(args.existing_runs)
    planned = {
        "phase": "native_dreamer4_imagination_repeatability",
        "out_dir": str(out_dir),
        "existing_runs": existing_runs,
        "new_seeds": seeds,
        "commands": [],
    }
    for seed in seeds:
        run_dir = out_dir / "runs" / f"seed_{seed}"
        planned["commands"].append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "command": build_command(args, seed, run_dir),
            }
        )
    write_json(out_dir / "planned_runs.json", planned)
    if args.dry_run:
        print(json.dumps(planned, indent=2))
        return 0

    runs: list[dict[str, Any]] = []
    for seed, existing_dir in existing_runs.items():
        runs.append(load_existing_run(seed, existing_dir, args))

    for seed in seeds:
        runs.append(run_seed(args, seed, out_dir))

    summary = summarize_suite(args, out_dir, runs, time.time() - started)
    write_json(out_dir / "summary.json", summary)
    write_report(summary, out_dir / "report.md")
    print(json.dumps(compact_summary(summary), indent=2))
    return 0


def run_seed(args: argparse.Namespace, seed: int, out_dir: Path) -> dict[str, Any]:
    run_dir = out_dir / "runs" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.force:
        return summarize_run(load_json(summary_path), seed=seed, source="new", run_dir=run_dir, args=args)

    command = build_command(args, seed, run_dir)
    write_json(run_dir / "command.json", command)
    started = time.time()
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
        run_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, text=True, check=False)

    metadata = {
        "seed": seed,
        "returncode": result.returncode,
        "elapsed_s": time.time() - started,
        "summary_path": str(summary_path),
    }
    write_json(run_dir / "runner_metadata.json", metadata)
    if result.returncode != 0 or not summary_path.exists():
        return {
            "seed": seed,
            "source": "new",
            "run_dir": str(run_dir),
            "status": "failed",
            "returncode": result.returncode,
            "elapsed_s": metadata["elapsed_s"],
            "pass": False,
            "failure": f"Training failed or missing summary.json. See {run_dir}.",
        }
    return summarize_run(load_json(summary_path), seed=seed, source="new", run_dir=run_dir, args=args)


def load_existing_run(seed: int, run_dir_str: str, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = resolve_path(run_dir_str)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {
            "seed": seed,
            "source": "existing",
            "run_dir": str(run_dir),
            "status": "missing",
            "pass": False,
            "failure": f"Missing summary.json at {summary_path}.",
        }
    return summarize_run(load_json(summary_path), seed=seed, source="existing", run_dir=run_dir, args=args)


def build_command(args: argparse.Namespace, seed: int, run_dir: Path) -> list[str]:
    train_args = [
        "sensenova_drone_agent/scripts/train_native_dreamer4_imagination.py",
        "--tasks-json",
        container_path(args.tasks_json, args.local),
        "--tokenizer-ckpt",
        container_path(args.tokenizer_ckpt, args.local),
        "--dynamics-ckpt",
        container_path(args.dynamics_ckpt, args.local),
        "--out-dir",
        container_path(repo_relative(run_dir), args.local),
        "--seq-len",
        str(args.seq_len),
        "--ctx-len",
        str(args.ctx_len),
        "--imagination-horizon",
        str(args.imagination_horizon),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--bc-steps",
        str(args.bc_steps),
        "--imagination-updates",
        str(args.imagination_updates),
        "--eval-batches",
        str(args.eval_batches),
        "--eval-seed",
        str(args.eval_seed),
        "--action-dim",
        str(args.action_dim),
        "--action-features",
        args.action_features,
        "--learning-rate",
        str(args.learning_rate),
        "--imagination-learning-rate",
        str(args.imagination_learning_rate),
        "--target-normalization",
        args.target_normalization,
        "--reward-clip",
        str(args.reward_clip),
        "--value-clip",
        str(args.value_clip),
        "--eval-holdout-fraction",
        str(args.eval_holdout_fraction),
        "--split-seed",
        str(args.split_seed),
        "--imagination-mode",
        args.imagination_mode,
        "--advantage-mode",
        args.advantage_mode,
        "--advantage-baseline",
        args.advantage_baseline,
        "--advantage-clip",
        str(args.advantage_clip),
        "--policy-loss-min-advantage-abs",
        str(args.policy_loss_min_advantage_abs),
        "--policy-loss-max-prior-mse",
        str(args.policy_loss_max_prior_mse),
        "--prior-weight",
        str(args.prior_weight),
        "--prior-hinge-weight",
        str(args.prior_hinge_weight),
        "--prior-hinge-target",
        str(args.prior_hinge_target),
        "--mean-prior-weight",
        str(args.mean_prior_weight),
        "--mean-prior-hinge-weight",
        str(args.mean_prior_hinge_weight),
        "--mean-prior-hinge-target",
        str(args.mean_prior_hinge_target),
        "--value-loss-weight",
        str(args.value_loss_weight),
        "--entropy-weight",
        str(args.entropy_weight),
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    for data_dir in args.data_dir or DEFAULT_DATA_DIRS:
        train_args.extend(["--data-dir", container_path(data_dir, args.local)])
    for frames_dir in args.frames_dir or DEFAULT_FRAME_DIRS:
        train_args.extend(["--frames-dir", container_path(frames_dir, args.local)])

    if args.local:
        return [sys.executable, *train_args]

    gpu_arg = "all" if args.gpu == "all" else f"device={args.gpu}"
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        gpu_arg,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "PYTHONPATH=/workspace/.pydeps:/workspace/dreamer4/dreamer4:/workspace/sensenova_drone_agent/scripts",
        "-v",
        f"{REPO_ROOT}:/workspace",
        "-w",
        "/workspace",
        args.docker_image,
        "python",
        *train_args,
    ]


def summarize_run(
    summary: dict[str, Any],
    *,
    seed: int,
    source: str,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    before = summary.get("before_imagination", {})
    after = summary.get("after_imagination", {})
    comparison = summary.get("comparison", {})
    after_per_task = after.get("per_task", {})
    prior_mse_after = as_float(after.get("policy_prior_mse"))
    prior_mse_delta = as_float(comparison.get("policy_prior_mse_delta"))
    after_policy_minus_bc = as_float(comparison.get("after_policy_minus_bc", after.get("policy_minus_bc")))
    after_policy_minus_zero = as_float(comparison.get("after_policy_minus_zero", after.get("policy_minus_zero")))
    policy_return_delta = as_float(comparison.get("policy_return_delta"))
    passed = (
        after_policy_minus_bc > args.min_policy_minus_bc
        and after_policy_minus_zero > args.min_policy_minus_zero
        and policy_return_delta > args.min_policy_return_delta
        and prior_mse_after <= args.max_prior_mse_after
        and prior_mse_delta <= args.max_prior_mse_delta
    )
    return {
        "seed": int(summary.get("config", {}).get("seed", seed)),
        "source": source,
        "run_dir": str(run_dir),
        "status": "completed",
        "pass": bool(passed),
        "before_policy_minus_bc": as_float(before.get("policy_minus_bc")),
        "after_policy_minus_bc": after_policy_minus_bc,
        "after_policy_minus_zero": after_policy_minus_zero,
        "policy_return_delta": policy_return_delta,
        "bc_prior_return_delta": as_float(comparison.get("bc_prior_return_delta")),
        "policy_prior_mse_before": as_float(before.get("policy_prior_mse")),
        "policy_prior_mse_after": prior_mse_after,
        "policy_prior_mse_delta": prior_mse_delta,
        "policy_action_abs_before": as_float(before.get("policy_action_abs")),
        "policy_action_abs_after": as_float(after.get("policy_action_abs")),
        "policy_action_abs_delta": as_float(comparison.get("policy_action_abs_delta")),
        "per_task_mean_policy_minus_bc": as_float(after_per_task.get("mean_policy_minus_bc")),
        "num_tasks_seen": int(after_per_task.get("num_tasks_seen", 0) or 0),
        "elapsed_s": as_float(summary.get("elapsed_s")),
    }


def summarize_suite(
    args: argparse.Namespace,
    out_dir: Path,
    runs: list[dict[str, Any]],
    elapsed_s: float,
) -> dict[str, Any]:
    completed = [row for row in runs if row.get("status") == "completed"]
    pass_count = sum(1 for row in completed if row.get("pass"))
    pass_fraction = pass_count / len(completed) if completed else 0.0
    metrics = {
        key: metric_stats([as_float(row.get(key)) for row in completed])
        for key in [
            "after_policy_minus_bc",
            "after_policy_minus_zero",
            "policy_return_delta",
            "policy_prior_mse_after",
            "policy_prior_mse_delta",
            "policy_action_abs_delta",
            "per_task_mean_policy_minus_bc",
        ]
    }
    repeatability_pass = bool(completed and pass_fraction >= args.min_pass_fraction and metrics["after_policy_minus_bc"]["mean"] > 0.0)
    return {
        "phase": "native_dreamer4_imagination_repeatability",
        "claim_boundary": (
            "This is repeatability evidence for policy improvement inside frozen learned Dreamer4-style "
            "dynamics. It is not a real-environment or drone-control claim."
        ),
        "out_dir": str(out_dir),
        "elapsed_s": elapsed_s,
        "thresholds": {
            "min_policy_minus_bc": args.min_policy_minus_bc,
            "min_policy_minus_zero": args.min_policy_minus_zero,
            "min_policy_return_delta": args.min_policy_return_delta,
            "max_prior_mse_after": args.max_prior_mse_after,
            "max_prior_mse_delta": args.max_prior_mse_delta,
            "min_pass_fraction": args.min_pass_fraction,
        },
        "config": {
            "seeds": parse_int_csv(args.seeds),
            "existing_runs": parse_existing_runs(args.existing_runs),
            "docker_image": None if args.local else args.docker_image,
            "gpu": None if args.local else args.gpu,
            "recipe": {
                "seq_len": args.seq_len,
                "ctx_len": args.ctx_len,
                "imagination_horizon": args.imagination_horizon,
                "bc_steps": args.bc_steps,
                "imagination_updates": args.imagination_updates,
                "eval_batches": args.eval_batches,
                "eval_seed": args.eval_seed,
                "action_dim": args.action_dim,
                "action_features": args.action_features,
                "learning_rate": args.learning_rate,
                "imagination_learning_rate": args.imagination_learning_rate,
                "target_normalization": args.target_normalization,
                "eval_holdout_fraction": args.eval_holdout_fraction,
                "split_seed": args.split_seed,
                "imagination_mode": args.imagination_mode,
                "advantage_mode": args.advantage_mode,
                "advantage_baseline": args.advantage_baseline,
                "policy_loss_min_advantage_abs": args.policy_loss_min_advantage_abs,
                "policy_loss_max_prior_mse": args.policy_loss_max_prior_mse,
                "prior_weight": args.prior_weight,
                "mean_prior_weight": args.mean_prior_weight,
            },
        },
        "runs": runs,
        "aggregate": {
            "completed_count": len(completed),
            "pass_count": pass_count,
            "pass_fraction": pass_fraction,
            "metrics": metrics,
        },
        "decision": {
            "repeatability_pass": repeatability_pass,
            "recommended_next_step": (
                "Add held-out rollout visualization and an ablation with policy-update disabled."
                if repeatability_pass
                else "Do not advance to transfer yet; tune the imagination objective and rerun seeds."
            ),
        },
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    aggregate = summary["aggregate"]
    decision = summary["decision"]
    lines = [
        "# Native Dreamer4 Imagination Repeatability",
        "",
        "## Decision",
        f"- Repeatability pass: `{json_bool(decision['repeatability_pass'])}`",
        f"- Completed runs: `{aggregate['completed_count']}`",
        f"- Passing runs: `{aggregate['pass_count']}`",
        f"- Pass fraction: `{aggregate['pass_fraction']:.3f}`",
        f"- Recommended next step: {decision['recommended_next_step']}",
        "",
        "## Aggregate Metrics",
    ]
    for key, stats in aggregate["metrics"].items():
        lines.append(
            f"- {key}: mean `{format_float(stats['mean'])}`, std `{format_float(stats['std'])}`, "
            f"min `{format_float(stats['min'])}`, max `{format_float(stats['max'])}`"
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            (
                "| seed | source | status | pass | after policy-BC | after policy-zero | "
                "policy delta | prior MSE after | prior MSE delta | per-task mean | run |"
            ),
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["runs"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("seed", "")),
                    str(row.get("source", "")),
                    str(row.get("status", "")),
                    f"`{json_bool(bool(row.get('pass')))}`",
                    format_float(row.get("after_policy_minus_bc")),
                    format_float(row.get("after_policy_minus_zero")),
                    format_float(row.get("policy_return_delta")),
                    format_float(row.get("policy_prior_mse_after")),
                    format_float(row.get("policy_prior_mse_delta")),
                    format_float(row.get("per_task_mean_policy_minus_bc")),
                    f"`{row.get('run_dir', '')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            summary["claim_boundary"],
            "",
            "This validates whether the calibrated policy-update recipe repeats inside the learned dynamics. "
            "It still does not show real robot, SOAR environment, Gazebo, or drone transfer.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": summary["phase"],
        "out_dir": summary["out_dir"],
        "decision": summary["decision"],
        "aggregate": summary["aggregate"],
        "runs": [
            {
                "seed": row.get("seed"),
                "source": row.get("source"),
                "status": row.get("status"),
                "pass": row.get("pass"),
                "after_policy_minus_bc": row.get("after_policy_minus_bc"),
                "policy_return_delta": row.get("policy_return_delta"),
                "policy_prior_mse_after": row.get("policy_prior_mse_after"),
            }
            for row in summary["runs"]
        ],
    }


def metric_stats(values: list[float]) -> dict[str, float | None]:
    valid = [value for value in values if math.isfinite(value)]
    if not valid:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(valid),
        "mean": mean(valid),
        "std": pstdev(valid) if len(valid) > 1 else 0.0,
        "min": min(valid),
        "max": max(valid),
    }


def parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_existing_runs(value: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        seed_str, sep, path = item.partition("=")
        if not sep:
            raise ValueError(f"existing run must use seed=path form: {item}")
        out[int(seed_str)] = path
    return out


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def container_path(path: str | Path, local: bool) -> str:
    text = str(path)
    if local:
        return text
    path_obj = Path(text)
    if path_obj.is_absolute():
        try:
            return "/workspace/" + str(path_obj.resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            return text
    return text


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def format_float(value: Any) -> str:
    val = as_float(value)
    if not math.isfinite(val):
        return "n/a"
    return f"{val:.6f}"


def json_bool(value: bool) -> str:
    return "true" if value else "false"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
