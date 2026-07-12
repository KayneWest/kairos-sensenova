#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated-seed PyBullet feature-policy comparisons and aggregate results."
    )
    parser.add_argument("--out-dir", default="sensenova_drone_agent/output/pybullet_drones_feature_policy_suite_v1")
    parser.add_argument(
        "--features",
        default="rgb_downsample,random_projection,cnn_pixels,resnet18_imagenet,kairos_vae_flat",
    )
    parser.add_argument("--seeds", default="", help="Comma-separated explicit seeds. Overrides --seed-start/--num-seeds.")
    parser.add_argument("--seed-start", type=int, default=150000)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--train-episodes", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--initial-xy-range", type=float, default=0.4)
    parser.add_argument("--initial-z-min", type=float, default=0.15)
    parser.add_argument("--initial-z-max", type=float, default=0.6)
    parser.add_argument("--kairos-device", default="cpu")
    parser.add_argument("--kairos-dtype", default="float32")
    parser.add_argument("--kairos-height", type=int, default=128)
    parser.add_argument("--kairos-width", type=int, default=128)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--eval-trace-frames", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args)
    started = time.time()
    run_summaries = []

    for idx, seed in enumerate(seeds):
        run_dir = out_dir / f"seed_{seed}"
        cmd = build_run_command(args=args, seed=seed, run_dir=run_dir)
        command_record = {
            "seed": seed,
            "run_dir": str(run_dir),
            "command": cmd,
        }
        print(f"[{idx + 1}/{len(seeds)}] seed={seed} -> {run_dir}", flush=True)
        (run_dir.parent / f"seed_{seed}_command.json").write_text(
            json.dumps(command_record, indent=2),
            encoding="utf-8",
        )
        if args.dry_run:
            print(" ".join(cmd), flush=True)
            continue
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Expected summary missing: {summary_path}")
        run_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    suite_summary = {
        "benchmark": "gym-pybullet-drones repeated-seed feature-policy suite",
        "elapsed_s": time.time() - started,
        "args": vars(args),
        "seeds": seeds,
        "runs": [
            {
                "seed": seed,
                "summary_path": str((out_dir / f"seed_{seed}" / "summary.json")),
            }
            for seed in seeds
        ],
        "aggregate": aggregate(run_summaries),
        "claim_boundary": (
            "Repeated-seed BC benchmark only. It is useful for representation comparison, "
            "not a claim of robust drone autonomy."
        ),
    }
    (out_dir / "suite_summary.json").write_text(json.dumps(suite_summary, indent=2), encoding="utf-8")
    write_report(suite_summary, out_dir / "suite_report.md")
    print(json.dumps(suite_summary["aggregate"], indent=2), flush=True)
    print(f"Wrote {out_dir / 'suite_summary.json'}", flush=True)
    return 0


def build_run_command(args: argparse.Namespace, seed: int, run_dir: Path) -> list[str]:
    return [
        str(PROJECT_ROOT / "scripts" / "run_pybullet_drones_feature_policy.sh"),
        "--out-dir",
        str(run_dir.relative_to(REPO_ROOT)),
        "--features",
        args.features,
        "--seed",
        str(seed),
        "--train-episodes",
        str(args.train_episodes),
        "--eval-episodes",
        str(args.eval_episodes),
        "--max-steps",
        str(args.max_steps),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-trace-frames",
        str(args.eval_trace_frames),
        "--initial-xy-range",
        str(args.initial_xy_range),
        "--initial-z-min",
        str(args.initial_z_min),
        "--initial-z-max",
        str(args.initial_z_max),
        "--kairos-device",
        args.kairos_device,
        "--kairos-dtype",
        args.kairos_dtype,
        "--kairos-height",
        str(args.kairos_height),
        "--kairos-width",
        str(args.kairos_width),
        "--torch-device",
        args.torch_device,
    ]


def aggregate(run_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_feature: dict[str, list[dict[str, Any]]] = {}
    for summary in run_summaries:
        for result in summary.get("results", []):
            by_feature.setdefault(result["feature"], []).append(result)

    rows = []
    for feature, results in by_feature.items():
        ok_results = [result for result in results if result.get("status") == "ok"]
        if not ok_results:
            rows.append(
                {
                    "feature": feature,
                    "runs": len(results),
                    "ok_runs": 0,
                    "status": "failed",
                    "errors": [result.get("error") for result in results],
                }
            )
            continue
        success = [float(result["eval"]["success_rate"]) for result in ok_results]
        distance = [float(result["eval"]["mean_final_distance_m"]) for result in ok_results]
        returns = [float(result["eval"]["mean_return"]) for result in ok_results]
        val_mse = [float(result["train"]["best_val_mse"]) for result in ok_results]
        rows.append(
            {
                "feature": feature,
                "runs": len(results),
                "ok_runs": len(ok_results),
                "success_rate_mean": safe_mean(success),
                "success_rate_std": safe_stdev(success),
                "mean_final_distance_m_mean": safe_mean(distance),
                "mean_final_distance_m_std": safe_stdev(distance),
                "mean_return_mean": safe_mean(returns),
                "mean_return_std": safe_stdev(returns),
                "best_val_mse_mean": safe_mean(val_mse),
                "best_val_mse_std": safe_stdev(val_mse),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("success_rate_mean", -1.0)),
            float(row.get("mean_final_distance_m_mean", 1e9)),
        ),
    )


def write_report(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# PyBullet Drone Feature Policy Suite",
        "",
        "Repeated-seed randomized-start behavior-cloning benchmark.",
        "",
        "## Configuration",
        "",
        f"- Seeds: `{summary['seeds']}`",
        f"- Features: `{summary['args']['features']}`",
        f"- Train episodes per seed: `{summary['args']['train_episodes']}`",
        f"- Eval episodes per seed: `{summary['args']['eval_episodes']}`",
        f"- Max steps: `{summary['args']['max_steps']}`",
        "",
        "## Aggregate Results",
        "",
        "| Feature | Runs | Success mean | Success std | Final distance mean | Final distance std | Return mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate"]:
        if row.get("status") == "failed":
            lines.append(f"| `{row['feature']}` | {row['runs']} | failed |  |  |  |  |")
            continue
        lines.append(
            "| `{feature}` | {ok_runs}/{runs} | {success_rate_mean:.3f} | {success_rate_std:.3f} | "
            "{mean_final_distance_m_mean:.4f} | {mean_final_distance_m_std:.4f} | "
            "{mean_return_mean:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            summary["claim_boundary"],
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds.strip():
        return [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    return [args.seed_start + idx * args.seed_stride for idx in range(args.num_seeds)]


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def safe_stdev(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def resolve_out_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
