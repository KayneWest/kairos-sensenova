#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "sensenova_drone_agent" / "scripts" / "run_pybullet_drones_imagination_policy.sh"


PRESETS: dict[str, dict[str, str]] = {
    "smoke": {
        "--collect-episodes": "2",
        "--max-steps": "16",
        "--world-epochs": "2",
        "--bc-epochs": "2",
        "--imagination-updates": "2",
        "--imagination-horizon": "2",
        "--batch-size": "8",
        "--hidden-dim": "64",
    },
    "small": {
        "--collect-episodes": "32",
        "--max-steps": "128",
        "--world-epochs": "100",
        "--bc-epochs": "80",
        "--imagination-updates": "250",
        "--imagination-horizon": "16",
        "--batch-size": "128",
        "--hidden-dim": "256",
    },
    "medium": {
        "--collect-episodes": "8",
        "--max-steps": "64",
        "--world-epochs": "50",
        "--bc-epochs": "40",
        "--imagination-updates": "100",
        "--imagination-horizon": "8",
        "--batch-size": "32",
        "--hidden-dim": "256",
    },
    "overnight": {
        "--collect-episodes": "96",
        "--max-steps": "160",
        "--world-epochs": "250",
        "--bc-epochs": "180",
        "--imagination-updates": "900",
        "--imagination-horizon": "24",
        "--batch-size": "256",
        "--hidden-dim": "384",
    },
}


FEATURE_ARGS: dict[str, list[str]] = {
    "kinematic": [
        "--feature",
        "kinematic",
    ],
    "rgb_downsample": [
        "--feature",
        "rgb_downsample",
        "--rgb-feature-size",
        "8",
        "--feature-stack",
        "4",
        "--feature-stack-deltas",
        "--include-prev-action-in-feature",
    ],
    "kairos_vae_flat": [
        "--feature",
        "kairos_vae_flat",
        "--feature-stack",
        "2",
        "--feature-stack-deltas",
        "--include-prev-action-in-feature",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-seed PyBullet visual/Kairos imagination suites with cached feature datasets."
        )
    )
    parser.add_argument("--out", default="sensenova_drone_agent/output/pybullet_drones_visual_imagination_suite_v1")
    parser.add_argument("--features", default="rgb_downsample,kairos_vae_flat")
    parser.add_argument("--seeds", default="170000")
    parser.add_argument("--eval-seeds", default="171000,171001,171002,171003,171004,171005")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--initial-xy-range", default="0.6")
    parser.add_argument("--initial-z-min", default="0.15")
    parser.add_argument("--initial-z-max", default="0.8")
    parser.add_argument("--behavior", default="random_mix")
    parser.add_argument("--random-action-prob", default="0.35")
    parser.add_argument("--behavior-noise", default="0.25")
    parser.add_argument("--dynamics-action-conditioning", choices=["concat", "action_token"], default="action_token")
    parser.add_argument("--dynamics-action-token-layers", default="2")
    parser.add_argument("--dynamics-action-token-heads", default="4")
    parser.add_argument("--world-training-mode", choices=["one_step", "sequence"], default="sequence")
    parser.add_argument("--sequence-length", default="8")
    parser.add_argument("--sequence-stride", default="1")
    parser.add_argument("--policy-lr", default="0.00005")
    parser.add_argument("--critic-lr", default="0.0005")
    parser.add_argument("--policy-std", default="0.08")
    parser.add_argument("--prior-weight", default="1.0")
    parser.add_argument("--lambda-return", default="0.95")
    parser.add_argument("--return-clip", default="20")
    parser.add_argument("--max-grad-norm", default="5.0")
    parser.add_argument("--kairos-device", default="cuda")
    parser.add_argument("--kairos-dtype", default="float32")
    parser.add_argument("--kairos-height", default="64")
    parser.add_argument("--kairos-width", default="64")
    parser.add_argument("--reuse-caches", action="store_true", default=True)
    parser.add_argument("--no-reuse-caches", action="store_false", dest="reuse_caches")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "runs"
    cache_dir = out_dir / "feature_caches"
    log_dir = out_dir / "logs"
    for path in (run_dir, cache_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    features = parse_csv(args.features)
    seeds = [int(item) for item in parse_csv(args.seeds)]
    unknown = sorted(set(features) - set(FEATURE_ARGS))
    if unknown:
        raise ValueError(f"Unknown feature(s): {unknown}. Valid: {sorted(FEATURE_ARGS)}")

    started = time.time()
    records = []
    commands = []
    for feature in features:
        for seed in seeds:
            record = run_suite_item(
                args=args,
                out_dir=out_dir,
                run_dir=run_dir,
                cache_dir=cache_dir,
                log_dir=log_dir,
                feature=feature,
                seed=seed,
            )
            records.append(record)
            commands.append(record["command"])

    summary = {
        "suite": "PyBullet visual/Kairos imagination fixed-seed suite",
        "elapsed_s": time.time() - started,
        "args": vars(args),
        "features": features,
        "train_seeds": seeds,
        "eval_seeds": [int(item) for item in parse_csv(args.eval_seeds)],
        "preset": PRESETS[args.preset],
        "records": records,
        "ranking": rank_records(records),
        "claim_boundary": (
            "This suite compares learned-simulator imagination policies under fixed PyBullet eval seeds. "
            "It is not proof of real drone autonomy or native Kairos action-conditioned simulation."
        ),
    }
    (out_dir / "suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "suite_commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary["ranking"], indent=2))
    print(f"Wrote {to_repo_cli_path(out_dir / 'suite_summary.json')}")
    return 0


def run_suite_item(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    run_dir: Path,
    cache_dir: Path,
    log_dir: Path,
    feature: str,
    seed: int,
) -> dict[str, Any]:
    item_name = f"{feature}_seed{seed}"
    item_out = run_dir / item_name
    cache_path = cache_dir / f"{item_name}.npz"
    stdout_path = log_dir / f"{item_name}.stdout.log"
    stderr_path = log_dir / f"{item_name}.stderr.log"
    command = build_command(args, item_out, cache_path, feature, seed)
    record: dict[str, Any] = {
        "feature": feature,
        "seed": seed,
        "out_dir": to_repo_cli_path(item_out),
        "cache_path": to_repo_cli_path(cache_path),
        "stdout_log": to_repo_cli_path(stdout_path),
        "stderr_log": to_repo_cli_path(stderr_path),
        "command": command,
        "status": "dry_run" if args.dry_run else "pending",
    }
    if args.dry_run:
        return record

    started = time.time()
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    record["elapsed_s"] = time.time() - started
    record["returncode"] = proc.returncode
    summary_path = item_out / "summary.json"
    if proc.returncode != 0:
        record["status"] = "failed"
        record["error"] = f"command exited with code {proc.returncode}"
        return record
    if not summary_path.exists():
        record["status"] = "failed"
        record["error"] = "summary.json was not created"
        return record
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record["status"] = "ok"
    record["summary_path"] = to_repo_cli_path(summary_path)
    record["dataset"] = summary.get("dataset", {})
    record["bc_prior_eval"] = summary.get("bc_prior_eval", {})
    record["after_imagination_eval"] = summary.get("after_imagination_eval", {})
    record["policy_selection"] = summary.get("policy_selection", {})
    record["imagination"] = summary.get("imagination", {})
    return record


def build_command(args: argparse.Namespace, out_dir: Path, cache_path: Path, feature: str, seed: int) -> list[str]:
    command = [
        str(WRAPPER),
        "--out-dir",
        to_repo_cli_path(out_dir),
        "--seed",
        str(seed),
        "--eval-seeds",
        args.eval_seeds,
        "--dataset-cache",
        to_repo_cli_path(cache_path),
    ]
    if args.reuse_caches and cache_path.exists():
        command.append("--reuse-dataset-cache")
    command.extend(FEATURE_ARGS[feature])
    if feature.startswith("kairos_"):
        command.extend(
            [
                "--kairos-device",
                args.kairos_device,
                "--kairos-dtype",
                args.kairos_dtype,
                "--kairos-height",
                args.kairos_height,
                "--kairos-width",
                args.kairos_width,
            ]
        )
    for key, value in PRESETS[args.preset].items():
        command.extend([key, value])
    command.extend(
        [
            "--initial-xy-range",
            args.initial_xy_range,
            "--initial-z-min",
            args.initial_z_min,
            "--initial-z-max",
            args.initial_z_max,
            "--behavior",
            args.behavior,
            "--random-action-prob",
            args.random_action_prob,
            "--behavior-noise",
            args.behavior_noise,
            "--dynamics-action-conditioning",
            args.dynamics_action_conditioning,
            "--dynamics-action-token-layers",
            args.dynamics_action_token_layers,
            "--dynamics-action-token-heads",
            args.dynamics_action_token_heads,
            "--world-training-mode",
            args.world_training_mode,
            "--sequence-length",
            args.sequence_length,
            "--sequence-stride",
            args.sequence_stride,
            "--imagination-objective",
            "pmpo",
            "--prior-weight",
            args.prior_weight,
            "--policy-lr",
            args.policy_lr,
            "--critic-lr",
            args.critic_lr,
            "--policy-std",
            args.policy_std,
            "--lambda-return",
            args.lambda_return,
            "--return-clip",
            args.return_clip,
            "--max-grad-norm",
            args.max_grad_norm,
        ]
    )
    return command


def rank_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_records = [record for record in records if record.get("status") == "ok"]

    def key(record: dict[str, Any]) -> tuple[float, float]:
        after = record.get("after_imagination_eval", {})
        return (
            float(after.get("success_rate", 0.0)),
            -float(after.get("mean_final_distance_m", float("inf"))),
        )

    ranked = sorted(ok_records, key=key, reverse=True)
    return [
        {
            "rank": idx + 1,
            "feature": record["feature"],
            "seed": record["seed"],
            "selected_actor": record.get("policy_selection", {}).get("selected_actor"),
            "bc_success": record.get("bc_prior_eval", {}).get("success_rate"),
            "after_success": record.get("after_imagination_eval", {}).get("success_rate"),
            "bc_final_distance_m": record.get("bc_prior_eval", {}).get("mean_final_distance_m"),
            "after_final_distance_m": record.get("after_imagination_eval", {}).get("mean_final_distance_m"),
            "summary_path": record.get("summary_path"),
        }
        for idx, record in enumerate(ranked)
    ]


def write_report(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# PyBullet Visual/Kairos Imagination Suite",
        "",
        "This suite runs fixed-seed learned-simulator imagination experiments for visual features.",
        "",
        "## Configuration",
        "",
        f"- Features: `{', '.join(summary['features'])}`",
        f"- Train seeds: `{', '.join(str(seed) for seed in summary['train_seeds'])}`",
        f"- Eval seeds: `{', '.join(str(seed) for seed in summary['eval_seeds'])}`",
        f"- Dynamics action conditioning: `{summary['args']['dynamics_action_conditioning']}`",
        f"- World training mode: `{summary['args']['world_training_mode']}`",
        f"- Records: `{len(summary['records'])}`",
        "",
        "## Ranking",
        "",
    ]
    if summary["ranking"]:
        for item in summary["ranking"]:
            lines.append(
                "- "
                f"#{item['rank']} `{item['feature']}` seed `{item['seed']}`: "
                f"success `{item['after_success']}`, final distance `{item['after_final_distance_m']}`, "
                f"selected `{item['selected_actor']}`"
            )
    else:
        lines.append("- No completed runs yet.")
    lines.extend(
        [
            "",
            "## Dreamer Reference Usage",
            "",
            "- Dreamer3 informs the train/eval/imagination split and fixed validation discipline.",
            "- Dreamer4 informs the separation between action-conditioned dynamics and policy/reward/value heads.",
            "- This suite is an adapter layer, not a direct Dreamer3/Dreamer4 port.",
            "",
            "## Claim Boundary",
            "",
            summary["claim_boundary"],
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must live under repo root {REPO_ROOT}: {path}") from exc
    return path


def to_repo_cli_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
