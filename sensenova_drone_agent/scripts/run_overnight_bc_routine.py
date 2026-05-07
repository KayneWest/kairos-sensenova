#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sensenova_drone.actions import DiscreteDroneAction


DEFAULT_GAZEBO_TOPIC = "/world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
DEFAULT_DEPTH_TOPIC = "/depth_camera"
DEFAULT_EPISODES_ROOT = PROJECT_ROOT / "data" / "bc_sft" / "episodes"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "bc_sft" / "manifests" / "bc_manifest.jsonl"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "bc_sft" / "manifests" / "bc_manifest_summary.json"
DEFAULT_TRAIN_OUT_DIR = PROJECT_ROOT / "output" / "bc_policy_baseline"
DEFAULT_EVAL_OUT_ROOT = PROJECT_ROOT / "output" / "closed_loop_eval"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "overnight_bc"


CURRICULA: dict[str, list[DiscreteDroneAction]] = {
    "forward_yaw_mix": [
        DiscreteDroneAction.HOVER,
        DiscreteDroneAction.FORWARD,
        DiscreteDroneAction.YAW_LEFT,
        DiscreteDroneAction.FORWARD,
        DiscreteDroneAction.YAW_RIGHT,
        DiscreteDroneAction.HOVER,
    ],
    "backward_strafe_mix": [
        DiscreteDroneAction.HOVER,
        DiscreteDroneAction.BACKWARD,
        DiscreteDroneAction.STRAFE_LEFT,
        DiscreteDroneAction.STRAFE_RIGHT,
        DiscreteDroneAction.BACKWARD,
        DiscreteDroneAction.HOVER,
    ],
    "altitude_mix": [
        DiscreteDroneAction.HOVER,
        DiscreteDroneAction.ASCEND,
        DiscreteDroneAction.FORWARD,
        DiscreteDroneAction.DESCEND,
        DiscreteDroneAction.BACKWARD,
        DiscreteDroneAction.HOVER,
    ],
    "turn_heavy": [
        DiscreteDroneAction.HOVER,
        DiscreteDroneAction.YAW_LEFT,
        DiscreteDroneAction.YAW_LEFT,
        DiscreteDroneAction.FORWARD,
        DiscreteDroneAction.YAW_RIGHT,
        DiscreteDroneAction.HOVER,
    ],
    "lateral_mix": [
        DiscreteDroneAction.HOVER,
        DiscreteDroneAction.FORWARD,
        DiscreteDroneAction.STRAFE_LEFT,
        DiscreteDroneAction.FORWARD,
        DiscreteDroneAction.STRAFE_RIGHT,
        DiscreteDroneAction.HOVER,
    ],
    "rare_action_mix": [
        DiscreteDroneAction.HOVER,
        DiscreteDroneAction.ASCEND,
        DiscreteDroneAction.FORWARD,
        DiscreteDroneAction.DESCEND,
        DiscreteDroneAction.BACKWARD,
        DiscreteDroneAction.STRAFE_LEFT,
        DiscreteDroneAction.STRAFE_RIGHT,
        DiscreteDroneAction.YAW_RIGHT,
        DiscreteDroneAction.HOVER,
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gazebo-topic", default=DEFAULT_GAZEBO_TOPIC)
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--collector-policy", choices=["scripted", "reactive_teacher"], default="reactive_teacher")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--episodes-per-cycle", type=int, default=6)
    parser.add_argument("--teacher-steps-per-episode", type=int, default=8)
    parser.add_argument("--teacher-goal-mode", choices=["random", "decision_game"], default="random")
    parser.add_argument("--teacher-target-families", default="forward,left,right")
    parser.add_argument("--teacher-goal-resample-attempts", type=int, default=8)
    parser.add_argument("--teacher-min-decision-score", type=float, default=0.55)
    parser.add_argument("--teacher-refresh-goal-every-step", action="store_true")
    parser.add_argument("--teacher-max-goal-vertical-m", type=float, default=0.0)
    parser.add_argument("--epochs-per-cycle", type=int, default=20)
    parser.add_argument("--eval-episodes-per-cycle", type=int, default=1)
    parser.add_argument("--eval-steps-per-episode", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--episodes-root", default=str(DEFAULT_EPISODES_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--train-out-dir", default=str(DEFAULT_TRAIN_OUT_DIR))
    parser.add_argument("--eval-out-root", default=str(DEFAULT_EVAL_OUT_ROOT))
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--bridge-wait-s", type=float, default=4.0)
    parser.add_argument("--command-duration-s", type=float, default=0.75)
    parser.add_argument("--settle-duration-s", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--sleep-between-episodes-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-connection-check", action="store_true")
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--i-understand-this-is-sitl", action="store_true", dest="i_understand_this_is_sitl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.i_understand_this_is_sitl:
        raise SystemExit("Refusing to run without --i-understand-this-is-sitl")

    run_id = datetime.now(timezone.utc).strftime("overnight_bc_%Y%m%dT%H%M%SZ")
    run_dir = Path(args.log_root).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    summary: dict[str, object] = {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "collector_policy": args.collector_policy,
        "teacher_goal_mode": args.teacher_goal_mode,
        "teacher_target_families": args.teacher_target_families,
        "cycles": [],
        "failed_episode_count": 0,
    }

    if not args.skip_connection_check:
        _run_command(
            [
                "docker",
                "compose",
                "run",
                "--rm",
                "tools",
                "python3",
                "scripts/verify_mavsdk_connection.py",
            ],
            cwd=PROJECT_ROOT,
            stdout_path=run_dir / "verify_connection.stdout.log",
            stderr_path=run_dir / "verify_connection.stderr.log",
        )

    failures = 0
    curriculum_names = list(CURRICULA.keys())

    for cycle_index in range(args.cycles):
        cycle_dir = run_dir / f"cycle_{cycle_index:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        if args.collector_policy == "scripted":
            chosen_curricula = [rng.choice(curriculum_names) for _ in range(args.episodes_per_cycle)]
        else:
            chosen_curricula = ["reactive_teacher" for _ in range(args.episodes_per_cycle)]
        cycle_record = {
            "cycle_index": cycle_index,
            "curricula": chosen_curricula,
            "collector_policy": args.collector_policy,
            "teacher_goal_mode": args.teacher_goal_mode,
            "teacher_target_families": args.teacher_target_families,
            "episode_runs": [],
        }

        for episode_index, curriculum_name in enumerate(chosen_curricula):
            episode_id = f"{run_id}_c{cycle_index:03d}_e{episode_index:03d}_{curriculum_name}"
            stdout_path = cycle_dir / f"{episode_id}.stdout.log"
            stderr_path = cycle_dir / f"{episode_id}.stderr.log"
            collector_command = [
                "docker",
                "compose",
                "run",
                "--rm",
                "tools",
                "python3",
                "scripts/collect_sitl_bc_episode.py",
                "--gazebo-topic",
                args.gazebo_topic,
                "--episodes-root",
                _path_for_tools_container(Path(args.episodes_root).expanduser()),
                "--episode-id",
                episode_id,
                "--policy",
                args.collector_policy,
                "--bridge-wait-s",
                str(args.bridge_wait_s),
                "--command-duration-s",
                str(args.command_duration_s),
                "--settle-duration-s",
                str(args.settle_duration_s),
                "--timeout",
                str(args.timeout),
                "--i-understand-this-is-sitl",
            ]
            if args.collector_policy == "scripted":
                actions = ",".join(action.value for action in CURRICULA[curriculum_name])
                collector_command.extend(["--actions", actions])
            else:
                collector_command.extend(
                    [
                        "--depth-topic",
                        args.depth_topic,
                        "--num-steps",
                        str(args.teacher_steps_per_episode),
                        "--teacher-seed",
                        str(args.seed + cycle_index * 1_000 + episode_index),
                        "--teacher-goal-mode",
                        args.teacher_goal_mode,
                        "--teacher-target-families",
                        args.teacher_target_families,
                        "--teacher-goal-resample-attempts",
                        str(args.teacher_goal_resample_attempts),
                        "--teacher-min-decision-score",
                        str(args.teacher_min_decision_score),
                        "--teacher-max-goal-vertical-m",
                        str(args.teacher_max_goal_vertical_m),
                    ]
                )
                if args.teacher_refresh_goal_every_step:
                    collector_command.append("--teacher-refresh-goal-every-step")

            try:
                _run_command(
                    collector_command,
                    cwd=PROJECT_ROOT,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                cycle_record["episode_runs"].append(
                    {
                        "episode_id": episode_id,
                        "curriculum": curriculum_name,
                        "status": "completed",
                    }
                )
            except subprocess.CalledProcessError as exc:
                failures += 1
                cycle_record["episode_runs"].append(
                    {
                        "episode_id": episode_id,
                        "curriculum": curriculum_name,
                        "status": "failed",
                        "returncode": exc.returncode,
                    }
                )
                summary["failed_episode_count"] = failures
                if failures >= args.max_failures:
                    cycle_record["aborted"] = True
                    summary["cycles"].append(cycle_record)
                    _write_summary(run_dir / "summary.json", summary)
                    raise RuntimeError(
                        f"Overnight routine aborted after {failures} collection failures."
                    ) from exc
            time.sleep(max(args.sleep_between_episodes_s, 0.0))

        _run_command(
            [
                "python3",
                "scripts/export_bc_dataset.py",
                "--episodes-root",
                str(Path(args.episodes_root).expanduser()),
                "--out-jsonl",
                str(Path(args.manifest).expanduser()),
                "--summary-json",
                str(Path(args.summary_json).expanduser()),
                "--val-ratio",
                str(args.val_ratio),
            ],
            cwd=PROJECT_ROOT,
            stdout_path=cycle_dir / "export.stdout.log",
            stderr_path=cycle_dir / "export.stderr.log",
        )

        _run_command(
            [
                "python3",
                "scripts/train_bc_policy.py",
                "--manifest",
                str(Path(args.manifest).expanduser()),
                "--out-dir",
                str(Path(args.train_out_dir).expanduser()),
                "--epochs",
                str(args.epochs_per_cycle),
                "--batch-size",
                str(args.batch_size),
                "--image-size",
                str(args.image_size),
                "--device",
                args.device,
                "--seed",
                str(args.seed + cycle_index),
            ],
            cwd=PROJECT_ROOT,
            stdout_path=cycle_dir / "train.stdout.log",
            stderr_path=cycle_dir / "train.stderr.log",
        )

        if args.eval_episodes_per_cycle > 0:
            _run_command(
                [
                    "docker",
                    "compose",
                    "run",
                    "--rm",
                    "tools",
                    "python3",
                    "scripts/eval_bc_policy_closed_loop.py",
                    "--checkpoint",
                    _path_for_tools_container(Path(args.train_out_dir).expanduser() / "best.pt"),
                    "--gazebo-topic",
                    args.gazebo_topic,
                    "--depth-topic",
                    args.depth_topic,
                    "--out-root",
                    _path_for_tools_container(Path(args.eval_out_root).expanduser()),
                    "--run-id",
                    f"{run_id}_cycle_{cycle_index:03d}",
                    "--episodes",
                    str(args.eval_episodes_per_cycle),
                    "--num-steps",
                    str(args.eval_steps_per_episode),
                    "--bridge-wait-s",
                    str(args.bridge_wait_s),
                    "--timeout",
                    str(args.timeout),
                    "--device",
                    args.device,
                    "--goal-seed",
                    str(args.seed + cycle_index * 10_000),
                    "--i-understand-this-is-sitl",
                ],
                cwd=PROJECT_ROOT,
                stdout_path=cycle_dir / "eval.stdout.log",
                stderr_path=cycle_dir / "eval.stderr.log",
            )

        _run_command(
            ["python3", "scripts/report_closed_loop_eval.py"],
            cwd=PROJECT_ROOT,
            stdout_path=cycle_dir / "report_closed_loop_eval.stdout.log",
            stderr_path=cycle_dir / "report_closed_loop_eval.stderr.log",
        )
        _run_command(
            ["python3", "scripts/report_episode_behavior.py"],
            cwd=PROJECT_ROOT,
            stdout_path=cycle_dir / "report_episode_behavior.stdout.log",
            stderr_path=cycle_dir / "report_episode_behavior.stderr.log",
        )
        _run_command(
            ["python3", "scripts/report_bc_progress.py"],
            cwd=PROJECT_ROOT,
            stdout_path=cycle_dir / "report_bc_progress.stdout.log",
            stderr_path=cycle_dir / "report_bc_progress.stderr.log",
        )
        _run_command(
            ["python3", "scripts/report_training_dashboard.py"],
            cwd=PROJECT_ROOT,
            stdout_path=cycle_dir / "report_training_dashboard.stdout.log",
            stderr_path=cycle_dir / "report_training_dashboard.stderr.log",
        )

        cycle_record["status"] = "completed"
        summary["cycles"].append(cycle_record)
        _write_summary(run_dir / "summary.json", summary)

    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_summary(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _path_for_tools_container(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return str(path)
    return str(relative)


if __name__ == "__main__":
    raise SystemExit(main())
