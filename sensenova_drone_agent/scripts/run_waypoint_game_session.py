#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_ROOT = PROJECT_ROOT / "output" / "waypoint_game_session"
DEFAULT_EPISODES_ROOT = PROJECT_ROOT / "data" / "waypoint_game" / "episodes"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "waypoint_game" / "manifests" / "bc_manifest.jsonl"
DEFAULT_MANIFEST_SUMMARY = PROJECT_ROOT / "data" / "waypoint_game" / "manifests" / "bc_manifest_summary.json"
DEFAULT_TRAIN_OUT_DIR = PROJECT_ROOT / "output" / "bc_policy_waypoint_game"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "output" / "closed_loop_eval_waypoint_game"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "waypoint_game_sessions"
DEFAULT_PROGRESS_OUT = PROJECT_ROOT / "output" / "waypoint_game_progress_report"
DEFAULT_BEHAVIOR_OUT = PROJECT_ROOT / "output" / "waypoint_game_behavior_report"
DEFAULT_EVAL_OUT = PROJECT_ROOT / "output" / "waypoint_game_eval_report"
DEFAULT_DASHBOARD_OUT = PROJECT_ROOT / "output" / "waypoint_game_dashboard"
DEFAULT_GAZEBO_TOPIC = "/world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
DEFAULT_DEPTH_TOPIC = "/depth_camera"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gazebo-topic", default=DEFAULT_GAZEBO_TOPIC)
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--episodes-per-cycle", type=int, default=4)
    parser.add_argument("--teacher-steps-per-episode", type=int, default=8)
    parser.add_argument("--epochs-per-cycle", type=int, default=8)
    parser.add_argument("--eval-episodes-per-cycle", type=int, default=1)
    parser.add_argument("--eval-steps-per-episode", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--teacher-target-families", default="forward,left,right")
    parser.add_argument("--teacher-min-decision-score", type=float, default=0.55)
    parser.add_argument("--teacher-goal-resample-attempts", type=int, default=8)
    parser.add_argument("--teacher-max-goal-vertical-m", type=float, default=0.0)
    parser.add_argument("--episodes-root", default=str(DEFAULT_EPISODES_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--summary-json", default=str(DEFAULT_MANIFEST_SUMMARY))
    parser.add_argument("--train-out-dir", default=str(DEFAULT_TRAIN_OUT_DIR))
    parser.add_argument("--eval-out-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--progress-out-dir", default=str(DEFAULT_PROGRESS_OUT))
    parser.add_argument("--behavior-out-dir", default=str(DEFAULT_BEHAVIOR_OUT))
    parser.add_argument("--eval-report-out-dir", default=str(DEFAULT_EVAL_OUT))
    parser.add_argument("--dashboard-out-dir", default=str(DEFAULT_DASHBOARD_OUT))
    parser.add_argument("--i-understand-this-is-sitl", action="store_true", dest="i_understand_this_is_sitl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.i_understand_this_is_sitl:
        raise SystemExit("Refusing to run without --i-understand-this-is-sitl")

    _run(
        [
            "python3",
            "scripts/run_overnight_bc_routine.py",
            "--gazebo-topic",
            args.gazebo_topic,
            "--depth-topic",
            args.depth_topic,
            "--collector-policy",
            "reactive_teacher",
            "--teacher-goal-mode",
            "decision_game",
            "--teacher-target-families",
            args.teacher_target_families,
            "--teacher-goal-resample-attempts",
            str(args.teacher_goal_resample_attempts),
            "--teacher-min-decision-score",
            str(args.teacher_min_decision_score),
            "--teacher-refresh-goal-every-step",
            "--teacher-max-goal-vertical-m",
            str(args.teacher_max_goal_vertical_m),
            "--cycles",
            str(args.cycles),
            "--episodes-per-cycle",
            str(args.episodes_per_cycle),
            "--teacher-steps-per-episode",
            str(args.teacher_steps_per_episode),
            "--epochs-per-cycle",
            str(args.epochs_per_cycle),
            "--eval-episodes-per-cycle",
            str(args.eval_episodes_per_cycle),
            "--eval-steps-per-episode",
            str(args.eval_steps_per_episode),
            "--batch-size",
            str(args.batch_size),
            "--image-size",
            str(args.image_size),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--episodes-root",
            args.episodes_root,
            "--manifest",
            args.manifest,
            "--summary-json",
            args.summary_json,
            "--train-out-dir",
            args.train_out_dir,
            "--eval-out-root",
            args.eval_out_root,
            "--log-root",
            args.log_root,
            "--i-understand-this-is-sitl",
        ]
    )

    _run(
        [
            "python3",
            "scripts/report_episode_behavior.py",
            "--episodes-root",
            args.episodes_root,
            "--out-dir",
            args.behavior_out_dir,
        ]
    )
    _run(
        [
            "python3",
            "scripts/report_bc_progress.py",
            "--log-root",
            args.log_root,
            "--episodes-root",
            args.episodes_root,
            "--manifest-summary",
            args.summary_json,
            "--baseline-summary",
            str(Path(args.train_out_dir) / "train_summary.json"),
            "--out-dir",
            args.progress_out_dir,
        ]
    )
    _run(
        [
            "python3",
            "scripts/report_closed_loop_eval.py",
            "--eval-root",
            args.eval_out_root,
            "--out-dir",
            args.eval_report_out_dir,
        ]
    )
    _run(
        [
            "python3",
            "scripts/report_training_dashboard.py",
            "--manifest-summary",
            args.summary_json,
            "--baseline-summary",
            str(Path(args.train_out_dir) / "train_summary.json"),
            "--progress-report",
            str(Path(args.progress_out_dir) / "index.html"),
            "--behavior-report",
            str(Path(args.behavior_out_dir) / "index.html"),
            "--eval-report",
            str(Path(args.eval_report_out_dir) / "index.html"),
            "--eval-summary",
            str(Path(args.eval_report_out_dir) / "dashboard_summary.json"),
            "--out-dir",
            args.dashboard_out_dir,
        ]
    )

    payload = {
        "episodes_root": str(Path(args.episodes_root).resolve()),
        "manifest_summary": str(Path(args.summary_json).resolve()),
        "train_summary": str((Path(args.train_out_dir) / "train_summary.json").resolve()),
        "behavior_report": str((Path(args.behavior_out_dir) / "index.html").resolve()),
        "progress_report": str((Path(args.progress_out_dir) / "index.html").resolve()),
        "eval_report": str((Path(args.eval_report_out_dir) / "index.html").resolve()),
        "dashboard_report": str((Path(args.dashboard_out_dir) / "index.html").resolve()),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
