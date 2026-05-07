#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
DEFAULT_GAZEBO_TOPIC_TEMPLATE = "/world/{world}/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
DEFAULT_DEPTH_TOPIC = "/depth_camera"
DEFAULT_EPISODES_ROOT = PROJECT_ROOT / "data" / "waypoint_game_multiscene" / "episodes"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "waypoint_game_multiscene" / "manifests" / "bc_manifest.jsonl"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "waypoint_game_multiscene" / "manifests" / "bc_manifest_summary.json"
DEFAULT_TRAIN_OUT_DIR = PROJECT_ROOT / "output" / "bc_policy_waypoint_multiscene"
DEFAULT_EVAL_OUT_ROOT = PROJECT_ROOT / "output" / "closed_loop_eval_waypoint_multiscene"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "waypoint_multiscene_sessions"
DEFAULT_PROGRESS_OUT = PROJECT_ROOT / "output" / "waypoint_multiscene_progress_report"
DEFAULT_BEHAVIOR_OUT = PROJECT_ROOT / "output" / "waypoint_multiscene_behavior_report"
DEFAULT_EVAL_REPORT_OUT = PROJECT_ROOT / "output" / "waypoint_multiscene_eval_report"
DEFAULT_DASHBOARD_OUT = PROJECT_ROOT / "output" / "waypoint_multiscene_dashboard"
SIM_IMAGE = "sensenova_drone_agent-px4-source-sim:local"
TOOLS_RUN_PREFIX = "sensenova_drone_agent-tools-run-"
TOOLS_SESSION_PREFIX = "sensenova_drone_agent-tools-session-"


@dataclass(frozen=True)
class Scenario:
    label: str
    world: str
    pose: str
    target_families: str
    max_goal_vertical_m: float = 0.0


SCENARIOS_DIVERSE_V1: list[Scenario] = [
    Scenario("forest_north", "forest", "6,0,1.8,0,0,1.5708", "forward,left,right"),
    Scenario("forest_east", "forest", "6,0,1.8,0,0,0.0", "forward,left,right"),
    Scenario("forest_southwest", "forest", "8,-2,1.8,0,0,2.35619", "forward,left,right"),
    Scenario("forest_altitude", "forest", "6,0,2.3,0,0,-1.5708", "left,right,ascend,descend", 1.2),
    Scenario("walls_east", "walls", "2,0,1.8,0,0,0.0", "forward,left,right"),
    Scenario("walls_west", "walls", "2,0,1.8,0,0,3.14159", "forward,left,right"),
    Scenario("walls_offset", "walls", "4,1,1.8,0,0,-1.5708", "forward,left,right"),
]

SCENARIOS_FOREST_AVOIDANCE_V1: list[Scenario] = [
    Scenario("forest_north", "forest", "6,0,1.8,0,0,1.5708", "left,right"),
    Scenario("forest_east", "forest", "6,0,1.8,0,0,0.0", "left,right"),
    Scenario("forest_south", "forest", "6,0,1.8,0,0,-1.5708", "left,right"),
    Scenario("forest_west", "forest", "6,0,1.8,0,0,3.14159", "left,right"),
    Scenario("forest_northeast", "forest", "8,1,1.8,0,0,0.7854", "left,right"),
    Scenario("forest_northwest", "forest", "8,-1,1.8,0,0,2.35619", "left,right"),
    Scenario("forest_southeast", "forest", "4,1,1.8,0,0,-0.7854", "left,right"),
    Scenario("forest_southwest", "forest", "8,-2,1.8,0,0,2.35619", "left,right"),
    Scenario("forest_offset_left", "forest", "10,-1,1.8,0,0,1.5708", "left,right"),
    Scenario("forest_offset_right", "forest", "8,1,1.8,0,0,-1.5708", "left,right"),
    Scenario("forest_close_north", "forest", "4,0,1.8,0,0,1.5708", "left,right"),
    Scenario("forest_close_east", "forest", "4,0,1.8,0,0,0.0", "left,right"),
    Scenario("forest_diagonal_a", "forest", "5,-1,1.8,0,0,1.0472", "left,right"),
    Scenario("forest_diagonal_b", "forest", "5,1,1.8,0,0,-1.0472", "left,right"),
    Scenario("forest_altitude_hi", "forest", "6,0,2.5,0,0,-1.5708", "left,right,ascend,descend", 1.2),
    Scenario("forest_altitude_mid", "forest", "8,0,2.2,0,0,0.0", "left,right,ascend,descend", 1.0),
    Scenario("forest_altitude_low", "forest", "6,-1,1.5,0,0,1.5708", "left,right,ascend", 1.0),
]

SCENARIOS_FOREST_AVOIDANCE_STABLE_V1: list[Scenario] = [
    Scenario("forest_north", "forest", "6,0,1.8,0,0,1.5708", "left,right"),
    Scenario("forest_east", "forest", "6,0,1.8,0,0,0.0", "left,right"),
    Scenario("forest_south", "forest", "6,0,1.8,0,0,-1.5708", "left,right"),
    Scenario("forest_west", "forest", "6,0,1.8,0,0,3.14159", "left,right"),
    Scenario("forest_northeast", "forest", "8,1,1.8,0,0,0.7854", "left,right"),
    Scenario("forest_northwest", "forest", "8,-1,1.8,0,0,2.35619", "left,right"),
    Scenario("forest_southeast", "forest", "4,1,1.8,0,0,-0.7854", "left,right"),
    Scenario("forest_southwest", "forest", "8,-2,1.8,0,0,2.35619", "left,right"),
]

SCENARIOS_FOREST_AVOIDANCE_ASYMMETRIC_V2: list[Scenario] = [
    Scenario("forest_east", "forest", "6,0,1.8,0,0,0.0", "left,right"),
    Scenario("forest_west", "forest", "6,0,1.8,0,0,3.14159", "left,right"),
    Scenario("forest_northeast", "forest", "8,1,1.8,0,0,0.7854", "left,right"),
    Scenario("forest_northwest", "forest", "8,-1,1.8,0,0,2.35619", "left,right"),
    Scenario("forest_southeast", "forest", "4,1,1.8,0,0,-0.7854", "left,right"),
    Scenario("forest_southwest", "forest", "8,-2,1.8,0,0,2.35619", "left,right"),
    Scenario("forest_offset_left", "forest", "10,-1,1.8,0,0,1.5708", "left,right"),
    Scenario("forest_offset_right", "forest", "8,1,1.8,0,0,-1.5708", "left,right"),
]

SCENARIOS_FOREST_AVOIDANCE_LOWALT_V3: list[Scenario] = [
    Scenario("forest_northeast", "forest", "8,1,1.8,0,0,0.7854", "left,right"),
    Scenario("forest_southeast", "forest", "4,1,1.8,0,0,-0.7854", "left,right"),
    Scenario("forest_offset_right", "forest", "8,1,1.8,0,0,-1.5708", "left,right"),
    Scenario("forest_diagonal_a", "forest", "5,-1,1.8,0,0,1.0472", "left,right"),
]

SCENARIOS_FOREST_LATERAL_BALANCE_V4: list[Scenario] = [
    Scenario("forest_northeast_left", "forest", "8,1,1.8,0,0,0.7854", "left"),
    Scenario("forest_close_north_left", "forest", "4,0,1.8,0,0,1.5708", "left"),
    Scenario("forest_southeast_right", "forest", "4,1,1.8,0,0,-0.7854", "right"),
    Scenario("forest_offset_right_right", "forest", "8,1,1.8,0,0,-1.5708", "right"),
]

SCENARIOS_FOREST_LEFT_ESCAPE_FOCUS_V5: list[Scenario] = [
    Scenario("forest_northeast_leftfocus", "forest", "8,1,1.8,0,0,0.7854", "left"),
    Scenario("forest_southeast_leftfocus", "forest", "4,1,1.8,0,0,-0.7854", "left"),
    Scenario("forest_offset_right_leftfocus", "forest", "8,1,1.8,0,0,-1.5708", "left"),
]

SCENARIOS_FOREST_LEFT_ESCAPE_DENSE_V6: list[Scenario] = [
    Scenario("forest_east_leftdense", "forest", "6,0,1.8,0,0,0.0", "left"),
    Scenario("forest_west_leftdense", "forest", "6,0,1.8,0,0,3.14159", "left"),
    Scenario("forest_close_north_leftdense", "forest", "4,0,1.8,0,0,1.5708", "left"),
    Scenario("forest_diagonal_b_leftdense", "forest", "5,1,1.8,0,0,-1.0472", "left"),
    Scenario("forest_southeast_leftdense", "forest", "4,1,1.8,0,0,-0.7854", "left"),
]

SCENARIOS_FOREST_RIGHT_ESCAPE_DENSE_V7: list[Scenario] = [
    Scenario("forest_southeast_rightdense", "forest", "4,1,1.8,0,0,-0.7854", "right"),
    Scenario("forest_offset_left_rightdense", "forest", "10,-1,1.8,0,0,1.5708", "right"),
    Scenario("forest_close_east_rightdense", "forest", "4,0,1.8,0,0,0.0", "right"),
    Scenario("forest_diagonal_a_rightdense", "forest", "5,-1,1.8,0,0,1.0472", "right"),
]

SCENARIOS_FOREST_BALANCED_ESCAPE_EVAL_V1: list[Scenario] = [
    Scenario("forest_east_leftdense", "forest", "6,0,1.8,0,0,0.0", "left"),
    Scenario("forest_west_leftdense", "forest", "6,0,1.8,0,0,3.14159", "left"),
    Scenario("forest_southeast_rightdense", "forest", "4,1,1.8,0,0,-0.7854", "right"),
    Scenario("forest_offset_left_rightdense", "forest", "10,-1,1.8,0,0,1.5708", "right"),
]

SCENARIOS_FOREST_RIGHT_ESCAPE_CURATED_V8: list[Scenario] = [
    Scenario("forest_northeast_rightcurated", "forest", "8,1,1.8,0,0,0.7854", "right"),
    Scenario("forest_offset_right_rightcurated", "forest", "8,1,1.8,0,0,-1.5708", "right"),
]

SCENARIOS_FOREST_BALANCED_ESCAPE_EVAL_V2: list[Scenario] = [
    Scenario("forest_east_leftdense", "forest", "6,0,1.8,0,0,0.0", "left"),
    Scenario("forest_west_leftdense", "forest", "6,0,1.8,0,0,3.14159", "left"),
    Scenario("forest_northeast_rightcurated", "forest", "8,1,1.8,0,0,0.7854", "right"),
    Scenario("forest_offset_right_rightcurated", "forest", "8,1,1.8,0,0,-1.5708", "right"),
]

SCENARIOS_FOREST_BALANCED_COLLECTION_V9: list[Scenario] = [
    Scenario("forest_east_leftdense", "forest", "6,0,1.8,0,0,0.0", "left"),
    Scenario("forest_west_leftdense", "forest", "6,0,1.8,0,0,3.14159", "left"),
    Scenario("forest_northeast_leftfocus", "forest", "8,1,1.8,0,0,0.7854", "left"),
    Scenario("forest_southeast_leftfocus", "forest", "4,1,1.8,0,0,-0.7854", "left"),
    Scenario("forest_northeast_rightcurated", "forest", "8,1,1.8,0,0,0.7854", "right"),
    Scenario("forest_offset_right_rightcurated", "forest", "8,1,1.8,0,0,-1.5708", "right"),
]

SCENARIOS_FOREST_BALANCED_ESCAPE_EVAL_V3: list[Scenario] = [
    Scenario("forest_east_leftdense", "forest", "6,0,1.8,0,0,0.0", "left"),
    Scenario("forest_west_leftdense", "forest", "6,0,1.8,0,0,3.14159", "left"),
    Scenario("forest_northeast_rightcurated", "forest", "8,1,1.8,0,0,0.7854", "right"),
    Scenario("forest_southeast_rightblocked_v9", "forest", "4,1,1.8,0,0,-0.7854", "right"),
]

SCENARIOS_FOREST_RIGHT_PROBE_V10: list[Scenario] = [
    Scenario("forest_northeast_rightcurated", "forest", "8,1,1.8,0,0,0.7854", "right"),
    Scenario("forest_offset_left_rightprobe", "forest", "10,-1,1.8,0,0,1.5708", "right"),
    Scenario("forest_close_east_rightprobe", "forest", "4,0,1.8,0,0,0.0", "right"),
    Scenario("forest_diagonal_a_rightprobe", "forest", "5,-1,1.8,0,0,1.0472", "right"),
]

SCENARIOS_FOREST_BALANCED_ESCAPE_EVAL_V10: list[Scenario] = [
    Scenario("forest_east_leftdense", "forest", "6,0,1.8,0,0,0.0", "left"),
    Scenario("forest_west_leftdense", "forest", "6,0,1.8,0,0,3.14159", "left"),
    Scenario("forest_northeast_rightcurated", "forest", "8,1,1.8,0,0,0.7854", "right"),
    Scenario("forest_close_east_rightprobe", "forest", "4,0,1.8,0,0,0.0", "right"),
]

SCENARIOS_FOREST_RIGHT_HARDNEG_V11: list[Scenario] = [
    Scenario("forest_northeast_rightcurated", "forest", "8,1,1.8,0,0,0.7854", "right"),
    Scenario("forest_close_east_rightprobe", "forest", "4,0,1.8,0,0,0.0", "right"),
    Scenario("forest_offset_left_rightprobe", "forest", "10,-1,1.8,0,0,1.5708", "right"),
]

SCENARIOS_FOREST_RIGHT_FOCUS_V11B: list[Scenario] = [
    Scenario("forest_close_east_rightprobe", "forest", "4,0,1.8,0,0,0.0", "right"),
    Scenario("forest_offset_left_rightprobe", "forest", "10,-1,1.8,0,0,1.5708", "right"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-set", choices=["diverse_v1", "forest_avoidance_v1", "forest_avoidance_stable_v1", "forest_avoidance_asymmetric_v2", "forest_avoidance_lowalt_v3", "forest_lateral_balance_v4", "forest_left_escape_focus_v5", "forest_left_escape_dense_v6", "forest_right_escape_dense_v7", "forest_balanced_escape_eval_v1", "forest_right_escape_curated_v8", "forest_balanced_escape_eval_v2", "forest_balanced_collection_v9", "forest_balanced_escape_eval_v3", "forest_right_probe_v10", "forest_balanced_escape_eval_v10", "forest_right_hardneg_v11", "forest_right_focus_v11b"], default="diverse_v1")
    parser.add_argument("--eval-scenario-set", choices=["", "diverse_v1", "forest_avoidance_v1", "forest_avoidance_stable_v1", "forest_avoidance_asymmetric_v2", "forest_avoidance_lowalt_v3", "forest_lateral_balance_v4", "forest_left_escape_focus_v5", "forest_left_escape_dense_v6", "forest_right_escape_dense_v7", "forest_balanced_escape_eval_v1", "forest_right_escape_curated_v8", "forest_balanced_escape_eval_v2", "forest_balanced_collection_v9", "forest_balanced_escape_eval_v3", "forest_right_probe_v10", "forest_balanced_escape_eval_v10", "forest_right_hardneg_v11", "forest_right_focus_v11b"], default="")
    parser.add_argument("--episodes-per-scenario", type=int, default=2)
    parser.add_argument("--teacher-steps-per-episode", type=int, default=8)
    parser.add_argument("--teacher-goal-resample-attempts", type=int, default=10)
    parser.add_argument("--teacher-min-decision-score", type=float, default=0.55)
    parser.add_argument("--teacher-front-blocked-threshold-m", type=float, default=2.8)
    parser.add_argument("--teacher-side-clearance-threshold-m", type=float, default=1.5)
    parser.add_argument("--teacher-front-preferred-threshold-m", type=float, default=3.5)
    parser.add_argument("--teacher-target-side-preference-margin-m", type=float, default=0.4)
    parser.add_argument("--teacher-require-initially-blocked", action="store_true")
    parser.add_argument("--teacher-require-initial-target-action", action="store_true")
    parser.add_argument("--teacher-min-initial-branch-score", type=float, default=0.0)
    parser.add_argument("--teacher-fixed-target-family-per-episode", action="store_true")
    parser.add_argument("--collector-policy", choices=["reactive_teacher", "reactive_obstacle_teacher"], default="reactive_teacher")
    parser.add_argument("--action-forward-m-s", type=float, default=0.3)
    parser.add_argument("--action-strafe-m-s", type=float, default=0.3)
    parser.add_argument("--action-vertical-m-s", type=float, default=0.3)
    parser.add_argument("--action-yawspeed-deg-s", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-command-loss-weight", type=float, default=0.25)
    parser.add_argument("--train-no-class-weights", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-frame-stack", type=int, default=1)
    parser.add_argument("--mirror-lateral-actions", action="store_true")
    parser.add_argument("--balanced-action-sampler", action="store_true")
    parser.add_argument("--eval-scenarios", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--eval-script", default="scripts/eval_bc_policy_closed_loop.py")
    parser.add_argument("--eval-controller", choices=["bc", "reactive_obstacle_teacher"], default="bc")
    parser.add_argument("--eval-collision-imminent-threshold-m", type=float, default=1.5)
    parser.add_argument("--eval-front-blocked-threshold-m", type=float, default=2.5)
    parser.add_argument("--eval-escape-front-clearance-threshold-m", type=float, default=2.5)
    parser.add_argument("--eval-clearance-progress-threshold-m", type=float, default=0.25)
    parser.add_argument("--eval-forward-probe-duration-s", type=float, default=1.5)
    parser.add_argument("--eval-forward-probe-min-progress-m", type=float, default=0.2)
    parser.add_argument("--eval-forward-probe-safe-front-m", type=float, default=1.8)
    parser.add_argument("--eval-initial-blocked-check-window-s", type=float, default=2.0)
    parser.add_argument("--eval-initial-blocked-check-max-samples", type=int, default=5)
    parser.add_argument("--eval-require-initially-blocked", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--bridge-wait-s", type=float, default=4.0)
    parser.add_argument("--takeoff-altitude-m", type=float, default=2.5)
    parser.add_argument("--min-offboard-ready-altitude-m", type=float, default=0.5)
    parser.add_argument("--command-duration-s", type=float, default=0.75)
    parser.add_argument("--settle-duration-s", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--collection-max-attempts-per-episode", type=int, default=2)
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--episodes-root", default=str(DEFAULT_EPISODES_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--train-out-dir", default=str(DEFAULT_TRAIN_OUT_DIR))
    parser.add_argument("--eval-out-root", default=str(DEFAULT_EVAL_OUT_ROOT))
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--progress-out-dir", default=str(DEFAULT_PROGRESS_OUT))
    parser.add_argument("--behavior-out-dir", default=str(DEFAULT_BEHAVIOR_OUT))
    parser.add_argument("--eval-report-out-dir", default=str(DEFAULT_EVAL_REPORT_OUT))
    parser.add_argument("--dashboard-out-dir", default=str(DEFAULT_DASHBOARD_OUT))
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--filter-worlds", default="")
    parser.add_argument("--required-decision-family", default="")
    parser.add_argument("--allowed-decision-families", default="")
    parser.add_argument("--require-decision-rich", action="store_true")
    parser.add_argument("--allowed-actions", default="")
    parser.add_argument("--followthrough-after-families", default="")
    parser.add_argument("--followthrough-family", default="")
    parser.add_argument("--followthrough-actions", default="")
    parser.add_argument("--followthrough-steps", type=int, default=0)
    parser.add_argument("--zero-goal-features", action="store_true")
    parser.add_argument("--eval-goal-feature-mode", choices=["recorded", "zeros"], default="recorded")
    parser.add_argument("--i-understand-this-is-sitl", action="store_true", dest="i_understand_this_is_sitl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.i_understand_this_is_sitl:
        raise SystemExit("Refusing to run without --i-understand-this-is-sitl")

    session_id = datetime.now(timezone.utc).strftime("waypoint_multiscene_%Y%m%dT%H%M%SZ")
    run_dir = Path(args.log_root).expanduser().resolve() / session_id
    cycle_dir = run_dir / "cycle_000"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    tools_container_name = f"{TOOLS_SESSION_PREFIX}{session_id}"

    scenarios = _scenario_set(args.scenario_set)
    eval_scenario_pool = _scenario_set(args.eval_scenario_set or args.scenario_set)
    summary: dict[str, object] = {
        "run_id": session_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_set": args.scenario_set,
        "eval_scenario_set": args.eval_scenario_set or args.scenario_set,
        "episodes_per_scenario": args.episodes_per_scenario,
        "skip_collection": bool(args.skip_collection),
        "skip_export": bool(args.skip_export),
        "skip_train": bool(args.skip_train),
        "scenarios": [asdict(s) for s in scenarios],
        "eval_scenarios": [asdict(s) for s in eval_scenario_pool],
        "collection_runs": [],
        "eval_runs": [],
    }

    try:
        _stop_all_sim_containers()
        _stop_all_tools_containers()
        _launch_tools_container(container_name=tools_container_name)
        if not args.skip_collection:
            for scenario_index, scenario in enumerate(scenarios):
                container_name = f"sensenova_drone_agent_multiscene_{session_id}_{scenario.label}"
                result = _run_collection_scenario(
                    args=args,
                    cycle_dir=cycle_dir,
                    session_id=session_id,
                    scenario_index=scenario_index,
                    scenario=scenario,
                    container_name=container_name,
                    tools_container_name=tools_container_name,
                )
                summary["collection_runs"].append(result)

        if not args.skip_export:
            _run_logged(
                _append_optional_args(
                    [
                        "python3",
                        "scripts/export_bc_dataset.py",
                        "--episodes-root",
                        args.episodes_root,
                        "--out-jsonl",
                        args.manifest,
                        "--summary-json",
                        args.summary_json,
                        "--val-ratio",
                        "0.15",
                    ],
                    {
                        "--include-worlds": args.filter_worlds,
                        "--required-decision-family": args.required_decision_family,
                        "--allowed-decision-families": args.allowed_decision_families,
                        "--allowed-actions": args.allowed_actions,
                        "--followthrough-after-families": args.followthrough_after_families,
                        "--followthrough-family": args.followthrough_family,
                        "--followthrough-actions": args.followthrough_actions,
                        "--followthrough-steps": str(args.followthrough_steps) if int(args.followthrough_steps) > 0 else "",
                    },
                ),
                stdout_path=cycle_dir / "export.stdout.log",
                stderr_path=cycle_dir / "export.stderr.log",
                extra_args=(["--require-decision-rich"] if args.require_decision_rich else None),
            )
        if not args.skip_train:
            _run_logged(
                [
                    "python3",
                    "scripts/train_bc_policy.py",
                    "--manifest",
                    args.manifest,
                    "--out-dir",
                    args.train_out_dir,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--image-size",
                    str(args.image_size),
                    "--learning-rate",
                    str(args.train_learning_rate),
                    "--command-loss-weight",
                    str(args.train_command_loss_weight),
                    "--device",
                    args.device,
                ],
                stdout_path=cycle_dir / "train.stdout.log",
                stderr_path=cycle_dir / "train.stderr.log",
                extra_args=(
                    (["--zero-goal-features"] if args.zero_goal_features else [])
                    + (["--frame-stack", str(args.train_frame_stack)] if int(args.train_frame_stack) > 1 else [])
                    + (["--mirror-lateral-actions"] if args.mirror_lateral_actions else [])
                    + (["--balanced-action-sampler"] if args.balanced_action_sampler else [])
                    + (["--no-class-weights"] if args.train_no_class_weights else [])
                ) or None,
            )

        # Refresh the long-lived tools session before eval. Collection can leave the
        # container in a bad state or it may have exited unexpectedly during the long
        # host-side train/export phase; eval should not inherit that stale session.
        _stop_container(tools_container_name)
        _launch_tools_container(container_name=tools_container_name)

        eval_scenarios = eval_scenario_pool[: max(1, min(args.eval_scenarios, len(eval_scenario_pool)))]
        for eval_index, scenario in enumerate(eval_scenarios):
            container_name = f"sensenova_drone_agent_multiscene_eval_{session_id}_{scenario.label}"
            result = _run_eval_scenario(
                args=args,
                cycle_dir=cycle_dir,
                session_id=session_id,
                eval_index=eval_index,
                scenario=scenario,
                container_name=container_name,
                tools_container_name=tools_container_name,
            )
            summary["eval_runs"].append(result)

        _run_logged(
            [
                "python3",
                "scripts/report_episode_behavior.py",
                "--episodes-root",
                args.episodes_root,
                "--out-dir",
                str(Path(args.behavior_out_dir).expanduser().resolve()),
            ],
            stdout_path=cycle_dir / "report_episode_behavior.stdout.log",
            stderr_path=cycle_dir / "report_episode_behavior.stderr.log",
        )
        _run_logged(
            [
                "python3",
                "scripts/report_bc_progress.py",
                "--log-root",
                args.log_root,
                "--episodes-root",
                args.episodes_root,
                "--manifest-summary",
                str(Path(args.summary_json).expanduser().resolve()),
                "--baseline-summary",
                str((Path(args.train_out_dir).expanduser().resolve() / "train_summary.json")),
                "--out-dir",
                str(Path(args.progress_out_dir).expanduser().resolve()),
            ],
            stdout_path=cycle_dir / "report_bc_progress.stdout.log",
            stderr_path=cycle_dir / "report_bc_progress.stderr.log",
        )
        _run_logged(
            [
                "python3",
                "scripts/report_closed_loop_eval.py",
                "--eval-root",
                str(Path(args.eval_out_root).expanduser().resolve()),
                "--out-dir",
                str(Path(args.eval_report_out_dir).expanduser().resolve()),
                "--run-id-prefix",
                session_id,
            ],
            stdout_path=cycle_dir / "report_closed_loop_eval.stdout.log",
            stderr_path=cycle_dir / "report_closed_loop_eval.stderr.log",
        )
        _run_logged(
            [
                "python3",
                "scripts/report_training_dashboard.py",
                "--manifest-summary",
                str(Path(args.summary_json).expanduser().resolve()),
                "--baseline-summary",
                str((Path(args.train_out_dir).expanduser().resolve() / "train_summary.json")),
                "--progress-report",
                str((Path(args.progress_out_dir).expanduser().resolve() / "index.html")),
                "--behavior-report",
                str((Path(args.behavior_out_dir).expanduser().resolve() / "index.html")),
                "--eval-report",
                str((Path(args.eval_report_out_dir).expanduser().resolve() / "index.html")),
                "--eval-summary",
                str((Path(args.eval_report_out_dir).expanduser().resolve() / "dashboard_summary.json")),
                "--out-dir",
                str(Path(args.dashboard_out_dir).expanduser().resolve()),
            ],
            stdout_path=cycle_dir / "report_training_dashboard.stdout.log",
            stderr_path=cycle_dir / "report_training_dashboard.stderr.log",
        )
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(run_dir / "summary.json", summary)
    finally:
        _stop_container(tools_container_name)
        _stop_all_sim_containers()
        _stop_all_tools_containers()

    payload = {
        "run_dir": str(run_dir),
        "summary": str((run_dir / "summary.json").resolve()),
        "manifest_summary": str(Path(args.summary_json).resolve()),
        "train_summary": str((Path(args.train_out_dir) / "train_summary.json").resolve()),
        "dashboard_report": str((Path(args.dashboard_out_dir) / "index.html").resolve()),
        "progress_report": str((Path(args.progress_out_dir) / "index.html").resolve()),
        "behavior_report": str((Path(args.behavior_out_dir) / "index.html").resolve()),
        "eval_report": str((Path(args.eval_report_out_dir) / "index.html").resolve()),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _run_collection_scenario(
    *,
    args: argparse.Namespace,
    cycle_dir: Path,
    session_id: str,
    scenario_index: int,
    scenario: Scenario,
    container_name: str,
    tools_container_name: str,
) -> dict[str, object]:
    _launch_sim(scenario=scenario, container_name=container_name)
    try:
        try:
            _wait_for_connection(tools_container_name=tools_container_name, timeout=args.timeout)
        except Exception as exc:
            return {
                "scenario": asdict(scenario),
                "episodes": [],
                "status": "failed",
                "error": f"connection_failed: {exc}",
            }
        topic = DEFAULT_GAZEBO_TOPIC_TEMPLATE.format(world=scenario.world)
        records: list[dict[str, object]] = []
        for episode_index in range(args.episodes_per_scenario):
            episode_id = f"{session_id}_s{scenario_index:03d}_{scenario.label}_e{episode_index:03d}"
            episode_dir = Path(args.episodes_root).expanduser().resolve() / episode_id
            episode_json_path = episode_dir / "episode.json"
            stdout_path = cycle_dir / f"{episode_id}.stdout.log"
            stderr_path = cycle_dir / f"{episode_id}.stderr.log"
            fixed_target_family = (
                _episode_target_family(scenario.target_families, episode_index)
                if args.teacher_fixed_target_family_per_episode
                else ""
            )
            tool_command = _append_optional_args(
                [
                    "python3",
                    "scripts/collect_sitl_bc_episode.py",
                    "--gazebo-topic",
                    topic,
                    "--depth-topic",
                    args.depth_topic,
                    "--episodes-root",
                    _container_path(args.episodes_root),
                    "--episode-id",
                    episode_id,
                    "--world-label",
                    scenario.world,
                    "--scenario-label",
                    scenario.label,
                    "--policy",
                    args.collector_policy,
                    "--num-steps",
                    str(args.teacher_steps_per_episode),
                    "--teacher-seed",
                    str(args.seed + scenario_index * 10_000 + episode_index),
                    "--teacher-goal-mode",
                    "decision_game",
                    "--teacher-target-families",
                    scenario.target_families,
                    "--teacher-goal-resample-attempts",
                    str(args.teacher_goal_resample_attempts),
                    "--teacher-min-decision-score",
                    str(args.teacher_min_decision_score),
                    "--teacher-front-blocked-threshold-m",
                    str(args.teacher_front_blocked_threshold_m),
                    "--teacher-side-clearance-threshold-m",
                    str(args.teacher_side_clearance_threshold_m),
                    "--teacher-front-preferred-threshold-m",
                    str(args.teacher_front_preferred_threshold_m),
                    "--teacher-target-side-preference-margin-m",
                    str(args.teacher_target_side_preference_margin_m),
                    "--teacher-min-initial-branch-score",
                    str(args.teacher_min_initial_branch_score),
                    "--teacher-max-goal-vertical-m",
                    str(scenario.max_goal_vertical_m),
                    "--teacher-refresh-goal-every-step",
                    "--bridge-wait-s",
                    str(args.bridge_wait_s),
                    "--takeoff-altitude-m",
                    str(args.takeoff_altitude_m),
                    "--min-offboard-ready-altitude-m",
                    str(args.min_offboard_ready_altitude_m),
                    "--command-duration-s",
                    str(args.command_duration_s),
                    "--action-forward-m-s",
                    str(args.action_forward_m_s),
                    "--action-strafe-m-s",
                    str(args.action_strafe_m_s),
                    "--action-vertical-m-s",
                    str(args.action_vertical_m_s),
                    "--action-yawspeed-deg-s",
                    str(args.action_yawspeed_deg_s),
                    "--settle-duration-s",
                    str(args.settle_duration_s),
                    "--timeout",
                    str(args.timeout),
                    "--i-understand-this-is-sitl",
                ],
                {
                    "--teacher-fixed-target-family": fixed_target_family,
                    "--teacher-require-initially-blocked": args.teacher_require_initially_blocked,
                    "--teacher-require-initial-target-action": args.teacher_require_initial_target_action,
                },
            )
            command = _tools_exec_command(tools_container_name, tool_command)
            try:
                for attempt in range(max(1, int(args.collection_max_attempts_per_episode))):
                    try:
                        _run_logged(
                            command,
                            stdout_path=stdout_path,
                            stderr_path=stderr_path,
                            timeout=float(args.timeout) + 30.0,
                        )
                        break
                    except subprocess.TimeoutExpired:
                        if episode_json_path.exists():
                            try:
                                episode_payload = json.loads(episode_json_path.read_text(encoding="utf-8"))
                            except Exception:
                                episode_payload = {}
                            status = str(episode_payload.get("status", "")).lower()
                            if status == "completed":
                                break
                            if status == "failed":
                                raise subprocess.CalledProcessError(returncode=124, cmd=command)
                        if attempt >= max(1, int(args.collection_max_attempts_per_episode)) - 1:
                            raise
                        _stop_container(container_name)
                        _launch_sim(scenario=scenario, container_name=container_name)
                        _wait_for_connection(tools_container_name=tools_container_name, timeout=args.timeout)
                        time.sleep(2.0)
                    except subprocess.CalledProcessError:
                        if _is_non_retryable_collection_failure(stderr_path):
                            raise
                        if attempt >= max(1, int(args.collection_max_attempts_per_episode)) - 1:
                            raise
                        _stop_container(container_name)
                        _launch_sim(scenario=scenario, container_name=container_name)
                        _wait_for_connection(tools_container_name=tools_container_name, timeout=args.timeout)
                        time.sleep(2.0)
            except subprocess.CalledProcessError as exc:
                records.append(
                    {
                        "episode_id": episode_id,
                        "status": "failed",
                        "returncode": exc.returncode,
                        "stdout_log": str(stdout_path.resolve()),
                        "stderr_log": str(stderr_path.resolve()),
                    }
                )
                continue
            records.append({"episode_id": episode_id, "status": "completed"})
        scenario_status = "completed"
        if records and any(record.get("status") == "failed" for record in records):
            scenario_status = "completed_with_failures"
        return {"scenario": asdict(scenario), "episodes": records, "status": scenario_status}
    finally:
        _stop_container(container_name)


def _run_eval_scenario(
    *,
    args: argparse.Namespace,
    cycle_dir: Path,
    session_id: str,
    eval_index: int,
    scenario: Scenario,
    container_name: str,
    tools_container_name: str,
) -> dict[str, object]:
    _launch_sim(scenario=scenario, container_name=container_name)
    try:
        try:
            _wait_for_connection(tools_container_name=tools_container_name, timeout=args.timeout)
        except Exception as exc:
            return {
                "scenario": asdict(scenario),
                "run_id": f"{session_id}_{scenario.label}",
                "status": "failed",
                "error": f"connection_failed: {exc}",
            }
        topic = DEFAULT_GAZEBO_TOPIC_TEMPLATE.format(world=scenario.world)
        run_id = f"{session_id}_{scenario.label}"
        eval_summary_path = Path(args.eval_out_root).expanduser().resolve() / run_id / "summary.json"
        stdout_path = cycle_dir / f"eval_{eval_index:03d}_{scenario.label}.stdout.log"
        stderr_path = cycle_dir / f"eval_{eval_index:03d}_{scenario.label}.stderr.log"
        try:
            for attempt in range(2):
                try:
                    _run_logged(
                        _eval_command(
                            args=args,
                            topic=topic,
                            scenario=scenario,
                            run_id=run_id,
                            tools_container_name=tools_container_name,
                        ),
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        timeout=float(args.timeout) + 30.0,
                    )
                    break
                except subprocess.TimeoutExpired:
                    if eval_summary_path.exists():
                        return {
                            "scenario": asdict(scenario),
                            "run_id": run_id,
                            "status": "completed_timeout",
                            "summary_path": str(eval_summary_path.resolve()),
                            "stdout_log": str(stdout_path.resolve()),
                            "stderr_log": str(stderr_path.resolve()),
                        }
                    if attempt >= 1:
                        raise
                    _stop_container(container_name)
                    _launch_sim(scenario=scenario, container_name=container_name)
                    _wait_for_connection(tools_container_name=tools_container_name, timeout=args.timeout)
                    time.sleep(2.0)
                except subprocess.CalledProcessError:
                    if _is_non_retryable_eval_failure(stderr_path):
                        raise
                    if attempt >= 1:
                        raise
                    _stop_container(container_name)
                    _launch_sim(scenario=scenario, container_name=container_name)
                    _wait_for_connection(tools_container_name=tools_container_name, timeout=args.timeout)
                    time.sleep(2.0)
            return {"scenario": asdict(scenario), "run_id": run_id, "status": "completed"}
        except subprocess.CalledProcessError as exc:
            return {
                "scenario": asdict(scenario),
                "run_id": run_id,
                "status": "failed",
                "error": f"eval_failed: rc={exc.returncode}",
                "stdout_log": str(stdout_path.resolve()),
                "stderr_log": str(stderr_path.resolve()),
            }
        except subprocess.TimeoutExpired:
            return {
                "scenario": asdict(scenario),
                "run_id": run_id,
                "status": "failed",
                "error": "eval_timeout_without_summary",
                "stdout_log": str(stdout_path.resolve()),
                "stderr_log": str(stderr_path.resolve()),
            }
    finally:
        _stop_container(container_name)


def _launch_sim(*, scenario: Scenario, container_name: str) -> None:
    _stop_all_sim_containers()
    env = os.environ.copy()
    env["HOST_UID"] = str(os.getuid())
    env["HOST_GID"] = str(os.getgid())
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "run",
            "-d",
            "--name",
            container_name,
            "-e",
            "PX4_SIM_MODEL=gz_x500_depth",
            "-e",
            f"PX4_GZ_WORLD={scenario.world}",
            "-e",
            f"PX4_GZ_MODEL_POSE={scenario.pose}",
            "-e",
            "HEADLESS=1",
            "sim",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3.0)


def _wait_for_connection(*, tools_container_name: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _container_is_running(tools_container_name):
            _launch_tools_container(container_name=tools_container_name)
        try:
            result = subprocess.run(
                _tools_exec_command(
                    tools_container_name,
                    [
                        "python3",
                        "scripts/verify_mavsdk_connection.py",
                        "--timeout",
                        "15",
                    ],
                ),
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20.0,
            )
        except subprocess.TimeoutExpired:
            _stop_container(tools_container_name)
            time.sleep(2.0)
            continue
        if result.returncode == 0:
            return
        time.sleep(5.0)
    raise RuntimeError("Timed out waiting for PX4 SITL to come online for the requested scenario.")


def _stop_all_sim_containers() -> None:
    result = subprocess.run(
        ["docker", "ps", "--filter", f"ancestor={SIM_IMAGE}", "--format", "{{.Names}}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for name in names:
        _stop_container(name)

def _stop_all_tools_containers() -> None:
    for name in _running_tools_containers():
        _stop_container(name)


def _running_tools_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith(TOOLS_RUN_PREFIX)
        or line.strip().startswith(TOOLS_SESSION_PREFIX)
    ]


def _stop_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _launch_tools_container(*, container_name: str) -> None:
    _stop_container(container_name)
    env = os.environ.copy()
    env["HOST_UID"] = str(os.getuid())
    env["HOST_GID"] = str(os.getgid())
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "run",
            "-d",
            "--name",
            container_name,
            "tools",
            "sleep",
            "infinity",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)


def _container_is_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    return result.stdout.strip().lower() == "true"


def _tools_exec_command(container_name: str, command: list[str]) -> list[str]:
    shell_command = " ".join(shlex.quote(token) for token in command)
    return [
        "docker",
        "exec",
        "-w",
        "/workspace",
        container_name,
        "bash",
        "-lc",
        f"source /opt/ros/${{ROS_DISTRO}}/setup.bash && {shell_command}",
    ]
def _run_logged(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    extra_args: list[str] | None = None,
    timeout: float | None = None,
) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    full_command = list(command)
    if extra_args:
        full_command.extend(extra_args)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        subprocess.run(full_command, cwd=PROJECT_ROOT, check=True, stdout=stdout_handle, stderr=stderr_handle, timeout=timeout)


def _is_non_retryable_collection_failure(stderr_path: Path) -> bool:
    if not stderr_path.exists():
        return False
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="ignore")
    non_retryable_markers = (
        "Reactive obstacle teacher episode is not initially blocked enough",
        "Reactive obstacle teacher episode initial action does not match target family",
        "Reactive obstacle teacher episode initial branch score is too low",
        "Reactive teacher episode is not initially blocked enough",
        "Reactive teacher episode initial action does not match target family",
        "Reactive teacher episode initial branch score is too low",
    )
    return any(marker in stderr_text for marker in non_retryable_markers)


def _is_non_retryable_eval_failure(stderr_path: Path) -> bool:
    if not stderr_path.exists():
        return False
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="ignore")
    non_retryable_markers = (
        "Scenario is not initially blocked enough for tree eval",
        "Policy closed-loop eval is not initially blocked enough",
    )
    return any(marker in stderr_text for marker in non_retryable_markers)


def _append_optional_args(command: list[str], optional_args: dict[str, object]) -> list[str]:
    full_command = list(command)
    for flag, value in optional_args.items():
        if isinstance(value, bool):
            if value:
                full_command.append(flag)
            continue
        if str(value).strip():
            full_command.extend([flag, str(value)])
    return full_command


def _container_path(path: str) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        raise RuntimeError(f"Path {resolved} must live under the project root {PROJECT_ROOT}.")
    return str(Path("/workspace") / relative)


def _scenario_set(name: str) -> list[Scenario]:
    if name == "diverse_v1":
        return list(SCENARIOS_DIVERSE_V1)
    if name == "forest_avoidance_v1":
        return list(SCENARIOS_FOREST_AVOIDANCE_V1)
    if name == "forest_avoidance_stable_v1":
        return list(SCENARIOS_FOREST_AVOIDANCE_STABLE_V1)
    if name == "forest_avoidance_asymmetric_v2":
        return list(SCENARIOS_FOREST_AVOIDANCE_ASYMMETRIC_V2)
    if name == "forest_avoidance_lowalt_v3":
        return list(SCENARIOS_FOREST_AVOIDANCE_LOWALT_V3)
    if name == "forest_lateral_balance_v4":
        return list(SCENARIOS_FOREST_LATERAL_BALANCE_V4)
    if name == "forest_left_escape_focus_v5":
        return list(SCENARIOS_FOREST_LEFT_ESCAPE_FOCUS_V5)
    if name == "forest_left_escape_dense_v6":
        return list(SCENARIOS_FOREST_LEFT_ESCAPE_DENSE_V6)
    if name == "forest_right_escape_dense_v7":
        return list(SCENARIOS_FOREST_RIGHT_ESCAPE_DENSE_V7)
    if name == "forest_balanced_escape_eval_v1":
        return list(SCENARIOS_FOREST_BALANCED_ESCAPE_EVAL_V1)
    if name == "forest_right_escape_curated_v8":
        return list(SCENARIOS_FOREST_RIGHT_ESCAPE_CURATED_V8)
    if name == "forest_balanced_escape_eval_v2":
        return list(SCENARIOS_FOREST_BALANCED_ESCAPE_EVAL_V2)
    if name == "forest_balanced_collection_v9":
        return list(SCENARIOS_FOREST_BALANCED_COLLECTION_V9)
    if name == "forest_balanced_escape_eval_v3":
        return list(SCENARIOS_FOREST_BALANCED_ESCAPE_EVAL_V3)
    if name == "forest_right_probe_v10":
        return list(SCENARIOS_FOREST_RIGHT_PROBE_V10)
    if name == "forest_balanced_escape_eval_v10":
        return list(SCENARIOS_FOREST_BALANCED_ESCAPE_EVAL_V10)
    if name == "forest_right_hardneg_v11":
        return list(SCENARIOS_FOREST_RIGHT_HARDNEG_V11)
    if name == "forest_right_focus_v11b":
        return list(SCENARIOS_FOREST_RIGHT_FOCUS_V11B)
    raise RuntimeError(f"Unsupported scenario set: {name}")


def _episode_target_family(raw_target_families: str, episode_index: int) -> str:
    families = [token.strip().lower() for token in raw_target_families.split(",") if token.strip()]
    if not families:
        return "forward"
    return families[episode_index % len(families)]


def _eval_command(
    *,
    args: argparse.Namespace,
    topic: str,
    scenario: Scenario,
    run_id: str,
    tools_container_name: str,
) -> list[str]:
    eval_script = args.eval_script.strip() or "scripts/eval_bc_policy_closed_loop.py"
    checkpoint_path = _container_path(str(_select_eval_checkpoint(Path(args.train_out_dir))))
    if eval_script.endswith("eval_tree_avoidance_policy.py"):
        return _tools_exec_command(
            tools_container_name,
            [
            "python3",
            eval_script,
            "--controller",
            args.eval_controller,
            "--checkpoint",
            checkpoint_path,
            "--gazebo-topic",
            topic,
            "--depth-topic",
            args.depth_topic,
            "--world-label",
            scenario.label,
            "--run-id",
            run_id,
            "--episodes",
            str(args.eval_episodes),
            "--num-steps",
            str(args.eval_steps),
            "--takeoff-altitude-m",
            str(args.takeoff_altitude_m),
            "--min-offboard-ready-altitude-m",
            str(args.min_offboard_ready_altitude_m),
            "--device",
            args.device,
            "--command-duration-s",
            str(args.command_duration_s),
            "--action-forward-m-s",
            str(args.action_forward_m_s),
            "--action-strafe-m-s",
            str(args.action_strafe_m_s),
            "--action-vertical-m-s",
            str(args.action_vertical_m_s),
            "--action-yawspeed-deg-s",
            str(args.action_yawspeed_deg_s),
            "--collision-imminent-threshold-m",
            str(args.eval_collision_imminent_threshold_m),
            "--front-blocked-threshold-m",
            str(args.eval_front_blocked_threshold_m),
            "--escape-front-clearance-threshold-m",
            str(args.eval_escape_front_clearance_threshold_m),
            "--clearance-progress-threshold-m",
            str(args.eval_clearance_progress_threshold_m),
            "--initial-blocked-check-window-s",
            str(args.eval_initial_blocked_check_window_s),
            "--initial-blocked-check-max-samples",
            str(args.eval_initial_blocked_check_max_samples),
            "--forward-probe-duration-s",
            str(args.eval_forward_probe_duration_s),
            "--forward-probe-min-progress-m",
            str(args.eval_forward_probe_min_progress_m),
            "--forward-probe-safe-front-m",
            str(args.eval_forward_probe_safe_front_m),
            "--teacher-front-blocked-threshold-m",
            str(args.teacher_front_blocked_threshold_m),
            "--teacher-side-clearance-threshold-m",
            str(args.teacher_side_clearance_threshold_m),
            "--teacher-front-preferred-threshold-m",
            str(args.teacher_front_preferred_threshold_m),
            "--out-root",
            _container_path(args.eval_out_root),
            *(
                ["--require-initially-blocked"]
                if args.eval_require_initially_blocked
                else []
            ),
            "--i-understand-this-is-sitl",
            ],
        )
    return _tools_exec_command(
        tools_container_name,
        [
            "python3",
            eval_script,
            "--checkpoint",
            checkpoint_path,
            "--gazebo-topic",
            topic,
            "--depth-topic",
            args.depth_topic,
            "--world-label",
            scenario.label,
            "--run-id",
            run_id,
            "--episodes",
            str(args.eval_episodes),
            "--num-steps",
            str(args.eval_steps),
            "--device",
            args.device,
            "--goal-feature-mode",
            args.eval_goal_feature_mode,
            "--command-duration-s",
            str(args.command_duration_s),
            "--action-forward-m-s",
            str(args.action_forward_m_s),
            "--action-strafe-m-s",
            str(args.action_strafe_m_s),
            "--action-vertical-m-s",
            str(args.action_vertical_m_s),
            "--action-yawspeed-deg-s",
            str(args.action_yawspeed_deg_s),
            "--out-root",
            _container_path(args.eval_out_root),
            "--i-understand-this-is-sitl",
        ],
    )


def _select_eval_checkpoint(train_out_dir: Path) -> Path:
    best_checkpoint = train_out_dir / "best.pt"
    if best_checkpoint.exists():
        return best_checkpoint
    last_checkpoint = train_out_dir / "last.pt"
    if last_checkpoint.exists():
        return last_checkpoint
    raise RuntimeError(f"No evaluation checkpoint found in {train_out_dir}")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
