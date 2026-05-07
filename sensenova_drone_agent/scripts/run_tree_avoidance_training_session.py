#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_multiscene_waypoint_training_session.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-profile", choices=["broad", "escape_split"], default="broad")
    parser.add_argument("--scenario-set", default="forest_avoidance_lowalt_v3")
    parser.add_argument("--eval-scenario-set", default="")
    parser.add_argument("--episodes-per-scenario", type=int, default=3)
    parser.add_argument("--teacher-steps-per-episode", type=int, default=8)
    parser.add_argument("--teacher-goal-resample-attempts", type=int, default=12)
    parser.add_argument("--teacher-min-decision-score", type=float, default=0.5)
    parser.add_argument("--teacher-front-blocked-threshold-m", type=float, default=2.0)
    parser.add_argument("--teacher-side-clearance-threshold-m", type=float, default=1.45)
    parser.add_argument("--teacher-front-preferred-threshold-m", type=float, default=2.5)
    parser.add_argument("--teacher-target-side-preference-margin-m", type=float, default=0.4)
    parser.add_argument("--teacher-require-initially-blocked", action="store_true")
    parser.add_argument("--teacher-require-initial-target-action", action="store_true")
    parser.add_argument("--teacher-min-initial-branch-score", type=float, default=0.0)
    parser.add_argument("--teacher-fixed-target-family-per-episode", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-command-loss-weight", type=float, default=0.25)
    parser.add_argument("--train-no-class-weights", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-frame-stack", type=int, default=4)
    parser.add_argument("--mirror-lateral-actions", action="store_true")
    parser.add_argument("--balanced-action-sampler", action="store_true")
    parser.add_argument("--eval-scenarios", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bridge-wait-s", type=float, default=4.0)
    parser.add_argument("--takeoff-altitude-m", type=float, default=0.8)
    parser.add_argument("--min-offboard-ready-altitude-m", type=float, default=0.4)
    parser.add_argument("--command-duration-s", type=float, default=1.0)
    parser.add_argument("--action-forward-m-s", type=float, default=0.4)
    parser.add_argument("--action-strafe-m-s", type=float, default=0.5)
    parser.add_argument("--action-vertical-m-s", type=float, default=0.35)
    parser.add_argument("--action-yawspeed-deg-s", type=float, default=12.0)
    parser.add_argument("--settle-duration-s", type=float, default=0.5)
    parser.add_argument("--eval-front-blocked-threshold-m", type=float, default=2.1)
    parser.add_argument("--eval-escape-front-clearance-threshold-m", type=float, default=2.0)
    parser.add_argument("--eval-clearance-progress-threshold-m", type=float, default=0.15)
    parser.add_argument("--eval-initial-blocked-check-window-s", type=float, default=2.0)
    parser.add_argument("--eval-initial-blocked-check-max-samples", type=int, default=5)
    parser.add_argument("--eval-forward-probe-duration-s", type=float, default=1.5)
    parser.add_argument("--eval-forward-probe-min-progress-m", type=float, default=0.1)
    parser.add_argument("--eval-forward-probe-safe-front-m", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--collection-max-attempts-per-episode", type=int, default=2)
    parser.add_argument("--episodes-root", default="data/tree_avoidance/episodes")
    parser.add_argument("--manifest", default="data/tree_avoidance/manifests/bc_manifest.jsonl")
    parser.add_argument("--summary-json", default="data/tree_avoidance/manifests/bc_manifest_summary.json")
    parser.add_argument("--train-out-dir", default="output/bc_policy_tree_avoidance")
    parser.add_argument("--eval-out-root", default="output/closed_loop_eval_tree_avoidance")
    parser.add_argument("--log-root", default="logs/tree_avoidance_sessions")
    parser.add_argument("--progress-out-dir", default="output/tree_avoidance_progress_report")
    parser.add_argument("--behavior-out-dir", default="output/tree_avoidance_behavior_report")
    parser.add_argument("--eval-report-out-dir", default="output/tree_avoidance_eval_report")
    parser.add_argument("--dashboard-out-dir", default="output/tree_avoidance_dashboard")
    parser.add_argument("--depth-topic", default="/depth_camera")
    parser.add_argument("--required-decision-family", default="")
    parser.add_argument("--allowed-decision-families", default="obstacle_avoidance,obstacle_cruise")
    parser.add_argument("--allowed-actions", default="forward,yaw_left,yaw_right,strafe_left,strafe_right")
    parser.add_argument("--followthrough-after-families", default="")
    parser.add_argument("--followthrough-family", default="")
    parser.add_argument("--followthrough-actions", default="")
    parser.add_argument("--followthrough-steps", type=int, default=0)
    parser.add_argument("--i-understand-this-is-sitl", action="store_true", dest="i_understand_this_is_sitl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.i_understand_this_is_sitl:
        raise SystemExit("Refusing to run without --i-understand-this-is-sitl")

    required_decision_family = args.required_decision_family
    allowed_decision_families = args.allowed_decision_families
    allowed_actions = args.allowed_actions
    followthrough_after_families = args.followthrough_after_families
    followthrough_family = args.followthrough_family
    followthrough_actions = args.followthrough_actions
    followthrough_steps = args.followthrough_steps

    if args.dataset_profile == "escape_split":
        required_decision_family = "obstacle_avoidance"
        allowed_decision_families = ""
        allowed_actions = "forward,yaw_left,yaw_right,strafe_left,strafe_right"
        followthrough_after_families = "obstacle_avoidance"
        followthrough_family = "obstacle_cruise"
        followthrough_actions = "forward"
        followthrough_steps = 2

    command = [
        sys.executable,
        str(RUNNER),
        "--scenario-set",
        args.scenario_set,
        *(["--eval-scenario-set", args.eval_scenario_set] if args.eval_scenario_set else []),
        "--episodes-per-scenario",
        str(args.episodes_per_scenario),
        "--teacher-steps-per-episode",
        str(args.teacher_steps_per_episode),
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
        *(["--teacher-require-initially-blocked"] if args.teacher_require_initially_blocked else []),
        *(["--teacher-require-initial-target-action"] if args.teacher_require_initial_target_action else []),
        "--teacher-min-initial-branch-score",
        str(args.teacher_min_initial_branch_score),
        *(["--teacher-fixed-target-family-per-episode"] if args.teacher_fixed_target_family_per_episode else []),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--train-learning-rate",
        str(args.train_learning_rate),
        "--train-command-loss-weight",
        str(args.train_command_loss_weight),
        "--device",
        args.device,
        "--train-frame-stack",
        str(args.train_frame_stack),
        *(["--mirror-lateral-actions"] if args.mirror_lateral_actions else []),
        *(["--balanced-action-sampler"] if args.balanced_action_sampler else []),
        *(["--train-no-class-weights"] if args.train_no_class_weights else []),
        "--eval-scenarios",
        str(args.eval_scenarios),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-steps",
        str(args.eval_steps),
        "--seed",
        str(args.seed),
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
        "--eval-front-blocked-threshold-m",
        str(args.eval_front_blocked_threshold_m),
        "--eval-escape-front-clearance-threshold-m",
        str(args.eval_escape_front_clearance_threshold_m),
        "--eval-clearance-progress-threshold-m",
        str(args.eval_clearance_progress_threshold_m),
        "--eval-initial-blocked-check-window-s",
        str(args.eval_initial_blocked_check_window_s),
        "--eval-initial-blocked-check-max-samples",
        str(args.eval_initial_blocked_check_max_samples),
        "--eval-forward-probe-duration-s",
        str(args.eval_forward_probe_duration_s),
        "--eval-forward-probe-min-progress-m",
        str(args.eval_forward_probe_min_progress_m),
        "--eval-forward-probe-safe-front-m",
        str(args.eval_forward_probe_safe_front_m),
        "--timeout",
        str(args.timeout),
        "--collection-max-attempts-per-episode",
        str(args.collection_max_attempts_per_episode),
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
        "--progress-out-dir",
        args.progress_out_dir,
        "--behavior-out-dir",
        args.behavior_out_dir,
        "--eval-report-out-dir",
        args.eval_report_out_dir,
        "--dashboard-out-dir",
        args.dashboard_out_dir,
        "--depth-topic",
        args.depth_topic,
        "--collector-policy",
        "reactive_obstacle_teacher",
        "--filter-worlds",
        "forest",
        "--required-decision-family",
        required_decision_family,
        "--allowed-decision-families",
        allowed_decision_families,
        "--allowed-actions",
        allowed_actions,
        "--followthrough-after-families",
        followthrough_after_families,
        "--followthrough-family",
        followthrough_family,
        "--followthrough-actions",
        followthrough_actions,
        "--followthrough-steps",
        str(followthrough_steps),
        "--zero-goal-features",
        "--eval-script",
        "scripts/eval_tree_avoidance_policy.py",
        "--eval-controller",
        "bc",
        "--eval-require-initially-blocked",
        "--i-understand-this-is-sitl",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
