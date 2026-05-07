#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import collect_sitl_bc_episode as sitl_runtime

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

from sensenova_drone.actions import build_action_cfg, discrete_to_command
from sensenova_drone.bc_data import BCEpisodeStep, save_episode_step
from sensenova_drone.bc_infer import load_bc_policy_runner
from sensenova_drone.eval.closed_loop import (
    ClosedLoopEpisodeSummary,
    aggregate_closed_loop_summaries,
    write_episode_contact_strip,
)
from sensenova_drone.expert_policy import (
    ReactiveDepthTeacherConfig,
    ReactiveDepthWaypointTeacher,
    compute_depth_clearances,
)
from sensenova_drone.memory import MemoryEntry, RealObservationMemory
from sensenova_drone.observation import Observation
from sensenova_drone.safety import SafetyShield


DEFAULT_GAZEBO_TOPIC = "/world/forest/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
DEFAULT_DEPTH_TOPIC = "/depth_camera"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "output" / "closed_loop_eval_tree_avoidance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--controller", choices=["bc", "reactive_obstacle_teacher"], default="bc")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--gazebo-topic", default=DEFAULT_GAZEBO_TOPIC)
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--connection", default=sitl_runtime.DEFAULT_CONNECTION)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--bridge-wait-s", type=float, default=4.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--takeoff-altitude-m", type=float, default=2.5)
    parser.add_argument("--min-offboard-ready-altitude-m", type=float, default=0.5)
    parser.add_argument("--command-duration-s", type=float, default=1.0)
    parser.add_argument("--action-forward-m-s", type=float, default=0.4)
    parser.add_argument("--action-strafe-m-s", type=float, default=0.5)
    parser.add_argument("--action-vertical-m-s", type=float, default=0.35)
    parser.add_argument("--action-yawspeed-deg-s", type=float, default=12.0)
    parser.add_argument("--settle-duration-s", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--world-label", default="forest")
    parser.add_argument("--collision-imminent-threshold-m", type=float, default=1.6)
    parser.add_argument("--front-blocked-threshold-m", type=float, default=2.5)
    parser.add_argument("--escape-front-clearance-threshold-m", type=float, default=2.5)
    parser.add_argument("--clearance-progress-threshold-m", type=float, default=0.25)
    parser.add_argument("--forward-probe-duration-s", type=float, default=1.5)
    parser.add_argument("--forward-probe-min-progress-m", type=float, default=0.2)
    parser.add_argument("--forward-probe-safe-front-m", type=float, default=1.8)
    parser.add_argument("--require-initially-blocked", action="store_true")
    parser.add_argument("--initial-blocked-check-window-s", type=float, default=2.0)
    parser.add_argument("--initial-blocked-check-max-samples", type=int, default=5)
    parser.add_argument("--teacher-front-blocked-threshold-m", type=float, default=2.5)
    parser.add_argument("--teacher-side-clearance-threshold-m", type=float, default=2.0)
    parser.add_argument("--teacher-front-preferred-threshold-m", type=float, default=4.0)
    parser.add_argument("--max-linear-speed-m-s", type=float, default=0.6)
    parser.add_argument("--max-yawspeed-deg-s", type=float, default=15.0)
    parser.add_argument("--max-duration-s", type=float, default=1.25)
    parser.add_argument("--i-understand-this-is-sitl", action="store_true", dest="i_understand_this_is_sitl")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.i_understand_this_is_sitl:
        raise SystemExit("Refusing to run without --i-understand-this-is-sitl")
    if not args.probe_only and args.episodes <= 0:
        raise SystemExit("--episodes must be > 0")
    if not args.probe_only and args.num_steps <= 0:
        raise SystemExit("--num-steps must be > 0")
    if not sitl_runtime.SAFE_LOCAL_CONNECTION.match(args.connection):
        raise SystemExit("Refusing to run because the connection string is not the local PX4 SITL UDP endpoint.")
    if args.controller == "bc" and not args.probe_only:
        if not args.checkpoint.strip():
            raise SystemExit("--checkpoint is required for --controller bc")
        if not Path(args.checkpoint).expanduser().exists():
            raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")


async def run_eval(args: argparse.Namespace) -> Path:
    run_id = args.run_id or datetime.now(timezone.utc).strftime("tree_eval_%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_root).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    bridge_process = None
    image_buffer = None
    depth_buffer = None
    stop_event = None
    spin_thread = None
    drone = System()
    zero_setpoint = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)

    action_cfg = build_action_cfg(
        duration_s=args.command_duration_s,
        forward_m_s=args.action_forward_m_s,
        strafe_m_s=args.action_strafe_m_s,
        vertical_m_s=args.action_vertical_m_s,
        yawspeed_deg_s=args.action_yawspeed_deg_s,
    )
    policy_runner = None
    if args.controller == "bc":
        policy_runner = load_bc_policy_runner(
            args.checkpoint,
            device=args.device,
            action_cfg=action_cfg,
        )
    teacher = ReactiveDepthWaypointTeacher(
        ReactiveDepthTeacherConfig(
            front_blocked_threshold_m=args.teacher_front_blocked_threshold_m,
            side_clearance_threshold_m=args.teacher_side_clearance_threshold_m,
            front_preferred_threshold_m=args.teacher_front_preferred_threshold_m,
        )
    )
    safety_shield = SafetyShield(
        {
            "safety": {
                "max_linear_speed_m_s": args.max_linear_speed_m_s,
                "max_yawspeed_deg_s": args.max_yawspeed_deg_s,
                "max_duration_s": args.max_duration_s,
                "allow_translation_without_pose": False,
            }
        }
    )

    run_summary: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": (
            str(Path(args.checkpoint).expanduser().resolve())
            if args.checkpoint.strip()
            else ""
        ),
        "controller": args.controller,
        "task_label": "tree_avoidance",
        "gazebo_topic": args.gazebo_topic,
        "depth_topic": args.depth_topic,
        "world_label": args.world_label,
        "episodes": [],
    }

    try:
        bridge_specs = [
            f"{args.gazebo_topic}@sensor_msgs/msg/Image@gz.msgs.Image",
            f"{args.depth_topic}@sensor_msgs/msg/Image@gz.msgs.Image",
        ]
        bridge_process = sitl_runtime.start_bridge(bridge_specs)
        await asyncio.sleep(args.bridge_wait_s)

        sitl_runtime.rclpy.init()
        image_buffer = sitl_runtime.LatestImageBuffer(args.gazebo_topic)
        depth_buffer = sitl_runtime.LatestDepthBuffer(args.depth_topic)
        stop_event, spin_thread = sitl_runtime.start_ros_spin([image_buffer, depth_buffer])

        await asyncio.wait_for(drone.connect(system_address=args.connection), timeout=args.timeout)
        await sitl_runtime.wait_connected(drone, timeout=args.timeout)
        await sitl_runtime.wait_for_frame(image_buffer, min_frame_count=0, timeout=args.timeout)
        await sitl_runtime.wait_for_depth_frame(depth_buffer, min_frame_count=0, timeout=args.timeout)

        await drone.action.set_takeoff_altitude(args.takeoff_altitude_m)
        await sitl_runtime.safe_arm(drone, timeout=min(args.timeout, 20.0))
        await drone.action.takeoff()
        await sitl_runtime.wait_for_async_value(drone.telemetry.in_air(), lambda value: bool(value), timeout=args.timeout)
        with suppress(Exception):
            await sitl_runtime.wait_for_offboard_ready_altitude(
                drone,
                min_relative_altitude_m=args.min_offboard_ready_altitude_m,
                timeout=min(args.timeout, 20.0),
            )
        await asyncio.sleep(2.0)

        await drone.offboard.set_velocity_body(zero_setpoint)
        await drone.offboard.start()

        last_frame_count = 0
        last_depth_frame_count = 0
        closed_loop_summaries: list[ClosedLoopEpisodeSummary] = []

        for episode_index in range(args.episodes):
            if policy_runner is not None:
                policy_runner.reset_history()
            memory = RealObservationMemory()
            episode_id = f"{run_id}_e{episode_index:03d}"
            episode_dir = run_dir / episode_id
            episode_dir.mkdir(parents=True, exist_ok=True)

            episode_payload: dict[str, Any] = {
                "episode_id": episode_id,
                "run_id": run_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_path": run_summary["checkpoint_path"],
                "controller": args.controller,
                "task_label": "tree_avoidance",
                "status": "running",
                "step_count": 0,
                "world_label": args.world_label,
                "actions": [],
            }
            step_payloads: list[dict[str, Any]] = []
            recovery_clear_steps = 0
            recovery_ready = False

            for step_index in range(args.num_steps):
                step_dir = episode_dir / f"step_{step_index:06d}"
                step_dir.mkdir(parents=True, exist_ok=True)

                frame_count, frame, timestamp_s, intrinsics = await sitl_runtime.wait_for_frame(
                    image_buffer,
                    min_frame_count=last_frame_count,
                    timeout=args.timeout,
                )
                last_frame_count = frame_count
                before_path = step_dir / "frame_before.png"
                frame.save(before_path)

                pose_before = await sitl_runtime.estimate_pose(drone)
                if pose_before is None:
                    raise RuntimeError("Failed to estimate pose before action.")

                depth_count, depth_frame, depth_timestamp_s, depth_encoding = await sitl_runtime.wait_for_depth_frame(
                    depth_buffer,
                    min_frame_count=last_depth_frame_count,
                    timeout=args.timeout,
                )
                last_depth_frame_count = depth_count
                depth_before = compute_depth_clearances(depth_frame)
                if step_index == 0 and args.require_initially_blocked:
                    front_before = depth_before.get("front_m")
                    if front_before is None or float(front_before) >= float(args.front_blocked_threshold_m):
                        (
                            last_depth_frame_count,
                            depth_before,
                            blocked_probe,
                        ) = await _recheck_initially_blocked(
                            depth_buffer=depth_buffer,
                            min_frame_count=last_depth_frame_count,
                            threshold_m=float(args.front_blocked_threshold_m),
                            window_s=float(args.initial_blocked_check_window_s),
                            max_samples=int(args.initial_blocked_check_max_samples),
                            timeout=float(args.timeout),
                        )
                        if not blocked_probe["blocked"]:
                            raise RuntimeError(
                                "Scenario is not initially blocked enough for tree eval: "
                                f"front_m={front_before}, "
                                f"min_front_m={blocked_probe['min_front_m']}, "
                                f"samples={blocked_probe['samples_m']}"
                            )

                collision_imminent = (
                    depth_before["front_m"] is not None
                    and float(depth_before["front_m"]) < float(args.collision_imminent_threshold_m)
                )
                observation = Observation(
                    frame_rgb=frame,
                    timestamp_s=timestamp_s,
                    pose=pose_before,
                    intrinsics=intrinsics,
                    metadata={"collision_imminent": collision_imminent},
                )

                if args.controller == "reactive_obstacle_teacher":
                    teacher_decision = teacher.choose_obstacle_avoidance_action(depth_image_m=depth_frame)
                    action = teacher_decision.action
                    proposed_command = discrete_to_command(action, action_cfg)
                    policy_meta: dict[str, Any] = {
                        "mode": "reactive_obstacle_teacher",
                        "reason": teacher_decision.reason,
                        "decision_profile": teacher_decision.diagnostics.get("decision_profile", {}),
                        "teacher_diagnostics": teacher_decision.diagnostics,
                    }
                else:
                    assert policy_runner is not None
                    prediction = policy_runner.predict(frame, goal_features=[0.0, 0.0, 0.0, 0.0])
                    action = prediction.action
                    proposed_command = prediction.command
                    policy_meta = {
                        "mode": "bc",
                        "action_index": prediction.action_index,
                        "confidence": prediction.confidence,
                        "probabilities": prediction.probabilities,
                        "raw_command_prediction": prediction.raw_command_prediction,
                        **prediction.metadata,
                    }

                executed_command = safety_shield.filter(proposed_command, observation, memory)
                command_setpoint = VelocityBodyYawspeed(
                    executed_command.forward_m_s,
                    executed_command.right_m_s,
                    executed_command.down_m_s,
                    executed_command.yawspeed_deg_s,
                )

                await drone.offboard.set_velocity_body(command_setpoint)
                await asyncio.sleep(executed_command.duration_s)
                await drone.offboard.set_velocity_body(zero_setpoint)
                await asyncio.sleep(args.settle_duration_s)

                frame_count, next_frame, next_timestamp_s, _ = await sitl_runtime.wait_for_frame(
                    image_buffer,
                    min_frame_count=last_frame_count,
                    timeout=args.timeout,
                )
                last_frame_count = frame_count
                after_path = step_dir / "frame_after.png"
                next_frame.save(after_path)

                pose_after = await sitl_runtime.estimate_pose(drone)
                if pose_after is None:
                    raise RuntimeError("Failed to estimate pose after action.")

                depth_count, depth_frame_after, _, _ = await sitl_runtime.wait_for_depth_frame(
                    depth_buffer,
                    min_frame_count=last_depth_frame_count,
                    timeout=args.timeout,
                )
                last_depth_frame_count = depth_count
                depth_after = compute_depth_clearances(depth_frame_after)
                front_before = _maybe_float(depth_before.get("front_m"))
                front_after = _maybe_float(depth_after.get("front_m"))
                front_delta = (
                    None
                    if front_before is None or front_after is None
                    else float(front_after - front_before)
                )
                pose_delta = _pose_delta_summary(pose_before, pose_after)
                recovery_clear_step = bool(
                    front_after is not None
                    and front_after >= float(args.escape_front_clearance_threshold_m)
                )
                recovery_clear_steps = recovery_clear_steps + 1 if recovery_clear_step else 0
                recovery_ready = recovery_clear_steps >= 2

                step = BCEpisodeStep(
                    episode_id=episode_id,
                    step_index=step_index,
                    action=action,
                    command=executed_command,
                    image_path=str(before_path.relative_to(episode_dir)),
                    next_image_path=str(after_path.relative_to(episode_dir)),
                    timestamp_s=timestamp_s,
                    pose=pose_before,
                    intrinsics=intrinsics,
                    metadata={
                        "frame_after_timestamp_s": next_timestamp_s,
                        "depth_before": depth_before,
                        "depth_after": depth_after,
                        "depth_encoding": depth_encoding,
                        "depth_timestamp_s": depth_timestamp_s,
                        "policy": policy_meta,
                        "eval": {
                            "safety_override": executed_command != proposed_command,
                            "collision_imminent": collision_imminent,
                            "front_delta_m": front_delta,
                            "local_motion": pose_delta,
                            "recovery_clear_step": recovery_clear_step,
                            "recovery_ready": recovery_ready,
                            "pose_after": {
                                "position_xyz": list(pose_after.position_xyz),
                                "orientation_xyzw": list(pose_after.orientation_xyzw),
                                "metadata": pose_after.metadata,
                            },
                        },
                    },
                )
                save_episode_step(step_dir, step)
                step_payload = step.to_dict()
                step_payloads.append(step_payload)
                episode_payload["actions"].append(action.value)
                memory.append(MemoryEntry(observation=observation, metadata={"source": "tree_eval"}))

                if recovery_ready:
                    break

            probe_summary = await run_forward_probe(
                drone=drone,
                image_buffer=image_buffer,
                depth_buffer=depth_buffer,
                memory=memory,
                safety_shield=safety_shield,
                action_cfg=action_cfg,
                zero_setpoint=zero_setpoint,
                last_frame_count=last_frame_count,
                last_depth_frame_count=last_depth_frame_count,
                settle_duration_s=args.settle_duration_s,
                timeout=args.timeout,
                collision_imminent_threshold_m=args.collision_imminent_threshold_m,
                probe_duration_s=args.forward_probe_duration_s,
            )
            last_frame_count = int(probe_summary["last_frame_count"])
            last_depth_frame_count = int(probe_summary["last_depth_frame_count"])
            episode_payload["probe"] = {
                key: value
                for key, value in probe_summary.items()
                if key not in {"last_frame_count", "last_depth_frame_count"}
            }
            probe_success = bool(
                probe_summary.get("executed_forward_m_s", 0.0) > 0.0
                and probe_summary.get("horizontal_progress_m") is not None
                and float(probe_summary["horizontal_progress_m"]) >= float(args.forward_probe_min_progress_m)
                and not bool(probe_summary.get("collision_imminent_after", True))
                and (
                    probe_summary.get("front_after_m") is None
                    or float(probe_summary["front_after_m"]) >= float(args.forward_probe_safe_front_m)
                )
            )
            episode_payload["status"] = "escaped_blocked_scene" if probe_success else "timeout"

            episode_payload["step_count"] = len(step_payloads)
            contact_strip = write_episode_contact_strip(episode_dir)
            if contact_strip is not None:
                episode_payload["contact_strip_path"] = contact_strip

            summary = summarize_tree_avoidance_episode(
                episode_dir=episode_dir,
                episode_payload=episode_payload,
                step_payloads=step_payloads,
                progress_threshold_m=args.clearance_progress_threshold_m,
            )
            episode_payload["summary"] = _summary_to_dict(summary)
            (episode_dir / "episode.json").write_text(json.dumps(episode_payload, indent=2), encoding="utf-8")
            closed_loop_summaries.append(summary)
            run_summary["episodes"].append(
                {
                    "episode_id": summary.episode_id,
                    "status": summary.status,
                    "step_count": summary.step_count,
                    "front_clearance_delta_m": summary.front_clearance_delta_m,
                    "front_clearance_improved": summary.front_clearance_improved,
                    "reached_goal": summary.reached_goal,
                    "stalled": summary.stalled,
                    "oscillation_rate": summary.oscillation_rate,
                    "mean_confidence": summary.mean_confidence,
                    "episode_dir": summary.episode_dir,
                    "contact_strip_path": summary.contact_strip_path,
                    "task_label": summary.task_label,
                }
            )

        run_summary["aggregate"] = aggregate_closed_loop_summaries(closed_loop_summaries)
        run_summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        print(json.dumps({"run_dir": str(run_dir), "summary": run_summary["aggregate"]}, indent=2))
        return run_dir
    finally:
        with suppress(Exception):
            await drone.offboard.set_velocity_body(zero_setpoint)
        with suppress(OffboardError, Exception):
            await drone.offboard.stop()
        if stop_event is not None:
            stop_event.set()
        if spin_thread is not None:
            spin_thread.join(timeout=5.0)
        if image_buffer is not None:
            image_buffer.destroy_node()
        if depth_buffer is not None:
            depth_buffer.destroy_node()
        with suppress(Exception):
            if sitl_runtime.rclpy.ok():
                sitl_runtime.rclpy.shutdown()
        sitl_runtime.stop_bridge(bridge_process)


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or datetime.now(timezone.utc).strftime("tree_probe_%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_root).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    bridge_process = None
    image_buffer = None
    depth_buffer = None
    stop_event = None
    spin_thread = None
    drone = System()
    zero_setpoint = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)

    probe_summary: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task_label": "tree_probe",
        "gazebo_topic": args.gazebo_topic,
        "depth_topic": args.depth_topic,
        "world_label": args.world_label,
    }

    try:
        bridge_specs = [
            f"{args.gazebo_topic}@sensor_msgs/msg/Image@gz.msgs.Image",
            f"{args.depth_topic}@sensor_msgs/msg/Image@gz.msgs.Image",
        ]
        bridge_process = sitl_runtime.start_bridge(bridge_specs)
        await asyncio.sleep(args.bridge_wait_s)

        sitl_runtime.rclpy.init()
        image_buffer = sitl_runtime.LatestImageBuffer(args.gazebo_topic)
        depth_buffer = sitl_runtime.LatestDepthBuffer(args.depth_topic)
        stop_event, spin_thread = sitl_runtime.start_ros_spin([image_buffer, depth_buffer])

        await asyncio.wait_for(drone.connect(system_address=args.connection), timeout=args.timeout)
        await sitl_runtime.wait_connected(drone, timeout=args.timeout)
        await sitl_runtime.wait_for_frame(image_buffer, min_frame_count=0, timeout=args.timeout)
        depth_count, depth_frame, _, _ = await sitl_runtime.wait_for_depth_frame(
            depth_buffer,
            min_frame_count=0,
            timeout=args.timeout,
        )

        await drone.action.set_takeoff_altitude(args.takeoff_altitude_m)
        await sitl_runtime.safe_arm(drone, timeout=min(args.timeout, 20.0))
        await drone.action.takeoff()
        await sitl_runtime.wait_for_async_value(drone.telemetry.in_air(), lambda value: bool(value), timeout=args.timeout)
        with suppress(Exception):
            await sitl_runtime.wait_for_offboard_ready_altitude(
                drone,
                min_relative_altitude_m=args.min_offboard_ready_altitude_m,
                timeout=min(args.timeout, 20.0),
            )
        await asyncio.sleep(2.0)

        clearances = compute_depth_clearances(depth_frame)
        depth_count, latest_clearances, blocked_probe = await _recheck_initially_blocked(
            depth_buffer=depth_buffer,
            min_frame_count=depth_count,
            threshold_m=float(args.front_blocked_threshold_m),
            window_s=float(args.initial_blocked_check_window_s),
            max_samples=int(args.initial_blocked_check_max_samples),
            timeout=float(args.timeout),
        )
        pose = await sitl_runtime.estimate_pose(drone)
        probe_summary["initial_clearances_m"] = clearances
        probe_summary["latest_clearances_m"] = latest_clearances
        probe_summary["blocked_probe"] = blocked_probe
        probe_summary["pose"] = pose.to_dict() if pose is not None else None
        probe_summary["finished_utc"] = datetime.now(timezone.utc).isoformat()

        (run_dir / "probe.json").write_text(json.dumps(probe_summary, indent=2), encoding="utf-8")
        print(json.dumps({"run_dir": str(run_dir), "probe": probe_summary}, indent=2), flush=True)
        return probe_summary
    finally:
        with suppress(Exception):
            await drone.offboard.set_velocity_body(zero_setpoint)
        with suppress(OffboardError, Exception):
            await drone.offboard.stop()
        if stop_event is not None:
            stop_event.set()
        if spin_thread is not None:
            spin_thread.join(timeout=5.0)
        if image_buffer is not None:
            image_buffer.destroy_node()
        if depth_buffer is not None:
            depth_buffer.destroy_node()
        with suppress(Exception):
            if sitl_runtime.rclpy.ok():
                sitl_runtime.rclpy.shutdown()
        sitl_runtime.stop_bridge(bridge_process)


def _local_pose_dict(pose) -> dict[str, float] | None:
    if pose is None:
        return None
    local = dict(pose.metadata.get("local_position_ned_m", {}))
    if not local:
        return None
    return {
        "north_m": float(local.get("north_m", 0.0)),
        "east_m": float(local.get("east_m", 0.0)),
        "down_m": float(local.get("down_m", 0.0)),
    }


def _yaw_deg(pose) -> float | None:
    if pose is None:
        return None
    attitude = dict(pose.metadata.get("attitude_euler_deg", {}))
    if "yaw_deg" not in attitude:
        return None
    return float(attitude["yaw_deg"])


def _pose_delta_summary(pose_before, pose_after) -> dict[str, float | None]:
    local_before = _local_pose_dict(pose_before)
    local_after = _local_pose_dict(pose_after)
    horizontal_m = None
    vertical_m = None
    north_delta_m = None
    east_delta_m = None
    if local_before is not None and local_after is not None:
        north_delta_m = float(local_after["north_m"] - local_before["north_m"])
        east_delta_m = float(local_after["east_m"] - local_before["east_m"])
        vertical_m = float(-(local_after["down_m"] - local_before["down_m"]))
        horizontal_m = float((north_delta_m**2 + east_delta_m**2) ** 0.5)
    yaw_before = _yaw_deg(pose_before)
    yaw_after = _yaw_deg(pose_after)
    yaw_delta_deg = None
    if yaw_before is not None and yaw_after is not None:
        yaw_delta_deg = float(yaw_after - yaw_before)
    return {
        "north_delta_m": north_delta_m,
        "east_delta_m": east_delta_m,
        "horizontal_m": horizontal_m,
        "vertical_m": vertical_m,
        "yaw_delta_deg": yaw_delta_deg,
    }


async def _recheck_initially_blocked(
    *,
    depth_buffer,
    min_frame_count: int,
    threshold_m: float,
    window_s: float,
    max_samples: int,
    timeout: float,
) -> tuple[int, dict[str, float | None], dict[str, Any]]:
    deadline = asyncio.get_running_loop().time() + max(0.0, window_s)
    current_frame_count = int(min_frame_count)
    samples: list[float] = []
    latest_clearances: dict[str, float | None] = {}
    min_front_m: float | None = None
    max_attempts = max(1, int(max_samples))

    for _ in range(max_attempts):
        remaining_window = deadline - asyncio.get_running_loop().time()
        if remaining_window <= 0.0 and samples:
            break
        per_wait_timeout = max(0.5, min(float(timeout), remaining_window if remaining_window > 0.0 else 0.5))
        depth_count, depth_frame, _, _ = await sitl_runtime.wait_for_depth_frame(
            depth_buffer,
            min_frame_count=current_frame_count,
            timeout=per_wait_timeout,
        )
        current_frame_count = depth_count
        latest_clearances = compute_depth_clearances(depth_frame)
        front_m = _maybe_float(latest_clearances.get("front_m"))
        if front_m is not None:
            samples.append(front_m)
            min_front_m = front_m if min_front_m is None else min(min_front_m, front_m)
            if front_m < threshold_m:
                return current_frame_count, latest_clearances, {
                    "blocked": True,
                    "min_front_m": min_front_m,
                    "samples_m": samples,
                }

    return current_frame_count, latest_clearances, {
        "blocked": False,
        "min_front_m": min_front_m,
        "samples_m": samples,
    }


async def run_forward_probe(
    *,
    drone: System,
    image_buffer,
    depth_buffer,
    memory: RealObservationMemory,
    safety_shield: SafetyShield,
    action_cfg: dict[str, dict[str, float]],
    zero_setpoint: VelocityBodyYawspeed,
    last_frame_count: int,
    last_depth_frame_count: int,
    settle_duration_s: float,
    timeout: float,
    collision_imminent_threshold_m: float,
    probe_duration_s: float,
) -> dict[str, Any]:
    frame_count, frame, timestamp_s, intrinsics = await sitl_runtime.wait_for_frame(
        image_buffer,
        min_frame_count=last_frame_count,
        timeout=timeout,
    )
    last_frame_count = frame_count
    pose_before = await sitl_runtime.estimate_pose(drone)
    if pose_before is None:
        raise RuntimeError("Failed to estimate pose before forward probe.")

    depth_count, depth_frame, _, _ = await sitl_runtime.wait_for_depth_frame(
        depth_buffer,
        min_frame_count=last_depth_frame_count,
        timeout=timeout,
    )
    last_depth_frame_count = depth_count
    depth_before = compute_depth_clearances(depth_frame)
    collision_imminent_before = (
        depth_before["front_m"] is not None
        and float(depth_before["front_m"]) < float(collision_imminent_threshold_m)
    )
    observation = Observation(
        frame_rgb=frame,
        timestamp_s=timestamp_s,
        pose=pose_before,
        intrinsics=intrinsics,
        metadata={"collision_imminent": collision_imminent_before},
    )
    probe_cfg = dict(action_cfg)
    probe_cfg["forward"] = {
        **probe_cfg.get("forward", {}),
        "duration_s": float(probe_duration_s),
    }
    proposed_command = discrete_to_command("forward", probe_cfg)
    executed_command = safety_shield.filter(proposed_command, observation, memory)
    command_setpoint = VelocityBodyYawspeed(
        executed_command.forward_m_s,
        executed_command.right_m_s,
        executed_command.down_m_s,
        executed_command.yawspeed_deg_s,
    )
    await drone.offboard.set_velocity_body(command_setpoint)
    await asyncio.sleep(executed_command.duration_s)
    await drone.offboard.set_velocity_body(zero_setpoint)
    await asyncio.sleep(settle_duration_s)

    pose_after = await sitl_runtime.estimate_pose(drone)
    if pose_after is None:
        raise RuntimeError("Failed to estimate pose after forward probe.")
    depth_count, depth_frame_after, _, _ = await sitl_runtime.wait_for_depth_frame(
        depth_buffer,
        min_frame_count=last_depth_frame_count,
        timeout=timeout,
    )
    last_depth_frame_count = depth_count
    depth_after = compute_depth_clearances(depth_frame_after)
    probe_motion = _pose_delta_summary(pose_before, pose_after)
    front_before = _maybe_float(depth_before.get("front_m"))
    front_after = _maybe_float(depth_after.get("front_m"))
    collision_imminent_after = (
        front_after is not None
        and float(front_after) < float(collision_imminent_threshold_m)
    )
    return {
        "proposed_forward_m_s": proposed_command.forward_m_s,
        "executed_forward_m_s": executed_command.forward_m_s,
        "executed_duration_s": executed_command.duration_s,
        "front_before_m": front_before,
        "front_after_m": front_after,
        "front_delta_m": (
            None if front_before is None or front_after is None else float(front_after - front_before)
        ),
        "collision_imminent_before": collision_imminent_before,
        "collision_imminent_after": collision_imminent_after,
        "horizontal_progress_m": probe_motion.get("horizontal_m"),
        "north_delta_m": probe_motion.get("north_delta_m"),
        "east_delta_m": probe_motion.get("east_delta_m"),
        "vertical_delta_m": probe_motion.get("vertical_m"),
        "yaw_delta_deg": probe_motion.get("yaw_delta_deg"),
        "local_pose_before": _local_pose_dict(pose_before),
        "local_pose_after": _local_pose_dict(pose_after),
        "safety_override": executed_command != proposed_command,
        "last_frame_count": last_frame_count,
        "last_depth_frame_count": last_depth_frame_count,
    }


def summarize_tree_avoidance_episode(
    *,
    episode_dir: str | Path,
    episode_payload: dict[str, Any],
    step_payloads: list[dict[str, Any]],
    progress_threshold_m: float,
) -> ClosedLoopEpisodeSummary:
    actions = [str(step.get("action", "unknown")) for step in step_payloads]
    safety_override_count = sum(
        1
        for step in step_payloads
        if bool(step.get("metadata", {}).get("eval", {}).get("safety_override", False))
    )
    collision_imminent_count = sum(
        1
        for step in step_payloads
        if bool(step.get("metadata", {}).get("eval", {}).get("collision_imminent", False))
    )
    front_before_values = [
        _maybe_float(step.get("metadata", {}).get("depth_before", {}).get("front_m"))
        for step in step_payloads
    ]
    front_after_values = [
        _maybe_float(step.get("metadata", {}).get("depth_after", {}).get("front_m"))
        for step in step_payloads
    ]
    valid_front_before = [value for value in front_before_values if value is not None]
    valid_front_after = [value for value in front_after_values if value is not None]
    initial_front = next((value for value in valid_front_before), None)
    final_front = next((value for value in reversed(valid_front_after) if value is not None), None)

    front_delta_values = [
        _maybe_float(step.get("metadata", {}).get("eval", {}).get("front_delta_m"))
        for step in step_payloads
    ]
    valid_front_deltas = [value for value in front_delta_values if value is not None]
    probe = dict(episode_payload.get("probe", {}))
    probe_progress_m = _maybe_float(probe.get("horizontal_progress_m")) or 0.0
    probe_front_after_m = _maybe_float(probe.get("front_after_m"))
    probe_front_delta_m = _maybe_float(probe.get("front_delta_m"))
    net_front_delta = (
        probe_front_delta_m
        if probe_front_delta_m is not None
        else (
            0.0
            if initial_front is None or final_front is None
            else float(final_front - initial_front)
        )
    )
    escaped_scene = bool(str(episode_payload.get("status", "")) == "escaped_blocked_scene")
    moved_toward_clearance = bool(escaped_scene or probe_progress_m >= float(progress_threshold_m))
    stalled = bool(
        len(step_payloads) >= 3
        and not escaped_scene
        and probe_progress_m < float(progress_threshold_m)
    )
    oscillation_flips = 0
    for prev, nxt in zip(actions[:-1], actions[1:]):
        if (prev, nxt) in {("yaw_left", "yaw_right"), ("yaw_right", "yaw_left"), ("strafe_left", "strafe_right"), ("strafe_right", "strafe_left")}:
            oscillation_flips += 1

    confidences = [
        _maybe_float(step.get("metadata", {}).get("policy", {}).get("confidence"))
        for step in step_payloads
    ]
    valid_confidences = [value for value in confidences if value is not None]

    return ClosedLoopEpisodeSummary(
        episode_id=str(episode_payload.get("episode_id", Path(episode_dir).name)),
        run_id=str(episode_payload.get("run_id", "unknown")),
        checkpoint_path=str(episode_payload.get("checkpoint_path", "")),
        status=str(episode_payload.get("status", "unknown")),
        step_count=int(episode_payload.get("step_count", len(step_payloads))),
        actions=actions,
        initial_distance_xy_m=None,
        final_distance_xy_m=None,
        net_progress_m=probe_progress_m,
        mean_step_progress_m=(
            float(sum(valid_front_deltas) / len(valid_front_deltas))
            if valid_front_deltas
            else 0.0
        ),
        progress_ratio=None,
        moved_toward_goal=moved_toward_clearance,
        reached_goal=escaped_scene,
        stalled=stalled,
        oscillation_flips=oscillation_flips,
        oscillation_rate=(oscillation_flips / max(len(actions) - 1, 1)) if actions else 0.0,
        safety_override_count=safety_override_count,
        collision_imminent_count=collision_imminent_count,
        mean_front_clearance_before_m=(
            float(sum(valid_front_before) / len(valid_front_before))
            if valid_front_before
            else None
        ),
        mean_front_clearance_after_m=(
            probe_front_after_m
            if probe_front_after_m is not None
            else (
                float(sum(valid_front_after) / len(valid_front_after))
                if valid_front_after
                else None
            )
        ),
        min_front_clearance_before_m=min(valid_front_before) if valid_front_before else None,
        min_front_clearance_after_m=(
            probe_front_after_m
            if probe_front_after_m is not None
            else (min(valid_front_after) if valid_front_after else None)
        ),
        front_clearance_delta_m=net_front_delta,
        front_clearance_improved=escaped_scene,
        mean_confidence=(
            float(sum(valid_confidences) / len(valid_confidences))
            if valid_confidences
            else None
        ),
        goal_reached_radius_m=0.0,
        altitude_tolerance_m=0.0,
        contact_strip_path=_maybe_str(episode_payload.get("contact_strip_path")),
        episode_dir=str(Path(episode_dir).resolve()),
        task_label="tree_avoidance",
    )


def _summary_to_dict(summary: ClosedLoopEpisodeSummary) -> dict[str, Any]:
    return {
        "episode_id": summary.episode_id,
        "run_id": summary.run_id,
        "checkpoint_path": summary.checkpoint_path,
        "status": summary.status,
        "step_count": summary.step_count,
        "actions": summary.actions,
        "initial_distance_xy_m": summary.initial_distance_xy_m,
        "final_distance_xy_m": summary.final_distance_xy_m,
        "net_progress_m": summary.net_progress_m,
        "mean_step_progress_m": summary.mean_step_progress_m,
        "progress_ratio": summary.progress_ratio,
        "moved_toward_goal": summary.moved_toward_goal,
        "reached_goal": summary.reached_goal,
        "stalled": summary.stalled,
        "oscillation_flips": summary.oscillation_flips,
        "oscillation_rate": summary.oscillation_rate,
        "safety_override_count": summary.safety_override_count,
        "collision_imminent_count": summary.collision_imminent_count,
        "mean_front_clearance_before_m": summary.mean_front_clearance_before_m,
        "mean_front_clearance_after_m": summary.mean_front_clearance_after_m,
        "min_front_clearance_before_m": summary.min_front_clearance_before_m,
        "min_front_clearance_after_m": summary.min_front_clearance_after_m,
        "front_clearance_delta_m": summary.front_clearance_delta_m,
        "front_clearance_improved": summary.front_clearance_improved,
        "mean_confidence": summary.mean_confidence,
        "goal_reached_radius_m": summary.goal_reached_radius_m,
        "altitude_tolerance_m": summary.altitude_tolerance_m,
        "contact_strip_path": summary.contact_strip_path,
        "episode_dir": summary.episode_dir,
        "task_label": summary.task_label,
    }


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


async def main_async() -> int:
    args = parse_args()
    validate_args(args)
    if args.probe_only:
        await run_probe(args)
    else:
        await run_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
