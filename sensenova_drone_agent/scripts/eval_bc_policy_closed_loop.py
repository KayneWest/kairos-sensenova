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

from sensenova_drone.actions import DiscreteDroneAction, build_action_cfg
from sensenova_drone.bc_data import BCEpisodeStep, save_episode_step
from sensenova_drone.bc_infer import load_bc_policy_runner
from sensenova_drone.eval.closed_loop import (
    aggregate_closed_loop_summaries,
    compute_goal_metrics,
    goal_feature_vector_from_metrics,
    summarize_closed_loop_episode,
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
DEFAULT_OUT_ROOT = PROJECT_ROOT / "output" / "closed_loop_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument("--command-duration-s", type=float, default=0.75)
    parser.add_argument("--action-forward-m-s", type=float, default=0.3)
    parser.add_argument("--action-strafe-m-s", type=float, default=0.3)
    parser.add_argument("--action-vertical-m-s", type=float, default=0.3)
    parser.add_argument("--action-yawspeed-deg-s", type=float, default=5.0)
    parser.add_argument("--settle-duration-s", type=float, default=0.5)
    parser.add_argument("--goal-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--world-label", default="forest")
    parser.add_argument("--goal-feature-mode", choices=["recorded", "zeros"], default="recorded")
    parser.add_argument("--collision-imminent-threshold-m", type=float, default=1.5)
    parser.add_argument("--teacher-min-goal-forward-m", type=float, default=3.0)
    parser.add_argument("--teacher-max-goal-forward-m", type=float, default=8.0)
    parser.add_argument("--teacher-max-goal-lateral-m", type=float, default=4.0)
    parser.add_argument("--teacher-max-goal-vertical-m", type=float, default=0.0)
    parser.add_argument("--teacher-goal-reached-radius-m", type=float, default=1.0)
    parser.add_argument("--teacher-altitude-tolerance-m", type=float, default=0.35)
    parser.add_argument("--teacher-heading-error-threshold-deg", type=float, default=15.0)
    parser.add_argument("--teacher-front-blocked-threshold-m", type=float, default=2.5)
    parser.add_argument("--teacher-side-clearance-threshold-m", type=float, default=2.0)
    parser.add_argument("--teacher-front-preferred-threshold-m", type=float, default=4.0)
    parser.add_argument("--max-linear-speed-m-s", type=float, default=0.5)
    parser.add_argument("--max-yawspeed-deg-s", type=float, default=10.0)
    parser.add_argument("--max-duration-s", type=float, default=1.0)
    parser.add_argument("--i-understand-this-is-sitl", action="store_true", dest="i_understand_this_is_sitl")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.i_understand_this_is_sitl:
        raise SystemExit("Refusing to run without --i-understand-this-is-sitl")
    if args.episodes <= 0:
        raise SystemExit("--episodes must be > 0")
    if args.num_steps <= 0:
        raise SystemExit("--num-steps must be > 0")
    if not sitl_runtime.SAFE_LOCAL_CONNECTION.match(args.connection):
        raise SystemExit("Refusing to run because the connection string is not the local PX4 SITL UDP endpoint.")
    if not Path(args.checkpoint).expanduser().exists():
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")


async def run_eval(args: argparse.Namespace) -> Path:
    run_id = args.run_id or datetime.now(timezone.utc).strftime("eval_%Y%m%dT%H%M%SZ")
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
    policy_runner = load_bc_policy_runner(
        args.checkpoint,
        device=args.device,
        action_cfg=action_cfg,
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
    teacher = ReactiveDepthWaypointTeacher(
        ReactiveDepthTeacherConfig(
            min_goal_forward_m=args.teacher_min_goal_forward_m,
            max_goal_forward_m=args.teacher_max_goal_forward_m,
            max_goal_lateral_m=args.teacher_max_goal_lateral_m,
            max_goal_vertical_m=args.teacher_max_goal_vertical_m,
            goal_reached_radius_m=args.teacher_goal_reached_radius_m,
            altitude_tolerance_m=args.teacher_altitude_tolerance_m,
            heading_error_threshold_deg=args.teacher_heading_error_threshold_deg,
            front_blocked_threshold_m=args.teacher_front_blocked_threshold_m,
            side_clearance_threshold_m=args.teacher_side_clearance_threshold_m,
            front_preferred_threshold_m=args.teacher_front_preferred_threshold_m,
        )
    )
    rng = random.Random(args.goal_seed or abs(hash(run_id)) % (2**31))

    run_summary: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(Path(args.checkpoint).expanduser().resolve()),
        "gazebo_topic": args.gazebo_topic,
        "depth_topic": args.depth_topic,
        "world_label": args.world_label,
        "episodes": [],
    }

    try:
        bridge_specs = [f"{args.gazebo_topic}@sensor_msgs/msg/Image@gz.msgs.Image"]
        if args.depth_topic.strip():
            bridge_specs.append(f"{args.depth_topic}@sensor_msgs/msg/Image@gz.msgs.Image")
        bridge_process = sitl_runtime.start_bridge(bridge_specs)
        await asyncio.sleep(args.bridge_wait_s)

        sitl_runtime.rclpy.init()
        image_buffer = sitl_runtime.LatestImageBuffer(args.gazebo_topic)
        nodes = [image_buffer]
        if args.depth_topic.strip():
            depth_buffer = sitl_runtime.LatestDepthBuffer(args.depth_topic)
            nodes.append(depth_buffer)
        stop_event, spin_thread = sitl_runtime.start_ros_spin(nodes)

        await asyncio.wait_for(drone.connect(system_address=args.connection), timeout=args.timeout)
        await sitl_runtime.wait_connected(drone, timeout=args.timeout)
        await sitl_runtime.wait_for_frame(image_buffer, min_frame_count=0, timeout=args.timeout)
        if depth_buffer is not None:
            await sitl_runtime.wait_for_depth_frame(depth_buffer, min_frame_count=0, timeout=args.timeout)

        await drone.action.set_takeoff_altitude(args.takeoff_altitude_m)
        await drone.action.arm()
        await drone.action.takeoff()
        await sitl_runtime.wait_for_async_value(drone.telemetry.in_air(), lambda value: bool(value), timeout=args.timeout)
        await asyncio.sleep(3.0)

        await drone.offboard.set_velocity_body(zero_setpoint)
        await drone.offboard.start()

        home_pose = await sitl_runtime.estimate_pose(drone)
        if home_pose is None:
            raise RuntimeError("Failed to estimate home pose.")

        last_frame_count = 0
        last_depth_frame_count = 0
        closed_loop_summaries = []

        for episode_index in range(args.episodes):
            memory = RealObservationMemory()
            policy_runner.reset_history()
            episode_id = f"{run_id}_e{episode_index:03d}"
            episode_dir = run_dir / episode_id
            episode_dir.mkdir(parents=True, exist_ok=True)

            start_pose = await sitl_runtime.estimate_pose(drone)
            if start_pose is None:
                raise RuntimeError("Failed to estimate start pose.")
            goal = teacher.sample_goal(home_pose=home_pose, current_pose=start_pose, rng=rng)

            episode_payload: dict[str, Any] = {
                "episode_id": episode_id,
                "run_id": run_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_path": str(Path(args.checkpoint).expanduser().resolve()),
                "status": "running",
                "step_count": 0,
                "goal_local_xyz_m": list(goal.local_position_xyz_m),
                "goal_metadata": goal.metadata,
                "goal_reached_radius_m": args.teacher_goal_reached_radius_m,
                "altitude_tolerance_m": args.teacher_altitude_tolerance_m,
                "world_label": args.world_label,
                "actions": [],
            }
            step_payloads: list[dict[str, Any]] = []

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

                depth_frame = None
                depth_encoding = None
                depth_timestamp_s = None
                if depth_buffer is not None:
                    depth_count, depth_frame, depth_timestamp_s, depth_encoding = await sitl_runtime.wait_for_depth_frame(
                        depth_buffer,
                        min_frame_count=last_depth_frame_count,
                        timeout=args.timeout,
                    )
                    last_depth_frame_count = depth_count

                goal_before = compute_goal_metrics(
                    home_pose=home_pose,
                    current_pose=pose_before,
                    goal_local_xyz_m=goal.local_position_xyz_m,
                )
                depth_before = compute_depth_clearances(depth_frame)
                goal_features = goal_feature_vector_from_metrics(goal_before)
                if args.goal_feature_mode == "zeros":
                    goal_features = [0.0, 0.0, 0.0, 0.0]
                prediction = policy_runner.predict(frame, goal_features=goal_features)
                proposed_command = prediction.command

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

                depth_after = None
                if depth_buffer is not None:
                    depth_count, depth_frame_after, _, _ = await sitl_runtime.wait_for_depth_frame(
                        depth_buffer,
                        min_frame_count=last_depth_frame_count,
                        timeout=args.timeout,
                    )
                    last_depth_frame_count = depth_count
                    depth_after = compute_depth_clearances(depth_frame_after)

                goal_after = compute_goal_metrics(
                    home_pose=home_pose,
                    current_pose=pose_after,
                    goal_local_xyz_m=goal.local_position_xyz_m,
                )
                progress_m = float(goal_before["distance_xy_m"] - goal_after["distance_xy_m"])
                reached_goal = teacher.goal_reached(
                    current_local_xyz_m=(
                        goal_after["current_local_x_m"],
                        goal_after["current_local_y_m"],
                        goal_after["current_local_z_m"],
                    ),
                    goal=goal,
                )

                step = BCEpisodeStep(
                    episode_id=episode_id,
                    step_index=step_index,
                    action=prediction.action,
                    command=executed_command,
                    image_path=str(before_path.relative_to(episode_dir)),
                    next_image_path=str(after_path.relative_to(episode_dir)),
                    timestamp_s=timestamp_s,
                    pose=pose_before,
                    intrinsics=intrinsics,
                    metadata={
                        "frame_after_timestamp_s": next_timestamp_s,
                        "policy": {
                            "action_index": prediction.action_index,
                            "confidence": prediction.confidence,
                            "probabilities": prediction.probabilities,
                            "raw_command_prediction": prediction.raw_command_prediction,
                            **prediction.metadata,
                        },
                        "goal_before": goal_before,
                        "goal_after": goal_after,
                        "goal_feature_vector": goal_features,
                        "goal_feature_mode": args.goal_feature_mode,
                        "depth_before": depth_before,
                        "depth_after": depth_after,
                        "depth_encoding": depth_encoding,
                        "depth_timestamp_s": depth_timestamp_s,
                        "eval": {
                            "safety_override": executed_command != proposed_command,
                            "collision_imminent": collision_imminent,
                            "progress_m": progress_m,
                            "reached_goal": reached_goal,
                            "pose_after": {
                                "position_xyz": list(pose_after.position_xyz),
                                "orientation_xyzw": list(pose_after.orientation_xyzw),
                                "metadata": pose_after.metadata,
                            },
                        },
                    },
                )
                save_episode_step(step_dir, step)
                step_payloads.append(step.to_dict())
                episode_payload["actions"].append(prediction.action.value)
                memory.append(MemoryEntry(observation=observation, metadata={"source": "closed_loop_eval"}))

                if reached_goal:
                    episode_payload["status"] = "goal_reached"
                    break
            else:
                episode_payload["status"] = "timeout"

            episode_payload["step_count"] = len(step_payloads)
            contact_strip = write_episode_contact_strip(episode_dir)
            if contact_strip is not None:
                episode_payload["contact_strip_path"] = contact_strip

            summary = summarize_closed_loop_episode(
                episode_dir=episode_dir,
                episode_payload=episode_payload,
                step_payloads=step_payloads,
                goal_reached_radius_m=args.teacher_goal_reached_radius_m,
                altitude_tolerance_m=args.teacher_altitude_tolerance_m,
            )
            episode_payload["summary"] = {
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
                "mean_confidence": summary.mean_confidence,
            }
            (episode_dir / "episode.json").write_text(json.dumps(episode_payload, indent=2), encoding="utf-8")
            closed_loop_summaries.append(summary)
            run_summary["episodes"].append(
                {
                    "episode_id": summary.episode_id,
                    "status": summary.status,
                    "step_count": summary.step_count,
                    "net_progress_m": summary.net_progress_m,
                    "reached_goal": summary.reached_goal,
                    "stalled": summary.stalled,
                    "oscillation_rate": summary.oscillation_rate,
                    "mean_confidence": summary.mean_confidence,
                    "episode_dir": summary.episode_dir,
                    "contact_strip_path": summary.contact_strip_path,
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
        with suppress(Exception):
            await sitl_runtime.safe_land(drone, timeout=args.timeout)
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


async def main_async() -> int:
    args = parse_args()
    validate_args(args)
    await run_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
