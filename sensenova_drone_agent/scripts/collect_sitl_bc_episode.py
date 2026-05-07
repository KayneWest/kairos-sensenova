#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    print(f"Missing dependency: {exc}", file=sys.stderr)
    raise SystemExit(1)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
except ModuleNotFoundError as exc:
    print(
        "ROS 2 Python packages are missing in this environment. "
        "Run this script inside the tools container with ROS sourced.",
        file=sys.stderr,
    )
    raise SystemExit(1)

try:
    from mavsdk import System
    from mavsdk.action import ActionError
    from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
except ModuleNotFoundError:
    print(
        "mavsdk is not installed in this Python environment. "
        "Run this script inside the tools container or the dedicated MAVSDK environment.",
        file=sys.stderr,
    )
    raise SystemExit(1)

if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
    # ROS 2 Jazzy's cv_bridge wheels in this image are built against NumPy 1.x.
    # Prefer the manual conversion path until the container is rebuilt with numpy<2.
    CvBridge = None
else:
    try:
        from cv_bridge import CvBridge
    except Exception:
        CvBridge = None

from PIL import Image

from sensenova_drone.actions import (
    DiscreteDroneAction,
    build_action_cfg,
    coerce_discrete_action,
    discrete_to_command,
)
from sensenova_drone.bc_data import BCEpisodeStep, save_episode_step
from sensenova_drone.expert_policy import (
    ReactiveDepthTeacherConfig,
    ReactiveDepthWaypointTeacher,
    latlon_alt_to_local_m,
    normalize_goal_features,
)
from sensenova_drone.observation import CameraIntrinsics, Pose


DEFAULT_CONNECTION = "udpin://0.0.0.0:14540"
SAFE_LOCAL_CONNECTION = re.compile(r"^udpin://(0\.0\.0\.0|127\.0\.0\.1|localhost):14540$")
DEFAULT_EPISODES_ROOT = PROJECT_ROOT / "data" / "bc_sft" / "episodes"


def convert_common_encoding(msg: RosImage) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    row_data = data.reshape((height, step))
    encoding = msg.encoding.lower()

    if encoding == "rgb8":
        trimmed = row_data[:, : width * 3]
        return trimmed.reshape((height, width, 3))
    if encoding == "bgr8":
        trimmed = row_data[:, : width * 3]
        bgr = trimmed.reshape((height, width, 3))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if encoding == "rgba8":
        trimmed = row_data[:, : width * 4]
        rgba = trimmed.reshape((height, width, 4))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    if encoding == "bgra8":
        trimmed = row_data[:, : width * 4]
        bgra = trimmed.reshape((height, width, 4))
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    if encoding == "mono8":
        trimmed = row_data[:, :width]
        gray = trimmed.reshape((height, width))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    raise ValueError(f"Unsupported encoding without cv_bridge: {msg.encoding}")


def convert_depth_encoding(msg: RosImage) -> np.ndarray:
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    row_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, step))
    encoding = msg.encoding.lower()

    if encoding in {"32fc1", "32fc"}:
        dtype = np.float32
        channels = 1
    elif encoding in {"16uc1", "16uc"}:
        dtype = np.uint16
        channels = 1
    elif encoding in {"16sc1", "16sc"}:
        dtype = np.int16
        channels = 1
    elif encoding in {"8uc1", "8uc", "mono8"}:
        dtype = np.uint8
        channels = 1
    else:
        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")

    bytes_per_value = np.dtype(dtype).itemsize
    trimmed = row_data[:, : width * channels * bytes_per_value]
    depth = np.frombuffer(trimmed.tobytes(), dtype=dtype).reshape((height, width, channels))
    depth = depth[..., 0].astype(np.float32)
    if encoding.startswith("16u") and np.nanpercentile(depth, 90) > 50.0:
        depth = depth / 1000.0
    depth[~np.isfinite(depth)] = np.nan
    depth[depth <= 0.0] = np.nan
    return depth


class LatestImageBuffer(Node):
    def __init__(self, topic: str):
        super().__init__("bc_episode_image_buffer")
        self._bridge = CvBridge() if CvBridge is not None else None
        self._lock = threading.Lock()
        self._frame: Image.Image | None = None
        self._frame_count = 0
        self._timestamp_s: float | None = None
        self._intrinsics: CameraIntrinsics | None = None
        self.create_subscription(RosImage, topic, self._callback, 10)

    def _callback(self, msg: RosImage) -> None:
        try:
            if self._bridge is not None:
                try:
                    cv_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                except Exception as exc:
                    self.get_logger().warning(
                        f"cv_bridge conversion failed, falling back to manual decoding: {exc}"
                    )
                    self._bridge = None
                    cv_frame = convert_common_encoding(msg)
            else:
                cv_frame = convert_common_encoding(msg)
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert image frame: {exc}")
            return

        if isinstance(cv_frame, np.ndarray):
            pil_image = Image.fromarray(np.asarray(cv_frame, dtype=np.uint8), mode="RGB")
        else:
            pil_image = Image.fromarray(np.asarray(cv_frame), mode="RGB")

        timestamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1_000_000_000.0
        intrinsics = CameraIntrinsics(width=int(msg.width), height=int(msg.height), frame_id=msg.header.frame_id)

        with self._lock:
            self._frame = pil_image
            self._frame_count += 1
            self._timestamp_s = timestamp_s
            self._intrinsics = intrinsics

    def snapshot(self) -> tuple[int, Image.Image | None, float | None, CameraIntrinsics | None]:
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
            intrinsics = self._intrinsics
            return self._frame_count, frame, self._timestamp_s, intrinsics


class LatestDepthBuffer(Node):
    def __init__(self, topic: str):
        super().__init__("bc_episode_depth_buffer")
        self._bridge = CvBridge() if CvBridge is not None else None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_count = 0
        self._timestamp_s: float | None = None
        self._encoding: str | None = None
        self.create_subscription(RosImage, topic, self._callback, 10)

    def _callback(self, msg: RosImage) -> None:
        try:
            if self._bridge is not None:
                try:
                    cv_frame = self._bridge.imgmsg_to_cv2(msg)
                    depth_frame = np.asarray(cv_frame, dtype=np.float32)
                except Exception as exc:
                    self.get_logger().warning(
                        f"cv_bridge depth conversion failed, falling back to manual decoding: {exc}"
                    )
                    self._bridge = None
                    depth_frame = convert_depth_encoding(msg)
            else:
                depth_frame = convert_depth_encoding(msg)
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert depth frame: {exc}")
            return

        if msg.encoding.lower().startswith("16u") and np.nanpercentile(depth_frame, 90) > 50.0:
            depth_frame = depth_frame / 1000.0
        depth_frame = np.asarray(depth_frame, dtype=np.float32)
        depth_frame[~np.isfinite(depth_frame)] = np.nan
        depth_frame[depth_frame <= 0.0] = np.nan

        timestamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1_000_000_000.0
        with self._lock:
            self._frame = depth_frame.copy()
            self._frame_count += 1
            self._timestamp_s = timestamp_s
            self._encoding = msg.encoding

    def snapshot(self) -> tuple[int, np.ndarray | None, float | None, str | None]:
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
            return self._frame_count, frame, self._timestamp_s, self._encoding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gazebo-topic", required=True, help="Gazebo image topic to bridge and record.")
    parser.add_argument("--depth-topic", default="", help="Optional Gazebo depth topic used by reactive teacher mode.")
    parser.add_argument("--connection", default=DEFAULT_CONNECTION)
    parser.add_argument("--episodes-root", default=str(DEFAULT_EPISODES_ROOT))
    parser.add_argument("--episode-id", default="")
    parser.add_argument("--world-label", default="unknown")
    parser.add_argument("--scenario-label", default="")
    parser.add_argument("--policy", choices=["scripted", "reactive_teacher", "reactive_obstacle_teacher"], default="scripted")
    parser.add_argument(
        "--actions",
        default="hover,forward,yaw_left,forward,yaw_right,hover",
        help="Comma-separated discrete action sequence.",
    )
    parser.add_argument("--num-steps", type=int, default=6, help="Used by reactive teacher mode.")
    parser.add_argument("--teacher-seed", type=int, default=0)
    parser.add_argument("--teacher-min-goal-forward-m", type=float, default=3.0)
    parser.add_argument("--teacher-max-goal-forward-m", type=float, default=8.0)
    parser.add_argument("--teacher-max-goal-lateral-m", type=float, default=4.0)
    parser.add_argument("--teacher-max-goal-vertical-m", type=float, default=0.0)
    parser.add_argument("--teacher-goal-mode", choices=["random", "decision_game"], default="random")
    parser.add_argument("--teacher-target-families", default="forward,left,right")
    parser.add_argument("--teacher-fixed-target-family", default="")
    parser.add_argument("--teacher-goal-resample-attempts", type=int, default=8)
    parser.add_argument("--teacher-min-decision-score", type=float, default=0.55)
    parser.add_argument("--teacher-require-initially-blocked", action="store_true")
    parser.add_argument("--teacher-require-initial-target-action", action="store_true")
    parser.add_argument("--teacher-min-initial-branch-score", type=float, default=0.0)
    parser.add_argument("--teacher-refresh-goal-every-step", action="store_true")
    parser.add_argument("--teacher-goal-reached-radius-m", type=float, default=1.0)
    parser.add_argument("--teacher-altitude-tolerance-m", type=float, default=0.35)
    parser.add_argument("--teacher-heading-error-threshold-deg", type=float, default=15.0)
    parser.add_argument("--teacher-front-blocked-threshold-m", type=float, default=2.5)
    parser.add_argument("--teacher-side-clearance-threshold-m", type=float, default=2.0)
    parser.add_argument("--teacher-front-preferred-threshold-m", type=float, default=4.0)
    parser.add_argument("--teacher-target-side-preference-margin-m", type=float, default=0.4)
    parser.add_argument("--takeoff-altitude-m", type=float, default=2.5)
    parser.add_argument("--min-offboard-ready-altitude-m", type=float, default=0.5)
    parser.add_argument("--command-duration-s", type=float, default=0.75)
    parser.add_argument("--action-forward-m-s", type=float, default=0.3)
    parser.add_argument("--action-strafe-m-s", type=float, default=0.3)
    parser.add_argument("--action-vertical-m-s", type=float, default=0.3)
    parser.add_argument("--action-yawspeed-deg-s", type=float, default=5.0)
    parser.add_argument("--settle-duration-s", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--bridge-wait-s", type=float, default=3.0)
    parser.add_argument("--i-understand-this-is-sitl", action="store_true", dest="i_understand_this_is_sitl")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.i_understand_this_is_sitl:
        raise SystemExit("Refusing to run without --i-understand-this-is-sitl")
    if not SAFE_LOCAL_CONNECTION.match(args.connection):
        raise SystemExit("Refusing to run because the connection string is not the local PX4 SITL UDP endpoint.")
    if args.policy in {"reactive_teacher", "reactive_obstacle_teacher"} and not args.depth_topic.strip():
        raise SystemExit("Reactive teacher modes require --depth-topic.")
    if args.policy in {"reactive_teacher", "reactive_obstacle_teacher"} and args.num_steps <= 0:
        raise SystemExit("Reactive teacher modes require --num-steps > 0.")
    if args.teacher_fixed_target_family.strip():
        allowed = {"forward", "left", "right", "ascend", "descend"}
        if args.teacher_fixed_target_family.strip().lower() not in allowed:
            raise SystemExit(f"Unsupported --teacher-fixed-target-family: {args.teacher_fixed_target_family}")


def parse_action_sequence(raw: str) -> list[DiscreteDroneAction]:
    return [coerce_discrete_action(token.strip()) for token in raw.split(",") if token.strip()]


def initial_action_matches_target_family(action: DiscreteDroneAction, target_family: str | None) -> bool:
    family = (target_family or "").strip().lower()
    allowed: dict[str, set[DiscreteDroneAction]] = {
        "forward": {DiscreteDroneAction.FORWARD},
        "left": {DiscreteDroneAction.STRAFE_LEFT, DiscreteDroneAction.YAW_LEFT},
        "right": {DiscreteDroneAction.STRAFE_RIGHT, DiscreteDroneAction.YAW_RIGHT},
        "ascend": {DiscreteDroneAction.ASCEND},
        "descend": {DiscreteDroneAction.DESCEND},
    }
    if family not in allowed:
        return True
    return action in allowed[family]


async def wait_connected(drone: System, timeout: float) -> None:
    async def _wait() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return

    await asyncio.wait_for(_wait(), timeout=timeout)


async def wait_for_async_value(async_iterable, predicate, timeout: float):
    async def _wait():
        async for value in async_iterable:
            if predicate(value):
                return value

    return await asyncio.wait_for(_wait(), timeout=timeout)


async def wait_for_offboard_ready_altitude(drone: System, *, min_relative_altitude_m: float, timeout: float) -> None:
    min_altitude = float(max(0.0, min_relative_altitude_m))

    async def _wait() -> None:
        async for position in drone.telemetry.position():
            if float(position.relative_altitude_m) >= min_altitude:
                return

    await asyncio.wait_for(_wait(), timeout=timeout)


async def safe_arm(drone: System, *, timeout: float, retry_delay_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            await drone.action.arm()
            return
        except ActionError as exc:
            last_error = exc
            print(f"Arming retry after MAVSDK action error: {exc}", file=sys.stderr)
        except Exception as exc:
            last_error = exc
            print(f"Arming retry after unexpected error: {exc}", file=sys.stderr)
        await asyncio.sleep(retry_delay_s)
    if last_error is not None:
        raise last_error
    raise TimeoutError("Timed out while attempting to arm PX4 SITL.")


async def wait_for_frame(buffer: LatestImageBuffer, *, min_frame_count: int, timeout: float) -> tuple[int, Image.Image, float | None, CameraIntrinsics | None]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame_count, frame, timestamp_s, intrinsics = buffer.snapshot()
        if frame is not None and frame_count > min_frame_count:
            return frame_count, frame, timestamp_s, intrinsics
        await asyncio.sleep(0.05)
    raise TimeoutError("Timed out waiting for a ROS image frame.")


async def wait_for_depth_frame(
    buffer: LatestDepthBuffer,
    *,
    min_frame_count: int,
    timeout: float,
) -> tuple[int, np.ndarray, float | None, str | None]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame_count, frame, timestamp_s, encoding = buffer.snapshot()
        if frame is not None and frame_count > min_frame_count:
            return frame_count, frame, timestamp_s, encoding
        await asyncio.sleep(0.05)
    raise TimeoutError("Timed out waiting for a ROS depth frame.")


def start_ros_spin(nodes: list[Node]) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def _spin() -> None:
        while rclpy.ok() and not stop_event.is_set():
            for index, node in enumerate(nodes):
                timeout_sec = 0.1 if index == 0 else 0.0
                rclpy.spin_once(node, timeout_sec=timeout_sec)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    return stop_event, thread


def start_bridge(bridges: list[str]) -> subprocess.Popen[str]:
    bridge_args = " ".join(f"'{bridge}'" for bridge in bridges)
    bridge_command = [
        "bash",
        "-lc",
        (
            "source /opt/ros/${ROS_DISTRO}/setup.bash && "
            f"ros2 run ros_gz_bridge parameter_bridge {bridge_args}"
        ),
    ]
    return subprocess.Popen(
        bridge_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )


def stop_bridge(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    with suppress(Exception):
        process.wait(timeout=5.0)


async def _latest_bool(async_iterable, *, timeout: float) -> bool | None:
    try:
        value = await wait_for_async_value(async_iterable, lambda _: True, timeout=timeout)
    except Exception:
        return None
    return bool(value)


async def safe_land(drone: System, timeout: float) -> None:
    probe_timeout = min(timeout, 5.0)
    land_timeout = min(timeout, 10.0)

    in_air = await _latest_bool(drone.telemetry.in_air(), timeout=probe_timeout)
    armed = await _latest_bool(drone.telemetry.armed(), timeout=probe_timeout)

    if in_air:
        with suppress(Exception):
            await asyncio.wait_for(drone.action.land(), timeout=land_timeout)
        with suppress(Exception):
            await wait_for_async_value(drone.telemetry.in_air(), lambda value: not bool(value), timeout=timeout)

    if armed or in_air:
        with suppress(Exception):
            await asyncio.wait_for(drone.action.disarm(), timeout=land_timeout)


async def estimate_pose(drone: System) -> Pose | None:
    try:
        position = await wait_for_async_value(drone.telemetry.position(), lambda _: True, timeout=5.0)
    except Exception:
        return None

    metadata: dict[str, Any] = {}
    orientation = (0.0, 0.0, 0.0, 1.0)
    with suppress(Exception):
        attitude = await wait_for_async_value(drone.telemetry.attitude_euler(), lambda _: True, timeout=2.0)
        metadata["attitude_euler_deg"] = {
            "roll_deg": float(attitude.roll_deg),
            "pitch_deg": float(attitude.pitch_deg),
            "yaw_deg": float(attitude.yaw_deg),
        }
    with suppress(Exception):
        local_state = await wait_for_async_value(
            drone.telemetry.position_velocity_ned(),
            lambda _: True,
            timeout=2.0,
        )
        metadata["local_position_ned_m"] = {
            "north_m": float(local_state.position.north_m),
            "east_m": float(local_state.position.east_m),
            "down_m": float(local_state.position.down_m),
        }
        metadata["local_velocity_ned_m_s"] = {
            "north_m_s": float(local_state.velocity.north_m_s),
            "east_m_s": float(local_state.velocity.east_m_s),
            "down_m_s": float(local_state.velocity.down_m_s),
        }
    metadata["relative_altitude_m"] = float(position.relative_altitude_m)

    return Pose(
        position_xyz=(
            float(position.latitude_deg),
            float(position.longitude_deg),
            float(position.absolute_altitude_m),
        ),
        orientation_xyzw=orientation,
        metadata=metadata,
    )


def build_teacher(args: argparse.Namespace) -> ReactiveDepthWaypointTeacher:
    return ReactiveDepthWaypointTeacher(
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
            decision_rich_threshold=args.teacher_min_decision_score,
            target_side_preference_margin_m=args.teacher_target_side_preference_margin_m,
        )
    )


def parse_target_families(raw: str) -> list[str]:
    allowed = {"forward", "left", "right", "ascend", "descend"}
    values = [token.strip().lower() for token in raw.split(",") if token.strip()]
    parsed = [value for value in values if value in allowed]
    return parsed or ["forward", "left", "right"]


def resolve_target_family(
    *,
    fixed_target_family: str,
    target_families: list[str],
    step_index: int,
) -> str | None:
    if fixed_target_family.strip():
        return fixed_target_family.strip().lower()
    return target_families[step_index % len(target_families)] if target_families else None


def select_reactive_teacher_goal(
    *,
    args: argparse.Namespace,
    teacher: ReactiveDepthWaypointTeacher,
    home_pose: Pose,
    current_pose: Pose,
    depth_frame: np.ndarray,
    teacher_rng: random.Random,
    target_family: str | None,
) -> tuple[Any, Any, dict[str, Any]]:
    best_goal = None
    best_decision = None
    best_profile = None
    best_score = float("-inf")

    attempts = max(int(args.teacher_goal_resample_attempts), 1)
    actual_attempts = 0
    for attempt_index in range(attempts):
        actual_attempts = attempt_index + 1
        candidate_goal = teacher.sample_goal(
            home_pose=home_pose,
            current_pose=current_pose,
            rng=teacher_rng,
            depth_image_m=depth_frame,
            mode=args.teacher_goal_mode,
            target_family=target_family,
        )
        candidate_decision = teacher.choose_action(
            home_pose=home_pose,
            current_pose=current_pose,
            goal=candidate_goal,
            depth_image_m=depth_frame,
        )
        candidate_profile = dict(candidate_decision.diagnostics.get("decision_profile", {}))
        candidate_score = float(candidate_profile.get("branch_score", 0.0))
        if candidate_score > best_score:
            best_goal = candidate_goal
            best_decision = candidate_decision
            best_profile = candidate_profile
            best_score = candidate_score
        if args.teacher_goal_mode != "decision_game":
            break
        if bool(candidate_profile.get("decision_rich", False)):
            best_goal = candidate_goal
            best_decision = candidate_decision
            best_profile = candidate_profile
            best_score = candidate_score
            break

    return best_goal, best_decision, {
        "attempt_count": actual_attempts,
        "selected_branch_score": best_score if best_score > float("-inf") else None,
        "selected_profile": best_profile or {},
        "target_family": target_family,
    }


async def run_episode(args: argparse.Namespace) -> Path:
    actions = parse_action_sequence(args.actions) if args.policy == "scripted" else []
    episode_id = args.episode_id or datetime.now(timezone.utc).strftime("episode_%Y%m%dT%H%M%SZ")
    episode_dir = Path(args.episodes_root).expanduser().resolve() / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    bridge_process: subprocess.Popen[str] | None = None
    image_buffer: LatestImageBuffer | None = None
    depth_buffer: LatestDepthBuffer | None = None
    stop_event: threading.Event | None = None
    spin_thread: threading.Thread | None = None
    drone = System()
    zero_setpoint = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    connected = False
    armed = False
    airborne = False
    offboard_started = False
    teacher = build_teacher(args) if args.policy in {"reactive_teacher", "reactive_obstacle_teacher"} else None
    teacher_rng = random.Random(args.teacher_seed or abs(hash(episode_id)) % (2**31))
    home_pose: Pose | None = None
    current_goal = None
    executed_actions: list[str] = []
    decision_family_counts: dict[str, int] = {}
    teacher_reason_counts: dict[str, int] = {}
    decision_rich_step_count = 0
    branch_scores: list[float] = []
    target_families = parse_target_families(args.teacher_target_families)
    action_cfg = build_action_cfg(
        duration_s=args.command_duration_s,
        forward_m_s=args.action_forward_m_s,
        strafe_m_s=args.action_strafe_m_s,
        vertical_m_s=args.action_vertical_m_s,
        yawspeed_deg_s=args.action_yawspeed_deg_s,
    )

    episode_summary = {
        "episode_id": episode_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gazebo_topic": args.gazebo_topic,
        "depth_topic": args.depth_topic,
        "world_label": args.world_label,
        "scenario_label": args.scenario_label,
        "connection": args.connection,
        "policy": args.policy,
        "teacher_goal_mode": ("obstacle_reflex" if args.policy == "reactive_obstacle_teacher" else args.teacher_goal_mode),
        "teacher_target_families": target_families,
        "teacher_fixed_target_family": args.teacher_fixed_target_family.strip().lower() or None,
        "actions": [action.value for action in actions],
        "status": "starting",
        "step_count": 0,
    }

    try:
        bridge_specs = [f"{args.gazebo_topic}@sensor_msgs/msg/Image@gz.msgs.Image"]
        if args.depth_topic.strip():
            bridge_specs.append(f"{args.depth_topic}@sensor_msgs/msg/Image@gz.msgs.Image")
        bridge_process = start_bridge(bridge_specs)
        await asyncio.sleep(args.bridge_wait_s)

        rclpy.init()
        image_buffer = LatestImageBuffer(args.gazebo_topic)
        nodes: list[Node] = [image_buffer]
        if args.depth_topic.strip():
            depth_buffer = LatestDepthBuffer(args.depth_topic)
            nodes.append(depth_buffer)
        stop_event, spin_thread = start_ros_spin(nodes)

        await asyncio.wait_for(drone.connect(system_address=args.connection), timeout=args.timeout)
        await wait_connected(drone, timeout=args.timeout)
        connected = True
        await wait_for_frame(image_buffer, min_frame_count=0, timeout=args.timeout)
        if depth_buffer is not None:
            await wait_for_depth_frame(depth_buffer, min_frame_count=0, timeout=args.timeout)

        await drone.action.set_takeoff_altitude(args.takeoff_altitude_m)
        await safe_arm(drone, timeout=min(args.timeout, 20.0))
        armed = True
        await drone.action.takeoff()
        await wait_for_async_value(drone.telemetry.in_air(), lambda value: bool(value), timeout=args.timeout)
        airborne = True
        with suppress(Exception):
            await wait_for_offboard_ready_altitude(
                drone,
                min_relative_altitude_m=args.min_offboard_ready_altitude_m,
                timeout=min(args.timeout, 20.0),
            )
        await asyncio.sleep(2.0)

        await drone.offboard.set_velocity_body(zero_setpoint)
        await drone.offboard.start()
        offboard_started = True

        last_frame_count = 0
        last_depth_frame_count = 0
        total_steps = len(actions) if args.policy == "scripted" else int(args.num_steps)
        for step_index in range(total_steps):
            step_dir = episode_dir / f"step_{step_index:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            frame_count, frame, timestamp_s, intrinsics = await wait_for_frame(
                image_buffer,
                min_frame_count=last_frame_count,
                timeout=args.timeout,
            )
            last_frame_count = frame_count
            before_path = step_dir / "frame_before.png"
            frame.save(before_path)

            pose = await estimate_pose(drone)
            if pose is None:
                raise RuntimeError("Failed to estimate pose before choosing an action.")
            if home_pose is None:
                home_pose = pose

            teacher_metadata: dict[str, Any] = {}
            if args.policy in {"reactive_teacher", "reactive_obstacle_teacher"}:
                if teacher is None or depth_buffer is None:
                    raise RuntimeError("Reactive teacher mode is missing its teacher or depth buffer.")

                depth_count, depth_frame, depth_timestamp_s, depth_encoding = await wait_for_depth_frame(
                    depth_buffer,
                    min_frame_count=last_depth_frame_count,
                    timeout=args.timeout,
                )
                last_depth_frame_count = depth_count

                if args.policy == "reactive_obstacle_teacher":
                    target_family = resolve_target_family(
                        fixed_target_family=args.teacher_fixed_target_family,
                        target_families=target_families,
                        step_index=step_index,
                    )
                    goal_selection_meta = {
                        "attempt_count": 0,
                        "selected_branch_score": None,
                        "selected_profile": {},
                        "target_family": target_family,
                    }
                    decision = teacher.choose_obstacle_avoidance_action(
                        depth_image_m=depth_frame,
                        target_family=target_family,
                    )
                else:
                    current_local = latlon_alt_to_local_m(home_pose, pose)
                    refresh_goal = bool(args.teacher_goal_mode == "decision_game" and args.teacher_refresh_goal_every_step)
                    if current_goal is None or refresh_goal or teacher.goal_reached(
                        current_local_xyz_m=current_local,
                        goal=current_goal,
                    ):
                        target_family = resolve_target_family(
                            fixed_target_family=args.teacher_fixed_target_family,
                            target_families=target_families,
                            step_index=step_index,
                        )
                        current_goal, decision, goal_selection_meta = select_reactive_teacher_goal(
                            args=args,
                            teacher=teacher,
                            home_pose=home_pose,
                            current_pose=pose,
                            depth_frame=depth_frame,
                            teacher_rng=teacher_rng,
                            target_family=target_family,
                        )
                    else:
                        goal_selection_meta = {
                            "attempt_count": 0,
                            "selected_branch_score": None,
                            "selected_profile": {},
                            "target_family": None,
                        }
                        decision = teacher.choose_action(
                            home_pose=home_pose,
                            current_pose=pose,
                            goal=current_goal,
                            depth_image_m=depth_frame,
                        )

                if args.policy == "reactive_teacher" and current_goal is None:
                    raise RuntimeError("Reactive teacher failed to produce a goal.")
                if decision is None:
                    raise RuntimeError("Reactive teacher failed to choose an action.")

                action = decision.action
                goal_features = dict(decision.diagnostics.get("goal_features", {}))
                decision_profile = dict(decision.diagnostics.get("decision_profile", {}))
                family = str(decision_profile.get("family", "unknown"))
                target_family = str(
                    decision_profile.get("target_family")
                    or goal_selection_meta.get("target_family")
                    or ""
                ).strip().lower()
                branch_score = float(decision_profile.get("branch_score", 0.0) or 0.0)
                if (
                    args.policy == "reactive_obstacle_teacher"
                    and step_index == 0
                ):
                    if args.teacher_require_initially_blocked and family != "obstacle_avoidance":
                        raise RuntimeError(
                            "Reactive obstacle teacher episode is not initially blocked enough: "
                            f"family={family}"
                        )
                    if args.teacher_min_initial_branch_score > 0.0 and branch_score < args.teacher_min_initial_branch_score:
                        raise RuntimeError(
                            "Reactive obstacle teacher episode initial branch score is too low: "
                            f"branch_score={branch_score:.3f} required>={args.teacher_min_initial_branch_score:.3f}"
                        )
                    if (
                        args.teacher_require_initial_target_action
                        and not initial_action_matches_target_family(action, target_family)
                    ):
                        raise RuntimeError(
                            "Reactive obstacle teacher episode initial action does not match target family: "
                            f"target_family={target_family or 'unknown'} action={action.value}"
                        )
                decision_family_counts[family] = decision_family_counts.get(family, 0) + 1
                teacher_reason_counts[decision.reason] = teacher_reason_counts.get(decision.reason, 0) + 1
                if bool(decision_profile.get("decision_rich", False)):
                    decision_rich_step_count += 1
                if decision_profile.get("branch_score") is not None:
                    branch_scores.append(float(decision_profile["branch_score"]))
                teacher_metadata = {
                    "teacher": {
                        "policy": (
                            "reactive_obstacle_reflex_v1"
                            if args.policy == "reactive_obstacle_teacher"
                            else "reactive_depth_waypoint_v1"
                        ),
                        "reason": decision.reason,
                        "goal_features": goal_features,
                        "goal_feature_vector": normalize_goal_features(
                            goal_features.get("forward_m", 0.0),
                            goal_features.get("right_m", 0.0),
                            goal_features.get("alt_error_m", 0.0),
                            goal_features.get("heading_error_deg", 0.0),
                        ),
                        "goal_mode": ("obstacle_reflex" if args.policy == "reactive_obstacle_teacher" else args.teacher_goal_mode),
                        "goal_selection": goal_selection_meta,
                        **decision.diagnostics,
                    },
                    "depth_topic": args.depth_topic,
                    "depth_timestamp_s": depth_timestamp_s,
                    "depth_encoding": depth_encoding,
                }
            else:
                action = actions[step_index]

            executed_actions.append(action.value)
            command = discrete_to_command(action, action_cfg)
            command_setpoint = VelocityBodyYawspeed(
                command.forward_m_s,
                command.right_m_s,
                command.down_m_s,
                command.yawspeed_deg_s,
            )

            await drone.offboard.set_velocity_body(command_setpoint)
            await asyncio.sleep(command.duration_s)
            await drone.offboard.set_velocity_body(zero_setpoint)
            await asyncio.sleep(args.settle_duration_s)

            frame_count, next_frame, next_timestamp_s, _ = await wait_for_frame(
                image_buffer,
                min_frame_count=last_frame_count,
                timeout=args.timeout,
            )
            last_frame_count = frame_count
            after_path = step_dir / "frame_after.png"
            next_frame.save(after_path)

            step = BCEpisodeStep(
                episode_id=episode_id,
                step_index=step_index,
                action=action,
                command=command,
                image_path=str(before_path.relative_to(episode_dir)),
                next_image_path=str(after_path.relative_to(episode_dir)),
                timestamp_s=timestamp_s,
                pose=pose,
                intrinsics=intrinsics,
                metadata={
                    "frame_after_timestamp_s": next_timestamp_s,
                    "gazebo_topic": args.gazebo_topic,
                    "world_label": args.world_label,
                    "scenario_label": args.scenario_label,
                    **teacher_metadata,
                },
            )
            save_episode_step(step_dir, step)

        episode_summary["status"] = "completed"
        episode_summary["actions"] = executed_actions
        episode_summary["step_count"] = total_steps
        episode_summary["decision_family_counts"] = decision_family_counts
        episode_summary["teacher_reason_counts"] = teacher_reason_counts
        episode_summary["decision_rich_step_count"] = decision_rich_step_count
        episode_summary["mean_branch_score"] = (
            sum(branch_scores) / len(branch_scores) if branch_scores else None
        )
        return episode_dir
    except Exception as exc:
        episode_summary["status"] = "failed"
        episode_summary["error"] = str(exc)
        raise
    finally:
        (episode_dir / "episode.json").write_text(json.dumps(episode_summary, indent=2), encoding="utf-8")
        if connected and offboard_started:
            with suppress(Exception):
                await drone.offboard.set_velocity_body(zero_setpoint)
            with suppress(OffboardError, Exception):
                await drone.offboard.stop()
        if connected and (armed or airborne):
            with suppress(Exception):
                await safe_land(drone, timeout=args.timeout)
        if stop_event is not None:
            stop_event.set()
        if spin_thread is not None:
            spin_thread.join(timeout=5.0)
        if image_buffer is not None:
            image_buffer.destroy_node()
        if depth_buffer is not None:
            depth_buffer.destroy_node()
        with suppress(Exception):
            if rclpy.ok():
                rclpy.shutdown()
        stop_bridge(bridge_process)


async def main_async() -> int:
    args = parse_args()
    validate_args(args)
    episode_dir = await run_episode(args)
    print(json.dumps({"episode_dir": str(episode_dir), "status": "completed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
