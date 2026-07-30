from __future__ import annotations

import inspect
import logging
from typing import Any, Awaitable, Callable

from sensenova_drone.observation import CameraIntrinsics, Pose


class StateEstimator:
    """
    Provides pose and camera calibration for each observation.

    In Gazebo/PX4 mode, pose may come from:
    - MAVSDK telemetry
    - ROS odometry topic
    - Gazebo pose topic
    - mocked pose fallback

    Intrinsics may come from:
    - ROS CameraInfo topic
    - Gazebo camera metadata
    - config fallback
    """

    def __init__(
        self,
        cfg: dict,
        *,
        pose_provider: Callable[[], Awaitable[Any] | Any] | None = None,
        camera_info_provider: Callable[[], Awaitable[Any] | Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.cfg = cfg
        self.pose_provider = pose_provider
        self.camera_info_provider = camera_info_provider
        self.logger = logger or logging.getLogger(__name__)

    async def estimate_pose(self) -> Pose | None:
        estimator_cfg = self.cfg.get("state_estimator", {})
        source = estimator_cfg.get("pose_source", "none")

        if source == "none":
            self.logger.warning("Pose unavailable: state_estimator.pose_source=none")
            return None

        if source == "mock":
            pose = self._pose_from_config(estimator_cfg.get("mock_pose", {}))
            if pose is None:
                self.logger.warning("Pose unavailable: mock pose source configured without values")
            return pose

        if source not in {"mavsdk", "ros_odom", "gazebo"}:
            self.logger.warning("Pose unavailable: unsupported pose source %s", source)
            return None

        if self.pose_provider is None:
            self.logger.warning("Pose unavailable: pose source %s configured but no provider is attached", source)
            return None

        raw_pose = await _resolve_provider(self.pose_provider)
        pose = _normalize_pose(raw_pose)
        if pose is None:
            self.logger.warning("Pose unavailable: provider for source %s returned %r", source, raw_pose)
        return pose

    async def get_intrinsics(self) -> CameraIntrinsics | None:
        camera_cfg = self.cfg.get("camera", {})

        if self.camera_info_provider is not None and camera_cfg.get("camera_info_topic"):
            raw_intrinsics = await _resolve_provider(self.camera_info_provider)
            intrinsics = _normalize_intrinsics(raw_intrinsics)
            if intrinsics is not None:
                return intrinsics
            self.logger.warning(
                "Camera intrinsics unavailable from camera_info_topic=%s; falling back to config",
                camera_cfg.get("camera_info_topic"),
            )

        intrinsics = self._intrinsics_from_config(camera_cfg)
        if intrinsics is None:
            self.logger.warning("Camera intrinsics unavailable: no config fallback and no live provider result")
        return intrinsics

    def _pose_from_config(self, pose_cfg: dict[str, Any]) -> Pose | None:
        if not pose_cfg:
            return None
        return _normalize_pose(pose_cfg)

    def _intrinsics_from_config(self, camera_cfg: dict[str, Any]) -> CameraIntrinsics | None:
        if not camera_cfg:
            return None

        width = camera_cfg.get("width")
        height = camera_cfg.get("height")
        if width is None or height is None:
            return None

        return CameraIntrinsics(
            width=int(width),
            height=int(height),
            fx=_maybe_float(camera_cfg.get("fx")),
            fy=_maybe_float(camera_cfg.get("fy")),
            cx=_maybe_float(camera_cfg.get("cx")),
            cy=_maybe_float(camera_cfg.get("cy")),
            metadata={"source": "config_fallback"},
        )


async def _resolve_provider(provider: Callable[[], Awaitable[Any] | Any] | Any) -> Any:
    value = provider() if callable(provider) else provider
    if inspect.isawaitable(value):
        return await value
    return value


def _normalize_pose(value: Any) -> Pose | None:
    if value is None:
        return None
    if isinstance(value, Pose):
        return value
    if isinstance(value, dict):
        return Pose.from_mapping(value)
    return None


def _normalize_intrinsics(value: Any) -> CameraIntrinsics | None:
    if value is None:
        return None
    if isinstance(value, CameraIntrinsics):
        return value
    if isinstance(value, dict):
        width = value.get("width")
        height = value.get("height")
        if width is None or height is None:
            return None
        return CameraIntrinsics.from_mapping(value)
    return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
