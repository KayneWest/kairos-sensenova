from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any

import numpy as np

from sensenova_drone.actions import DiscreteDroneAction
from sensenova_drone.observation import Pose


EARTH_RADIUS_M = 6_378_137.0


@dataclass
class LocalWaypointGoal:
    local_position_xyz_m: tuple[float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TeacherDecision:
    action: DiscreteDroneAction
    reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionPointProfile:
    family: str
    branch_score: float
    decision_rich: bool
    target_family: str | None = None
    supporting_clearance_m: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReactiveDepthTeacherConfig:
    min_goal_forward_m: float = 3.0
    max_goal_forward_m: float = 8.0
    max_goal_lateral_m: float = 4.0
    max_goal_vertical_m: float = 0.0
    goal_reached_radius_m: float = 1.0
    altitude_tolerance_m: float = 0.35
    heading_error_threshold_deg: float = 15.0
    front_blocked_threshold_m: float = 2.5
    side_clearance_threshold_m: float = 2.0
    front_preferred_threshold_m: float = 4.0
    decision_rich_threshold: float = 0.55
    target_side_preference_margin_m: float = 0.4


def yaw_deg_from_pose(pose: Pose | None) -> float | None:
    if pose is None:
        return None
    attitude = pose.metadata.get("attitude_euler_deg", {})
    yaw_deg = attitude.get("yaw_deg")
    if yaw_deg is None:
        return None
    return float(yaw_deg)


def latlon_alt_to_local_m(home_pose: Pose, pose: Pose) -> tuple[float, float, float]:
    home_lat_deg, home_lon_deg, home_alt_m = home_pose.position_xyz
    lat_deg, lon_deg, alt_m = pose.position_xyz

    mean_lat_rad = math.radians((home_lat_deg + lat_deg) * 0.5)
    north_m = math.radians(lat_deg - home_lat_deg) * EARTH_RADIUS_M
    east_m = math.radians(lon_deg - home_lon_deg) * EARTH_RADIUS_M * math.cos(mean_lat_rad)
    up_m = alt_m - home_alt_m
    return (east_m, north_m, up_m)


def world_delta_to_body_m(delta_east_m: float, delta_north_m: float, yaw_deg: float) -> tuple[float, float]:
    yaw_rad = math.radians(yaw_deg)
    forward_m = delta_east_m * math.sin(yaw_rad) + delta_north_m * math.cos(yaw_rad)
    right_m = delta_east_m * math.cos(yaw_rad) - delta_north_m * math.sin(yaw_rad)
    return forward_m, right_m


def normalize_goal_features(
    forward_m: float,
    right_m: float,
    alt_error_m: float,
    heading_error_deg: float,
) -> list[float]:
    return [
        float(np.clip(forward_m / 10.0, -2.0, 2.0)),
        float(np.clip(right_m / 5.0, -2.0, 2.0)),
        float(np.clip(alt_error_m / 3.0, -2.0, 2.0)),
        float(np.clip(heading_error_deg / 180.0, -1.0, 1.0)),
    ]


class ReactiveDepthWaypointTeacher:
    def __init__(self, cfg: ReactiveDepthTeacherConfig | None = None):
        self.cfg = cfg or ReactiveDepthTeacherConfig()

    def sample_goal(
        self,
        *,
        home_pose: Pose,
        current_pose: Pose,
        rng: random.Random,
        depth_image_m: np.ndarray | None = None,
        mode: str = "random",
        target_family: str | None = None,
    ) -> LocalWaypointGoal:
        if mode == "decision_game":
            return self._sample_decision_game_goal(
                home_pose=home_pose,
                current_pose=current_pose,
                rng=rng,
                depth_image_m=depth_image_m,
                target_family=target_family,
            )
        return self._sample_random_goal(
            home_pose=home_pose,
            current_pose=current_pose,
            rng=rng,
        )

    def _sample_random_goal(
        self,
        *,
        home_pose: Pose,
        current_pose: Pose,
        rng: random.Random,
    ) -> LocalWaypointGoal:
        current_local = latlon_alt_to_local_m(home_pose, current_pose)
        yaw_deg = yaw_deg_from_pose(current_pose) or 0.0

        forward_m = rng.uniform(self.cfg.min_goal_forward_m, self.cfg.max_goal_forward_m)
        right_m = rng.uniform(-self.cfg.max_goal_lateral_m, self.cfg.max_goal_lateral_m)
        up_m = rng.uniform(-self.cfg.max_goal_vertical_m, self.cfg.max_goal_vertical_m)

        yaw_rad = math.radians(yaw_deg)
        delta_east_m = forward_m * math.sin(yaw_rad) + right_m * math.cos(yaw_rad)
        delta_north_m = forward_m * math.cos(yaw_rad) - right_m * math.sin(yaw_rad)

        return LocalWaypointGoal(
            local_position_xyz_m=(
                current_local[0] + delta_east_m,
                current_local[1] + delta_north_m,
                current_local[2] + up_m,
            ),
            metadata={
                "sampled_body_offset_m": {
                    "forward_m": forward_m,
                    "right_m": right_m,
                    "up_m": up_m,
                },
                "yaw_deg_at_sample": yaw_deg,
                "goal_mode": "random",
            },
        )

    def _sample_decision_game_goal(
        self,
        *,
        home_pose: Pose,
        current_pose: Pose,
        rng: random.Random,
        depth_image_m: np.ndarray | None,
        target_family: str | None,
    ) -> LocalWaypointGoal:
        current_local = latlon_alt_to_local_m(home_pose, current_pose)
        yaw_deg = yaw_deg_from_pose(current_pose) or 0.0
        clearances = compute_depth_clearances(depth_image_m)

        family = target_family or self._choose_decision_game_family(clearances, rng)
        lateral_peak = max(1.5, self.cfg.max_goal_lateral_m)
        forward_low = max(2.0, min(self.cfg.min_goal_forward_m, self.cfg.max_goal_forward_m))
        forward_mid = max(forward_low, min(self.cfg.max_goal_forward_m, max(self.cfg.min_goal_forward_m, 4.0)))

        if family == "left":
            forward_m = rng.uniform(forward_low, forward_mid)
            right_m = -rng.uniform(1.5, lateral_peak)
            up_m = 0.0
        elif family == "right":
            forward_m = rng.uniform(forward_low, forward_mid)
            right_m = rng.uniform(1.5, lateral_peak)
            up_m = 0.0
        elif family == "ascend" and self.cfg.max_goal_vertical_m > 0.0:
            forward_m = rng.uniform(forward_low, forward_mid)
            right_m = rng.uniform(-0.75, 0.75)
            up_m = rng.uniform(max(0.4, self.cfg.altitude_tolerance_m * 1.5), self.cfg.max_goal_vertical_m)
        elif family == "descend" and self.cfg.max_goal_vertical_m > 0.0:
            forward_m = rng.uniform(forward_low, forward_mid)
            right_m = rng.uniform(-0.75, 0.75)
            up_m = -rng.uniform(max(0.4, self.cfg.altitude_tolerance_m * 1.5), self.cfg.max_goal_vertical_m)
        else:
            family = "forward"
            forward_m = rng.uniform(max(forward_low, self.cfg.min_goal_forward_m), self.cfg.max_goal_forward_m)
            right_m = rng.uniform(-0.75, 0.75)
            up_m = 0.0

        yaw_rad = math.radians(yaw_deg)
        delta_east_m = forward_m * math.sin(yaw_rad) + right_m * math.cos(yaw_rad)
        delta_north_m = forward_m * math.cos(yaw_rad) - right_m * math.sin(yaw_rad)

        return LocalWaypointGoal(
            local_position_xyz_m=(
                current_local[0] + delta_east_m,
                current_local[1] + delta_north_m,
                current_local[2] + up_m,
            ),
            metadata={
                "sampled_body_offset_m": {
                    "forward_m": forward_m,
                    "right_m": right_m,
                    "up_m": up_m,
                },
                "yaw_deg_at_sample": yaw_deg,
                "goal_mode": "decision_game",
                "target_family": family,
                "depth_clearances_at_sample_m": clearances,
            },
        )

    def _choose_decision_game_family(
        self,
        clearances: dict[str, float | None],
        rng: random.Random,
    ) -> str:
        front_m = clearances.get("front_m")
        left_m = clearances.get("left_m")
        right_m = clearances.get("right_m")
        up_m = clearances.get("up_m")

        left_open = left_m is None or left_m >= self.cfg.side_clearance_threshold_m
        right_open = right_m is None or right_m >= self.cfg.side_clearance_threshold_m
        front_blocked = front_m is not None and front_m < self.cfg.front_blocked_threshold_m

        weighted: list[str] = []
        if front_blocked:
            if left_open:
                weighted.extend(["left"] * (4 if (left_m or 0.0) >= (right_m or 0.0) else 2))
            if right_open:
                weighted.extend(["right"] * (4 if (right_m or 0.0) > (left_m or 0.0) else 2))
            if self.cfg.max_goal_vertical_m > 0.0 and (up_m is None or up_m >= self.cfg.front_preferred_threshold_m):
                weighted.extend(["ascend"] * 2)
        else:
            weighted.extend(["forward"] * 4)
            if left_open:
                weighted.extend(["left"] * 2)
            if right_open:
                weighted.extend(["right"] * 2)
            if self.cfg.max_goal_vertical_m > 0.0 and (up_m is None or up_m >= self.cfg.front_preferred_threshold_m):
                weighted.append("ascend")

        if not weighted:
            weighted = ["forward", "left", "right"]
        return rng.choice(weighted)

    def goal_reached(
        self,
        *,
        current_local_xyz_m: tuple[float, float, float],
        goal: LocalWaypointGoal,
    ) -> bool:
        dx = goal.local_position_xyz_m[0] - current_local_xyz_m[0]
        dy = goal.local_position_xyz_m[1] - current_local_xyz_m[1]
        dz = goal.local_position_xyz_m[2] - current_local_xyz_m[2]
        distance_xy_m = math.hypot(dx, dy)
        return (
            distance_xy_m <= self.cfg.goal_reached_radius_m
            and abs(dz) <= self.cfg.altitude_tolerance_m
        )

    def choose_action(
        self,
        *,
        home_pose: Pose,
        current_pose: Pose,
        goal: LocalWaypointGoal,
        depth_image_m: np.ndarray | None,
    ) -> TeacherDecision:
        yaw_deg = yaw_deg_from_pose(current_pose) or 0.0
        current_local = latlon_alt_to_local_m(home_pose, current_pose)

        delta_east_m = goal.local_position_xyz_m[0] - current_local[0]
        delta_north_m = goal.local_position_xyz_m[1] - current_local[1]
        delta_up_m = goal.local_position_xyz_m[2] - current_local[2]
        goal_forward_m, goal_right_m = world_delta_to_body_m(delta_east_m, delta_north_m, yaw_deg)
        goal_heading_error_deg = math.degrees(math.atan2(goal_right_m, max(goal_forward_m, 1e-6)))

        clearances = compute_depth_clearances(depth_image_m)
        front_m = clearances["front_m"]
        left_m = clearances["left_m"]
        right_m = clearances["right_m"]
        up_m = clearances["up_m"]

        diagnostics = {
            "goal_features": {
                "forward_m": goal_forward_m,
                "right_m": goal_right_m,
                "alt_error_m": delta_up_m,
                "heading_error_deg": goal_heading_error_deg,
            },
            "goal_local_xyz_m": list(goal.local_position_xyz_m),
            "current_local_xyz_m": list(current_local),
            "depth_clearance_m": clearances,
            "goal_metadata": goal.metadata,
        }

        front_blocked = front_m is not None and front_m < self.cfg.front_blocked_threshold_m
        if front_blocked:
            turn_left_by_goal = goal_right_m < 0.0
            depth_bias = (left_m or 0.0) - (right_m or 0.0)
            clearer_left = (left_m or 0.0) >= (right_m or 0.0)
            if abs(depth_bias) < 0.25 and abs(goal_right_m) > 0.75:
                clearer_left = turn_left_by_goal
            side_clear = max(left_m or 0.0, right_m or 0.0) >= self.cfg.side_clearance_threshold_m
            if side_clear:
                return self._finalize_decision(
                    action=DiscreteDroneAction.STRAFE_LEFT if clearer_left else DiscreteDroneAction.STRAFE_RIGHT,
                    reason="front_blocked_strafe",
                    diagnostics=diagnostics,
                )
            if up_m is not None and up_m > self.cfg.front_preferred_threshold_m and delta_up_m > self.cfg.altitude_tolerance_m:
                return self._finalize_decision(
                    action=DiscreteDroneAction.ASCEND,
                    reason="front_blocked_ascend",
                    diagnostics=diagnostics,
                )
            return self._finalize_decision(
                action=DiscreteDroneAction.YAW_LEFT if clearer_left else DiscreteDroneAction.YAW_RIGHT,
                reason="front_blocked_yaw",
                diagnostics=diagnostics,
            )

        if abs(delta_up_m) > self.cfg.altitude_tolerance_m:
            return self._finalize_decision(
                action=DiscreteDroneAction.ASCEND if delta_up_m > 0.0 else DiscreteDroneAction.DESCEND,
                reason="altitude_correction",
                diagnostics=diagnostics,
            )

        if abs(goal_heading_error_deg) > self.cfg.heading_error_threshold_deg:
            return self._finalize_decision(
                action=DiscreteDroneAction.YAW_RIGHT if goal_heading_error_deg > 0.0 else DiscreteDroneAction.YAW_LEFT,
                reason="heading_correction",
                diagnostics=diagnostics,
            )

        if goal_forward_m < -1.0:
            return self._finalize_decision(
                action=DiscreteDroneAction.BACKWARD,
                reason="goal_behind",
                diagnostics=diagnostics,
            )

        return self._finalize_decision(
            action=DiscreteDroneAction.FORWARD,
            reason="goal_progress",
            diagnostics=diagnostics,
        )

    def choose_obstacle_avoidance_action(
        self,
        *,
        depth_image_m: np.ndarray | None,
        target_family: str | None = None,
    ) -> TeacherDecision:
        clearances = compute_depth_clearances(depth_image_m)
        front_m = clearances["front_m"]
        left_m = clearances["left_m"]
        right_m = clearances["right_m"]
        up_m = clearances["up_m"]

        diagnostics = {
            "goal_features": {
                "forward_m": 0.0,
                "right_m": 0.0,
                "alt_error_m": 0.0,
                "heading_error_deg": 0.0,
            },
            "depth_clearance_m": clearances,
            "goal_metadata": {
                "goal_mode": "obstacle_reflex",
                "target_family": target_family,
            },
        }

        front_blocked = front_m is not None and front_m < self.cfg.front_blocked_threshold_m
        if front_blocked:
            clearer_left = (left_m or 0.0) >= (right_m or 0.0)
            side_clear = max(left_m or 0.0, right_m or 0.0) >= self.cfg.side_clearance_threshold_m
            if side_clear:
                preferred_left = self._preferred_lateral_side(
                    target_family=target_family,
                    left_m=left_m,
                    right_m=right_m,
                )
                if preferred_left is not None:
                    clearer_left = preferred_left
                return self._finalize_decision(
                    action=DiscreteDroneAction.STRAFE_LEFT if clearer_left else DiscreteDroneAction.STRAFE_RIGHT,
                    reason="front_blocked_strafe",
                    diagnostics=diagnostics,
                )
            if up_m is not None and up_m > self.cfg.front_preferred_threshold_m:
                return self._finalize_decision(
                    action=DiscreteDroneAction.ASCEND,
                    reason="front_blocked_ascend",
                    diagnostics=diagnostics,
                )
            return self._finalize_decision(
                action=DiscreteDroneAction.YAW_LEFT if clearer_left else DiscreteDroneAction.YAW_RIGHT,
                reason="front_blocked_yaw",
                diagnostics=diagnostics,
            )

        return self._finalize_decision(
            action=DiscreteDroneAction.FORWARD,
            reason="front_clear_forward",
            diagnostics=diagnostics,
        )

    def _preferred_lateral_side(
        self,
        *,
        target_family: str | None,
        left_m: float | None,
        right_m: float | None,
    ) -> bool | None:
        preferred = (target_family or "").strip().lower()
        if preferred not in {"left", "right"}:
            return None

        left_clear = left_m is None or left_m >= self.cfg.side_clearance_threshold_m
        right_clear = right_m is None or right_m >= self.cfg.side_clearance_threshold_m
        if preferred == "left":
            if not left_clear:
                return None
            if right_m is not None and left_m is not None:
                if left_m + self.cfg.target_side_preference_margin_m < right_m:
                    return None
            return True

        if not right_clear:
            return None
        if left_m is not None and right_m is not None:
            if right_m + self.cfg.target_side_preference_margin_m < left_m:
                return None
        return False

    def _finalize_decision(
        self,
        *,
        action: DiscreteDroneAction,
        reason: str,
        diagnostics: dict[str, Any],
    ) -> TeacherDecision:
        profile = self.profile_decision(
            action=action,
            reason=reason,
            diagnostics=diagnostics,
        )
        payload = dict(diagnostics)
        payload["decision_profile"] = {
            "family": profile.family,
            "branch_score": profile.branch_score,
            "decision_rich": profile.decision_rich,
            "target_family": profile.target_family,
            "supporting_clearance_m": profile.supporting_clearance_m,
            **profile.metadata,
        }
        return TeacherDecision(action=action, reason=reason, diagnostics=payload)

    def profile_decision(
        self,
        *,
        action: DiscreteDroneAction,
        reason: str,
        diagnostics: dict[str, Any],
    ) -> DecisionPointProfile:
        goal_features = dict(diagnostics.get("goal_features", {}))
        clearances = dict(diagnostics.get("depth_clearance_m", {}))
        goal_metadata = dict(diagnostics.get("goal_metadata", {}))

        forward_m = float(goal_features.get("forward_m", 0.0))
        right_m = float(goal_features.get("right_m", 0.0))
        alt_error_m = float(goal_features.get("alt_error_m", 0.0))
        heading_error_deg = abs(float(goal_features.get("heading_error_deg", 0.0)))

        front_m = _maybe_float(clearances.get("front_m"))
        left_m = _maybe_float(clearances.get("left_m"))
        right_clear_m = _maybe_float(clearances.get("right_m"))
        up_m = _maybe_float(clearances.get("up_m"))

        family = "goal_progress"
        support_m = front_m
        if reason.startswith("front_blocked"):
            family = "obstacle_avoidance"
            support_m = max(value for value in [left_m, right_clear_m, up_m] if value is not None) if any(
                value is not None for value in [left_m, right_clear_m, up_m]
            ) else None
            blocked_score = _clamp01((self.cfg.front_blocked_threshold_m - (front_m or 0.0)) / max(self.cfg.front_blocked_threshold_m, 1e-6))
            side_support = _clamp01(max((left_m or 0.0), (right_clear_m or 0.0), (up_m or 0.0)) / max(self.cfg.front_preferred_threshold_m, 1.0))
            lateral_pull = _clamp01(abs(right_m) / max(self.cfg.max_goal_lateral_m, 1.5))
            branch_score = 0.45 * blocked_score + 0.30 * side_support + 0.25 * lateral_pull
        elif reason == "heading_correction":
            family = "goal_turn"
            branch_score = 0.55 * _clamp01(heading_error_deg / max(self.cfg.heading_error_threshold_deg * 2.0, 1.0))
            branch_score += 0.45 * _clamp01(abs(right_m) / max(self.cfg.max_goal_lateral_m, 1.5))
        elif reason == "altitude_correction":
            family = "altitude"
            support_m = up_m
            branch_score = 0.65 * _clamp01(abs(alt_error_m) / max(self.cfg.altitude_tolerance_m * 3.0, 0.5))
            branch_score += 0.20 * _clamp01(abs(forward_m) / max(self.cfg.max_goal_forward_m, 1.0))
            branch_score += 0.15 * _clamp01((up_m or self.cfg.front_preferred_threshold_m) / max(self.cfg.front_preferred_threshold_m, 1.0))
        elif reason == "goal_behind":
            family = "reverse"
            branch_score = 0.5 + 0.5 * _clamp01(abs(forward_m) / max(self.cfg.max_goal_forward_m, 1.0))
        elif reason == "front_clear_forward":
            family = "obstacle_cruise"
            branch_score = 0.40 * _clamp01((front_m or self.cfg.front_preferred_threshold_m) / max(self.cfg.front_preferred_threshold_m, 1.0))
            branch_score += 0.60 * (1.0 - _clamp01(abs(right_m) / max(self.cfg.max_goal_lateral_m, 1.5)))
        else:
            family = "goal_progress"
            branch_score = 0.35 * _clamp01(abs(forward_m) / max(self.cfg.max_goal_forward_m, 1.0))
            branch_score += 0.35 * _clamp01((front_m or self.cfg.front_preferred_threshold_m) / max(self.cfg.front_preferred_threshold_m, 1.0))
            branch_score += 0.30 * (1.0 - _clamp01(heading_error_deg / 45.0))

        branch_score = _clamp01(branch_score)
        decision_rich = branch_score >= self.cfg.decision_rich_threshold
        return DecisionPointProfile(
            family=family,
            branch_score=branch_score,
            decision_rich=decision_rich,
            target_family=str(goal_metadata.get("target_family")) if goal_metadata.get("target_family") else None,
            supporting_clearance_m=support_m,
            metadata={
                "goal_mode": goal_metadata.get("goal_mode"),
                "selected_action": action.value,
            },
        )


def compute_depth_clearances(depth_image_m: np.ndarray | None) -> dict[str, float | None]:
    if depth_image_m is None or depth_image_m.size == 0:
        return {
            "front_m": None,
            "left_m": None,
            "right_m": None,
            "up_m": None,
            "down_m": None,
        }

    image = np.asarray(depth_image_m, dtype=np.float32)
    if image.ndim == 3:
        image = image[..., 0]

    height, width = image.shape
    regions = {
        "front_m": image[int(0.30 * height): int(0.70 * height), int(0.35 * width): int(0.65 * width)],
        "left_m": image[int(0.30 * height): int(0.70 * height), int(0.05 * width): int(0.35 * width)],
        "right_m": image[int(0.30 * height): int(0.70 * height), int(0.65 * width): int(0.95 * width)],
        "up_m": image[int(0.05 * height): int(0.35 * height), int(0.35 * width): int(0.65 * width)],
        "down_m": image[int(0.65 * height): int(0.95 * height), int(0.35 * width): int(0.65 * width)],
    }
    return {name: _region_clearance(region) for name, region in regions.items()}


def _region_clearance(region: np.ndarray) -> float | None:
    values = np.asarray(region, dtype=np.float32)
    valid = values[np.isfinite(values) & (values > 0.05)]
    if valid.size == 0:
        return None
    return float(np.quantile(valid, 0.25))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
