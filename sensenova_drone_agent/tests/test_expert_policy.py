from __future__ import annotations

import numpy as np
import random

from sensenova_drone.actions import DiscreteDroneAction
from sensenova_drone.expert_policy import (
    LocalWaypointGoal,
    ReactiveDepthWaypointTeacher,
    compute_depth_clearances,
    latlon_alt_to_local_m,
)
from sensenova_drone.observation import Pose


def make_pose(*, lat: float = 47.0, lon: float = 8.0, alt: float = 10.0, yaw_deg: float = 0.0) -> Pose:
    return Pose(
        position_xyz=(lat, lon, alt),
        metadata={
            "attitude_euler_deg": {
                "yaw_deg": yaw_deg,
            }
        },
    )


def test_compute_depth_clearances_prefers_opener_side():
    depth = np.full((100, 100), 6.0, dtype=np.float32)
    depth[30:70, 35:65] = 1.0
    depth[30:70, 5:35] = 5.5
    depth[30:70, 65:95] = 2.0
    clearances = compute_depth_clearances(depth)
    assert clearances["front_m"] is not None
    assert clearances["left_m"] > clearances["right_m"]


def test_reactive_teacher_strafes_when_front_is_blocked():
    teacher = ReactiveDepthWaypointTeacher()
    home_pose = make_pose()
    current_pose = make_pose(yaw_deg=0.0)
    goal = LocalWaypointGoal(local_position_xyz_m=(0.0, 6.0, 0.0))

    depth = np.full((120, 160), 6.0, dtype=np.float32)
    depth[36:84, 56:104] = 1.2
    depth[36:84, 8:56] = 5.0
    depth[36:84, 104:152] = 1.5

    decision = teacher.choose_action(
        home_pose=home_pose,
        current_pose=current_pose,
        goal=goal,
        depth_image_m=depth,
    )
    assert decision.action == DiscreteDroneAction.STRAFE_LEFT
    assert decision.reason == "front_blocked_strafe"


def test_reactive_obstacle_teacher_uses_depth_only_for_blocked_scene():
    teacher = ReactiveDepthWaypointTeacher()
    depth = np.full((120, 160), 6.0, dtype=np.float32)
    depth[36:84, 56:104] = 1.2
    depth[36:84, 8:56] = 1.5
    depth[36:84, 104:152] = 5.2

    decision = teacher.choose_obstacle_avoidance_action(
        depth_image_m=depth,
    )
    assert decision.action == DiscreteDroneAction.STRAFE_RIGHT
    assert decision.reason == "front_blocked_strafe"
    profile = dict(decision.diagnostics.get("decision_profile", {}))
    assert profile.get("family") == "obstacle_avoidance"


def test_reactive_obstacle_teacher_can_prefer_left_when_both_sides_are_safe():
    teacher = ReactiveDepthWaypointTeacher()
    depth = np.full((120, 160), 6.0, dtype=np.float32)
    depth[36:84, 56:104] = 1.2
    depth[36:84, 8:56] = 4.9
    depth[36:84, 104:152] = 5.1

    decision = teacher.choose_obstacle_avoidance_action(
        depth_image_m=depth,
        target_family="left",
    )
    assert decision.action == DiscreteDroneAction.STRAFE_LEFT
    assert decision.reason == "front_blocked_strafe"


def test_reactive_obstacle_teacher_does_not_force_left_if_right_is_much_clearer():
    teacher = ReactiveDepthWaypointTeacher()
    depth = np.full((120, 160), 6.0, dtype=np.float32)
    depth[36:84, 56:104] = 1.2
    depth[36:84, 8:56] = 2.1
    depth[36:84, 104:152] = 5.4

    decision = teacher.choose_obstacle_avoidance_action(
        depth_image_m=depth,
        target_family="left",
    )
    assert decision.action == DiscreteDroneAction.STRAFE_RIGHT
    assert decision.reason == "front_blocked_strafe"


def test_reactive_teacher_yaws_toward_goal_when_clear():
    teacher = ReactiveDepthWaypointTeacher()
    home_pose = make_pose()
    current_pose = make_pose(yaw_deg=0.0)
    goal = LocalWaypointGoal(local_position_xyz_m=(5.0, 4.0, 0.0))

    depth = np.full((120, 160), 6.0, dtype=np.float32)
    decision = teacher.choose_action(
        home_pose=home_pose,
        current_pose=current_pose,
        goal=goal,
        depth_image_m=depth,
    )
    assert decision.action == DiscreteDroneAction.YAW_RIGHT
    assert decision.reason == "heading_correction"
    profile = dict(decision.diagnostics.get("decision_profile", {}))
    assert profile.get("family") == "goal_turn"
    assert "branch_score" in profile


def test_latlon_alt_to_local_m_preserves_altitude_delta():
    home_pose = make_pose(alt=10.0)
    current_pose = make_pose(lat=47.00001, lon=8.00001, alt=12.5)
    local_xyz = latlon_alt_to_local_m(home_pose, current_pose)
    assert abs(local_xyz[2] - 2.5) < 1e-6


def test_decision_game_goal_sampling_preserves_target_family_metadata():
    teacher = ReactiveDepthWaypointTeacher()
    home_pose = make_pose()
    current_pose = make_pose(yaw_deg=0.0)
    depth = np.full((120, 160), 6.0, dtype=np.float32)

    goal = teacher.sample_goal(
        home_pose=home_pose,
        current_pose=current_pose,
        rng=random.Random(0),
        depth_image_m=depth,
        mode="decision_game",
        target_family="left",
    )

    assert goal.metadata["goal_mode"] == "decision_game"
    assert goal.metadata["target_family"] == "left"
