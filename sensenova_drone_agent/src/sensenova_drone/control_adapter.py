from __future__ import annotations

from sensenova_drone.actions import (
    DiscreteDroneAction,
    actions_to_kairos_prompt_suffix,
    discrete_to_command,
)
from sensenova_drone.observation import CameraIntrinsics, Pose
from sensenova_drone.world_state import ActionSequence, KairosActionCondition


DEFAULT_CAMERA_CONTROL_ORIGIN = [
    0.0,
    0.532139961,
    0.946026558,
    0.5,
    0.5,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
]


class DroneToKairosControlAdapter:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def make_action_sequence(
        self,
        actions: list[DiscreteDroneAction],
    ) -> ActionSequence:
        commands = [
            discrete_to_command(action, self.cfg.get("actions", {}))
            for action in actions
        ]
        return ActionSequence(
            actions=actions,
            commands=commands,
            horizon_steps=len(actions),
        )

    def make_kairos_action_condition(
        self,
        current_pose: Pose | None,
        action_sequence: ActionSequence,
        camera_intrinsics: CameraIntrinsics | None,
    ) -> KairosActionCondition:
        prompt_suffix = actions_to_kairos_prompt_suffix(action_sequence.actions)
        camera_control_direction, unsupported_actions = self._map_action_sequence_to_kairos_camera_control(
            action_sequence.actions
        )
        predicted_pose_trajectory = self._predict_pose_trajectory(
            current_pose=current_pose,
            commands=action_sequence.commands,
        )
        camera_control_origin = list(
            self.cfg.get("kairos", {}).get(
                "camera_control_origin",
                DEFAULT_CAMERA_CONTROL_ORIGIN,
            )
        )
        supported_actions = [
            action.value for action in action_sequence.actions if action.value not in unsupported_actions
        ]

        return KairosActionCondition(
            action_sequence=action_sequence,
            prompt_suffix=prompt_suffix,
            camera_control_direction=camera_control_direction,
            camera_control_speed=self.cfg.get("kairos", {}).get("camera_control_speed", 1.0),
            camera_control_origin=camera_control_origin,
            predicted_pose_trajectory=predicted_pose_trajectory,
            intrinsics=camera_intrinsics,
            metadata={
                "mapping_note": (
                    "Kairos public camera controls are approximate. "
                    "Do not assume exact drone dynamics."
                ),
                "native_camera_control_supported": camera_control_direction is not None,
                "native_camera_control_supported_actions": supported_actions,
                "unsupported_actions": unsupported_actions,
                "native_camera_control_partial": bool(supported_actions) and bool(unsupported_actions),
            },
        )

    def _map_action_sequence_to_kairos_camera_control(
        self,
        actions: list[DiscreteDroneAction],
    ) -> tuple[str | None, list[str]]:
        yaw_direction: str | None = None
        vertical_direction: str | None = None
        unsupported_actions: list[str] = []

        for action in actions:
            if action == DiscreteDroneAction.YAW_LEFT:
                yaw_direction = "Left"
            elif action == DiscreteDroneAction.YAW_RIGHT:
                yaw_direction = "Right"
            elif action == DiscreteDroneAction.ASCEND:
                vertical_direction = "Up"
            elif action == DiscreteDroneAction.DESCEND:
                vertical_direction = "Down"
            elif action == DiscreteDroneAction.HOVER:
                continue
            else:
                unsupported_actions.append(action.value)

        if yaw_direction and vertical_direction:
            return f"{yaw_direction}{vertical_direction}", unsupported_actions
        if yaw_direction:
            return yaw_direction, unsupported_actions
        if vertical_direction:
            return vertical_direction, unsupported_actions
        return None, unsupported_actions

    def _predict_pose_trajectory(
        self,
        current_pose: Pose | None,
        commands: list,
    ) -> list[Pose] | None:
        """
        Approximate future pose trajectory from current pose and candidate commands.

        MVP returns None.
        """

        _ = (current_pose, commands)
        return None
