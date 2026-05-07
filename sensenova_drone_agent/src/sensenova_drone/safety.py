from __future__ import annotations

from dataclasses import replace

from sensenova_drone.actions import DiscreteDroneAction, DroneCommand
from sensenova_drone.memory import RealObservationMemory
from sensenova_drone.observation import Observation


class SafetyShield:
    """
    Final command filter before anything is sent to PX4 SITL.

    The MVP implementation clamps speeds and duration to conservative limits
    and can optionally refuse translational motion when pose is unavailable.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def filter(
        self,
        proposed_command: DroneCommand,
        observation: Observation,
        memory: RealObservationMemory,
    ) -> DroneCommand:
        safety_cfg = self.cfg.get("safety", {})
        max_linear = float(safety_cfg.get("max_linear_speed_m_s", 0.5))
        max_yaw = float(safety_cfg.get("max_yawspeed_deg_s", 10.0))
        max_duration = float(safety_cfg.get("max_duration_s", 1.0))
        allow_translation_without_pose = bool(safety_cfg.get("allow_translation_without_pose", True))

        filtered = replace(
            proposed_command,
            forward_m_s=_clamp(proposed_command.forward_m_s, -max_linear, max_linear),
            right_m_s=_clamp(proposed_command.right_m_s, -max_linear, max_linear),
            down_m_s=_clamp(proposed_command.down_m_s, -max_linear, max_linear),
            yawspeed_deg_s=_clamp(proposed_command.yawspeed_deg_s, -max_yaw, max_yaw),
            duration_s=_clamp(proposed_command.duration_s, 0.0, max_duration),
            metadata={**proposed_command.metadata, "memory_size": len(memory)},
        )

        if observation.metadata.get("collision_imminent"):
            evasive = replace(
                filtered,
                forward_m_s=min(filtered.forward_m_s, 0.0),
                down_m_s=min(filtered.down_m_s, 0.0),
                metadata={**filtered.metadata, "reason": "collision_imminent_evasive_only"},
            )
            if (
                abs(evasive.forward_m_s) < 1e-6
                and abs(evasive.right_m_s) < 1e-6
                and abs(evasive.down_m_s) < 1e-6
                and abs(evasive.yawspeed_deg_s) < 1e-6
            ):
                return replace(
                    evasive,
                    source_action=DiscreteDroneAction.HOVER,
                )
            return evasive

        if not allow_translation_without_pose and observation.pose is None:
            return replace(
                filtered,
                forward_m_s=0.0,
                right_m_s=0.0,
                down_m_s=0.0,
                source_action=DiscreteDroneAction.HOVER,
                metadata={**filtered.metadata, "reason": "pose_unavailable"},
            )

        return filtered


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
