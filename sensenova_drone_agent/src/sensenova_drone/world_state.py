from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sensenova_drone.actions import DiscreteDroneAction, DroneCommand
from sensenova_drone.observation import CameraIntrinsics, Observation, Pose


@dataclass
class ObservationEncoding:
    """
    Encoded representation of a real observation.

    This may contain:
    - Kairos VAE latent, if accessible
    - image embedding, if accessible
    - path to the current frame, as a fallback
    - metadata required to reconstruct or reuse the observation
    """

    latent: Any | None = None
    image_features: Any | None = None
    frame_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldState:
    """
    The current real state used by the planner.

    This state must be built from real Gazebo observations,
    not from generated Kairos frames.
    """

    observation: Observation
    encoding: ObservationEncoding
    pose: Pose | None
    intrinsics: CameraIntrinsics | None
    memory_size: int


@dataclass
class ActionSequence:
    actions: list[DiscreteDroneAction]
    commands: list[DroneCommand]
    horizon_steps: int


@dataclass
class KairosActionCondition:
    """
    The model-side conditioning derived from a drone action sequence.

    For Kairos/Sensenova, this may include:
    - prompt suffix
    - camera_control_direction
    - camera_control_speed
    - approximate predicted pose trajectory
    - camera intrinsics
    - future extension points for native action tokens
    """

    action_sequence: ActionSequence
    prompt_suffix: str = ""
    camera_control_direction: str | None = None
    camera_control_speed: float | None = None
    camera_control_origin: list[float] | None = None
    predicted_pose_trajectory: list[Pose] | None = None
    intrinsics: CameraIntrinsics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictedFuture:
    """
    Temporary hypothesis produced by Kairos.

    This must never be appended to RealObservationMemory as truth.
    """

    action_condition: KairosActionCondition
    video_path: str | None = None
    latents: Any | None = None
    final_frame_path: str | None = None
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
