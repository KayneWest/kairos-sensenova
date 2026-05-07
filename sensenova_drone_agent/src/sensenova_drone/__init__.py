"""Core package for the Sensenova drone inference scaffold."""

from sensenova_drone.actions import (
    DiscreteDroneAction,
    DroneCommand,
    action_to_kairos_prompt_suffix,
    coerce_discrete_action,
    discrete_to_command,
)
from sensenova_drone.bc_data import ACTION_VOCAB, BCEpisodeStep, export_bc_manifest
from sensenova_drone.bc_infer import BCPrediction, BCPolicyRunner, load_bc_policy_runner
from sensenova_drone.bc_model import ImageBCPolicy
from sensenova_drone.control_adapter import DroneToKairosControlAdapter
from sensenova_drone.grounded_world_model import (
    BCGroundedWorldModelAdapter,
    GroundedMovementProposal,
    GroundedWorldModelMovementPlanner,
)
from sensenova_drone.kairos_adapter import (
    KairosWorldModelAdapter,
    PythonKairosAdapter,
    SubprocessKairosAdapter,
)
from sensenova_drone.loop import ClosedLoopAgent
from sensenova_drone.memory import MemoryEntry, RealObservationMemory
from sensenova_drone.observation import CameraIntrinsics, Observation, Pose
from sensenova_drone.observation_adapter import ObservationAdapter
from sensenova_drone.planner import CandidatePlan, KairosMPCPlanner
from sensenova_drone.policy import PolicyOutput, PolicyRuntimePlanner, RuntimeModePlanner
from sensenova_drone.safety import SafetyShield
from sensenova_drone.scoring import GoalSpec, RolloutScorer, ScoreResult
from sensenova_drone.state_estimator import StateEstimator
from sensenova_drone.telemetry import TelemetryLogger
from sensenova_drone.world_state import (
    ActionSequence,
    KairosActionCondition,
    ObservationEncoding,
    PredictedFuture,
    WorldState,
)

__all__ = [
    "ACTION_VOCAB",
    "ActionSequence",
    "BCEpisodeStep",
    "BCGroundedWorldModelAdapter",
    "BCPolicyRunner",
    "BCPrediction",
    "CameraIntrinsics",
    "CandidatePlan",
    "ClosedLoopAgent",
    "DiscreteDroneAction",
    "DroneCommand",
    "DroneToKairosControlAdapter",
    "GoalSpec",
    "GroundedMovementProposal",
    "GroundedWorldModelMovementPlanner",
    "ImageBCPolicy",
    "KairosActionCondition",
    "KairosMPCPlanner",
    "KairosWorldModelAdapter",
    "MemoryEntry",
    "Observation",
    "ObservationAdapter",
    "ObservationEncoding",
    "PolicyOutput",
    "PolicyRuntimePlanner",
    "Pose",
    "PredictedFuture",
    "PythonKairosAdapter",
    "RealObservationMemory",
    "RolloutScorer",
    "RuntimeModePlanner",
    "SafetyShield",
    "ScoreResult",
    "StateEstimator",
    "SubprocessKairosAdapter",
    "TelemetryLogger",
    "WorldState",
    "action_to_kairos_prompt_suffix",
    "coerce_discrete_action",
    "discrete_to_command",
    "export_bc_manifest",
    "load_bc_policy_runner",
]
