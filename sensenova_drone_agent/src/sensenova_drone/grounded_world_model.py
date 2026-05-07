from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sensenova_drone.actions import DiscreteDroneAction, DroneCommand, discrete_to_command
from sensenova_drone.bc_infer import BCPolicyRunner
from sensenova_drone.memory import RealObservationMemory
from sensenova_drone.policy import PolicyOutput
from sensenova_drone.scoring import GoalSpec
from sensenova_drone.world_state import ObservationEncoding, WorldState


@dataclass
class GroundedMovementProposal:
    """
    Movement hypothesis produced from the current real world-model state.

    The next observation must still come from Gazebo/PX4 after executing the
    command. This object is an action decision, not simulated state.
    """

    action: DiscreteDroneAction
    command: DroneCommand | None = None
    confidence: float | None = None
    latent_state: Any | None = None
    raw_output: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GroundedWorldModelMovementPlanner:
    """
    Planner for the grounded autoregressive control loop.

    At each step:
        real frame -> world-model state/proposal -> movement command

    The environment supplies the next frame after actuation. Generated frames
    are never accepted as the next state.
    """

    def __init__(
        self,
        world_model,
        *,
        action_cfg: dict[str, Any] | None = None,
        cfg: dict[str, Any] | None = None,
    ):
        self.world_model = world_model
        self.action_cfg = action_cfg or {}
        self.cfg = cfg or {}

    def plan(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        goal: GoalSpec,
        episode_step_dir: str | None = None,
    ) -> PolicyOutput:
        proposal = self._propose_movement(world_state, memory, goal, episode_step_dir)
        command = proposal.command or discrete_to_command(proposal.action, self.action_cfg)

        return PolicyOutput(
            action=proposal.action,
            command=command,
            confidence=proposal.confidence,
            action_logits=proposal.raw_output,
            metadata={
                "mode": "grounded_world_model",
                "runtime_mode": "grounded_world_model",
                "decision_rule": "world_model.propose_movement(real_observation_state)",
                "next_observation_source": "real_gazebo_camera_after_actuation",
                "generated_rollouts_used_as_state": False,
                "latent_available": proposal.latent_state is not None,
                "world_state_memory_size": world_state.memory_size,
                "proposal_metadata": dict(proposal.metadata),
            },
        )

    def _propose_movement(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        goal: GoalSpec,
        episode_step_dir: str | None,
    ) -> GroundedMovementProposal:
        proposer = getattr(self.world_model, "propose_movement", None)
        if proposer is None:
            proposer = getattr(self.world_model, "propose_action", None)
        if proposer is None:
            raise RuntimeError(
                "Grounded world-model mode requires world_model.propose_movement(...) "
                "or world_model.propose_action(...)."
            )

        proposal = proposer(
            world_state=world_state,
            memory=memory,
            goal=goal,
            episode_step_dir=episode_step_dir,
        )
        if proposal is None:
            raise RuntimeError("Grounded world-model proposer returned None.")
        if isinstance(proposal, GroundedMovementProposal):
            return proposal
        if isinstance(proposal, PolicyOutput):
            return GroundedMovementProposal(
                action=proposal.action,
                command=proposal.command,
                confidence=proposal.confidence,
                raw_output=proposal.action_logits,
                metadata=dict(proposal.metadata or {}),
            )
        raise TypeError(
            "Grounded world-model proposer must return GroundedMovementProposal "
            f"or PolicyOutput, got {type(proposal)!r}."
        )


class BCGroundedWorldModelAdapter:
    """
    Concrete grounded action adapter backed by the current BC visual policy.

    This is intentionally labeled as a surrogate: it gives us the correct
    grounded action loop now, while the native Kairos latent/h_t path can later
    replace this class without changing the drone loop.
    """

    def __init__(self, policy_runner: BCPolicyRunner):
        self.policy_runner = policy_runner

    def encode_observation(self, frame_rgb, frame_path: str | None = None) -> ObservationEncoding:
        _ = frame_rgb
        return ObservationEncoding(
            latent=None,
            image_features=None,
            frame_path=frame_path,
            metadata={
                "backend": "bc_grounded_surrogate",
                "latent_available": False,
                "native_kairos_state": False,
            },
        )

    def encode_observation_and_memory(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        goal: GoalSpec | None = None,
    ):
        _ = (world_state, memory, goal)
        return None

    def propose_movement(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        goal: GoalSpec,
        episode_step_dir: str | None = None,
    ) -> GroundedMovementProposal:
        _ = (memory, episode_step_dir)
        image = world_state.observation.frame_rgb
        if image is None:
            image = world_state.encoding.frame_path
        if image is None:
            raise RuntimeError("BC grounded adapter requires a real frame or frame_path.")

        goal_features = _goal_features(world_state, goal)
        prediction = self.policy_runner.predict(image, goal_features=goal_features)
        return GroundedMovementProposal(
            action=prediction.action,
            command=prediction.command,
            confidence=prediction.confidence,
            raw_output=prediction.probabilities,
            metadata={
                "backend": "bc_grounded_surrogate",
                "native_kairos_state": False,
                "goal_features": goal_features,
                "prediction_metadata": dict(prediction.metadata),
            },
        )


def _goal_features(world_state: WorldState, goal: GoalSpec) -> list[float]:
    for source in (
        goal.metadata.get("goal_features"),
        world_state.observation.metadata.get("goal_features"),
        world_state.encoding.metadata.get("goal_features"),
    ):
        if source is None:
            continue
        values = [float(value) for value in source]
        if len(values) != 4:
            raise ValueError(f"Expected 4 goal features, got {len(values)}.")
        return values
    return [0.0, 0.0, 0.0, 0.0]
