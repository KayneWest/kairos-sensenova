from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sensenova_drone.actions import DroneCommand
from sensenova_drone.memory import RealObservationMemory
from sensenova_drone.observation import Observation
from sensenova_drone.world_state import ActionSequence, PredictedFuture


@dataclass
class GoalSpec:
    prompt: str
    success_description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    total: float
    reward: float
    safety_penalty: float
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class RolloutScorer:
    """
    Scores candidate futures for the MPC planner.

    The initial implementation is intentionally simple. It preserves the
    contract the planner needs now and can later be replaced by a learned
    reward model without changing the planner interface.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def score(
        self,
        observation: Observation,
        memory: RealObservationMemory,
        goal: GoalSpec,
        action_sequence: ActionSequence,
        command: DroneCommand,
        predicted_future: PredictedFuture | None,
    ) -> ScoreResult:
        scoring_cfg = self.cfg.get("scoring", {})
        notes: list[str] = []

        reward = float(scoring_cfg.get("base_reward", 0.0))
        if predicted_future is None:
            notes.append("no_rollout")
        elif predicted_future.success:
            reward += float(scoring_cfg.get("rollout_success_bonus", 1.0))
        else:
            reward += float(scoring_cfg.get("rollout_failure_bonus", -0.25))

        if goal.prompt.strip():
            reward += float(scoring_cfg.get("goal_prompt_bonus", 0.1))

        action_bias = scoring_cfg.get("action_bias", {})
        if action_sequence.actions:
            reward += float(action_bias.get(action_sequence.actions[0].value, 0.0))

        if predicted_future is not None and predicted_future.metadata.get("dry_run"):
            notes.append("dry_run_rollout")

        safety_penalty = self._safety_penalty(command)
        total = reward - safety_penalty

        return ScoreResult(
            total=total,
            reward=reward,
            safety_penalty=safety_penalty,
            components={
                "reward": reward,
                "safety_penalty": safety_penalty,
                "memory_size_bonus": 0.0 if len(memory) == 0 else float(scoring_cfg.get("memory_size_bonus", 0.0)),
            },
            notes=notes,
        )

    def _safety_penalty(self, command: DroneCommand) -> float:
        safety_cfg = self.cfg.get("safety", {})
        max_linear = float(safety_cfg.get("max_linear_speed_m_s", 0.5))
        max_yaw = float(safety_cfg.get("max_yawspeed_deg_s", 10.0))
        max_duration = float(safety_cfg.get("max_duration_s", 1.0))

        linear_over = max(
            0.0,
            abs(command.forward_m_s) - max_linear,
            abs(command.right_m_s) - max_linear,
            abs(command.down_m_s) - max_linear,
        )
        yaw_over = max(0.0, abs(command.yawspeed_deg_s) - max_yaw)
        duration_over = max(0.0, command.duration_s - max_duration)

        return linear_over + (yaw_over / max(max_yaw, 1.0)) + duration_over
