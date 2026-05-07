from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sensenova_drone.actions import DiscreteDroneAction, DroneCommand, coerce_discrete_action
from sensenova_drone.control_adapter import DroneToKairosControlAdapter
from sensenova_drone.memory import RealObservationMemory
from sensenova_drone.scoring import GoalSpec, RolloutScorer, ScoreResult
from sensenova_drone.safety import SafetyShield
from sensenova_drone.world_state import (
    ActionSequence,
    KairosActionCondition,
    PredictedFuture,
    WorldState,
)


@dataclass
class CandidatePlan:
    action_sequence: ActionSequence
    action_condition: KairosActionCondition
    proposed_command: DroneCommand
    predicted_future: PredictedFuture | None = None
    score: float | None = None
    score_result: ScoreResult | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class KairosMPCPlanner:
    """
    Receding-horizon planner.

    At each real timestep:
        1. encode real observation
        2. sample candidate action sequences
        3. rollout Kairos future for each candidate
        4. score each future
        5. choose argmax
        6. return first command only
    """

    def __init__(
        self,
        world_model,
        scorer: RolloutScorer,
        safety_shield: SafetyShield,
        control_adapter: DroneToKairosControlAdapter,
        cfg: dict,
    ):
        self.world_model = world_model
        self.scorer = scorer
        self.safety_shield = safety_shield
        self.control_adapter = control_adapter
        self.cfg = cfg

    def sample_action_sequences(self) -> list[list[DiscreteDroneAction]]:
        raw_actions = self.cfg.get("planner", {}).get(
            "candidate_actions",
            ["hover", "yaw_left", "yaw_right"],
        )

        sequences: list[list[DiscreteDroneAction]] = []
        for item in raw_actions:
            if isinstance(item, (list, tuple)):
                sequences.append([coerce_discrete_action(value) for value in item])
            else:
                sequences.append([coerce_discrete_action(item)])
        return sequences

    def plan(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        goal: GoalSpec,
        episode_step_dir: str,
    ) -> CandidatePlan:
        candidates: list[CandidatePlan] = []
        best_plan: CandidatePlan | None = None
        best_score = float("-inf")

        for idx, action_list in enumerate(self.sample_action_sequences()):
            action_sequence = self.control_adapter.make_action_sequence(action_list)
            action_condition = self.control_adapter.make_kairos_action_condition(
                current_pose=world_state.pose,
                action_sequence=action_sequence,
                camera_intrinsics=world_state.intrinsics,
            )

            proposed_command = action_sequence.commands[0]
            prefiltered_command = self.safety_shield.filter(
                proposed_command,
                observation=world_state.observation,
                memory=memory,
            )

            candidate = CandidatePlan(
                action_sequence=action_sequence,
                action_condition=action_condition,
                proposed_command=prefiltered_command,
                diagnostics={
                    "candidate_index": idx,
                    "prefilter_changed_command": prefiltered_command != proposed_command,
                },
            )

            if self.cfg.get("planner", {}).get("use_kairos_rollouts", True):
                predicted_future = self.world_model.rollout_from_state(
                    world_state=world_state,
                    memory=memory,
                    action_condition=action_condition,
                    goal=goal,
                    out_dir=f"{episode_step_dir}/candidate_{idx}_{action_list[0].value}",
                    return_type=self.cfg.get("planner", {}).get("rollout_return_type", "video"),
                )
                candidate.predicted_future = predicted_future
            else:
                predicted_future = None

            score_result = self.scorer.score(
                observation=world_state.observation,
                memory=memory,
                goal=goal,
                action_sequence=action_sequence,
                command=prefiltered_command,
                predicted_future=predicted_future,
            )

            candidate.score = score_result.total
            candidate.score_result = score_result
            candidates.append(candidate)

            if score_result.total > best_score:
                best_score = score_result.total
                best_plan = candidate

        if best_plan is None:
            raise RuntimeError("Planner produced no valid candidate plan.")

        best_plan.diagnostics["all_candidates"] = [self._candidate_summary(candidate) for candidate in candidates]
        best_plan.diagnostics["decision_rule"] = "argmax_A R(Kairos rollout under candidate action sequence A)"
        return best_plan

    def _candidate_summary(self, candidate: CandidatePlan) -> dict[str, Any]:
        return {
            "action_sequence": [action.value for action in candidate.action_sequence.actions],
            "command": {
                "forward_m_s": candidate.proposed_command.forward_m_s,
                "right_m_s": candidate.proposed_command.right_m_s,
                "down_m_s": candidate.proposed_command.down_m_s,
                "yawspeed_deg_s": candidate.proposed_command.yawspeed_deg_s,
                "duration_s": candidate.proposed_command.duration_s,
            },
            "kairos_condition": {
                "prompt_suffix": candidate.action_condition.prompt_suffix,
                "camera_control_direction": candidate.action_condition.camera_control_direction,
                "camera_control_speed": candidate.action_condition.camera_control_speed,
            },
            "rollout_success": bool(candidate.predicted_future and candidate.predicted_future.success),
            "score": candidate.score,
        }
