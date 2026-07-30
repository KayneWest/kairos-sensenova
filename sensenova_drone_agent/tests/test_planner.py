from sensenova_drone.actions import DiscreteDroneAction
from sensenova_drone.actions import DroneCommand
from sensenova_drone.control_adapter import DroneToKairosControlAdapter
from sensenova_drone.memory import MemoryEntry, RealObservationMemory
from sensenova_drone.observation import Observation
from sensenova_drone.planner import KairosMPCPlanner
from sensenova_drone.safety import SafetyShield
from sensenova_drone.scoring import GoalSpec, ScoreResult
from sensenova_drone.world_state import ObservationEncoding, PredictedFuture, WorldState


class StubWorldModel:
    def rollout_from_state(self, world_state, memory, action_condition, goal, out_dir, return_type="video"):
        _ = (world_state, memory, goal, out_dir, return_type)
        return PredictedFuture(
            action_condition=action_condition,
            success=action_condition.action_sequence.actions[0] == DiscreteDroneAction.YAW_LEFT,
            metadata={"backend": "stub"},
        )


class StubScorer:
    def score(self, observation, memory, goal, action_sequence, command, predicted_future):
        _ = (observation, memory, goal, command, predicted_future)
        score_map = {
            DiscreteDroneAction.HOVER: 0.1,
            DiscreteDroneAction.YAW_LEFT: 1.0,
            DiscreteDroneAction.YAW_RIGHT: 0.2,
        }
        total = score_map[action_sequence.actions[0]]
        return ScoreResult(total=total, reward=total, safety_penalty=0.0, components={"stub": total})


def test_mpc_planner_uses_argmax_candidate_selection(tmp_path) -> None:
    cfg = {
        "planner": {
            "candidate_actions": ["hover", "yaw_left", "yaw_right"],
            "use_kairos_rollouts": True,
        },
        "actions": {},
        "kairos": {"camera_control_speed": 1.0},
        "safety": {},
    }
    planner = KairosMPCPlanner(
        world_model=StubWorldModel(),
        scorer=StubScorer(),
        safety_shield=SafetyShield(cfg),
        control_adapter=DroneToKairosControlAdapter(cfg),
        cfg=cfg,
    )

    memory = RealObservationMemory([MemoryEntry(observation=Observation(frame_rgb="frame"))])
    world_state = WorldState(
        observation=Observation(frame_rgb="frame"),
        encoding=ObservationEncoding(frame_path="frame.png", metadata={"backend": "subprocess"}),
        pose=None,
        intrinsics=None,
        memory_size=len(memory),
    )

    plan = planner.plan(
        world_state=world_state,
        memory=memory,
        goal=GoalSpec(prompt="find a safe path"),
        episode_step_dir=str(tmp_path),
    )

    assert plan.action_sequence.actions[0] == DiscreteDroneAction.YAW_LEFT
    assert plan.score == 1.0
    assert plan.diagnostics["decision_rule"] == "argmax_A R(Kairos rollout under candidate action sequence A)"
    assert len(plan.diagnostics["all_candidates"]) == 3
    assert [candidate["action_sequence"] for candidate in plan.diagnostics["all_candidates"]] == [
        ["hover"],
        ["yaw_left"],
        ["yaw_right"],
    ]


def test_safety_shield_allows_evasive_motion_when_collision_imminent() -> None:
    shield = SafetyShield({"safety": {"max_linear_speed_m_s": 0.5, "max_yawspeed_deg_s": 10.0, "max_duration_s": 1.0}})
    memory = RealObservationMemory()
    observation = Observation(frame_rgb="frame", metadata={"collision_imminent": True})

    filtered = shield.filter(
        DroneCommand(
            forward_m_s=0.3,
            right_m_s=-0.3,
            down_m_s=-0.3,
            yawspeed_deg_s=5.0,
            duration_s=0.75,
            source_action=DiscreteDroneAction.STRAFE_LEFT,
        ),
        observation,
        memory,
    )

    assert filtered.forward_m_s == 0.0
    assert filtered.right_m_s == -0.3
    assert filtered.down_m_s == -0.3
    assert filtered.yawspeed_deg_s == 5.0
    assert filtered.source_action == DiscreteDroneAction.STRAFE_LEFT
