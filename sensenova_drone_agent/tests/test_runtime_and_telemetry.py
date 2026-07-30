import json

from sensenova_drone.actions import DiscreteDroneAction, DroneCommand
from sensenova_drone.grounded_world_model import (
    GroundedMovementProposal,
    GroundedWorldModelMovementPlanner,
)
from sensenova_drone.memory import RealObservationMemory
from sensenova_drone.observation import CameraIntrinsics, Observation, Pose
from sensenova_drone.planner import CandidatePlan
from sensenova_drone.policy import PolicyOutput, RuntimeModePlanner
from sensenova_drone.scoring import GoalSpec
from sensenova_drone.telemetry import TelemetryLogger
from sensenova_drone.world_state import (
    ActionSequence,
    KairosActionCondition,
    ObservationEncoding,
    WorldState,
)


class StubMPCPlanner:
    def plan(self, world_state, memory, goal, episode_step_dir):
        _ = (world_state, memory, goal, episode_step_dir)
        action_sequence = ActionSequence(
            actions=[DiscreteDroneAction.YAW_LEFT],
            commands=[DroneCommand(yawspeed_deg_s=-5.0, duration_s=0.5, source_action=DiscreteDroneAction.YAW_LEFT)],
            horizon_steps=1,
        )
        condition = KairosActionCondition(action_sequence=action_sequence, prompt_suffix="The camera slowly yaws left.")
        return CandidatePlan(
            action_sequence=action_sequence,
            action_condition=condition,
            proposed_command=action_sequence.commands[0],
            diagnostics={
                "decision_rule": "argmax_A R(Kairos rollout under candidate action sequence A)",
                "all_candidates": [
                    {
                        "action_sequence": ["hover"],
                        "kairos_condition": {
                            "prompt_suffix": "The camera remains mostly stable.",
                            "camera_control_direction": None,
                            "camera_control_speed": 1.0,
                        },
                        "rollout_success": True,
                        "score": 0.12,
                    }
                ],
            },
        )


class LowConfidencePolicyPlanner:
    def plan(self, world_state, memory, goal, episode_step_dir=None):
        _ = (world_state, memory, goal, episode_step_dir)
        return PolicyOutput(
            action=DiscreteDroneAction.HOVER,
            command=DroneCommand(duration_s=0.5, source_action=DiscreteDroneAction.HOVER),
            confidence=0.25,
            metadata={"mode": "policy_head"},
        )


class StubGroundedWorldModel:
    def propose_movement(self, world_state, memory, goal, episode_step_dir=None):
        _ = (world_state, memory, goal, episode_step_dir)
        return GroundedMovementProposal(
            action=DiscreteDroneAction.STRAFE_RIGHT,
            command=DroneCommand(
                right_m_s=0.3,
                duration_s=0.5,
                source_action=DiscreteDroneAction.STRAFE_RIGHT,
            ),
            confidence=0.8,
            latent_state={"stub": True},
            metadata={"backend": "stub_world_model"},
        )


def test_runtime_mode_hybrid_falls_back_to_mpc(tmp_path) -> None:
    cfg = {
        "runtime": {"mode": "hybrid"},
        "policy": {"confidence_threshold": 0.75},
        "hybrid": {"use_policy_when_confident": True, "fallback_to_mpc": True},
    }
    router = RuntimeModePlanner(
        mpc_planner=StubMPCPlanner(),
        policy_planner=LowConfidencePolicyPlanner(),
        cfg=cfg,
    )

    world_state = WorldState(
        observation=Observation(frame_rgb="frame"),
        encoding=ObservationEncoding(frame_path="frame.png"),
        pose=None,
        intrinsics=None,
        memory_size=0,
    )
    plan = router.plan(world_state, RealObservationMemory(), GoalSpec(prompt="goal"), str(tmp_path))
    assert isinstance(plan, CandidatePlan)
    assert plan.diagnostics["hybrid_decision"] == "mpc_fallback"


def test_runtime_mode_grounded_world_model_uses_real_feedback_planner(tmp_path) -> None:
    cfg = {"runtime": {"mode": "grounded_world_model"}}
    router = RuntimeModePlanner(
        mpc_planner=StubMPCPlanner(),
        grounded_world_model_planner=GroundedWorldModelMovementPlanner(
            world_model=StubGroundedWorldModel(),
            cfg=cfg,
        ),
        cfg=cfg,
    )

    world_state = WorldState(
        observation=Observation(frame_rgb="frame"),
        encoding=ObservationEncoding(frame_path="frame.png", metadata={"backend": "stub"}),
        pose=None,
        intrinsics=None,
        memory_size=0,
    )

    plan = router.plan(world_state, RealObservationMemory(), GoalSpec(prompt="avoid trees"), str(tmp_path))

    assert isinstance(plan, PolicyOutput)
    assert plan.action == DiscreteDroneAction.STRAFE_RIGHT
    assert plan.command.right_m_s == 0.3
    assert plan.metadata["runtime_mode"] == "grounded_world_model"
    assert plan.metadata["decision_rule"] == "world_model.propose_movement(real_observation_state)"
    assert plan.metadata["next_observation_source"] == "real_gazebo_camera_after_actuation"
    assert plan.metadata["generated_rollouts_used_as_state"] is False


def test_telemetry_logger_writes_required_loop_fields(tmp_path) -> None:
    logger = TelemetryLogger(tmp_path)
    observation = Observation(
        frame_rgb="frame",
        pose=Pose(position_xyz=(0.0, 0.0, 2.0)),
        intrinsics=CameraIntrinsics(width=640, height=480),
        metadata={"frame_path": "/tmp/fake.png"},
    )
    world_state = WorldState(
        observation=observation,
        encoding=ObservationEncoding(frame_path="/tmp/fake.png", metadata={"backend": "subprocess", "latent_available": False}),
        pose=observation.pose,
        intrinsics=observation.intrinsics,
        memory_size=12,
    )

    action_sequence = ActionSequence(
        actions=[DiscreteDroneAction.YAW_LEFT],
        commands=[DroneCommand(yawspeed_deg_s=-5.0, duration_s=0.5, source_action=DiscreteDroneAction.YAW_LEFT)],
        horizon_steps=1,
    )
    condition = KairosActionCondition(
        action_sequence=action_sequence,
        prompt_suffix="The camera slowly yaws left.",
        camera_control_direction="Left",
        camera_control_speed=1.0,
    )
    plan = CandidatePlan(
        action_sequence=action_sequence,
        action_condition=condition,
        proposed_command=action_sequence.commands[0],
        diagnostics={
            "decision_rule": "argmax_A R(Kairos rollout under candidate action sequence A)",
            "all_candidates": [
                {
                    "action_sequence": ["hover"],
                    "kairos_condition": {
                        "prompt_suffix": "The camera remains mostly stable.",
                        "camera_control_direction": None,
                        "camera_control_speed": 1.0,
                    },
                    "rollout_success": True,
                    "score": 0.12,
                }
            ],
        },
    )

    logger.log_step(
        observation=observation,
        world_state=world_state,
        plan=plan,
        executed_command=action_sequence.commands[0],
    )

    telemetry_path = tmp_path / "step_000001" / "telemetry.json"
    payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert payload["pose_T_t"]["position_xyz"] == [0.0, 0.0, 2.0]
    assert payload["intrinsics_K_t"]["width"] == 640
    assert payload["observation_encoding"]["backend"] == "subprocess"
    assert payload["candidate_action_sequences"] == [["hover"]]
    assert payload["decision_rule"] == "argmax_A R(Kairos rollout under candidate action sequence A)"
    assert payload["generated_rollouts_used_as_state"] is False


def test_telemetry_logger_writes_grounded_world_model_fields(tmp_path) -> None:
    logger = TelemetryLogger(tmp_path)
    observation = Observation(
        frame_rgb="frame",
        pose=Pose(position_xyz=(0.0, 0.0, 2.0)),
        intrinsics=CameraIntrinsics(width=640, height=480),
        metadata={"frame_path": "/tmp/fake.png"},
    )
    world_state = WorldState(
        observation=observation,
        encoding=ObservationEncoding(
            frame_path="/tmp/fake.png",
            metadata={"backend": "bc_grounded_surrogate", "latent_available": False},
        ),
        pose=observation.pose,
        intrinsics=observation.intrinsics,
        memory_size=3,
    )
    command = DroneCommand(
        right_m_s=-0.3,
        duration_s=0.5,
        source_action=DiscreteDroneAction.STRAFE_LEFT,
    )
    plan = PolicyOutput(
        action=DiscreteDroneAction.STRAFE_LEFT,
        command=command,
        confidence=0.7,
        metadata={
            "mode": "grounded_world_model",
            "decision_rule": "world_model.propose_movement(real_observation_state)",
            "proposal_metadata": {"backend": "stub_world_model"},
        },
    )

    logger.log_step(
        observation=observation,
        world_state=world_state,
        plan=plan,
        executed_command=command,
    )

    payload = json.loads((tmp_path / "step_000001" / "telemetry.json").read_text(encoding="utf-8"))
    assert payload["decision_rule"] == "world_model.propose_movement(real_observation_state)"
    assert payload["candidate_action_sequences"] == [["strafe_left"]]
    assert payload["candidate_rollouts"][0]["world_model_proposal"]["backend"] == "stub_world_model"
    assert payload["generated_rollouts_used_as_state"] is False
