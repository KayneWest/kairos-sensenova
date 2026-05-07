import json
from pathlib import Path

from PIL import Image

from sensenova_drone.actions import DiscreteDroneAction, DroneCommand
from sensenova_drone.kairos_adapter import SubprocessKairosAdapter
from sensenova_drone.memory import RealObservationMemory
from sensenova_drone.observation import Observation
from sensenova_drone.scoring import GoalSpec
from sensenova_drone.world_state import (
    ActionSequence,
    KairosActionCondition,
    ObservationEncoding,
    WorldState,
)


def test_subprocess_adapter_creates_dry_run_request_when_execution_disabled(tmp_path) -> None:
    template_path = tmp_path / "template.json"
    template_path.write_text(
        json.dumps(
            {
                "prompt": "",
                "input_image": "",
                "output_dir": "",
            }
        ),
        encoding="utf-8",
    )

    cfg = {
        "kairos": {
            "execute_subprocess": False,
            "template_request_json": str(template_path),
            "repo_root": str(tmp_path),
            "camera_control_speed": 1.0,
        }
    }
    adapter = SubprocessKairosAdapter(cfg)
    frame_path = tmp_path / "real_frame.png"
    frame_path.write_bytes(b"not-a-real-png-but-copyable")
    action_sequence = ActionSequence(
        actions=[DiscreteDroneAction.YAW_RIGHT],
        commands=[DroneCommand(yawspeed_deg_s=5.0, duration_s=0.5, source_action=DiscreteDroneAction.YAW_RIGHT)],
        horizon_steps=1,
    )
    action_condition = KairosActionCondition(
        action_sequence=action_sequence,
        prompt_suffix="The camera slowly yaws right.",
        camera_control_direction="Right",
        camera_control_speed=1.0,
        camera_control_origin=[0.0] * 19,
    )
    world_state = WorldState(
        observation=Observation(frame_rgb="frame"),
        encoding=ObservationEncoding(frame_path=str(frame_path), metadata={"backend": "subprocess"}),
        pose=None,
        intrinsics=None,
        memory_size=1,
    )

    predicted_future = adapter.rollout_from_state(
        world_state=world_state,
        memory=RealObservationMemory(),
        action_condition=action_condition,
        goal=GoalSpec(prompt="look for an opening"),
        out_dir=str(tmp_path / "candidate"),
    )

    assert predicted_future.success is False
    assert predicted_future.metadata["dry_run"] is True

    request_path = tmp_path / "candidate" / "candidate_config.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["input_image"].endswith("input_frame.png")
    assert "look for an opening" in request["prompt"]
    assert "slowly yaws right" in request["prompt"]
    assert request["camera_control_direction"] == "Right"
    assert request["camera_control_speed"] == 1.0
    assert request["camera_control_origin"] == [0.0] * 19


def test_subprocess_adapter_uses_input_video_fallback_when_camera_control_runtime_is_unsupported(tmp_path) -> None:
    template_path = tmp_path / "template.json"
    template_path.write_text(
        json.dumps(
            {
                "prompt": "",
                "input_image": "",
                "output_dir": "",
                "num_frames": 9,
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "kairos_config.py"
    config_path.write_text('dit_config = {"has_image_input": False}\n', encoding="utf-8")

    cfg = {
        "kairos": {
            "execute_subprocess": False,
            "template_request_json": str(template_path),
            "repo_root": str(tmp_path),
            "config_file": str(config_path),
            "camera_control_speed": 1.0,
            "enable_action_conditioned_input_video_fallback": True,
        }
    }
    adapter = SubprocessKairosAdapter(cfg)
    frame_path = tmp_path / "real_frame.png"
    Image.new("RGB", (64, 48), color=(64, 96, 128)).save(frame_path)

    action_sequence = ActionSequence(
        actions=[DiscreteDroneAction.YAW_LEFT],
        commands=[DroneCommand(yawspeed_deg_s=-5.0, duration_s=0.5, source_action=DiscreteDroneAction.YAW_LEFT)],
        horizon_steps=1,
    )
    action_condition = KairosActionCondition(
        action_sequence=action_sequence,
        prompt_suffix="The camera slowly yaws left.",
        camera_control_direction="Left",
        camera_control_speed=1.0,
        camera_control_origin=[0.0] * 19,
    )
    world_state = WorldState(
        observation=Observation(frame_rgb="frame"),
        encoding=ObservationEncoding(frame_path=str(frame_path), metadata={"backend": "subprocess"}),
        pose=None,
        intrinsics=None,
        memory_size=1,
    )

    predicted_future = adapter.rollout_from_state(
        world_state=world_state,
        memory=RealObservationMemory(),
        action_condition=action_condition,
        goal=GoalSpec(prompt="look for an opening"),
        out_dir=str(tmp_path / "candidate_fallback"),
    )

    assert predicted_future.success is False
    assert predicted_future.metadata["dry_run"] is True
    assert predicted_future.metadata["conditioning_backend"] == "synthetic_input_video_fallback"

    request_path = tmp_path / "candidate_fallback" / "candidate_config.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert "camera_control_direction" not in request
    assert "camera_control_speed" not in request
    assert "camera_control_origin" not in request

    input_video_dir = Path(request["input_video"])
    assert input_video_dir.is_dir()
    assert len(sorted(input_video_dir.glob("frame_*.png"))) == 9
