from __future__ import annotations

from PIL import Image
import pytest

from sensenova_drone.bc_data import ACTION_VOCAB
from sensenova_drone.bc_infer import load_bc_policy_runner
from sensenova_drone.bc_model import ImageBCPolicy


def test_load_bc_policy_runner_predicts(tmp_path):
    torch = pytest.importorskip("torch")

    checkpoint_path = tmp_path / "checkpoint.pt"
    model = ImageBCPolicy(num_actions=len(ACTION_VOCAB), goal_feature_dim=4, frame_stack=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"image_size": 16, "frame_stack": 2},
            "epoch": 3,
            "action_vocab": ACTION_VOCAB,
            "metrics": {"val": {"accuracy": 0.5}},
        },
        checkpoint_path,
    )

    image_path = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), color=(64, 128, 192)).save(image_path)

    runner = load_bc_policy_runner(checkpoint_path, device="cpu")
    prediction = runner.predict(image_path, goal_features=[0.1, -0.2, 0.0, 0.3])

    assert prediction.action.value in ACTION_VOCAB
    assert prediction.action_index >= 0
    assert len(prediction.probabilities) == len(ACTION_VOCAB)
    assert prediction.metadata["epoch"] == 3
    assert prediction.metadata["image_size"] == 16
    assert prediction.metadata["frame_stack"] == 2


def test_bc_policy_runner_resets_frame_history(tmp_path):
    torch = pytest.importorskip("torch")

    checkpoint_path = tmp_path / "checkpoint.pt"
    model = ImageBCPolicy(num_actions=len(ACTION_VOCAB), goal_feature_dim=4, frame_stack=3)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"image_size": 16, "frame_stack": 3},
            "epoch": 1,
            "action_vocab": ACTION_VOCAB,
            "metrics": {},
        },
        checkpoint_path,
    )

    frame_a = tmp_path / "frame_a.png"
    frame_b = tmp_path / "frame_b.png"
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(frame_a)
    Image.new("RGB", (16, 16), color=(0, 255, 0)).save(frame_b)

    runner = load_bc_policy_runner(checkpoint_path, device="cpu")
    runner.predict(frame_a, goal_features=[0.0, 0.0, 0.0, 0.0])
    runner.predict(frame_b, goal_features=[0.0, 0.0, 0.0, 0.0])
    runner.reset_history()
    prediction = runner.predict(frame_b, goal_features=[0.0, 0.0, 0.0, 0.0])

    assert prediction.metadata["frame_stack"] == 3
