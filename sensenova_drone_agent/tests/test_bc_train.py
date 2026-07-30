from __future__ import annotations

from PIL import Image
import pytest

from sensenova_drone.bc_train import (
    BCManifestDataset,
    _extract_goal_feature_vector,
    _mirrored_action_index,
)


def test_extract_goal_feature_vector_uses_teacher_metadata():
    record = {
        "metadata": {
            "teacher": {
                "goal_features": {
                    "forward_m": 5.0,
                    "right_m": -2.5,
                    "alt_error_m": 1.5,
                    "heading_error_deg": -45.0,
                }
            }
        }
    }
    features = _extract_goal_feature_vector(record)
    assert features == [0.5, -0.5, 0.5, -0.25]


def test_bc_manifest_dataset_returns_goal_features(tmp_path):
    pytest.importorskip("torch")
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), color=(128, 128, 128)).save(image_path)
    record = {
        "image_path": str(image_path),
        "action_index": 2,
        "command": {
            "forward_m_s": 0.1,
            "right_m_s": 0.0,
            "down_m_s": 0.0,
            "yawspeed_deg_s": 1.0,
            "duration_s": 0.5,
        },
        "metadata": {
            "teacher": {
                "goal_features": {
                    "forward_m": 2.0,
                    "right_m": 1.0,
                    "alt_error_m": -0.6,
                    "heading_error_deg": 18.0,
                }
            }
        },
    }
    dataset = BCManifestDataset([record], image_size=16)
    sample = dataset[0]
    assert list(sample["goal_features"].shape) == [4]
    assert sample["goal_features"].tolist() == [0.2, 0.2, -0.2, 0.1]


def test_bc_manifest_dataset_stacks_previous_frames(tmp_path):
    pytest.importorskip("torch")
    image_a = tmp_path / "frame_a.png"
    image_b = tmp_path / "frame_b.png"
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(image_a)
    Image.new("RGB", (16, 16), color=(0, 255, 0)).save(image_b)

    records = [
        {
            "episode_id": "episode_001",
            "step_index": 0,
            "image_path": str(image_a),
            "action_index": 1,
            "command": {
                "forward_m_s": 0.0,
                "right_m_s": 0.0,
                "down_m_s": 0.0,
                "yawspeed_deg_s": 1.0,
                "duration_s": 0.5,
            },
            "metadata": {"teacher": {"goal_features": {}}},
        },
        {
            "episode_id": "episode_001",
            "step_index": 1,
            "image_path": str(image_b),
            "action_index": 2,
            "command": {
                "forward_m_s": 0.0,
                "right_m_s": 0.0,
                "down_m_s": 0.0,
                "yawspeed_deg_s": 1.0,
                "duration_s": 0.5,
            },
            "metadata": {"teacher": {"goal_features": {}}},
        },
    ]
    dataset = BCManifestDataset(records, image_size=16, frame_stack=2)
    sample = dataset[1]
    assert list(sample["image"].shape) == [6, 16, 16]


def test_bc_manifest_dataset_mirrors_lateral_actions(tmp_path):
    pytest.importorskip("torch")
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(image_path)
    record = {
        "image_path": str(image_path),
        "action_index": 7,  # strafe_left
        "command": {
            "forward_m_s": 0.0,
            "right_m_s": -0.5,
            "down_m_s": 0.0,
            "yawspeed_deg_s": 0.0,
            "duration_s": 1.0,
        },
        "metadata": {
            "teacher": {
                "goal_features": {
                    "forward_m": 0.0,
                    "right_m": -2.0,
                    "alt_error_m": 0.0,
                    "heading_error_deg": -30.0,
                }
            }
        },
    }
    dataset = BCManifestDataset([record], image_size=16, mirror_lateral_actions=True)
    assert len(dataset) == 2
    mirrored = dataset[1]
    assert int(mirrored["action_index"].item()) == _mirrored_action_index(7)
    assert mirrored["command"].tolist()[1] == 0.5
    assert mirrored["goal_features"].tolist()[1] == 0.4
    assert mirrored["goal_features"].tolist()[3] == pytest.approx(30.0 / 180.0)
