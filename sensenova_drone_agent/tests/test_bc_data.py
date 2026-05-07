from __future__ import annotations

import json

from sensenova_drone.actions import DiscreteDroneAction, DroneCommand
from sensenova_drone.bc_data import (
    BCEpisodeStep,
    episode_split_map,
    export_bc_manifest,
    load_episode_steps,
)
from sensenova_drone.observation import CameraIntrinsics, Pose


def test_save_load_and_export_bc_manifest(tmp_path):
    episodes_root = tmp_path / "episodes"
    episode_dir = episodes_root / "episode_001"
    step_dir = episode_dir / "step_000000"
    step_dir.mkdir(parents=True)

    before_path = step_dir / "frame_before.png"
    after_path = step_dir / "frame_after.png"
    before_path.write_bytes(b"fake-before")
    after_path.write_bytes(b"fake-after")

    step = BCEpisodeStep(
        episode_id="episode_001",
        step_index=0,
        action=DiscreteDroneAction.FORWARD,
        command=DroneCommand(
            forward_m_s=0.3,
            duration_s=0.75,
            source_action=DiscreteDroneAction.FORWARD,
        ),
        image_path=str(before_path),
        next_image_path=str(after_path),
        timestamp_s=123.0,
        pose=Pose(position_xyz=(1.0, 2.0, 3.0)),
        intrinsics=CameraIntrinsics(width=640, height=480),
        metadata={
            "source": "test",
            "world_label": "forest",
            "scenario_label": "forest_north",
            "teacher": {
                "reason": "heading_correction",
                "decision_profile": {
                    "family": "goal_turn",
                    "target_family": "left",
                    "decision_rich": True,
                    "branch_score": 0.75,
                },
            },
        },
    )

    json_path = step_dir / "step.json"
    json_path.write_text(json.dumps(step.to_dict(), indent=2), encoding="utf-8")

    loaded = load_episode_steps(episode_dir)
    assert len(loaded) == 1
    assert loaded[0].action == DiscreteDroneAction.FORWARD
    assert loaded[0].command.source_action == DiscreteDroneAction.FORWARD
    assert loaded[0].pose is not None
    assert loaded[0].intrinsics is not None

    manifest_path = tmp_path / "manifest.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = export_bc_manifest(
        episodes_root=episodes_root,
        out_jsonl=manifest_path,
        val_ratio=0.0,
        summary_json=summary_path,
    )

    assert manifest_path.is_file()
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 1
    assert records[0]["action"] == "forward"
    assert records[0]["split"] == "train"
    assert summary["num_examples"] == 1
    assert summary["counts_by_action"]["forward"] == 1
    assert summary["counts_by_teacher_reason"]["heading_correction"] == 1
    assert summary["counts_by_decision_family"]["goal_turn"] == 1
    assert summary["counts_by_target_family"]["left"] == 1
    assert summary["counts_by_world"]["forest"] == 1
    assert summary["counts_by_scenario"]["forest_north"] == 1
    assert summary["decision_rich_examples"] == 1


def test_episode_split_map_guarantees_val_episode_for_multi_episode_export():
    split_map = episode_split_map(
        [
            "episode_001",
            "episode_002",
            "episode_003",
            "episode_004",
            "episode_005",
        ],
        val_ratio=0.1,
    )
    assert set(split_map.values()) == {"train", "val"}
    assert sum(1 for split in split_map.values() if split == "val") == 1


def test_export_bc_manifest_filters_by_world_and_decision_family(tmp_path):
    episodes_root = tmp_path / "episodes"
    step_dir = episodes_root / "episode_forest" / "step_000000"
    step_dir.mkdir(parents=True)
    (step_dir / "frame_before.png").write_bytes(b"x")
    (step_dir / "frame_after.png").write_bytes(b"y")
    forest_step = BCEpisodeStep(
        episode_id="episode_forest",
        step_index=0,
        action=DiscreteDroneAction.YAW_LEFT,
        command=DroneCommand(source_action=DiscreteDroneAction.YAW_LEFT),
        image_path=str(step_dir / "frame_before.png"),
        next_image_path=str(step_dir / "frame_after.png"),
        metadata={
            "world_label": "forest",
            "teacher": {
                "decision_profile": {
                    "family": "obstacle_avoidance",
                    "decision_rich": True,
                }
            },
        },
    )
    (step_dir / "step.json").write_text(json.dumps(forest_step.to_dict()), encoding="utf-8")

    other_step_dir = episodes_root / "episode_walls" / "step_000000"
    other_step_dir.mkdir(parents=True)
    (other_step_dir / "frame_before.png").write_bytes(b"x")
    (other_step_dir / "frame_after.png").write_bytes(b"y")
    walls_step = BCEpisodeStep(
        episode_id="episode_walls",
        step_index=0,
        action=DiscreteDroneAction.FORWARD,
        command=DroneCommand(source_action=DiscreteDroneAction.FORWARD),
        image_path=str(other_step_dir / "frame_before.png"),
        next_image_path=str(other_step_dir / "frame_after.png"),
        metadata={
            "world_label": "walls",
            "teacher": {
                "decision_profile": {
                    "family": "goal_progress",
                    "decision_rich": False,
                }
            },
        },
    )
    (other_step_dir / "step.json").write_text(json.dumps(walls_step.to_dict()), encoding="utf-8")

    summary = export_bc_manifest(
        episodes_root=episodes_root,
        out_jsonl=tmp_path / "manifest.jsonl",
        val_ratio=0.0,
        include_worlds={"forest"},
        required_decision_family="obstacle_avoidance",
        require_decision_rich=True,
        allowed_actions={"yaw_left", "yaw_right"},
    )
    assert summary["num_examples"] == 1
    assert summary["num_episodes"] == 1
    assert summary["num_source_episodes"] == 2
    assert summary["counts_by_world"]["forest"] == 1


def test_export_bc_manifest_splits_only_selected_episodes(tmp_path):
    episodes_root = tmp_path / "episodes"
    for episode_id, world_label in [
        ("episode_match_a", "forest"),
        ("episode_match_b", "forest"),
        ("episode_filtered", "walls"),
    ]:
        step_dir = episodes_root / episode_id / "step_000000"
        step_dir.mkdir(parents=True)
        (step_dir / "frame_before.png").write_bytes(b"x")
        (step_dir / "frame_after.png").write_bytes(b"y")
        family = "obstacle_avoidance" if world_label == "forest" else "goal_progress"
        action = DiscreteDroneAction.STRAFE_LEFT if world_label == "forest" else DiscreteDroneAction.FORWARD
        step = BCEpisodeStep(
            episode_id=episode_id,
            step_index=0,
            action=action,
            command=DroneCommand(source_action=action),
            image_path=str(step_dir / "frame_before.png"),
            next_image_path=str(step_dir / "frame_after.png"),
            metadata={
                "world_label": world_label,
                "teacher": {"decision_profile": {"family": family}},
            },
        )
        (step_dir / "step.json").write_text(json.dumps(step.to_dict()), encoding="utf-8")

    summary = export_bc_manifest(
        episodes_root=episodes_root,
        out_jsonl=tmp_path / "manifest.jsonl",
        val_ratio=0.5,
        include_worlds={"forest"},
        required_decision_family="obstacle_avoidance",
    )
    assert summary["num_source_episodes"] == 3
    assert summary["num_episodes"] == 2
    assert summary["num_examples"] == 2
    assert summary["train_examples"] == 1
    assert summary["val_examples"] == 1
    assert summary["val_episode_count"] == 1
