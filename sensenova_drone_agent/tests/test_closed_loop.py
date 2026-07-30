from __future__ import annotations

import json

from sensenova_drone.actions import build_action_cfg
from sensenova_drone.eval.closed_loop import load_eval_episode_summary


def test_build_action_cfg_applies_tree_overrides() -> None:
    cfg = build_action_cfg(
        duration_s=1.0,
        forward_m_s=0.4,
        strafe_m_s=0.5,
        vertical_m_s=0.35,
        yawspeed_deg_s=12.0,
    )

    assert cfg["forward"]["forward_m_s"] == 0.4
    assert cfg["strafe_left"]["right_m_s"] == -0.5
    assert cfg["strafe_right"]["right_m_s"] == 0.5
    assert cfg["ascend"]["down_m_s"] == -0.35
    assert cfg["yaw_left"]["yawspeed_deg_s"] == -12.0
    assert cfg["yaw_right"]["yawspeed_deg_s"] == 12.0


def test_load_eval_episode_summary_uses_precomputed_tree_summary(tmp_path) -> None:
    episode_dir = tmp_path / "episode_000"
    episode_dir.mkdir(parents=True)
    summary = {
        "episode_id": "episode_000",
        "run_id": "tree_eval_demo",
        "checkpoint_path": "/tmp/checkpoint.pt",
        "status": "escaped_blocked_scene",
        "step_count": 4,
        "actions": ["strafe_left", "strafe_left"],
        "initial_distance_xy_m": None,
        "final_distance_xy_m": None,
        "net_progress_m": 0.9,
        "mean_step_progress_m": 0.45,
        "progress_ratio": None,
        "moved_toward_goal": True,
        "reached_goal": True,
        "stalled": False,
        "oscillation_flips": 0,
        "oscillation_rate": 0.0,
        "safety_override_count": 0,
        "collision_imminent_count": 2,
        "mean_front_clearance_before_m": 1.2,
        "mean_front_clearance_after_m": 2.6,
        "min_front_clearance_before_m": 1.1,
        "min_front_clearance_after_m": 2.5,
        "front_clearance_delta_m": 1.4,
        "front_clearance_improved": True,
        "mean_confidence": 0.8,
        "goal_reached_radius_m": 0.0,
        "altitude_tolerance_m": 0.0,
        "contact_strip_path": None,
        "episode_dir": str(episode_dir),
        "task_label": "tree_avoidance",
    }
    payload = {
        "episode_id": "episode_000",
        "run_id": "tree_eval_demo",
        "task_label": "tree_avoidance",
        "summary": summary,
    }
    (episode_dir / "episode.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_eval_episode_summary(episode_dir)

    assert loaded.task_label == "tree_avoidance"
    assert loaded.reached_goal is True
    assert loaded.front_clearance_delta_m == 1.4
    assert loaded.net_progress_m == 0.9
