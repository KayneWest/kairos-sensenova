from __future__ import annotations

import pytest

from sensenova_drone.eval.closed_loop import aggregate_closed_loop_summaries, summarize_closed_loop_episode


def test_summarize_closed_loop_episode_detects_progress_and_oscillation(tmp_path):
    episode_payload = {
        "episode_id": "eval_001",
        "run_id": "run_001",
        "checkpoint_path": "/tmp/best.pt",
        "status": "timeout",
        "step_count": 4,
        "contact_strip_path": str(tmp_path / "contact_strip.png"),
    }
    step_payloads = [
        {
            "action": "yaw_left",
            "metadata": {
                "goal_before": {"distance_xy_m": 8.0},
                "goal_after": {"distance_xy_m": 7.0, "goal_alt_error_m": 0.0},
                "policy": {"confidence": 0.7},
                "eval": {"progress_m": 1.0, "safety_override": False, "collision_imminent": False},
            },
        },
        {
            "action": "yaw_right",
            "metadata": {
                "goal_before": {"distance_xy_m": 7.0},
                "goal_after": {"distance_xy_m": 6.8, "goal_alt_error_m": 0.0},
                "policy": {"confidence": 0.6},
                "eval": {"progress_m": 0.2, "safety_override": True, "collision_imminent": False},
            },
        },
        {
            "action": "yaw_left",
            "metadata": {
                "goal_before": {"distance_xy_m": 6.8},
                "goal_after": {"distance_xy_m": 6.7, "goal_alt_error_m": 0.0},
                "policy": {"confidence": 0.65},
                "eval": {"progress_m": 0.1, "safety_override": False, "collision_imminent": True},
            },
        },
        {
            "action": "forward",
            "metadata": {
                "goal_before": {"distance_xy_m": 6.7},
                "goal_after": {"distance_xy_m": 6.4, "goal_alt_error_m": 0.0},
                "policy": {"confidence": 0.8},
                "eval": {"progress_m": 0.3, "safety_override": False, "collision_imminent": False},
            },
        },
    ]

    summary = summarize_closed_loop_episode(
        episode_dir=tmp_path,
        episode_payload=episode_payload,
        step_payloads=step_payloads,
        goal_reached_radius_m=1.0,
        altitude_tolerance_m=0.35,
    )

    assert summary.net_progress_m == pytest.approx(1.6)
    assert summary.moved_toward_goal is True
    assert summary.oscillation_flips == 2
    assert summary.safety_override_count == 1
    assert summary.collision_imminent_count == 1
    assert summary.mean_confidence is not None

    aggregate = aggregate_closed_loop_summaries([summary])
    assert aggregate["success_rate"] == 0.0
    assert aggregate["moved_toward_goal_rate"] == 1.0
    assert aggregate["mean_net_progress_m"] == pytest.approx(1.6)
