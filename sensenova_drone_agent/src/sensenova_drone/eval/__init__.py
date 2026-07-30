"""Evaluation utilities for rollout experiments and closed-loop SITL evaluation."""

from sensenova_drone.eval.closed_loop import (
    ClosedLoopEpisodeSummary,
    aggregate_closed_loop_summaries,
    compute_goal_metrics,
    goal_feature_vector_from_metrics,
    load_eval_episode_summary,
    summarize_closed_loop_episode,
    write_episode_contact_strip,
)
try:
    from sensenova_drone.eval.contact_sheet import make_video_contact_sheet
except ModuleNotFoundError:  # pragma: no cover - optional utility dependency
    make_video_contact_sheet = None
try:
    from sensenova_drone.eval.video_motion import estimate_motion_strength
except ModuleNotFoundError:  # pragma: no cover - optional utility dependency
    estimate_motion_strength = None

__all__ = [
    "ClosedLoopEpisodeSummary",
    "aggregate_closed_loop_summaries",
    "compute_goal_metrics",
    "estimate_motion_strength",
    "goal_feature_vector_from_metrics",
    "load_eval_episode_summary",
    "make_video_contact_sheet",
    "summarize_closed_loop_episode",
    "write_episode_contact_strip",
]
