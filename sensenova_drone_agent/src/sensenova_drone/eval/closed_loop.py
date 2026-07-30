from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw

from sensenova_drone.expert_policy import latlon_alt_to_local_m, normalize_goal_features, world_delta_to_body_m
from sensenova_drone.observation import Pose


OSCILLATION_PAIRS = {
    ("yaw_left", "yaw_right"),
    ("yaw_right", "yaw_left"),
    ("strafe_left", "strafe_right"),
    ("strafe_right", "strafe_left"),
    ("forward", "backward"),
    ("backward", "forward"),
}


@dataclass
class ClosedLoopEpisodeSummary:
    episode_id: str
    run_id: str
    checkpoint_path: str
    status: str
    step_count: int
    actions: list[str]
    initial_distance_xy_m: float | None
    final_distance_xy_m: float | None
    net_progress_m: float
    mean_step_progress_m: float
    progress_ratio: float | None
    moved_toward_goal: bool
    reached_goal: bool
    stalled: bool
    oscillation_flips: int
    oscillation_rate: float
    safety_override_count: int
    collision_imminent_count: int
    mean_front_clearance_before_m: float | None
    mean_front_clearance_after_m: float | None
    min_front_clearance_before_m: float | None
    min_front_clearance_after_m: float | None
    front_clearance_delta_m: float | None
    front_clearance_improved: bool
    mean_confidence: float | None
    goal_reached_radius_m: float
    altitude_tolerance_m: float
    contact_strip_path: str | None
    episode_dir: str
    task_label: str = "waypoint"


def compute_goal_metrics(
    *,
    home_pose: Pose,
    current_pose: Pose,
    goal_local_xyz_m: tuple[float, float, float],
) -> dict[str, float]:
    yaw_deg = float(current_pose.metadata.get("attitude_euler_deg", {}).get("yaw_deg", 0.0))
    current_local = latlon_alt_to_local_m(home_pose, current_pose)
    delta_east_m = goal_local_xyz_m[0] - current_local[0]
    delta_north_m = goal_local_xyz_m[1] - current_local[1]
    delta_up_m = goal_local_xyz_m[2] - current_local[2]
    forward_m, right_m = world_delta_to_body_m(delta_east_m, delta_north_m, yaw_deg)
    heading_error_deg = math.degrees(math.atan2(right_m, max(forward_m, 1e-6)))
    distance_xy_m = math.hypot(delta_east_m, delta_north_m)
    distance_3d_m = math.sqrt(delta_east_m**2 + delta_north_m**2 + delta_up_m**2)
    return {
        "current_local_x_m": current_local[0],
        "current_local_y_m": current_local[1],
        "current_local_z_m": current_local[2],
        "goal_forward_m": forward_m,
        "goal_right_m": right_m,
        "goal_alt_error_m": delta_up_m,
        "goal_heading_error_deg": heading_error_deg,
        "distance_xy_m": distance_xy_m,
        "distance_3d_m": distance_3d_m,
    }


def goal_feature_vector_from_metrics(metrics: Mapping[str, Any]) -> list[float]:
    return normalize_goal_features(
        float(metrics.get("goal_forward_m", 0.0)),
        float(metrics.get("goal_right_m", 0.0)),
        float(metrics.get("goal_alt_error_m", 0.0)),
        float(metrics.get("goal_heading_error_deg", 0.0)),
    )


def summarize_closed_loop_episode(
    *,
    episode_dir: str | Path,
    episode_payload: Mapping[str, Any],
    step_payloads: list[Mapping[str, Any]],
    goal_reached_radius_m: float,
    altitude_tolerance_m: float,
) -> ClosedLoopEpisodeSummary:
    actions = [str(step.get("action", "unknown")) for step in step_payloads]
    safety_override_count = sum(
        1
        for step in step_payloads
        if bool(step.get("metadata", {}).get("eval", {}).get("safety_override", False))
    )
    collision_imminent_count = sum(
        1
        for step in step_payloads
        if bool(step.get("metadata", {}).get("eval", {}).get("collision_imminent", False))
    )
    front_before_values = [
        _maybe_float(step.get("metadata", {}).get("depth_before", {}).get("front_m"))
        for step in step_payloads
    ]
    front_after_values = [
        _maybe_float(step.get("metadata", {}).get("depth_after", {}).get("front_m"))
        for step in step_payloads
    ]
    valid_front_before = [value for value in front_before_values if value is not None]
    valid_front_after = [value for value in front_after_values if value is not None]

    distance_series = [
        _maybe_float(step.get("metadata", {}).get("goal_before", {}).get("distance_xy_m"))
        for step in step_payloads
    ]
    post_distance_series = [
        _maybe_float(step.get("metadata", {}).get("goal_after", {}).get("distance_xy_m"))
        for step in step_payloads
    ]
    progress_values = [
        _maybe_float(step.get("metadata", {}).get("eval", {}).get("progress_m"))
        for step in step_payloads
    ]
    valid_progress = [value for value in progress_values if value is not None]
    initial_distance = next((value for value in distance_series if value is not None), None)
    final_distance = next((value for value in reversed(post_distance_series) if value is not None), None)

    oscillation_flips = 0
    for prev, nxt in zip(actions[:-1], actions[1:]):
        if (prev, nxt) in OSCILLATION_PAIRS:
            oscillation_flips += 1

    confidences = [
        _maybe_float(step.get("metadata", {}).get("policy", {}).get("confidence"))
        for step in step_payloads
    ]
    valid_confidences = [value for value in confidences if value is not None]

    reached_goal = False
    if step_payloads:
        final_goal_after = dict(step_payloads[-1].get("metadata", {}).get("goal_after", {}))
        final_alt_error = abs(float(final_goal_after.get("goal_alt_error_m", 0.0)))
        final_dist = _maybe_float(final_goal_after.get("distance_xy_m"))
        reached_goal = (
            final_dist is not None
            and final_dist <= float(goal_reached_radius_m)
            and final_alt_error <= float(altitude_tolerance_m)
        )

    net_progress = float(0.0 if initial_distance is None or final_distance is None else initial_distance - final_distance)
    mean_step_progress = float(sum(valid_progress) / len(valid_progress)) if valid_progress else 0.0
    progress_ratio = None
    if initial_distance is not None and initial_distance > 1e-6 and final_distance is not None:
        progress_ratio = float(max(0.0, min(1.0, net_progress / initial_distance)))

    stalled = bool(
        len(step_payloads) >= 3
        and not reached_goal
        and net_progress < max(0.75, 0.1 * (initial_distance or 0.0))
    )
    moved_toward_goal = bool(net_progress > 0.25)
    mean_front_before = (
        float(sum(valid_front_before) / len(valid_front_before))
        if valid_front_before
        else None
    )
    mean_front_after = (
        float(sum(valid_front_after) / len(valid_front_after))
        if valid_front_after
        else None
    )
    min_front_before = min(valid_front_before) if valid_front_before else None
    min_front_after = min(valid_front_after) if valid_front_after else None
    front_clearance_delta = (
        None
        if mean_front_before is None or mean_front_after is None
        else float(mean_front_after - mean_front_before)
    )
    front_clearance_improved = bool(front_clearance_delta is not None and front_clearance_delta > 0.15)

    return ClosedLoopEpisodeSummary(
        episode_id=str(episode_payload.get("episode_id", Path(episode_dir).name)),
        run_id=str(episode_payload.get("run_id", "unknown")),
        checkpoint_path=str(episode_payload.get("checkpoint_path", "")),
        status=str(episode_payload.get("status", "unknown")),
        step_count=int(episode_payload.get("step_count", len(step_payloads))),
        actions=actions,
        initial_distance_xy_m=initial_distance,
        final_distance_xy_m=final_distance,
        net_progress_m=net_progress,
        mean_step_progress_m=mean_step_progress,
        progress_ratio=progress_ratio,
        moved_toward_goal=moved_toward_goal,
        reached_goal=reached_goal,
        stalled=stalled,
        oscillation_flips=oscillation_flips,
        oscillation_rate=(oscillation_flips / max(len(actions) - 1, 1)) if actions else 0.0,
        safety_override_count=safety_override_count,
        collision_imminent_count=collision_imminent_count,
        mean_front_clearance_before_m=mean_front_before,
        mean_front_clearance_after_m=mean_front_after,
        min_front_clearance_before_m=min_front_before,
        min_front_clearance_after_m=min_front_after,
        front_clearance_delta_m=front_clearance_delta,
        front_clearance_improved=front_clearance_improved,
        mean_confidence=(
            float(sum(valid_confidences) / len(valid_confidences))
            if valid_confidences
            else None
        ),
        goal_reached_radius_m=float(goal_reached_radius_m),
        altitude_tolerance_m=float(altitude_tolerance_m),
        contact_strip_path=_maybe_str(episode_payload.get("contact_strip_path")),
        episode_dir=str(Path(episode_dir).resolve()),
        task_label=str(episode_payload.get("task_label", "waypoint")),
    )


def aggregate_closed_loop_summaries(summaries: list[ClosedLoopEpisodeSummary]) -> dict[str, Any]:
    if not summaries:
        return {
            "task_label": "waypoint",
            "num_episodes": 0,
            "success_rate": 0.0,
            "moved_toward_goal_rate": 0.0,
            "stall_rate": 0.0,
            "mean_net_progress_m": 0.0,
            "mean_progress_ratio": 0.0,
            "mean_oscillation_rate": 0.0,
            "safety_override_rate": 0.0,
            "collision_imminent_rate": 0.0,
            "front_clearance_improved_rate": 0.0,
            "mean_front_clearance_delta_m": 0.0,
            "mean_front_clearance_before_m": 0.0,
            "mean_front_clearance_after_m": 0.0,
            "action_counts": {},
        }

    action_counts = Counter(action for summary in summaries for action in summary.actions)
    mean_progress_ratio_values = [summary.progress_ratio for summary in summaries if summary.progress_ratio is not None]

    return {
        "task_label": summaries[0].task_label,
        "num_episodes": len(summaries),
        "success_rate": sum(1 for summary in summaries if summary.reached_goal) / len(summaries),
        "moved_toward_goal_rate": sum(1 for summary in summaries if summary.moved_toward_goal) / len(summaries),
        "stall_rate": sum(1 for summary in summaries if summary.stalled) / len(summaries),
        "mean_net_progress_m": sum(summary.net_progress_m for summary in summaries) / len(summaries),
        "mean_progress_ratio": (
            sum(mean_progress_ratio_values) / len(mean_progress_ratio_values)
            if mean_progress_ratio_values
            else 0.0
        ),
        "mean_oscillation_rate": sum(summary.oscillation_rate for summary in summaries) / len(summaries),
        "safety_override_rate": (
            sum(summary.safety_override_count for summary in summaries)
            / max(sum(summary.step_count for summary in summaries), 1)
        ),
        "collision_imminent_rate": (
            sum(summary.collision_imminent_count for summary in summaries)
            / max(sum(summary.step_count for summary in summaries), 1)
        ),
        "front_clearance_improved_rate": (
            sum(1 for summary in summaries if summary.front_clearance_improved) / len(summaries)
        ),
        "mean_front_clearance_delta_m": (
            sum(summary.front_clearance_delta_m for summary in summaries if summary.front_clearance_delta_m is not None)
            / max(sum(1 for summary in summaries if summary.front_clearance_delta_m is not None), 1)
        ),
        "mean_front_clearance_before_m": (
            sum(summary.mean_front_clearance_before_m for summary in summaries if summary.mean_front_clearance_before_m is not None)
            / max(sum(1 for summary in summaries if summary.mean_front_clearance_before_m is not None), 1)
        ),
        "mean_front_clearance_after_m": (
            sum(summary.mean_front_clearance_after_m for summary in summaries if summary.mean_front_clearance_after_m is not None)
            / max(sum(1 for summary in summaries if summary.mean_front_clearance_after_m is not None), 1)
        ),
        "action_counts": dict(action_counts),
    }


def load_eval_episode_summary(episode_dir: str | Path) -> ClosedLoopEpisodeSummary:
    episode_path = Path(episode_dir)
    episode_payload = json.loads((episode_path / "episode.json").read_text(encoding="utf-8"))
    precomputed_summary = episode_payload.get("summary")
    if isinstance(precomputed_summary, dict) and str(precomputed_summary.get("task_label", "")) == "tree_avoidance":
        payload = dict(precomputed_summary)
        payload.setdefault("episode_dir", str(episode_path.resolve()))
        payload.setdefault("contact_strip_path", episode_payload.get("contact_strip_path"))
        payload.setdefault("task_label", "tree_avoidance")
        return ClosedLoopEpisodeSummary(**payload)
    step_payloads = [
        json.loads(step_path.read_text(encoding="utf-8"))
        for step_path in sorted(episode_path.glob("step_*/step.json"))
    ]
    return summarize_closed_loop_episode(
        episode_dir=episode_path,
        episode_payload=episode_payload,
        step_payloads=step_payloads,
        goal_reached_radius_m=float(episode_payload.get("goal_reached_radius_m", 1.0)),
        altitude_tolerance_m=float(episode_payload.get("altitude_tolerance_m", 0.35)),
    )


def write_episode_contact_strip(
    episode_dir: str | Path,
    *,
    out_path: str | Path | None = None,
    max_frames: int = 8,
) -> str | None:
    episode_path = Path(episode_dir)
    frame_paths = sorted(episode_path.rglob("frame_before.png"))
    after_paths = sorted(episode_path.rglob("frame_after.png"))
    if after_paths:
        frame_paths = frame_paths + [after_paths[-1]]
    if not frame_paths:
        return None

    selected_paths = _sample_paths(frame_paths, max_count=max_frames)
    thumbs = [Image.open(path).convert("RGB").resize((256, 144), Image.BILINEAR) for path in selected_paths]
    label_height = 28
    canvas = Image.new("RGB", (256 * len(thumbs), 144 + label_height), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    for index, frame in enumerate(thumbs):
        left = index * 256
        canvas.paste(frame, (left, label_height))
        draw.text((left + 8, 6), f"s{index}", fill=(255, 255, 255))

    output = Path(out_path) if out_path is not None else episode_path / "contact_strip.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return str(output.resolve())


def _sample_paths(paths: list[Path], *, max_count: int) -> list[Path]:
    if len(paths) <= max_count:
        return paths
    selected: list[Path] = []
    for index in range(max_count):
        source_index = round(index * (len(paths) - 1) / max(max_count - 1, 1))
        selected.append(paths[source_index])
    return selected


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
