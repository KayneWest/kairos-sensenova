#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sensenova_drone.midtraining import load_sequence_cache, build_valid_anchors, cache_summary  # noqa: E402


DEFAULT_ZIP = "sensenova_drone_agent/data/robotics/soar/soar-dataset-numpy.zip"
DEFAULT_OUT = "sensenova_drone_agent/data/robotics/soar/sequence_caches/soar_rgb32_small.npz"


@dataclass
class SoarTrajectory:
    path: str
    robot: str
    scene: str
    policy: str
    date: str
    outcome: str
    traj: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a small SOAR numpy-zip subset into the generic phase-2 sequence cache schema. "
            "This reads the zip directly and uses ffmpeg for frame extraction, so full extraction is not required."
        )
    )
    parser.add_argument("--zip", default=DEFAULT_ZIP)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--max-trajectories", type=int, default=64)
    parser.add_argument("--max-steps-per-trajectory", type=int, default=64)
    parser.add_argument("--frame-size", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--action-aggregation",
        choices=["sample", "mean", "sum", "last"],
        default="sample",
        help=(
            "How to map high-rate SOAR actions onto exported frame steps. sample preserves the old behavior "
            "and stores actions[frame_idx]. mean/sum/last aggregate actions over the interval from the current "
            "exported frame to the next exported frame."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-outcomes", default="success,failure")
    parser.add_argument(
        "--selection-mode",
        choices=["outcome_balanced", "task_balanced"],
        default="outcome_balanced",
        help=(
            "outcome_balanced preserves the original success/failure-balanced random sampling. "
            "task_balanced selects repeated language tasks, preferably with both successes and failures."
        ),
    )
    parser.add_argument("--target-task-count", type=int, default=64)
    parser.add_argument("--min-trajectories-per-task", type=int, default=4)
    parser.add_argument("--max-trajectories-per-task", type=int, default=8)
    parser.add_argument("--require-both-outcomes-per-task", action="store_true")
    parser.add_argument(
        "--reward-mode",
        choices=[
            "final_success",
            "trajectory_success",
            "linear_success_progress",
            "signed_trajectory_success",
            "signed_final_success",
            "signed_linear_success_progress",
            "linear_success_progress_with_action_penalty",
        ],
        default="final_success",
        help=(
            "How to convert SOAR success.txt into per-step rewards. final_success gives reward 1 only on the "
            "last exported frame of successful trajectories. trajectory_success gives reward 1 to every frame of "
            "successful trajectories. linear_success_progress ramps from 0 to 1 across successful trajectories. "
            "signed_* modes give failures negative rewards. *_action_penalty subtracts a small normalized action cost."
        ),
    )
    parser.add_argument("--reward-action-penalty", type=float, default=0.01)
    parser.add_argument(
        "--feature",
        choices=["rgb_flat", "kairos_vae", "kairos_vae_flat"],
        default="rgb_flat",
        help=(
            "Feature representation to write as z. rgb_flat is a cheap placeholder. "
            "kairos_vae writes pooled Wan VAE stats. kairos_vae_flat writes the full flattened Wan VAE latent."
        ),
    )
    parser.add_argument("--kairos-config", default="kairos/configs/kairos_4b_config_DMD.py")
    parser.add_argument("--kairos-device", default="cuda")
    parser.add_argument("--kairos-dtype", default="bfloat16")
    parser.add_argument("--kairos-height", type=int, default=128)
    parser.add_argument("--kairos-width", type=int, default=128)
    parser.add_argument("--kairos-tiled", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = resolve_path(args.zip)
    out_path = resolve_path(args.out)
    summary_path = resolve_path(args.summary_json) if args.summary_json else out_path.with_suffix(".summary.json")

    trajectories = discover_trajectories(zip_path)
    include_outcomes = {item.strip() for item in args.include_outcomes.split(",") if item.strip()}
    if args.selection_mode == "task_balanced":
        selected, selection_metadata = select_task_balanced_trajectories(
            zip_path,
            trajectories,
            max_trajectories=args.max_trajectories,
            include_outcomes=include_outcomes,
            seed=args.seed,
            target_task_count=args.target_task_count,
            min_trajectories_per_task=args.min_trajectories_per_task,
            max_trajectories_per_task=args.max_trajectories_per_task,
            require_both_outcomes=args.require_both_outcomes_per_task,
        )
    else:
        selected = select_trajectories(
            trajectories,
            max_trajectories=args.max_trajectories,
            include_outcomes=include_outcomes,
            seed=args.seed,
        )
        selection_metadata = {"mode": "outcome_balanced"}
    if args.dry_run:
        summary = {
            "zip": str(zip_path),
            "trajectory_count": len(trajectories),
            "selected_count": len(selected),
            "selection": selection_metadata,
            "feature": args.feature,
            "reward_mode": args.reward_mode,
            "reward_action_penalty": float(args.reward_action_penalty),
            "frame_extract_size": frame_extract_size(args),
            "frame_stride": int(args.frame_stride),
            "action_aggregation": args.action_aggregation,
            "kairos": kairos_feature_config(args),
            "selected_examples": [asdict(item) for item in selected[:20]],
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    if not selected:
        raise RuntimeError("No SOAR trajectories selected.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cache, records, feature_metadata = export_cache(zip_path, selected, args)
    np.savez_compressed(
        out_path,
        z=cache["z"],
        action=cache["action"],
        reward=cache["reward"],
        episode=cache["episode"],
        step=cache["step"],
        task_id=cache["task_id"],
        done=cache["done"],
    )
    loaded = load_sequence_cache(out_path)
    anchors = build_valid_anchors(loaded, context_len=8, mtp_horizon=8)
    summary = {
        "zip": str(zip_path),
        "out": str(out_path),
        "trajectory_count": len(trajectories),
        "selected_count": len(selected),
        "selection": selection_metadata,
        "exported_steps": int(cache["z"].shape[0]),
        "feature": args.feature,
        "reward_mode": args.reward_mode,
        "reward_action_penalty": float(args.reward_action_penalty),
        "frame_size": args.frame_size,
        "frame_stride": int(args.frame_stride),
        "action_aggregation": args.action_aggregation,
        "frame_extract_size": frame_extract_size(args),
        "kairos": kairos_feature_config(args),
        "feature_metadata": feature_metadata,
        "action_dim": int(cache["action"].shape[1]) if cache["action"].ndim == 2 else None,
        "z_dim": int(cache["z"].shape[1]) if cache["z"].ndim == 2 else None,
        "success_steps": int(np.sum(cache["reward"] > 0.0)),
        "reward_positive_fraction": float(np.mean(cache["reward"] > 0.0)) if cache["reward"].size else 0.0,
        "reward_sum": float(np.sum(cache["reward"])) if cache["reward"].size else 0.0,
        "midtraining_cache": cache_summary(loaded, anchors),
        "records_preview": records[:20],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def discover_trajectories(zip_path: Path) -> list[SoarTrajectory]:
    with zipfile.ZipFile(zip_path) as archive:
        action_dirs = sorted({name.rsplit("/", 1)[0] for name in archive.namelist() if name.endswith("/actions.npy")})
    trajectories: list[SoarTrajectory] = []
    for path in action_dirs:
        parts = path.split("/")
        # soar-dataset-local/<robot>/<scene>/<policy>/<date>/<outcome>/<traj>
        if len(parts) < 7:
            continue
        trajectories.append(
            SoarTrajectory(
                path=path,
                robot=parts[1],
                scene=parts[2],
                policy=parts[3],
                date=parts[4],
                outcome=parts[5],
                traj=parts[6],
            )
        )
    return trajectories


def select_trajectories(
    trajectories: list[SoarTrajectory],
    *,
    max_trajectories: int,
    include_outcomes: set[str],
    seed: int,
) -> list[SoarTrajectory]:
    filtered = [traj for traj in trajectories if traj.outcome in include_outcomes]
    rng = np.random.default_rng(seed)
    by_outcome: dict[str, list[SoarTrajectory]] = {}
    for traj in filtered:
        by_outcome.setdefault(traj.outcome, []).append(traj)
    selected: list[SoarTrajectory] = []
    outcomes = sorted(by_outcome)
    if not outcomes:
        return []
    per = max(1, math.ceil(max_trajectories / len(outcomes)))
    for outcome in outcomes:
        items = list(by_outcome[outcome])
        rng.shuffle(items)
        selected.extend(items[:per])
    rng.shuffle(selected)
    return selected[:max_trajectories]


def select_task_balanced_trajectories(
    zip_path: Path,
    trajectories: list[SoarTrajectory],
    *,
    max_trajectories: int,
    include_outcomes: set[str],
    seed: int,
    target_task_count: int,
    min_trajectories_per_task: int,
    max_trajectories_per_task: int,
    require_both_outcomes: bool,
) -> tuple[list[SoarTrajectory], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    filtered = [traj for traj in trajectories if traj.outcome in include_outcomes]
    task_groups: dict[str, dict[str, list[SoarTrajectory]]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for traj in filtered:
            language = read_text(archive, f"{traj.path}/language_task.txt").strip()
            task_groups.setdefault(language, {}).setdefault(traj.outcome, []).append(traj)

    candidates: list[tuple[str, dict[str, list[SoarTrajectory]]]] = []
    both_outcome_count = 0
    for language, outcome_groups in task_groups.items():
        total = sum(len(items) for items in outcome_groups.values())
        has_both = all(outcome in outcome_groups and outcome_groups[outcome] for outcome in include_outcomes)
        both_outcome_count += int(has_both)
        if total < min_trajectories_per_task:
            continue
        if require_both_outcomes and not has_both:
            continue
        candidates.append((language, outcome_groups))

    if not candidates:
        raise RuntimeError("No task-balanced SOAR candidates matched the requested filters.")
    rng.shuffle(candidates)
    selected_tasks = candidates[: max(1, min(int(target_task_count), len(candidates)))]
    per_task_goal = max(1, math.ceil(max_trajectories / max(1, len(selected_tasks))))
    per_task_goal = min(per_task_goal, max(1, int(max_trajectories_per_task)))

    selected: list[SoarTrajectory] = []
    selected_paths: set[str] = set()
    task_counts: dict[str, dict[str, int]] = {}
    leftovers: list[tuple[str, SoarTrajectory]] = []
    for language, outcome_groups in selected_tasks:
        picks, remaining = sample_task_trajectories(
            outcome_groups,
            rng=rng,
            limit=per_task_goal,
            include_outcomes=include_outcomes,
        )
        for traj in picks:
            if traj.path not in selected_paths and len(selected) < max_trajectories:
                selected.append(traj)
                selected_paths.add(traj.path)
                task_counts.setdefault(language, {}).setdefault(traj.outcome, 0)
                task_counts[language][traj.outcome] += 1
        leftovers.extend((language, traj) for traj in remaining)

    rng.shuffle(leftovers)
    for language, traj in leftovers:
        if len(selected) >= max_trajectories:
            break
        if traj.path in selected_paths:
            continue
        current_task_count = sum(task_counts.get(language, {}).values())
        if current_task_count >= max_trajectories_per_task:
            continue
        selected.append(traj)
        selected_paths.add(traj.path)
        task_counts.setdefault(language, {}).setdefault(traj.outcome, 0)
        task_counts[language][traj.outcome] += 1

    rng.shuffle(selected)
    metadata = {
        "mode": "task_balanced",
        "filtered_trajectories": len(filtered),
        "candidate_tasks": len(candidates),
        "tasks_with_both_outcomes": both_outcome_count,
        "selected_tasks": len(task_counts),
        "selected_trajectories": len(selected),
        "target_task_count": int(target_task_count),
        "min_trajectories_per_task": int(min_trajectories_per_task),
        "max_trajectories_per_task": int(max_trajectories_per_task),
        "require_both_outcomes_per_task": bool(require_both_outcomes),
        "selected_task_counts_preview": dict(list(task_counts.items())[:20]),
    }
    return selected[:max_trajectories], metadata


def sample_task_trajectories(
    outcome_groups: dict[str, list[SoarTrajectory]],
    *,
    rng: np.random.Generator,
    limit: int,
    include_outcomes: set[str],
) -> tuple[list[SoarTrajectory], list[SoarTrajectory]]:
    groups = {outcome: list(outcome_groups.get(outcome, [])) for outcome in sorted(include_outcomes)}
    for items in groups.values():
        rng.shuffle(items)
    picks: list[SoarTrajectory] = []
    while len(picks) < limit and any(groups.values()):
        for outcome in sorted(groups):
            if len(picks) >= limit:
                break
            if groups[outcome]:
                picks.append(groups[outcome].pop())
    remaining = [traj for items in groups.values() for traj in items]
    return picks, remaining


def export_cache(
    zip_path: Path,
    selected: list[SoarTrajectory],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    z_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    reward_rows: list[float] = []
    episode_rows: list[int] = []
    step_rows: list[int] = []
    task_rows: list[int] = []
    done_rows: list[bool] = []
    records: list[dict[str, Any]] = []
    task_ids: dict[str, int] = {}
    feature_encoder = make_feature_encoder(args)
    feature_metadata: dict[str, Any] = {
        "backend": args.feature,
        "feature": args.feature,
        "frame_extract_size": frame_extract_size(args),
    }

    tmp_root = Path(tempfile.mkdtemp(prefix="soar_export_"))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for episode_idx, traj in enumerate(selected):
                traj_tmp = tmp_root / f"episode_{episode_idx:05d}"
                traj_tmp.mkdir(parents=True, exist_ok=True)
                actions = np.load(io.BytesIO(archive.read(f"{traj.path}/actions.npy"))).astype(np.float32)
                success = read_text(archive, f"{traj.path}/success.txt").strip().lower() == "true"
                language = read_text(archive, f"{traj.path}/language_task.txt").strip()
                task_id = task_ids.setdefault(language, len(task_ids))
                video_member = f"{traj.path}/trajectory.mp4"
                video_path = traj_tmp / "trajectory.mp4"
                video_path.write_bytes(archive.read(video_member))
                frames_dir = traj_tmp / "frames"
                extract_width, extract_height = frame_extract_size(args)
                frames = extract_frames(video_path, frames_dir, width=extract_width, height=extract_height)
                usable = min(len(frames), actions.shape[0], int(args.max_steps_per_trajectory))
                if args.frame_stride > 1:
                    frame_indices = list(range(0, usable, int(args.frame_stride)))
                else:
                    frame_indices = list(range(usable))
                for local_step, frame_idx in enumerate(frame_indices):
                    feature, metadata = encode_frame_feature(
                        frames[frame_idx],
                        args=args,
                        feature_encoder=feature_encoder,
                    )
                    if len(z_rows) == 0:
                        feature_metadata.update(metadata)
                    action_end = action_interval_end(
                        frame_indices=frame_indices,
                        local_step=local_step,
                        usable=usable,
                        frame_stride=int(args.frame_stride),
                    )
                    action = aggregate_actions(
                        actions,
                        start=frame_idx,
                        end=action_end,
                        mode=args.action_aggregation,
                    )
                    z_rows.append(feature)
                    action_rows.append(action)
                    reward_rows.append(
                        soar_reward(
                            success=success,
                            local_step=local_step,
                            step_count=len(frame_indices),
                            mode=args.reward_mode,
                            action=action,
                            action_penalty=float(args.reward_action_penalty),
                        )
                    )
                    episode_rows.append(episode_idx)
                    step_rows.append(local_step)
                    task_rows.append(task_id)
                    done_rows.append(local_step == len(frame_indices) - 1)
                records.append(
                    {
                        "episode": episode_idx,
                        "trajectory": traj.path,
                        "language_task": language,
                        "task_id": task_id,
                        "success": success,
                        "reward_mode": args.reward_mode,
                        "actions_shape": list(actions.shape),
                        "frames": len(frames),
                        "exported_steps": len(frame_indices),
                        "frame_stride": int(args.frame_stride),
                        "action_aggregation": args.action_aggregation,
                    }
                )
    finally:
        if args.keep_temp:
            print(f"Kept temp dir: {tmp_root}", file=sys.stderr)
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)

    if not z_rows:
        raise RuntimeError("No rows exported from selected SOAR trajectories.")
    return {
        "z": np.stack(z_rows).astype(np.float32),
        "action": np.stack(action_rows).astype(np.float32),
        "reward": np.asarray(reward_rows, dtype=np.float32),
        "episode": np.asarray(episode_rows, dtype=np.int64),
        "step": np.asarray(step_rows, dtype=np.int64),
        "task_id": np.asarray(task_rows, dtype=np.int64),
        "done": np.asarray(done_rows, dtype=bool),
    }, records, feature_metadata


def action_interval_end(
    *,
    frame_indices: list[int],
    local_step: int,
    usable: int,
    frame_stride: int,
) -> int:
    if local_step + 1 < len(frame_indices):
        return int(frame_indices[local_step + 1])
    return min(int(usable), int(frame_indices[local_step]) + max(1, int(frame_stride)))


def aggregate_actions(actions: np.ndarray, *, start: int, end: int, mode: str) -> np.ndarray:
    start = int(start)
    end = max(start + 1, int(end))
    window = actions[start:end].astype(np.float32)
    if window.size == 0:
        window = actions[start : start + 1].astype(np.float32)
    if mode == "sample":
        return window[0].astype(np.float32)
    if mode == "mean":
        return np.mean(window, axis=0).astype(np.float32)
    if mode == "sum":
        return np.sum(window, axis=0).astype(np.float32)
    if mode == "last":
        return window[-1].astype(np.float32)
    raise ValueError(f"Unsupported action aggregation mode: {mode}")


def soar_reward(
    *,
    success: bool,
    local_step: int,
    step_count: int,
    mode: str,
    action: np.ndarray | None = None,
    action_penalty: float = 0.01,
) -> float:
    progress = float((local_step + 1) / max(1, step_count))
    terminal = local_step == step_count - 1
    action_cost = 0.0
    if action is not None:
        action_cost = float(action_penalty) * float(np.linalg.norm(np.asarray(action, dtype=np.float32)[:-1]))
    if not success:
        if mode == "signed_trajectory_success":
            return -1.0
        if mode == "signed_final_success":
            return -1.0 if terminal else 0.0
        if mode == "signed_linear_success_progress":
            return -progress
        if mode == "linear_success_progress_with_action_penalty":
            return -action_cost
        return 0.0
    if mode == "final_success":
        return 1.0 if local_step == step_count - 1 else 0.0
    if mode == "trajectory_success":
        return 1.0
    if mode == "linear_success_progress":
        return progress
    if mode == "signed_trajectory_success":
        return 1.0
    if mode == "signed_final_success":
        return 1.0 if terminal else 0.0
    if mode == "signed_linear_success_progress":
        return progress
    if mode == "linear_success_progress_with_action_penalty":
        return progress - action_cost
    raise ValueError(f"Unsupported reward mode: {mode}")


def extract_frames(video_path: Path, frames_dir: Path, *, width: int, height: int) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = frames_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"scale={int(width)}:{int(height)}",
        "-q:v",
        "3",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)
    return sorted(frames_dir.glob("frame_*.jpg"))


def load_frame(path: Path, *, frame_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((frame_size, frame_size))
        return np.asarray(image, dtype=np.uint8)


def make_feature_encoder(args: argparse.Namespace):
    if args.feature == "rgb_flat":
        return None
    from sensenova_drone.kairos_features import KairosVAEFeatureExtractor

    return KairosVAEFeatureExtractor(
        config_file=args.kairos_config,
        repo_root=REPO_ROOT,
        device=args.kairos_device,
        dtype=args.kairos_dtype,
        height=int(args.kairos_height),
        width=int(args.kairos_width),
        tiled=bool(args.kairos_tiled),
    )


def encode_frame_feature(
    frame_path: Path,
    *,
    args: argparse.Namespace,
    feature_encoder: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    if args.feature == "rgb_flat":
        frame = load_frame(frame_path, frame_size=int(args.frame_size))
        feature = frame.reshape(-1).astype(np.float32) / 255.0
        return feature, {
            "backend": "rgb_flat",
            "feature_dim": int(feature.shape[0]),
            "frame_size": int(args.frame_size),
        }

    if feature_encoder is None:
        raise RuntimeError(f"Feature encoder was not initialized for {args.feature}.")
    payload = feature_encoder.encode_image(frame_path)
    if args.feature == "kairos_vae_flat":
        feature_tensor = payload["latent"].reshape(-1)
    elif args.feature == "kairos_vae":
        feature_tensor = payload["image_features"]
    else:
        raise ValueError(f"Unsupported feature: {args.feature}")

    feature = feature_tensor.detach().cpu().numpy().astype(np.float32)
    metadata = dict(payload["metadata"])
    metadata.update(
        {
            "feature": args.feature,
            "feature_dim": int(feature.shape[0]),
            "feature_mean": float(np.mean(feature)),
            "feature_std": float(np.std(feature)),
        }
    )
    return feature, metadata


def frame_extract_size(args: argparse.Namespace) -> list[int]:
    if args.feature == "rgb_flat":
        return [int(args.frame_size), int(args.frame_size)]
    return [int(args.kairos_width), int(args.kairos_height)]


def kairos_feature_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if not str(args.feature).startswith("kairos_vae"):
        return None
    return {
        "config": str(args.kairos_config),
        "device": str(args.kairos_device),
        "dtype": str(args.kairos_dtype),
        "height": int(args.kairos_height),
        "width": int(args.kairos_width),
        "tiled": bool(args.kairos_tiled),
    }


def read_text(archive: zipfile.ZipFile, member: str) -> str:
    return archive.read(member).decode("utf-8", errors="replace")


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
