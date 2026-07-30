#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_soar_sequence_cache import (  # noqa: E402
    DEFAULT_ZIP,
    SoarTrajectory,
    aggregate_actions,
    discover_trajectories,
    extract_frames,
    read_text,
    select_task_balanced_trajectories,
    select_trajectories,
    soar_reward,
)


DEFAULT_OUT = "sensenova_drone_agent/data/robotics/soar/dreamer4_soar_small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a subset of the SOAR numpy ZIP into the unofficial Dreamer4 WMDataset layout: "
            "raw task .pt files plus 128x128 frame shards."
        )
    )
    parser.add_argument("--zip", default=DEFAULT_ZIP)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-trajectories", type=int, default=16)
    parser.add_argument("--max-steps-per-trajectory", type=int, default=80)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-outcomes", default="success,failure")
    parser.add_argument(
        "--selection-mode",
        choices=["outcome_balanced", "task_balanced"],
        default="task_balanced",
    )
    parser.add_argument("--target-task-count", type=int, default=4)
    parser.add_argument("--min-trajectories-per-task", type=int, default=2)
    parser.add_argument("--max-trajectories-per-task", type=int, default=4)
    parser.add_argument("--require-both-outcomes-per-task", action="store_true")
    parser.add_argument(
        "--action-aggregation",
        choices=["sample", "mean", "sum", "last"],
        default="sum",
    )
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
        default="trajectory_success",
    )
    parser.add_argument("--reward-action-penalty", type=float, default=0.01)
    parser.add_argument("--task-name-mode", choices=["language", "scene"], default="language")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = resolve_path(args.zip)
    out_dir = resolve_path(args.out)
    raw_dir = out_dir / "raw"
    frames_dir = out_dir / "frames"
    tasks_json = out_dir / "tasks.json"
    summary_path = out_dir / "summary.json"

    trajectories = discover_trajectories(zip_path)
    include_outcomes = {item.strip() for item in args.include_outcomes.split(",") if item.strip()}
    if args.selection_mode == "task_balanced":
        selected, selection = select_task_balanced_trajectories(
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
        selection = {"mode": "outcome_balanced"}

    if args.dry_run:
        payload = {
            "zip": str(zip_path),
            "out": str(out_dir),
            "trajectory_count": len(trajectories),
            "selected_count": len(selected),
            "selection": selection,
            "selected_preview": [traj.path for traj in selected[:20]],
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    if out_dir.exists():
        shutil.rmtree(out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    records, task_payloads, task_frame_buffers = export_selected(zip_path, selected, args)
    task_summaries: dict[str, Any] = {}
    for task_name, payload in sorted(task_payloads.items()):
        task_raw = {
            "episode": torch.tensor(payload["episode"], dtype=torch.int64),
            "action": torch.tensor(np.stack(payload["action"]).astype(np.float32)),
            "reward": torch.tensor(payload["reward"], dtype=torch.float32),
        }
        torch.save(task_raw, raw_dir / f"{task_name}.pt")
        write_frame_shards(
            task_name=task_name,
            frames=torch.cat(task_frame_buffers[task_name], dim=0),
            frames_dir=frames_dir,
            shard_size=int(args.shard_size),
        )
        task_summaries[task_name] = {
            "steps": int(len(payload["reward"])),
            "episodes": int(len(set(payload["episode"]))),
            "reward_sum": float(np.sum(payload["reward"])),
            "success_episodes": int(payload["success_episodes"]),
            "failure_episodes": int(payload["failure_episodes"]),
        }

    task_meta = {
        task_name: {
            "action_dim": 7,
            "text": payload["text"],
        }
        for task_name, payload in sorted(task_payloads.items())
    }
    tasks_json.write_text(json.dumps(task_meta, indent=2), encoding="utf-8")
    summary = {
        "zip": str(zip_path),
        "out": str(out_dir),
        "raw_dir": str(raw_dir),
        "frames_dir": str(frames_dir),
        "tasks_json": str(tasks_json),
        "trajectory_count": len(trajectories),
        "selected_count": len(selected),
        "selection": selection,
        "frame_size": int(args.frame_size),
        "frame_stride": int(args.frame_stride),
        "shard_size": int(args.shard_size),
        "action_aggregation": args.action_aggregation,
        "reward_mode": args.reward_mode,
        "tasks": task_summaries,
        "records_preview": records[:20],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def export_selected(
    zip_path: Path,
    selected: list[SoarTrajectory],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[torch.Tensor]]]:
    task_payloads: dict[str, dict[str, Any]] = {}
    task_frame_buffers: dict[str, list[torch.Tensor]] = defaultdict(list)
    task_episode_offsets: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="soar_dreamer4_"))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for selected_idx, traj in enumerate(selected):
                language = read_text(archive, f"{traj.path}/language_task.txt").strip()
                task_name = safe_task_name(language if args.task_name_mode == "language" else traj.scene)
                payload = task_payloads.setdefault(
                    task_name,
                    {
                        "episode": [],
                        "action": [],
                        "reward": [],
                        "text": language,
                        "success_episodes": 0,
                        "failure_episodes": 0,
                    },
                )
                episode_id = task_episode_offsets[task_name]
                task_episode_offsets[task_name] += 1

                traj_tmp = tmp_root / f"traj_{selected_idx:05d}"
                traj_tmp.mkdir(parents=True, exist_ok=True)
                actions = np.load(io.BytesIO(archive.read(f"{traj.path}/actions.npy"))).astype(np.float32)
                success = read_text(archive, f"{traj.path}/success.txt").strip().lower() == "true"
                payload["success_episodes"] += int(success)
                payload["failure_episodes"] += int(not success)

                video_path = traj_tmp / "trajectory.mp4"
                video_path.write_bytes(archive.read(f"{traj.path}/trajectory.mp4"))
                extracted = extract_frames(
                    video_path,
                    traj_tmp / "frames",
                    width=int(args.frame_size),
                    height=int(args.frame_size),
                )
                usable = min(len(extracted), int(actions.shape[0]), int(args.max_steps_per_trajectory))
                frame_indices = list(range(0, usable, max(1, int(args.frame_stride))))
                if len(frame_indices) < 2:
                    continue

                frames_tensor = torch.stack([load_frame_chw(extracted[idx], size=int(args.frame_size)) for idx in frame_indices], dim=0)
                task_frame_buffers[task_name].append(frames_tensor)

                for local_step, frame_idx in enumerate(frame_indices):
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
                    payload["episode"].append(episode_id)
                    payload["action"].append(action)
                    payload["reward"].append(
                        soar_reward(
                            success=success,
                            local_step=local_step,
                            step_count=len(frame_indices),
                            mode=args.reward_mode,
                            action=action,
                            action_penalty=float(args.reward_action_penalty),
                        )
                    )

                records.append(
                    {
                        "trajectory": traj.path,
                        "task": task_name,
                        "language": language,
                        "episode": int(episode_id),
                        "success": bool(success),
                        "raw_actions": list(actions.shape),
                        "raw_frames": len(extracted),
                        "exported_steps": len(frame_indices),
                    }
                )
    finally:
        if args.keep_temp:
            print(f"Kept temp dir: {tmp_root}", file=sys.stderr)
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)
    if not records:
        raise RuntimeError("No SOAR trajectories were exported.")
    return records, task_payloads, task_frame_buffers


def write_frame_shards(*, task_name: str, frames: torch.Tensor, frames_dir: Path, shard_size: int) -> None:
    task_dir = frames_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    shard_size = max(1, int(shard_size))
    for shard_idx, start in enumerate(range(0, int(frames.shape[0]), shard_size)):
        shard = frames[start : start + shard_size].contiguous()
        torch.save({"frames": shard}, task_dir / f"{task_name}_shard{shard_idx:04d}.pt")


def load_frame_chw(path: Path, *, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size))
        arr = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def action_interval_end(*, frame_indices: list[int], local_step: int, usable: int, frame_stride: int) -> int:
    if local_step + 1 < len(frame_indices):
        return int(frame_indices[local_step + 1])
    return min(int(usable), int(frame_indices[local_step]) + max(1, int(frame_stride)))


def safe_task_name(value: str) -> str:
    slug = []
    for ch in value.strip().lower():
        if ch.isalnum():
            slug.append(ch)
        elif slug and slug[-1] != "-":
            slug.append("-")
    out = "".join(slug).strip("-")
    return out or "soar-task"


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
