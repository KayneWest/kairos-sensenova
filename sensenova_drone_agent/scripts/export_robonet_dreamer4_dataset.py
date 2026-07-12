#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_TFDS_NAME = "robonet/robonet_sample_64"
DEFAULT_TFDS_DIR = "sensenova_drone_agent/data/robotics/robonet/tfds"
DEFAULT_TAR = "sensenova_drone_agent/data/robotics/robonet/raw/robonet_sampler.tar.gz"
DEFAULT_OUT = "sensenova_drone_agent/data/robotics/robonet/dreamer4_robonet_sample_64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the TFDS RoboNet sample into the local Dreamer4 WMDataset layout. "
            "The output can be passed to train_native_dreamer4_imagination.py as a raw/frame source."
        )
    )
    parser.add_argument("--tfds-name", default=DEFAULT_TFDS_NAME)
    parser.add_argument("--tfds-data-dir", default=DEFAULT_TFDS_DIR)
    parser.add_argument("--source", choices=["tfds", "tar"], default="tar")
    parser.add_argument("--tar", default=DEFAULT_TAR)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=700, help="<=0 means all examples in the split.")
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-name", default="robonet_sample_64")
    parser.add_argument(
        "--task-mode",
        choices=["fixed", "filename_parent", "filename_prefix", "robot_name"],
        default="fixed",
        help="How to group trajectories into WMDataset tasks.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["zero", "action_magnitude"],
        default="zero",
        help="RoboNet has no task reward in TFDS; this controls placeholder rewards.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source == "tar":
        return export_from_tar(args)

    try:
        import tensorflow_datasets as tfds
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tensorflow_datasets is required. Install tensorflow-datasets and tensorflow, "
            "or run this inside a data-prep image."
        ) from exc

    tfds_dir = resolve_path(args.tfds_data_dir)
    out_dir = resolve_path(args.out)
    raw_dir = out_dir / "raw"
    frames_dir = out_dir / "frames"
    tasks_json = out_dir / "tasks.json"
    summary_path = out_dir / "summary.json"

    builder = tfds.builder(args.tfds_name, data_dir=str(tfds_dir))
    if args.download_if_missing:
        builder.download_and_prepare()
    info = builder.info
    dry_payload = {
        "tfds_name": args.tfds_name,
        "tfds_data_dir": str(tfds_dir),
        "out": str(out_dir),
        "split": args.split,
        "version": str(info.version),
        "features": repr(info.features),
        "max_trajectories": int(args.max_trajectories),
        "frame_size": int(args.frame_size),
        "frame_stride": int(args.frame_stride),
        "task_mode": args.task_mode,
        "reward_mode": args.reward_mode,
    }
    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({**dry_payload, "dry_run": True}, indent=2), encoding="utf-8")
        print(json.dumps({**dry_payload, "dry_run": True}, indent=2))
        return 0

    if out_dir.exists():
        shutil.rmtree(out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    dataset = builder.as_dataset(split=args.split, shuffle_files=False)
    if args.max_trajectories > 0:
        dataset = dataset.take(int(args.max_trajectories))
    examples = tfds.as_numpy(dataset)

    rng = np.random.default_rng(int(args.seed))
    task_payloads: dict[str, dict[str, Any]] = {}
    task_frame_buffers: dict[str, list[torch.Tensor]] = defaultdict(list)
    task_episode_offsets: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []

    count = 0
    for example_idx, example in enumerate(examples):
        video = np.asarray(example["video"])
        actions = np.asarray(example["actions"], dtype=np.float32)
        filename = decode_text(example.get("filename", f"example_{example_idx:06d}"))
        if video.ndim != 4:
            continue
        if actions.ndim == 1:
            actions = actions[:, None]
        usable = min(int(video.shape[0]), max(0, int(actions.shape[0])))
        if usable < 2:
            continue
        task_name = task_name_for(filename, args)
        payload = task_payloads.setdefault(
            task_name,
            {
                "episode": [],
                "action": [],
                "reward": [],
                "text": f"RoboNet robot-object interaction: {task_name}",
                "filenames": [],
            },
        )
        episode_id = task_episode_offsets[task_name]
        task_episode_offsets[task_name] += 1
        payload["filenames"].append(filename)

        frame_indices = np.arange(0, usable, max(1, int(args.frame_stride)), dtype=np.int64)
        if frame_indices.shape[0] < 2:
            continue
        frames_tensor = torch.stack(
            [resize_frame_to_chw(video[int(idx)], size=int(args.frame_size)) for idx in frame_indices],
            dim=0,
        )
        task_frame_buffers[task_name].append(frames_tensor)
        action_rows = build_action_rows(actions, frame_indices, usable)
        rewards = build_rewards(action_rows, mode=args.reward_mode)

        payload["episode"].extend([episode_id] * int(frame_indices.shape[0]))
        payload["action"].extend(action_rows)
        payload["reward"].extend(rewards)
        records.append(
            {
                "example_idx": int(example_idx),
                "filename": filename,
                "task": task_name,
                "episode": int(episode_id),
                "raw_frames": int(video.shape[0]),
                "raw_actions": int(actions.shape[0]),
                "exported_steps": int(frame_indices.shape[0]),
                "action_dim": int(action_rows.shape[-1]),
            }
        )
        count += 1

    if not task_payloads:
        raise RuntimeError("No RoboNet trajectories exported.")

    task_meta: dict[str, dict[str, Any]] = {}
    task_summaries: dict[str, Any] = {}
    for task_name, payload in sorted(task_payloads.items()):
        actions_np = np.stack(payload["action"]).astype(np.float32)
        task_raw = {
            "episode": torch.tensor(payload["episode"], dtype=torch.int64),
            "action": torch.tensor(actions_np, dtype=torch.float32),
            "reward": torch.tensor(payload["reward"], dtype=torch.float32),
        }
        torch.save(task_raw, raw_dir / f"{task_name}.pt")
        frames = torch.cat(task_frame_buffers[task_name], dim=0)
        write_frame_shards(task_name, frames, frames_dir=frames_dir, shard_size=int(args.shard_size))
        task_meta[task_name] = {
            "action_dim": int(actions_np.shape[-1]),
            "text": payload["text"],
        }
        task_summaries[task_name] = {
            "steps": int(actions_np.shape[0]),
            "episodes": int(len(set(payload["episode"]))),
            "action_dim": int(actions_np.shape[-1]),
            "reward_sum": float(np.sum(payload["reward"])),
            "action_abs_mean": float(np.mean(np.abs(actions_np))),
            "example_filenames": payload["filenames"][:10],
        }

    tasks_json.write_text(json.dumps(task_meta, indent=2), encoding="utf-8")
    summary = {
        **dry_payload,
        "dry_run": False,
        "raw_dir": str(raw_dir),
        "frames_dir": str(frames_dir),
        "tasks_json": str(tasks_json),
        "exported_trajectories": int(count),
        "tasks": task_summaries,
        "records_preview": records[:20],
        "seed_probe": int(rng.integers(0, 2**31 - 1)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def export_from_tar(args: argparse.Namespace) -> int:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for --source tar") from exc

    tar_path = resolve_path(args.tar)
    out_dir = resolve_path(args.out)
    raw_dir = out_dir / "raw"
    frames_dir = out_dir / "frames"
    tasks_json = out_dir / "tasks.json"
    summary_path = out_dir / "summary.json"

    with tarfile.open(tar_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(".hdf5")]
    members = sorted(members, key=lambda item: item.name)
    archive_trajectory_count = len(members)
    rng = np.random.default_rng(int(args.seed))
    if args.max_trajectories > 0 and len(members) > int(args.max_trajectories):
        order = rng.permutation(len(members))[: int(args.max_trajectories)]
        members = [members[int(idx)] for idx in order]

    dry_payload = {
        "source": "tar",
        "tar": str(tar_path),
        "out": str(out_dir),
        "archive_trajectories": int(archive_trajectory_count),
        "selected_trajectories": int(len(members)),
        "max_trajectories": int(args.max_trajectories),
        "frame_size": int(args.frame_size),
        "frame_stride": int(args.frame_stride),
        "task_mode": args.task_mode,
        "reward_mode": args.reward_mode,
    }
    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({**dry_payload, "dry_run": True}, indent=2), encoding="utf-8")
        print(json.dumps({**dry_payload, "dry_run": True}, indent=2))
        return 0

    if out_dir.exists():
        shutil.rmtree(out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    task_payloads: dict[str, dict[str, Any]] = {}
    task_frame_buffers: dict[str, list[torch.Tensor]] = defaultdict(list)
    task_episode_offsets: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []

    with tarfile.open(tar_path, "r:gz") as archive, tempfile.TemporaryDirectory(prefix="robonet_tar_export_") as tmp:
        tmp_root = Path(tmp)
        for selected_idx, member in enumerate(members):
            archive.extract(member, tmp_root)
            h5_path = tmp_root / member.name
            filename = Path(member.name).name
            with h5py.File(h5_path, "r") as handle:
                video_bytes = handle["env"]["cam0_video"]["frames"][:].tobytes()
                raw_actions = handle["policy"]["actions"][:].astype(np.float32)
            video_path = tmp_root / f"video_{selected_idx:06d}.mp4"
            frames_tmp = tmp_root / f"frames_{selected_idx:06d}"
            frames_tmp.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(video_bytes)
            frame_paths = extract_video_frames(video_path, frames_tmp, size=int(args.frame_size))
            if not frame_paths:
                continue
            actions = pad_actions(raw_actions, width=5)
            usable = min(len(frame_paths), int(actions.shape[0]) + 1)
            if usable < 2:
                continue
            task_name = task_name_for(filename, args)
            payload = task_payloads.setdefault(
                task_name,
                {
                    "episode": [],
                    "action": [],
                    "reward": [],
                    "text": f"RoboNet robot-object interaction: {task_name}",
                    "filenames": [],
                },
            )
            episode_id = task_episode_offsets[task_name]
            task_episode_offsets[task_name] += 1
            payload["filenames"].append(filename)

            frame_indices = np.arange(0, usable, max(1, int(args.frame_stride)), dtype=np.int64)
            if frame_indices.shape[0] < 2:
                continue
            frames_tensor = torch.stack([load_png_chw(frame_paths[int(idx)], size=int(args.frame_size)) for idx in frame_indices], dim=0)
            action_rows = align_transition_actions_to_exported_frames(actions, frame_indices)
            rewards = build_rewards(action_rows, mode=args.reward_mode)
            task_frame_buffers[task_name].append(frames_tensor)
            payload["episode"].extend([episode_id] * int(frame_indices.shape[0]))
            payload["action"].extend(action_rows)
            payload["reward"].extend(rewards)
            records.append(
                {
                    "selected_idx": int(selected_idx),
                    "filename": filename,
                    "task": task_name,
                    "episode": int(episode_id),
                    "decoded_frames": int(len(frame_paths)),
                    "raw_actions": int(raw_actions.shape[0]),
                    "exported_steps": int(frame_indices.shape[0]),
                    "action_dim": int(action_rows.shape[-1]),
                }
            )
            h5_path.unlink(missing_ok=True)
            video_path.unlink(missing_ok=True)
            shutil.rmtree(frames_tmp, ignore_errors=True)

    write_wm_dataset_outputs(
        raw_dir=raw_dir,
        frames_dir=frames_dir,
        tasks_json=tasks_json,
        task_payloads=task_payloads,
        task_frame_buffers=task_frame_buffers,
        shard_size=int(args.shard_size),
    )
    summary = {
        **dry_payload,
        "dry_run": False,
        "raw_dir": str(raw_dir),
        "frames_dir": str(frames_dir),
        "tasks_json": str(tasks_json),
        "exported_trajectories": int(len(records)),
        "tasks": summarize_task_payloads(task_payloads),
        "records_preview": records[:20],
        "seed_probe": int(rng.integers(0, 2**31 - 1)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def write_wm_dataset_outputs(
    *,
    raw_dir: Path,
    frames_dir: Path,
    tasks_json: Path,
    task_payloads: dict[str, dict[str, Any]],
    task_frame_buffers: dict[str, list[torch.Tensor]],
    shard_size: int,
) -> None:
    if not task_payloads:
        raise RuntimeError("No RoboNet trajectories exported.")
    task_meta: dict[str, dict[str, Any]] = {}
    for task_name, payload in sorted(task_payloads.items()):
        actions_np = np.stack(payload["action"]).astype(np.float32)
        task_raw = {
            "episode": torch.tensor(payload["episode"], dtype=torch.int64),
            "action": torch.tensor(actions_np, dtype=torch.float32),
            "reward": torch.tensor(payload["reward"], dtype=torch.float32),
        }
        torch.save(task_raw, raw_dir / f"{task_name}.pt")
        frames = torch.cat(task_frame_buffers[task_name], dim=0)
        write_frame_shards(task_name, frames, frames_dir=frames_dir, shard_size=shard_size)
        task_meta[task_name] = {
            "action_dim": int(actions_np.shape[-1]),
            "text": payload["text"],
        }
    tasks_json.write_text(json.dumps(task_meta, indent=2), encoding="utf-8")


def summarize_task_payloads(task_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    task_summaries: dict[str, Any] = {}
    for task_name, payload in sorted(task_payloads.items()):
        actions_np = np.stack(payload["action"]).astype(np.float32)
        task_summaries[task_name] = {
            "steps": int(actions_np.shape[0]),
            "episodes": int(len(set(payload["episode"]))),
            "action_dim": int(actions_np.shape[-1]),
            "reward_sum": float(np.sum(payload["reward"])),
            "action_abs_mean": float(np.mean(np.abs(actions_np))),
            "example_filenames": payload["filenames"][:10],
        }
    return task_summaries


def build_action_rows(actions: np.ndarray, frame_indices: np.ndarray, usable: int) -> np.ndarray:
    rows = []
    for idx in frame_indices:
        action_idx = min(int(idx), int(actions.shape[0]) - 1, usable - 1)
        rows.append(actions[action_idx])
    return np.stack(rows).astype(np.float32)


def align_transition_actions_to_exported_frames(actions: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
    rows = []
    for local_idx, frame_idx in enumerate(frame_indices):
        if local_idx == 0:
            rows.append(np.zeros(actions.shape[-1], dtype=np.float32))
            continue
        prev_frame_idx = int(frame_indices[local_idx - 1])
        current_frame_idx = int(frame_idx)
        start = min(prev_frame_idx, int(actions.shape[0]) - 1)
        end = min(max(prev_frame_idx + 1, current_frame_idx), int(actions.shape[0]))
        rows.append(actions[start:end].mean(axis=0).astype(np.float32))
    return np.stack(rows).astype(np.float32)


def pad_actions(actions: np.ndarray, *, width: int) -> np.ndarray:
    if actions.ndim == 1:
        actions = actions[:, None]
    actions = actions.astype(np.float32)
    if actions.shape[-1] >= width:
        return actions[:, :width]
    out = np.zeros((actions.shape[0], width), dtype=np.float32)
    out[:, : actions.shape[-1]] = actions
    return out


def build_rewards(actions: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "zero":
        return np.zeros(actions.shape[0], dtype=np.float32)
    if mode == "action_magnitude":
        return np.linalg.norm(actions, axis=-1).astype(np.float32)
    raise ValueError(f"unknown reward mode: {mode}")


def write_frame_shards(task_name: str, frames: torch.Tensor, *, frames_dir: Path, shard_size: int) -> None:
    task_dir = frames_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    for shard_idx, start in enumerate(range(0, int(frames.shape[0]), int(shard_size))):
        shard = frames[start : start + int(shard_size)].contiguous()
        torch.save({"frames": shard}, task_dir / f"frames_shard_{shard_idx:05d}.pt")


def extract_video_frames(video_path: Path, frames_dir: Path, *, size: int) -> list[Path]:
    pattern = frames_dir / "frame_%06d.png"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={size}:{size}",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)
    return sorted(frames_dir.glob("frame_*.png"))


def load_png_chw(path: Path, *, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def resize_frame_to_chw(frame: np.ndarray, *, size: int) -> torch.Tensor:
    if frame.ndim != 3:
        raise ValueError(f"expected HWC frame, got shape={frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    image = Image.fromarray(frame)
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def task_name_for(filename: str, args: argparse.Namespace) -> str:
    if args.task_mode == "fixed":
        return safe_task_name(args.task_name)
    path = Path(filename)
    if args.task_mode == "filename_parent":
        parent = path.parent.name if path.parent.name else args.task_name
        return safe_task_name(parent)
    if args.task_mode == "filename_prefix":
        stem = path.stem or filename
        return safe_task_name(stem.split("_", 1)[0] or args.task_name)
    if args.task_mode == "robot_name":
        stem = path.stem or filename
        return safe_task_name(stem.rsplit("_traj", 1)[0] or args.task_name)
    raise ValueError(f"unknown task mode: {args.task_mode}")


def safe_task_name(value: str) -> str:
    out = []
    for ch in value.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {"-", "_", " ", "/", "."}:
            out.append("_")
    name = "".join(out).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name or "robonet"


def decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_text(value.item())
    return str(value)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
