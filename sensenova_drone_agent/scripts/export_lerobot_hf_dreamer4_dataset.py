#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_IN = "sensenova_drone_agent/data/robotics/hf_action_sources/IPEC_COMMUNITY_bridge_orig_lerobot"
DEFAULT_OUT = "sensenova_drone_agent/data/robotics/hf_action_exports/bridge_orig_lerobot_dreamer4"
DEFAULT_CAMERA_PRIORITY = (
    "observation.images.exterior_image_1_left,"
    "observation.images.image,"
    "observation.images.image_0,"
    "observation.images.exterior_1_left,"
    "observation.images.wrist_image_left,"
    "observation.images.wrist_left"
)


@dataclass
class TaskBuffer:
    text: str
    episodes: list[int] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[np.ndarray] = field(default_factory=list)
    shard_frames: list[Any] = field(default_factory=list)
    shard_idx: int = 0
    next_episode_id: int = 0
    filenames: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export LeRobot-style per-episode parquet + MP4 datasets into the local Dreamer4 "
            "WMDataset layout. This supports compact OXE mirrors such as IPEC-COMMUNITY/*_lerobot."
        )
    )
    parser.add_argument("--input", default=DEFAULT_IN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--episodes-glob", default="data/**/episode_*.parquet")
    parser.add_argument("--max-trajectories", type=int, default=5000, help="<=0 exports all matching episodes.")
    parser.add_argument(
        "--paired-video-parquets-only",
        action="store_true",
        help="Only consider parquet episodes with a matching per-episode MP4 for the selected camera priority.",
    )
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--action-column", default="action")
    parser.add_argument("--reward-column", default="reward")
    parser.add_argument("--camera-priority", default=DEFAULT_CAMERA_PRIORITY)
    parser.add_argument("--task-mode", choices=["fixed", "task_index", "language"], default="fixed")
    parser.add_argument("--reward-mode", choices=["zero", "parquet", "success_last", "action_magnitude"], default="zero")
    parser.add_argument("--first-action-zero", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_path(args.input)
    out_dir = resolve_path(args.out)
    raw_dir = out_dir / "raw"
    frames_dir = out_dir / "frames"
    tasks_json = out_dir / "tasks.json"
    summary_path = out_dir / "summary.json"
    dataset_name = safe_task_name(args.dataset_name or root.name)
    camera_priority = parse_csv(args.camera_priority)
    all_parquet_paths = sorted(root.glob(args.episodes_glob))
    if args.paired_video_parquets_only:
        parquet_paths = [
            path for path in all_parquet_paths
            if find_episode_video(root, path, camera_priority)[0] is not None
        ]
    else:
        parquet_paths = list(all_parquet_paths)
    rng = np.random.default_rng(int(args.seed))
    if args.max_trajectories > 0 and len(parquet_paths) > int(args.max_trajectories):
        selected = sorted(rng.choice(len(parquet_paths), size=int(args.max_trajectories), replace=False).tolist())
        parquet_paths = [parquet_paths[int(idx)] for idx in selected]

    dry_payload = {
        "source": "lerobot_hf_episode_video",
        "input": str(root),
        "out": str(out_dir),
        "dataset_name": dataset_name,
        "episode_parquets_considered": int(len(all_parquet_paths)),
        "episode_parquets_selected": int(len(parquet_paths)),
        "paired_video_parquets_only": bool(args.paired_video_parquets_only),
        "max_trajectories": int(args.max_trajectories),
        "frame_size": int(args.frame_size),
        "frame_stride": int(args.frame_stride),
        "action_column": args.action_column,
        "reward_mode": args.reward_mode,
        "task_mode": args.task_mode,
        "camera_priority": camera_priority,
    }
    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({**dry_payload, "dry_run": True}, indent=2), encoding="utf-8")
        print(json.dumps({**dry_payload, "dry_run": True}, indent=2))
        return 0
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required; run this inside the Dreamer training image") from exc
    if not root.exists():
        raise FileNotFoundError(root)
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"{out_dir} exists; pass --force to rebuild")
        shutil.rmtree(out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    task_texts = load_task_texts(root)
    buffers: dict[str, TaskBuffer] = {}
    records = []
    skipped: dict[str, int] = defaultdict(int)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="lerobot_export_") as tmp:
        tmp_root = Path(tmp)
        for idx, parquet_path in enumerate(parquet_paths):
            try:
                table = pq.read_table(parquet_path)
                actions = table_column_to_numpy(table, args.action_column)
                if actions is None:
                    skipped["missing_action"] += 1
                    continue
                if actions.ndim == 1:
                    actions = actions[:, None]
                actions = actions.astype(np.float32)
                video_path, camera_name = find_episode_video(root, parquet_path, parse_csv(args.camera_priority))
                if video_path is None:
                    skipped["missing_video"] += 1
                    continue
                frame_tmp = tmp_root / f"frames_{idx:07d}"
                frame_tmp.mkdir(parents=True, exist_ok=True)
                frame_paths = extract_video_frames(video_path, frame_tmp, size=int(args.frame_size))
                usable = min(len(frame_paths), int(actions.shape[0]))
                if usable < 2:
                    skipped["too_short"] += 1
                    shutil.rmtree(frame_tmp, ignore_errors=True)
                    continue
                frame_indices = np.arange(0, usable, max(1, int(args.frame_stride)), dtype=np.int64)
                if frame_indices.shape[0] < 2:
                    skipped["too_short_after_stride"] += 1
                    shutil.rmtree(frame_tmp, ignore_errors=True)
                    continue
                task_name, task_text = resolve_task(args, table, dataset_name, task_texts)
                buffer = buffers.setdefault(task_name, TaskBuffer(text=task_text))
                episode_id = buffer.next_episode_id
                buffer.next_episode_id += 1
                sampled_actions = actions[frame_indices].astype(np.float32)
                if args.first_action_zero and sampled_actions.shape[0] > 0:
                    sampled_actions[0] = 0.0
                rewards = build_rewards(table, sampled_actions, frame_indices, mode=args.reward_mode, reward_column=args.reward_column)
                frames = load_frames(frame_paths, frame_indices, size=int(args.frame_size))
                append_task(
                    task_name=task_name,
                    buffer=buffer,
                    frames=frames,
                    actions=sampled_actions,
                    rewards=rewards,
                    episode_id=episode_id,
                    frames_dir=frames_dir,
                    shard_size=int(args.shard_size),
                )
                buffer.filenames.append(str(parquet_path.relative_to(root)))
                records.append(
                    {
                        "episode_parquet": str(parquet_path.relative_to(root)),
                        "video": str(video_path.relative_to(root)),
                        "camera": camera_name,
                        "task": task_name,
                        "episode": int(episode_id),
                        "raw_rows": int(actions.shape[0]),
                        "decoded_frames": int(len(frame_paths)),
                        "exported_steps": int(frame_indices.shape[0]),
                        "action_dim": int(sampled_actions.shape[-1]),
                    }
                )
                shutil.rmtree(frame_tmp, ignore_errors=True)
            except Exception as exc:
                skipped[f"error:{type(exc).__name__}"] += 1
                print(f"[warn] skipped {parquet_path}: {exc}", flush=True)
                continue
            if (idx + 1) % 100 == 0:
                print(f"[export] processed={idx + 1} exported={len(records)} skipped={dict(skipped)}", flush=True)

    flush_all(buffers, frames_dir=frames_dir, shard_size=int(args.shard_size))
    write_raw_outputs(buffers, raw_dir=raw_dir, tasks_json=tasks_json)
    summary = {
        **dry_payload,
        "dry_run": False,
        "elapsed_s": time.time() - started,
        "raw_dir": str(raw_dir),
        "frames_dir": str(frames_dir),
        "tasks_json": str(tasks_json),
        "exported_trajectories": int(len(records)),
        "skipped": dict(skipped),
        "tasks": summarize_buffers(buffers),
        "records_preview": records[:20],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def resolve_task(
    args: argparse.Namespace,
    table: Any,
    dataset_name: str,
    task_texts: dict[int, str],
) -> tuple[str, str]:
    if args.task_mode == "fixed":
        return dataset_name, f"LeRobot action-video source: {dataset_name}"
    task_index = 0
    if "task_index" in table.column_names and table.num_rows:
        task_index = int(table["task_index"][0].as_py())
    if args.task_mode == "task_index":
        task_text = task_texts.get(task_index, f"{dataset_name} task {task_index}")
        return safe_task_name(f"{dataset_name}_task_{task_index:05d}"), task_text
    for name in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        if name in table.column_names and table.num_rows:
            text = table[name][0].as_py()
            if text:
                return safe_task_name(f"{dataset_name}_{str(text)[:80]}"), str(text)
    task_text = task_texts.get(task_index, f"{dataset_name} task {task_index}")
    return safe_task_name(f"{dataset_name}_{task_text[:80]}"), task_text


def load_task_texts(root: Path) -> dict[int, str]:
    path = root / "meta" / "tasks.jsonl"
    if not path.exists():
        return {}
    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = row.get("task_index", row.get("index", row.get("id")))
            text = row.get("task", row.get("text", row.get("language_instruction", row.get("name"))))
            if idx is None or text is None:
                continue
            out[int(idx)] = str(text)
    return out


def find_episode_video(root: Path, parquet_path: Path, camera_priority: list[str]) -> tuple[Path | None, str | None]:
    rel = parquet_path.relative_to(root)
    chunk = parquet_path.parent.name
    stem = parquet_path.stem
    for camera in camera_priority:
        candidate = root / "videos" / chunk / camera / f"{stem}.mp4"
        if candidate.exists():
            return candidate, camera
    video_chunk = root / "videos" / chunk
    if video_chunk.exists():
        matches = sorted(video_chunk.glob(f"*/{stem}.mp4"))
        if matches:
            return matches[0], matches[0].parent.name
    # Some LeRobot mirrors store chunked videos as videos/<camera>/<chunk>/file-*.mp4.
    # Those need a timestamp-aware decoder; this exporter intentionally handles the
    # per-episode MP4 layout first.
    return None, None


def table_column_to_numpy(table: Any, column: str) -> np.ndarray | None:
    if column not in table.column_names:
        return None
    values = table[column].to_pylist()
    return np.asarray(values, dtype=np.float32)


def build_rewards(table: Any, actions: np.ndarray, frame_indices: np.ndarray, *, mode: str, reward_column: str) -> np.ndarray:
    if mode == "zero":
        return np.zeros(actions.shape[0], dtype=np.float32)
    if mode == "action_magnitude":
        return np.linalg.norm(actions, axis=-1).astype(np.float32)
    if mode == "parquet" and reward_column in table.column_names:
        rewards = np.asarray(table[reward_column].to_pylist(), dtype=np.float32)
        usable_indices = np.minimum(frame_indices, max(0, rewards.shape[0] - 1))
        return rewards[usable_indices].astype(np.float32)
    if mode == "success_last":
        out = np.zeros(actions.shape[0], dtype=np.float32)
        success = False
        if "is_episode_successful" in table.column_names and table.num_rows:
            success = bool(table["is_episode_successful"][0].as_py())
        if success and out.shape[0]:
            out[-1] = 1.0
        return out
    return np.zeros(actions.shape[0], dtype=np.float32)


def append_task(
    *,
    task_name: str,
    buffer: TaskBuffer,
    frames: Any,
    actions: np.ndarray,
    rewards: np.ndarray,
    episode_id: int,
    frames_dir: Path,
    shard_size: int,
) -> None:
    buffer.episodes.extend([int(episode_id)] * int(actions.shape[0]))
    buffer.actions.append(actions.astype(np.float32))
    buffer.rewards.append(rewards.astype(np.float32))
    buffer.shard_frames.append(frames)
    flush_if_needed(task_name, buffer, frames_dir=frames_dir, shard_size=shard_size)


def flush_if_needed(task_name: str, buffer: TaskBuffer, *, frames_dir: Path, shard_size: int) -> None:
    import torch

    total = sum(int(frames.shape[0]) for frames in buffer.shard_frames)
    if total < shard_size:
        return
    frames = torch.cat(buffer.shard_frames, dim=0)
    start = 0
    while start + shard_size <= int(frames.shape[0]):
        write_frame_shard(task_name, frames[start : start + shard_size], buffer, frames_dir=frames_dir)
        start += shard_size
    buffer.shard_frames = [frames[start:].contiguous()] if start < int(frames.shape[0]) else []


def flush_all(buffers: dict[str, TaskBuffer], *, frames_dir: Path, shard_size: int) -> None:
    import torch

    for task_name, buffer in buffers.items():
        if not buffer.shard_frames:
            continue
        frames = torch.cat(buffer.shard_frames, dim=0)
        for start in range(0, int(frames.shape[0]), int(shard_size)):
            write_frame_shard(task_name, frames[start : start + int(shard_size)], buffer, frames_dir=frames_dir)
        buffer.shard_frames = []


def write_frame_shard(task_name: str, frames: Any, buffer: TaskBuffer, *, frames_dir: Path) -> None:
    import torch

    task_dir = frames_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"frames": frames.contiguous()}, task_dir / f"frames_shard_{buffer.shard_idx:05d}.pt")
    buffer.shard_idx += 1


def write_raw_outputs(buffers: dict[str, TaskBuffer], *, raw_dir: Path, tasks_json: Path) -> None:
    import torch

    task_meta: dict[str, dict[str, Any]] = {}
    for task_name, buffer in sorted(buffers.items()):
        if not buffer.actions:
            continue
        actions = np.concatenate(buffer.actions, axis=0).astype(np.float32)
        rewards = np.concatenate(buffer.rewards, axis=0).astype(np.float32)
        raw = {
            "episode": torch.tensor(buffer.episodes, dtype=torch.int64),
            "action": torch.tensor(actions, dtype=torch.float32),
            "reward": torch.tensor(rewards, dtype=torch.float32),
        }
        torch.save(raw, raw_dir / f"{task_name}.pt")
        task_meta[task_name] = {
            "action_dim": int(actions.shape[-1]),
            "text": buffer.text,
        }
    tasks_json.write_text(json.dumps(task_meta, indent=2), encoding="utf-8")


def summarize_buffers(buffers: dict[str, TaskBuffer]) -> dict[str, Any]:
    out = {}
    for task_name, buffer in sorted(buffers.items()):
        if not buffer.actions:
            continue
        actions = np.concatenate(buffer.actions, axis=0).astype(np.float32)
        rewards = np.concatenate(buffer.rewards, axis=0).astype(np.float32)
        out[task_name] = {
            "steps": int(actions.shape[0]),
            "episodes": int(len(set(buffer.episodes))),
            "action_dim": int(actions.shape[-1]),
            "reward_sum": float(rewards.sum()),
            "action_abs_mean": float(np.mean(np.abs(actions))),
            "frame_shards": int(buffer.shard_idx),
            "example_files": buffer.filenames[:10],
        }
    return out


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


def load_frames(frame_paths: list[Path], frame_indices: np.ndarray, *, size: int) -> Any:
    import torch

    tensors = [load_png_chw(frame_paths[int(idx)], size=size) for idx in frame_indices]
    return torch.stack(tensors, dim=0)


def load_png_chw(path: Path, *, size: int) -> Any:
    import torch

    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def safe_task_name(value: str) -> str:
    out = []
    for ch in str(value).strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {"-", "_", " ", "/", ".", ":"}:
            out.append("_")
    name = "".join(out).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name[:160] or "lerobot"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
