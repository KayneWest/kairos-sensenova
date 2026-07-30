#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


DEFAULT_OUT = "sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_v1"
DEFAULT_PROCGEN_ENVS = "coinrun,bigfish,jumper"
DEFAULT_VIZDOOM_SCENARIOS = "basic"


@dataclass
class TaskWriter:
    task_name: str
    text: str
    raw_dir: Path
    frames_dir: Path
    frame_size: int
    shard_size: int
    action_dim: int
    action_labels: list[str]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    episodes: list[int] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    frame_buffer: list[torch.Tensor] = field(default_factory=list)
    preview_frames: list[np.ndarray] = field(default_factory=list)
    shard_index: int = 0
    total_frames: int = 0
    episode_count: int = 0
    return_by_episode: list[float] = field(default_factory=list)
    length_by_episode: list[int] = field(default_factory=list)

    def append(self, *, episode: int, frame_rgb: np.ndarray, action: np.ndarray, reward: float) -> None:
        frame_chw = resize_chw_uint8(frame_rgb, self.frame_size)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"Task {self.task_name} expected action shape {(self.action_dim,)}, got {action.shape}"
            )

        self.episodes.append(int(episode))
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.frame_buffer.append(torch.from_numpy(frame_chw))
        self.total_frames += 1

        if len(self.preview_frames) < 8:
            self.preview_frames.append(np.transpose(frame_chw, (1, 2, 0)))
        if len(self.frame_buffer) >= self.shard_size:
            self.flush_frame_shard()

    def record_episode(self, *, episode_return: float, length: int) -> None:
        self.episode_count += 1
        self.return_by_episode.append(float(episode_return))
        self.length_by_episode.append(int(length))

    def flush_frame_shard(self) -> None:
        if not self.frame_buffer:
            return
        task_dir = self.frames_dir / self.task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        frames = torch.stack(self.frame_buffer, dim=0).contiguous()
        out_path = task_dir / f"{self.task_name}_shard{self.shard_index:04d}.pt"
        torch.save({"frames": frames}, out_path)
        self.frame_buffer.clear()
        self.shard_index += 1

    def finalize(self) -> dict[str, Any]:
        self.flush_frame_shard()
        if not self.actions:
            raise RuntimeError(f"Task {self.task_name} has no collected frames.")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        raw = {
            "episode": torch.tensor(self.episodes, dtype=torch.int64),
            "action": torch.tensor(np.stack(self.actions).astype(np.float32), dtype=torch.float32),
            "reward": torch.tensor(np.asarray(self.rewards, dtype=np.float32), dtype=torch.float32),
        }
        torch.save(raw, self.raw_dir / f"{self.task_name}.pt")
        return {
            "source": self.source,
            "task_name": self.task_name,
            "text": self.text,
            "frames": int(self.total_frames),
            "episodes": int(self.episode_count),
            "action_dim": int(self.action_dim),
            "action_labels": self.action_labels,
            "reward_sum": float(np.sum(self.rewards)),
            "mean_return": safe_mean(self.return_by_episode),
            "mean_length": safe_mean(self.length_by_episode),
            "shards": int(self.shard_index),
            "metadata": self.metadata,
        }

    def write_preview(self, out_dir: Path) -> str | None:
        if not self.preview_frames:
            return None
        preview_dir = out_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        out_path = preview_dir / f"{self.task_name}.png"
        make_contact_sheet(self.preview_frames, out_path, self.task_name)
        return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect action-identifiable Procgen/ViZDoom rollouts into the native Dreamer4 WMDataset "
            "layout: raw/*.pt, frames/<task>/*_shard*.pt, tasks.json."
        )
    )
    parser.add_argument("--source", choices=["procgen", "vizdoom", "all"], default="all")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=180000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-seq-len", type=int, default=8)
    parser.add_argument(
        "--policy",
        choices=["random", "action_blocks"],
        default="random",
        help="Action sampling policy. action_blocks repeats one sampled action for --action-block-steps frames.",
    )
    parser.add_argument("--action-block-steps", type=int, default=8)

    parser.add_argument("--procgen-envs", default=DEFAULT_PROCGEN_ENVS)
    parser.add_argument("--procgen-distribution-mode", default="easy")
    parser.add_argument("--procgen-num-levels", type=int, default=0)
    parser.add_argument("--procgen-start-level", type=int, default=0)

    parser.add_argument("--vizdoom-scenarios", default=DEFAULT_VIZDOOM_SCENARIOS)
    parser.add_argument("--vizdoom-frame-repeat", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out)
    raw_dir = out_dir / "raw"
    frames_dir = out_dir / "frames"

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    selected_sources = ["procgen", "vizdoom"] if args.source == "all" else [args.source]
    writers: list[TaskWriter] = []

    started = time.time()
    if "procgen" in selected_sources:
        writers.extend(collect_procgen(args, raw_dir=raw_dir, frames_dir=frames_dir))
    if "vizdoom" in selected_sources:
        writers.extend(collect_vizdoom(args, raw_dir=raw_dir, frames_dir=frames_dir))

    task_summaries: dict[str, Any] = {}
    task_meta: dict[str, Any] = {}
    for writer in writers:
        summary = writer.finalize()
        preview_path = writer.write_preview(out_dir)
        if preview_path:
            summary["preview"] = preview_path
        task_summaries[writer.task_name] = summary
        task_meta[writer.task_name] = {
            "action_dim": int(writer.action_dim),
            "text": writer.text,
            "source": writer.source,
            "action_labels": writer.action_labels,
            **writer.metadata,
        }

    tasks_json = out_dir / "tasks.json"
    tasks_json.write_text(json.dumps(task_meta, indent=2, sort_keys=True), encoding="utf-8")

    validation = None
    if args.validate:
        validation = validate_wm_dataset(
            raw_dir=raw_dir,
            frames_dir=frames_dir,
            tasks_json=tasks_json,
            seq_len=int(args.validate_seq_len),
            frame_size=int(args.frame_size),
            action_dim=max((w.action_dim for w in writers), default=1),
            shard_size=int(args.shard_size),
        )

    summary = {
        "phase": "game_action_dreamer4_collection",
        "created_unix_s": started,
        "completed_unix_s": time.time(),
        "out": str(out_dir),
        "raw_dir": str(raw_dir),
        "frames_dir": str(frames_dir),
        "tasks_json": str(tasks_json),
        "source": args.source,
        "episodes_per_task": int(args.episodes),
        "max_steps": int(args.max_steps),
        "frame_size": int(args.frame_size),
        "shard_size": int(args.shard_size),
        "seed": int(args.seed),
        "tasks": task_summaries,
        "validation": validation,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary, indent=2))
    return 0


def collect_procgen(args: argparse.Namespace, *, raw_dir: Path, frames_dir: Path) -> list[TaskWriter]:
    from procgen import ProcgenEnv

    rng = np.random.default_rng(int(args.seed))
    writers: list[TaskWriter] = []
    env_names = split_csv(args.procgen_envs)
    for env_offset, env_name in enumerate(env_names):
        env = ProcgenEnv(
            num_envs=1,
            env_name=env_name,
            num_levels=int(args.procgen_num_levels),
            start_level=int(args.procgen_start_level),
            distribution_mode=str(args.procgen_distribution_mode),
        )
        action_dim = int(env.action_space.n)
        obs = env.reset()
        writer = TaskWriter(
            task_name=safe_task_name(f"procgen-{env_name}-random"),
            text=f"Procgen {env_name}: random discrete action-conditioned gameplay.",
            raw_dir=raw_dir,
            frames_dir=frames_dir,
            frame_size=int(args.frame_size),
            shard_size=int(args.shard_size),
            action_dim=action_dim,
            action_labels=[f"action_{idx}" for idx in range(action_dim)],
            source="procgen",
            metadata={
                "env_name": env_name,
                "distribution_mode": args.procgen_distribution_mode,
                "num_levels": int(args.procgen_num_levels),
                "start_level": int(args.procgen_start_level),
                "action_encoding": "one_hot_discrete",
                "policy": args.policy,
                "action_block_steps": int(args.action_block_steps),
            },
        )
        try:
            for episode_index in range(int(args.episodes)):
                episode_id = env_offset * 1_000_000 + episode_index
                frame = extract_rgb(obs)
                writer.append(
                    episode=episode_id,
                    frame_rgb=frame,
                    action=np.zeros(action_dim, dtype=np.float32),
                    reward=0.0,
                )
                episode_return = 0.0
                length = 0
                action_id = int(rng.integers(0, action_dim))
                for _ in range(int(args.max_steps)):
                    if args.policy == "random" or length % max(1, int(args.action_block_steps)) == 0:
                        action_id = int(rng.integers(0, action_dim))
                    obs, rewards, dones, _infos = env.step(np.asarray([action_id], dtype=np.int32))
                    reward = float(np.asarray(rewards).reshape(-1)[0])
                    done = bool(np.asarray(dones).reshape(-1)[0])
                    if done:
                        next_frame = frame.copy()
                    else:
                        next_frame = extract_rgb(obs)
                    writer.append(
                        episode=episode_id,
                        frame_rgb=next_frame,
                        action=one_hot(action_id, action_dim),
                        reward=reward,
                    )
                    episode_return += reward
                    length += 1
                    frame = next_frame
                    if done:
                        break
                writer.record_episode(episode_return=episode_return, length=length)
        finally:
            env.close()
        writers.append(writer)
    return writers


def collect_vizdoom(args: argparse.Namespace, *, raw_dir: Path, frames_dir: Path) -> list[TaskWriter]:
    from vizdoom import DoomGame, scenarios_path

    rng = np.random.default_rng(int(args.seed) + 100_000)
    writers: list[TaskWriter] = []
    for scenario_offset, scenario in enumerate(split_csv(args.vizdoom_scenarios)):
        game = DoomGame()
        game.load_config(str(Path(scenarios_path) / f"{scenario}.cfg"))
        game.set_window_visible(False)
        game.init()

        try:
            button_count = int(game.get_available_buttons_size())
            button_names = [str(button).split(".")[-1] for button in game.get_available_buttons()]
            doom_actions, action_labels = build_vizdoom_action_choices(button_count, button_names)
            action_dim = len(doom_actions)
            writer = TaskWriter(
                task_name=safe_task_name(f"vizdoom-{scenario}-random"),
                text=f"ViZDoom {scenario}: random first-person action-conditioned gameplay.",
                raw_dir=raw_dir,
                frames_dir=frames_dir,
                frame_size=int(args.frame_size),
                shard_size=int(args.shard_size),
                action_dim=action_dim,
                action_labels=action_labels,
                source="vizdoom",
                metadata={
                    "scenario": scenario,
                    "button_names": button_names,
                    "doom_button_count": button_count,
                    "frame_repeat": int(args.vizdoom_frame_repeat),
                    "action_encoding": "one_hot_action_choice",
                    "policy": args.policy,
                    "action_block_steps": int(args.action_block_steps),
                },
            )

            for episode_index in range(int(args.episodes)):
                episode_id = scenario_offset * 1_000_000 + episode_index
                game.new_episode()
                state = game.get_state()
                if state is None:
                    continue
                frame = extract_rgb(state.screen_buffer)
                writer.append(
                    episode=episode_id,
                    frame_rgb=frame,
                    action=np.zeros(action_dim, dtype=np.float32),
                    reward=0.0,
                )
                episode_return = 0.0
                length = 0
                action_id = int(rng.integers(0, action_dim))
                for _ in range(int(args.max_steps)):
                    if args.policy == "random" or length % max(1, int(args.action_block_steps)) == 0:
                        action_id = int(rng.integers(0, action_dim))
                    reward = float(game.make_action(doom_actions[action_id], int(args.vizdoom_frame_repeat)))
                    done = bool(game.is_episode_finished())
                    if done:
                        next_frame = frame.copy()
                    else:
                        state = game.get_state()
                        next_frame = frame.copy() if state is None else extract_rgb(state.screen_buffer)
                    writer.append(
                        episode=episode_id,
                        frame_rgb=next_frame,
                        action=one_hot(action_id, action_dim),
                        reward=reward,
                    )
                    episode_return += reward
                    length += 1
                    frame = next_frame
                    if done:
                        break
                writer.record_episode(episode_return=episode_return, length=length)
        finally:
            game.close()
        writers.append(writer)
    return writers


def build_vizdoom_action_choices(button_count: int, button_names: list[str]) -> tuple[list[list[int]], list[str]]:
    actions: list[list[int]] = [[0 for _ in range(button_count)]]
    labels = ["noop"]
    for idx in range(button_count):
        action = [0 for _ in range(button_count)]
        action[idx] = 1
        actions.append(action)
        label = button_names[idx] if idx < len(button_names) else f"button_{idx}"
        labels.append(label)
    return actions, labels


def validate_wm_dataset(
    *,
    raw_dir: Path,
    frames_dir: Path,
    tasks_json: Path,
    seq_len: int,
    frame_size: int,
    action_dim: int,
    shard_size: int,
) -> dict[str, Any]:
    dreamer4_dir = REPO_ROOT / "dreamer4" / "dreamer4"
    if str(dreamer4_dir) not in sys.path:
        sys.path.insert(0, str(dreamer4_dir))
    from wm_dataset import WMDataset  # noqa: E402

    dataset = WMDataset(
        data_dir=str(raw_dir),
        frames_dir=str(frames_dir),
        seq_len=int(seq_len),
        img_size=int(frame_size),
        action_dim=int(action_dim),
        shard_size=int(shard_size),
        tasks_json=str(tasks_json),
        verbose=False,
        strict_tasks=False,
    )
    sample = dataset[0]
    return {
        "success": True,
        "num_windows": int(len(dataset)),
        "num_tasks": int(dataset.num_tasks),
        "tasks": list(dataset.tasks),
        "sample_obs_shape": list(sample["obs"].shape),
        "sample_act_shape": list(sample["act"].shape),
        "sample_rew_shape": list(sample["rew"].shape),
        "action_dim": int(action_dim),
        "seq_len": int(seq_len),
    }


def extract_rgb(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        frame = obs.get("rgb")
    else:
        frame = obs
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"Expected image-like observation, got shape {frame.shape}")
    if frame.shape[0] in {1, 3, 4} and frame.shape[-1] not in {1, 3, 4}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return np.clip(frame, 0, 255).astype(np.uint8)


def resize_chw_uint8(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(extract_rgb(frame_rgb)).resize((int(size), int(size)))
    arr = np.asarray(image, dtype=np.uint8)
    return np.transpose(arr, (2, 0, 1)).copy()


def one_hot(index: int, dim: int) -> np.ndarray:
    out = np.zeros(int(dim), dtype=np.float32)
    out[int(index)] = 1.0
    return out


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def safe_task_name(value: str) -> str:
    slug = []
    for ch in value.strip().lower():
        if ch.isalnum():
            slug.append(ch)
        elif slug and slug[-1] != "-":
            slug.append("-")
    out = "".join(slug).strip("-")
    return out or "game-action-task"


def safe_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def make_contact_sheet(frames: list[np.ndarray], out_path: Path, label: str) -> None:
    if not frames:
        return
    cell = 128
    label_h = 24
    pil_frames = [Image.fromarray(frame).resize((cell, cell)) for frame in frames]
    sheet = Image.new("RGB", (cell * len(pil_frames), cell + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), label[:100], fill=(0, 0, 0))
    for idx, frame in enumerate(pil_frames):
        sheet.paste(frame, (idx * cell, label_h))
    sheet.save(out_path)


def write_report(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Game Action Dreamer4 Dataset",
        "",
        "This dataset is intended to create action-identifiable pixel/action sequences for native Dreamer4-style dynamics training.",
        "",
        "## Layout",
        "",
        f"- Raw transition tensors: `{summary['raw_dir']}`",
        f"- Frame shards: `{summary['frames_dir']}`",
        f"- Task metadata: `{summary['tasks_json']}`",
        "",
        "## Tasks",
        "",
    ]
    for task, info in sorted(summary["tasks"].items()):
        lines.extend(
            [
                f"- `{task}`: source=`{info['source']}`, frames=`{info['frames']}`, episodes=`{info['episodes']}`, action_dim=`{info['action_dim']}`, mean_return=`{info['mean_return']}`",
            ]
        )
    if summary.get("validation"):
        validation = summary["validation"]
        lines.extend(
            [
                "",
                "## Validation",
                "",
                f"- WMDataset load success: `{validation['success']}`",
                f"- Valid windows: `{validation['num_windows']}`",
                f"- Sample obs shape: `{validation['sample_obs_shape']}`",
                f"- Sample action shape: `{validation['sample_act_shape']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Training Hook",
            "",
            "Use `raw/` as `--data_dirs`, `frames/` as `--frame_dirs`, and `tasks.json` as `--tasks_json` for native Dreamer4 tokenizer/dynamics experiments.",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
