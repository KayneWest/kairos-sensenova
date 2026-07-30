#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal Procgen random-policy visual benchmark.")
    parser.add_argument("--out-dir", default="sensenova_drone_agent/output/procgen_coinrun_random_v1")
    parser.add_argument("--env-name", default="coinrun")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=130000)
    parser.add_argument("--num-levels", type=int, default=0)
    parser.add_argument("--start-level", type=int, default=0)
    parser.add_argument("--distribution-mode", default="easy")
    parser.add_argument("--trace-frames", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from procgen import ProcgenEnv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "procgen is not installed. Build with "
            "`./sensenova_drone_agent/scripts/build_procgen_benchmark_image.sh` "
            "and run via `./sensenova_drone_agent/scripts/run_procgen_benchmark.sh`."
        ) from exc

    rng = np.random.default_rng(args.seed)
    env = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_name,
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
    )
    obs = env.reset()
    action_n = int(env.action_space.n)

    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
    completed = []
    trace_frames = []

    steps_executed = 0
    termination_reason = "max_steps"
    for step in range(args.max_steps):
        steps_executed = step + 1
        if len(trace_frames) < args.trace_frames:
            trace_frames.append(extract_rgb(obs, env_index=0))
        actions = rng.integers(0, action_n, size=(args.num_envs,), dtype=np.int64)
        obs, rewards, dones, infos = env.step(actions)
        del infos
        episode_returns += np.asarray(rewards, dtype=np.float64)
        episode_lengths += 1

        for idx, done in enumerate(np.asarray(dones).astype(bool)):
            if done:
                completed.append(
                    {
                        "episode": len(completed),
                        "env_index": idx,
                        "return": float(episode_returns[idx]),
                        "length": int(episode_lengths[idx]),
                    }
                )
                episode_returns[idx] = 0.0
                episode_lengths[idx] = 0
                if len(completed) >= args.episodes:
                    termination_reason = "episodes_requested_completed"
                    break
        if len(completed) >= args.episodes:
            break

    env.close()

    episodes_path = out_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as f:
        for record in completed:
            f.write(json.dumps(record) + "\n")

    trace_path = out_dir / "random_trace.png"
    make_contact_sheet(trace_frames, trace_path, label=f"procgen/{args.env_name} random")

    summary = {
        "benchmark": "Procgen",
        "source": "https://github.com/openai/procgen",
        "env_name": args.env_name,
        "episodes_requested": args.episodes,
        "episodes_completed": len(completed),
        "completed_requested": len(completed) >= args.episodes,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "steps_executed": steps_executed,
        "termination_reason": termination_reason,
        "truncated_by_max_steps": len(completed) < args.episodes,
        "seed": args.seed,
        "num_levels": args.num_levels,
        "start_level": args.start_level,
        "distribution_mode": args.distribution_mode,
        "policy": "random",
        "mean_return": float(np.mean([r["return"] for r in completed])) if completed else None,
        "mean_length": float(np.mean([r["length"] for r in completed])) if completed else None,
        "episodes_path": str(episodes_path),
        "trace_contact_sheet": str(trace_path),
        "note": "This is only a dependency/API smoke test. Learned visual policies are not implemented here yet.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary, indent=2))
    return 0


def extract_rgb(obs: Any, env_index: int = 0) -> np.ndarray:
    if isinstance(obs, dict):
        frame = obs.get("rgb")
    else:
        frame = obs
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[env_index]
    if frame.shape[0] in {3, 4} and frame.shape[-1] not in {3, 4}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return np.clip(frame, 0, 255).astype(np.uint8)


def make_contact_sheet(frames: list[np.ndarray], out_path: Path, label: str) -> None:
    if not frames:
        return
    pil_frames = [Image.fromarray(frame).resize((128, 128)) for frame in frames]
    label_h = 24
    sheet = Image.new("RGB", (128 * len(pil_frames), 128 + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), label, fill=(0, 0, 0))
    for idx, frame in enumerate(pil_frames):
        sheet.paste(frame, (128 * idx, label_h))
    sheet.save(out_path)


def write_report(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Procgen Smoke Benchmark",
        "",
        "Source: https://github.com/openai/procgen",
        "",
        "## Result",
        "",
        f"- Environment: `{summary['env_name']}`",
        f"- Policy: `{summary['policy']}`",
        f"- Episodes completed: `{summary['episodes_completed']}`",
        f"- Completed requested episodes: `{summary['completed_requested']}`",
        f"- Termination reason: `{summary['termination_reason']}`",
        f"- Mean return: `{summary['mean_return']}`",
        f"- Mean length: `{summary['mean_length']}`",
        "",
        "This confirms the Procgen dependency/API path only. It is not yet a Kairos control result.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def resolve_out_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
