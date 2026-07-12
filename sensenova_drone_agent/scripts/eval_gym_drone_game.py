#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sensenova_drone.bc_data import ACTION_VOCAB
from sensenova_drone.gym_drone_game import DroneGameConfig, DroneMazeEnv

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


class StateDQN(nn.Module if nn is not None else object):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        if nn is None:
            raise RuntimeError("torch is required for DQN evaluation.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state):
        return self.net(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Gym drone-game DQN checkpoint on held-out seeds.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--enabled-actions", default="", help="Override comma-separated action names.")
    parser.add_argument("--trace-episodes", type=int, default=6)
    parser.add_argument("--trace-frames", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    if torch is None or nn is None:
        raise RuntimeError("torch is required for DQN evaluation.")

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    env_config = checkpoint.get("env_config") or {}
    env_cfg = DroneGameConfig(**env_config)
    enabled_actions = parse_enabled_actions(
        args.enabled_actions
        or checkpoint.get("args", {}).get("enabled_actions")
        or ",".join(checkpoint.get("summary", {}).get("enabled_actions", []))
        or "hover,yaw_left,yaw_right,forward,strafe_left,strafe_right"
    )

    env = DroneMazeEnv(env_cfg)
    obs, _ = env.reset(seed=args.seed)
    policy = StateDQN(
        state_dim=int(obs["state"].shape[0]),
        num_actions=len(checkpoint.get("action_vocab") or ACTION_VOCAB),
    ).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    episodes = []
    action_counts: Counter[str] = Counter()
    trace_rows: list[tuple[str, list[np.ndarray]]] = []

    for episode_idx in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode_idx)
        total_reward = 0.0
        done = False
        step_idx = 0
        episode_actions: Counter[str] = Counter()
        min_front = float("inf")
        trace_frames: list[np.ndarray] = []
        info: dict[str, Any] = {}

        while not done:
            if episode_idx < args.trace_episodes and len(trace_frames) < args.trace_frames:
                trace_frames.append(obs["image"])
            action = greedy_action(policy, obs["state"], enabled_actions, device=device)
            obs, reward, terminated, truncated, info = env.step(action)
            action_name = ACTION_VOCAB[action]
            action_counts[action_name] += 1
            episode_actions[action_name] += 1
            total_reward += float(reward)
            front = info.get("clearance_m", {}).get("front_m")
            if front is not None:
                min_front = min(min_front, float(front))
            done = bool(terminated or truncated)
            step_idx += 1

        episode = {
            "episode_index": episode_idx,
            "seed": args.seed + episode_idx,
            "return": total_reward,
            "length": step_idx,
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "out_of_bounds": bool(info.get("out_of_bounds", False)),
            "truncated": bool(info.get("truncated", False)),
            "distance_to_goal_m": info.get("distance_to_goal_m"),
            "min_front_clearance_m": None if min_front == float("inf") else min_front,
            "action_counts": dict(episode_actions),
        }
        episodes.append(episode)
        if trace_frames:
            trace_rows.append((f"seed {args.seed + episode_idx}", trace_frames))

    summary = summarize(
        checkpoint_path=str(Path(args.checkpoint).resolve()),
        out_dir=out_dir,
        episodes=episodes,
        action_counts=action_counts,
        enabled_actions=enabled_actions,
        env_cfg=env_cfg,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode) + "\n")
    if trace_rows:
        make_trace_sheet(trace_rows, out_dir / "trace_contact_sheet.png")
    write_report(out_dir, summary)
    print(json.dumps(summary, indent=2))
    return 0


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def parse_enabled_actions(raw: str) -> list[int]:
    names = [item.strip() for item in str(raw).split(",") if item.strip()]
    indices = []
    for name in names:
        if name not in ACTION_VOCAB:
            raise ValueError(f"Unsupported action: {name!r}")
        indices.append(ACTION_VOCAB.index(name))
    if not indices:
        raise ValueError("No enabled actions provided.")
    return list(dict.fromkeys(indices))


def greedy_action(policy: StateDQN, state: np.ndarray, enabled_actions: list[int], *, device: str) -> int:
    with torch.no_grad():
        tensor = torch.tensor(state[None, :], dtype=torch.float32, device=device)
        logits = policy(tensor)
        mask = torch.full_like(logits, fill_value=-1e9)
        mask[:, enabled_actions] = logits[:, enabled_actions]
        return int(torch.argmax(mask, dim=-1).item())


def summarize(
    *,
    checkpoint_path: str,
    out_dir: Path,
    episodes: list[dict[str, Any]],
    action_counts: Counter[str],
    enabled_actions: list[int],
    env_cfg: DroneGameConfig,
) -> dict[str, Any]:
    returns = [float(item["return"]) for item in episodes]
    lengths = [int(item["length"]) for item in episodes]
    successes = [1.0 if item["success"] else 0.0 for item in episodes]
    collisions = [1.0 if item["collision"] else 0.0 for item in episodes]
    out_of_bounds = [1.0 if item["out_of_bounds"] else 0.0 for item in episodes]
    truncated = [1.0 if item["truncated"] else 0.0 for item in episodes]
    front_values = [
        float(item["min_front_clearance_m"])
        for item in episodes
        if item["min_front_clearance_m"] is not None
    ]
    return {
        "checkpoint_path": checkpoint_path,
        "out_dir": str(out_dir.resolve()),
        "episodes": len(episodes),
        "success_rate": mean(successes),
        "collision_rate": mean(collisions),
        "out_of_bounds_rate": mean(out_of_bounds),
        "timeout_rate": mean(truncated),
        "mean_return": mean(returns),
        "median_return": float(np.median(returns)) if returns else 0.0,
        "mean_length": mean(lengths),
        "mean_min_front_clearance_m": mean(front_values) if front_values else None,
        "action_counts": dict(action_counts),
        "enabled_actions": [ACTION_VOCAB[index] for index in enabled_actions],
        "env_config": env_cfg.__dict__,
        "report_path": str((out_dir / "index.html").resolve()),
        "trace_contact_sheet": str((out_dir / "trace_contact_sheet.png").resolve()),
    }


def mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def make_trace_sheet(rows: list[tuple[str, list[np.ndarray]]], out_path: Path) -> None:
    if not rows:
        return
    frame_h, frame_w = rows[0][1][0].shape[:2]
    label_w = 96
    label_h = 16
    cols = max(len(frames) for _, frames in rows)
    sheet = Image.new("RGB", (label_w + frame_w * cols, label_h + frame_h * len(rows)), color=(28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    for col in range(cols):
        draw.text((label_w + col * frame_w + 4, 2), f"t={col}", fill=(240, 240, 240))
    for row_idx, (label, frames) in enumerate(rows):
        y = label_h + row_idx * frame_h
        draw.text((4, y + 4), label, fill=(240, 240, 240))
        for col, frame in enumerate(frames):
            sheet.paste(Image.fromarray(frame.astype(np.uint8), mode="RGB"), (label_w + col * frame_w, y))
    sheet.save(out_path)


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Gym Drone Game Eval</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f4f1e8; color: #202020; }}
    table {{ border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #c9c1ad; padding: 6px 8px; text-align: left; }}
    th {{ background: #292f25; color: white; }}
    img {{ max-width: 100%; border: 1px solid #9f967f; background: white; }}
    code {{ background: #ebe4d4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Gym Drone Game Eval</h1>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Episodes</td><td>{summary['episodes']}</td></tr>
    <tr><td>Success Rate</td><td>{summary['success_rate']:.4f}</td></tr>
    <tr><td>Collision Rate</td><td>{summary['collision_rate']:.4f}</td></tr>
    <tr><td>Timeout Rate</td><td>{summary['timeout_rate']:.4f}</td></tr>
    <tr><td>Mean Return</td><td>{summary['mean_return']:.4f}</td></tr>
    <tr><td>Mean Length</td><td>{summary['mean_length']:.2f}</td></tr>
  </table>
  <h2>Trace Contact Sheet</h2>
  <img src="trace_contact_sheet.png" />
  <h2>Checkpoint</h2>
  <p><code>{summary['checkpoint_path']}</code></p>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
