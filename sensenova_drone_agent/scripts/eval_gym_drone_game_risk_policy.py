#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
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
from sensenova_drone.risk_aware_policy import load_risk_aware_policy_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a risk-aware visual policy in the Gym drone game.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=410000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--enabled-actions", default="hover,yaw_left,yaw_right,forward,strafe_left,strafe_right")
    parser.add_argument("--trace-episodes", type=int, default=6)
    parser.add_argument("--trace-frames", type=int, default=10)
    parser.add_argument("--no-shield", action="store_true")
    parser.add_argument("--shield-collision-threshold", type=float, default=None)
    parser.add_argument("--shield-front-clearance-m", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enabled_actions = parse_enabled_actions(args.enabled_actions)
    runner = load_risk_aware_policy_runner(
        args.checkpoint,
        device=args.device,
        shield_enabled=not args.no_shield,
        shield_collision_threshold=args.shield_collision_threshold,
        shield_front_clearance_m=args.shield_front_clearance_m,
    )
    env_cfg = DroneGameConfig()
    episodes = []
    action_counts: Counter[str] = Counter()
    unshielded_counts: Counter[str] = Counter()
    shield_counts: Counter[str] = Counter()
    trace_rows: list[tuple[str, list[np.ndarray]]] = []

    for episode_idx in range(args.episodes):
        env = DroneMazeEnv(env_cfg)
        obs, info = env.reset(seed=args.seed + episode_idx)
        runner.reset_history()
        done = False
        total_reward = 0.0
        step_idx = 0
        min_front = float("inf")
        episode_actions: Counter[str] = Counter()
        trace_frames: list[np.ndarray] = []
        risk_values: list[float] = []
        clearance_predictions: list[float] = []
        shielded_steps = 0

        while not done:
            if episode_idx < args.trace_episodes and len(trace_frames) < args.trace_frames:
                trace_frames.append(obs["image"])
            prediction = runner.predict(
                Image.fromarray(obs["image"]),
                goal_features=goal_features(info),
                enabled_actions=enabled_actions,
            )
            action_index = prediction.action_index
            unshielded_counts[str(prediction.metadata.get("unshielded_action"))] += 1
            if prediction.shield_reason:
                shield_counts[prediction.shield_reason] += 1
                shielded_steps += 1
            obs, reward, terminated, truncated, info = env.step(action_index)
            action_name = ACTION_VOCAB[action_index]
            total_reward += float(reward)
            action_counts[action_name] += 1
            episode_actions[action_name] += 1
            risk_values.append(float(prediction.collision_risk))
            clearance_predictions.append(float(prediction.front_clearance_m))
            front = info.get("clearance_m", {}).get("front_m")
            if front is not None:
                min_front = min(min_front, float(front))
            done = bool(terminated or truncated)
            step_idx += 1

        episodes.append(
            {
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
                "mean_predicted_collision_risk": mean(risk_values),
                "mean_predicted_front_clearance_m": mean(clearance_predictions),
                "shielded_steps": shielded_steps,
                "action_counts": dict(episode_actions),
            }
        )
        if trace_frames:
            trace_rows.append((f"seed {args.seed + episode_idx}", trace_frames))

    summary = summarize(
        checkpoint_path=str(Path(args.checkpoint).resolve()),
        out_dir=out_dir,
        episodes=episodes,
        action_counts=action_counts,
        unshielded_counts=unshielded_counts,
        shield_counts=shield_counts,
        enabled_actions=enabled_actions,
        shield_enabled=not args.no_shield,
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


def parse_enabled_actions(raw: str) -> list[int]:
    indices = []
    for name in [item.strip() for item in raw.split(",") if item.strip()]:
        if name not in ACTION_VOCAB:
            raise ValueError(f"Unsupported action: {name!r}")
        indices.append(ACTION_VOCAB.index(name))
    return list(dict.fromkeys(indices))


def goal_features(info: dict[str, Any]) -> list[float]:
    forward, right = info.get("goal_body_xy_m") or [0.0, 0.0]
    heading_error_deg = math.degrees(math.atan2(float(right), max(float(forward), 1e-6)))
    return [
        float(np.clip(float(forward) / 10.0, -2.0, 2.0)),
        float(np.clip(float(right) / 5.0, -2.0, 2.0)),
        0.0,
        float(np.clip(heading_error_deg / 180.0, -1.0, 1.0)),
    ]


def summarize(
    *,
    checkpoint_path: str,
    out_dir: Path,
    episodes: list[dict[str, Any]],
    action_counts: Counter[str],
    unshielded_counts: Counter[str],
    shield_counts: Counter[str],
    enabled_actions: list[int],
    shield_enabled: bool,
) -> dict[str, Any]:
    returns = [float(item["return"]) for item in episodes]
    lengths = [int(item["length"]) for item in episodes]
    successes = [1.0 if item["success"] else 0.0 for item in episodes]
    collisions = [1.0 if item["collision"] else 0.0 for item in episodes]
    timeouts = [1.0 if item["truncated"] else 0.0 for item in episodes]
    out_of_bounds = [1.0 if item["out_of_bounds"] else 0.0 for item in episodes]
    front_values = [float(item["min_front_clearance_m"]) for item in episodes if item["min_front_clearance_m"] is not None]
    shielded_steps = [float(item["shielded_steps"]) for item in episodes]
    return {
        "checkpoint_path": checkpoint_path,
        "out_dir": str(out_dir.resolve()),
        "episodes": len(episodes),
        "shield_enabled": shield_enabled,
        "success_rate": mean(successes),
        "collision_rate": mean(collisions),
        "timeout_rate": mean(timeouts),
        "out_of_bounds_rate": mean(out_of_bounds),
        "mean_return": mean(returns),
        "median_return": float(np.median(returns)) if returns else 0.0,
        "mean_length": mean(lengths),
        "mean_min_front_clearance_m": mean(front_values) if front_values else None,
        "mean_shielded_steps": mean(shielded_steps),
        "action_counts": dict(action_counts),
        "unshielded_action_counts": dict(unshielded_counts),
        "shield_counts": dict(shield_counts),
        "enabled_actions": [ACTION_VOCAB[index] for index in enabled_actions],
        "report_path": str((out_dir / "index.html").resolve()),
        "trace_contact_sheet": str((out_dir / "trace_contact_sheet.png").resolve()),
    }


def mean(values) -> float:
    values = list(values)
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
  <title>Risk-Aware Gym Drone Policy Eval</title>
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
  <h1>Risk-Aware Gym Drone Policy Eval</h1>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Episodes</td><td>{summary['episodes']}</td></tr>
    <tr><td>Shield Enabled</td><td>{summary['shield_enabled']}</td></tr>
    <tr><td>Success Rate</td><td>{summary['success_rate']:.4f}</td></tr>
    <tr><td>Collision Rate</td><td>{summary['collision_rate']:.4f}</td></tr>
    <tr><td>Timeout Rate</td><td>{summary['timeout_rate']:.4f}</td></tr>
    <tr><td>Mean Return</td><td>{summary['mean_return']:.4f}</td></tr>
    <tr><td>Mean Shielded Steps</td><td>{summary['mean_shielded_steps']:.2f}</td></tr>
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
