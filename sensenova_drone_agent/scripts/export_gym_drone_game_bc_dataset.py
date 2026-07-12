#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sensenova_drone.actions import DiscreteDroneAction, discrete_to_command
from sensenova_drone.bc_data import ACTION_TO_INDEX, ACTION_VOCAB
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
            raise RuntimeError("torch is required for DQN dataset export.")
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
    parser = argparse.ArgumentParser(description="Export Gym drone-game episodes to the existing BC image manifest format.")
    parser.add_argument("--checkpoint", default="", help="DQN checkpoint. Omit for heuristic/random policy.")
    parser.add_argument("--policy", choices=["dqn", "heuristic", "random"], default="dqn")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=300000)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--enabled-actions", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    manifest_path = Path(args.manifest)
    summary_path = Path(args.summary_json)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint_payload = None
    policy = None
    env_cfg = DroneGameConfig()
    enabled_actions = parse_enabled_actions(
        args.enabled_actions or "hover,yaw_left,yaw_right,forward,strafe_left,strafe_right"
    )
    if args.policy == "dqn":
        if torch is None or nn is None:
            raise RuntimeError("torch is required for DQN export.")
        if not args.checkpoint:
            raise RuntimeError("--checkpoint is required when --policy=dqn.")
        checkpoint_payload = torch.load(args.checkpoint, map_location=device)
        env_cfg = DroneGameConfig(**dict(checkpoint_payload.get("env_config") or {}))
        enabled_actions = parse_enabled_actions(
            args.enabled_actions
            or checkpoint_payload.get("args", {}).get("enabled_actions")
            or ",".join(checkpoint_payload.get("summary", {}).get("enabled_actions", []))
            or ",".join(ACTION_VOCAB)
        )
        probe_env = DroneMazeEnv(env_cfg)
        probe_obs, _ = probe_env.reset(seed=args.seed)
        policy = StateDQN(
            state_dim=int(probe_obs["state"].shape[0]),
            num_actions=len(checkpoint_payload.get("action_vocab") or ACTION_VOCAB),
        ).to(device)
        policy.load_state_dict(checkpoint_payload["policy_state_dict"])
        policy.eval()

    records: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    counts_by_action: Counter[str] = Counter()
    counts_by_status: Counter[str] = Counter()

    for episode_index in range(args.episodes):
        episode_id = f"gym_game_{args.seed + episode_index:06d}"
        episode_dir = out_root / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        env = DroneMazeEnv(env_cfg)
        obs, info = env.reset(seed=args.seed + episode_index)
        episode_records: list[dict[str, Any]] = []
        total_reward = 0.0
        done = False
        step_index = 0
        action_counts: Counter[str] = Counter()
        while not done:
            before_info = dict(info)
            action_index = choose_action(
                args.policy,
                env,
                obs,
                policy=policy,
                enabled_actions=enabled_actions,
                device=device,
            )
            action_name = ACTION_VOCAB[action_index]
            action = DiscreteDroneAction(action_name)
            step_dir = episode_dir / f"step_{step_index:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            image_path = step_dir / "frame_before.png"
            Image.fromarray(obs["image"]).save(image_path)

            next_obs, reward, terminated, truncated, info = env.step(action_index)
            next_image_path = step_dir / "frame_after.png"
            Image.fromarray(next_obs["image"]).save(next_image_path)
            total_reward += float(reward)
            action_counts[action_name] += 1
            record = build_record(
                episode_id=episode_id,
                step_index=step_index,
                image_path=image_path,
                next_image_path=next_image_path,
                action=action,
                before_info=before_info,
                after_info=info,
                reward=reward,
                source_policy=args.policy,
            )
            episode_records.append(record)
            obs = next_obs
            done = bool(terminated or truncated)
            step_index += 1

        success = bool(info.get("success", False))
        collision = bool(info.get("collision", False))
        status = "success" if success else "collision" if collision else "timeout"
        counts_by_status[status] += 1
        if success or not args.success_only:
            records.extend(episode_records)
            counts_by_action.update(action_counts)
        episode_summaries.append(
            {
                "episode_id": episode_id,
                "status": status,
                "success": success,
                "collision": collision,
                "length": step_index,
                "return": total_reward,
                "included": bool(success or not args.success_only),
                "action_counts": dict(action_counts),
            }
        )

    split_by_episode = build_split_map([item["episode_id"] for item in episode_summaries if item["included"]], args.val_ratio)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            record["split"] = split_by_episode.get(str(record["episode_id"]), "train")
            handle.write(json.dumps(record) + "\n")

    summary = {
        "policy": args.policy,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "episodes_requested": args.episodes,
        "episodes_included": sum(1 for item in episode_summaries if item["included"]),
        "num_examples": len(records),
        "success_only": bool(args.success_only),
        "counts_by_status": dict(counts_by_status),
        "counts_by_action": {action: counts_by_action.get(action, 0) for action in ACTION_VOCAB},
        "manifest": str(manifest_path.resolve()),
        "out_root": str(out_root.resolve()),
        "episode_summaries": episode_summaries,
        "enabled_actions": [ACTION_VOCAB[index] for index in enabled_actions],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return requested


def parse_enabled_actions(raw: str) -> list[int]:
    indices = []
    for name in [item.strip() for item in str(raw).split(",") if item.strip()]:
        if name not in ACTION_VOCAB:
            raise ValueError(f"Unsupported action: {name!r}")
        indices.append(ACTION_VOCAB.index(name))
    if not indices:
        raise ValueError("No enabled actions provided.")
    return list(dict.fromkeys(indices))


def choose_action(
    mode: str,
    env: DroneMazeEnv,
    obs: dict[str, np.ndarray],
    *,
    policy: StateDQN | None,
    enabled_actions: list[int],
    device: str,
) -> int:
    if mode == "heuristic":
        return env.expert_action_index()
    if mode == "random":
        return int(random.choice(enabled_actions))
    if policy is None or torch is None:
        raise RuntimeError("DQN policy is not loaded.")
    with torch.no_grad():
        state = torch.tensor(obs["state"][None, :], dtype=torch.float32, device=device)
        logits = policy(state)
        mask = torch.full_like(logits, fill_value=-1e9)
        mask[:, enabled_actions] = logits[:, enabled_actions]
        return int(torch.argmax(mask, dim=-1).item())


def build_record(
    *,
    episode_id: str,
    step_index: int,
    image_path: Path,
    next_image_path: Path,
    action: DiscreteDroneAction,
    before_info: dict[str, Any],
    after_info: dict[str, Any],
    reward: float,
    source_policy: str,
) -> dict[str, Any]:
    command = discrete_to_command(action)
    goal_forward, goal_right = before_info.get("goal_body_xy_m") or [0.0, 0.0]
    heading_error_deg = math.degrees(math.atan2(float(goal_right), max(float(goal_forward), 1e-6)))
    return {
        "episode_id": episode_id,
        "step_index": step_index,
        "action": action.value,
        "action_index": ACTION_TO_INDEX[action.value],
        "command": {
            "forward_m_s": command.forward_m_s,
            "right_m_s": command.right_m_s,
            "down_m_s": command.down_m_s,
            "yawspeed_deg_s": command.yawspeed_deg_s,
            "duration_s": command.duration_s,
            "source_action": action.value,
        },
        "image_path": str(image_path),
        "next_image_path": str(next_image_path),
        "timestamp_s": float(step_index),
        "pose": None,
        "intrinsics": None,
        "metadata": {
            "world_label": "gym_drone_game",
            "scenario_label": "gym_drone_game",
            "reward": float(reward),
            "source_policy": source_policy,
            "teacher": {
                "reason": f"{source_policy}_action",
                "goal_features": {
                    "forward_m": float(goal_forward),
                    "right_m": float(goal_right),
                    "alt_error_m": 0.0,
                    "heading_error_deg": heading_error_deg,
                },
                "depth_clearance_m": before_info.get("clearance_m", {}),
                "after": {
                    "success": bool(after_info.get("success", False)),
                    "collision": bool(after_info.get("collision", False)),
                    "distance_to_goal_m": after_info.get("distance_to_goal_m"),
                    "clearance_m": after_info.get("clearance_m", {}),
                },
            },
        },
    }


def build_split_map(episode_ids: list[str], val_ratio: float) -> dict[str, str]:
    if not episode_ids:
        return {}
    rng = random.Random(0)
    ids = list(dict.fromkeys(episode_ids))
    rng.shuffle(ids)
    num_val = max(1, int(round(len(ids) * float(val_ratio)))) if len(ids) > 1 and val_ratio > 0.0 else 0
    val_ids = set(ids[:num_val])
    return {episode_id: ("val" if episode_id in val_ids else "train") for episode_id in ids}


if __name__ == "__main__":
    raise SystemExit(main())
