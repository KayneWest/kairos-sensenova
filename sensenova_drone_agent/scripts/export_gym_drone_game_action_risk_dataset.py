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

from sensenova_drone.bc_data import ACTION_VOCAB
from sensenova_drone.gym_drone_game import DroneGameConfig, DroneMazeEnv

try:
    import torch
    from scripts.train_gym_drone_game import StateDQN
except ModuleNotFoundError:
    torch = None
    StateDQN = None


DEFAULT_ACTIONS = "hover,yaw_left,yaw_right,forward,strafe_left,strafe_right"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export action-conditioned branch-risk labels from the Gym drone game.")
    parser.add_argument("--checkpoint", default="output/gym_drone_game_dqn_overnight_20260509T032655Z/best.pt")
    parser.add_argument("--policy", choices=["dqn", "heuristic", "random", "mixed"], default="mixed")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=600000)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--enabled-actions", default=DEFAULT_ACTIONS)
    parser.add_argument("--candidate-actions", default=DEFAULT_ACTIONS)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--image-height", type=int, default=72)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--obstacle-count", type=int, default=14)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    out_root = Path(args.out_root)
    manifest_path = Path(args.manifest)
    summary_path = Path(args.summary_json)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    env_cfg = DroneGameConfig(
        image_width=args.image_width,
        image_height=args.image_height,
        max_episode_steps=args.max_episode_steps,
        obstacle_count=args.obstacle_count,
    )
    enabled_actions = parse_actions(args.enabled_actions)
    candidate_actions = parse_actions(args.candidate_actions)
    policy = load_dqn_policy(args.checkpoint, env_cfg, device=device) if args.policy in {"dqn", "mixed"} else None

    records: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    branch_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()

    for episode_index in range(args.episodes):
        seed = args.seed + episode_index
        episode_id = f"action_risk_{seed:06d}"
        env = DroneMazeEnv(env_cfg)
        obs, info = env.reset(seed=seed)
        done = False
        step_index = 0
        total_reward = 0.0
        episode_records: list[dict[str, Any]] = []
        while not done:
            image_dir = out_root / episode_id
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"step_{step_index:06d}.png"
            Image.fromarray(obs["image"]).save(image_path)
            before_info = dict(info)
            branches = [build_branch_record(env, action_index, before_info) for action_index in candidate_actions]
            best_action = max(branches, key=lambda item: item["utility"])
            for branch in branches:
                record = {
                    "episode_id": episode_id,
                    "step_index": step_index,
                    "image_path": str(image_path),
                    "candidate_action": branch["action"],
                    "candidate_action_index": branch["action_index"],
                    "best_action": best_action["action"],
                    "best_action_index": best_action["action_index"],
                    "is_best_action": branch["action_index"] == best_action["action_index"],
                    "goal_features": goal_features(before_info),
                    "labels": branch,
                    "metadata": {
                        "source": "gym_drone_game_branch_eval",
                        "behavior_policy": args.policy,
                        "before": select_info_fields(before_info),
                    },
                }
                episode_records.append(record)
                if branch["collision"]:
                    branch_counts["collision"] += 1
                if branch["near_miss"]:
                    branch_counts["near_miss"] += 1
                if branch["success"]:
                    branch_counts["success"] += 1
            behavior_action = choose_behavior_action(
                args.policy,
                env,
                obs,
                info,
                policy=policy,
                enabled_actions=enabled_actions,
                device=device,
                rng=rng,
            )
            obs, reward, terminated, truncated, info = env.step(behavior_action)
            total_reward += float(reward)
            behavior_counts[ACTION_VOCAB[behavior_action]] += 1
            done = bool(terminated or truncated)
            step_index += 1
        records.extend(episode_records)
        episode_summaries.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "length": step_index,
                "return": total_reward,
                "success": bool(info.get("success", False)),
                "collision": bool(info.get("collision", False)),
                "out_of_bounds": bool(info.get("out_of_bounds", False)),
                "timeout": bool(info.get("truncated", False)),
            }
        )

    split_map = split_by_episode([item["episode_id"] for item in episode_summaries], args.val_ratio)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            record["split"] = split_map.get(str(record["episode_id"]), "train")
            handle.write(json.dumps(record) + "\n")
    summary = {
        "policy": args.policy,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "episodes": args.episodes,
        "num_examples": len(records),
        "num_states": sum(item["length"] for item in episode_summaries),
        "candidate_actions": [ACTION_VOCAB[index] for index in candidate_actions],
        "enabled_actions": [ACTION_VOCAB[index] for index in enabled_actions],
        "branch_counts": dict(branch_counts),
        "behavior_counts": dict(behavior_counts),
        "manifest": str(manifest_path.resolve()),
        "out_root": str(out_root.resolve()),
        "env_config": env_cfg.__dict__,
        "episode_summaries": episode_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "episode_summaries"}, indent=2))
    return 0


def build_branch_record(env: DroneMazeEnv, action_index: int, before_info: dict[str, Any]) -> dict[str, Any]:
    branch = env.branch_step(action_index)
    after = branch["after"]
    before_distance = float(before_info.get("distance_to_goal_m", 0.0))
    after_distance = float(after.get("distance_to_goal_m", before_distance))
    progress_m = before_distance - after_distance
    after_clearance = dict(after.get("clearance_m", {}))
    front_after_m = float(after_clearance.get("front_m", env.cfg.max_depth_m))
    collision = bool(after.get("collision", False))
    out_of_bounds = bool(after.get("out_of_bounds", False))
    success = bool(after.get("success", False))
    near_miss = bool(front_after_m < 1.25 or collision)
    reward = float(branch["reward"])
    near_miss_penalty = 5.0 * max(0.0, (2.0 - front_after_m) / 2.0)
    non_progress_penalty = 0.08 if branch["action"] in {"hover", "yaw_left", "yaw_right"} else 0.0
    utility = (
        reward
        + 5.0 * (1.0 if success else 0.0)
        + 2.5 * progress_m
        + 0.6 * (front_after_m / max(env.cfg.max_depth_m, 1e-6))
        - 16.0 * (1.0 if collision else 0.0)
        - 5.0 * (1.0 if out_of_bounds else 0.0)
        - near_miss_penalty
        - non_progress_penalty
    )
    return {
        "action": branch["action"],
        "action_index": int(action_index),
        "reward": reward,
        "utility": float(utility),
        "progress_m": float(progress_m),
        "front_clearance_after_m": front_after_m,
        "collision": collision,
        "success": success,
        "out_of_bounds": out_of_bounds,
        "truncated": bool(after.get("truncated", False)),
        "near_miss": near_miss,
        "near_miss_penalty": float(near_miss_penalty),
        "after": select_info_fields(after),
    }


def choose_behavior_action(
    mode: str,
    env: DroneMazeEnv,
    obs: dict[str, np.ndarray],
    info: dict[str, Any],
    *,
    policy,
    enabled_actions: list[int],
    device: str,
    rng: random.Random,
) -> int:
    if mode == "heuristic":
        return env.expert_action_index()
    if mode == "random":
        return int(rng.choice(enabled_actions))
    if mode == "dqn":
        return choose_dqn_action(policy, obs, enabled_actions=enabled_actions, device=device)

    front = float(info.get("clearance_m", {}).get("front_m", env.cfg.max_depth_m))
    roll = rng.random()
    if front < env.cfg.front_blocked_threshold_m * 1.25:
        if roll < 0.35 and ACTION_VOCAB.index("forward") in enabled_actions:
            return ACTION_VOCAB.index("forward")
        if roll < 0.65:
            return int(rng.choice(enabled_actions))
        if policy is not None:
            return choose_dqn_action(policy, obs, enabled_actions=enabled_actions, device=device)
        return env.expert_action_index()
    if roll < 0.60 and policy is not None:
        return choose_dqn_action(policy, obs, enabled_actions=enabled_actions, device=device)
    if roll < 0.80:
        return env.expert_action_index()
    return int(rng.choice(enabled_actions))


def load_dqn_policy(checkpoint_path: str, env_cfg: DroneGameConfig, *, device: str):
    if torch is None or StateDQN is None:
        raise RuntimeError("torch is required for DQN-backed action-risk export.")
    payload = torch.load(checkpoint_path, map_location=device)
    probe_env = DroneMazeEnv(env_cfg)
    obs, _ = probe_env.reset(seed=0)
    model = StateDQN(int(obs["state"].shape[0]), len(payload.get("action_vocab") or ACTION_VOCAB)).to(device)
    model.load_state_dict(payload["policy_state_dict"])
    model.eval()
    return model


def choose_dqn_action(policy, obs: dict[str, np.ndarray], *, enabled_actions: list[int], device: str) -> int:
    with torch.no_grad():
        state = torch.tensor(obs["state"][None, :], dtype=torch.float32, device=device)
        logits = policy(state)
        mask = torch.full_like(logits, fill_value=-1e9)
        mask[:, enabled_actions] = logits[:, enabled_actions]
        return int(torch.argmax(mask, dim=-1).item())


def goal_features(info: dict[str, Any]) -> list[float]:
    forward, right = info.get("goal_body_xy_m") or [0.0, 0.0]
    heading_error_deg = math.degrees(math.atan2(float(right), max(float(forward), 1e-6)))
    return [
        float(np.clip(float(forward) / 10.0, -2.0, 2.0)),
        float(np.clip(float(right) / 5.0, -2.0, 2.0)),
        0.0,
        float(np.clip(heading_error_deg / 180.0, -1.0, 1.0)),
    ]


def select_info_fields(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_index": info.get("step_index"),
        "position_xy_m": info.get("position_xy_m"),
        "yaw_deg": info.get("yaw_deg"),
        "goal_body_xy_m": info.get("goal_body_xy_m"),
        "distance_to_goal_m": info.get("distance_to_goal_m"),
        "clearance_m": info.get("clearance_m", {}),
        "collision": bool(info.get("collision", False)),
        "success": bool(info.get("success", False)),
        "out_of_bounds": bool(info.get("out_of_bounds", False)),
        "truncated": bool(info.get("truncated", False)),
    }


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return requested


def parse_actions(raw: str) -> list[int]:
    indices = []
    for name in [item.strip() for item in raw.split(",") if item.strip()]:
        if name not in ACTION_VOCAB:
            raise ValueError(f"Unsupported action: {name!r}")
        indices.append(ACTION_VOCAB.index(name))
    if not indices:
        raise ValueError("No actions provided.")
    return list(dict.fromkeys(indices))


def split_by_episode(episode_ids: list[str], val_ratio: float) -> dict[str, str]:
    ids = list(dict.fromkeys(episode_ids))
    rng = random.Random(0)
    rng.shuffle(ids)
    num_val = max(1, int(round(len(ids) * float(val_ratio)))) if len(ids) > 1 and val_ratio > 0.0 else 0
    val_ids = set(ids[:num_val])
    return {episode_id: ("val" if episode_id in val_ids else "train") for episode_id in ids}


if __name__ == "__main__":
    raise SystemExit(main())
