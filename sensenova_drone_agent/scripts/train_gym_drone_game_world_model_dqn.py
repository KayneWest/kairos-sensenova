#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sensenova_drone.bc_data import ACTION_VOCAB
from sensenova_drone.gym_drone_game import DroneGameConfig, DroneMazeEnv
from scripts.train_gym_drone_game_world_model import ActionConditionedWorldModel

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


DEFAULT_ENABLED_ACTIONS = "hover,yaw_left,yaw_right,forward,strafe_left,strafe_right"


class WorldModelDQN(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 256):
        if nn is None:
            raise RuntimeError("torch is required for WorldModelDQN.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(num_actions)),
        )

    def forward(self, features):
        return self.net(features)


class LatentReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._items: deque[tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        feature: np.ndarray,
        action: int,
        reward: float,
        next_feature: np.ndarray,
        done: bool,
        next_enabled_actions: list[int],
    ) -> None:
        self._items.append(
            (
                np.asarray(feature, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_feature, dtype=np.float32),
                bool(done),
                enabled_action_mask(next_enabled_actions),
            )
        )

    def sample(self, batch_size: int, device: str) -> dict[str, Any]:
        batch = random.sample(self._items, int(batch_size))
        features, actions, rewards, next_features, dones, next_action_masks = zip(*batch)
        return {
            "feature": torch.tensor(np.asarray(features), dtype=torch.float32, device=device),
            "action": torch.tensor(actions, dtype=torch.long, device=device),
            "reward": torch.tensor(rewards, dtype=torch.float32, device=device),
            "next_feature": torch.tensor(np.asarray(next_features), dtype=torch.float32, device=device),
            "done": torch.tensor(dones, dtype=torch.float32, device=device),
            "next_action_mask": torch.tensor(np.asarray(next_action_masks), dtype=torch.bool, device=device),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train DQN in the drone Gym game using frozen action-conditioned world-model "
            "features instead of privileged simulator state."
        )
    )
    parser.add_argument("--world-model-checkpoint", default=str(PROJECT_ROOT / "output" / "gym_drone_game_world_model_v1" / "best.pt"))
    parser.add_argument(
        "--encoder-source",
        choices=["pretrained", "random"],
        default="pretrained",
        help="Use pretrained world-model weights or a frozen random encoder with the same architecture.",
    )
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "output" / "gym_drone_game_world_model_dqn"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--world-size-m", type=float, default=16.0)
    parser.add_argument("--obstacle-count", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--target-update-steps", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=15000)
    parser.add_argument(
        "--expert-mix-start",
        type=float,
        default=0.05,
        help="Optional decaying heuristic exploration. Set 0 for pure epsilon-greedy RL.",
    )
    parser.add_argument("--expert-mix-end", type=float, default=0.0)
    parser.add_argument("--expert-mix-decay-steps", type=int, default=10000)
    parser.add_argument("--extra-collision-penalty", type=float, default=0.0)
    parser.add_argument("--extra-out-of-bounds-penalty", type=float, default=0.0)
    parser.add_argument("--near-obstacle-threshold-m", type=float, default=1.2)
    parser.add_argument("--near-obstacle-penalty", type=float, default=0.0)
    parser.add_argument("--forward-low-clearance-threshold-m", type=float, default=2.2)
    parser.add_argument("--forward-low-clearance-penalty", type=float, default=0.0)
    parser.add_argument("--clearance-recovery-bonus", type=float, default=0.0)
    parser.add_argument(
        "--shield-front-clearance-m",
        type=float,
        default=None,
        help="If set, train and evaluate with FORWARD masked when front clearance is below this threshold.",
    )
    parser.add_argument("--enabled-actions", default=DEFAULT_ENABLED_ACTIONS)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=910000)
    parser.add_argument("--trace-episodes", type=int, default=4)
    parser.add_argument("--trace-frames", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    if torch is None or nn is None or F is None:
        raise RuntimeError("torch is required. Use the existing Docker tools image.")

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    device = resolve_device(args.device)
    world_model, wm_config = load_world_model(
        args.world_model_checkpoint,
        device=device,
        encoder_source=args.encoder_source,
    )
    latent_dim = int(wm_config["latent_dim"])
    image_width = int(wm_config["image_width"])
    image_height = int(wm_config["image_height"])
    feature_dim = latent_dim + 4
    enabled_actions = parse_enabled_actions(args.enabled_actions)
    env_cfg = DroneGameConfig(
        world_size_m=args.world_size_m,
        obstacle_count=args.obstacle_count,
        image_width=image_width,
        image_height=image_height,
        max_episode_steps=args.max_episode_steps,
    )

    q_net = WorldModelDQN(feature_dim, len(ACTION_VOCAB), hidden_dim=args.hidden_dim).to(device)
    target_net = WorldModelDQN(feature_dim, len(ACTION_VOCAB), hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=args.learning_rate)
    replay = LatentReplayBuffer(args.replay_size)

    env = DroneMazeEnv(env_cfg)
    obs, info = env.reset(seed=args.seed)
    feature = encode_env_feature(world_model, obs["image"], info, device=device)
    completed_episodes: list[dict[str, Any]] = []
    loss_values: deque[float] = deque(maxlen=200)
    episode_return = 0.0
    episode_shaped_return = 0.0
    episode_length = 0
    episode_success = False
    episode_collision = False
    episode_timeout = False
    episode_actions: Counter[str] = Counter()
    train_shielded_steps = 0
    best_success_rate = -1.0

    for step in range(1, args.total_steps + 1):
        epsilon = linear_schedule(args.epsilon_start, args.epsilon_end, step, args.epsilon_decay_steps)
        expert_mix = linear_schedule(args.expert_mix_start, args.expert_mix_end, step, args.expert_mix_decay_steps)
        current_enabled_actions = shielded_enabled_actions(
            info,
            enabled_actions,
            shield_front_clearance_m=args.shield_front_clearance_m,
        )
        if len(current_enabled_actions) != len(enabled_actions):
            train_shielded_steps += 1
        action = select_action(
            q_net,
            feature,
            env,
            enabled_action_indices=current_enabled_actions,
            epsilon=epsilon,
            expert_mix=expert_mix,
            device=device,
        )
        before_info = info
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = bool(terminated or truncated)
        next_feature = encode_env_feature(world_model, next_obs["image"], next_info, device=device)
        next_enabled_actions = shielded_enabled_actions(
            next_info,
            enabled_actions,
            shield_front_clearance_m=args.shield_front_clearance_m,
        )
        shaped_reward, _shaping_terms = shape_safety_reward(
            reward,
            action,
            before_info=before_info,
            after_info=next_info,
            args=args,
        )
        replay.add(feature, action, shaped_reward, next_feature, done, next_enabled_actions)

        episode_return += float(reward)
        episode_shaped_return += float(shaped_reward)
        episode_length += 1
        episode_success = episode_success or bool(next_info.get("success", False))
        episode_collision = episode_collision or bool(next_info.get("collision", False))
        episode_timeout = episode_timeout or bool(next_info.get("truncated", False))
        episode_actions[ACTION_VOCAB[action]] += 1

        feature = next_feature
        obs = next_obs
        info = next_info

        if len(replay) >= max(args.warmup_steps, args.batch_size):
            batch = replay.sample(args.batch_size, device)
            loss = optimize_dqn(
                q_net,
                target_net,
                optimizer,
                batch,
                gamma=args.gamma,
                enabled_action_indices=enabled_actions,
            )
            loss_values.append(loss)

        if step % args.target_update_steps == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            completed_episodes.append(
                {
                    "step": step,
                    "return": episode_return,
                    "shaped_return": episode_shaped_return,
                    "length": episode_length,
                    "success": episode_success,
                    "collision": episode_collision,
                    "timeout": episode_timeout,
                    "action_counts": dict(episode_actions),
                }
            )
            obs, info = env.reset()
            feature = encode_env_feature(world_model, obs["image"], info, device=device)
            episode_return = 0.0
            episode_shaped_return = 0.0
            episode_length = 0
            episode_success = False
            episode_collision = False
            episode_timeout = False
            episode_actions = Counter()

        if step == 1 or step % args.eval_every == 0 or step == args.total_steps:
            eval_summary = evaluate_policy(
                q_net,
                world_model,
                env_cfg,
                episodes=args.eval_episodes,
                seed=args.eval_seed,
                device=device,
                enabled_action_indices=enabled_actions,
                shield_front_clearance_m=args.shield_front_clearance_m,
                trace_path=out_dir / "latest_eval_trace.png",
                trace_episodes=args.trace_episodes,
                trace_frames=args.trace_frames,
            )
            train_summary = {
                "step": step,
                "epsilon": epsilon,
                "expert_mix": expert_mix,
                "replay_size": len(replay),
                "mean_loss_200": mean_or_none(loss_values),
                "episodes_completed": len(completed_episodes),
                "recent_train_return": mean_or_none([episode["return"] for episode in completed_episodes[-20:]]),
                "recent_train_shaped_return": mean_or_none([episode["shaped_return"] for episode in completed_episodes[-20:]]),
                "recent_train_success_rate": mean_or_none([1.0 if episode["success"] else 0.0 for episode in completed_episodes[-20:]]),
                "recent_train_collision_rate": mean_or_none([1.0 if episode["collision"] else 0.0 for episode in completed_episodes[-20:]]),
                "reward_shaping": reward_shaping_config(args),
                "train_shielded_steps": train_shielded_steps,
                "shield_front_clearance_m": args.shield_front_clearance_m,
                "eval": eval_summary,
            }
            append_jsonl(metrics_path, train_summary)
            save_checkpoint(
                out_dir / "last.pt",
                q_net,
                target_net,
                optimizer,
                world_model,
                args=args,
                env_cfg=env_cfg,
                wm_config=wm_config,
                step=step,
                summary=train_summary,
            )
            if eval_summary["success_rate"] >= best_success_rate:
                best_success_rate = float(eval_summary["success_rate"])
                save_checkpoint(
                    out_dir / "best.pt",
                    q_net,
                    target_net,
                    optimizer,
                    world_model,
                    args=args,
                    env_cfg=env_cfg,
                    wm_config=wm_config,
                    step=step,
                    summary=train_summary,
                )
            write_dashboard(out_dir, train_summary, metrics_path)
            print(json.dumps(train_summary, indent=2), flush=True)

    final_summary = {
        "out_dir": str(out_dir.resolve()),
        "world_model_checkpoint": str(Path(args.world_model_checkpoint).resolve()),
        "total_steps": int(args.total_steps),
        "best_success_rate": best_success_rate,
        "num_completed_train_episodes": len(completed_episodes),
        "latest_checkpoint": str((out_dir / "last.pt").resolve()),
        "best_checkpoint": str((out_dir / "best.pt").resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "dashboard_path": str((out_dir / "index.html").resolve()),
        "env_config": env_cfg.__dict__,
        "world_model_config": wm_config,
        "encoder_source": args.encoder_source,
        "enabled_actions": [ACTION_VOCAB[index] for index in enabled_actions],
        "method": "DQN over frozen action-conditioned world-model encoder features plus goal features.",
        "reward_shaping": reward_shaping_config(args),
        "shield_front_clearance_m": args.shield_front_clearance_m,
    }
    (out_dir / "summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    return 0


def load_world_model(
    checkpoint_path: str | Path,
    *,
    device: str,
    encoder_source: str,
) -> tuple[ActionConditionedWorldModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    wm_config = dict(payload["config"])
    model = ActionConditionedWorldModel(
        num_actions=len(ACTION_VOCAB),
        image_width=int(wm_config["image_width"]),
        image_height=int(wm_config["image_height"]),
        latent_dim=int(wm_config["latent_dim"]),
    ).to(device)
    if encoder_source == "pretrained":
        model.load_state_dict(payload["model_state_dict"])
    elif encoder_source != "random":
        raise ValueError(f"Unsupported encoder source: {encoder_source!r}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, wm_config


def encode_env_feature(
    world_model: ActionConditionedWorldModel,
    image: np.ndarray,
    info: dict[str, Any],
    *,
    device: str,
) -> np.ndarray:
    with torch.no_grad():
        image_tensor = image_to_tensor(image, device=device)
        latent = world_model.encode(image_tensor)
        goal = torch.tensor([goal_features_from_info(info)], dtype=torch.float32, device=device)
        feature = torch.cat([latent, goal], dim=1)
    return feature.detach().cpu().numpy()[0].astype(np.float32)


def image_to_tensor(image: np.ndarray, *, device: str):
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.tensor(array[None, ...], dtype=torch.float32, device=device)


def goal_features_from_info(info: dict[str, Any]) -> list[float]:
    forward, right = info.get("goal_body_xy_m") or [0.0, 0.0]
    heading = math.degrees(math.atan2(float(right), max(float(forward), 1e-6)))
    return [
        float(np.clip(float(forward) / 10.0, -2.0, 2.0)),
        float(np.clip(float(right) / 5.0, -2.0, 2.0)),
        0.0,
        float(np.clip(heading / 180.0, -1.0, 1.0)),
    ]


def shape_safety_reward(
    reward: float,
    action: int,
    *,
    before_info: dict[str, Any],
    after_info: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    terms: dict[str, float] = {}
    if after_info.get("collision"):
        terms["extra_collision_penalty"] = -abs(float(args.extra_collision_penalty))
    if after_info.get("out_of_bounds"):
        terms["extra_out_of_bounds_penalty"] = -abs(float(args.extra_out_of_bounds_penalty))

    clearances = dict(after_info.get("clearance_m") or {})
    front_clearance = float(clearances.get("front_m", args.near_obstacle_threshold_m))
    near_deficit = max(0.0, float(args.near_obstacle_threshold_m) - front_clearance)
    if near_deficit > 0.0 and args.near_obstacle_penalty > 0.0:
        terms["near_obstacle_penalty"] = -float(args.near_obstacle_penalty) * near_deficit

    before_front = float(dict(before_info.get("clearance_m") or {}).get("front_m", args.forward_low_clearance_threshold_m))
    if (
        ACTION_VOCAB[int(action)] == "forward"
        and before_front < float(args.forward_low_clearance_threshold_m)
        and args.forward_low_clearance_penalty > 0.0
    ):
        deficit = float(args.forward_low_clearance_threshold_m) - before_front
        terms["forward_low_clearance_penalty"] = -float(args.forward_low_clearance_penalty) * deficit

    front_delta = float(after_info.get("front_delta_m", 0.0))
    if front_delta > 0.0 and args.clearance_recovery_bonus > 0.0:
        terms["clearance_recovery_bonus"] = float(args.clearance_recovery_bonus) * front_delta

    shaped = float(reward) + float(sum(terms.values()))
    return shaped, terms


def reward_shaping_config(args: argparse.Namespace) -> dict[str, float]:
    return {
        "extra_collision_penalty": float(args.extra_collision_penalty),
        "extra_out_of_bounds_penalty": float(args.extra_out_of_bounds_penalty),
        "near_obstacle_threshold_m": float(args.near_obstacle_threshold_m),
        "near_obstacle_penalty": float(args.near_obstacle_penalty),
        "forward_low_clearance_threshold_m": float(args.forward_low_clearance_threshold_m),
        "forward_low_clearance_penalty": float(args.forward_low_clearance_penalty),
        "clearance_recovery_bonus": float(args.clearance_recovery_bonus),
    }


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def parse_enabled_actions(raw: str) -> list[int]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("--enabled-actions must include at least one action.")
    indices = []
    for name in names:
        if name not in ACTION_VOCAB:
            raise ValueError(f"Unsupported action in --enabled-actions: {name!r}")
        indices.append(ACTION_VOCAB.index(name))
    return list(dict.fromkeys(indices))


def enabled_action_mask(enabled_action_indices: list[int]) -> np.ndarray:
    mask = np.zeros(len(ACTION_VOCAB), dtype=np.bool_)
    mask[list(enabled_action_indices)] = True
    return mask


def shielded_enabled_actions(
    info: dict[str, Any],
    enabled_action_indices: list[int],
    *,
    shield_front_clearance_m: float | None,
) -> list[int]:
    if shield_front_clearance_m is None:
        return enabled_action_indices
    forward_index = ACTION_VOCAB.index("forward")
    if forward_index not in enabled_action_indices:
        return enabled_action_indices
    front = dict(info.get("clearance_m") or {}).get("front_m")
    if front is None or float(front) >= float(shield_front_clearance_m):
        return enabled_action_indices
    shielded = [index for index in enabled_action_indices if index != forward_index]
    return shielded or enabled_action_indices


def linear_schedule(start: float, end: float, step: int, decay_steps: int) -> float:
    if decay_steps <= 0:
        return float(end)
    mix = min(1.0, max(0.0, float(step) / float(decay_steps)))
    return float(start + mix * (end - start))


def select_action(
    q_net: WorldModelDQN,
    feature: np.ndarray,
    env: DroneMazeEnv,
    *,
    enabled_action_indices: list[int],
    epsilon: float,
    expert_mix: float,
    device: str,
) -> int:
    if random.random() < expert_mix:
        expert_index = env.expert_action_index()
        if expert_index in enabled_action_indices:
            return int(expert_index)
    if random.random() < epsilon:
        return int(random.choice(enabled_action_indices))
    with torch.no_grad():
        tensor = torch.tensor(feature[None, :], dtype=torch.float32, device=device)
        q_values = q_net(tensor)
        return masked_argmax(q_values, enabled_action_indices)


def masked_argmax(q_values, enabled_action_indices: list[int]) -> int:
    mask = torch.full_like(q_values, fill_value=-1e9)
    mask[:, enabled_action_indices] = q_values[:, enabled_action_indices]
    return int(torch.argmax(mask, dim=-1).item())


def mask_disabled_actions(q_values, enabled_action_indices: list[int]):
    mask = torch.full_like(q_values, fill_value=-1e9)
    mask[:, enabled_action_indices] = q_values[:, enabled_action_indices]
    return mask


def mask_disabled_actions_tensor(q_values, action_mask):
    return torch.where(action_mask, q_values, torch.full_like(q_values, fill_value=-1e9))


def optimize_dqn(
    q_net: WorldModelDQN,
    target_net: WorldModelDQN,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    *,
    gamma: float,
    enabled_action_indices: list[int],
) -> float:
    q_values = q_net(batch["feature"]).gather(1, batch["action"].unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        if "next_action_mask" in batch:
            next_online_q = mask_disabled_actions_tensor(q_net(batch["next_feature"]), batch["next_action_mask"])
        else:
            next_online_q = mask_disabled_actions(q_net(batch["next_feature"]), enabled_action_indices)
        next_policy_actions = torch.argmax(next_online_q, dim=1)
        next_target_q = target_net(batch["next_feature"]).gather(1, next_policy_actions.unsqueeze(1)).squeeze(1)
        expected = batch["reward"] + float(gamma) * (1.0 - batch["done"]) * next_target_q
    loss = F.smooth_l1_loss(q_values, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach().cpu().item())


def evaluate_policy(
    q_net: WorldModelDQN,
    world_model: ActionConditionedWorldModel,
    env_cfg: DroneGameConfig,
    *,
    episodes: int,
    seed: int,
    device: str,
    enabled_action_indices: list[int],
    shield_front_clearance_m: float | None,
    trace_path: Path,
    trace_episodes: int,
    trace_frames: int,
) -> dict[str, Any]:
    env = DroneMazeEnv(env_cfg)
    returns = []
    lengths = []
    successes = []
    collisions = []
    timeouts = []
    out_of_bounds = []
    min_fronts = []
    action_counts: Counter[str] = Counter()
    trace_rows: list[tuple[str, list[np.ndarray]]] = []
    shielded_steps = 0

    for episode_idx in range(int(episodes)):
        obs, info = env.reset(seed=seed + episode_idx)
        done = False
        episode_return = 0.0
        step_idx = 0
        min_front = float("inf")
        frames: list[np.ndarray] = []
        while not done:
            if episode_idx < trace_episodes and len(frames) < trace_frames:
                frames.append(obs["image"])
            feature = encode_env_feature(world_model, obs["image"], info, device=device)
            with torch.no_grad():
                q_values = q_net(torch.tensor(feature[None, :], dtype=torch.float32, device=device))
                current_enabled_actions = shielded_enabled_actions(
                    info,
                    enabled_action_indices,
                    shield_front_clearance_m=shield_front_clearance_m,
                )
                if len(current_enabled_actions) != len(enabled_action_indices):
                    shielded_steps += 1
                action = masked_argmax(q_values, current_enabled_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            action_counts[ACTION_VOCAB[action]] += 1
            front = info.get("clearance_m", {}).get("front_m")
            if front is not None:
                min_front = min(min_front, float(front))
            step_idx += 1
            done = bool(terminated or truncated)

        returns.append(episode_return)
        lengths.append(step_idx)
        successes.append(1.0 if info.get("success") else 0.0)
        collisions.append(1.0 if info.get("collision") else 0.0)
        timeouts.append(1.0 if info.get("truncated") else 0.0)
        out_of_bounds.append(1.0 if info.get("out_of_bounds") else 0.0)
        if min_front != float("inf"):
            min_fronts.append(min_front)
        if frames:
            trace_rows.append((f"seed {seed + episode_idx}", frames))

    if trace_rows:
        make_trace_sheet(trace_rows, trace_path)
    return {
        "episodes": int(episodes),
        "success_rate": mean(successes),
        "collision_rate": mean(collisions),
        "timeout_rate": mean(timeouts),
        "out_of_bounds_rate": mean(out_of_bounds),
        "mean_return": mean(returns),
        "mean_length": mean(lengths),
        "mean_min_front_clearance_m": mean(min_fronts) if min_fronts else None,
        "action_counts": dict(action_counts),
        "shielded_steps": int(shielded_steps),
        "shield_front_clearance_m": shield_front_clearance_m,
        "trace_contact_sheet": str(trace_path.resolve()) if trace_rows else None,
    }


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def save_checkpoint(
    path: Path,
    q_net: WorldModelDQN,
    target_net: WorldModelDQN,
    optimizer: torch.optim.Optimizer,
    world_model: ActionConditionedWorldModel,
    *,
    args: argparse.Namespace,
    env_cfg: DroneGameConfig,
    wm_config: dict[str, Any],
    step: int,
    summary: dict[str, Any],
) -> None:
    torch.save(
        {
            "q_state_dict": q_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "world_model_checkpoint": str(Path(args.world_model_checkpoint).resolve()),
            "world_model_state_dict": {
                key: value.detach().cpu()
                for key, value in world_model.state_dict().items()
            },
            "world_model_config": wm_config,
            "encoder_source": args.encoder_source,
            "env_config": env_cfg.__dict__,
            "args": vars(args),
            "step": int(step),
            "summary": summary,
            "action_vocab": ACTION_VOCAB,
            "model_type": "frozen_world_model_dqn",
            "input_dim": int(wm_config["latent_dim"]) + 4,
            "hidden_dim": int(args.hidden_dim),
        },
        path,
    )


def write_dashboard(out_dir: Path, latest: dict[str, Any], metrics_path: Path) -> None:
    rows = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines()[-80:]:
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append(
            "<tr>"
            f"<td>{item['step']}</td>"
            f"<td>{item['epsilon']:.3f}</td>"
            f"<td>{item['expert_mix']:.3f}</td>"
            f"<td>{html_float(item.get('mean_loss_200'))}</td>"
            f"<td>{html_float(item.get('recent_train_success_rate'))}</td>"
            f"<td>{html_float(item.get('recent_train_collision_rate'))}</td>"
            f"<td>{html_float(item.get('recent_train_shaped_return'))}</td>"
            f"<td>{item['eval']['success_rate']:.3f}</td>"
            f"<td>{item['eval']['collision_rate']:.3f}</td>"
            f"<td>{item['eval']['timeout_rate']:.3f}</td>"
            f"<td>{item['eval']['mean_return']:.3f}</td>"
            "</tr>"
        )
    trace = latest.get("eval", {}).get("trace_contact_sheet")
    trace_rel = Path(trace).name if trace else ""
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>World-Model Feature DQN</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f4f1e8; color: #202020; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #c9c1ad; padding: 6px 8px; text-align: right; }}
    th {{ background: #292f25; color: white; }}
    img {{ max-width: 100%; border: 1px solid #9f967f; background: white; }}
    code {{ background: #ebe4d4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>World-Model Feature DQN</h1>
  <p>Frozen world model encoder -> latent z_t + goal features -> DQN Q-values -> action.</p>
  <p>Latest step: <code>{latest['step']}</code></p>
  <p>Eval success: <strong>{latest['eval']['success_rate']:.3f}</strong>,
     collision: <strong>{latest['eval']['collision_rate']:.3f}</strong>,
     timeout: <strong>{latest['eval']['timeout_rate']:.3f}</strong>,
     mean return: <strong>{latest['eval']['mean_return']:.3f}</strong></p>
  <h2>Latest Eval Trace</h2>
  {'<img src="' + trace_rel + '" />' if trace_rel else '<p>No trace yet.</p>'}
  <h2>Metrics</h2>
  <table>
    <thead><tr><th>Step</th><th>Epsilon</th><th>Expert Mix</th><th>Loss</th><th>Train Success</th><th>Train Collision</th><th>Train Shaped Return</th><th>Eval Success</th><th>Eval Collision</th><th>Eval Timeout</th><th>Eval Return</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item) + "\n")


def mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def mean_or_none(values) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))


def html_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
