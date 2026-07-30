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
from scripts.train_gym_drone_game_world_model_dqn import (
    enabled_action_mask,
    goal_features_from_info,
    html_float,
    linear_schedule,
    mean,
    mean_or_none,
    parse_enabled_actions,
    resolve_device,
    shielded_enabled_actions,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


DEFAULT_ENABLED_ACTIONS = "hover,yaw_left,yaw_right,forward,strafe_left,strafe_right"


class ResidualBlock(nn.Module if nn is not None else object):
    def __init__(self, channels: int):
        if nn is None:
            raise RuntimeError("torch is required for ResidualBlock.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return F.relu(x + self.net(x), inplace=True)


class CnnGoalDQN(nn.Module if nn is not None else object):
    """
    Generic strong pixel-control baseline.

    This is not pretrained. It learns image features end-to-end from environment reward.
    """

    def __init__(self, num_actions: int, hidden_dim: int = 256, goal_dim: int = 4):
        if nn is None:
            raise RuntimeError("torch is required for CnnGoalDQN.")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + int(goal_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, int(num_actions)),
        )

    def forward(self, image, goal_features):
        return self.head(torch.cat([self.encoder(image), goal_features], dim=1))


class PixelReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._items: deque[tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, bool, np.ndarray]] = deque(
            maxlen=self.capacity
        )

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        image: np.ndarray,
        goal_features: list[float],
        action: int,
        reward: float,
        next_image: np.ndarray,
        next_goal_features: list[float],
        done: bool,
        next_enabled_actions: list[int],
    ) -> None:
        self._items.append(
            (
                np.asarray(image, dtype=np.uint8),
                np.asarray(goal_features, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_image, dtype=np.uint8),
                np.asarray(next_goal_features, dtype=np.float32),
                bool(done),
                enabled_action_mask(next_enabled_actions),
            )
        )

    def sample(self, batch_size: int, device: str, *, random_shift_pixels: int) -> dict[str, Any]:
        batch = random.sample(self._items, int(batch_size))
        images, goals, actions, rewards, next_images, next_goals, dones, next_action_masks = zip(*batch)
        image = tensor_from_hwc_uint8(np.asarray(images), device=device)
        next_image = tensor_from_hwc_uint8(np.asarray(next_images), device=device)
        if random_shift_pixels > 0:
            image = random_shift(image, int(random_shift_pixels))
            next_image = random_shift(next_image, int(random_shift_pixels))
        return {
            "image": image,
            "goal": torch.tensor(np.asarray(goals), dtype=torch.float32, device=device),
            "action": torch.tensor(actions, dtype=torch.long, device=device),
            "reward": torch.tensor(rewards, dtype=torch.float32, device=device),
            "next_image": next_image,
            "next_goal": torch.tensor(np.asarray(next_goals), dtype=torch.float32, device=device),
            "done": torch.tensor(dones, dtype=torch.float32, device=device),
            "next_action_mask": torch.tensor(np.asarray(next_action_masks), dtype=torch.bool, device=device),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a generic CNN+goal DQN baseline on the Gym drone game.")
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "output" / "gym_drone_game_cnn_dqn"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--image-height", type=int, default=48)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--world-size-m", type=float, default=16.0)
    parser.add_argument("--obstacle-count", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--target-update-steps", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10000)
    parser.add_argument("--expert-mix-start", type=float, default=0.10)
    parser.add_argument("--expert-mix-end", type=float, default=0.0)
    parser.add_argument("--expert-mix-decay-steps", type=int, default=10000)
    parser.add_argument("--shield-front-clearance-m", type=float, default=None)
    parser.add_argument("--random-shift-pixels", type=int, default=4)
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
    enabled_actions = parse_enabled_actions(args.enabled_actions)
    env_cfg = DroneGameConfig(
        world_size_m=args.world_size_m,
        obstacle_count=args.obstacle_count,
        image_width=args.image_width,
        image_height=args.image_height,
        max_episode_steps=args.max_episode_steps,
    )
    q_net = CnnGoalDQN(len(ACTION_VOCAB), hidden_dim=args.hidden_dim).to(device)
    target_net = CnnGoalDQN(len(ACTION_VOCAB), hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=args.learning_rate)
    replay = PixelReplayBuffer(args.replay_size)

    env = DroneMazeEnv(env_cfg)
    obs, info = env.reset(seed=args.seed)
    completed_episodes: list[dict[str, Any]] = []
    loss_values: deque[float] = deque(maxlen=200)
    episode_return = 0.0
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
            obs["image"],
            goal_features_from_info(info),
            env,
            enabled_action_indices=current_enabled_actions,
            epsilon=epsilon,
            expert_mix=expert_mix,
            device=device,
        )
        before_image = obs["image"]
        before_goal = goal_features_from_info(info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        next_enabled_actions = shielded_enabled_actions(
            info,
            enabled_actions,
            shield_front_clearance_m=args.shield_front_clearance_m,
        )
        replay.add(
            before_image,
            before_goal,
            action,
            reward,
            obs["image"],
            goal_features_from_info(info),
            done,
            next_enabled_actions,
        )

        episode_return += float(reward)
        episode_length += 1
        episode_success = episode_success or bool(info.get("success", False))
        episode_collision = episode_collision or bool(info.get("collision", False))
        episode_timeout = episode_timeout or bool(info.get("truncated", False))
        episode_actions[ACTION_VOCAB[action]] += 1

        if len(replay) >= max(args.warmup_steps, args.batch_size):
            batch = replay.sample(args.batch_size, device, random_shift_pixels=args.random_shift_pixels)
            loss = optimize_dqn(q_net, target_net, optimizer, batch, gamma=args.gamma)
            loss_values.append(loss)

        if step % args.target_update_steps == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            completed_episodes.append(
                {
                    "step": step,
                    "return": episode_return,
                    "length": episode_length,
                    "success": episode_success,
                    "collision": episode_collision,
                    "timeout": episode_timeout,
                    "action_counts": dict(episode_actions),
                }
            )
            obs, info = env.reset()
            episode_return = 0.0
            episode_length = 0
            episode_success = False
            episode_collision = False
            episode_timeout = False
            episode_actions = Counter()

        if step == 1 or step % args.eval_every == 0 or step == args.total_steps:
            eval_summary = evaluate_policy(
                q_net,
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
                "recent_train_success_rate": mean_or_none([1.0 if episode["success"] else 0.0 for episode in completed_episodes[-20:]]),
                "recent_train_collision_rate": mean_or_none([1.0 if episode["collision"] else 0.0 for episode in completed_episodes[-20:]]),
                "train_shielded_steps": train_shielded_steps,
                "shield_front_clearance_m": args.shield_front_clearance_m,
                "eval": eval_summary,
            }
            append_jsonl(metrics_path, train_summary)
            save_checkpoint(out_dir / "last.pt", q_net, target_net, optimizer, env_cfg, args, step, train_summary)
            if eval_summary["success_rate"] >= best_success_rate:
                best_success_rate = float(eval_summary["success_rate"])
                save_checkpoint(out_dir / "best.pt", q_net, target_net, optimizer, env_cfg, args, step, train_summary)
            write_dashboard(out_dir, train_summary, metrics_path)
            print(json.dumps(train_summary, indent=2), flush=True)

    final_summary = {
        "out_dir": str(out_dir.resolve()),
        "total_steps": int(args.total_steps),
        "best_success_rate": best_success_rate,
        "num_completed_train_episodes": len(completed_episodes),
        "latest_checkpoint": str((out_dir / "last.pt").resolve()),
        "best_checkpoint": str((out_dir / "best.pt").resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "dashboard_path": str((out_dir / "index.html").resolve()),
        "env_config": env_cfg.__dict__,
        "model_type": "cnn_goal_dqn",
        "enabled_actions": [ACTION_VOCAB[index] for index in enabled_actions],
        "shield_front_clearance_m": args.shield_front_clearance_m,
        "random_shift_pixels": args.random_shift_pixels,
    }
    (out_dir / "summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    return 0


def select_action(
    q_net: CnnGoalDQN,
    image: np.ndarray,
    goal_features: list[float],
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
        q_values = q_net(
            tensor_from_hwc_uint8(np.asarray([image]), device=device),
            torch.tensor([goal_features], dtype=torch.float32, device=device),
        )
        return masked_argmax(q_values, enabled_action_indices)


def optimize_dqn(
    q_net: CnnGoalDQN,
    target_net: CnnGoalDQN,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    *,
    gamma: float,
) -> float:
    q_values = q_net(batch["image"], batch["goal"]).gather(1, batch["action"].unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_online_q = mask_disabled_actions_tensor(q_net(batch["next_image"], batch["next_goal"]), batch["next_action_mask"])
        next_policy_actions = torch.argmax(next_online_q, dim=1)
        next_target_q = target_net(batch["next_image"], batch["next_goal"]).gather(1, next_policy_actions.unsqueeze(1)).squeeze(1)
        expected = batch["reward"] + float(gamma) * (1.0 - batch["done"]) * next_target_q
    loss = F.smooth_l1_loss(q_values, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach().cpu().item())


def evaluate_policy(
    q_net: CnnGoalDQN,
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
            current_enabled_actions = shielded_enabled_actions(
                info,
                enabled_action_indices,
                shield_front_clearance_m=shield_front_clearance_m,
            )
            if len(current_enabled_actions) != len(enabled_action_indices):
                shielded_steps += 1
            with torch.no_grad():
                q_values = q_net(
                    tensor_from_hwc_uint8(np.asarray([obs["image"]]), device=device),
                    torch.tensor([goal_features_from_info(info)], dtype=torch.float32, device=device),
                )
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


def tensor_from_hwc_uint8(array: np.ndarray, *, device: str):
    chw = np.transpose(np.asarray(array, dtype=np.uint8), (0, 3, 1, 2))
    return torch.tensor(chw, dtype=torch.float32, device=device) / 255.0


def random_shift(image, pad: int):
    if pad <= 0:
        return image
    padded = F.pad(image, (pad, pad, pad, pad), mode="replicate")
    batch, _, height, width = image.shape
    out = torch.empty_like(image)
    for idx in range(batch):
        top = int(torch.randint(0, pad * 2 + 1, (1,), device=image.device).item())
        left = int(torch.randint(0, pad * 2 + 1, (1,), device=image.device).item())
        out[idx] = padded[idx, :, top : top + height, left : left + width]
    return out


def masked_argmax(q_values, enabled_action_indices: list[int]) -> int:
    mask = torch.full_like(q_values, fill_value=-1e9)
    mask[:, enabled_action_indices] = q_values[:, enabled_action_indices]
    return int(torch.argmax(mask, dim=-1).item())


def mask_disabled_actions_tensor(q_values, action_mask):
    return torch.where(action_mask, q_values, torch.full_like(q_values, fill_value=-1e9))


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
    q_net: CnnGoalDQN,
    target_net: CnnGoalDQN,
    optimizer: torch.optim.Optimizer,
    env_cfg: DroneGameConfig,
    args: argparse.Namespace,
    step: int,
    summary: dict[str, Any],
) -> None:
    torch.save(
        {
            "q_state_dict": q_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "env_config": env_cfg.__dict__,
            "args": vars(args),
            "step": int(step),
            "summary": summary,
            "action_vocab": ACTION_VOCAB,
            "model_type": "cnn_goal_dqn",
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
  <title>CNN Goal DQN</title>
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
  <h1>CNN Goal DQN</h1>
  <p>Generic CNN from pixels + goal features -> DQN action. No world-model pretraining.</p>
  <p>Latest step: <code>{latest['step']}</code></p>
  <p>Eval success: <strong>{latest['eval']['success_rate']:.3f}</strong>,
     collision: <strong>{latest['eval']['collision_rate']:.3f}</strong>,
     timeout: <strong>{latest['eval']['timeout_rate']:.3f}</strong>,
     mean return: <strong>{latest['eval']['mean_return']:.3f}</strong></p>
  <h2>Latest Eval Trace</h2>
  {'<img src="' + trace_rel + '" />' if trace_rel else '<p>No trace yet.</p>'}
  <h2>Metrics</h2>
  <table>
    <thead><tr><th>Step</th><th>Epsilon</th><th>Expert Mix</th><th>Loss</th><th>Train Success</th><th>Train Collision</th><th>Eval Success</th><th>Eval Collision</th><th>Eval Timeout</th><th>Eval Return</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
