#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import random
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
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


class ImageDQN(nn.Module if nn is not None else object):
    def __init__(self, in_channels: int, num_actions: int, hidden_dim: int = 256):
        if nn is None:
            raise RuntimeError("torch is required for image DQN training.")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, image):
        return self.head(self.encoder(image))


class ImageReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._items: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, image: np.ndarray, action: int, reward: float, next_image: np.ndarray, done: bool) -> None:
        self._items.append((image.astype(np.uint8), int(action), float(reward), next_image.astype(np.uint8), bool(done)))

    def sample(self, batch_size: int, device: str) -> dict[str, Any]:
        batch = random.sample(self._items, int(batch_size))
        images, actions, rewards, next_images, dones = zip(*batch)
        return {
            "image": tensor_from_uint8_stack(np.asarray(images), device=device),
            "action": torch.tensor(actions, dtype=torch.long, device=device),
            "reward": torch.tensor(rewards, dtype=torch.float32, device=device),
            "next_image": tensor_from_uint8_stack(np.asarray(next_images), device=device),
            "done": torch.tensor(dones, dtype=torch.float32, device=device),
        }


class FrameStack:
    def __init__(self, frame_stack: int):
        self.frame_stack = max(1, int(frame_stack))
        self.frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)

    def reset(self, image_hwc: np.ndarray) -> np.ndarray:
        self.frames.clear()
        frame = image_to_chw_uint8(image_hwc)
        for _ in range(self.frame_stack):
            self.frames.append(frame.copy())
        return self.as_array()

    def append(self, image_hwc: np.ndarray) -> np.ndarray:
        self.frames.append(image_to_chw_uint8(image_hwc))
        while len(self.frames) < self.frame_stack:
            self.frames.appendleft(self.frames[0].copy())
        return self.as_array()

    def as_array(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an image-based DQN on the Gym drone game.")
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "output" / "gym_drone_game_image_dqn"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--image-height", type=int, default=48)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=25000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--target-update-steps", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.08)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20000)
    parser.add_argument("--expert-mix-start", type=float, default=0.35)
    parser.add_argument("--expert-mix-end", type=float, default=0.02)
    parser.add_argument("--expert-mix-decay-steps", type=int, default=20000)
    parser.add_argument("--enabled-actions", default="hover,yaw_left,yaw_right,forward,strafe_left,strafe_right")
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--save-trace-frames", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    if torch is None or nn is None or F is None:
        raise RuntimeError("torch is required. Use the existing drone-sim environment with torch installed.")

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    env_cfg = DroneGameConfig(
        image_width=args.image_width,
        image_height=args.image_height,
        max_episode_steps=args.max_episode_steps,
    )
    env = DroneMazeEnv(env_cfg)
    obs, _ = env.reset(seed=args.seed)
    frame_stacker = FrameStack(args.frame_stack)
    stacked_image = frame_stacker.reset(obs["image"])
    enabled_action_indices = parse_enabled_actions(args.enabled_actions)

    policy = ImageDQN(stacked_image.shape[0], len(ACTION_VOCAB)).to(device)
    target = ImageDQN(stacked_image.shape[0], len(ACTION_VOCAB)).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    replay = ImageReplayBuffer(args.replay_size)

    episode_return = 0.0
    episode_length = 0
    episode_success = False
    episode_collision = False
    episode_action_counts: Counter[str] = Counter()
    completed_episodes: list[dict[str, Any]] = []
    loss_values: deque[float] = deque(maxlen=200)
    best_success_rate = -1.0

    for step in range(1, args.total_steps + 1):
        epsilon = linear_schedule(args.epsilon_start, args.epsilon_end, step, args.epsilon_decay_steps)
        expert_mix = linear_schedule(args.expert_mix_start, args.expert_mix_end, step, args.expert_mix_decay_steps)
        action = select_action(
            policy,
            stacked_image,
            env,
            enabled_action_indices=enabled_action_indices,
            epsilon=epsilon,
            expert_mix=expert_mix,
            device=device,
        )
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_stacked_image = frame_stacker.append(next_obs["image"])
        done = terminated or truncated
        replay.add(stacked_image, action, reward, next_stacked_image, done)

        episode_return += float(reward)
        episode_length += 1
        episode_success = episode_success or bool(info.get("success", False))
        episode_collision = episode_collision or bool(info.get("collision", False))
        episode_action_counts[ACTION_VOCAB[action]] += 1
        stacked_image = next_stacked_image

        if len(replay) >= max(args.warmup_steps, args.batch_size):
            batch = replay.sample(args.batch_size, device)
            loss = optimize_dqn(
                policy,
                target,
                optimizer,
                batch,
                gamma=args.gamma,
                enabled_action_indices=enabled_action_indices,
            )
            loss_values.append(loss)

        if step % args.target_update_steps == 0:
            target.load_state_dict(policy.state_dict())

        if done:
            completed_episodes.append(
                {
                    "step": step,
                    "return": episode_return,
                    "length": episode_length,
                    "success": episode_success,
                    "collision": episode_collision,
                    "action_counts": dict(episode_action_counts),
                }
            )
            obs, _ = env.reset()
            stacked_image = frame_stacker.reset(obs["image"])
            episode_return = 0.0
            episode_length = 0
            episode_success = False
            episode_collision = False
            episode_action_counts = Counter()

        if step == 1 or step % args.eval_every == 0 or step == args.total_steps:
            eval_summary = evaluate_policy(
                policy,
                env_cfg,
                frame_stack=args.frame_stack,
                episodes=args.eval_episodes,
                seed=args.seed + step,
                device=device,
                enabled_action_indices=enabled_action_indices,
                trace_path=out_dir / "latest_eval_trace.png",
                max_trace_frames=args.save_trace_frames,
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
                "eval": eval_summary,
            }
            append_jsonl(metrics_path, train_summary)
            save_checkpoint(out_dir / "last.pt", policy, target, optimizer, env_cfg, args, step, train_summary)
            if eval_summary["success_rate"] >= best_success_rate:
                best_success_rate = float(eval_summary["success_rate"])
                save_checkpoint(out_dir / "best.pt", policy, target, optimizer, env_cfg, args, step, train_summary)
            write_dashboard(out_dir, train_summary, metrics_path)
            print(json.dumps(train_summary, indent=2), flush=True)

    final_summary = {
        "out_dir": str(out_dir.resolve()),
        "total_steps": args.total_steps,
        "best_success_rate": best_success_rate,
        "num_completed_train_episodes": len(completed_episodes),
        "latest_checkpoint": str((out_dir / "last.pt").resolve()),
        "best_checkpoint": str((out_dir / "best.pt").resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "dashboard_path": str((out_dir / "index.html").resolve()),
        "env_config": env_cfg.__dict__,
        "frame_stack": args.frame_stack,
        "enabled_actions": [ACTION_VOCAB[index] for index in enabled_action_indices],
    }
    (out_dir / "summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    return 0


def image_to_chw_uint8(image_hwc: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(image_hwc, dtype=np.uint8), (2, 0, 1))


def tensor_from_uint8_stack(array: np.ndarray, *, device: str):
    return torch.tensor(array, dtype=torch.float32, device=device) / 255.0


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def linear_schedule(start: float, end: float, step: int, decay_steps: int) -> float:
    if decay_steps <= 0:
        return float(end)
    mix = min(1.0, max(0.0, float(step) / float(decay_steps)))
    return float(start + mix * (end - start))


def parse_enabled_actions(raw: str) -> list[int]:
    indices = []
    for name in [item.strip() for item in raw.split(",") if item.strip()]:
        if name not in ACTION_VOCAB:
            raise ValueError(f"Unsupported action in --enabled-actions: {name!r}")
        indices.append(ACTION_VOCAB.index(name))
    if not indices:
        raise ValueError("--enabled-actions must include at least one action.")
    return list(dict.fromkeys(indices))


def select_action(
    policy: ImageDQN,
    image: np.ndarray,
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
            return expert_index
    if random.random() < epsilon:
        return int(random.choice(enabled_action_indices))
    with torch.no_grad():
        q_values = policy(tensor_from_uint8_stack(image[None, ...], device=device))
        return int(torch.argmax(mask_disabled_actions(q_values, enabled_action_indices), dim=-1).item())


def mask_disabled_actions(logits, enabled_action_indices: list[int]):
    mask = torch.full_like(logits, fill_value=-1e9)
    mask[:, enabled_action_indices] = logits[:, enabled_action_indices]
    return mask


def optimize_dqn(
    policy: ImageDQN,
    target: ImageDQN,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    *,
    gamma: float,
    enabled_action_indices: list[int],
) -> float:
    q_values = policy(batch["image"]).gather(1, batch["action"].unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = mask_disabled_actions(target(batch["next_image"]), enabled_action_indices).max(dim=1).values
        expected = batch["reward"] + float(gamma) * (1.0 - batch["done"]) * next_q
    loss = F.smooth_l1_loss(q_values, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach().cpu().item())


def evaluate_policy(
    policy: ImageDQN,
    env_cfg: DroneGameConfig,
    *,
    frame_stack: int,
    episodes: int,
    seed: int,
    device: str,
    enabled_action_indices: list[int],
    trace_path: Path,
    max_trace_frames: int,
) -> dict[str, Any]:
    env = DroneMazeEnv(env_cfg)
    stacker = FrameStack(frame_stack)
    returns = []
    lengths = []
    successes = []
    collisions = []
    action_counts: Counter[str] = Counter()
    trace_frames: list[np.ndarray] = []

    for episode_idx in range(int(episodes)):
        obs, _ = env.reset(seed=seed + episode_idx)
        stacked_image = stacker.reset(obs["image"])
        episode_return = 0.0
        done = False
        step_idx = 0
        info: dict[str, Any] = {}
        while not done:
            if episode_idx == 0 and len(trace_frames) < max_trace_frames:
                trace_frames.append(obs["image"])
            with torch.no_grad():
                q_values = policy(tensor_from_uint8_stack(stacked_image[None, ...], device=device))
                action = int(torch.argmax(mask_disabled_actions(q_values, enabled_action_indices), dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            stacked_image = stacker.append(obs["image"])
            episode_return += float(reward)
            action_counts[ACTION_VOCAB[action]] += 1
            step_idx += 1
            done = bool(terminated or truncated)
        returns.append(episode_return)
        lengths.append(step_idx)
        successes.append(1.0 if info.get("success") else 0.0)
        collisions.append(1.0 if info.get("collision") else 0.0)

    if trace_frames:
        save_trace_contact_sheet(trace_frames, trace_path)

    return {
        "episodes": int(episodes),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "collision_rate": float(np.mean(collisions)) if collisions else 0.0,
        "action_counts": dict(action_counts),
        "trace_path": str(trace_path.resolve()) if trace_frames else None,
    }


def save_trace_contact_sheet(frames: list[np.ndarray], out_path: Path) -> None:
    pil_frames = [Image.fromarray(frame.astype(np.uint8), mode="RGB") for frame in frames]
    width, height = pil_frames[0].size
    label_h = 16
    sheet = Image.new("RGB", (width * len(pil_frames), height + label_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for idx, frame in enumerate(pil_frames):
        x = idx * width
        sheet.paste(frame, (x, label_h))
        draw.text((x + 3, 2), f"t={idx}", fill=(240, 240, 240))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def save_checkpoint(
    path: Path,
    policy: ImageDQN,
    target: ImageDQN,
    optimizer: torch.optim.Optimizer,
    env_cfg: DroneGameConfig,
    args: argparse.Namespace,
    step: int,
    summary: dict[str, Any],
) -> None:
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "target_state_dict": target.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "env_config": env_cfg.__dict__,
            "args": vars(args),
            "step": int(step),
            "action_vocab": ACTION_VOCAB,
            "summary": summary,
            "model_type": "image_dqn",
        },
        path,
    )


def write_dashboard(out_dir: Path, latest: dict[str, Any], metrics_path: Path) -> None:
    rows = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines()[-50:]:
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
            f"<td>{item['eval']['success_rate']:.3f}</td>"
            f"<td>{item['eval']['collision_rate']:.3f}</td>"
            f"<td>{item['eval']['mean_return']:.3f}</td>"
            "</tr>"
        )
    trace = latest.get("eval", {}).get("trace_path")
    trace_rel = Path(trace).name if trace else ""
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Image DQN Drone Game</title>
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
  <h1>Image DQN Drone Game</h1>
  <p>Latest step: <code>{latest['step']}</code></p>
  <p>Eval success: <strong>{latest['eval']['success_rate']:.3f}</strong>,
     collision: <strong>{latest['eval']['collision_rate']:.3f}</strong>,
     mean return: <strong>{latest['eval']['mean_return']:.3f}</strong></p>
  <h2>Latest Eval Trace</h2>
  {'<img src="' + trace_rel + '" />' if trace_rel else '<p>No trace yet.</p>'}
  <h2>Metrics</h2>
  <table>
    <thead><tr><th>Step</th><th>Epsilon</th><th>Expert Mix</th><th>Loss</th><th>Train Success</th><th>Eval Success</th><th>Eval Collision</th><th>Eval Return</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item) + "\n")


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
