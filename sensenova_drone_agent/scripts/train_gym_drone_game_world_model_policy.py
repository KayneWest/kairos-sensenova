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
from scripts.train_gym_drone_game_world_model import (
    ActionConditionedWorldModel,
    load_image,
    load_manifest,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object


class WorldModelPolicyDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], *, image_width: int, image_height: int):
        if torch is None:
            raise RuntimeError("torch is required for WorldModelPolicyDataset.")
        self.records = records
        self.image_width = int(image_width)
        self.image_height = int(image_height)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return {
            "image": load_image(record["image_path"], self.image_width, self.image_height),
            "goal_features": torch.tensor(extract_goal_features(record), dtype=torch.float32),
            "action_index": torch.tensor(int(record["action_index"]), dtype=torch.long),
        }


class FrozenWorldModelPolicy(nn.Module if nn is not None else object):
    def __init__(self, world_model: ActionConditionedWorldModel, *, latent_dim: int, num_actions: int):
        if nn is None:
            raise RuntimeError("torch is required for FrozenWorldModelPolicy.")
        super().__init__()
        self.world_model = world_model
        for parameter in self.world_model.parameters():
            parameter.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(int(latent_dim) + 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, int(num_actions)),
        )

    def forward(self, image, goal_features):
        with torch.no_grad():
            latent = self.world_model.encode(image)
        return self.head(torch.cat([latent, goal_features], dim=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a movement policy head on top of a frozen pixel world-model encoder.")
    parser.add_argument("--world-model-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=800000)
    parser.add_argument("--enabled-actions", default="hover,yaw_left,yaw_right,forward,strafe_left,strafe_right")
    parser.add_argument("--trace-episodes", type=int, default=8)
    parser.add_argument("--trace-frames", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    if torch is None or nn is None or F is None or DataLoader is None:
        raise RuntimeError("torch is required.")

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    wm_payload = torch.load(args.world_model_checkpoint, map_location=device)
    wm_config = dict(wm_payload["config"])
    world_model = ActionConditionedWorldModel(
        num_actions=len(ACTION_VOCAB),
        image_width=int(wm_config["image_width"]),
        image_height=int(wm_config["image_height"]),
        latent_dim=int(wm_config["latent_dim"]),
    ).to(device)
    world_model.load_state_dict(wm_payload["model_state_dict"])
    world_model.eval()

    records = load_manifest(args.manifest)
    train_records = [record for record in records if record.get("split", "train") == "train"]
    val_records = [record for record in records if record.get("split") == "val"]
    train_loader = DataLoader(
        WorldModelPolicyDataset(train_records, image_width=wm_config["image_width"], image_height=wm_config["image_height"]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        WorldModelPolicyDataset(val_records, image_width=wm_config["image_width"], image_height=wm_config["image_height"]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    policy = FrozenWorldModelPolicy(
        world_model,
        latent_dim=int(wm_config["latent_dim"]),
        num_actions=len(ACTION_VOCAB),
    ).to(device)
    optimizer = torch.optim.AdamW(policy.head.parameters(), lr=args.learning_rate)
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(policy, train_loader, device=device, optimizer=optimizer)
        val_metrics = run_epoch(policy, val_loader, device=device, optimizer=None)
        epoch_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_metrics)
        checkpoint = {
            "policy_state_dict": policy.head.state_dict(),
            "world_model_checkpoint": str(Path(args.world_model_checkpoint).resolve()),
            "config": vars(args),
            "world_model_config": wm_config,
            "epoch": epoch,
            "metrics": epoch_metrics,
            "action_vocab": ACTION_VOCAB,
            "model_type": "frozen_world_model_policy_head",
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            torch.save(checkpoint, out_dir / "best.pt")
        print(json.dumps(epoch_metrics), flush=True)

    best_payload = torch.load(out_dir / "best.pt", map_location=device)
    policy.head.load_state_dict(best_payload["policy_state_dict"])
    eval_summary = evaluate_policy(
        policy,
        image_width=int(wm_config["image_width"]),
        image_height=int(wm_config["image_height"]),
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        enabled_actions=parse_enabled_actions(args.enabled_actions),
        device=device,
        trace_path=out_dir / "trace_contact_sheet.png",
        trace_episodes=args.trace_episodes,
        trace_frames=args.trace_frames,
    )
    summary = {
        "world_model_checkpoint": str(Path(args.world_model_checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "out_dir": str(out_dir.resolve()),
        "device": device,
        "num_train": len(train_records),
        "num_val": len(val_records),
        "best_val_loss": best_val_loss,
        "history": history,
        "eval": eval_summary,
        "best_checkpoint": str((out_dir / "best.pt").resolve()),
        "last_checkpoint": str((out_dir / "last.pt").resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir, summary)
    print(json.dumps(summary, indent=2))
    return 0


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def run_epoch(policy: FrozenWorldModelPolicy, loader, *, device: str, optimizer) -> dict[str, Any]:
    train_mode = optimizer is not None
    policy.train(mode=train_mode)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in loader:
        image = batch["image"].to(device)
        goal_features = batch["goal_features"].to(device)
        action_index = batch["action_index"].to(device)
        with torch.set_grad_enabled(train_mode):
            logits = policy(image, goal_features)
            loss = F.cross_entropy(logits, action_index)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = int(image.shape[0])
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_correct += int((torch.argmax(logits, dim=1) == action_index).sum().item())
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "num_examples": total_examples,
    }


def extract_goal_features(record: dict[str, Any]) -> list[float]:
    teacher = dict(record.get("metadata", {}).get("teacher", {}))
    goal = dict(teacher.get("goal_features", {}))
    forward = float(goal.get("forward_m", 0.0))
    right = float(goal.get("right_m", 0.0))
    alt = float(goal.get("alt_error_m", 0.0))
    heading = float(goal.get("heading_error_deg", 0.0))
    return [
        float(np.clip(forward / 10.0, -2.0, 2.0)),
        float(np.clip(right / 5.0, -2.0, 2.0)),
        float(np.clip(alt / 3.0, -2.0, 2.0)),
        float(np.clip(heading / 180.0, -1.0, 1.0)),
    ]


def goal_features_from_info(info: dict[str, Any]) -> list[float]:
    forward, right = info.get("goal_body_xy_m") or [0.0, 0.0]
    heading = math.degrees(math.atan2(float(right), max(float(forward), 1e-6)))
    return [
        float(np.clip(float(forward) / 10.0, -2.0, 2.0)),
        float(np.clip(float(right) / 5.0, -2.0, 2.0)),
        0.0,
        float(np.clip(heading / 180.0, -1.0, 1.0)),
    ]


def parse_enabled_actions(raw: str) -> list[int]:
    indices = []
    for name in [item.strip() for item in raw.split(",") if item.strip()]:
        if name not in ACTION_VOCAB:
            raise ValueError(f"Unsupported action: {name!r}")
        indices.append(ACTION_VOCAB.index(name))
    if not indices:
        raise ValueError("No enabled actions.")
    return list(dict.fromkeys(indices))


def masked_argmax(logits, enabled_actions: list[int]) -> int:
    mask = torch.full_like(logits, fill_value=-1e9)
    mask[:, enabled_actions] = logits[:, enabled_actions]
    return int(torch.argmax(mask, dim=-1).item())


def evaluate_policy(
    policy: FrozenWorldModelPolicy,
    *,
    image_width: int,
    image_height: int,
    episodes: int,
    seed: int,
    enabled_actions: list[int],
    device: str,
    trace_path: Path,
    trace_episodes: int,
    trace_frames: int,
) -> dict[str, Any]:
    env = DroneMazeEnv(DroneGameConfig(image_width=image_width, image_height=image_height))
    returns = []
    lengths = []
    successes = []
    collisions = []
    timeouts = []
    action_counts: Counter[str] = Counter()
    trace_rows: list[tuple[str, list[np.ndarray]]] = []

    for episode_idx in range(episodes):
        obs, info = env.reset(seed=seed + episode_idx)
        episode_return = 0.0
        frames: list[np.ndarray] = []
        done = False
        step_idx = 0
        while not done:
            if episode_idx < trace_episodes and len(frames) < trace_frames:
                frames.append(obs["image"])
            image = image_tensor(obs["image"], device=device)
            goal = torch.tensor([goal_features_from_info(info)], dtype=torch.float32, device=device)
            with torch.no_grad():
                action = masked_argmax(policy(image, goal), enabled_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            action_counts[ACTION_VOCAB[action]] += 1
            episode_return += float(reward)
            step_idx += 1
            done = bool(terminated or truncated)
        returns.append(episode_return)
        lengths.append(step_idx)
        successes.append(1.0 if info.get("success") else 0.0)
        collisions.append(1.0 if info.get("collision") else 0.0)
        timeouts.append(1.0 if info.get("truncated") else 0.0)
        if frames:
            trace_rows.append((f"seed {seed + episode_idx}", frames))

    if trace_rows:
        make_trace_sheet(trace_rows, trace_path)
    return {
        "episodes": episodes,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "collision_rate": float(np.mean(collisions)) if collisions else 0.0,
        "timeout_rate": float(np.mean(timeouts)) if timeouts else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "action_counts": dict(action_counts),
        "trace_contact_sheet": str(trace_path.resolve()) if trace_rows else None,
    }


def image_tensor(image: np.ndarray, *, device: str):
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.tensor(array[None, ...], dtype=torch.float32, device=device)


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
    last = summary["history"][-1]
    eval_summary = summary["eval"]
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>World Model Policy Probe</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f4f1e8; color: #202020; }}
    table {{ border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #c9c1ad; padding: 6px 8px; text-align: right; }}
    th {{ background: #292f25; color: white; }}
    img {{ max-width: 100%; border: 1px solid #9f967f; background: white; }}
    code {{ background: #ebe4d4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>World Model Policy Probe</h1>
  <p>Frozen world-model encoder -> policy head -> drone action.</p>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Last Val Accuracy</td><td>{last['val']['accuracy']:.4f}</td></tr>
    <tr><td>Eval Success</td><td>{eval_summary['success_rate']:.4f}</td></tr>
    <tr><td>Eval Collision</td><td>{eval_summary['collision_rate']:.4f}</td></tr>
    <tr><td>Eval Timeout</td><td>{eval_summary['timeout_rate']:.4f}</td></tr>
    <tr><td>Mean Return</td><td>{eval_summary['mean_return']:.4f}</td></tr>
  </table>
  <h2>Closed-Loop Trace</h2>
  <img src="trace_contact_sheet.png" />
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
