#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import shutil
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
from scripts.train_gym_drone_game_world_model import ActionConditionedWorldModel, load_image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None
    Categorical = None
    DataLoader = None
    Dataset = object


@dataclass
class Dreamer4LiteConfig:
    world_model_checkpoint: str
    bc_manifest: str
    risk_manifest: str
    out_dir: str
    supervised_epochs: int = 3
    imagination_updates: int = 500
    batch_size: int = 128
    learning_rate: float = 1e-3
    imagination_learning_rate: float = 3e-4
    imagination_horizon: int = 8
    gamma: float = 0.97
    bc_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    risk_loss_weight: float = 0.5
    kl_to_prior_weight: float = 0.3
    entropy_weight: float = 0.01
    collision_penalty: float = 8.0
    out_of_bounds_penalty: float = 5.0
    reward_scale: float = 12.0
    eval_episodes: int = 128
    eval_seed: int = 920000
    device: str = "auto"
    seed: int = 0
    num_workers: int = 0
    enabled_actions: str = "hover,yaw_left,yaw_right,forward,strafe_left,strafe_right"


class BCDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], *, image_width: int, image_height: int):
        if torch is None:
            raise RuntimeError("torch is required for BCDataset.")
        self.records = records
        self.image_width = int(image_width)
        self.image_height = int(image_height)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        metadata = dict(record.get("metadata", {}))
        teacher = dict(metadata.get("teacher", {}))
        after = dict(teacher.get("after", {}))
        reward = float(metadata.get("reward", 0.0))
        return {
            "image": load_record_image(record["image_path"], self.image_width, self.image_height),
            "goal": torch.tensor(goal_features_from_record(record), dtype=torch.float32),
            "action": torch.tensor(int(record["action_index"]), dtype=torch.long),
            "reward_norm": torch.tensor(np.clip(reward / 12.0, -1.0, 1.0), dtype=torch.float32),
            "success": torch.tensor(1.0 if after.get("success", False) else 0.0, dtype=torch.float32),
            "collision": torch.tensor(1.0 if after.get("collision", False) else 0.0, dtype=torch.float32),
            "out_of_bounds": torch.tensor(0.0, dtype=torch.float32),
            "value_norm": torch.tensor(np.clip(reward / 12.0, -1.0, 1.0), dtype=torch.float32),
        }


class RiskDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        image_width: int,
        image_height: int,
        value_targets: dict[tuple[str, int], float],
        reward_scale: float,
    ):
        if torch is None:
            raise RuntimeError("torch is required for RiskDataset.")
        self.records = records
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.value_targets = value_targets
        self.reward_scale = float(reward_scale)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        labels = dict(record["labels"])
        key = (str(record["episode_id"]), int(record["step_index"]))
        value_target = self.value_targets.get(key, float(labels.get("utility", labels.get("reward", 0.0))))
        return {
            "image": load_record_image(record["image_path"], self.image_width, self.image_height),
            "goal": torch.tensor(record.get("goal_features") or [0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "candidate_action": torch.tensor(int(record["candidate_action_index"]), dtype=torch.long),
            "best_action": torch.tensor(int(record["best_action_index"]), dtype=torch.long),
            "reward_norm": torch.tensor(
                float(np.clip(float(labels.get("utility", labels.get("reward", 0.0))) / max(self.reward_scale, 1e-6), -1.0, 1.0)),
                dtype=torch.float32,
            ),
            "value_norm": torch.tensor(float(np.clip(value_target / max(self.reward_scale, 1e-6), -1.0, 1.0)), dtype=torch.float32),
            "collision": torch.tensor(1.0 if labels.get("collision", False) else 0.0, dtype=torch.float32),
            "success": torch.tensor(1.0 if labels.get("success", False) else 0.0, dtype=torch.float32),
            "out_of_bounds": torch.tensor(1.0 if labels.get("out_of_bounds", False) else 0.0, dtype=torch.float32),
        }


class Dreamer4LiteAgent(nn.Module if nn is not None else object):
    def __init__(
        self,
        world_model: ActionConditionedWorldModel,
        *,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        action_embed_dim: int = 32,
    ):
        if nn is None:
            raise RuntimeError("torch is required for Dreamer4LiteAgent.")
        super().__init__()
        self.world_model = world_model
        for parameter in self.world_model.parameters():
            parameter.requires_grad_(False)
        self.num_actions = int(num_actions)
        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
        state_dim = int(latent_dim) + 4
        critic_input_dim = state_dim + action_embed_dim
        self.prior_policy = make_mlp(state_dim, hidden_dim, num_actions)
        self.policy = make_mlp(state_dim, hidden_dim, num_actions)
        self.value = make_mlp(state_dim, hidden_dim, 1)
        self.reward = make_mlp(critic_input_dim, hidden_dim, 1)
        self.collision = make_mlp(critic_input_dim, hidden_dim, 1)
        self.success = make_mlp(critic_input_dim, hidden_dim, 1)
        self.out_of_bounds = make_mlp(critic_input_dim, hidden_dim, 1)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.world_model.encode(image)

    def state_features(self, latent: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.cat([latent, goal], dim=1)

    def action_features(self, latent: torch.Tensor, goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.cat([latent, goal, self.action_embedding(action.long())], dim=1)

    def forward_from_image(self, image: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode_image(image)
        return self.forward_from_latent(latent, goal)

    def forward_from_latent(self, latent: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.state_features(latent, goal)
        return {
            "latent": latent,
            "prior_logits": self.prior_policy(state),
            "policy_logits": self.policy(state),
            "value_norm": self.value(state).squeeze(-1),
        }

    def predict_action_outcomes(
        self,
        latent: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.action_features(latent, goal, action)
        return {
            "reward_norm": self.reward(features).squeeze(-1),
            "collision_logit": self.collision(features).squeeze(-1),
            "success_logit": self.success(features).squeeze(-1),
            "out_of_bounds_logit": self.out_of_bounds(features).squeeze(-1),
        }

    def imagine_next_latent(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_embed = self.world_model.action_embed(action.long())
        return self.world_model.transition(torch.cat([latent, action_embed], dim=1))


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dreamer4-lite training on Sensenova world-model features.")
    parser.add_argument("--world-model-checkpoint", default="sensenova_drone_agent/output/gym_drone_game_world_model_v1/best.pt")
    parser.add_argument("--bc-manifest", default="sensenova_drone_agent/data/gym_drone_game_dqn_teacher_v2_all/manifests/bc_manifest.jsonl")
    parser.add_argument("--risk-manifest", default="sensenova_drone_agent/data/gym_drone_game_action_risk_v2/manifests/action_risk_manifest.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--supervised-epochs", type=int, default=3)
    parser.add_argument("--imagination-updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--imagination-learning-rate", type=float, default=3e-4)
    parser.add_argument("--imagination-horizon", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--kl-to-prior-weight", type=float, default=0.3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--collision-penalty", type=float, default=8.0)
    parser.add_argument("--out-of-bounds-penalty", type=float, default=5.0)
    parser.add_argument("--reward-scale", type=float, default=12.0)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=920000)
    parser.add_argument("--enabled-actions", default="hover,yaw_left,yaw_right,forward,strafe_left,strafe_right")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    if torch is None or nn is None or F is None or DataLoader is None or Categorical is None:
        raise RuntimeError("torch is required for Dreamer4-lite training.")
    args = parse_args()
    config = Dreamer4LiteConfig(**vars(args))
    summary = train(config)
    print(json.dumps(summary, indent=2))
    return 0


def train(config: Dreamer4LiteConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    wm_payload = safe_torch_load(config.world_model_checkpoint, device)
    wm_config = dict(wm_payload["config"])
    image_width = int(wm_config["image_width"])
    image_height = int(wm_config["image_height"])
    latent_dim = int(wm_config["latent_dim"])
    world_model = ActionConditionedWorldModel(
        num_actions=len(ACTION_VOCAB),
        image_width=image_width,
        image_height=image_height,
        latent_dim=latent_dim,
    ).to(device)
    world_model.load_state_dict(wm_payload["model_state_dict"])
    world_model.eval()

    bc_records = load_jsonl(config.bc_manifest)
    risk_records = load_jsonl(config.risk_manifest)
    train_bc = [record for record in bc_records if record.get("split", "train") == "train"]
    val_bc = [record for record in bc_records if record.get("split") == "val"]
    train_risk = [record for record in risk_records if record.get("split", "train") == "train"]
    val_risk = [record for record in risk_records if record.get("split") == "val"]
    value_targets = build_value_targets(risk_records)
    if not train_bc:
        raise RuntimeError("No train examples in BC manifest.")
    if not train_risk:
        raise RuntimeError("No train examples in risk manifest.")

    train_bc_loader = DataLoader(
        BCDataset(train_bc, image_width=image_width, image_height=image_height),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_bc_loader = DataLoader(
        BCDataset(val_bc, image_width=image_width, image_height=image_height),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    ) if val_bc else None
    train_risk_loader = DataLoader(
        RiskDataset(train_risk, image_width=image_width, image_height=image_height, value_targets=value_targets, reward_scale=config.reward_scale),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_risk_loader = DataLoader(
        RiskDataset(val_risk, image_width=image_width, image_height=image_height, value_targets=value_targets, reward_scale=config.reward_scale),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    ) if val_risk else None

    agent = Dreamer4LiteAgent(
        world_model,
        latent_dim=latent_dim,
        num_actions=len(ACTION_VOCAB),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            *agent.prior_policy.parameters(),
            *agent.policy.parameters(),
            *agent.value.parameters(),
            *agent.reward.parameters(),
            *agent.collision.parameters(),
            *agent.success.parameters(),
            *agent.out_of_bounds.parameters(),
        ],
        lr=config.learning_rate,
    )
    history: list[dict[str, Any]] = []
    best_val = float("inf")

    for epoch in range(1, config.supervised_epochs + 1):
        train_metrics = run_supervised_epoch(
            agent,
            bc_loader=train_bc_loader,
            risk_loader=train_risk_loader,
            device=device,
            optimizer=optimizer,
            config=config,
        )
        val_metrics = run_supervised_epoch(
            agent,
            bc_loader=val_bc_loader,
            risk_loader=val_risk_loader,
            device=device,
            optimizer=None,
            config=config,
        )
        epoch_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_metrics)
        val_loss = float(val_metrics.get("loss", train_metrics["loss"]))
        payload = checkpoint_payload(agent, config, wm_config, epoch, epoch_metrics)
        torch.save(payload, out_dir / "last_supervised.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, out_dir / "best_supervised.pt")
        print(json.dumps({"phase": "supervised", **epoch_metrics}), flush=True)

    agent.load_state_dict(safe_torch_load(out_dir / "best_supervised.pt", device)["agent_state_dict"])
    enabled_actions = parse_enabled_actions(config.enabled_actions)
    supervised_eval = evaluate_agent(
        agent,
        device=device,
        image_width=image_width,
        image_height=image_height,
        episodes=config.eval_episodes,
        seed=config.eval_seed,
        enabled_actions=enabled_actions,
        trace_path=out_dir / "supervised_trace_contact_sheet.png",
    )
    print(json.dumps({"phase": "supervised_eval", **supervised_eval}), flush=True)

    imagination_history = run_imagination_training(
        agent,
        seed_loader=train_bc_loader,
        device=device,
        config=config,
        enabled_actions=enabled_actions,
    )
    final_payload = checkpoint_payload(
        agent,
        config,
        wm_config,
        int(config.supervised_epochs),
        {"supervised_history": history, "imagination_history": imagination_history},
    )
    torch.save(final_payload, out_dir / "dreamer4_lite_final.pt")
    final_eval = evaluate_agent(
        agent,
        device=device,
        image_width=image_width,
        image_height=image_height,
        episodes=config.eval_episodes,
        seed=config.eval_seed + 10000,
        enabled_actions=enabled_actions,
        trace_path=out_dir / "final_trace_contact_sheet.png",
    )
    supervised_score = deployment_score(supervised_eval)
    final_score = deployment_score(final_eval)
    if final_score >= supervised_score:
        recommended_phase = "imagination"
        recommended_checkpoint = out_dir / "dreamer4_lite_final.pt"
    else:
        recommended_phase = "supervised"
        recommended_checkpoint = out_dir / "best_supervised.pt"
    selected_checkpoint = out_dir / "selected_checkpoint.pt"
    shutil.copy2(recommended_checkpoint, selected_checkpoint)

    summary = {
        "method": "dreamer4_lite",
        "interpretation": (
            "Frozen Sensenova world-model encoder/dynamics + supervised BC/reward/value heads "
            "+ KL-constrained imagination fine-tuning."
        ),
        "world_model_checkpoint": str(Path(config.world_model_checkpoint).resolve()),
        "bc_manifest": str(Path(config.bc_manifest).resolve()),
        "risk_manifest": str(Path(config.risk_manifest).resolve()),
        "out_dir": str(out_dir.resolve()),
        "device": device,
        "num_bc_train": len(train_bc),
        "num_bc_val": len(val_bc),
        "num_risk_train": len(train_risk),
        "num_risk_val": len(val_risk),
        "enabled_actions": [ACTION_VOCAB[index] for index in enabled_actions],
        "supervised_history": history,
        "supervised_eval": supervised_eval,
        "imagination_history": imagination_history,
        "final_eval": final_eval,
        "supervised_deployment_score": supervised_score,
        "final_deployment_score": final_score,
        "recommended_phase": recommended_phase,
        "imagination_rl_ready": bool(recommended_phase == "imagination"),
        "best_supervised_checkpoint": str((out_dir / "best_supervised.pt").resolve()),
        "final_checkpoint": str((out_dir / "dreamer4_lite_final.pt").resolve()),
        "selected_checkpoint": str(selected_checkpoint.resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir, summary)
    return summary


def deployment_score(eval_summary: dict[str, Any]) -> float:
    return (
        100.0 * float(eval_summary.get("success_rate", 0.0))
        + float(eval_summary.get("mean_return", 0.0))
        - 50.0 * float(eval_summary.get("collision_rate", 0.0))
        - 10.0 * float(eval_summary.get("timeout_rate", 0.0))
    )


def run_supervised_epoch(
    agent: Dreamer4LiteAgent,
    *,
    bc_loader,
    risk_loader,
    device: str,
    optimizer,
    config: Dreamer4LiteConfig,
) -> dict[str, Any]:
    train_mode = optimizer is not None
    agent.train(mode=train_mode)
    totals = Counter()
    bc_iter = iter(bc_loader) if bc_loader is not None else None
    risk_iter = iter(risk_loader) if risk_loader is not None else None
    steps = max(len(bc_loader) if bc_loader is not None else 0, len(risk_loader) if risk_loader is not None else 0)
    if steps <= 0:
        return {"loss": 0.0, "num_examples": 0}

    for _ in range(steps):
        losses = []
        if bc_iter is not None:
            try:
                bc_batch = next(bc_iter)
            except StopIteration:
                bc_iter = iter(bc_loader)
                bc_batch = next(bc_iter)
            bc_loss, bc_metrics = supervised_bc_loss(agent, bc_batch, device=device)
            losses.append(config.bc_loss_weight * bc_loss)
            for key, value in bc_metrics.items():
                totals[key] += value
        if risk_iter is not None:
            try:
                risk_batch = next(risk_iter)
            except StopIteration:
                risk_iter = iter(risk_loader)
                risk_batch = next(risk_iter)
            risk_loss, risk_metrics = supervised_risk_loss(agent, risk_batch, device=device, config=config)
            losses.append(risk_loss)
            for key, value in risk_metrics.items():
                totals[key] += value
        if not losses:
            continue
        loss = sum(losses)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=5.0)
            optimizer.step()
        totals["loss"] += float(loss.item())
        totals["steps"] += 1

    denom = max(int(totals["steps"]), 1)
    metrics = {key: float(value) / denom for key, value in totals.items() if key != "steps"}
    metrics["num_steps"] = denom
    return metrics


def supervised_bc_loss(agent: Dreamer4LiteAgent, batch: dict[str, Any], *, device: str):
    image = batch["image"].to(device)
    goal = batch["goal"].to(device)
    action = batch["action"].to(device)
    outputs = agent.forward_from_image(image, goal)
    prior_loss = F.cross_entropy(outputs["prior_logits"], action)
    policy_loss = F.cross_entropy(outputs["policy_logits"], action)
    value_loss = F.mse_loss(outputs["value_norm"], batch["value_norm"].to(device))
    with torch.no_grad():
        pred = torch.argmax(outputs["policy_logits"], dim=1)
        acc = float((pred == action).float().mean().item())
    return prior_loss + policy_loss + 0.25 * value_loss, {
        "bc_loss": float((prior_loss + policy_loss).item()),
        "bc_accuracy": acc,
    }


def supervised_risk_loss(
    agent: Dreamer4LiteAgent,
    batch: dict[str, Any],
    *,
    device: str,
    config: Dreamer4LiteConfig,
):
    image = batch["image"].to(device)
    goal = batch["goal"].to(device)
    candidate_action = batch["candidate_action"].to(device)
    best_action = batch["best_action"].to(device)
    latent = agent.encode_image(image)
    state = agent.forward_from_latent(latent, goal)
    outcomes = agent.predict_action_outcomes(latent, goal, candidate_action)
    bc_loss = F.cross_entropy(state["policy_logits"], best_action)
    prior_loss = F.cross_entropy(state["prior_logits"], best_action)
    reward_loss = F.mse_loss(outcomes["reward_norm"], batch["reward_norm"].to(device))
    value_loss = F.mse_loss(state["value_norm"], batch["value_norm"].to(device))
    collision_loss = F.binary_cross_entropy_with_logits(outcomes["collision_logit"], batch["collision"].to(device))
    success_loss = F.binary_cross_entropy_with_logits(outcomes["success_logit"], batch["success"].to(device))
    oob_loss = F.binary_cross_entropy_with_logits(outcomes["out_of_bounds_logit"], batch["out_of_bounds"].to(device))
    risk_loss = collision_loss + success_loss + oob_loss
    loss = (
        config.bc_loss_weight * (bc_loss + prior_loss)
        + config.reward_loss_weight * reward_loss
        + config.value_loss_weight * value_loss
        + config.risk_loss_weight * risk_loss
    )
    with torch.no_grad():
        pred = torch.argmax(state["policy_logits"], dim=1)
        acc = float((pred == best_action).float().mean().item())
    return loss, {
        "risk_bc_loss": float((bc_loss + prior_loss).item()),
        "risk_best_action_accuracy": acc,
        "reward_loss": float(reward_loss.item()),
        "value_loss": float(value_loss.item()),
        "risk_loss": float(risk_loss.item()),
    }


def run_imagination_training(
    agent: Dreamer4LiteAgent,
    *,
    seed_loader,
    device: str,
    config: Dreamer4LiteConfig,
    enabled_actions: list[int],
) -> list[dict[str, Any]]:
    if config.imagination_updates <= 0:
        return []
    freeze_module(agent.world_model)
    freeze_module(agent.prior_policy)
    freeze_module(agent.reward)
    freeze_module(agent.collision)
    freeze_module(agent.success)
    freeze_module(agent.out_of_bounds)
    optimizer = torch.optim.AdamW(
        [*agent.policy.parameters(), *agent.value.parameters()],
        lr=config.imagination_learning_rate,
    )
    history: list[dict[str, Any]] = []
    loader_iter = iter(seed_loader)
    enabled_tensor = torch.tensor(enabled_actions, dtype=torch.long, device=device)
    for update in range(1, config.imagination_updates + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(seed_loader)
            batch = next(loader_iter)
        image = batch["image"].to(device)
        goal = batch["goal"].to(device)
        with torch.no_grad():
            latent = agent.encode_image(image)
        metrics = imagination_update(
            agent,
            latent=latent,
            goal=goal,
            enabled_actions=enabled_tensor,
            optimizer=optimizer,
            config=config,
        )
        if update == 1 or update % max(1, config.imagination_updates // 10) == 0 or update == config.imagination_updates:
            item = {"update": update, **metrics}
            history.append(item)
            print(json.dumps({"phase": "imagination", **item}), flush=True)
    unfreeze_module(agent.prior_policy)
    unfreeze_module(agent.reward)
    unfreeze_module(agent.collision)
    unfreeze_module(agent.success)
    unfreeze_module(agent.out_of_bounds)
    return history


def imagination_update(
    agent: Dreamer4LiteAgent,
    *,
    latent: torch.Tensor,
    goal: torch.Tensor,
    enabled_actions: torch.Tensor,
    optimizer,
    config: Dreamer4LiteConfig,
) -> dict[str, float]:
    latents = []
    values = []
    log_probs = []
    rewards = []
    kls = []
    entropies = []
    risks = []
    current = latent.detach()
    for _ in range(config.imagination_horizon):
        state = agent.forward_from_latent(current, goal)
        logits = mask_logits(state["policy_logits"], enabled_actions)
        prior_logits = mask_logits(state["prior_logits"].detach(), enabled_actions)
        dist = Categorical(logits=logits)
        action = dist.sample()
        prior_probs = torch.softmax(prior_logits, dim=-1)
        policy_log_probs = torch.log_softmax(logits, dim=-1)
        kl = torch.sum(torch.softmax(logits, dim=-1) * (policy_log_probs - torch.log(prior_probs + 1e-8)), dim=-1)
        outcomes = agent.predict_action_outcomes(current, goal, action)
        collision = torch.sigmoid(outcomes["collision_logit"])
        oob = torch.sigmoid(outcomes["out_of_bounds_logit"])
        success = torch.sigmoid(outcomes["success_logit"])
        reward = (
            outcomes["reward_norm"] * config.reward_scale
            + 2.0 * success
            - config.collision_penalty * collision
            - config.out_of_bounds_penalty * oob
        )
        latents.append(current)
        values.append(state["value_norm"] * config.reward_scale)
        log_probs.append(dist.log_prob(action))
        rewards.append(reward)
        kls.append(kl)
        entropies.append(dist.entropy())
        risks.append(collision)
        with torch.no_grad():
            current = agent.imagine_next_latent(current, action)

    with torch.no_grad():
        bootstrap = agent.forward_from_latent(current, goal)["value_norm"] * config.reward_scale
        returns = []
        running = bootstrap
        for reward in reversed(rewards):
            running = reward + config.gamma * running
            returns.append(running)
        returns = list(reversed(returns))

    values_t = torch.stack(values, dim=0)
    returns_t = torch.stack(returns, dim=0)
    log_probs_t = torch.stack(log_probs, dim=0)
    advantages = returns_t.detach() - values_t.detach()
    positive = advantages >= 0.0
    negative = ~positive
    policy_terms = []
    if positive.any():
        policy_terms.append(-0.5 * log_probs_t[positive].mean())
    if negative.any():
        policy_terms.append(0.5 * log_probs_t[negative].mean())
    policy_loss = sum(policy_terms) if policy_terms else torch.zeros((), device=latent.device)
    value_loss = F.mse_loss(values_t, returns_t.detach())
    kl_loss = torch.stack(kls, dim=0).mean()
    entropy = torch.stack(entropies, dim=0).mean()
    loss = policy_loss + 0.5 * value_loss + config.kl_to_prior_weight * kl_loss - config.entropy_weight * entropy
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([*agent.policy.parameters(), *agent.value.parameters()], max_norm=5.0)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "kl_to_prior": float(kl_loss.item()),
        "entropy": float(entropy.item()),
        "imagined_reward": float(torch.stack(rewards, dim=0).mean().item()),
        "imagined_collision_risk": float(torch.stack(risks, dim=0).mean().item()),
        "positive_advantage_fraction": float(positive.float().mean().item()),
    }


def evaluate_agent(
    agent: Dreamer4LiteAgent,
    *,
    device: str,
    image_width: int,
    image_height: int,
    episodes: int,
    seed: int,
    enabled_actions: list[int],
    trace_path: Path,
) -> dict[str, Any]:
    env = DroneMazeEnv(DroneGameConfig(image_width=image_width, image_height=image_height))
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    collisions: list[float] = []
    timeouts: list[float] = []
    action_counts: Counter[str] = Counter()
    traces: list[tuple[str, list[np.ndarray]]] = []
    agent.eval()
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        frames: list[np.ndarray] = []
        steps = 0
        while not done:
            if episode < 8 and len(frames) < 10:
                frames.append(obs["image"])
            image = image_to_batch(obs["image"], width=image_width, height=image_height, device=device)
            goal = torch.tensor([goal_features_from_info(info)], dtype=torch.float32, device=device)
            with torch.no_grad():
                latent = agent.encode_image(image)
                logits = mask_logits(agent.forward_from_latent(latent, goal)["policy_logits"], torch.tensor(enabled_actions, dtype=torch.long, device=device))
                action = int(torch.argmax(logits, dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            action_counts[ACTION_VOCAB[action]] += 1
            done = bool(terminated or truncated)
            steps += 1
        returns.append(total)
        lengths.append(steps)
        successes.append(1.0 if info.get("success") else 0.0)
        collisions.append(1.0 if info.get("collision") else 0.0)
        timeouts.append(1.0 if info.get("truncated") else 0.0)
        if frames:
            traces.append((f"seed {seed + episode}", frames))
    if traces:
        make_trace_sheet(traces, trace_path)
    return {
        "episodes": int(episodes),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "collision_rate": float(np.mean(collisions)) if collisions else 0.0,
        "timeout_rate": float(np.mean(timeouts)) if timeouts else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "action_counts": dict(action_counts),
        "trace_contact_sheet": str(trace_path.resolve()) if traces else None,
    }


def checkpoint_payload(
    agent: Dreamer4LiteAgent,
    config: Dreamer4LiteConfig,
    world_model_config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "agent_state_dict": agent.state_dict(),
        "config": asdict(config),
        "world_model_config": world_model_config,
        "epoch": epoch,
        "metrics": metrics,
        "action_vocab": ACTION_VOCAB,
        "model_type": "dreamer4_lite_world_model_bc_reward_imagination",
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_record_image(path: str | Path, width: int, height: int):
    return load_image(resolve_record_path(path), width, height)


def resolve_record_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    project_relative = PROJECT_ROOT / candidate
    if project_relative.exists():
        return project_relative
    repo_relative = PROJECT_ROOT.parent / candidate
    if repo_relative.exists():
        return repo_relative
    return candidate


def build_value_targets(records: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    values: dict[tuple[str, int], float] = {}
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in records:
        labels = dict(record.get("labels", {}))
        key = (str(record["episode_id"]), int(record["step_index"]))
        grouped[key].append(float(labels.get("utility", labels.get("reward", 0.0))))
    for key, items in grouped.items():
        values[key] = max(items)
    return values


def goal_features_from_record(record: dict[str, Any]) -> list[float]:
    teacher = dict(record.get("metadata", {}).get("teacher", {}))
    goal = dict(teacher.get("goal_features", {}))
    return [
        float(np.clip(float(goal.get("forward_m", 0.0)) / 10.0, -2.0, 2.0)),
        float(np.clip(float(goal.get("right_m", 0.0)) / 5.0, -2.0, 2.0)),
        float(np.clip(float(goal.get("alt_error_m", 0.0)) / 3.0, -2.0, 2.0)),
        float(np.clip(float(goal.get("heading_error_deg", 0.0)) / 180.0, -1.0, 1.0)),
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


def mask_logits(logits: torch.Tensor, enabled_actions: torch.Tensor) -> torch.Tensor:
    masked = torch.full_like(logits, fill_value=-1e9)
    masked.index_copy_(1, enabled_actions, logits.index_select(1, enabled_actions))
    return masked


def parse_enabled_actions(raw: str) -> list[int]:
    indices: list[int] = []
    for name in [item.strip() for item in raw.split(",") if item.strip()]:
        if name not in ACTION_VOCAB:
            raise ValueError(f"Unsupported action: {name!r}")
        indices.append(ACTION_VOCAB.index(name))
    if not indices:
        raise ValueError("No enabled actions.")
    return list(dict.fromkeys(indices))


def image_to_batch(image: np.ndarray, *, width: int, height: int, device: str):
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").resize((width, height), Image.BILINEAR)
    array = np.asarray(pil, dtype=np.float32) / 255.0
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
    supervised_eval = summary["supervised_eval"]
    final_eval = summary["final_eval"]
    imagination_rows = []
    for item in summary["imagination_history"]:
        imagination_rows.append(
            "<tr>"
            f"<td>{item['update']}</td>"
            f"<td>{item['loss']:.4f}</td>"
            f"<td>{item['imagined_reward']:.4f}</td>"
            f"<td>{item['imagined_collision_risk']:.4f}</td>"
            f"<td>{item['kl_to_prior']:.4f}</td>"
            f"<td>{item['entropy']:.4f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Dreamer4-Lite Drone Game</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f4f1e8; color: #202020; }}
    table {{ border-collapse: collapse; background: white; margin: 16px 0; }}
    th, td {{ border: 1px solid #c9c1ad; padding: 6px 8px; text-align: right; }}
    th {{ background: #292f25; color: white; }}
    td:first-child, th:first-child {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #9f967f; background: white; }}
    code {{ background: #ebe4d4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Dreamer4-Lite Drone Game</h1>
  <p>Frozen Sensenova world-model encoder/dynamics + BC/reward/value heads + KL-constrained imagination update.</p>
  <h2>Closed-Loop Evaluation</h2>
  <table>
    <tr><th>Metric</th><th>Supervised</th><th>After Imagination</th></tr>
    <tr><td>Success Rate</td><td>{supervised_eval['success_rate']:.4f}</td><td>{final_eval['success_rate']:.4f}</td></tr>
    <tr><td>Collision Rate</td><td>{supervised_eval['collision_rate']:.4f}</td><td>{final_eval['collision_rate']:.4f}</td></tr>
    <tr><td>Timeout Rate</td><td>{supervised_eval['timeout_rate']:.4f}</td><td>{final_eval['timeout_rate']:.4f}</td></tr>
    <tr><td>Mean Return</td><td>{supervised_eval['mean_return']:.4f}</td><td>{final_eval['mean_return']:.4f}</td></tr>
    <tr><td>Mean Length</td><td>{supervised_eval['mean_length']:.2f}</td><td>{final_eval['mean_length']:.2f}</td></tr>
    <tr><td>Deployment Score</td><td>{summary['supervised_deployment_score']:.4f}</td><td>{summary['final_deployment_score']:.4f}</td></tr>
  </table>
  <p><strong>Recommended checkpoint:</strong> {summary['recommended_phase']} (<code>{Path(summary['selected_checkpoint']).name}</code>)</p>
  <h2>Imagination Training</h2>
  <table>
    <tr><th>Update</th><th>Loss</th><th>Imagined Reward</th><th>Collision Risk</th><th>KL Prior</th><th>Entropy</th></tr>
    {''.join(imagination_rows)}
  </table>
  <h2>Supervised Trace</h2>
  <img src="supervised_trace_contact_sheet.png" />
  <h2>Final Trace</h2>
  <img src="final_trace_contact_sheet.png" />
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def unfreeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(True)


def safe_torch_load(path: str | Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return requested


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
