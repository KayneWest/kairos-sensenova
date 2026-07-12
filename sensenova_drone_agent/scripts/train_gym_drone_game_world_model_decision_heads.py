#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
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

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    F = None
    DataLoader = None
    Dataset = None

from sensenova_drone.bc_data import ACTION_VOCAB
from sensenova_drone.world_model_decision import WorldModelDecisionHeads
from scripts.train_gym_drone_game_world_model import ActionConditionedWorldModel, load_manifest


@dataclass
class WMDecisionTrainConfig:
    world_model_checkpoint: str
    bc_manifest: str
    risk_manifest: str
    out_dir: str
    epochs: int = 6
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "auto"
    seed: int = 0
    num_workers: int = 0
    max_depth_m: float = 10.0
    utility_scale: float = 12.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.15
    binary_loss_weight: float = 1.0
    clearance_loss_weight: float = 0.5
    progress_loss_weight: float = 0.35
    utility_loss_weight: float = 0.5
    freeze_world_model: bool = True
    planner_policy_logprob_weight: float = 0.75
    planner_collision_penalty: float = 8.0
    planner_out_of_bounds_penalty: float = 5.0
    planner_success_bonus: float = 5.0
    planner_progress_weight: float = 3.0
    planner_clearance_weight: float = 0.4
    planner_hover_penalty: float = 0.5
    planner_yaw_penalty: float = 0.05


class BCDecisionDataset(Dataset if Dataset is not None else object):
    def __init__(self, records: list[dict[str, Any]], *, image_width: int, image_height: int):
        if torch is None:
            raise RuntimeError("torch is required.")
        self.records = records
        self.image_width = int(image_width)
        self.image_height = int(image_height)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        metadata = dict(record.get("metadata", {}))
        reward = float(metadata.get("reward", 0.0))
        return {
            "image": load_image(record["image_path"], self.image_width, self.image_height),
            "goal_features": torch.tensor(extract_bc_goal_features(record), dtype=torch.float32),
            "action_index": torch.tensor(int(record["action_index"]), dtype=torch.long),
            "value_target": torch.tensor(float(np.clip(reward / 12.0, -1.0, 1.0)), dtype=torch.float32),
        }


class ActionRiskDecisionDataset(Dataset if Dataset is not None else object):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        image_width: int,
        image_height: int,
        max_depth_m: float,
        utility_scale: float,
    ):
        if torch is None:
            raise RuntimeError("torch is required.")
        self.records = records
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.max_depth_m = float(max_depth_m)
        self.utility_scale = float(utility_scale)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        labels = dict(record["labels"])
        return {
            "image": load_image(record["image_path"], self.image_width, self.image_height),
            "goal_features": torch.tensor(record.get("goal_features") or [0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "action_index": torch.tensor(int(record["candidate_action_index"]), dtype=torch.long),
            "collision": torch.tensor(1.0 if labels.get("collision", False) else 0.0, dtype=torch.float32),
            "success": torch.tensor(1.0 if labels.get("success", False) else 0.0, dtype=torch.float32),
            "out_of_bounds": torch.tensor(1.0 if labels.get("out_of_bounds", False) else 0.0, dtype=torch.float32),
            "front_clearance_norm": torch.tensor(
                float(np.clip(float(labels.get("front_clearance_after_m", self.max_depth_m)) / max(self.max_depth_m, 1e-6), 0.0, 1.0)),
                dtype=torch.float32,
            ),
            "progress_norm": torch.tensor(float(np.clip(float(labels.get("progress_m", 0.0)), -1.0, 1.0)), dtype=torch.float32),
            "utility_norm": torch.tensor(float(np.clip(float(labels.get("utility", 0.0)) / max(self.utility_scale, 1e-6), -1.0, 1.0)), dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train decision heads on top of a pretrained Gym pixel world model.")
    parser.add_argument("--world-model-checkpoint", default="output/gym_drone_game_world_model_v1/best.pt")
    parser.add_argument("--bc-manifest", default="data/gym_drone_game_dqn_teacher_v2_all/manifests/bc_manifest.jsonl")
    parser.add_argument("--risk-manifest", default="data/gym_drone_game_action_risk_v2/manifests/action_risk_manifest.jsonl")
    parser.add_argument("--out-dir", default="output/gym_drone_game_world_model_decision_heads_v1")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--utility-scale", type=float, default=12.0)
    parser.add_argument("--finetune-world-model", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = WMDecisionTrainConfig(
        world_model_checkpoint=args.world_model_checkpoint,
        bc_manifest=args.bc_manifest,
        risk_manifest=args.risk_manifest,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        max_depth_m=args.max_depth_m,
        utility_scale=args.utility_scale,
        freeze_world_model=not args.finetune_world_model,
    )
    summary = train_decision_heads(config)
    print(json.dumps(summary, indent=2))
    return 0


def train_decision_heads(config: WMDecisionTrainConfig) -> dict[str, Any]:
    if torch is None or F is None or DataLoader is None:
        raise RuntimeError("torch is required.")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wm_payload = torch.load(config.world_model_checkpoint, map_location=device)
    wm_config = dict(wm_payload["config"])
    world_model = ActionConditionedWorldModel(
        num_actions=len(ACTION_VOCAB),
        image_width=int(wm_config["image_width"]),
        image_height=int(wm_config["image_height"]),
        latent_dim=int(wm_config["latent_dim"]),
    ).to(device)
    world_model.load_state_dict(wm_payload["model_state_dict"])
    model = WorldModelDecisionHeads(
        world_model,
        latent_dim=int(wm_config["latent_dim"]),
        num_actions=len(ACTION_VOCAB),
        freeze_world_model=config.freeze_world_model,
    ).to(device)

    bc_records = load_manifest(config.bc_manifest)
    risk_records = load_manifest(config.risk_manifest)
    bc_train = [record for record in bc_records if record.get("split", "train") == "train"]
    bc_val = [record for record in bc_records if record.get("split") == "val"]
    risk_train = [record for record in risk_records if record.get("split", "train") == "train"]
    risk_val = [record for record in risk_records if record.get("split") == "val"]
    image_width = int(wm_config["image_width"])
    image_height = int(wm_config["image_height"])
    bc_train_loader = DataLoader(
        BCDecisionDataset(bc_train, image_width=image_width, image_height=image_height),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    bc_val_loader = DataLoader(
        BCDecisionDataset(bc_val, image_width=image_width, image_height=image_height),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    risk_train_loader = DataLoader(
        ActionRiskDecisionDataset(
            risk_train,
            image_width=image_width,
            image_height=image_height,
            max_depth_m=config.max_depth_m,
            utility_scale=config.utility_scale,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    risk_val_loader = DataLoader(
        ActionRiskDecisionDataset(
            risk_val,
            image_width=image_width,
            image_height=image_height,
            max_depth_m=config.max_depth_m,
            utility_scale=config.utility_scale,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    pos_weights = compute_pos_weights(risk_train)
    pos_weights = {
        key: (value.to(device) if value is not None else None)
        for key, value in pos_weights.items()
    }
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_checkpoint = out_dir / "best.pt"
    last_checkpoint = out_dir / "last.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            bc_train_loader,
            risk_train_loader,
            device=device,
            optimizer=optimizer,
            config=config,
            pos_weights=pos_weights,
        )
        val_metrics = run_epoch(
            model,
            bc_val_loader,
            risk_val_loader,
            device=device,
            optimizer=None,
            config=config,
            pos_weights=pos_weights,
        )
        metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(metrics)
        selection_metric = val_metrics["loss"]
        payload = {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "world_model_config": wm_config,
            "world_model_checkpoint": str(Path(config.world_model_checkpoint).resolve()),
            "epoch": epoch,
            "metrics": metrics,
            "action_vocab": ACTION_VOCAB,
            "model_type": "world_model_decision_heads",
        }
        torch.save(payload, last_checkpoint)
        if selection_metric < best_metric:
            best_metric = float(selection_metric)
            torch.save(payload, best_checkpoint)
        print(json.dumps(metrics), flush=True)

    summary = {
        "world_model_checkpoint": str(Path(config.world_model_checkpoint).resolve()),
        "bc_manifest": str(Path(config.bc_manifest).resolve()),
        "risk_manifest": str(Path(config.risk_manifest).resolve()),
        "out_dir": str(out_dir.resolve()),
        "device": device,
        "num_bc_train": len(bc_train),
        "num_bc_val": len(bc_val),
        "num_risk_train": len(risk_train),
        "num_risk_val": len(risk_val),
        "best_metric": best_metric,
        "history": history,
        "best_checkpoint": str(best_checkpoint.resolve()),
        "last_checkpoint": str(last_checkpoint.resolve()),
        "config": asdict(config),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir, summary)
    return summary


def run_epoch(model, bc_loader, risk_loader, *, device: str, optimizer, config: WMDecisionTrainConfig, pos_weights) -> dict[str, Any]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    bc_iter = iter(bc_loader)
    risk_iter = iter(risk_loader)
    steps = max(len(bc_loader), len(risk_loader))
    totals: dict[str, float] = {
        "examples_bc": 0.0,
        "examples_risk": 0.0,
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "risk_loss": 0.0,
        "policy_correct": 0.0,
        "collision_correct": 0.0,
        "success_correct": 0.0,
        "clearance_mae_m": 0.0,
        "utility_mae": 0.0,
    }
    for _ in range(steps):
        try:
            bc_batch = next(bc_iter)
        except StopIteration:
            bc_iter = iter(bc_loader)
            bc_batch = next(bc_iter)
        try:
            risk_batch = next(risk_iter)
        except StopIteration:
            risk_iter = iter(risk_loader)
            risk_batch = next(risk_iter)

        with torch.set_grad_enabled(train_mode):
            policy_loss, value_loss, bc_stats = compute_policy_loss(model, bc_batch, device=device)
            risk_loss, risk_stats = compute_risk_loss(model, risk_batch, device=device, config=config, pos_weights=pos_weights)
            loss = (
                config.policy_loss_weight * policy_loss
                + config.value_loss_weight * value_loss
                + risk_loss
            )
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        bc_n = bc_stats["num_examples"]
        risk_n = risk_stats["num_examples"]
        totals["examples_bc"] += bc_n
        totals["examples_risk"] += risk_n
        totals["loss"] += float(loss.item())
        totals["policy_loss"] += float(policy_loss.item())
        totals["value_loss"] += float(value_loss.item())
        totals["risk_loss"] += float(risk_loss.item())
        totals["policy_correct"] += bc_stats["correct"]
        totals["collision_correct"] += risk_stats["collision_correct"]
        totals["success_correct"] += risk_stats["success_correct"]
        totals["clearance_mae_m"] += risk_stats["clearance_mae_m"] * risk_n
        totals["utility_mae"] += risk_stats["utility_mae"] * risk_n

    denom_steps = max(steps, 1)
    bc_n = max(totals["examples_bc"], 1.0)
    risk_n = max(totals["examples_risk"], 1.0)
    return {
        "loss": totals["loss"] / denom_steps,
        "policy_loss": totals["policy_loss"] / denom_steps,
        "value_loss": totals["value_loss"] / denom_steps,
        "risk_loss": totals["risk_loss"] / denom_steps,
        "policy_accuracy": totals["policy_correct"] / bc_n,
        "collision_accuracy": totals["collision_correct"] / risk_n,
        "success_accuracy": totals["success_correct"] / risk_n,
        "clearance_mae_m": totals["clearance_mae_m"] / risk_n,
        "utility_mae": totals["utility_mae"] / risk_n,
        "num_bc_examples": int(totals["examples_bc"]),
        "num_risk_examples": int(totals["examples_risk"]),
    }


def compute_policy_loss(model, batch, *, device: str):
    image = batch["image"].to(device)
    goal = batch["goal_features"].to(device)
    action = batch["action_index"].to(device)
    value_target = batch["value_target"].to(device)
    outputs = model.policy(image, goal)
    policy_loss = F.cross_entropy(outputs["action_logits"], action)
    value_loss = F.mse_loss(outputs["value"], value_target)
    correct = int((torch.argmax(outputs["action_logits"], dim=1) == action).sum().item())
    return policy_loss, value_loss, {"num_examples": int(image.shape[0]), "correct": correct}


def compute_risk_loss(model, batch, *, device: str, config: WMDecisionTrainConfig, pos_weights):
    image = batch["image"].to(device)
    goal = batch["goal_features"].to(device)
    action = batch["action_index"].to(device)
    collision = batch["collision"].to(device)
    success = batch["success"].to(device)
    out_of_bounds = batch["out_of_bounds"].to(device)
    clearance = batch["front_clearance_norm"].to(device)
    progress = batch["progress_norm"].to(device)
    utility = batch["utility_norm"].to(device)
    outputs = model.risk(image, goal, action)
    collision_loss = F.binary_cross_entropy_with_logits(
        outputs["collision_logit"],
        collision,
        pos_weight=pos_weights.get("collision"),
    )
    success_loss = F.binary_cross_entropy_with_logits(
        outputs["success_logit"],
        success,
        pos_weight=pos_weights.get("success"),
    )
    oob_loss = F.binary_cross_entropy_with_logits(
        outputs["out_of_bounds_logit"],
        out_of_bounds,
        pos_weight=pos_weights.get("out_of_bounds"),
    )
    clearance_loss = F.mse_loss(torch.sigmoid(outputs["front_clearance_norm"]), clearance)
    progress_loss = F.mse_loss(outputs["progress_norm"], progress)
    utility_loss = F.mse_loss(outputs["utility_norm"], utility)
    loss = (
        config.binary_loss_weight * (collision_loss + success_loss + oob_loss)
        + config.clearance_loss_weight * clearance_loss
        + config.progress_loss_weight * progress_loss
        + config.utility_loss_weight * utility_loss
    )
    n = int(image.shape[0])
    stats = {
        "num_examples": n,
        "collision_correct": int(((torch.sigmoid(outputs["collision_logit"]) >= 0.5) == (collision >= 0.5)).sum().item()),
        "success_correct": int(((torch.sigmoid(outputs["success_logit"]) >= 0.5) == (success >= 0.5)).sum().item()),
        "clearance_mae_m": float(torch.abs(torch.sigmoid(outputs["front_clearance_norm"]) - clearance).mean().item()) * config.max_depth_m,
        "utility_mae": float(torch.abs(outputs["utility_norm"] - utility).mean().item()) * config.utility_scale,
    }
    return loss, stats


def compute_pos_weights(records: list[dict[str, Any]]) -> dict[str, Any]:
    if torch is None:
        return {}
    weights = {}
    for key in ["collision", "success", "out_of_bounds"]:
        positives = sum(1 for record in records if bool(record["labels"].get(key, False)))
        negatives = max(0, len(records) - positives)
        weights[key] = (
            torch.tensor(float(negatives) / float(positives), dtype=torch.float32)
            if positives > 0
            else None
        )
    return weights


def load_image(path: str | Path, width: int, height: int):
    image = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def extract_bc_goal_features(record: dict[str, Any]) -> list[float]:
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


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    rows = []
    for item in summary["history"]:
        val = item["val"]
        rows.append(
            "<tr>"
            f"<td>{item['epoch']}</td>"
            f"<td>{item['train']['loss']:.4f}</td>"
            f"<td>{val['loss']:.4f}</td>"
            f"<td>{val['policy_accuracy']:.4f}</td>"
            f"<td>{val['collision_accuracy']:.4f}</td>"
            f"<td>{val['success_accuracy']:.4f}</td>"
            f"<td>{val['clearance_mae_m']:.3f}</td>"
            f"<td>{val['utility_mae']:.3f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>World Model Decision Heads</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f4f1e8; color: #202020; }}
    table {{ border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #c9c1ad; padding: 6px 8px; text-align: right; }}
    th {{ background: #292f25; color: white; }}
    td:first-child, th:first-child {{ text-align: left; }}
    code {{ background: #ebe4d4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>World Model Decision Heads</h1>
  <p>Best metric: <code>{summary['best_metric']:.6f}</code></p>
  <table>
    <thead><tr><th>Epoch</th><th>Train Loss</th><th>Val Loss</th><th>Policy Acc</th><th>Collision Acc</th><th>Success Acc</th><th>Clearance MAE m</th><th>Utility MAE</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
