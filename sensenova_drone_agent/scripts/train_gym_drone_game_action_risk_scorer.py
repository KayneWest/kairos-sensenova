#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
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

from sensenova_drone.action_risk_model import ActionRiskVisualScorer
from sensenova_drone.bc_data import ACTION_VOCAB


@dataclass
class ActionRiskTrainConfig:
    manifest_path: str
    out_dir: str
    epochs: int = 6
    batch_size: int = 128
    learning_rate: float = 1e-3
    image_size: int = 96
    frame_stack: int = 1
    device: str = "auto"
    num_workers: int = 0
    seed: int = 0
    max_depth_m: float = 10.0
    reward_scale: float = 12.0
    binary_loss_weight: float = 1.0
    clearance_loss_weight: float = 0.75
    progress_loss_weight: float = 0.5
    reward_loss_weight: float = 0.5
    use_pos_weight: bool = True
    planner_collision_penalty: float = 8.0
    planner_out_of_bounds_penalty: float = 5.0
    planner_success_bonus: float = 5.0
    planner_progress_weight: float = 2.0
    planner_clearance_weight: float = 0.5


class ActionRiskDataset(Dataset if Dataset is not None else object):
    def __init__(self, records: list[dict[str, Any]], *, image_size: int, max_depth_m: float, reward_scale: float):
        if torch is None:
            raise RuntimeError("torch is required to instantiate ActionRiskDataset.")
        self.records = records
        self.image_size = int(image_size)
        self.max_depth_m = float(max_depth_m)
        self.reward_scale = float(reward_scale)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        labels = dict(record["labels"])
        return {
            "image": load_image(record["image_path"], self.image_size),
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
            "reward_norm": torch.tensor(float(np.clip(float(labels.get("utility", labels.get("reward", 0.0))) / max(self.reward_scale, 1e-6), -1.0, 1.0)), dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train action-conditioned visual risk scorer.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--reward-scale", type=float, default=12.0)
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--binary-loss-weight", type=float, default=1.0)
    parser.add_argument("--clearance-loss-weight", type=float, default=0.75)
    parser.add_argument("--progress-loss-weight", type=float, default=0.5)
    parser.add_argument("--reward-loss-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ActionRiskTrainConfig(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        max_depth_m=args.max_depth_m,
        reward_scale=args.reward_scale,
        binary_loss_weight=args.binary_loss_weight,
        clearance_loss_weight=args.clearance_loss_weight,
        progress_loss_weight=args.progress_loss_weight,
        reward_loss_weight=args.reward_loss_weight,
        use_pos_weight=not args.no_pos_weight,
    )
    summary = train_action_risk_scorer(config)
    print(json.dumps(summary, indent=2))
    return 0


def train_action_risk_scorer(config: ActionRiskTrainConfig) -> dict[str, Any]:
    if torch is None or F is None or DataLoader is None:
        raise RuntimeError("torch is required for action-risk scorer training.")
    seed_everything(config.seed)
    records = load_records(config.manifest_path)
    train_records = [record for record in records if record.get("split", "train") == "train"]
    val_records = [record for record in records if record.get("split", "train") == "val"]
    if not train_records:
        raise RuntimeError("No training records in action-risk manifest.")
    device = resolve_device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = ActionRiskDataset(
        train_records,
        image_size=config.image_size,
        max_depth_m=config.max_depth_m,
        reward_scale=config.reward_scale,
    )
    val_dataset = ActionRiskDataset(
        val_records,
        image_size=config.image_size,
        max_depth_m=config.max_depth_m,
        reward_scale=config.reward_scale,
    ) if val_records else None
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers) if val_dataset else None
    model = ActionRiskVisualScorer(num_actions=len(ACTION_VOCAB), goal_feature_dim=4, frame_stack=config.frame_stack).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    pos_weights = compute_pos_weights(train_records) if config.use_pos_weight else {}
    pos_weights = {
        key: (value.to(device) if value is not None else None)
        for key, value in pos_weights.items()
    }
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_checkpoint = out_dir / "best.pt"
    last_checkpoint = out_dir / "last.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer, config=config, pos_weights=pos_weights)
        val_metrics = run_epoch(model, val_loader, device=device, optimizer=None, config=config, pos_weights=pos_weights) if val_loader else None
        metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(metrics)
        selection_metric = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "action_vocab": ACTION_VOCAB,
            "metrics": metrics,
        }
        torch.save(payload, last_checkpoint)
        if selection_metric < best_metric:
            best_metric = selection_metric
            torch.save(payload, best_checkpoint)

    summary = {
        "manifest_path": str(Path(config.manifest_path).resolve()),
        "out_dir": str(out_dir.resolve()),
        "device": device,
        "num_train": len(train_records),
        "num_val": len(val_records),
        "best_metric": best_metric,
        "history": history,
        "best_checkpoint": str(best_checkpoint.resolve()),
        "last_checkpoint": str(last_checkpoint.resolve()),
        "config": asdict(config),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir, summary)
    return summary


def run_epoch(model, loader, *, device: str, optimizer, config: ActionRiskTrainConfig, pos_weights) -> dict[str, Any]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    totals = Counter()
    for batch in loader:
        images = batch["image"].to(device)
        goal = batch["goal_features"].to(device)
        actions = batch["action_index"].to(device)
        collision = batch["collision"].to(device)
        success = batch["success"].to(device)
        out_of_bounds = batch["out_of_bounds"].to(device)
        clearance = batch["front_clearance_norm"].to(device)
        progress = batch["progress_norm"].to(device)
        reward = batch["reward_norm"].to(device)
        with torch.set_grad_enabled(train_mode):
            outputs = model(images, goal, actions)
            collision_loss = bce(outputs["collision_logit"], collision, pos_weights.get("collision"))
            success_loss = bce(outputs["success_logit"], success, pos_weights.get("success"))
            oob_loss = bce(outputs["out_of_bounds_logit"], out_of_bounds, pos_weights.get("out_of_bounds"))
            clearance_loss = F.mse_loss(torch.sigmoid(outputs["front_clearance_norm"]), clearance)
            progress_loss = F.mse_loss(outputs["progress_norm"], progress)
            reward_loss = F.mse_loss(outputs["reward_norm"], reward)
            loss = (
                config.binary_loss_weight * (collision_loss + success_loss + oob_loss)
                + config.clearance_loss_weight * clearance_loss
                + config.progress_loss_weight * progress_loss
                + config.reward_loss_weight * reward_loss
            )
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        n = int(images.shape[0])
        totals["examples"] += n
        for name, value in [
            ("loss", loss),
            ("collision_loss", collision_loss),
            ("success_loss", success_loss),
            ("out_of_bounds_loss", oob_loss),
            ("clearance_loss", clearance_loss),
            ("progress_loss", progress_loss),
            ("reward_loss", reward_loss),
        ]:
            totals[name] += float(value.item()) * n
        totals["collision_correct"] += int(((torch.sigmoid(outputs["collision_logit"]) >= 0.5) == (collision >= 0.5)).sum().item())
        totals["success_correct"] += int(((torch.sigmoid(outputs["success_logit"]) >= 0.5) == (success >= 0.5)).sum().item())
        totals["out_of_bounds_correct"] += int(((torch.sigmoid(outputs["out_of_bounds_logit"]) >= 0.5) == (out_of_bounds >= 0.5)).sum().item())
        totals["clearance_mae_m"] += float(torch.abs(torch.sigmoid(outputs["front_clearance_norm"]) - clearance).mean().item()) * n * config.max_depth_m
        totals["progress_mae_m"] += float(torch.abs(outputs["progress_norm"] - progress).mean().item()) * n
        totals["reward_mae"] += float(torch.abs(outputs["reward_norm"] - reward).mean().item()) * n * config.reward_scale
    n = max(int(totals["examples"]), 1)
    return {
        "loss": totals["loss"] / n,
        "collision_loss": totals["collision_loss"] / n,
        "success_loss": totals["success_loss"] / n,
        "out_of_bounds_loss": totals["out_of_bounds_loss"] / n,
        "clearance_loss": totals["clearance_loss"] / n,
        "progress_loss": totals["progress_loss"] / n,
        "reward_loss": totals["reward_loss"] / n,
        "collision_accuracy": totals["collision_correct"] / n,
        "success_accuracy": totals["success_correct"] / n,
        "out_of_bounds_accuracy": totals["out_of_bounds_correct"] / n,
        "clearance_mae_m": totals["clearance_mae_m"] / n,
        "progress_mae_m": totals["progress_mae_m"] / n,
        "reward_mae": totals["reward_mae"] / n,
        "num_examples": n,
    }


def bce(logits, target, pos_weight):
    if pos_weight is None:
        return F.binary_cross_entropy_with_logits(logits, target)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_image(path: str | Path, image_size: int):
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def compute_pos_weights(records: list[dict[str, Any]]) -> dict[str, Any]:
    weights = {}
    for key in ["collision", "success", "out_of_bounds"]:
        positives = sum(1 for record in records if bool(record["labels"].get(key, False)))
        negatives = max(0, len(records) - positives)
        if positives > 0:
            weights[key] = torch.tensor(float(negatives) / float(positives), dtype=torch.float32)
        else:
            weights[key] = None
    return weights


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
        val = item.get("val") or {}
        rows.append(
            "<tr>"
            f"<td>{item['epoch']}</td>"
            f"<td>{item['train']['loss']:.4f}</td>"
            f"<td>{val.get('loss', 0.0):.4f}</td>"
            f"<td>{val.get('collision_accuracy', 0.0):.4f}</td>"
            f"<td>{val.get('success_accuracy', 0.0):.4f}</td>"
            f"<td>{val.get('clearance_mae_m', 0.0):.3f}</td>"
            f"<td>{val.get('reward_mae', 0.0):.3f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Action-Conditioned Risk Scorer Training</title>
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
  <h1>Action-Conditioned Risk Scorer Training</h1>
  <p>Best metric: <code>{summary['best_metric']:.6f}</code></p>
  <table>
    <thead><tr><th>Epoch</th><th>Train Loss</th><th>Val Loss</th><th>Collision Acc</th><th>Success Acc</th><th>Clearance MAE m</th><th>Reward MAE</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
