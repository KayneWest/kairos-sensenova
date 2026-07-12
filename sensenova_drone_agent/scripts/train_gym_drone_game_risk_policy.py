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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ModuleNotFoundError:
    torch = None
    F = None
    DataLoader = None
    Dataset = None
    WeightedRandomSampler = None

from sensenova_drone.bc_data import ACTION_VOCAB
from sensenova_drone.bc_train import (
    _build_dataset_entries,
    _build_episode_step_index,
    _extract_goal_feature_vector,
    _load_stacked_image_tensor,
    _mirrored_action_index,
    load_manifest_records,
    split_manifest_records,
)
from sensenova_drone.risk_aware_policy import RiskAwareVisualPolicy, extract_risk_labels


@dataclass
class RiskPolicyTrainConfig:
    manifest_path: str
    out_dir: str
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 1e-3
    image_size: int = 96
    frame_stack: int = 1
    device: str = "auto"
    num_workers: int = 0
    seed: int = 0
    max_depth_m: float = 10.0
    action_loss_weight: float = 1.0
    command_loss_weight: float = 0.15
    collision_loss_weight: float = 1.0
    stall_loss_weight: float = 0.35
    clearance_loss_weight: float = 0.5
    progress_loss_weight: float = 0.35
    use_class_weights: bool = True
    balanced_action_sampler: bool = False
    mirror_lateral_actions: bool = False
    shield_collision_threshold: float = 0.55
    shield_front_clearance_m: float = 1.1


class RiskManifestDataset(Dataset if Dataset is not None else object):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        image_size: int,
        frame_stack: int,
        max_depth_m: float,
        mirror_lateral_actions: bool,
    ):
        if torch is None:
            raise RuntimeError("torch is required to instantiate RiskManifestDataset.")
        self.records = records
        self.image_size = int(image_size)
        self.frame_stack = max(1, int(frame_stack))
        self.max_depth_m = float(max_depth_m)
        self._episode_steps = _build_episode_step_index(records)
        self._entries = _build_dataset_entries(records, mirror_lateral_actions=mirror_lateral_actions)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record, mirrored = self._entries[index]
        image = _load_stacked_image_tensor(
            self._stack_frame_paths(record),
            self.image_size,
            mirror=mirrored,
        )
        action_index = _mirrored_action_index(int(record["action_index"])) if mirrored else int(record["action_index"])
        goal_features = torch.tensor(_extract_goal_feature_vector(record, mirrored=mirrored), dtype=torch.float32)
        command = _command_tensor(record, mirrored=mirrored)
        labels = extract_risk_labels(record, max_depth_m=self.max_depth_m)
        return {
            "image": image,
            "goal_features": goal_features,
            "action_index": torch.tensor(action_index, dtype=torch.long),
            "command": command,
            "collision": torch.tensor(labels.collision_risk, dtype=torch.float32),
            "stall": torch.tensor(labels.stall_risk, dtype=torch.float32),
            "front_clearance_norm": torch.tensor(labels.front_clearance_norm, dtype=torch.float32),
            "progress_norm": torch.tensor(labels.progress_norm, dtype=torch.float32),
        }

    @property
    def action_indices(self) -> list[int]:
        values: list[int] = []
        for record, mirrored in self._entries:
            action_index = int(record["action_index"])
            values.append(_mirrored_action_index(action_index) if mirrored else action_index)
        return values

    def _stack_frame_paths(self, record: dict[str, Any]) -> list[str]:
        if self.frame_stack <= 1:
            return [str(record["image_path"])]
        episode_id = str(record.get("episode_id", ""))
        step_index = int(record.get("step_index", 0))
        steps = self._episode_steps.get(episode_id, {})
        paths: list[str] = []
        for offset in range(self.frame_stack - 1, -1, -1):
            candidate = steps.get(step_index - offset)
            if candidate is None:
                candidate = steps.get(0) or record
            paths.append(str(candidate["image_path"]))
        return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train risk-aware visual policy for the Gym drone game.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--command-loss-weight", type=float, default=0.15)
    parser.add_argument("--collision-loss-weight", type=float, default=1.0)
    parser.add_argument("--stall-loss-weight", type=float, default=0.35)
    parser.add_argument("--clearance-loss-weight", type=float, default=0.5)
    parser.add_argument("--progress-loss-weight", type=float, default=0.35)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--balanced-action-sampler", action="store_true")
    parser.add_argument("--mirror-lateral-actions", action="store_true")
    parser.add_argument("--shield-collision-threshold", type=float, default=0.55)
    parser.add_argument("--shield-front-clearance-m", type=float, default=1.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = RiskPolicyTrainConfig(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        max_depth_m=args.max_depth_m,
        action_loss_weight=args.action_loss_weight,
        command_loss_weight=args.command_loss_weight,
        collision_loss_weight=args.collision_loss_weight,
        stall_loss_weight=args.stall_loss_weight,
        clearance_loss_weight=args.clearance_loss_weight,
        progress_loss_weight=args.progress_loss_weight,
        use_class_weights=not args.no_class_weights,
        balanced_action_sampler=args.balanced_action_sampler,
        mirror_lateral_actions=args.mirror_lateral_actions,
        shield_collision_threshold=args.shield_collision_threshold,
        shield_front_clearance_m=args.shield_front_clearance_m,
    )
    summary = train_risk_policy(config)
    print(json.dumps(summary, indent=2))
    return 0


def train_risk_policy(config: RiskPolicyTrainConfig) -> dict[str, Any]:
    if torch is None or F is None or DataLoader is None:
        raise RuntimeError("torch is required for risk-aware policy training.")
    _seed_everything(config.seed)
    records = load_manifest_records(config.manifest_path)
    if not records:
        raise RuntimeError("The manifest is empty.")
    train_records, val_records = split_manifest_records(records)
    if not train_records:
        raise RuntimeError("The manifest has no training records.")

    device = resolve_device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = RiskManifestDataset(
        train_records,
        image_size=config.image_size,
        frame_stack=config.frame_stack,
        max_depth_m=config.max_depth_m,
        mirror_lateral_actions=config.mirror_lateral_actions,
    )
    train_sampler = _build_balanced_action_sampler(train_dataset.action_indices) if config.balanced_action_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
    )
    val_loader = None
    if val_records:
        val_dataset = RiskManifestDataset(
            val_records,
            image_size=config.image_size,
            frame_stack=config.frame_stack,
            max_depth_m=config.max_depth_m,
            mirror_lateral_actions=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
    model = RiskAwareVisualPolicy(
        num_actions=len(ACTION_VOCAB),
        goal_feature_dim=4,
        frame_stack=config.frame_stack,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    class_weights = _compute_class_weights(train_dataset.action_indices).to(device) if config.use_class_weights else None
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_checkpoint = out_dir / "best.pt"
    last_checkpoint = out_dir / "last.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            config=config,
            class_weights=class_weights,
        )
        val_metrics = None
        if val_loader is not None:
            val_metrics = _run_epoch(
                model,
                val_loader,
                device=device,
                optimizer=None,
                config=config,
                class_weights=class_weights,
            )
        epoch_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_metrics)
        selection_metric = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "action_vocab": ACTION_VOCAB,
            "metrics": epoch_metrics,
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
        "action_vocab": ACTION_VOCAB,
        "best_metric": best_metric,
        "history": history,
        "best_checkpoint": str(best_checkpoint.resolve()),
        "last_checkpoint": str(last_checkpoint.resolve()),
        "config": asdict(config),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir, summary)
    return summary


def _run_epoch(
    model,
    data_loader,
    *,
    device: str,
    optimizer,
    config: RiskPolicyTrainConfig,
    class_weights,
) -> dict[str, Any]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    totals = {
        "examples": 0,
        "loss": 0.0,
        "action_loss": 0.0,
        "command_loss": 0.0,
        "collision_loss": 0.0,
        "stall_loss": 0.0,
        "clearance_loss": 0.0,
        "progress_loss": 0.0,
        "correct": 0,
        "collision_correct": 0,
        "stall_correct": 0,
        "clearance_mae_m": 0.0,
        "progress_mae_m": 0.0,
    }
    for batch in data_loader:
        images = batch["image"].to(device)
        goal_features = batch["goal_features"].to(device)
        action_targets = batch["action_index"].to(device)
        command_targets = batch["command"].to(device)
        collision_targets = batch["collision"].to(device)
        stall_targets = batch["stall"].to(device)
        clearance_targets = batch["front_clearance_norm"].to(device)
        progress_targets = batch["progress_norm"].to(device)
        with torch.set_grad_enabled(train_mode):
            outputs = model(images, goal_features=goal_features)
            action_loss = F.cross_entropy(outputs["action_logits"], action_targets, weight=class_weights)
            command_loss = F.mse_loss(outputs["command_pred"], command_targets)
            collision_loss = F.binary_cross_entropy_with_logits(outputs["collision_logit"], collision_targets)
            stall_loss = F.binary_cross_entropy_with_logits(outputs["stall_logit"], stall_targets)
            clearance_loss = F.mse_loss(torch.sigmoid(outputs["front_clearance_norm"]), clearance_targets)
            progress_loss = F.mse_loss(outputs["progress_norm"], progress_targets)
            loss = (
                config.action_loss_weight * action_loss
                + config.command_loss_weight * command_loss
                + config.collision_loss_weight * collision_loss
                + config.stall_loss_weight * stall_loss
                + config.clearance_loss_weight * clearance_loss
                + config.progress_loss_weight * progress_loss
            )
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = int(images.shape[0])
        totals["examples"] += batch_size
        totals["loss"] += float(loss.item()) * batch_size
        totals["action_loss"] += float(action_loss.item()) * batch_size
        totals["command_loss"] += float(command_loss.item()) * batch_size
        totals["collision_loss"] += float(collision_loss.item()) * batch_size
        totals["stall_loss"] += float(stall_loss.item()) * batch_size
        totals["clearance_loss"] += float(clearance_loss.item()) * batch_size
        totals["progress_loss"] += float(progress_loss.item()) * batch_size
        totals["correct"] += int((torch.argmax(outputs["action_logits"], dim=1) == action_targets).sum().item())
        collision_pred = torch.sigmoid(outputs["collision_logit"]) >= 0.5
        stall_pred = torch.sigmoid(outputs["stall_logit"]) >= 0.5
        totals["collision_correct"] += int((collision_pred == (collision_targets >= 0.5)).sum().item())
        totals["stall_correct"] += int((stall_pred == (stall_targets >= 0.5)).sum().item())
        totals["clearance_mae_m"] += float(
            torch.abs(torch.sigmoid(outputs["front_clearance_norm"]) - clearance_targets).mean().item()
        ) * batch_size * config.max_depth_m
        totals["progress_mae_m"] += float(torch.abs(outputs["progress_norm"] - progress_targets).mean().item()) * batch_size
    n = max(int(totals["examples"]), 1)
    return {
        "loss": totals["loss"] / n,
        "action_loss": totals["action_loss"] / n,
        "command_loss": totals["command_loss"] / n,
        "collision_loss": totals["collision_loss"] / n,
        "stall_loss": totals["stall_loss"] / n,
        "clearance_loss": totals["clearance_loss"] / n,
        "progress_loss": totals["progress_loss"] / n,
        "accuracy": totals["correct"] / n,
        "collision_accuracy": totals["collision_correct"] / n,
        "stall_accuracy": totals["stall_correct"] / n,
        "clearance_mae_m": totals["clearance_mae_m"] / n,
        "progress_mae_m": totals["progress_mae_m"] / n,
        "num_examples": n,
    }


def _command_tensor(record: dict[str, Any], *, mirrored: bool):
    right_m_s = float(record["command"]["right_m_s"])
    yawspeed_deg_s = float(record["command"]["yawspeed_deg_s"])
    if mirrored:
        right_m_s *= -1.0
        yawspeed_deg_s *= -1.0
    return torch.tensor(
        [
            float(record["command"]["forward_m_s"]),
            right_m_s,
            float(record["command"]["down_m_s"]),
            yawspeed_deg_s,
            float(record["command"]["duration_s"]),
        ],
        dtype=torch.float32,
    )


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return requested


def _compute_class_weights(action_indices: list[int]):
    counts = [0 for _ in ACTION_VOCAB]
    for action_index in action_indices:
        counts[int(action_index)] += 1
    nonzero = [count for count in counts if count > 0]
    if not nonzero:
        return torch.ones(len(ACTION_VOCAB), dtype=torch.float32)
    mean_count = float(sum(nonzero)) / float(len(nonzero))
    return torch.tensor(
        [0.0 if count <= 0 else mean_count / float(count) for count in counts],
        dtype=torch.float32,
    )


def _build_balanced_action_sampler(action_indices: list[int]):
    if WeightedRandomSampler is None or not action_indices:
        return None
    counts = [0 for _ in ACTION_VOCAB]
    for action_index in action_indices:
        counts[int(action_index)] += 1
    weights = [0.0 if counts[int(index)] <= 0 else 1.0 / float(counts[int(index)]) for index in action_indices]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(action_indices), replacement=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    rows = []
    for item in summary["history"]:
        train = item["train"]
        val = item.get("val") or {}
        rows.append(
            "<tr>"
            f"<td>{item['epoch']}</td>"
            f"<td>{train['loss']:.4f}</td>"
            f"<td>{train['accuracy']:.4f}</td>"
            f"<td>{val.get('loss', 0.0):.4f}</td>"
            f"<td>{val.get('accuracy', 0.0):.4f}</td>"
            f"<td>{val.get('collision_accuracy', 0.0):.4f}</td>"
            f"<td>{val.get('clearance_mae_m', 0.0):.3f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Risk-Aware Gym Drone Policy Training</title>
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
  <h1>Risk-Aware Gym Drone Policy Training</h1>
  <p>Best metric: <code>{summary['best_metric']:.6f}</code></p>
  <table>
    <thead><tr><th>Epoch</th><th>Train Loss</th><th>Train Acc</th><th>Val Loss</th><th>Val Acc</th><th>Val Collision Acc</th><th>Val Clearance MAE m</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
