#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sensenova_drone.midtraining import (  # noqa: E402
    BehaviorCloningMidtrainingHead,
    CONTROL_MODES,
    MidtrainingSequenceDataset,
    build_valid_anchors,
    cache_summary,
    compute_normalizer,
    load_sequence_cache,
    make_smoke_sequence_cache,
    split_anchors_by_episode,
    split_anchors_by_task_episode,
    split_anchors,
)

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
except ModuleNotFoundError:
    torch = None
    F = None
    DataLoader = None
    WeightedRandomSampler = None


@dataclass
class MidtrainingConfig:
    sequence_cache: str
    out_dir: str
    context_len: int = 8
    mtp_horizon: int = 8
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.0
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    reward_loss_weight: float = 1.0
    value_loss_weight: float = 0.2
    val_ratio: float = 0.1
    split_mode: str = "episode"
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0
    control_mode: str = "normal"
    control_seed: int = 0
    agent_token_isolation: bool = True
    bc_positive_reward_only: bool = False
    relevant_sample_fraction: float = 0.0
    relevant_reward_threshold: float = 0.0
    early_stopping_patience: int = 0
    early_stopping_metric: str = "loss"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-2 behavior-cloning midtraining: train policy/reward heads on frozen world-model "
            "latents using Dreamer-style multi-token prediction."
        )
    )
    parser.add_argument("--sequence-cache", default="")
    parser.add_argument("--out-dir", default="sensenova_drone_agent/output/bc_midtraining_v1")
    parser.add_argument("--context-len", type=int, default=8)
    parser.add_argument("--mtp-horizon", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--split-mode",
        choices=["episode", "episode_task", "anchor"],
        default="episode",
        help=(
            "episode holds out whole trajectories; episode_task holds out whole trajectories within tasks and keeps "
            "singleton tasks in train; anchor matches the older random-window split."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--control-mode",
        choices=sorted(CONTROL_MODES),
        default="normal",
        help=(
            "Baseline/control perturbation. normal keeps data aligned; shuffle_targets breaks future action/reward "
            "alignment; shuffle_z_context breaks visual-latent alignment; zero_z_context removes visual latents; "
            "zero_prev_actions removes previous-action context."
        ),
    )
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument(
        "--no-agent-token-isolation",
        action="store_true",
        help="Disable Dreamer-style attention isolation between world/context tokens and agent/task tokens.",
    )
    parser.add_argument(
        "--bc-positive-reward-only",
        action="store_true",
        help="Apply the action BC loss only to target steps with positive raw reward; reward/value losses still use all data.",
    )
    parser.add_argument(
        "--relevant-sample-fraction",
        type=float,
        default=0.0,
        help=(
            "If >0, sample train windows from a mixture where this fraction is reward-relevant "
            "and the remainder is uniform/non-relevant. Dreamer 4 uses a 0.5 relevant mixture."
        ),
    )
    parser.add_argument(
        "--relevant-reward-threshold",
        type=float,
        default=0.0,
        help="A train window is reward-relevant if any raw target reward is greater than this threshold.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument(
        "--early-stopping-metric",
        choices=["loss", "action_mse", "bc_action_mse", "all_action_mse"],
        default="loss",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate cache/windows and write a report without training.")
    parser.add_argument("--make-smoke-cache", default="", help="Create a tiny synthetic cache at this path and exit.")
    parser.add_argument("--smoke-episodes", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=48)
    parser.add_argument("--smoke-z-dim", type=int, default=16)
    parser.add_argument("--smoke-action-dim", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.make_smoke_cache:
        path = resolve_path(args.make_smoke_cache)
        make_smoke_sequence_cache(
            path,
            episodes=args.smoke_episodes,
            steps=args.smoke_steps,
            z_dim=args.smoke_z_dim,
            action_dim=args.smoke_action_dim,
        )
        print(json.dumps({"smoke_cache": str(path)}, indent=2))
        return 0
    if not args.sequence_cache:
        raise SystemExit("--sequence-cache is required unless --make-smoke-cache is provided.")

    config = MidtrainingConfig(
        sequence_cache=str(resolve_path(args.sequence_cache)),
        out_dir=str(resolve_path(args.out_dir)),
        context_len=args.context_len,
        mtp_horizon=args.mtp_horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        reward_loss_weight=args.reward_loss_weight,
        value_loss_weight=args.value_loss_weight,
        val_ratio=args.val_ratio,
        split_mode=args.split_mode,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        control_mode=args.control_mode,
        control_seed=args.control_seed,
        agent_token_isolation=not args.no_agent_token_isolation,
        bc_positive_reward_only=args.bc_positive_reward_only,
        relevant_sample_fraction=args.relevant_sample_fraction,
        relevant_reward_threshold=args.relevant_reward_threshold,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
    )
    if args.dry_run:
        summary = inspect_cache(config)
    else:
        summary = train(config)
    print(json.dumps(compact_summary_for_stdout(summary), indent=2))
    return 0


def inspect_cache(config: MidtrainingConfig) -> dict[str, Any]:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_sequence_cache(config.sequence_cache)
    anchors = build_valid_anchors(cache, context_len=config.context_len, mtp_horizon=config.mtp_horizon)
    summary = {
        "phase": "bc_midtraining_inspection",
        "config": asdict(config),
        "cache": cache_summary(cache, anchors),
        "training_mixture": training_mixture_summary(cache, anchors, config),
        "ready": bool(len(anchors) > 0),
        "schema": sequence_cache_schema(),
    }
    write_report(summary, out_dir / "inspection_report.md")
    (out_dir / "inspection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train(config: MidtrainingConfig) -> dict[str, Any]:
    if torch is None or F is None or DataLoader is None:
        raise RuntimeError("torch is required for BC midtraining.")
    seed_everything(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    started = time.time()
    cache = load_sequence_cache(config.sequence_cache)
    anchors = build_valid_anchors(cache, context_len=config.context_len, mtp_horizon=config.mtp_horizon)
    if len(anchors) <= 0:
        raise RuntimeError("No valid sequence anchors. Check episode/step metadata and context/horizon lengths.")
    if config.split_mode == "episode":
        train_anchors, val_anchors = split_anchors_by_episode(
            cache, anchors, val_ratio=config.val_ratio, seed=config.seed
        )
    elif config.split_mode == "episode_task":
        train_anchors, val_anchors = split_anchors_by_task_episode(
            cache, anchors, val_ratio=config.val_ratio, seed=config.seed
        )
    elif config.split_mode == "anchor":
        train_anchors, val_anchors = split_anchors(anchors, val_ratio=config.val_ratio, seed=config.seed)
    else:
        raise ValueError(f"Unsupported split_mode={config.split_mode!r}")
    if len(train_anchors) <= 0:
        raise RuntimeError("No train anchors after split.")
    normalizer = compute_normalizer(cache)
    train_ds = MidtrainingSequenceDataset(
        cache,
        train_anchors,
        normalizer,
        context_len=config.context_len,
        mtp_horizon=config.mtp_horizon,
        control_mode=config.control_mode,
        control_seed=config.control_seed,
    )
    val_ds = (
        MidtrainingSequenceDataset(
            cache,
            val_anchors,
            normalizer,
            context_len=config.context_len,
            mtp_horizon=config.mtp_horizon,
            control_mode=config.control_mode,
            control_seed=config.control_seed + 100003,
        )
        if len(val_anchors) > 0
        else None
    )
    train_sampler, mixture_summary = build_relevant_sampler(cache, train_anchors, config)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )
        if val_ds is not None
        else None
    )
    device = resolve_device(config.device)
    model = BehaviorCloningMidtrainingHead(
        z_dim=cache.z_dim,
        action_dim=cache.action_dim,
        hidden_dim=config.hidden_dim,
        context_len=config.context_len,
        mtp_horizon=config.mtp_horizon,
        num_tasks=cache.num_tasks,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        agent_token_isolation=config.agent_token_isolation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_by_metric: dict[str, float] = {}
    best_checkpoint_by_metric: dict[str, str] = {}
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, device=device, config=config)
        val_metrics = run_epoch(model, val_loader, optimizer=None, device=device, config=config) if val_loader else {}
        epoch_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_metrics)
        val_loss = float(val_metrics.get("loss", train_metrics["loss"]))
        payload = checkpoint_payload(model, config, cache, normalizer, epoch, epoch_metrics)
        torch.save(payload, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, out_dir / "best.pt")
        for metric_name in ["loss", "action_mse", "bc_action_mse", "all_action_mse", "reward_mse", "value_mse"]:
            if metric_name not in val_metrics:
                continue
            metric_value = float(val_metrics[metric_name])
            if metric_value < best_by_metric.get(metric_name, float("inf")):
                best_by_metric[metric_name] = metric_value
                metric_path = out_dir / f"best_{metric_name}.pt"
                torch.save(payload, metric_path)
                best_checkpoint_by_metric[metric_name] = str(metric_path.resolve())
        monitor_value = float(val_metrics.get(config.early_stopping_metric, train_metrics.get(config.early_stopping_metric, val_loss)))
        if monitor_value <= best_by_metric.get(config.early_stopping_metric, float("inf")) + 1e-12:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(json.dumps({"phase": "bc_midtraining", **epoch_metrics}), flush=True)
        if config.early_stopping_patience > 0 and epochs_without_improvement >= config.early_stopping_patience:
            break

    summary = {
        "phase": "bc_midtraining",
        "interpretation": (
            "Phase-2 agent-head preconditioning. The frozen world-model features are not trained here; "
            "only policy/reward/value heads learn from dataset behavior."
        ),
        "elapsed_s": time.time() - started,
        "config": asdict(config),
        "cache": cache_summary(cache, anchors),
        "train_anchors": int(len(train_anchors)),
        "val_anchors": int(len(val_anchors)),
        "history": history,
        "best_metrics": best_metrics(history),
        "duration_analysis": duration_analysis(history),
        "best_checkpoint_by_metric": best_checkpoint_by_metric,
        "training_mixture": mixture_summary,
        "normalizer": normalizer.to_dict(),
        "best_checkpoint": str((out_dir / "best.pt").resolve()),
        "last_checkpoint": str((out_dir / "last.pt").resolve()),
        "schema": sequence_cache_schema(),
        "next_phase": "imagination_rl_after_bc_midtraining",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, out_dir / "report.md")
    return summary


def run_epoch(model, loader, *, optimizer, device: str, config: MidtrainingConfig) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    totals: dict[str, float] = {}
    if loader is None:
        return {}
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["z_context"], batch["prev_action_context"], batch["task_id"])
        bc_action_loss = masked_action_mse(
            outputs["action_pred"],
            batch["target_action"],
            batch["target_reward_raw"] > 0.0,
        )
        if config.bc_positive_reward_only:
            action_loss = bc_action_loss if bc_action_loss is not None else outputs["action_pred"].sum() * 0.0
        else:
            action_loss = F.mse_loss(outputs["action_pred"], batch["target_action"])
        reward_loss = F.mse_loss(outputs["reward_pred"], batch["target_reward"])
        value_target = discounted_returns(batch["target_reward"])
        value_loss = F.mse_loss(outputs["value"], value_target)
        loss = (
            action_loss
            + config.reward_loss_weight * reward_loss
            + config.value_loss_weight * value_loss
        )
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        with torch.no_grad():
            first_action_mse = F.mse_loss(outputs["action_pred"][:, 0], batch["target_action"][:, 0])
            final_action_mse = F.mse_loss(outputs["action_pred"][:, -1], batch["target_action"][:, -1])
            add_metric(totals, "loss", float(loss.detach().cpu().item()))
            add_metric(totals, "action_mse", float(action_loss.detach().cpu().item()))
            all_action_mse = F.mse_loss(outputs["action_pred"], batch["target_action"])
            add_metric(totals, "all_action_mse", float(all_action_mse.detach().cpu().item()))
            if bc_action_loss is not None:
                add_metric(totals, "bc_action_mse", float(bc_action_loss.detach().cpu().item()))
                add_metric(totals, "bc_action_batches", 1.0)
            add_metric(totals, "reward_mse", float(reward_loss.detach().cpu().item()))
            add_metric(totals, "value_mse", float(value_loss.detach().cpu().item()))
            add_metric(totals, "first_action_mse", float(first_action_mse.detach().cpu().item()))
            add_metric(totals, "final_action_mse", float(final_action_mse.detach().cpu().item()))
            add_metric(totals, "batches", 1.0)
    batches = max(totals.pop("batches", 1.0), 1.0)
    bc_action_batches = max(totals.pop("bc_action_batches", 0.0), 0.0)
    metrics = {key: value / batches for key, value in totals.items()}
    if bc_action_batches > 0:
        metrics["bc_action_mse"] = totals.get("bc_action_mse", 0.0) / bc_action_batches
    return metrics | {"batches": batches}


def build_relevant_sampler(cache, train_anchors: np.ndarray, config: MidtrainingConfig):
    fraction = float(config.relevant_sample_fraction)
    summary = training_mixture_summary(cache, train_anchors, config)
    if fraction <= 0.0 or WeightedRandomSampler is None or torch is None:
        return None, summary | {"sampler_enabled": False}
    fraction = min(max(fraction, 0.0), 1.0)
    relevant = reward_relevant_mask(cache, train_anchors, config.mtp_horizon, config.relevant_reward_threshold)
    relevant_count = int(np.sum(relevant))
    non_relevant_count = int(relevant.size - relevant_count)
    if relevant_count <= 0 or non_relevant_count <= 0:
        return None, summary | {"sampler_enabled": False, "fallback_reason": "missing_relevant_or_non_relevant"}
    weights = np.empty(relevant.shape[0], dtype=np.float64)
    weights[relevant] = fraction / max(relevant_count, 1)
    weights[~relevant] = (1.0 - fraction) / max(non_relevant_count, 1)
    generator = torch.Generator()
    generator.manual_seed(int(config.seed))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(train_anchors.shape[0]),
        replacement=True,
        generator=generator,
    )
    return sampler, summary | {"sampler_enabled": True}


def training_mixture_summary(cache, anchors: np.ndarray, config: MidtrainingConfig) -> dict[str, Any]:
    relevant = reward_relevant_mask(cache, anchors, config.mtp_horizon, config.relevant_reward_threshold)
    relevant_count = int(np.sum(relevant))
    total = int(relevant.shape[0])
    return {
        "requested_relevant_sample_fraction": float(config.relevant_sample_fraction),
        "relevant_reward_threshold": float(config.relevant_reward_threshold),
        "train_windows": total,
        "relevant_windows": relevant_count,
        "non_relevant_windows": int(total - relevant_count),
        "raw_relevant_fraction": float(relevant_count / total) if total else 0.0,
    }


def reward_relevant_mask(
    cache,
    anchors: np.ndarray,
    mtp_horizon: int,
    relevant_reward_threshold: float,
) -> np.ndarray:
    mask = np.zeros(int(anchors.shape[0]), dtype=bool)
    for idx, anchor in enumerate(np.asarray(anchors, dtype=np.int64)):
        target = cache.reward[int(anchor) : int(anchor) + int(mtp_horizon) + 1]
        mask[idx] = bool(np.any(target > float(relevant_reward_threshold)))
    return mask


def masked_action_mse(pred, target, mask):
    expanded = mask.unsqueeze(-1).expand_as(pred)
    if not bool(expanded.any().detach().cpu().item()):
        return None
    diff = pred[expanded] - target[expanded]
    return torch.mean(diff * diff)


def discounted_returns(reward_seq, gamma: float = 0.97):
    ret = torch.zeros_like(reward_seq[:, 0])
    for idx in reversed(range(reward_seq.shape[1])):
        ret = reward_seq[:, idx] + gamma * ret
    return ret


def best_metrics(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {}

    def best_for(metric: str) -> dict[str, Any]:
        candidates = [
            item
            for item in history
            if item.get("val") and metric in item["val"]
        ]
        if not candidates:
            return {}
        best = min(candidates, key=lambda item: float(item["val"][metric]))
        return {"epoch": int(best["epoch"]), metric: float(best["val"][metric]), "val": best["val"]}

    return {
        "best_val_loss": best_for("loss"),
        "best_val_action_mse": best_for("action_mse"),
        "best_val_bc_action_mse": best_for("bc_action_mse"),
        "best_val_all_action_mse": best_for("all_action_mse"),
        "best_val_reward_mse": best_for("reward_mse"),
        "best_val_value_mse": best_for("value_mse"),
    }


def duration_analysis(history: list[dict[str, Any]], *, tail: int = 10) -> dict[str, Any]:
    values = [
        (int(item["epoch"]), float(item.get("val", {}).get("action_mse")))
        for item in history
        if item.get("val") and "action_mse" in item["val"]
    ]
    if not values:
        return {}
    first_epoch, first_value = values[0]
    last_epoch, last_value = values[-1]
    best_epoch, best_value = min(values, key=lambda item: item[1])
    tail_values = values[-min(tail, len(values)) :]
    tail_best_epoch, tail_best_value = min(tail_values, key=lambda item: item[1])
    previous_values = values[: max(0, len(values) - len(tail_values))]
    previous_best = min((value for _, value in previous_values), default=first_value)
    tail_relative_gain = (previous_best - tail_best_value) / max(abs(previous_best), 1e-9)
    return {
        "first_epoch": first_epoch,
        "first_val_action_mse": first_value,
        "last_epoch": last_epoch,
        "last_val_action_mse": last_value,
        "best_epoch": best_epoch,
        "best_val_action_mse": best_value,
        "tail_best_epoch": tail_best_epoch,
        "tail_best_val_action_mse": tail_best_value,
        "tail_relative_gain": tail_relative_gain,
        "best_at_final_epoch": best_epoch == last_epoch,
    }


def checkpoint_payload(
    model,
    config: MidtrainingConfig,
    cache,
    normalizer,
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_type": "behavior_cloning_midtraining_head",
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "cache": cache_summary(cache),
        "normalizer": normalizer.to_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "claim_boundary": (
            "This is phase-2 behavior cloning on frozen world-model features. "
            "It is not imagination RL and does not update Kairos/Sensenova."
        ),
    }


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def add_metric(totals: dict[str, float], key: str, value: float) -> None:
    totals[key] = totals.get(key, 0.0) + float(value)


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def sequence_cache_schema() -> dict[str, Any]:
    return {
        "required": {
            "z": "(N, z_dim) frozen world-model/Kairos features",
            "action": "(N, action_dim) action labels",
        },
        "optional": {
            "reward": "(N,) scalar reward/success/hindsight score; defaults to zeros",
            "episode": "(N,) trajectory id; defaults to one episode",
            "step": "(N,) timestep; defaults to arange(N)",
            "task_id": "(N,) integer task id; defaults to zeros",
        },
        "target": "h_t -> action[t:t+L], reward[t:t+L]",
    }


def write_report(summary: dict[str, Any], out_path: Path) -> None:
    cache = summary["cache"]
    lines = [
        "# Behavior Cloning Midtraining",
        "",
        "This is phase 2: train policy/reward/value heads on frozen world-model features.",
        "",
        "## Result",
        "",
        f"- Phase: `{summary['phase']}`",
        f"- Steps: `{cache['steps']}`",
        f"- Valid anchors: `{cache.get('valid_anchors', 'n/a')}`",
        f"- Feature dim: `{cache['z_dim']}`",
        f"- Action dim: `{cache['action_dim']}`",
        f"- Episodes: `{cache['episodes']}`",
        f"- Tasks: `{cache['tasks']}`",
        f"- Control mode: `{summary.get('config', {}).get('control_mode', 'normal')}`",
        f"- Split mode: `{summary.get('config', {}).get('split_mode', 'episode')}`",
        f"- Agent-token isolation: `{summary.get('config', {}).get('agent_token_isolation', True)}`",
    ]
    mixture = summary.get("training_mixture", {})
    if mixture:
        lines.extend(
            [
                f"- Relevant sample fraction requested: `{mixture.get('requested_relevant_sample_fraction')}`",
                f"- Relevant windows: `{mixture.get('relevant_windows')}`",
                f"- Non-relevant windows: `{mixture.get('non_relevant_windows')}`",
                f"- Relevant sampler enabled: `{mixture.get('sampler_enabled', False)}`",
            ]
        )
    if "train_anchors" in summary:
        lines.extend(
            [
                f"- Train anchors: `{summary['train_anchors']}`",
                f"- Val anchors: `{summary['val_anchors']}`",
                f"- Best checkpoint: `{summary['best_checkpoint']}`",
            ]
        )
        best = summary.get("best_metrics", {}).get("best_val_action_mse", {})
        duration = summary.get("duration_analysis", {})
        if best:
            lines.append(f"- Best val action MSE: `{best.get('action_mse')}` at epoch `{best.get('epoch')}`")
        if duration:
            lines.append(f"- Tail relative gain: `{duration.get('tail_relative_gain')}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This does not train the world model and does not run imagination RL. It prepares the behavioral prior for the next phase.",
            "",
            "## Cache Schema",
            "",
            "```text",
            "z       required frozen features, shape (N, z_dim)",
            "action  required labels, shape (N, action_dim)",
            "reward  optional scalar, shape (N,)",
            "episode optional trajectory id, shape (N,)",
            "step    optional timestep, shape (N,)",
            "task_id optional task id, shape (N,)",
            "```",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_summary_for_stdout(summary: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "phase": summary.get("phase"),
        "cache": summary.get("cache"),
        "schema": summary.get("schema"),
    }
    if "config" in summary:
        compact["control"] = {
            "mode": summary["config"].get("control_mode", "normal"),
            "seed": summary["config"].get("control_seed", 0),
        }
    if "ready" in summary:
        compact["ready"] = summary["ready"]
    if "train_anchors" in summary:
        compact["train_anchors"] = summary.get("train_anchors")
        compact["val_anchors"] = summary.get("val_anchors")
        compact["best_checkpoint"] = summary.get("best_checkpoint")
        history = summary.get("history") or []
        if history:
            compact["first_epoch"] = history[0]
            compact["last_epoch"] = history[-1]
    if "normalizer" in summary:
        normalizer = summary["normalizer"]
        compact["normalizer_summary"] = {
            "z_dim": len(normalizer.get("z_mean", [])),
            "action_dim": len(normalizer.get("action_mean", [])),
            "reward_mean": normalizer.get("reward_mean"),
            "reward_std": normalizer.get("reward_std"),
        }
    if "training_mixture" in summary:
        compact["training_mixture"] = summary["training_mixture"]
    return compact


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
