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
    ActionConditionedLatentDynamics,
    DYNAMICS_CONTROL_MODES,
    LatentDynamicsSequenceDataset,
    build_valid_anchors,
    build_valid_dynamics_anchors,
    cache_summary,
    compute_normalizer,
    load_sequence_cache,
    make_smoke_sequence_cache,
    split_anchors,
    split_anchors_by_episode,
    split_anchors_by_task_episode,
)

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    torch = None
    F = None
    DataLoader = None


@dataclass
class DynamicsConfig:
    sequence_cache: str
    out_dir: str
    context_len: int = 8
    prediction_horizon: int = 8
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.0
    predict_delta: bool = True
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    split_mode: str = "episode_task"
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0
    control_mode: str = "normal"
    control_seed: int = 0
    future_action_offset: int = 0
    min_future_action_rms: float = 0.0
    min_target_delta_rms: float = 0.0
    early_stopping_patience: int = 0
    early_stopping_metric: str = "z_mse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an action-conditioned latent dynamics model over frozen Kairos/Wan features: "
            "z_context + action_sequence + task -> future z."
        )
    )
    parser.add_argument("--sequence-cache", default="")
    parser.add_argument("--out-dir", default="sensenova_drone_agent/output/action_conditioned_latent_dynamics_v1")
    parser.add_argument("--context-len", type=int, default=8)
    parser.add_argument("--prediction-horizon", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-predict-delta", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=["episode", "episode_task", "anchor"], default="episode_task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--control-mode", choices=sorted(DYNAMICS_CONTROL_MODES), default="normal")
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument(
        "--future-action-offset",
        type=int,
        default=0,
        help=(
            "Shift future action conditioning relative to target future z. "
            "0 means a[t:t+H-1] predicts z[t+1:t+H]; -1 means a[t-1:t+H-2]."
        ),
    )
    parser.add_argument(
        "--min-future-action-rms",
        type=float,
        default=0.0,
        help="Keep only anchors whose future action sequence RMS is at least this value.",
    )
    parser.add_argument(
        "--min-target-delta-rms",
        type=float,
        default=0.0,
        help="Keep only anchors whose target future z delta RMS is at least this value.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument(
        "--early-stopping-metric",
        choices=["loss", "z_mse", "delta_mse"],
        default="z_mse",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--make-smoke-cache", default="")
    parser.add_argument("--smoke-episodes", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=48)
    parser.add_argument("--smoke-z-dim", type=int, default=16)
    parser.add_argument("--smoke-action-dim", type=int, default=3)
    parser.add_argument("--smoke-persistence", type=float, default=0.65)
    parser.add_argument("--smoke-action-scale", type=float, default=0.35)
    parser.add_argument("--smoke-noise-scale", type=float, default=0.02)
    parser.add_argument("--smoke-random-action-fraction", type=float, default=0.65)
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
            persistence=args.smoke_persistence,
            action_scale=args.smoke_action_scale,
            noise_scale=args.smoke_noise_scale,
            random_action_fraction=args.smoke_random_action_fraction,
        )
        print(json.dumps({"smoke_cache": str(path)}, indent=2))
        return 0
    if not args.sequence_cache:
        raise SystemExit("--sequence-cache is required unless --make-smoke-cache is provided.")

    config = DynamicsConfig(
        sequence_cache=str(resolve_path(args.sequence_cache)),
        out_dir=str(resolve_path(args.out_dir)),
        context_len=args.context_len,
        prediction_horizon=args.prediction_horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        predict_delta=not args.no_predict_delta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        split_mode=args.split_mode,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        control_mode=args.control_mode,
        control_seed=args.control_seed,
        future_action_offset=args.future_action_offset,
        min_future_action_rms=args.min_future_action_rms,
        min_target_delta_rms=args.min_target_delta_rms,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
    )
    summary = inspect_cache(config) if args.dry_run else train(config)
    print(json.dumps(compact_summary(summary), indent=2))
    return 0


def inspect_cache(config: DynamicsConfig) -> dict[str, Any]:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_sequence_cache(config.sequence_cache)
    anchors = build_valid_dynamics_anchors(
        cache,
        context_len=config.context_len,
        prediction_horizon=config.prediction_horizon,
        future_action_offset=config.future_action_offset,
    )
    anchors, anchor_filter = filter_dynamics_anchors(cache, anchors, config)
    summary = {
        "phase": "action_conditioned_latent_dynamics_inspection",
        "config": asdict(config),
        "cache": cache_summary(cache, anchors),
        "anchor_filter": anchor_filter,
        "ready": bool(len(anchors) > 0),
        "schema": dynamics_schema(),
    }
    write_json(out_dir / "inspection_summary.json", summary)
    write_report(summary, out_dir / "inspection_report.md")
    return summary


def train(config: DynamicsConfig) -> dict[str, Any]:
    if torch is None or F is None or DataLoader is None:
        raise RuntimeError("torch is required for latent dynamics training.")
    seed_everything(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", asdict(config))

    started = time.time()
    cache = load_sequence_cache(config.sequence_cache)
    anchors = build_valid_dynamics_anchors(
        cache,
        context_len=config.context_len,
        prediction_horizon=config.prediction_horizon,
        future_action_offset=config.future_action_offset,
    )
    anchors, anchor_filter = filter_dynamics_anchors(cache, anchors, config)
    if len(anchors) <= 0:
        raise RuntimeError("No valid sequence anchors. Check episode/step metadata and context/horizon lengths.")
    train_anchors, val_anchors = split_for_config(cache, anchors, config)
    if len(train_anchors) <= 0:
        raise RuntimeError("No train anchors after split.")

    normalizer = compute_normalizer(cache)
    train_ds = LatentDynamicsSequenceDataset(
        cache,
        train_anchors,
        normalizer,
        context_len=config.context_len,
        prediction_horizon=config.prediction_horizon,
        future_action_offset=config.future_action_offset,
        control_mode=config.control_mode,
        control_seed=config.control_seed,
    )
    val_ds = (
        LatentDynamicsSequenceDataset(
            cache,
            val_anchors,
            normalizer,
            context_len=config.context_len,
            prediction_horizon=config.prediction_horizon,
            future_action_offset=config.future_action_offset,
            control_mode=config.control_mode,
            control_seed=config.control_seed + 100003,
        )
        if len(val_anchors) > 0
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
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
    model = ActionConditionedLatentDynamics(
        z_dim=cache.z_dim,
        action_dim=cache.action_dim,
        hidden_dim=config.hidden_dim,
        context_len=config.context_len,
        prediction_horizon=config.prediction_horizon,
        num_tasks=cache.num_tasks,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        predict_delta=config.predict_delta,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: list[dict[str, Any]] = []
    best_by_metric: dict[str, float] = {}
    best_checkpoint_by_metric: dict[str, str] = {}
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_metrics = run_epoch(model, val_loader, optimizer=None, device=device) if val_loader else {}
        epoch_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_metrics)
        payload = checkpoint_payload(model, config, cache, normalizer, epoch, epoch_metrics)
        torch.save(payload, out_dir / "last.pt")
        for metric_name in ["loss", "z_mse", "delta_mse", "persistence_mse"]:
            metric_value = float(val_metrics.get(metric_name, train_metrics.get(metric_name, float("inf"))))
            if metric_value < best_by_metric.get(metric_name, float("inf")):
                best_by_metric[metric_name] = metric_value
                metric_path = out_dir / f"best_{metric_name}.pt"
                torch.save(payload, metric_path)
                best_checkpoint_by_metric[metric_name] = str(metric_path.resolve())
                if metric_name == "loss":
                    torch.save(payload, out_dir / "best.pt")
        monitor_value = float(val_metrics.get(config.early_stopping_metric, train_metrics.get(config.early_stopping_metric, float("inf"))))
        if monitor_value <= best_by_metric.get(config.early_stopping_metric, float("inf")) + 1e-12:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(json.dumps({"phase": "action_conditioned_latent_dynamics", **epoch_metrics}), flush=True)
        if config.early_stopping_patience > 0 and epochs_without_improvement >= config.early_stopping_patience:
            break

    summary = {
        "phase": "action_conditioned_latent_dynamics",
        "interpretation": (
            "Phase-2 controllable latent simulator. Frozen Kairos/Wan features remain fixed; "
            "this trains a small action-conditioned dynamics model over those features."
        ),
        "elapsed_s": time.time() - started,
        "config": asdict(config),
        "cache": cache_summary(cache, anchors),
        "anchor_filter": anchor_filter,
        "train_anchors": int(len(train_anchors)),
        "val_anchors": int(len(val_anchors)),
        "history": history,
        "best_metrics": best_metrics(history),
        "duration_analysis": duration_analysis(history),
        "best_checkpoint_by_metric": best_checkpoint_by_metric,
        "normalizer": normalizer.to_dict(),
        "schema": dynamics_schema(),
        "claim_boundary": "This is not imagination RL. It trains the action-conditioned latent simulator needed before RL.",
    }
    write_json(out_dir / "summary.json", summary)
    write_report(summary, out_dir / "report.md")
    return summary


def run_epoch(model, loader, *, optimizer, device: str) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    totals: dict[str, float] = {}
    if loader is None:
        return {}
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["z_context"], batch["action_context"], batch["future_action"], batch["task_id"])
        predicted = outputs["predicted_z"]
        target = batch["target_z"]
        z_loss = F.mse_loss(predicted, target)
        pred_delta = predicted - batch["z_context"][:, -1:, :]
        target_delta = target - batch["z_context"][:, -1:, :]
        delta_loss = F.mse_loss(pred_delta, target_delta)
        loss = z_loss
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        with torch.no_grad():
            persistence = batch["z_context"][:, -1:, :].expand_as(target)
            persistence_mse = F.mse_loss(persistence, target)
            first_step_mse = F.mse_loss(predicted[:, 0], target[:, 0])
            final_step_mse = F.mse_loss(predicted[:, -1], target[:, -1])
            add_metric(totals, "loss", float(loss.detach().cpu().item()))
            add_metric(totals, "z_mse", float(z_loss.detach().cpu().item()))
            add_metric(totals, "delta_mse", float(delta_loss.detach().cpu().item()))
            add_metric(totals, "persistence_mse", float(persistence_mse.detach().cpu().item()))
            add_metric(totals, "first_step_z_mse", float(first_step_mse.detach().cpu().item()))
            add_metric(totals, "final_step_z_mse", float(final_step_mse.detach().cpu().item()))
            add_metric(totals, "batches", 1.0)
    batches = max(totals.pop("batches", 1.0), 1.0)
    metrics = {key: value / batches for key, value in totals.items()}
    if metrics.get("persistence_mse", 0.0) > 0:
        metrics["z_mse_vs_persistence_ratio"] = metrics["z_mse"] / metrics["persistence_mse"]
    return metrics | {"batches": batches}


def split_for_config(cache, anchors: np.ndarray, config: DynamicsConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.split_mode == "episode":
        return split_anchors_by_episode(cache, anchors, val_ratio=config.val_ratio, seed=config.seed)
    if config.split_mode == "episode_task":
        return split_anchors_by_task_episode(cache, anchors, val_ratio=config.val_ratio, seed=config.seed)
    if config.split_mode == "anchor":
        return split_anchors(anchors, val_ratio=config.val_ratio, seed=config.seed)
    raise ValueError(f"Unsupported split_mode={config.split_mode!r}")


def filter_dynamics_anchors(cache, anchors: np.ndarray, config: DynamicsConfig) -> tuple[np.ndarray, dict[str, Any]]:
    anchors = np.asarray(anchors, dtype=np.int64)
    if anchors.size <= 0:
        return anchors, {
            "enabled": False,
            "input_anchors": 0,
            "kept_anchors": 0,
            "min_future_action_rms": float(config.min_future_action_rms),
            "min_target_delta_rms": float(config.min_target_delta_rms),
        }
    min_future_action_rms = float(config.min_future_action_rms)
    min_target_delta_rms = float(config.min_target_delta_rms)
    if min_future_action_rms <= 0.0 and min_target_delta_rms <= 0.0:
        return anchors, {
            "enabled": False,
            "input_anchors": int(anchors.size),
            "kept_anchors": int(anchors.size),
            "min_future_action_rms": min_future_action_rms,
            "min_target_delta_rms": min_target_delta_rms,
        }

    keep: list[int] = []
    action_rms_values: list[float] = []
    delta_rms_values: list[float] = []
    for anchor in anchors:
        anchor_int = int(anchor)
        future_action_start = anchor_int + int(config.future_action_offset)
        future_action_end = future_action_start + int(config.prediction_horizon)
        target_start = anchor_int + 1
        target_end = anchor_int + int(config.prediction_horizon) + 1
        future_action = cache.action[future_action_start:future_action_end].astype(np.float32)
        target_z = cache.z[target_start:target_end].astype(np.float32)
        z0 = cache.z[anchor_int : anchor_int + 1].astype(np.float32)
        future_action_rms = float(np.sqrt(np.mean(np.square(future_action)))) if future_action.size else 0.0
        target_delta_rms = float(np.sqrt(np.mean(np.square(target_z - z0)))) if target_z.size else 0.0
        action_rms_values.append(future_action_rms)
        delta_rms_values.append(target_delta_rms)
        if future_action_rms < min_future_action_rms:
            continue
        if target_delta_rms < min_target_delta_rms:
            continue
        keep.append(anchor_int)

    action_stats = summarize_values(action_rms_values)
    delta_stats = summarize_values(delta_rms_values)
    return np.asarray(keep, dtype=np.int64), {
        "enabled": True,
        "input_anchors": int(anchors.size),
        "kept_anchors": int(len(keep)),
        "kept_fraction": float(len(keep) / max(1, anchors.size)),
        "min_future_action_rms": min_future_action_rms,
        "min_target_delta_rms": min_target_delta_rms,
        "future_action_rms": action_stats,
        "target_delta_rms": delta_stats,
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def best_metrics(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {}

    def best_for(metric: str) -> dict[str, Any]:
        candidates = [item for item in history if item.get("val") and metric in item["val"]]
        if not candidates:
            return {}
        best = min(candidates, key=lambda item: float(item["val"][metric]))
        return {"epoch": int(best["epoch"]), metric: float(best["val"][metric]), "val": best["val"]}

    return {
        "best_val_loss": best_for("loss"),
        "best_val_z_mse": best_for("z_mse"),
        "best_val_delta_mse": best_for("delta_mse"),
        "best_val_persistence_mse": best_for("persistence_mse"),
    }


def duration_analysis(history: list[dict[str, Any]], *, tail: int = 10) -> dict[str, Any]:
    values = [
        (int(item["epoch"]), float(item.get("val", {}).get("z_mse")))
        for item in history
        if item.get("val") and "z_mse" in item["val"]
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
        "first_val_z_mse": first_value,
        "last_epoch": last_epoch,
        "last_val_z_mse": last_value,
        "best_epoch": best_epoch,
        "best_val_z_mse": best_value,
        "tail_best_epoch": tail_best_epoch,
        "tail_best_val_z_mse": tail_best_value,
        "tail_relative_gain": tail_relative_gain,
        "best_at_final_epoch": best_epoch == last_epoch,
    }


def checkpoint_payload(model, config, cache, normalizer, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_type": "action_conditioned_latent_dynamics",
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "cache": cache_summary(cache),
        "normalizer": normalizer.to_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "claim_boundary": "Frozen-feature action-conditioned latent dynamics. Kairos/Sensenova is not updated.",
    }


def dynamics_schema() -> dict[str, Any]:
    return {
        "input": {
            "z_context": "(B, C, z_dim) frozen latent context z[t-C+1:t]",
            "action_context": "(B, C, action_dim) aligned action context a[t-C+1:t]",
            "future_action": "(B, H, action_dim) candidate/teacher action sequence a[t:t+H-1]",
            "task_id": "(B,) task conditioning",
        },
        "target": "future latent sequence z[t+1:t+H]",
        "future_action_offset": "Configured offset shifts candidate actions before rollout for alignment audits.",
        "decision_gate": "normal dynamics should beat shuffled/zero future-action controls and persistence.",
    }


def write_report(summary: dict[str, Any], out_path: Path) -> None:
    cache = summary["cache"]
    lines = [
        "# Action-Conditioned Latent Dynamics",
        "",
        "This trains the controllable simulator piece: `z_context + actions -> future_z`.",
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
    ]
    if "train_anchors" in summary:
        best = summary.get("best_metrics", {}).get("best_val_z_mse", {})
        lines.extend(
            [
                f"- Train anchors: `{summary['train_anchors']}`",
                f"- Val anchors: `{summary['val_anchors']}`",
                f"- Best val z MSE: `{best.get('z_mse')}` at epoch `{best.get('epoch')}`",
                f"- Best z checkpoint: `{summary.get('best_checkpoint_by_metric', {}).get('z_mse')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            summary.get("claim_boundary", "This is not imagination RL."),
            "",
            "## Gate",
            "",
            "A usable simulator must beat persistence and shuffled/zero-action controls on held-out episodes.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
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
        compact["best_metrics"] = summary.get("best_metrics")
        compact["best_checkpoint_by_metric"] = summary.get("best_checkpoint_by_metric")
        history = summary.get("history") or []
        if history:
            compact["first_epoch"] = history[0]
            compact["last_epoch"] = history[-1]
    return compact


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
