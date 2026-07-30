#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
    BehaviorCloningMidtrainingHead,
    Normalizer,
    SequenceCache,
    cache_summary,
    compute_normalizer,
    load_sequence_cache,
    make_smoke_sequence_cache,
    normalize_np,
    split_anchors,
    split_anchors_by_episode,
    split_anchors_by_task_episode,
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
    Dataset = object  # type: ignore[assignment]


@dataclass
class SoarDreamerLiteConfig:
    sequence_cache: str
    out_dir: str
    stage: str = "all"
    checkpoint: str = ""
    context_len: int = 8
    prediction_horizon: int = 4
    mtp_horizon: int = 8
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    head_hidden_dim: int = 0
    head_layers: int = 1
    dynamics_architecture: str = "pooled"
    dynamics_residual_mode: str = "none"
    dropout: float = 0.0
    predict_delta: bool = True
    future_action_offset: int = 0
    future_action_window: int = 1
    future_action_reduce: str = "mean"
    motion_filter_quantile: float = 0.0
    min_motion_norm: float = 0.0
    delta_loss_weight: float = 0.0
    contrastive_loss_weight: float = 0.0
    contrastive_margin: float = 0.02
    dynamics_rollout_loss_weight: float = 0.0
    dynamics_rollout_contrastive_loss_weight: float = 0.0
    dynamics_rollout_contrastive_margin: float = 0.02
    dynamics_rollout_horizon: int = 0
    bc_epochs: int = 20
    imagination_epochs: int = 10
    imagination_horizon: int = 8
    batch_size: int = 128
    learning_rate: float = 1e-3
    imagination_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dynamics_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    value_loss_weight: float = 0.2
    reward_target_mode: str = "normalized"
    reward_loss_type: str = "mse"
    value_target_mode: str = "normalized_discounted_sum"
    value_loss_type: str = "mse"
    value_huber_beta: float = 1.0
    imagination_value_loss_weight: float = 0.5
    imagination_train_value_head: bool = True
    real_value_replay_loss_weight: float = 0.0
    prior_loss_weight: float = 0.3
    gamma: float = 0.997
    val_ratio: float = 0.1
    split_mode: str = "episode_task"
    dynamics_bc_metric: str = "loss"
    dynamics_bc_early_stop_patience: int = 0
    dynamics_bc_min_delta: float = 0.0
    agent_bc_metric: str = "loss"
    agent_bc_early_stop_patience: int = 0
    agent_bc_min_delta: float = 0.0
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0
    agent_token_isolation: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SOAR-only Dreamer-lite training: frozen Kairos/SOAR latents -> action-conditioned "
            "latent dynamics -> isolated agent-token BC/reward/value heads -> imagination training."
        )
    )
    parser.add_argument("--sequence-cache", default="")
    parser.add_argument("--out-dir", default="sensenova_drone_agent/output/soar_dreamer_lite_v1")
    parser.add_argument(
        "--stage",
        choices=["inspect", "dynamics_bc", "agent_bc", "agent_bc_imagination", "imagination", "all"],
        default="all",
    )
    parser.add_argument("--checkpoint", default="", help="Checkpoint to load before imagination-only training.")
    parser.add_argument("--context-len", type=int, default=8)
    parser.add_argument("--prediction-horizon", type=int, default=4)
    parser.add_argument("--mtp-horizon", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-hidden-dim", type=int, default=0)
    parser.add_argument("--head-layers", type=int, default=1)
    parser.add_argument(
        "--dynamics-architecture",
        choices=["pooled", "action_query"],
        default="pooled",
        help="Latent dynamics architecture. pooled is the legacy rollout-token model; action_query interleaves action and query tokens.",
    )
    parser.add_argument(
        "--dynamics-residual-mode",
        choices=["none", "action_gated"],
        default="none",
        help="Optional residual adapter. action_gated predicts a gated residual over persistence.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-predict-delta", action="store_true")
    parser.add_argument(
        "--future-action-offset",
        type=int,
        default=0,
        help="Offset for future action conditioning. Use to test action/frame lag; 0 means action[t:t+H-1].",
    )
    parser.add_argument(
        "--future-action-window",
        type=int,
        default=1,
        help="Number of consecutive future action rows to aggregate for each future action token.",
    )
    parser.add_argument(
        "--future-action-reduce",
        choices=["mean", "sum", "first"],
        default="mean",
        help="Reduction for --future-action-window aggregation.",
    )
    parser.add_argument(
        "--motion-filter-quantile",
        type=float,
        default=0.0,
        help="Keep only anchors whose future latent motion is at or above this quantile. 0 disables it.",
    )
    parser.add_argument(
        "--min-motion-norm",
        type=float,
        default=0.0,
        help="Absolute minimum future latent motion norm for anchor retention. 0 disables it.",
    )
    parser.add_argument(
        "--delta-loss-weight",
        type=float,
        default=0.0,
        help="Extra loss on predicted latent deltas z[t+k]-z[t], emphasizing motion over static reconstruction.",
    )
    parser.add_argument(
        "--contrastive-loss-weight",
        type=float,
        default=0.0,
        help="Extra loss requiring true future actions to beat shuffled and zero-action futures.",
    )
    parser.add_argument("--contrastive-margin", type=float, default=0.02)
    parser.add_argument(
        "--dynamics-rollout-loss-weight",
        type=float,
        default=0.0,
        help="Autoregressive dynamics consistency loss weight using true future actions.",
    )
    parser.add_argument(
        "--dynamics-rollout-contrastive-loss-weight",
        type=float,
        default=0.0,
        help="Autoregressive true-vs-shuffled/zero action contrastive loss weight.",
    )
    parser.add_argument("--dynamics-rollout-contrastive-margin", type=float, default=0.02)
    parser.add_argument(
        "--dynamics-rollout-horizon",
        type=int,
        default=0,
        help="Autoregressive dynamics training horizon. 0 disables rollout-consistency targets.",
    )
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--imagination-epochs", type=int, default=10)
    parser.add_argument("--imagination-horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--imagination-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dynamics-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--reward-target-mode",
        choices=["normalized", "raw"],
        default="normalized",
        help="Reward target scale for agent BC. Use raw with BCE for binary SOAR success labels.",
    )
    parser.add_argument(
        "--reward-loss-type",
        choices=["mse", "bce"],
        default="mse",
        help="Reward loss. BCE treats reward head outputs as logits and reports calibrated probabilities.",
    )
    parser.add_argument(
        "--value-target-mode",
        choices=["normalized_discounted_sum", "raw_discounted_sum", "raw_discounted_mean"],
        default="normalized_discounted_sum",
        help="Value target scale. raw_* modes avoid reward-normalization artifacts for SOAR success labels.",
    )
    parser.add_argument("--value-loss-type", choices=["mse", "huber"], default="mse")
    parser.add_argument("--value-huber-beta", type=float, default=1.0)
    parser.add_argument("--imagination-value-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--imagination-freeze-value-head",
        action="store_true",
        help="Keep the calibrated BC value head fixed during imagination; train only action heads.",
    )
    parser.add_argument(
        "--real-value-replay-loss-weight",
        type=float,
        default=0.0,
        help="Additional value-head calibration loss on real SOAR contexts during imagination.",
    )
    parser.add_argument("--prior-loss-weight", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--split-mode",
        choices=["episode", "episode_task", "episode_task_outcome", "anchor"],
        default="episode_task",
    )
    parser.add_argument(
        "--dynamics-bc-metric",
        choices=[
            "loss",
            "dynamics_mse",
            "delta_mse",
            "contrastive_margin_loss",
            "autoregressive_rollout_mse",
            "autoregressive_rollout_contrastive_margin_loss",
        ],
        default="loss",
        help="Validation metric used for dynamics checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--dynamics-bc-early-stop-patience",
        type=int,
        default=0,
        help="Stop dynamics BC after this many non-improving validation epochs. 0 disables early stopping.",
    )
    parser.add_argument(
        "--dynamics-bc-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation improvement for dynamics BC early stopping/checkpoint replacement.",
    )
    parser.add_argument(
        "--agent-bc-metric",
        choices=["loss", "action_mse", "reward_mse", "value_mse"],
        default="loss",
        help="Validation metric used for frozen agent-BC checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--agent-bc-early-stop-patience",
        type=int,
        default=0,
        help="Stop frozen agent BC after this many non-improving validation epochs. 0 disables early stopping.",
    )
    parser.add_argument(
        "--agent-bc-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation improvement for frozen agent-BC early stopping.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-agent-token-isolation", action="store_true")
    parser.add_argument("--make-smoke-cache", default="", help="Create a synthetic action-causal cache and exit.")
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
    if args.mtp_horizon + 1 < args.prediction_horizon:
        raise SystemExit("--mtp-horizon + 1 must be >= --prediction-horizon for imagination action plans.")

    config = SoarDreamerLiteConfig(
        sequence_cache=str(resolve_path(args.sequence_cache)),
        out_dir=str(resolve_path(args.out_dir)),
        stage=args.stage,
        checkpoint=str(resolve_path(args.checkpoint)) if args.checkpoint else "",
        context_len=args.context_len,
        prediction_horizon=args.prediction_horizon,
        mtp_horizon=args.mtp_horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_hidden_dim=args.head_hidden_dim,
        head_layers=args.head_layers,
        dynamics_architecture=args.dynamics_architecture,
        dynamics_residual_mode=args.dynamics_residual_mode,
        dropout=args.dropout,
        predict_delta=not args.no_predict_delta,
        future_action_offset=args.future_action_offset,
        future_action_window=args.future_action_window,
        future_action_reduce=args.future_action_reduce,
        motion_filter_quantile=args.motion_filter_quantile,
        min_motion_norm=args.min_motion_norm,
        delta_loss_weight=args.delta_loss_weight,
        contrastive_loss_weight=args.contrastive_loss_weight,
        contrastive_margin=args.contrastive_margin,
        dynamics_rollout_loss_weight=args.dynamics_rollout_loss_weight,
        dynamics_rollout_contrastive_loss_weight=args.dynamics_rollout_contrastive_loss_weight,
        dynamics_rollout_contrastive_margin=args.dynamics_rollout_contrastive_margin,
        dynamics_rollout_horizon=args.dynamics_rollout_horizon,
        bc_epochs=args.bc_epochs,
        imagination_epochs=args.imagination_epochs,
        imagination_horizon=args.imagination_horizon,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        imagination_learning_rate=args.imagination_learning_rate,
        weight_decay=args.weight_decay,
        dynamics_loss_weight=args.dynamics_loss_weight,
        action_loss_weight=args.action_loss_weight,
        reward_loss_weight=args.reward_loss_weight,
        value_loss_weight=args.value_loss_weight,
        reward_target_mode=args.reward_target_mode,
        reward_loss_type=args.reward_loss_type,
        value_target_mode=args.value_target_mode,
        value_loss_type=args.value_loss_type,
        value_huber_beta=args.value_huber_beta,
        imagination_value_loss_weight=args.imagination_value_loss_weight,
        imagination_train_value_head=not args.imagination_freeze_value_head,
        real_value_replay_loss_weight=args.real_value_replay_loss_weight,
        prior_loss_weight=args.prior_loss_weight,
        gamma=args.gamma,
        val_ratio=args.val_ratio,
        split_mode=args.split_mode,
        dynamics_bc_metric=args.dynamics_bc_metric,
        dynamics_bc_early_stop_patience=args.dynamics_bc_early_stop_patience,
        dynamics_bc_min_delta=args.dynamics_bc_min_delta,
        agent_bc_metric=args.agent_bc_metric,
        agent_bc_early_stop_patience=args.agent_bc_early_stop_patience,
        agent_bc_min_delta=args.agent_bc_min_delta,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        agent_token_isolation=not args.no_agent_token_isolation,
    )
    if config.stage == "inspect":
        summary = inspect(config)
    else:
        summary = train(config)
    print(json.dumps(compact_summary(summary), indent=2))
    return 0


def inspect(config: SoarDreamerLiteConfig) -> dict[str, Any]:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_sequence_cache(config.sequence_cache)
    anchors = build_dreamer_anchors(
        cache,
        context_len=config.context_len,
        prediction_horizon=config.prediction_horizon,
        mtp_horizon=config.mtp_horizon,
        future_action_offset=config.future_action_offset,
        future_action_window=config.future_action_window,
        dynamics_rollout_horizon=config.dynamics_rollout_horizon,
    )
    anchors, motion_filter = filter_motion_anchors(cache, anchors, config)
    train_anchors, val_anchors = split_for_config(cache, anchors, config)
    summary = {
        "phase": "soar_dreamer_lite_inspection",
        "config": asdict(config),
        "cache": cache_summary(cache, anchors),
        "anchors": {
            "total": int(len(anchors)),
            "train": int(len(train_anchors)),
            "val": int(len(val_anchors)),
        },
        "motion_filter": motion_filter,
        "schema": schema_summary(),
        "ready": bool(len(train_anchors) > 0),
    }
    write_json(out_dir / "inspection_summary.json", summary)
    write_report(summary, out_dir / "inspection_report.md")
    return summary


def train(config: SoarDreamerLiteConfig) -> dict[str, Any]:
    if torch is None or nn is None or F is None or DataLoader is None:
        raise RuntimeError("torch is required for SOAR Dreamer-lite training.")
    seed_everything(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", asdict(config))

    cache = load_sequence_cache(config.sequence_cache)
    anchors = build_dreamer_anchors(
        cache,
        context_len=config.context_len,
        prediction_horizon=config.prediction_horizon,
        mtp_horizon=config.mtp_horizon,
        future_action_offset=config.future_action_offset,
        future_action_window=config.future_action_window,
        dynamics_rollout_horizon=config.dynamics_rollout_horizon,
    )
    anchors, motion_filter = filter_motion_anchors(cache, anchors, config)
    if len(anchors) <= 0:
        raise RuntimeError("No valid anchors. Check context/horizon lengths and episode/step metadata.")
    train_anchors, val_anchors = split_for_config(cache, anchors, config)
    if len(train_anchors) <= 0:
        raise RuntimeError("No train anchors after split.")
    normalizer = compute_normalizer(cache)
    train_ds = SoarDreamerLiteDataset(cache, train_anchors, normalizer, config)
    val_ds = SoarDreamerLiteDataset(cache, val_anchors, normalizer, config) if len(val_anchors) else None
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if val_ds is not None
        else None
    )
    device = resolve_device(config.device)
    model = SoarDreamerLiteModel(config, cache).to(device)
    if config.checkpoint:
        load_checkpoint(model, Path(config.checkpoint), device)

    started = time.time()
    summary: dict[str, Any] = {
        "phase": "soar_dreamer_lite",
        "config": asdict(config),
        "cache": cache_summary(cache, anchors),
        "anchors": {
            "total": int(len(anchors)),
            "train": int(len(train_anchors)),
            "val": int(len(val_anchors)),
        },
        "motion_filter": motion_filter,
        "normalizer": normalizer.to_dict(),
        "schema": schema_summary(),
        "bc_history": [],
        "agent_bc_history": [],
        "agent_bc_best": {},
        "agent_bc_early_stop": {},
        "agent_calibration_eval": {},
        "imagination_history": [],
        "dynamics_control_eval": {},
        "dynamics_control_eval_before_agent": {},
        "dynamics_control_eval_after_agent": {},
    }

    if config.stage in {"agent_bc", "agent_bc_imagination"}:
        if not config.checkpoint:
            raise RuntimeError("--checkpoint is required for frozen-dynamics agent stages.")
        summary["dynamics_control_eval_before_agent"] = evaluate_dynamics_controls(
            model=model,
            loader=val_loader or train_loader,
            device=device,
        )
        summary["agent_bc_history"] = train_agent_bc(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            out_dir=out_dir,
            normalizer=normalizer,
        )
        summary["agent_bc_best"] = best_history_row(summary["agent_bc_history"], config.agent_bc_metric)
        summary["agent_bc_early_stop"] = {
            "enabled": bool(config.agent_bc_early_stop_patience > 0 and val_loader is not None),
            "patience": int(config.agent_bc_early_stop_patience),
            "min_delta": float(config.agent_bc_min_delta),
            "metric": config.agent_bc_metric,
            "triggered": bool(any(row.get("early_stop_triggered") for row in summary["agent_bc_history"])),
            "epochs_run": int(len(summary["agent_bc_history"])),
        }
        summary["agent_calibration_eval"] = evaluate_agent_calibration(
            model=model,
            loader=val_loader or train_loader,
            config=config,
            device=device,
        )
        summary["dynamics_control_eval_after_agent"] = evaluate_dynamics_controls(
            model=model,
            loader=val_loader or train_loader,
            device=device,
        )
        summary["dynamics_control_eval"] = summary["dynamics_control_eval_after_agent"]

    if config.stage in {"dynamics_bc", "all"} and config.bc_epochs > 0:
        summary["bc_history"] = train_dynamics_and_bc(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            out_dir=out_dir,
            normalizer=normalizer,
        )
        best_dynamics_checkpoint = out_dir / "best_dynamics_bc.pt"
        if best_dynamics_checkpoint.exists():
            load_checkpoint(model, best_dynamics_checkpoint, device)
        summary["dynamics_control_eval"] = evaluate_dynamics_controls(
            model=model,
            loader=val_loader or train_loader,
            device=device,
        )

    if config.stage in {"imagination", "agent_bc_imagination", "all"} and config.imagination_epochs > 0:
        if config.stage == "imagination" and not config.checkpoint:
            candidate = out_dir / "best_dynamics_bc.pt"
            if candidate.exists():
                load_checkpoint(model, candidate, device)
        if not summary["dynamics_control_eval"]:
            summary["dynamics_control_eval"] = evaluate_dynamics_controls(
                model=model,
                loader=val_loader or train_loader,
                device=device,
            )
        summary["imagination_history"] = train_in_imagination(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            out_dir=out_dir,
            normalizer=normalizer,
        )

    summary["elapsed_s"] = float(time.time() - started)
    summary["checkpoint_last"] = str(out_dir / "last.pt")
    save_checkpoint(out_dir / "last.pt", model, config, normalizer, summary)
    write_json(out_dir / "summary.json", summary)
    write_report(summary, out_dir / "report.md")
    return summary


if nn is not None:

    class SoarDreamerLiteModel(nn.Module):
        def __init__(self, config: SoarDreamerLiteConfig, cache: SequenceCache):
            super().__init__()
            self.dynamics = ActionConditionedLatentDynamics(
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
                head_hidden_dim=config.head_hidden_dim or None,
                head_layers=config.head_layers,
                architecture=config.dynamics_architecture,
                residual_mode=config.dynamics_residual_mode,
            )
            self.agent = BehaviorCloningMidtrainingHead(
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
                head_hidden_dim=config.head_hidden_dim or None,
                head_layers=config.head_layers,
            )


class SoarDreamerLiteDataset(Dataset):
    def __init__(
        self,
        cache: SequenceCache,
        anchors: np.ndarray,
        normalizer: Normalizer,
        config: SoarDreamerLiteConfig,
    ):
        if torch is None:
            raise RuntimeError("torch is required for SoarDreamerLiteDataset.")
        self.cache = cache
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.context_len = int(config.context_len)
        self.prediction_horizon = int(config.prediction_horizon)
        self.mtp_horizon = int(config.mtp_horizon)
        self.future_action_offset = int(config.future_action_offset)
        self.future_action_window = max(1, int(config.future_action_window))
        self.future_action_reduce = config.future_action_reduce
        self.dynamics_rollout_horizon = int(config.dynamics_rollout_horizon)
        self.z = torch.from_numpy(normalize_np(cache.z, normalizer.z_mean, normalizer.z_std))
        self.action = torch.from_numpy(normalize_np(cache.action, normalizer.action_mean, normalizer.action_std))
        reward = ((cache.reward.astype(np.float32) - normalizer.reward_mean) / normalizer.reward_std).astype(np.float32)
        self.reward = torch.from_numpy(reward)
        self.reward_raw = torch.from_numpy(cache.reward.astype(np.float32))
        self.done = torch.from_numpy(cache.done.astype(bool))
        self.task_id = torch.from_numpy(cache.task_id.astype(np.int64))

    def __len__(self) -> int:
        return int(self.anchors.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        anchor = int(self.anchors[index])
        z_context_start = anchor - self.context_len + 1
        target_z_start = anchor + 1
        target_z_end = anchor + self.prediction_horizon + 1
        future_action_start = anchor + self.future_action_offset
        rollout_horizon = max(0, self.dynamics_rollout_horizon)
        rollout_target_end = anchor + rollout_horizon + 1
        target_mtp_end = anchor + self.mtp_horizon + 1
        item = {
            "z_context": self.z[z_context_start : anchor + 1],
            "action_context": self.action[z_context_start : anchor + 1],
            "future_action": self._future_action_sequence(future_action_start, self.prediction_horizon),
            "target_z": self.z[target_z_start:target_z_end],
            "target_action": self.action[anchor:target_mtp_end],
            "target_reward": self.reward[anchor:target_mtp_end],
            "target_reward_raw": self.reward_raw[anchor:target_mtp_end],
            "target_done": self.done[anchor:target_mtp_end],
            "task_id": self.task_id[anchor],
        }
        if rollout_horizon > 0:
            item["rollout_future_action"] = self._future_action_sequence(
                future_action_start,
                rollout_horizon + self.prediction_horizon - 1,
            )
            item["rollout_target_z"] = self.z[target_z_start:rollout_target_end]
        return item

    def _future_action_sequence(self, start: int, horizon: int) -> torch.Tensor:
        actions = []
        for offset in range(int(horizon)):
            window = self.action[start + offset : start + offset + self.future_action_window]
            if self.future_action_reduce == "sum":
                actions.append(window.sum(dim=0))
            elif self.future_action_reduce == "first":
                actions.append(window[0])
            else:
                actions.append(window.mean(dim=0))
        return torch.stack(actions, dim=0)


def train_dynamics_and_bc(
    *,
    model: Any,
    train_loader: Any,
    val_loader: Any,
    config: SoarDreamerLiteConfig,
    device: Any,
    out_dir: Path,
    normalizer: Normalizer,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    stale_epochs = 0
    for epoch in range(1, config.bc_epochs + 1):
        train_metrics = run_dynamics_bc_epoch(model, train_loader, config, device, optimizer=optimizer)
        val_metrics = run_dynamics_bc_epoch(model, val_loader, config, device, optimizer=None) if val_loader else {}
        metric_name = config.dynamics_bc_metric
        metric_source = val_metrics if val_metrics else train_metrics
        metric = float(metric_source.get(metric_name, metric_source.get("loss", train_metrics["loss"])))
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "selection_metric": metric_name,
            "selection_metric_value": metric,
            "best_selection_metric": best_metric if best_metric < float("inf") else None,
        }
        history.append(row)
        emit_progress("dynamics_bc", row)
        if metric < best_metric - float(config.dynamics_bc_min_delta):
            best_metric = metric
            stale_epochs = 0
            save_checkpoint(out_dir / "best_dynamics_bc.pt", model, config, normalizer, {"bc_epoch": row})
        else:
            stale_epochs += 1
        row["stale_epochs"] = stale_epochs
        row["best_selection_metric"] = best_metric
        if (
            val_metrics
            and config.dynamics_bc_early_stop_patience > 0
            and stale_epochs >= config.dynamics_bc_early_stop_patience
        ):
            break
    return history


def train_agent_bc(
    *,
    model: Any,
    train_loader: Any,
    val_loader: Any,
    config: SoarDreamerLiteConfig,
    device: Any,
    out_dir: Path,
    normalizer: Normalizer,
) -> list[dict[str, Any]]:
    for param in model.dynamics.parameters():
        param.requires_grad_(False)
    for param in model.agent.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.agent.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    stale_epochs = 0
    for epoch in range(1, config.bc_epochs + 1):
        train_metrics = run_agent_bc_epoch(model, train_loader, config, device, optimizer=optimizer)
        val_metrics = run_agent_bc_epoch(model, val_loader, config, device, optimizer=None) if val_loader else {}
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        metric = agent_bc_selection_metric(row, config)
        row["selection_metric"] = metric
        row["selection_metric_name"] = config.agent_bc_metric
        improved = metric < (best_metric - float(config.agent_bc_min_delta))
        if improved:
            best_metric = metric
            stale_epochs = 0
            row["best"] = True
            save_checkpoint(out_dir / "best_agent_bc.pt", model, config, normalizer, {"agent_bc_epoch": row})
        else:
            stale_epochs += 1
            row["best"] = False
        row["stale_epochs"] = stale_epochs
        if (
            val_loader is not None
            and config.agent_bc_early_stop_patience > 0
            and stale_epochs >= config.agent_bc_early_stop_patience
        ):
            row["early_stop_triggered"] = True
            history.append(row)
            emit_progress("agent_bc", row)
            break
        row["early_stop_triggered"] = False
        history.append(row)
        emit_progress("agent_bc", row)
    best_agent_checkpoint = out_dir / "best_agent_bc.pt"
    if best_agent_checkpoint.exists():
        load_checkpoint(model, best_agent_checkpoint, device)
    return history


def agent_bc_selection_metric(row: dict[str, Any], config: SoarDreamerLiteConfig) -> float:
    metrics = row.get("val") or row.get("train") or {}
    fallback = row.get("train") or {}
    metric_name = config.agent_bc_metric
    if metric_name in metrics:
        return float(metrics[metric_name])
    if "loss" in metrics:
        return float(metrics["loss"])
    if metric_name in fallback:
        return float(fallback[metric_name])
    return float(fallback.get("loss", float("inf")))


def best_history_row(history: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    if not history:
        return {}

    def metric(row: dict[str, Any]) -> float:
        if "selection_metric" in row:
            return float(row["selection_metric"])
        metrics = row.get("val") or row.get("train") or {}
        return float(metrics.get(metric_name, metrics.get("loss", float("inf"))))

    return min(history, key=metric)


def evaluate_dynamics_controls(*, model: Any, loader: Any, device: Any) -> dict[str, Any]:
    model.eval()
    totals = {
        "normal_mse": 0.0,
        "shuffle_future_actions_mse": 0.0,
        "zero_future_actions_mse": 0.0,
        "persistence_mse": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            batch_size = int(batch["z_context"].shape[0])
            normal = model.dynamics(
                batch["z_context"],
                batch["action_context"],
                batch["future_action"],
                batch["task_id"],
            )["predicted_z"]
            if batch_size > 1:
                perm = torch.randperm(batch_size, device=batch["future_action"].device)
                shuffled_future_action = batch["future_action"][perm]
            else:
                shuffled_future_action = torch.roll(batch["future_action"], shifts=1, dims=1)
            shuffled = model.dynamics(
                batch["z_context"],
                batch["action_context"],
                shuffled_future_action,
                batch["task_id"],
            )["predicted_z"]
            zeroed = model.dynamics(
                batch["z_context"],
                batch["action_context"],
                torch.zeros_like(batch["future_action"]),
                batch["task_id"],
            )["predicted_z"]
            persistence = batch["z_context"][:, -1:, :].expand_as(batch["target_z"])
            target = batch["target_z"]
            totals["normal_mse"] += float(F.mse_loss(normal, target).detach().cpu()) * batch_size
            totals["shuffle_future_actions_mse"] += float(F.mse_loss(shuffled, target).detach().cpu()) * batch_size
            totals["zero_future_actions_mse"] += float(F.mse_loss(zeroed, target).detach().cpu()) * batch_size
            totals["persistence_mse"] += float(F.mse_loss(persistence, target).detach().cpu()) * batch_size
            count += batch_size
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    normal = max(metrics["normal_mse"], 1e-12)
    persistence = max(metrics["persistence_mse"], 1e-12)
    metrics["normal_over_persistence"] = metrics["normal_mse"] / persistence
    metrics["shuffle_over_normal"] = metrics["shuffle_future_actions_mse"] / normal
    metrics["zero_over_normal"] = metrics["zero_future_actions_mse"] / normal
    metrics["action_conditioning_detected"] = bool(
        metrics["normal_mse"] < metrics["persistence_mse"]
        and metrics["shuffle_future_actions_mse"] > metrics["normal_mse"]
        and metrics["zero_future_actions_mse"] > metrics["normal_mse"]
    )
    metrics["strict_gate_passed"] = bool(
        metrics["normal_over_persistence"] <= 0.95
        and metrics["shuffle_over_normal"] >= 1.05
        and metrics["zero_over_normal"] >= 1.05
    )
    if metrics["strict_gate_passed"]:
        metrics["action_conditioning_strength"] = "strong"
    elif metrics["action_conditioning_detected"]:
        metrics["action_conditioning_strength"] = "weak"
    else:
        metrics["action_conditioning_strength"] = "none"
    return metrics


def evaluate_agent_calibration(
    *,
    model: Any,
    loader: Any,
    config: SoarDreamerLiteConfig,
    device: Any,
) -> dict[str, Any]:
    model.eval()
    rewards_pred: list[Any] = []
    rewards_target: list[Any] = []
    values_pred: list[Any] = []
    values_target: list[Any] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model.agent(batch["z_context"], batch["action_context"], batch["task_id"])
            reward_target = reward_targets(batch, config)
            reward_pred = reward_predictions_for_metrics(out["reward_pred"], config)
            value_target = value_targets(batch, config)
            rewards_pred.append(reward_pred.reshape(-1).detach().cpu())
            rewards_target.append(reward_target.reshape(-1).detach().cpu())
            values_pred.append(out["value"].reshape(-1).detach().cpu())
            values_target.append(value_target.reshape(-1).detach().cpu())
    if not rewards_pred:
        return {}
    reward_pred_t = torch.cat(rewards_pred)
    reward_target_t = torch.cat(rewards_target)
    value_pred_t = torch.cat(values_pred)
    value_target_t = torch.cat(values_target)
    calibration = {
        "reward_target_mode": config.reward_target_mode,
        "reward_loss_type": config.reward_loss_type,
        "value_target_mode": config.value_target_mode,
        "value_loss_type": config.value_loss_type,
        "reward_brier": float(F.mse_loss(reward_pred_t, reward_target_t).item()),
        "reward_mae": float(F.l1_loss(reward_pred_t, reward_target_t).item()),
        "reward_accuracy": float(((reward_pred_t >= 0.5) == (reward_target_t >= 0.5)).float().mean().item())
        if float(reward_target_t.min().item()) >= 0.0 and float(reward_target_t.max().item()) <= 1.0
        else 0.0,
        "reward_ece_10": binary_ece(reward_pred_t, reward_target_t, bins=10),
        "reward_target_mean": float(reward_target_t.mean().item()),
        "reward_pred_mean": float(reward_pred_t.mean().item()),
        "reward_target_positive_fraction": float((reward_target_t >= 0.5).float().mean().item())
        if float(reward_target_t.min().item()) >= 0.0 and float(reward_target_t.max().item()) <= 1.0
        else 0.0,
        "value_mse": float(F.mse_loss(value_pred_t, value_target_t).item()),
        "value_mae": float(F.l1_loss(value_pred_t, value_target_t).item()),
        "value_target_mean": float(value_target_t.mean().item()),
        "value_pred_mean": float(value_pred_t.mean().item()),
        "value_corr": pearson_corr(value_pred_t, value_target_t),
    }
    return calibration


def binary_ece(prob: Any, target: Any, *, bins: int) -> float:
    if float(target.min().item()) < 0.0 or float(target.max().item()) > 1.0:
        return 0.0
    prob = prob.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    edges = torch.linspace(0.0, 1.0, int(bins) + 1)
    total = max(int(prob.numel()), 1)
    error = prob.new_tensor(0.0)
    for index in range(int(bins)):
        lo = edges[index]
        hi = edges[index + 1]
        if index == int(bins) - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not bool(mask.any()):
            continue
        confidence = prob[mask].mean()
        accuracy = target[mask].mean()
        error = error + (mask.float().mean() * torch.abs(confidence - accuracy))
    return float(error.item())


def pearson_corr(left: Any, right: Any) -> float:
    left = left.float()
    right = right.float()
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(right_centered)
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((left_centered * right_centered).sum().div(denom).item())


def run_dynamics_bc_epoch(
    model: Any,
    loader: Any,
    config: SoarDreamerLiteConfig,
    device: Any,
    *,
    optimizer: Any | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            losses = dynamics_bc_losses(model, batch, config)
            loss = losses["loss"]
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_size = int(batch["z_context"].shape[0])
        count += batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def run_agent_bc_epoch(
    model: Any,
    loader: Any,
    config: SoarDreamerLiteConfig,
    device: Any,
    *,
    optimizer: Any | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.dynamics.eval()
    model.agent.train(training)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            losses = agent_bc_losses(model, batch, config)
            loss = losses["loss"]
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.agent.parameters(), 1.0)
                optimizer.step()
        batch_size = int(batch["z_context"].shape[0])
        count += batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def agent_bc_losses(model: Any, batch: dict[str, Any], config: SoarDreamerLiteConfig) -> dict[str, Any]:
    agent_out = model.agent(batch["z_context"], batch["action_context"], batch["task_id"])
    action_loss = F.mse_loss(agent_out["action_pred"], batch["target_action"])
    reward_target = reward_targets(batch, config)
    reward_loss = reward_prediction_loss(agent_out["reward_pred"], reward_target, config)
    value_target = value_targets(batch, config)
    value_loss = value_prediction_loss(agent_out["value"], value_target, config)
    reward_pred_metric = reward_predictions_for_metrics(agent_out["reward_pred"], config)
    reward_mse = F.mse_loss(reward_pred_metric, reward_target)
    reward_mae = F.l1_loss(reward_pred_metric, reward_target)
    reward_accuracy = binary_accuracy(reward_pred_metric, reward_target)
    value_mse = F.mse_loss(agent_out["value"], value_target)
    value_mae = F.l1_loss(agent_out["value"], value_target)
    loss = (
        config.action_loss_weight * action_loss
        + config.reward_loss_weight * reward_loss
        + config.value_loss_weight * value_loss
    )
    return {
        "loss": loss,
        "action_mse": action_loss,
        "reward_loss": reward_loss,
        "reward_mse": reward_mse,
        "reward_mae": reward_mae,
        "reward_accuracy": reward_accuracy,
        "reward_target_mean": reward_target.mean(),
        "reward_pred_mean": reward_pred_metric.mean(),
        "value_loss": value_loss,
        "value_mse": value_mse,
        "value_mae": value_mae,
        "value_target_mean": value_target.mean(),
        "value_pred_mean": agent_out["value"].mean(),
    }


def reward_targets(batch: dict[str, Any], config: SoarDreamerLiteConfig) -> Any:
    if config.reward_target_mode == "raw":
        return batch["target_reward_raw"].float()
    if config.reward_target_mode == "normalized":
        return batch["target_reward"].float()
    raise ValueError(f"Unsupported reward_target_mode: {config.reward_target_mode}")


def reward_prediction_loss(prediction: Any, target: Any, config: SoarDreamerLiteConfig) -> Any:
    if config.reward_loss_type == "bce":
        return F.binary_cross_entropy_with_logits(prediction, target.clamp(0.0, 1.0))
    if config.reward_loss_type == "mse":
        return F.mse_loss(prediction, target)
    raise ValueError(f"Unsupported reward_loss_type: {config.reward_loss_type}")


def reward_predictions_for_metrics(prediction: Any, config: SoarDreamerLiteConfig) -> Any:
    if config.reward_loss_type == "bce":
        return torch.sigmoid(prediction)
    return prediction


def reward_predictions_for_imagination(prediction: Any, config: SoarDreamerLiteConfig) -> Any:
    if config.reward_loss_type == "bce":
        return torch.sigmoid(prediction)
    return prediction


def value_targets(batch: dict[str, Any], config: SoarDreamerLiteConfig) -> Any:
    if config.value_target_mode == "normalized_discounted_sum":
        return discounted_first_return(batch["target_reward"], batch["target_done"], gamma=config.gamma)
    if config.value_target_mode == "raw_discounted_sum":
        return discounted_first_return(batch["target_reward_raw"], batch["target_done"], gamma=config.gamma)
    if config.value_target_mode == "raw_discounted_mean":
        total = discounted_first_return(batch["target_reward_raw"], batch["target_done"], gamma=config.gamma)
        weights = discounted_horizon_weights(
            batch["target_reward_raw"].shape[1],
            done=batch["target_done"],
            gamma=config.gamma,
            device=batch["target_reward_raw"].device,
        )
        return total / weights.clamp_min(1e-6)
    raise ValueError(f"Unsupported value_target_mode: {config.value_target_mode}")


def value_prediction_loss(prediction: Any, target: Any, config: SoarDreamerLiteConfig) -> Any:
    if config.value_loss_type == "huber":
        return F.smooth_l1_loss(prediction, target, beta=float(config.value_huber_beta))
    if config.value_loss_type == "mse":
        return F.mse_loss(prediction, target)
    raise ValueError(f"Unsupported value_loss_type: {config.value_loss_type}")


def binary_accuracy(prediction: Any, target: Any) -> Any:
    if float(target.detach().min().cpu()) < 0.0 or float(target.detach().max().cpu()) > 1.0:
        return prediction.new_tensor(0.0)
    return ((prediction >= 0.5) == (target >= 0.5)).float().mean()


def dynamics_bc_losses(model: Any, batch: dict[str, Any], config: SoarDreamerLiteConfig) -> dict[str, Any]:
    dynamics_out = model.dynamics(
        batch["z_context"],
        batch["action_context"],
        batch["future_action"],
        batch["task_id"],
    )
    agent_out = model.agent(batch["z_context"], batch["action_context"], batch["task_id"])
    predicted_z = dynamics_out["predicted_z"]
    target_z = batch["target_z"]
    true_mse_per_sample = F.mse_loss(predicted_z, target_z, reduction="none").mean(dim=(1, 2))
    dyn_loss = true_mse_per_sample.mean()
    last_z = batch["z_context"][:, -1:, :]
    delta_loss = F.mse_loss(predicted_z - last_z, target_z - last_z)
    rollout_loss = predicted_z.new_tensor(0.0)
    rollout_contrastive_loss = predicted_z.new_tensor(0.0)
    rollout_contrastive_shuffle_mse = predicted_z.new_tensor(0.0)
    rollout_contrastive_zero_mse = predicted_z.new_tensor(0.0)
    if (
        (config.dynamics_rollout_loss_weight > 0 or config.dynamics_rollout_contrastive_loss_weight > 0)
        and config.dynamics_rollout_horizon > 0
        and "rollout_future_action" in batch
        and "rollout_target_z" in batch
    ):
        rollout_metrics = autoregressive_dynamics_rollout_losses(model, batch, config)
        rollout_loss = rollout_metrics["rollout_mse"]
        rollout_contrastive_loss = rollout_metrics["rollout_contrastive_loss"]
        rollout_contrastive_shuffle_mse = rollout_metrics["rollout_shuffle_mse"]
        rollout_contrastive_zero_mse = rollout_metrics["rollout_zero_mse"]
    contrastive_loss = predicted_z.new_tensor(0.0)
    shuffle_mse = predicted_z.new_tensor(0.0)
    zero_mse = predicted_z.new_tensor(0.0)
    if config.contrastive_loss_weight > 0:
        batch_size = int(batch["future_action"].shape[0])
        if batch_size > 1:
            perm = torch.randperm(batch_size, device=batch["future_action"].device)
            shuffled_future_action = batch["future_action"][perm]
        else:
            shuffled_future_action = torch.roll(batch["future_action"], shifts=1, dims=1)
        shuffled_z = model.dynamics(
            batch["z_context"],
            batch["action_context"],
            shuffled_future_action,
            batch["task_id"],
        )["predicted_z"]
        zeroed_z = model.dynamics(
            batch["z_context"],
            batch["action_context"],
            torch.zeros_like(batch["future_action"]),
            batch["task_id"],
        )["predicted_z"]
        shuffle_mse_per_sample = F.mse_loss(shuffled_z, target_z, reduction="none").mean(dim=(1, 2))
        zero_mse_per_sample = F.mse_loss(zeroed_z, target_z, reduction="none").mean(dim=(1, 2))
        margin = float(config.contrastive_margin)
        contrastive_loss = (
            F.relu(margin + true_mse_per_sample - shuffle_mse_per_sample).mean()
            + F.relu(margin + true_mse_per_sample - zero_mse_per_sample).mean()
        )
        shuffle_mse = shuffle_mse_per_sample.mean()
        zero_mse = zero_mse_per_sample.mean()
    action_loss = F.mse_loss(agent_out["action_pred"], batch["target_action"])
    reward_target = reward_targets(batch, config)
    reward_loss = reward_prediction_loss(agent_out["reward_pred"], reward_target, config)
    value_target = value_targets(batch, config)
    value_loss = value_prediction_loss(agent_out["value"], value_target, config)
    reward_pred_metric = reward_predictions_for_metrics(agent_out["reward_pred"], config)
    reward_mse = F.mse_loss(reward_pred_metric, reward_target)
    value_mse = F.mse_loss(agent_out["value"], value_target)
    loss = (
        config.dynamics_loss_weight * dyn_loss
        + config.delta_loss_weight * delta_loss
        + config.contrastive_loss_weight * contrastive_loss
        + config.dynamics_rollout_loss_weight * rollout_loss
        + config.dynamics_rollout_contrastive_loss_weight * rollout_contrastive_loss
        + config.action_loss_weight * action_loss
        + config.reward_loss_weight * reward_loss
        + config.value_loss_weight * value_loss
    )
    return {
        "loss": loss,
        "dynamics_mse": dyn_loss,
        "delta_mse": delta_loss,
        "autoregressive_rollout_mse": rollout_loss,
        "autoregressive_rollout_contrastive_margin_loss": rollout_contrastive_loss,
        "autoregressive_rollout_contrastive_shuffle_mse": rollout_contrastive_shuffle_mse,
        "autoregressive_rollout_contrastive_zero_mse": rollout_contrastive_zero_mse,
        "contrastive_margin_loss": contrastive_loss,
        "contrastive_shuffle_mse": shuffle_mse,
        "contrastive_zero_mse": zero_mse,
        "target_delta_norm": torch.linalg.vector_norm(target_z - last_z, dim=-1).mean(),
        "action_mse": action_loss,
        "reward_loss": reward_loss,
        "reward_mse": reward_mse,
        "reward_accuracy": binary_accuracy(reward_pred_metric, reward_target),
        "value_loss": value_loss,
        "value_mse": value_mse,
        "value_mae": F.l1_loss(agent_out["value"], value_target),
    }


def autoregressive_dynamics_rollout_losses(
    model: Any,
    batch: dict[str, Any],
    config: SoarDreamerLiteConfig,
) -> dict[str, Any]:
    z_context = batch["z_context"]
    action_context = batch["action_context"]
    future_actions = batch["rollout_future_action"]
    target_z = batch["rollout_target_z"]
    task_id = batch["task_id"]
    horizon = min(int(config.dynamics_rollout_horizon), int(target_z.shape[1]))
    target = target_z[:, :horizon, :]
    predicted = autoregressive_dynamics_rollout_prediction(
        model,
        z_context,
        action_context,
        future_actions,
        task_id,
        prediction_horizon=config.prediction_horizon,
        rollout_horizon=horizon,
    )
    rollout_mse_per_sample = F.mse_loss(predicted, target, reduction="none").mean(dim=(1, 2))
    rollout_mse = rollout_mse_per_sample.mean()
    batch_size = int(future_actions.shape[0])
    if batch_size > 1:
        perm = torch.randperm(batch_size, device=future_actions.device)
        shuffled_actions = future_actions[perm]
    else:
        shuffled_actions = torch.roll(future_actions, shifts=1, dims=1)
    shuffled = autoregressive_dynamics_rollout_prediction(
        model,
        z_context,
        action_context,
        shuffled_actions,
        task_id,
        prediction_horizon=config.prediction_horizon,
        rollout_horizon=horizon,
    )
    zeroed = autoregressive_dynamics_rollout_prediction(
        model,
        z_context,
        action_context,
        torch.zeros_like(future_actions),
        task_id,
        prediction_horizon=config.prediction_horizon,
        rollout_horizon=horizon,
    )
    shuffle_mse_per_sample = F.mse_loss(shuffled, target, reduction="none").mean(dim=(1, 2))
    zero_mse_per_sample = F.mse_loss(zeroed, target, reduction="none").mean(dim=(1, 2))
    margin = float(config.dynamics_rollout_contrastive_margin)
    contrastive = (
        F.relu(margin + rollout_mse_per_sample - shuffle_mse_per_sample).mean()
        + F.relu(margin + rollout_mse_per_sample - zero_mse_per_sample).mean()
    )
    return {
        "rollout_mse": rollout_mse,
        "rollout_contrastive_loss": contrastive,
        "rollout_shuffle_mse": shuffle_mse_per_sample.mean(),
        "rollout_zero_mse": zero_mse_per_sample.mean(),
    }


def autoregressive_dynamics_rollout_prediction(
    model: Any,
    z_context: Any,
    action_context: Any,
    future_actions: Any,
    task_id: Any,
    *,
    prediction_horizon: int,
    rollout_horizon: int,
) -> Any:
    predictions: list[Any] = []
    for step in range(int(rollout_horizon)):
        action_plan = future_actions[:, step : step + int(prediction_horizon), :]
        dynamics_out = model.dynamics(z_context, action_context, action_plan, task_id)
        next_z = dynamics_out["predicted_z"][:, 0, :]
        first_action = action_plan[:, 0, :]
        predictions.append(next_z)
        z_context = torch.cat([z_context[:, 1:, :], next_z[:, None, :]], dim=1)
        action_context = torch.cat([action_context[:, 1:, :], first_action[:, None, :]], dim=1)
    return torch.stack(predictions, dim=1)


def train_in_imagination(
    *,
    model: Any,
    train_loader: Any,
    val_loader: Any,
    config: SoarDreamerLiteConfig,
    device: Any,
    out_dir: Path,
    normalizer: Normalizer,
) -> list[dict[str, Any]]:
    prior_agent = copy.deepcopy(model.agent).to(device).eval()
    reward_agent = copy.deepcopy(model.agent).to(device).eval()
    for module in [model.dynamics, prior_agent, reward_agent, model.agent]:
        for param in module.parameters():
            param.requires_grad_(False)
    for head in model.agent.action_heads:
        for param in head.parameters():
            param.requires_grad_(True)
    if config.imagination_train_value_head:
        for param in model.agent.value_head.parameters():
            param.requires_grad_(True)

    params = [param for param in model.agent.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable imagination parameters. Check action/value head freezing config.")
    optimizer = torch.optim.AdamW(
        params,
        lr=config.imagination_learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    for epoch in range(1, config.imagination_epochs + 1):
        train_metrics = run_imagination_epoch(
            model,
            prior_agent,
            reward_agent,
            train_loader,
            config,
            device,
            optimizer=optimizer,
        )
        val_metrics = (
            run_imagination_epoch(
                model,
                prior_agent,
                reward_agent,
                val_loader,
                config,
                device,
                optimizer=None,
            )
            if val_loader
            else {}
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        emit_progress("imagination", row)
        metric = float(val_metrics.get("loss", train_metrics["loss"]))
        if metric < best_metric:
            best_metric = metric
            save_checkpoint(out_dir / "best_imagination.pt", model, config, normalizer, {"imagination_epoch": row})
    return history


def run_imagination_epoch(
    model: Any,
    prior_agent: Any,
    reward_agent: Any,
    loader: Any,
    config: SoarDreamerLiteConfig,
    device: Any,
    *,
    optimizer: Any | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.agent.train(training)
    model.dynamics.eval()
    prior_agent.eval()
    reward_agent.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            losses = imagination_losses(model, prior_agent, reward_agent, batch, config)
            loss = losses["loss"]
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.agent.parameters() if p.requires_grad], 1.0)
                optimizer.step()
        batch_size = int(batch["z_context"].shape[0])
        count += batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def imagination_losses(
    model: Any,
    prior_agent: Any,
    reward_agent: Any,
    batch: dict[str, Any],
    config: SoarDreamerLiteConfig,
) -> dict[str, Any]:
    z_context = batch["z_context"]
    action_context = batch["action_context"]
    task_id = batch["task_id"]
    rewards: list[Any] = []
    values: list[Any] = []
    prior_penalties: list[Any] = []

    for _ in range(config.imagination_horizon):
        agent_out = model.agent(z_context, action_context, task_id)
        action_plan = agent_out["action_pred"][:, : config.prediction_horizon, :]
        with torch.no_grad():
            prior_plan = prior_agent(z_context, action_context, task_id)["action_pred"][
                :, : config.prediction_horizon, :
            ]
        dynamics_out = model.dynamics(z_context, action_context, action_plan, task_id)
        first_action = action_plan[:, 0, :]
        next_z = dynamics_out["predicted_z"][:, 0, :]
        next_z_context = torch.cat([z_context[:, 1:, :], next_z[:, None, :]], dim=1)
        next_action_context = torch.cat([action_context[:, 1:, :], first_action[:, None, :]], dim=1)
        reward_pred = reward_agent(next_z_context, next_action_context, task_id)["reward_pred"][:, 0]
        reward = reward_predictions_for_imagination(reward_pred, config)
        rewards.append(reward)
        values.append(agent_out["value"])
        prior_penalties.append(F.mse_loss(action_plan, prior_plan, reduction="none").mean(dim=(1, 2)))
        z_context = next_z_context
        action_context = next_action_context

    final_value = model.agent(z_context, action_context, task_id)["value"]
    returns = lambda_returns_from_rewards(rewards, final_value, gamma=config.gamma)
    values_t = torch.stack(values, dim=1)
    returns_t = torch.stack(returns, dim=1)
    prior_t = torch.stack(prior_penalties, dim=1)
    policy_loss = -returns_t[:, 0].mean()
    value_loss = F.mse_loss(values_t, returns_t.detach())
    prior_loss = prior_t.mean()
    real_agent_out = model.agent(batch["z_context"], batch["action_context"], batch["task_id"])
    real_value_target = value_targets(batch, config)
    real_value_replay_loss = value_prediction_loss(real_agent_out["value"], real_value_target, config)
    value_loss_weight = float(config.imagination_value_loss_weight) if config.imagination_train_value_head else 0.0
    real_value_replay_loss_weight = (
        float(config.real_value_replay_loss_weight) if config.imagination_train_value_head else 0.0
    )
    loss = (
        policy_loss
        + value_loss_weight * value_loss
        + config.prior_loss_weight * prior_loss
        + real_value_replay_loss_weight * real_value_replay_loss
    )
    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_mse": value_loss,
        "value_loss_weight": value_loss.new_tensor(value_loss_weight),
        "real_value_replay_loss": real_value_replay_loss,
        "real_value_replay_loss_weight": real_value_replay_loss.new_tensor(real_value_replay_loss_weight),
        "prior_mse": prior_loss,
        "mean_imagined_reward": torch.stack(rewards, dim=1).mean(),
        "mean_imagined_return0": returns_t[:, 0].mean(),
    }


def discounted_first_return(reward: Any, done: Any, *, gamma: float) -> Any:
    running = torch.zeros_like(reward[:, 0])
    for index in reversed(range(reward.shape[1])):
        running = reward[:, index] + float(gamma) * running * (~done[:, index]).float()
    return running


def discounted_horizon_weights(length: int, *, done: Any, gamma: float, device: Any) -> Any:
    length = int(length)
    weights = torch.zeros(done.shape[0], dtype=torch.float32, device=device)
    running_mask = torch.ones(done.shape[0], dtype=torch.float32, device=device)
    discount = 1.0
    for index in range(length):
        weights = weights + running_mask * discount
        running_mask = running_mask * (~done[:, index]).float()
        discount *= float(gamma)
    return weights


def lambda_returns_from_rewards(rewards: list[Any], final_value: Any, *, gamma: float) -> list[Any]:
    running = final_value
    returns: list[Any] = []
    for reward in reversed(rewards):
        running = reward + float(gamma) * running
        returns.append(running)
    returns.reverse()
    return returns


def build_dreamer_anchors(
    cache: SequenceCache,
    *,
    context_len: int,
    prediction_horizon: int,
    mtp_horizon: int,
    future_action_offset: int = 0,
    future_action_window: int = 1,
    dynamics_rollout_horizon: int = 0,
) -> np.ndarray:
    context_len = int(context_len)
    prediction_horizon = int(prediction_horizon)
    mtp_horizon = int(mtp_horizon)
    future_action_offset = int(future_action_offset)
    future_action_window = max(1, int(future_action_window))
    dynamics_rollout_horizon = int(dynamics_rollout_horizon)
    if context_len < 1:
        raise ValueError("context_len must be >= 1")
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon must be >= 1")
    if mtp_horizon < 0:
        raise ValueError("mtp_horizon must be >= 0")
    anchors: list[int] = []
    n = cache.num_steps
    for anchor in range(context_len - 1, n):
        future_action_start = anchor + future_action_offset
        rollout_action_horizon = max(prediction_horizon, dynamics_rollout_horizon + prediction_horizon - 1)
        future_action_end = future_action_start + rollout_action_horizon + future_action_window - 2
        span_start = min(anchor - context_len + 1, future_action_start)
        span_end = max(anchor + prediction_horizon, anchor + mtp_horizon, anchor + dynamics_rollout_horizon, future_action_end)
        if span_start < 0 or span_end >= n:
            continue
        eps = cache.episode[span_start : span_end + 1]
        steps = cache.step[span_start : span_end + 1]
        if eps.size != (span_end - span_start + 1):
            continue
        if not np.all(eps == eps[0]):
            continue
        if not np.all(np.diff(steps) == 1):
            continue
        anchors.append(anchor)
    return np.asarray(anchors, dtype=np.int64)


def filter_motion_anchors(
    cache: SequenceCache,
    anchors: np.ndarray,
    config: SoarDreamerLiteConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    anchors = np.asarray(anchors, dtype=np.int64)
    quantile = float(config.motion_filter_quantile)
    min_motion_norm = float(config.min_motion_norm)
    metadata: dict[str, Any] = {
        "enabled": bool(quantile > 0 or min_motion_norm > 0),
        "input_anchors": int(anchors.size),
        "kept_anchors": int(anchors.size),
        "motion_filter_quantile": quantile,
        "min_motion_norm": min_motion_norm,
        "threshold": 0.0,
    }
    if anchors.size <= 0 or not metadata["enabled"]:
        return anchors, metadata
    if quantile < 0 or quantile >= 1:
        raise ValueError("--motion-filter-quantile must be in [0, 1).")

    motion = compute_anchor_motion_norms(cache, anchors, prediction_horizon=config.prediction_horizon)
    threshold = min_motion_norm
    if quantile > 0:
        threshold = max(threshold, float(np.quantile(motion, quantile)))
    keep = motion >= threshold
    filtered = anchors[keep]
    metadata.update(
        {
            "kept_anchors": int(filtered.size),
            "threshold": float(threshold),
            "motion_min": float(np.min(motion)) if motion.size else 0.0,
            "motion_mean": float(np.mean(motion)) if motion.size else 0.0,
            "motion_median": float(np.median(motion)) if motion.size else 0.0,
            "motion_max": float(np.max(motion)) if motion.size else 0.0,
        }
    )
    return filtered, metadata


def compute_anchor_motion_norms(
    cache: SequenceCache,
    anchors: np.ndarray,
    *,
    prediction_horizon: int,
) -> np.ndarray:
    values: list[float] = []
    horizon = int(prediction_horizon)
    for anchor in np.asarray(anchors, dtype=np.int64):
        start = int(anchor) + 1
        end = start + horizon
        future = cache.z[start:end]
        reference = cache.z[int(anchor) : int(anchor) + 1]
        if future.shape[0] != horizon:
            values.append(0.0)
            continue
        delta = future - reference
        values.append(float(np.linalg.norm(delta, axis=1).mean()))
    return np.asarray(values, dtype=np.float32)


def split_for_config(
    cache: SequenceCache,
    anchors: np.ndarray,
    config: SoarDreamerLiteConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.split_mode == "anchor":
        return split_anchors(anchors, val_ratio=config.val_ratio, seed=config.seed)
    if config.split_mode == "episode":
        return split_anchors_by_episode(cache, anchors, val_ratio=config.val_ratio, seed=config.seed)
    if config.split_mode == "episode_task":
        return split_anchors_by_task_episode(cache, anchors, val_ratio=config.val_ratio, seed=config.seed)
    if config.split_mode == "episode_task_outcome":
        return split_anchors_by_task_episode_outcome(cache, anchors, val_ratio=config.val_ratio, seed=config.seed)
    raise ValueError(f"Unsupported split_mode: {config.split_mode}")


def split_anchors_by_task_episode_outcome(
    cache: SequenceCache,
    anchors: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hold out episodes while preserving task and binary success/failure outcome.

    Reward/value calibration is very sensitive to success-rate shift. This split
    groups episodes by their dominant task and whether the trajectory ever has a
    positive reward, then holds out episodes within each group when possible.
    """

    anchors = np.asarray(anchors, dtype=np.int64)
    if anchors.size <= 0:
        return anchors, anchors
    rng = np.random.default_rng(seed)
    episode_ids = np.unique(cache.episode[anchors])
    groups: dict[tuple[int, int], list[int]] = {}
    for episode in episode_ids:
        indices = np.flatnonzero(cache.episode == episode)
        if indices.size <= 0:
            continue
        values, counts = np.unique(cache.task_id[indices], return_counts=True)
        task = int(values[np.argmax(counts)])
        outcome = int(np.max(cache.reward[indices]) > 0.0)
        groups.setdefault((task, outcome), []).append(int(episode))

    val_episodes: set[int] = set()
    for episodes in groups.values():
        episodes_array = np.asarray(episodes, dtype=np.int64)
        if episodes_array.size <= 1:
            continue
        shuffled = episodes_array.copy()
        rng.shuffle(shuffled)
        val_count = int(round(episodes_array.size * float(val_ratio)))
        val_count = min(max(val_count, 1), episodes_array.size - 1)
        val_episodes.update(int(ep) for ep in shuffled[:val_count])

    anchor_episodes = cache.episode[anchors]
    val_mask = np.asarray([int(ep) in val_episodes for ep in anchor_episodes], dtype=bool)
    train = np.sort(anchors[~val_mask])
    val = np.sort(anchors[val_mask])
    return train, val


def save_checkpoint(
    path: Path,
    model: Any,
    config: SoarDreamerLiteConfig,
    normalizer: Normalizer,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "normalizer": normalizer.to_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(model: Any, path: Path, device: Any) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}


def resolve_device(device: str) -> Any:
    if torch is None:
        raise RuntimeError("torch is required.")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def emit_progress(stage: str, row: dict[str, Any]) -> None:
    print(json.dumps({"progress": stage, **row}, sort_keys=True), flush=True)


def write_report(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SOAR Dreamer-Lite Training",
        "",
        "This run treats Kairos/Sensenova features as frozen visual world-model latents and trains the missing SOAR control stack on top.",
        "",
        "## Objective",
        "",
        "1. Encode SOAR video as frozen `z_t`.",
        "2. Learn `z_context + action_sequence -> future_z`.",
        "3. Train isolated agent-token action, reward, and value heads.",
        "4. Freeze dynamics and improve the policy/value heads in imagined rollouts.",
        "",
        "## Cache",
        "",
        f"- Steps: {summary.get('cache', {}).get('steps')}",
        f"- Episodes: {summary.get('cache', {}).get('episodes')}",
        f"- Tasks: {summary.get('cache', {}).get('tasks')}",
        f"- Latent dim: {summary.get('cache', {}).get('z_dim')}",
        f"- Action dim: {summary.get('cache', {}).get('action_dim')}",
        f"- Valid anchors: {summary.get('anchors', {}).get('total')}",
        "",
        "## Action Grounding Controls",
        "",
        f"- Dynamics architecture: {summary.get('config', {}).get('dynamics_architecture')}",
        f"- Dynamics residual mode: {summary.get('config', {}).get('dynamics_residual_mode')}",
        f"- Future action offset: {summary.get('config', {}).get('future_action_offset')}",
        f"- Future action window: {summary.get('config', {}).get('future_action_window')}",
        f"- Future action reduce: {summary.get('config', {}).get('future_action_reduce')}",
        f"- Reward target mode: {summary.get('config', {}).get('reward_target_mode')}",
        f"- Reward loss type: {summary.get('config', {}).get('reward_loss_type')}",
        f"- Value target mode: {summary.get('config', {}).get('value_target_mode')}",
        f"- Value loss type: {summary.get('config', {}).get('value_loss_type')}",
        f"- Imagination trains value head: {summary.get('config', {}).get('imagination_train_value_head')}",
        f"- Real value replay loss weight: {summary.get('config', {}).get('real_value_replay_loss_weight')}",
        f"- Imagination value loss weight: {summary.get('config', {}).get('imagination_value_loss_weight')}",
        f"- Prior loss weight: {summary.get('config', {}).get('prior_loss_weight')}",
        f"- Delta loss weight: {summary.get('config', {}).get('delta_loss_weight')}",
        f"- Contrastive loss weight: {summary.get('config', {}).get('contrastive_loss_weight')}",
        f"- Contrastive margin: {summary.get('config', {}).get('contrastive_margin')}",
        f"- Dynamics rollout loss weight: {summary.get('config', {}).get('dynamics_rollout_loss_weight')}",
        f"- Dynamics rollout contrastive loss weight: {summary.get('config', {}).get('dynamics_rollout_contrastive_loss_weight')}",
        f"- Dynamics rollout contrastive margin: {summary.get('config', {}).get('dynamics_rollout_contrastive_margin')}",
        f"- Dynamics rollout horizon: {summary.get('config', {}).get('dynamics_rollout_horizon')}",
        f"- Dynamics BC checkpoint metric: {summary.get('config', {}).get('dynamics_bc_metric')}",
        f"- Dynamics BC early stop patience: {summary.get('config', {}).get('dynamics_bc_early_stop_patience')}",
        f"- Motion filter: {summary.get('motion_filter')}",
        "",
        "## Final Metrics",
        "",
    ]
    bc_history = summary.get("bc_history") or []
    if bc_history:
        last = bc_history[-1]
        lines.append(f"- Dynamics/BC final train loss: {last.get('train', {}).get('loss')}")
        lines.append(f"- Dynamics/BC final val loss: {last.get('val', {}).get('loss')}")
    agent_bc_history = summary.get("agent_bc_history") or []
    if agent_bc_history:
        last = agent_bc_history[-1]
        best = summary.get("agent_bc_best") or best_history_row(
            agent_bc_history,
            str(summary.get("config", {}).get("agent_bc_metric", "loss")),
        )
        early_stop = summary.get("agent_bc_early_stop") or {}
        lines.append(f"- Frozen-dynamics agent BC selection metric: {summary.get('config', {}).get('agent_bc_metric')}")
        lines.append(f"- Frozen-dynamics agent BC early stop: {early_stop}")
        lines.append(f"- Frozen-dynamics agent BC best epoch: {best.get('epoch')}")
        lines.append(f"- Frozen-dynamics agent BC best val loss: {best.get('val', {}).get('loss')}")
        lines.append(f"- Frozen-dynamics agent BC best val action MSE: {best.get('val', {}).get('action_mse')}")
        lines.append(f"- Frozen-dynamics agent BC best val reward MSE: {best.get('val', {}).get('reward_mse')}")
        lines.append(f"- Frozen-dynamics agent BC best val value MSE: {best.get('val', {}).get('value_mse')}")
        lines.append(f"- Frozen-dynamics agent BC final train loss: {last.get('train', {}).get('loss')}")
        lines.append(f"- Frozen-dynamics agent BC final val loss: {last.get('val', {}).get('loss')}")
        lines.append(f"- Frozen-dynamics agent BC final val action MSE: {last.get('val', {}).get('action_mse')}")
    calibration = summary.get("agent_calibration_eval") or {}
    if calibration:
        lines.append(f"- Agent reward Brier: {calibration.get('reward_brier')}")
        lines.append(f"- Agent reward accuracy: {calibration.get('reward_accuracy')}")
        lines.append(f"- Agent reward ECE@10: {calibration.get('reward_ece_10')}")
        lines.append(f"- Agent reward target/pred mean: {calibration.get('reward_target_mean')} / {calibration.get('reward_pred_mean')}")
        lines.append(f"- Agent value MSE: {calibration.get('value_mse')}")
        lines.append(f"- Agent value MAE: {calibration.get('value_mae')}")
        lines.append(f"- Agent value target/pred mean: {calibration.get('value_target_mean')} / {calibration.get('value_pred_mean')}")
        lines.append(f"- Agent value corr: {calibration.get('value_corr')}")
    control_eval = summary.get("dynamics_control_eval") or {}
    if control_eval:
        lines.append(f"- Action conditioning detected: {control_eval.get('action_conditioning_detected')}")
        lines.append(f"- Strict action-conditioning gate passed: {control_eval.get('strict_gate_passed')}")
        lines.append(f"- Action-conditioning strength: {control_eval.get('action_conditioning_strength')}")
        lines.append(f"- Normal/persistence MSE: {control_eval.get('normal_over_persistence')}")
        lines.append(f"- Shuffled/normal MSE: {control_eval.get('shuffle_over_normal')}")
        lines.append(f"- Zero/normal MSE: {control_eval.get('zero_over_normal')}")
    imagination_history = summary.get("imagination_history") or []
    if imagination_history:
        last = imagination_history[-1]
        lines.append(f"- Imagination final train loss: {last.get('train', {}).get('loss')}")
        lines.append(f"- Imagination final val loss: {last.get('val', {}).get('loss')}")
        lines.append(f"- Imagination mean reward: {last.get('train', {}).get('mean_imagined_reward')}")
        lines.append(f"- Imagination value loss weight: {last.get('train', {}).get('value_loss_weight')}")
        lines.append(f"- Imagination real-value replay loss: {last.get('train', {}).get('real_value_replay_loss')}")
    if not bc_history and not agent_bc_history and not imagination_history:
        lines.append("- No training stage was run.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def schema_summary() -> dict[str, Any]:
    return {
        "frozen_visual_state": "z_t from SOAR/Kairos cache",
        "dynamics": "z[t-C+1:t], a[t-C+1:t], a[t:t+H-1], task -> z[t+1:t+H]",
        "agent": "agent token reads z/action/task context; context/world tokens cannot read agent/task tokens",
        "bc_targets": "action[t:t+L], reward[t:t+L], discounted value target",
        "imagination": "freeze dynamics and frozen reward/prior heads; optimize action heads, with optional value-head updates or real-value replay calibration",
        "frozen_agent_bc": "load strict-gated dynamics checkpoint, freeze dynamics, train only policy/reward/value heads",
        "action_grounding": "optional future-action offsets, motion-filtered anchors, delta-z loss, true-vs-shuffled/zero contrastive loss, and autoregressive rollout-consistency loss",
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "phase": summary.get("phase"),
        "out_dir": summary.get("config", {}).get("out_dir"),
        "stage": summary.get("config", {}).get("stage"),
        "cache": summary.get("cache"),
        "anchors": summary.get("anchors"),
        "motion_filter": summary.get("motion_filter"),
        "elapsed_s": summary.get("elapsed_s"),
    }
    bc_history = summary.get("bc_history") or []
    if bc_history:
        compact["bc_final"] = bc_history[-1]
    agent_bc_history = summary.get("agent_bc_history") or []
    if agent_bc_history:
        compact["agent_bc_final"] = agent_bc_history[-1]
        compact["agent_bc_best"] = summary.get("agent_bc_best") or best_history_row(
            agent_bc_history,
            str(summary.get("config", {}).get("agent_bc_metric", "loss")),
        )
        compact["agent_bc_early_stop"] = summary.get("agent_bc_early_stop")
    imagination_history = summary.get("imagination_history") or []
    if imagination_history:
        compact["imagination_final"] = imagination_history[-1]
    if summary.get("dynamics_control_eval"):
        compact["dynamics_control_eval"] = summary.get("dynamics_control_eval")
    if summary.get("agent_calibration_eval"):
        compact["agent_calibration_eval"] = summary.get("agent_calibration_eval")
    return compact


if __name__ == "__main__":
    raise SystemExit(main())
