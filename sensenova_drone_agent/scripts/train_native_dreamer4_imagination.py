#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DREAMER4_ROOT = REPO_ROOT / "dreamer4" / "dreamer4"
for item in [str(DREAMER4_ROOT), str(PROJECT_ROOT / "scripts")]:
    if item not in sys.path:
        sys.path.insert(0, item)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    from torch.utils.data import DataLoader, Subset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("torch is required for native Dreamer4 imagination training.") from exc

from train_dynamics import (  # noqa: E402
    DEFAULT_ACTION_DIM,
    Dynamics,
    align_actions_to_frames,
    load_frozen_tokenizer_from_pt_ckpt,
    make_tau_schedule,
    pack_bottleneck_to_spatial,
    sample_one_timestep_packed,
    temporal_patchify,
)
from wm_dataset import WMDataset, collate_batch, normalize_action_features  # noqa: E402
from residual_adapter_runtime import (  # noqa: E402
    infer_adapter_action_overrides,
    wrap_dynamics_with_residual_adapter,
    write_residual_adapter_info,
)


@dataclass
class NativeImaginationConfig:
    data_dirs: list[str]
    frame_dirs: list[str]
    tasks_json: str
    tokenizer_ckpt: str
    dynamics_ckpt: str
    out_dir: str
    residual_adapter_ckpt: str = ""
    seq_len: int = 16
    ctx_len: int = 8
    imagination_horizon: int = 8
    batch_size: int = 4
    num_workers: int = 2
    bc_steps: int = 1200
    imagination_updates: int = 400
    eval_batches: int = 64
    action_dim: int = DEFAULT_ACTION_DIM
    raw_action_dim: int = DEFAULT_ACTION_DIM
    action_features: str = "current"
    policy_action_source: str = "expanded"
    action_chunk_len: int = 1
    action_frame_offset: int = 0
    hidden_dim: int = 512
    learning_rate: float = 3e-4
    imagination_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gamma: float = 0.997
    prior_weight: float = 0.3
    entropy_weight: float = 1e-3
    reward_loss_weight: float = 1.0
    value_loss_weight: float = 0.25
    target_normalization: str = "none"
    min_target_std: float = 1e-3
    reward_clip: float = 0.0
    value_clip: float = 0.0
    eval_holdout_fraction: float = 0.1
    split_seed: int = 0
    imagination_mode: str = "train"
    select_best_imagination: bool = False
    imagination_eval_every: int = 0
    min_imagination_selection_update: int = 0
    best_imagination_metric: str = "policy_minus_bc"
    detach_policy_log_prob: bool = True
    imagination_dynamics_action_mode: str = "policy"
    imagination_agent_action_context_mode: str = "policy"
    reward_value_action_context_mode: str = "policy"
    reward_contrast_weight: float = 0.0
    reward_contrast_margin: float = 0.05
    reward_contrast_start: int = 0
    reward_contrast_every: int = 1
    reward_contrast_negative_modes: str = "zero,shuffle"
    reward_contrast_horizon: int = 1
    reward_contrast_positive_threshold: float = 0.0
    reward_contrast_min_action_norm: float = 0.0
    causal_policy_mode: str = "off"
    causal_policy_negative_modes: str = "zero,shuffle"
    causal_policy_min_margin: float = 0.0
    causal_shortfall_policy_weight: float = 0.0
    causal_shortfall_margin: float = -1.0
    eval_causal_dynamics: bool = False
    source_eval_sources: str = ""
    source_eval_batches: int = 0
    source_gate_hard_sources: str = "all,soar"
    source_gate_soft_sources: str = "droid"
    source_gate_soft_min_margin: float = -0.005
    aux_inverse_weight: float = 0.0
    aux_effect_weight: float = 0.0
    aux_action_effect_min_norm: float = 0.0
    advantage_mode: str = "raw_sign"
    advantage_baseline: str = "value"
    advantage_clip: float = 2.0
    policy_loss_min_advantage_abs: float = 0.0
    policy_loss_max_prior_mse: float = 0.0
    prior_hinge_weight: float = 0.0
    prior_hinge_target: float = 0.01
    mean_prior_weight: float = 0.0
    mean_prior_hinge_weight: float = 0.0
    mean_prior_hinge_target: float = 0.005
    log_std_init: float = -1.0
    require_non_noop: bool = False
    no_op_threshold: float = 0.0
    min_non_noop_steps: int = 1
    reward_filter_mode: str = "none"
    reward_signal_threshold: float = 0.0
    min_reward_signal_steps: int = 1
    train_sampling_mode: str = "shuffle"
    train_balance_spec: str = "hf_expert_positive=0.25,hf_mixed_positive=0.25,hf_mixed_zero=0.25,soar_game_positive=0.25"
    train_balance_return_threshold: float = 0.0
    train_balanced_samples: int = 0
    train_balance_seed: int = 0
    train_action_active_threshold: float = 0.0
    train_min_action_active_steps: int = 1
    freeze_value_during_imagination: bool = True
    eval_d: float = 0.25
    eval_seed: int = 0
    seed: int = 0
    device: str = "cuda"


class AgentHeads(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int,
        log_std_init: float,
        *,
        z_dim: int,
        single_action_dim: int,
    ):
        super().__init__()
        self.prior = make_mlp(input_dim, hidden_dim, action_dim)
        self.policy = make_mlp(input_dim, hidden_dim, action_dim)
        self.reward = make_mlp(input_dim, hidden_dim, 1)
        self.value = make_mlp(input_dim, hidden_dim, 1)
        self.inverse = make_mlp(2 * int(z_dim), hidden_dim, int(single_action_dim))
        self.effect = make_mlp(int(z_dim) + int(single_action_dim), hidden_dim, int(z_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def prior_mean(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.prior(features))

    def policy_mean(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.policy(features))

    def reward_pred(self, features: torch.Tensor) -> torch.Tensor:
        return self.reward(features).squeeze(-1)

    def value_pred(self, features: torch.Tensor) -> torch.Tensor:
        return self.value(features).squeeze(-1)

    def inverse_pred(self, z_pair: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.inverse(z_pair))

    def effect_pred(self, z_action: torch.Tensor) -> torch.Tensor:
        return self.effect(z_action)

    def action_dist(self, features: torch.Tensor) -> Normal:
        mean = self.policy_mean(features)
        std = self.log_std.exp().clamp(0.05, 2.0).view(1, -1).expand_as(mean)
        return Normal(mean, std)


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def imagination_metric_value(eval_payload: dict[str, Any], config: NativeImaginationConfig) -> float:
    if config.best_imagination_metric == "policy_minus_bc_plus_dyn_shuffle":
        return float(eval_payload.get("policy_minus_bc", 0.0)) + float(eval_payload.get("policy_minus_dyn_shuffle", 0.0))
    if config.best_imagination_metric == "policy_minus_bc_causal_gate":
        policy_minus_bc = float(eval_payload.get("policy_minus_bc", 0.0))
        policy_minus_dyn_shuffle = float(eval_payload.get("policy_minus_dyn_shuffle", 0.0))
        if policy_minus_bc > 0.0 and policy_minus_dyn_shuffle >= float(config.causal_policy_min_margin):
            return policy_minus_bc
        # Keep failed checkpoints ordered for diagnostics without allowing them to beat passed gates.
        return -1e6 + policy_minus_bc + policy_minus_dyn_shuffle
    if config.best_imagination_metric == "policy_minus_bc_zero_causal_gate":
        policy_minus_bc = float(eval_payload.get("policy_minus_bc", 0.0))
        policy_minus_zero = float(eval_payload.get("policy_minus_zero", 0.0))
        policy_minus_dyn_zero = float(eval_payload.get("policy_minus_dyn_zero", 0.0))
        policy_minus_dyn_shuffle = float(eval_payload.get("policy_minus_dyn_shuffle", 0.0))
        margin = float(config.causal_policy_min_margin)
        if (
            policy_minus_bc > 0.0
            and policy_minus_zero >= 0.0
            and policy_minus_dyn_zero >= margin
            and policy_minus_dyn_shuffle >= margin
        ):
            return policy_minus_bc
        # Rank failed checkpoints by distance to the hard gate for diagnostics.
        return (
            -1e6
            + policy_minus_bc
            + min(policy_minus_zero, 0.0)
            + min(policy_minus_dyn_zero - margin, 0.0)
            + min(policy_minus_dyn_shuffle - margin, 0.0)
        )
    if config.best_imagination_metric == "policy_minus_bc_zero_causal_gate_source_aware":
        return source_aware_imagination_metric_value(eval_payload, config)
    return float(eval_payload.get(config.best_imagination_metric, eval_payload.get("policy", -float("inf"))))


def source_aware_imagination_metric_value(eval_payload: dict[str, Any], config: NativeImaginationConfig) -> float:
    margin = float(config.causal_policy_min_margin)
    soft_min = float(config.source_gate_soft_min_margin)
    source_eval = eval_payload.get("source_eval", {})
    hard_sources = parse_source_names(config.source_gate_hard_sources, default=("all", "soar"))
    soft_sources = parse_source_names(config.source_gate_soft_sources, default=("droid",))
    penalty = 0.0

    for source in hard_sources:
        payload = source_eval.get(source, eval_payload if source == "all" else None)
        if payload is None:
            penalty -= margin
            continue
        penalty += zero_causal_gate_shortfall(payload, margin=margin)

    for source in soft_sources:
        payload = source_eval.get(source, eval_payload if source == "all" else None)
        if payload is None:
            continue
        penalty += soft_source_gate_shortfall(payload, soft_min=soft_min)

    policy_minus_bc = float(eval_payload.get("policy_minus_bc", 0.0))
    if penalty >= 0.0:
        return policy_minus_bc
    return -1e6 + policy_minus_bc + penalty


def zero_causal_gate_shortfall(payload: dict[str, Any], *, margin: float) -> float:
    return (
        min(float(payload.get("policy_minus_bc", 0.0)), 0.0)
        + min(float(payload.get("policy_minus_zero", 0.0)), 0.0)
        + min(float(payload.get("policy_minus_dyn_zero", 0.0)) - margin, 0.0)
        + min(float(payload.get("policy_minus_dyn_shuffle", 0.0)) - margin, 0.0)
    )


def soft_source_gate_shortfall(payload: dict[str, Any], *, soft_min: float) -> float:
    return (
        min(float(payload.get("policy_minus_bc", 0.0)), 0.0)
        + min(float(payload.get("policy_minus_zero", 0.0)), 0.0)
        + min(float(payload.get("policy_minus_dyn_zero", 0.0)) - soft_min, 0.0)
        + min(float(payload.get("policy_minus_dyn_shuffle", 0.0)) - soft_min, 0.0)
    )


def parse_source_names(value: str, *, default: tuple[str, ...]) -> list[str]:
    sources = [item.strip().lower() for item in str(value).replace("+", ",").split(",") if item.strip()]
    return sources or list(default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First native Dreamer4-style imagination test on frozen dynamics.")
    parser.add_argument("--data-dir", action="append", required=True)
    parser.add_argument("--frames-dir", action="append", required=True)
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--tokenizer-ckpt", required=True)
    parser.add_argument("--dynamics-ckpt", required=True)
    parser.add_argument(
        "--residual-adapter-ckpt",
        default="",
        help="Optional residual action adapter checkpoint to wrap around the frozen dynamics model.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--imagination-horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bc-steps", type=int, default=1200)
    parser.add_argument("--imagination-updates", type=int, default=400)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--raw-action-dim", type=int, default=None)
    parser.add_argument("--action-features", default=None)
    parser.add_argument("--policy-action-source", choices=["expanded", "raw"], default="expanded")
    parser.add_argument("--action-chunk-len", type=int, default=1)
    parser.add_argument("--action-frame-offset", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--imagination-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--prior-weight", type=float, default=0.3)
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--target-normalization", choices=["none", "global", "per_task"], default="none")
    parser.add_argument("--min-target-std", type=float, default=1e-3)
    parser.add_argument("--reward-clip", type=float, default=0.0)
    parser.add_argument("--value-clip", type=float, default=0.0)
    parser.add_argument("--eval-holdout-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--imagination-mode", choices=["train", "no_update"], default="train")
    parser.add_argument("--select-best-imagination", action="store_true")
    parser.add_argument("--imagination-eval-every", type=int, default=0)
    parser.add_argument(
        "--min-imagination-selection-update",
        type=int,
        default=0,
        help="Minimum imagination update eligible for best-checkpoint selection; use >0 to exclude pure BC.",
    )
    parser.add_argument(
        "--best-imagination-metric",
        choices=[
            "policy",
            "policy_minus_bc",
            "policy_minus_zero",
            "causal_policy_gain",
            "policy_minus_dyn_zero",
            "policy_minus_dyn_shuffle",
            "policy_minus_bc_plus_dyn_shuffle",
            "policy_minus_bc_causal_gate",
            "policy_minus_bc_zero_causal_gate",
            "policy_minus_bc_zero_causal_gate_source_aware",
        ],
        default="policy_minus_bc",
    )
    parser.add_argument(
        "--no-detach-policy-log-prob",
        action="store_true",
        help="Ablation: use the old reparameterized-sample log_prob path instead of score-function PMPO.",
    )
    parser.add_argument(
        "--imagination-dynamics-action-mode",
        choices=["policy", "zero", "shuffle"],
        default="policy",
        help="Ablation: corrupt the actions fed to dynamics during imagination updates.",
    )
    parser.add_argument(
        "--imagination-agent-action-context-mode",
        choices=["policy", "zero", "shuffle"],
        default="policy",
        help="Ablation: corrupt the action history seen by policy/reward/value features during imagination updates.",
    )
    parser.add_argument(
        "--reward-value-action-context-mode",
        choices=["policy", "zero"],
        default="policy",
        help="Use policy action history or zeroed action features for reward/value heads.",
    )
    parser.add_argument("--reward-contrast-weight", type=float, default=0.0)
    parser.add_argument("--reward-contrast-margin", type=float, default=0.05)
    parser.add_argument("--reward-contrast-start", type=int, default=0)
    parser.add_argument("--reward-contrast-every", type=int, default=1)
    parser.add_argument("--reward-contrast-negative-modes", default="zero,shuffle")
    parser.add_argument("--reward-contrast-horizon", type=int, default=1)
    parser.add_argument("--reward-contrast-positive-threshold", type=float, default=0.0)
    parser.add_argument("--reward-contrast-min-action-norm", type=float, default=0.0)
    parser.add_argument(
        "--causal-policy-mode",
        choices=["off", "advantage", "gate", "advantage_gate"],
        default="off",
        help=(
            "Use counterfactual dynamics rollouts inside PMPO. advantage replaces advantages "
            "with real-minus-counterfactual return; gate masks tiny causal margins; "
            "advantage_gate does both."
        ),
    )
    parser.add_argument("--causal-policy-negative-modes", default="zero,shuffle")
    parser.add_argument("--causal-policy-min-margin", type=float, default=0.0)
    parser.add_argument(
        "--causal-shortfall-policy-weight",
        type=float,
        default=0.0,
        help="Extra score-function penalty on sampled actions whose imagined return fails causal controls.",
    )
    parser.add_argument(
        "--causal-shortfall-margin",
        type=float,
        default=-1.0,
        help="Margin for the causal shortfall penalty. Negative means reuse causal-policy-min-margin.",
    )
    parser.add_argument(
        "--eval-causal-dynamics",
        action="store_true",
        help="Evaluate policy rollouts under zero/shuffled dynamics actions and report causal_policy_gain.",
    )
    parser.add_argument(
        "--source-eval-sources",
        default="",
        help="Optional source breakdown for best-checkpoint selection, e.g. all,soar,droid.",
    )
    parser.add_argument(
        "--source-eval-batches",
        type=int,
        default=0,
        help="Eval batches per source-aware selection eval. Zero reuses eval-batches.",
    )
    parser.add_argument("--source-gate-hard-sources", default="all,soar")
    parser.add_argument("--source-gate-soft-sources", default="droid")
    parser.add_argument("--source-gate-soft-min-margin", type=float, default=-0.005)
    parser.add_argument("--aux-inverse-weight", type=float, default=0.0)
    parser.add_argument("--aux-effect-weight", type=float, default=0.0)
    parser.add_argument("--aux-action-effect-min-norm", type=float, default=0.0)
    parser.add_argument("--advantage-mode", choices=["raw_sign", "centered_sign", "weighted"], default="raw_sign")
    parser.add_argument("--advantage-baseline", choices=["value", "bc_return"], default="value")
    parser.add_argument("--advantage-clip", type=float, default=2.0)
    parser.add_argument("--policy-loss-min-advantage-abs", type=float, default=0.0)
    parser.add_argument("--policy-loss-max-prior-mse", type=float, default=0.0)
    parser.add_argument("--prior-hinge-weight", type=float, default=0.0)
    parser.add_argument("--prior-hinge-target", type=float, default=0.01)
    parser.add_argument("--mean-prior-weight", type=float, default=0.0)
    parser.add_argument("--mean-prior-hinge-weight", type=float, default=0.0)
    parser.add_argument("--mean-prior-hinge-target", type=float, default=0.005)
    parser.add_argument("--log-std-init", type=float, default=-1.0)
    parser.add_argument("--require-non-noop", action="store_true")
    parser.add_argument("--no-op-threshold", type=float, default=0.0)
    parser.add_argument("--min-non-noop-steps", type=int, default=1)
    parser.add_argument(
        "--reward-filter-mode",
        choices=["none", "positive_sum", "abs_sum", "any_positive", "any_abs"],
        default="none",
    )
    parser.add_argument("--reward-signal-threshold", type=float, default=0.0)
    parser.add_argument("--min-reward-signal-steps", type=int, default=1)
    parser.add_argument("--train-sampling-mode", choices=["shuffle", "dreamer4_reward_mixture"], default="shuffle")
    parser.add_argument(
        "--train-balance-spec",
        default="hf_expert_positive=0.25,hf_mixed_positive=0.25,hf_mixed_zero=0.25,soar_game_positive=0.25",
    )
    parser.add_argument("--train-balance-return-threshold", type=float, default=0.0)
    parser.add_argument("--train-balanced-samples", type=int, default=0)
    parser.add_argument("--train-balance-seed", type=int, default=0)
    parser.add_argument("--train-action-active-threshold", type=float, default=0.0)
    parser.add_argument("--train-min-action-active-steps", type=int, default=1)
    parser.add_argument("--train-value-during-imagination", action="store_true")
    parser.add_argument("--eval-d", type=float, default=0.25)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    tokenizer_ckpt = resolve_path(args.tokenizer_ckpt)
    dynamics_ckpt = resolve_path(args.dynamics_ckpt)
    dyn_ckpt = torch.load(dynamics_ckpt, map_location="cpu", weights_only=False)
    dyn_args = dict(dyn_ckpt.get("args", {}))
    adapter_ckpt = resolve_path(args.residual_adapter_ckpt) if args.residual_adapter_ckpt else None
    adapter_action_overrides = infer_adapter_action_overrides(adapter_ckpt)
    action_dim = int(
        args.action_dim
        if args.action_dim is not None
        else adapter_action_overrides.get("action_dim", dyn_args.get("action_dim", DEFAULT_ACTION_DIM))
    )
    action_features = str(
        args.action_features
        if args.action_features is not None
        else adapter_action_overrides.get("action_features", dyn_args.get("action_features", "current"))
    )
    raw_action_dim = int(
        args.raw_action_dim
        if args.raw_action_dim is not None
        else infer_raw_action_dim(resolve_path(args.tasks_json), fallback=action_dim)
    )
    dyn_args["action_dim"] = action_dim
    dyn_args["action_features"] = action_features

    config = NativeImaginationConfig(
        data_dirs=[str(resolve_path(path)) for path in args.data_dir],
        frame_dirs=[str(resolve_path(path)) for path in args.frames_dir],
        tasks_json=str(resolve_path(args.tasks_json)),
        tokenizer_ckpt=str(tokenizer_ckpt),
        dynamics_ckpt=str(dynamics_ckpt),
        residual_adapter_ckpt=str(adapter_ckpt) if adapter_ckpt else "",
        out_dir=str(out_dir),
        seq_len=int(args.seq_len),
        ctx_len=int(args.ctx_len),
        imagination_horizon=int(args.imagination_horizon),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        bc_steps=int(args.bc_steps),
        imagination_updates=int(args.imagination_updates),
        eval_batches=int(args.eval_batches),
        action_dim=action_dim,
        raw_action_dim=raw_action_dim,
        action_features=action_features,
        policy_action_source=str(args.policy_action_source),
        action_chunk_len=max(1, int(args.action_chunk_len)),
        action_frame_offset=int(args.action_frame_offset),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        imagination_learning_rate=float(args.imagination_learning_rate),
        weight_decay=float(args.weight_decay),
        gamma=float(args.gamma),
        prior_weight=float(args.prior_weight),
        entropy_weight=float(args.entropy_weight),
        reward_loss_weight=float(args.reward_loss_weight),
        value_loss_weight=float(args.value_loss_weight),
        target_normalization=str(args.target_normalization),
        min_target_std=float(args.min_target_std),
        reward_clip=float(args.reward_clip),
        value_clip=float(args.value_clip),
        eval_holdout_fraction=float(args.eval_holdout_fraction),
        split_seed=int(args.split_seed),
        imagination_mode=str(args.imagination_mode),
        select_best_imagination=bool(args.select_best_imagination),
        imagination_eval_every=int(args.imagination_eval_every),
        min_imagination_selection_update=max(0, int(args.min_imagination_selection_update)),
        best_imagination_metric=str(args.best_imagination_metric),
        detach_policy_log_prob=not bool(args.no_detach_policy_log_prob),
        imagination_dynamics_action_mode=str(args.imagination_dynamics_action_mode),
        imagination_agent_action_context_mode=str(args.imagination_agent_action_context_mode),
        reward_value_action_context_mode=str(args.reward_value_action_context_mode),
        reward_contrast_weight=float(args.reward_contrast_weight),
        reward_contrast_margin=float(args.reward_contrast_margin),
        reward_contrast_start=int(args.reward_contrast_start),
        reward_contrast_every=int(args.reward_contrast_every),
        reward_contrast_negative_modes=str(args.reward_contrast_negative_modes),
        reward_contrast_horizon=max(1, int(args.reward_contrast_horizon)),
        reward_contrast_positive_threshold=float(args.reward_contrast_positive_threshold),
        reward_contrast_min_action_norm=float(args.reward_contrast_min_action_norm),
        causal_policy_mode=str(args.causal_policy_mode),
        causal_policy_negative_modes=str(args.causal_policy_negative_modes),
        causal_policy_min_margin=float(args.causal_policy_min_margin),
        causal_shortfall_policy_weight=float(args.causal_shortfall_policy_weight),
        causal_shortfall_margin=float(args.causal_shortfall_margin),
        eval_causal_dynamics=bool(args.eval_causal_dynamics)
        or str(args.best_imagination_metric)
        in {
            "causal_policy_gain",
            "policy_minus_dyn_zero",
            "policy_minus_dyn_shuffle",
            "policy_minus_bc_plus_dyn_shuffle",
            "policy_minus_bc_causal_gate",
            "policy_minus_bc_zero_causal_gate",
            "policy_minus_bc_zero_causal_gate_source_aware",
        },
        source_eval_sources=str(args.source_eval_sources),
        source_eval_batches=max(0, int(args.source_eval_batches)),
        source_gate_hard_sources=str(args.source_gate_hard_sources),
        source_gate_soft_sources=str(args.source_gate_soft_sources),
        source_gate_soft_min_margin=float(args.source_gate_soft_min_margin),
        aux_inverse_weight=float(args.aux_inverse_weight),
        aux_effect_weight=float(args.aux_effect_weight),
        aux_action_effect_min_norm=float(args.aux_action_effect_min_norm),
        advantage_mode=str(args.advantage_mode),
        advantage_baseline=str(args.advantage_baseline),
        advantage_clip=float(args.advantage_clip),
        policy_loss_min_advantage_abs=float(args.policy_loss_min_advantage_abs),
        policy_loss_max_prior_mse=float(args.policy_loss_max_prior_mse),
        prior_hinge_weight=float(args.prior_hinge_weight),
        prior_hinge_target=float(args.prior_hinge_target),
        mean_prior_weight=float(args.mean_prior_weight),
        mean_prior_hinge_weight=float(args.mean_prior_hinge_weight),
        mean_prior_hinge_target=float(args.mean_prior_hinge_target),
        log_std_init=float(args.log_std_init),
        require_non_noop=bool(args.require_non_noop),
        no_op_threshold=float(args.no_op_threshold),
        min_non_noop_steps=max(1, int(args.min_non_noop_steps)),
        reward_filter_mode=str(args.reward_filter_mode),
        reward_signal_threshold=float(args.reward_signal_threshold),
        min_reward_signal_steps=max(1, int(args.min_reward_signal_steps)),
        train_sampling_mode=str(args.train_sampling_mode),
        train_balance_spec=str(args.train_balance_spec),
        train_balance_return_threshold=float(args.train_balance_return_threshold),
        train_balanced_samples=int(args.train_balanced_samples),
        train_balance_seed=int(args.train_balance_seed),
        train_action_active_threshold=float(args.train_action_active_threshold),
        train_min_action_active_steps=max(1, int(args.train_min_action_active_steps)),
        freeze_value_during_imagination=not bool(args.train_value_during_imagination),
        eval_d=float(args.eval_d),
        eval_seed=int(args.eval_seed),
        seed=int(args.seed),
        device=str(device),
    )
    write_json(out_dir / "config.json", asdict(config))

    encoder, _decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(tokenizer_ckpt), device=device)
    dynamics = build_dynamics(dyn_args, tok_args).to(device)
    dynamics.load_state_dict(dyn_ckpt["dynamics"], strict=True)
    dynamics.eval()
    for param in dynamics.parameters():
        param.requires_grad_(False)
    residual_adapter_info = None
    if adapter_ckpt is not None:
        dynamics, residual_adapter_info = wrap_dynamics_with_residual_adapter(
            base=dynamics,
            adapter_ckpt=adapter_ckpt,
            dyn_args=dyn_args,
            tok_args=tok_args,
            device=device,
        )
        write_residual_adapter_info(out_dir / "residual_adapter_info.json", residual_adapter_info)

    train_dataset = WMDataset(
        data_dir=config.data_dirs,
        frames_dir=config.frame_dirs,
        seq_len=config.seq_len,
        img_size=128,
        action_dim=config.action_dim,
        raw_action_dim=config.raw_action_dim,
        tasks_json=config.tasks_json,
        tasks=None,
        strict_tasks=False,
        action_features=config.action_features,
        require_non_noop=config.require_non_noop,
        no_op_threshold=config.no_op_threshold,
        min_non_noop_steps=config.min_non_noop_steps,
        reward_filter_mode=config.reward_filter_mode,
        reward_signal_threshold=config.reward_signal_threshold,
        min_reward_signal_steps=config.min_reward_signal_steps,
        verbose=True,
    )
    eval_dataset = WMDataset(
        data_dir=config.data_dirs,
        frames_dir=config.frame_dirs,
        seq_len=config.seq_len,
        img_size=128,
        action_dim=config.action_dim,
        raw_action_dim=config.raw_action_dim,
        tasks_json=config.tasks_json,
        tasks=None,
        strict_tasks=False,
        action_features=config.action_features,
        require_non_noop=config.require_non_noop,
        no_op_threshold=config.no_op_threshold,
        min_non_noop_steps=config.min_non_noop_steps,
        reward_filter_mode=config.reward_filter_mode,
        reward_signal_threshold=config.reward_signal_threshold,
        min_reward_signal_steps=config.min_reward_signal_steps,
        verbose=False,
    )
    split_info = apply_episode_holdout_split(
        train_dataset,
        eval_dataset,
        holdout_fraction=config.eval_holdout_fraction,
        seed=config.split_seed,
    )
    train_loader_dataset: Any = train_dataset
    train_sampling_info: dict[str, Any] = {
        "mode": "shuffle",
        "sample_count": int(len(train_dataset)),
        "source_dataset_windows": int(len(train_dataset)),
    }
    if config.train_sampling_mode == "dreamer4_reward_mixture":
        train_indices, train_sampling_info = build_dreamer4_reward_mixture_indices(
            train_dataset,
            config=config,
            seed=config.train_balance_seed if config.train_balance_seed else config.seed + 32452843,
        )
        train_loader_dataset = Subset(train_dataset, train_indices)
    eval_indices, eval_sampling_info = build_balanced_eval_indices(
        eval_dataset,
        num_batches=config.eval_batches,
        batch_size=config.batch_size,
        seed=config.eval_seed if config.eval_seed else config.seed + 15485863,
    )
    eval_subset = Subset(eval_dataset, eval_indices)
    loader = DataLoader(
        train_loader_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_batch,
        persistent_workers=config.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_batch,
        persistent_workers=config.num_workers > 0,
    )
    source_eval_loaders = build_source_eval_loaders(config) if config.source_eval_sources else {}
    batches = cycle_loader(loader)
    target_stats = build_target_stats(train_dataset, config, device)
    write_json(out_dir / "target_stats.json", target_stats_for_json(target_stats, train_dataset.tasks))
    write_json(out_dir / "split_info.json", split_info)
    write_json(out_dir / "train_sampling_info.json", train_sampling_info)
    write_json(out_dir / "eval_sampling_info.json", eval_sampling_info)

    z_dim = int(tok_args.get("n_latents", 16) // int(dyn_args.get("packing_factor", 2))) * int(
        tok_args.get("d_bottleneck", 32)
    ) * int(dyn_args.get("packing_factor", 2))
    policy_dim = policy_single_action_dim(config)
    feature_dim = 2 * z_dim + policy_dim
    policy_output_dim = policy_dim * config.action_chunk_len
    agent = AgentHeads(
        feature_dim,
        policy_output_dim,
        config.hidden_dim,
        config.log_std_init,
        z_dim=z_dim,
        single_action_dim=policy_dim,
    ).to(device)

    bc_history = train_supervised_heads(
        agent=agent,
        encoder=encoder,
        dynamics=dynamics,
        batches=batches,
        tok_args=tok_args,
        dyn_args=dyn_args,
        config=config,
        device=device,
        target_stats=target_stats,
    )
    torch.save({"agent": agent.state_dict(), "config": asdict(config), "bc_history": bc_history}, out_dir / "bc_prior.pt")

    set_eval_seed(config)
    before_eval = evaluate_policies(
        agent=agent,
        encoder=encoder,
        dynamics=dynamics,
        loader=eval_loader,
        tok_args=tok_args,
        dyn_args=dyn_args,
        config=config,
        device=device,
        task_names=eval_dataset.tasks,
    )

    if config.imagination_mode == "no_update":
        imagination_history = [{"mode": "no_update", "update": 0}]
        best_selection_info = None
    else:
        imagination_history, best_selection_info = run_imagination_training(
            agent=agent,
            encoder=encoder,
            dynamics=dynamics,
            batches=batches,
            eval_loader=eval_loader,
            source_eval_loaders=source_eval_loaders,
            initial_eval=before_eval,
            tok_args=tok_args,
            dyn_args=dyn_args,
            config=config,
            device=device,
            target_stats=target_stats,
            task_names=eval_dataset.tasks,
        )
    torch.save(
        {
            "agent": agent.state_dict(),
            "config": asdict(config),
            "bc_history": bc_history,
            "imagination_history": imagination_history,
        },
        out_dir / "after_imagination.pt",
    )
    if best_selection_info is not None:
        write_json(out_dir / "best_imagination_selection.json", best_selection_info)

    set_eval_seed(config)
    after_eval = evaluate_policies(
        agent=agent,
        encoder=encoder,
        dynamics=dynamics,
        loader=eval_loader,
        tok_args=tok_args,
        dyn_args=dyn_args,
        config=config,
        device=device,
        task_names=eval_dataset.tasks,
    )

    summary = {
        "phase": "native_dreamer4_imagination_test",
        "config": asdict(config),
        "bc_history_tail": bc_history[-5:],
        "imagination_history_tail": imagination_history[-5:],
        "before_imagination": before_eval,
        "after_imagination": after_eval,
        "comparison": compare_eval(before_eval, after_eval),
        "best_imagination_selection": best_selection_info,
        "target_stats": target_stats_for_json(target_stats, train_dataset.tasks),
        "residual_adapter_info": residual_adapter_info,
        "split_info": split_info,
        "train_sampling_info": train_sampling_info,
        "eval_sampling_info": eval_sampling_info,
        "claim_boundary": (
            "This is a learned-dynamics imagination test only. It does not prove real-environment control."
        ),
        "elapsed_s": float(time.time() - started),
    }
    write_json(out_dir / "summary.json", summary)
    write_report(summary, out_dir / "report.md")
    print(json.dumps(compact_summary(summary), indent=2))
    return 0


def build_dynamics(dyn_args: dict[str, Any], tok_args: dict[str, Any]) -> Dynamics:
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    packing_factor = int(dyn_args.get("packing_factor", 2))
    assert n_latents % packing_factor == 0
    n_spatial = n_latents // packing_factor
    d_spatial = d_bottleneck * packing_factor
    return Dynamics(
        d_model=int(dyn_args.get("d_model_dyn", 128)),
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=int(dyn_args.get("n_register", 4)),
        n_agent=int(dyn_args.get("n_agent", 1)),
        n_heads=int(dyn_args.get("n_heads", 4)),
        depth=int(dyn_args.get("dyn_depth", 3)),
        k_max=int(dyn_args.get("k_max", 8)),
        dropout=0.0,
        mlp_ratio=float(dyn_args.get("mlp_ratio", 4.0)),
        time_every=int(dyn_args.get("time_every", 1)),
        space_mode=str(dyn_args.get("space_mode", "wm_agent_isolated")),
        scale_pos_embeds=bool(dyn_args.get("scale_pos_embeds", False)),
        action_dim=int(dyn_args.get("action_dim", DEFAULT_ACTION_DIM)),
    )


def infer_raw_action_dim(tasks_json: Path, *, fallback: int) -> int:
    try:
        payload = json.loads(tasks_json.read_text(encoding="utf-8"))
    except Exception:
        return int(fallback)
    dims = []
    if isinstance(payload, dict):
        for row in payload.values():
            if isinstance(row, dict) and "action_dim" in row:
                try:
                    dims.append(int(row["action_dim"]))
                except Exception:
                    continue
    return max(dims) if dims else int(fallback)


def policy_single_action_dim(config: NativeImaginationConfig) -> int:
    if config.policy_action_source == "raw":
        return int(config.raw_action_dim)
    return int(config.action_dim)


def apply_episode_holdout_split(
    train_dataset: WMDataset,
    eval_dataset: WMDataset,
    *,
    holdout_fraction: float,
    seed: int,
) -> dict[str, Any]:
    if train_dataset.tasks != eval_dataset.tasks:
        raise ValueError("train/eval datasets must have identical task order for split filtering.")

    holdout_fraction = max(0.0, min(0.95, float(holdout_fraction)))
    if holdout_fraction <= 0.0:
        train_total = recompute_cum_counts(train_dataset)
        eval_total = recompute_cum_counts(eval_dataset)
        return {
            "mode": "shared_all_windows",
            "holdout_fraction": holdout_fraction,
            "seed": int(seed),
            "train_windows": train_total,
            "eval_windows": eval_total,
            "tasks": [],
        }

    rows = []
    for task_idx, task in enumerate(train_dataset.tasks):
        starts = train_dataset.valid_starts[task_idx]
        if starts.numel() <= 1:
            train_dataset.valid_starts[task_idx] = starts
            eval_dataset.valid_starts[task_idx] = starts[:0]
            rows.append(
                {
                    "task_id": task_idx,
                    "task": task,
                    "total_windows": int(starts.numel()),
                    "train_windows": int(starts.numel()),
                    "eval_windows": 0,
                    "split_unit": "none_too_small",
                }
            )
            continue

        episodes = train_dataset.ep[task_idx][starts].to(torch.int64)
        unique_eps = torch.unique(episodes)
        generator = torch.Generator().manual_seed(int(seed) + task_idx * 1009)
        if unique_eps.numel() > 1:
            holdout_eps_n = int(round(float(unique_eps.numel()) * holdout_fraction))
            holdout_eps_n = max(1, min(int(unique_eps.numel()) - 1, holdout_eps_n))
            perm = torch.randperm(int(unique_eps.numel()), generator=generator)
            holdout_eps = unique_eps[perm[:holdout_eps_n]]
            eval_mask = torch.isin(episodes, holdout_eps)
            split_unit = "episode"
        else:
            holdout_n = int(round(float(starts.numel()) * holdout_fraction))
            holdout_n = max(1, min(int(starts.numel()) - 1, holdout_n))
            perm = torch.randperm(int(starts.numel()), generator=generator)
            eval_mask = torch.zeros(int(starts.numel()), dtype=torch.bool)
            eval_mask[perm[:holdout_n]] = True
            split_unit = "window"

        train_starts = starts[~eval_mask]
        eval_starts = starts[eval_mask]
        if train_starts.numel() == 0 or eval_starts.numel() == 0:
            split = max(1, int(starts.numel() * (1.0 - holdout_fraction)))
            split = min(split, int(starts.numel()) - 1)
            train_starts = starts[:split]
            eval_starts = starts[split:]
            split_unit = "window_fallback"

        train_dataset.valid_starts[task_idx] = train_starts.contiguous()
        eval_dataset.valid_starts[task_idx] = eval_starts.contiguous()
        rows.append(
            {
                "task_id": task_idx,
                "task": task,
                "total_windows": int(starts.numel()),
                "train_windows": int(train_starts.numel()),
                "eval_windows": int(eval_starts.numel()),
                "split_unit": split_unit,
            }
        )

    train_total = recompute_cum_counts(train_dataset)
    eval_total = recompute_cum_counts(eval_dataset)
    return {
        "mode": "episode_holdout",
        "holdout_fraction": holdout_fraction,
        "seed": int(seed),
        "train_windows": train_total,
        "eval_windows": eval_total,
        "tasks": rows,
    }


def recompute_cum_counts(dataset: WMDataset) -> int:
    total = 0
    cum_counts = []
    for starts in dataset.valid_starts:
        total += int(starts.numel())
        cum_counts.append(total)
    if total <= 0:
        raise ValueError("dataset split produced zero valid windows.")
    dataset._cum_counts = cum_counts
    return total


def build_balanced_eval_indices(
    dataset: WMDataset,
    *,
    num_batches: int,
    batch_size: int,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    total_samples = max(1, int(num_batches) * int(batch_size))
    prev_cum = [0] + [int(x) for x in dataset._cum_counts[:-1]]
    task_rows = []
    for task_idx, starts in enumerate(dataset.valid_starts):
        count = int(starts.numel())
        if count <= 0:
            continue
        task_rows.append((task_idx, count, prev_cum[task_idx]))
    if not task_rows:
        raise ValueError("eval split produced no non-empty tasks.")

    generator = torch.Generator().manual_seed(int(seed))
    indices: list[int] = []
    per_task_counts = {int(task_idx): 0 for task_idx, _count, _base in task_rows}
    for sample_idx in range(total_samples):
        task_idx, count, base = task_rows[sample_idx % len(task_rows)]
        local_idx = int(torch.randint(count, (1,), generator=generator).item())
        indices.append(int(base + local_idx))
        per_task_counts[int(task_idx)] += 1

    task_counts = [
        {
            "task_id": int(task_idx),
            "task": dataset.tasks[task_idx],
            "available_windows": int(count),
            "sampled_windows": int(per_task_counts[int(task_idx)]),
        }
        for task_idx, count, _base in task_rows
    ]
    return indices, {
        "mode": "balanced_task_round_robin_with_replacement",
        "seed": int(seed),
        "requested_batches": int(num_batches),
        "batch_size": int(batch_size),
        "sample_count": int(len(indices)),
        "num_tasks_sampled": int(len(task_rows)),
        "min_samples_per_task": int(min(per_task_counts.values())),
        "max_samples_per_task": int(max(per_task_counts.values())),
        "tasks": task_counts,
    }


def build_source_eval_loaders(config: NativeImaginationConfig) -> dict[str, tuple[DataLoader, list[str], NativeImaginationConfig]]:
    loaders: dict[str, tuple[DataLoader, list[str], NativeImaginationConfig]] = {}
    for source in parse_source_names(config.source_eval_sources, default=()):
        if source == "all":
            continue
        data_dirs, frame_dirs = select_source_paths(config, source)
        if not data_dirs:
            continue
        source_config = NativeImaginationConfig(**asdict(config))
        source_config.data_dirs = data_dirs
        source_config.frame_dirs = frame_dirs
        if int(source_config.source_eval_batches) > 0:
            source_config.eval_batches = int(source_config.source_eval_batches)
        train_shadow = WMDataset(
            data_dir=source_config.data_dirs,
            frames_dir=source_config.frame_dirs,
            seq_len=source_config.seq_len,
            img_size=128,
            action_dim=source_config.action_dim,
            raw_action_dim=source_config.raw_action_dim,
            tasks_json=source_config.tasks_json,
            tasks=None,
            strict_tasks=False,
            action_features=source_config.action_features,
            verbose=False,
        )
        eval_dataset = WMDataset(
            data_dir=source_config.data_dirs,
            frames_dir=source_config.frame_dirs,
            seq_len=source_config.seq_len,
            img_size=128,
            action_dim=source_config.action_dim,
            raw_action_dim=source_config.raw_action_dim,
            tasks_json=source_config.tasks_json,
            tasks=None,
            strict_tasks=False,
            action_features=source_config.action_features,
            require_non_noop=source_config.require_non_noop,
            no_op_threshold=source_config.no_op_threshold,
            min_non_noop_steps=source_config.min_non_noop_steps,
            reward_filter_mode=source_config.reward_filter_mode,
            reward_signal_threshold=source_config.reward_signal_threshold,
            min_reward_signal_steps=source_config.min_reward_signal_steps,
            verbose=False,
        )
        apply_episode_holdout_split(
            train_shadow,
            eval_dataset,
            holdout_fraction=source_config.eval_holdout_fraction,
            seed=source_config.split_seed,
        )
        eval_indices, _eval_sampling_info = build_balanced_eval_indices(
            eval_dataset,
            num_batches=source_config.eval_batches,
            batch_size=source_config.batch_size,
            seed=source_config.eval_seed if source_config.eval_seed else source_config.seed + 15485863,
        )
        loader = DataLoader(
            Subset(eval_dataset, eval_indices),
            batch_size=source_config.batch_size,
            shuffle=False,
            num_workers=source_config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_batch,
            persistent_workers=source_config.num_workers > 0,
        )
        loaders[source] = (loader, eval_dataset.tasks, source_config)
    return loaders


def select_source_paths(config: NativeImaginationConfig, source: str) -> tuple[list[str], list[str]]:
    source = source.lower()
    if source == "all":
        return list(config.data_dirs), list(config.frame_dirs)
    selected_data: list[str] = []
    selected_frames: list[str] = []
    for data_dir, frame_dir in zip(config.data_dirs, config.frame_dirs):
        text = f"{data_dir} {frame_dir}".lower()
        if source == "soar" and "robotics/soar" in text:
            selected_data.append(data_dir)
            selected_frames.append(frame_dir)
        elif source == "droid" and ("droid" in text or "hf_action_exports" in text):
            selected_data.append(data_dir)
            selected_frames.append(frame_dir)
    return selected_data, selected_frames


def attach_source_eval(
    *,
    eval_payload: dict[str, Any],
    agent: AgentHeads,
    encoder: Any,
    dynamics: Dynamics,
    source_eval_loaders: dict[str, tuple[DataLoader, list[str], NativeImaginationConfig]],
    tok_args: dict[str, Any],
    dyn_args: dict[str, Any],
    config: NativeImaginationConfig,
    device: torch.device,
) -> dict[str, Any]:
    sources = parse_source_names(config.source_eval_sources, default=())
    if not sources:
        return eval_payload
    payload = dict(eval_payload)
    source_eval: dict[str, Any] = {}
    for source in sources:
        if source == "all":
            source_eval[source] = strip_large_eval(eval_payload)
            continue
        bundle = source_eval_loaders.get(source)
        if bundle is None:
            continue
        loader, task_names, source_config = bundle
        set_eval_seed(source_config)
        source_eval[source] = strip_large_eval(
            evaluate_policies(
                agent=agent,
                encoder=encoder,
                dynamics=dynamics,
                loader=loader,
                tok_args=tok_args,
                dyn_args=dyn_args,
                config=source_config,
                device=device,
                task_names=task_names,
            )
        )
    payload["source_eval"] = source_eval
    return payload


def parse_train_balance_spec(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for part in str(spec).replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid train balance spec item '{item}', expected bucket=weight.")
        name, value = item.split("=", 1)
        name = name.strip()
        weight = float(value)
        if weight < 0:
            raise ValueError(f"Train balance weight for {name} must be non-negative, got {weight}.")
        if weight > 0:
            weights[name] = weight
    if not weights:
        raise ValueError("train_balance_spec produced no positive-weight buckets.")
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def classify_demo_source(path: str) -> str:
    text = str(path)
    if "nicklashansen_dreamer4/expert" in text or "/expert/" in text:
        return "hf_expert"
    if "nicklashansen_dreamer4/mixed-small" in text or "nicklashansen_dreamer4/mixed-large" in text:
        return "hf_mixed"
    if "/mixed-small/" in text or "/mixed-large/" in text:
        return "hf_mixed"
    if "robotics/hf_action_exports" in text or "droid_lerobot" in text:
        return "hf_robot"
    if "robotics/soar" in text or "game_action_sources" in text:
        return "soar_game"
    if "robotics/robonet" in text:
        return "robonet"
    return "other"


def reward_window_stats_for_starts(dataset: WMDataset, task_idx: int, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rew = torch.nan_to_num(dataset.rew[task_idx].float(), nan=0.0, posinf=0.0, neginf=0.0)
    transitions = rew[1:]
    if starts.numel() == 0:
        return starts.new_zeros((0,), dtype=torch.float32), starts.new_zeros((0,), dtype=torch.int32)
    start_count = int(rew.shape[0]) - int(dataset.T)
    if start_count <= 0:
        return starts.new_zeros((starts.numel(),), dtype=torch.float32), starts.new_zeros((starts.numel(),), dtype=torch.int32)
    end = starts.to(torch.long) + (int(dataset.T) - 1)
    prev = starts.to(torch.long) - 1
    prev_mask = prev >= 0

    cs_reward = torch.cumsum(transitions, dim=0)
    prev_reward = torch.zeros_like(starts, dtype=cs_reward.dtype)
    prev_reward[prev_mask] = cs_reward[prev[prev_mask]]
    reward_sum = cs_reward[end] - prev_reward

    positive = transitions > 0.0
    cs_pos = torch.cumsum(positive.to(torch.int32), dim=0)
    prev_pos = torch.zeros_like(starts, dtype=cs_pos.dtype)
    prev_pos[prev_mask] = cs_pos[prev[prev_mask]]
    positive_count = cs_pos[end] - prev_pos
    return reward_sum, positive_count


def action_window_stats_for_starts(
    dataset: WMDataset,
    task_idx: int,
    starts: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    raw_dim = int(dataset._raw_act_dims[task_idx])
    if raw_dim <= 0 or starts.numel() == 0:
        return starts.new_zeros((int(starts.numel()),), dtype=torch.int32)
    act = torch.nan_to_num(dataset.act[task_idx].float(), nan=0.0, posinf=0.0, neginf=0.0)
    transitions = act[1:, :raw_dim]
    if transitions.numel() == 0:
        return starts.new_zeros((int(starts.numel()),), dtype=torch.int32)
    active = transitions.norm(dim=-1) > float(threshold)
    cs_active = torch.cumsum(active.to(torch.int32), dim=0)
    end = starts.to(torch.long) + (int(dataset.T) - 1)
    prev = starts.to(torch.long) - 1
    prev_mask = prev >= 0
    prev_active = torch.zeros_like(starts, dtype=cs_active.dtype)
    prev_active[prev_mask] = cs_active[prev[prev_mask]]
    return cs_active[end] - prev_active


def candidate_train_buckets(
    *,
    source: str,
    ret: float,
    has_positive: bool,
    active_count: int,
    config: NativeImaginationConfig,
) -> set[str]:
    threshold = float(config.train_balance_return_threshold)
    min_active = max(1, int(config.train_min_action_active_steps))
    is_positive = ret > threshold or has_positive
    is_zero = abs(ret) <= threshold and not has_positive
    is_active = active_count >= min_active
    buckets: set[str] = set()
    sources = {source, "all"}
    for prefix in sources:
        if is_positive:
            buckets.add(f"{prefix}_positive")
        if is_zero:
            buckets.add(f"{prefix}_zero")
        if is_active:
            buckets.add(f"{prefix}_active")
        if is_positive and is_active:
            buckets.add(f"{prefix}_positive_active")
    # Backward-compatible names used by the existing launchers.
    if source == "soar_game" and has_positive:
        buckets.add("soar_game_positive")
    return buckets


def build_dreamer4_reward_mixture_indices(
    dataset: WMDataset,
    *,
    config: NativeImaginationConfig,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    requested_weights = parse_train_balance_spec(config.train_balance_spec)
    threshold = float(config.train_balance_return_threshold)
    bucket_task_indices: dict[str, dict[str, list[int]]] = {name: {} for name in requested_weights}
    bucket_available = {name: 0 for name in requested_weights}

    prev_cum = [0] + [int(x) for x in dataset._cum_counts[:-1]]
    for task_idx, starts in enumerate(dataset.valid_starts):
        if starts.numel() == 0:
            continue
        starts = starts.cpu().to(torch.long)
        reward_sum, positive_count = reward_window_stats_for_starts(dataset, task_idx, starts)
        active_count = action_window_stats_for_starts(
            dataset,
            task_idx,
            starts,
            threshold=float(config.train_action_active_threshold),
        ).cpu()
        reward_sum = reward_sum.cpu()
        positive_count = positive_count.cpu()
        base = int(prev_cum[task_idx])
        seg_cum = [int(x) for x in dataset.seg_cum_frames[task_idx]]
        demo_paths = dataset.demo_paths[task_idx]
        task_name = str(dataset.tasks[task_idx])

        for local_idx, start in enumerate(starts.tolist()):
            seg_idx = bisect.bisect_right(seg_cum, int(start))
            if seg_idx >= len(demo_paths):
                continue
            source = classify_demo_source(str(demo_paths[seg_idx]))
            ret = float(reward_sum[local_idx].item())
            has_positive = int(positive_count[local_idx].item()) > 0
            global_idx = base + int(local_idx)
            matched = candidate_train_buckets(
                source=source,
                ret=ret,
                has_positive=has_positive,
                active_count=int(active_count[local_idx].item()),
                config=config,
            )
            for bucket in matched:
                if bucket not in bucket_task_indices:
                    continue
                by_task = bucket_task_indices[bucket].setdefault(task_name, [])
                by_task.append(global_idx)
                bucket_available[bucket] += 1

    active_weights = {name: weight for name, weight in requested_weights.items() if bucket_available.get(name, 0) > 0}
    if not active_weights:
        raise ValueError(
            "Balanced train sampler found no non-empty buckets. "
            f"requested={requested_weights}, available={bucket_available}"
        )
    active_total = sum(active_weights.values())
    active_weights = {name: weight / active_total for name, weight in active_weights.items()}

    sample_count = int(config.train_balanced_samples)
    if sample_count <= 0:
        sample_count = max(1024, (int(config.bc_steps) + int(config.imagination_updates)) * int(config.batch_size))

    generator = torch.Generator().manual_seed(int(seed))
    bucket_names = list(active_weights.keys())
    bucket_weights = torch.tensor([active_weights[name] for name in bucket_names], dtype=torch.float32)
    sampled_buckets = torch.multinomial(bucket_weights, sample_count, replacement=True, generator=generator)
    indices: list[int] = []
    sampled_counts = {name: 0 for name in requested_weights}
    sampled_task_counts: dict[str, dict[str, int]] = {name: {} for name in requested_weights}

    for bucket_id in sampled_buckets.tolist():
        bucket = bucket_names[int(bucket_id)]
        tasks = sorted(bucket_task_indices[bucket].keys())
        task_id = int(torch.randint(len(tasks), (1,), generator=generator).item())
        task = tasks[task_id]
        choices = bucket_task_indices[bucket][task]
        choice_id = int(torch.randint(len(choices), (1,), generator=generator).item())
        indices.append(int(choices[choice_id]))
        sampled_counts[bucket] += 1
        sampled_task_counts[bucket][task] = sampled_task_counts[bucket].get(task, 0) + 1

    bucket_rows = []
    for bucket in requested_weights:
        task_counts = sampled_task_counts.get(bucket, {})
        bucket_rows.append(
            {
                "bucket": bucket,
                "requested_weight": float(requested_weights.get(bucket, 0.0)),
                "active_weight": float(active_weights.get(bucket, 0.0)),
                "available_windows": int(bucket_available.get(bucket, 0)),
                "available_tasks": int(len(bucket_task_indices.get(bucket, {}))),
                "sampled_windows": int(sampled_counts.get(bucket, 0)),
                "sampled_tasks": int(len(task_counts)),
                "top_sampled_tasks": [
                    {"task": task, "sampled_windows": int(count)}
                    for task, count in sorted(task_counts.items(), key=lambda item: item[1], reverse=True)[:20]
                ],
            }
        )

    return indices, {
        "mode": "dreamer4_reward_mixture",
        "seed": int(seed),
        "sample_count": int(len(indices)),
        "source_dataset_windows": int(len(dataset)),
        "return_threshold": float(threshold),
        "requested_weights": requested_weights,
        "active_weights": active_weights,
        "buckets": bucket_rows,
    }


def train_supervised_heads(
    *,
    agent: AgentHeads,
    encoder: Any,
    dynamics: Dynamics,
    batches: Iterable[dict[str, torch.Tensor]],
    tok_args: dict[str, Any],
    dyn_args: dict[str, Any],
    config: NativeImaginationConfig,
    device: torch.device,
    target_stats: dict[str, torch.Tensor],
) -> list[dict[str, float]]:
    opt = torch.optim.AdamW(agent.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    agent.train()
    for step in range(1, config.bc_steps + 1):
        batch = prepare_batch(next(batches), encoder, tok_args, dyn_args, config, device)
        policy_features, reward_value_features, target_actions, masks, rewards, values = supervised_targets(
            batch, config, target_stats
        )
        prior = agent.prior_mean(policy_features)
        policy = agent.policy_mean(policy_features)
        reward_pred = agent.reward_pred(reward_value_features)
        value_pred = agent.value_pred(reward_value_features)
        action_loss = masked_mse(prior, target_actions, masks) + masked_mse(policy, target_actions, masks)
        reward_loss = F.mse_loss(reward_pred, rewards)
        value_loss = F.mse_loss(value_pred, values)
        aux_inverse_loss, aux_effect_loss, aux_metrics = auxiliary_inverse_effect_losses(
            agent=agent,
            batch=batch,
            config=config,
        )
        reward_contrast = torch.zeros((), device=device, dtype=reward_loss.dtype)
        reward_contrast_metrics: dict[str, float] = {}
        if (
            config.reward_contrast_weight > 0.0
            and step >= int(config.reward_contrast_start)
            and (step % max(1, int(config.reward_contrast_every)) == 0)
        ):
            reward_contrast, reward_contrast_metrics = reward_counterfactual_loss(
                agent=agent,
                dynamics=dynamics,
                batch=batch,
                dyn_args=dyn_args,
                config=config,
            )
        loss = (
            action_loss
            + config.reward_loss_weight * reward_loss
            + config.value_loss_weight * value_loss
            + config.reward_contrast_weight * reward_contrast
            + config.aux_inverse_weight * aux_inverse_loss
            + config.aux_effect_weight * aux_effect_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 5.0)
        opt.step()
        if step == 1 or step % max(1, config.bc_steps // 10) == 0 or step == config.bc_steps:
            item = {
                "step": step,
                "loss": float(loss.item()),
                "action_mse": float((action_loss / 2.0).item()),
                "reward_mse": float(reward_loss.item()),
                "value_mse": float(value_loss.item()),
                "reward_contrast": float(reward_contrast.detach().item()),
                "aux_inverse_mse": float(aux_inverse_loss.detach().item()),
                "aux_effect_mse": float(aux_effect_loss.detach().item()),
                **reward_contrast_metrics,
                **aux_metrics,
            }
            history.append(item)
            print(json.dumps({"phase": "bc", **item}), flush=True)
    return history


def run_imagination_training(
    *,
    agent: AgentHeads,
    encoder: Any,
    dynamics: Dynamics,
    batches: Iterable[dict[str, torch.Tensor]],
    eval_loader: Any,
    source_eval_loaders: dict[str, tuple[DataLoader, list[str], NativeImaginationConfig]],
    initial_eval: dict[str, Any],
    tok_args: dict[str, Any],
    dyn_args: dict[str, Any],
    config: NativeImaginationConfig,
    device: torch.device,
    target_stats: dict[str, torch.Tensor],
    task_names: list[str] | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any] | None]:
    freeze_module(agent.prior)
    freeze_module(agent.reward)
    params = list(agent.policy.parameters()) + [agent.log_std]
    if not config.freeze_value_during_imagination:
        params += list(agent.value.parameters())
    opt = torch.optim.AdamW(params, lr=config.imagination_learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    best_selection: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -float("inf")
    if config.select_best_imagination:
        initial_eval = attach_source_eval(
            eval_payload=initial_eval,
            agent=agent,
            encoder=encoder,
            dynamics=dynamics,
            source_eval_loaders=source_eval_loaders,
            tok_args=tok_args,
            dyn_args=dyn_args,
            config=config,
            device=device,
        )
        initial_metric = imagination_metric_value(initial_eval, config)
        initial_eligible = int(config.min_imagination_selection_update) <= 0
        if initial_eligible:
            best_metric = initial_metric
            best_state = state_dict_cpu(agent)
        best_selection = {
            "enabled": True,
            "selected_update": 0 if initial_eligible else None,
            "min_selection_update": int(config.min_imagination_selection_update),
            "metric": config.best_imagination_metric,
            "metric_value": initial_metric if initial_eligible else None,
            "eval": strip_large_eval(initial_eval) if initial_eligible else None,
            "history": [
                {
                    "update": 0,
                    "eligible": initial_eligible,
                    "metric_value": initial_metric,
                    "source_gate_pass": bool(initial_metric > -1e5),
                    "policy": float(initial_eval.get("policy", 0.0)),
                    "policy_minus_bc": float(initial_eval.get("policy_minus_bc", 0.0)),
                    "policy_minus_zero": float(initial_eval.get("policy_minus_zero", 0.0)),
                    "causal_policy_gain": float(initial_eval.get("causal_policy_gain", 0.0)),
                    "policy_minus_dyn_zero": float(initial_eval.get("policy_minus_dyn_zero", 0.0)),
                    "policy_minus_dyn_shuffle": float(initial_eval.get("policy_minus_dyn_shuffle", 0.0)),
                    "source_eval": {
                        name: {
                            "policy_minus_bc": float(payload.get("policy_minus_bc", 0.0)),
                            "policy_minus_zero": float(payload.get("policy_minus_zero", 0.0)),
                            "policy_minus_dyn_zero": float(payload.get("policy_minus_dyn_zero", 0.0)),
                            "policy_minus_dyn_shuffle": float(payload.get("policy_minus_dyn_shuffle", 0.0)),
                        }
                        for name, payload in initial_eval.get("source_eval", {}).items()
                    },
                }
            ],
        }
    agent.train()
    for update in range(1, config.imagination_updates + 1):
        batch = prepare_batch(next(batches), encoder, tok_args, dyn_args, config, device)
        metrics = imagination_update(agent, dynamics, batch, dyn_args, config)
        opt.zero_grad(set_to_none=True)
        metrics["loss_tensor"].backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        item = {key: float(value) for key, value in metrics.items() if key != "loss_tensor"}
        item["update"] = update
        if update == 1 or update % max(1, config.imagination_updates // 10) == 0 or update == config.imagination_updates:
            history.append(item)
            print(json.dumps({"phase": "imagination", **item}), flush=True)
        if (
            config.select_best_imagination
            and config.imagination_eval_every > 0
            and (update % config.imagination_eval_every == 0 or update == config.imagination_updates)
        ):
            set_eval_seed(config)
            eval_payload = evaluate_policies(
                agent=agent,
                encoder=encoder,
                dynamics=dynamics,
                loader=eval_loader,
                tok_args=tok_args,
                dyn_args=dyn_args,
                config=config,
                device=device,
                task_names=task_names,
            )
            eval_payload = attach_source_eval(
                eval_payload=eval_payload,
                agent=agent,
                encoder=encoder,
                dynamics=dynamics,
                source_eval_loaders=source_eval_loaders,
                tok_args=tok_args,
                dyn_args=dyn_args,
                config=config,
                device=device,
            )
            metric_value = imagination_metric_value(eval_payload, config)
            eligible = int(update) >= int(config.min_imagination_selection_update)
            source_eval = eval_payload.get("source_eval", {})
            source_gate_pass = metric_value > -1e5
            eval_row = {
                "update": int(update),
                "eligible": bool(eligible),
                "metric_value": metric_value,
                "source_gate_pass": bool(source_gate_pass),
                "policy": float(eval_payload.get("policy", 0.0)),
                "policy_minus_bc": float(eval_payload.get("policy_minus_bc", 0.0)),
                "policy_minus_zero": float(eval_payload.get("policy_minus_zero", 0.0)),
                "causal_policy_gain": float(eval_payload.get("causal_policy_gain", 0.0)),
                "policy_minus_dyn_zero": float(eval_payload.get("policy_minus_dyn_zero", 0.0)),
                "policy_minus_dyn_shuffle": float(eval_payload.get("policy_minus_dyn_shuffle", 0.0)),
                "source_eval": {
                    name: {
                        "policy_minus_bc": float(payload.get("policy_minus_bc", 0.0)),
                        "policy_minus_zero": float(payload.get("policy_minus_zero", 0.0)),
                        "policy_minus_dyn_zero": float(payload.get("policy_minus_dyn_zero", 0.0)),
                        "policy_minus_dyn_shuffle": float(payload.get("policy_minus_dyn_shuffle", 0.0)),
                    }
                    for name, payload in source_eval.items()
                },
            }
            assert best_selection is not None
            best_selection["history"].append(eval_row)
            print(json.dumps({"phase": "imagination_eval", **eval_row}), flush=True)
            if eligible and metric_value > best_metric:
                best_metric = metric_value
                best_state = state_dict_cpu(agent)
                best_selection.update(
                    {
                        "selected_update": int(update),
                        "metric_value": metric_value,
                        "eval": strip_large_eval(eval_payload),
                    }
                )
            agent.train()
    if best_state is not None:
        agent.load_state_dict(best_state, strict=True)
        if best_selection is not None:
            print(json.dumps({"phase": "imagination_best_selected", **{k: v for k, v in best_selection.items() if k != "eval" and k != "history"}}), flush=True)
    unfreeze_module(agent.prior)
    unfreeze_module(agent.reward)
    return history, best_selection


def auxiliary_inverse_effect_losses(
    *,
    agent: AgentHeads,
    batch: dict[str, torch.Tensor],
    config: NativeImaginationConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    z = batch["z"]
    if z.shape[1] < 2:
        zero = z.new_zeros(())
        return zero, zero, {"aux_action_effect_active_fraction": 0.0}

    z_flat = z.reshape(z.shape[0], z.shape[1], -1)
    z0 = z_flat[:, :-1]
    z1 = z_flat[:, 1:]
    action = policy_target_actions(batch, config)[:, :-1]
    mask = policy_target_masks(batch, config)[:, :-1]
    active = mask.float().sum(dim=-1) > 0
    if config.aux_action_effect_min_norm > 0:
        denom = mask.float().sum(dim=-1).clamp_min(1.0)
        action_norm = ((action.float() * mask.float()).pow(2).sum(dim=-1) / denom).sqrt()
        active &= action_norm > float(config.aux_action_effect_min_norm)

    if not active.any():
        zero = z.new_zeros(())
        return zero, zero, {"aux_action_effect_active_fraction": 0.0}

    action_dim = policy_single_action_dim(config)
    z0_f = z0.reshape(-1, z0.shape[-1])
    z1_f = z1.reshape(-1, z1.shape[-1])
    action_f = action.reshape(-1, action_dim)
    mask_f = mask.reshape(-1, action_dim)
    active_f = active.reshape(-1)

    inv_input = torch.cat([z0_f, z1_f], dim=-1)
    inv_pred = agent.inverse_pred(inv_input)
    inv_per = masked_mse(inv_pred, action_f, mask_f, reduce=False)
    inverse_loss = inv_per[active_f].mean()

    effect_input = torch.cat([z0_f, action_f * mask_f], dim=-1)
    effect_target = z1_f - z0_f
    effect_pred = agent.effect_pred(effect_input)
    effect_per = (effect_pred.float() - effect_target.float()).pow(2).mean(dim=-1)
    effect_loss = effect_per[active_f].mean()

    return inverse_loss, effect_loss, {
        "aux_action_effect_active_fraction": float(active.float().mean().detach().item()),
    }


def reward_counterfactual_loss(
    *,
    agent: AgentHeads,
    dynamics: Dynamics,
    batch: dict[str, torch.Tensor],
    dyn_args: dict[str, Any],
    config: NativeImaginationConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    action_index = min(int(config.ctx_len), batch["z"].shape[1] - 1)
    horizon = min(max(1, int(config.reward_contrast_horizon)), batch["z"].shape[1] - action_index)
    z_seq = batch["z"][:, : config.ctx_len].detach()
    dyn_action_seq = batch["actions"][:, : config.ctx_len].detach()
    dyn_mask_seq = batch["mask"][:, : config.ctx_len].detach()
    policy_action_seq = policy_context_actions(batch, config)[:, : config.ctx_len].detach()
    target_actions = policy_target_actions(batch, config)
    target_masks = policy_target_masks(batch, config)

    true_actions = target_actions[:, action_index : action_index + horizon].detach()
    true_masks = target_masks[:, action_index : action_index + horizon].detach()
    positive = (
        batch["rewards"][:, action_index : action_index + horizon].sum(dim=1)
        > float(config.reward_contrast_positive_threshold)
    )
    if config.reward_contrast_min_action_norm > 0:
        denom = true_masks.sum(dim=(1, 2)).clamp_min(1.0)
        action_norm = ((true_actions * true_masks).pow(2).sum(dim=(1, 2)) / denom).sqrt()
        positive &= action_norm > float(config.reward_contrast_min_action_norm)
    if not positive.any():
        zero = batch["z"].new_zeros(())
        return zero, {
            "reward_contrast_true_minus_zero": 0.0,
            "reward_contrast_true_minus_shuffle": 0.0,
            "reward_contrast_positive_fraction": 0.0,
        }

    true_reward = reward_for_counterfactual_actions(
        agent=agent,
        dynamics=dynamics,
        z_seq=z_seq,
        dyn_action_seq=dyn_action_seq,
        dyn_mask_seq=dyn_mask_seq,
        policy_action_seq=policy_action_seq,
        actions=true_actions,
        action_masks=true_masks,
        dyn_args=dyn_args,
        config=config,
    )

    modes = [mode.strip() for mode in str(config.reward_contrast_negative_modes).replace("+", ",").split(",") if mode.strip()]
    modes = modes or ["zero", "shuffle"]
    losses = []
    metrics: dict[str, float] = {"reward_contrast_positive_fraction": float(positive.float().mean().detach().item())}
    for mode in modes:
        if mode in {"zero", "noop"}:
            wrong_actions = torch.zeros_like(true_actions)
            wrong_masks = true_masks
            metric_key = "reward_contrast_true_minus_zero"
        elif mode in {"shuffle", "batch"}:
            if true_actions.shape[0] <= 1:
                continue
            perm = torch.randperm(true_actions.shape[0], device=true_actions.device)
            wrong_actions = true_actions[perm]
            wrong_masks = true_masks[perm]
            metric_key = "reward_contrast_true_minus_shuffle"
        else:
            raise ValueError(f"unknown reward contrast negative mode: {mode}")

        wrong_reward = reward_for_counterfactual_actions(
            agent=agent,
            dynamics=dynamics,
            z_seq=z_seq,
            dyn_action_seq=dyn_action_seq,
            dyn_mask_seq=dyn_mask_seq,
            policy_action_seq=policy_action_seq,
            actions=wrong_actions,
            action_masks=wrong_masks,
            dyn_args=dyn_args,
            config=config,
        )
        margin = true_reward - wrong_reward
        metrics[metric_key] = float(margin[positive].mean().detach().item())
        losses.append(torch.relu(float(config.reward_contrast_margin) - margin)[positive].mean())

    for key in ["reward_contrast_true_minus_zero", "reward_contrast_true_minus_shuffle"]:
        metrics.setdefault(key, 0.0)
    if not losses:
        return batch["z"].new_zeros(()), metrics
    return sum(losses) / len(losses), metrics


def reward_for_counterfactual_actions(
    *,
    agent: AgentHeads,
    dynamics: Dynamics,
    z_seq: torch.Tensor,
    dyn_action_seq: torch.Tensor,
    dyn_mask_seq: torch.Tensor,
    policy_action_seq: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    dyn_args: dict[str, Any],
    config: NativeImaginationConfig,
) -> torch.Tensor:
    rewards = []
    discounts = []
    for t in range(actions.shape[1]):
        action = actions[:, t]
        action_mask = action_masks[:, t]
        if config.policy_action_source == "raw":
            dyn_action, dyn_action_mask = expand_raw_action_step(policy_action_seq, action, action_mask, config)
        else:
            dyn_action, dyn_action_mask = action, action_mask
        next_z = imagine_next(
            dynamics,
            z_seq,
            dyn_action_seq,
            dyn_mask_seq,
            dyn_action,
            dyn_action_mask,
            dyn_args,
            config,
        )
        next_z_seq = torch.cat([z_seq, next_z[:, None, :, :]], dim=1)[:, -config.ctx_len :]
        next_dyn_actions = torch.cat([dyn_action_seq, dyn_action[:, None, :]], dim=1)[:, -config.ctx_len :]
        next_dyn_masks = torch.cat([dyn_mask_seq, dyn_action_mask[:, None, :]], dim=1)[:, -config.ctx_len :]
        next_policy_actions = torch.cat([policy_action_seq, action[:, None, :]], dim=1)[:, -config.ctx_len :]
        rewards.append(agent.reward_pred(reward_value_features(next_z_seq, next_policy_actions, config)))
        discounts.append(float(config.gamma) ** t)
        z_seq = next_z_seq.detach()
        dyn_action_seq, dyn_mask_seq = next_dyn_actions.detach(), next_dyn_masks.detach()
        policy_action_seq = next_policy_actions.detach()
    discount_t = torch.tensor(discounts, device=actions.device, dtype=actions.dtype).view(-1, 1)
    return (torch.stack(rewards, dim=0) * discount_t).sum(dim=0)


def imagination_update(
    agent: AgentHeads,
    dynamics: Dynamics,
    batch: dict[str, torch.Tensor],
    dyn_args: dict[str, Any],
    config: NativeImaginationConfig,
) -> dict[str, Any]:
    z_seq = batch["z"][:, : config.ctx_len].detach()
    dyn_action_seq = batch["actions"][:, : config.ctx_len].detach()
    dyn_mask_seq = batch["mask"][:, : config.ctx_len].detach()
    policy_action_seq = policy_context_actions(batch, config)[:, : config.ctx_len].detach()
    policy_mask_seq = policy_context_masks(batch, config)[:, : config.ctx_len].detach()
    policy_action_seq, policy_mask_seq = corrupt_imagination_agent_action_context(
        policy_action_seq,
        policy_mask_seq,
        config,
    )
    current_mask = policy_mask_seq[:, -1]

    log_probs = []
    values = []
    rewards = []
    prior_mses = []
    mean_prior_mses = []
    entropies = []
    action_mags = []
    sampled_actions = []
    sampled_masks = []
    for _ in range(config.imagination_horizon):
        features = state_features(z_seq, policy_action_seq, max_context=config.ctx_len)
        value_features = reward_value_features(z_seq, policy_action_seq, config)
        dist = agent.action_dist(features)
        raw_action_flat = dist.rsample()
        action_flat = torch.tanh(raw_action_flat)
        action, action_mask = first_chunk_action(action_flat, current_mask, config)
        # PMPO is a score-function update. Detach the sampled action so log_prob
        # trains the policy mean toward high-advantage samples instead of canceling
        # through the reparameterized sample path. Keep the old path available as
        # an explicit ablation.
        log_prob_action = raw_action_flat.detach() if config.detach_policy_log_prob else raw_action_flat
        log_prob = first_chunk_log_prob(dist.log_prob(log_prob_action), action_mask, config)
        entropy = first_chunk_log_prob(dist.entropy(), action_mask, config)
        with torch.no_grad():
            prior_mean_flat = agent.prior_mean(features)
            prior_mean, _ = first_chunk_action(prior_mean_flat, current_mask, config)
        policy_mean_flat = agent.policy_mean(features)
        policy_mean, _ = first_chunk_action(policy_mean_flat, current_mask, config)
        prior_mse = masked_mse(action, prior_mean, current_mask, reduce=False)
        mean_prior_mse = masked_mse(policy_mean, prior_mean, current_mask, reduce=False)
        if config.policy_action_source == "raw":
            dyn_action, dyn_action_mask = expand_raw_action_step(policy_action_seq, action, current_mask, config)
        else:
            dyn_action, dyn_action_mask = action, current_mask
        dyn_action_for_model, dyn_action_mask_for_model = corrupt_imagination_dynamics_action(
            dyn_action,
            dyn_action_mask,
            config,
        )
        next_z = imagine_next(
            dynamics,
            z_seq,
            dyn_action_seq,
            dyn_mask_seq,
            dyn_action_for_model,
            dyn_action_mask_for_model,
            dyn_args,
            config,
        )
        next_dyn_actions = torch.cat([dyn_action_seq, dyn_action_for_model[:, None, :]], dim=1)[:, -config.ctx_len :]
        next_dyn_masks = torch.cat([dyn_mask_seq, dyn_action_mask_for_model[:, None, :]], dim=1)[:, -config.ctx_len :]
        agent_context_action, agent_context_mask = corrupt_imagination_agent_action_context(
            action,
            current_mask,
            config,
        )
        next_policy_actions = torch.cat([policy_action_seq, agent_context_action[:, None, :]], dim=1)[
            :, -config.ctx_len :
        ]
        next_policy_masks = torch.cat([policy_mask_seq, agent_context_mask[:, None, :]], dim=1)[:, -config.ctx_len :]
        next_z_seq = torch.cat([z_seq, next_z[:, None, :, :]], dim=1)[:, -config.ctx_len :]
        next_reward_features = reward_value_features(next_z_seq, next_policy_actions, config)
        with torch.no_grad():
            reward = agent.reward_pred(next_reward_features)
        value = agent.value_pred(value_features)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        prior_mses.append(prior_mse)
        mean_prior_mses.append(mean_prior_mse)
        entropies.append(entropy)
        action_mags.append((action.abs() * current_mask).sum(dim=-1) / current_mask.sum(dim=-1).clamp_min(1.0))
        sampled_actions.append(action.detach())
        sampled_masks.append(current_mask.detach())
        z_seq = next_z_seq.detach()
        dyn_action_seq, dyn_mask_seq = next_dyn_actions.detach(), next_dyn_masks.detach()
        policy_action_seq, policy_mask_seq = next_policy_actions.detach(), next_policy_masks.detach()

    values_t = torch.stack(values, dim=0)
    rewards_t = torch.stack(rewards, dim=0)
    log_probs_t = torch.stack(log_probs, dim=0)
    prior_mse_t = torch.stack(prior_mses, dim=0)
    mean_prior_mse_t = torch.stack(mean_prior_mses, dim=0)
    entropy_t = torch.stack(entropies, dim=0)
    returns_t = lambda_returns(rewards_t, values_t.detach(), config.gamma)
    causal_advantages = None
    causal_margin_mean = torch.zeros((), device=values_t.device, dtype=values_t.dtype)
    causal_positive_fraction = torch.zeros((), device=values_t.device, dtype=values_t.dtype)
    causal_shortfall_policy_loss = torch.zeros((), device=values_t.device, dtype=values_t.dtype)
    causal_shortfall_fraction = torch.zeros((), device=values_t.device, dtype=values_t.dtype)
    if config.causal_policy_mode != "off":
        with torch.no_grad():
            discounts = torch.tensor(
                [config.gamma**i for i in range(rewards_t.shape[0])],
                device=rewards_t.device,
                dtype=rewards_t.dtype,
            ).view(-1, 1)
            real_returns = (rewards_t.detach() * discounts).sum(dim=0)
            actions_t = torch.stack(sampled_actions, dim=1)
            masks_t = torch.stack(sampled_masks, dim=1)
            cf_returns = []
            for mode in parse_modes(config.causal_policy_negative_modes, default=("zero", "shuffle")):
                if mode not in {"zero", "noop", "shuffle", "batch"}:
                    raise ValueError(f"unknown causal policy negative mode: {mode}")
                dyn_mode = "zero" if mode in {"zero", "noop"} else "shuffle"
                cf_rewards = rollout_rewards_for_action_sequence(
                    agent=agent,
                    dynamics=dynamics,
                    batch=batch,
                    dyn_args=dyn_args,
                    config=config,
                    actions=actions_t,
                    action_masks=masks_t,
                    dynamics_action_mode=dyn_mode,
                )
                cf_returns.append((cf_rewards * discounts).sum(dim=0))
            if cf_returns:
                counterfactual_returns = torch.stack(cf_returns, dim=0).max(dim=0).values
                causal_traj_advantage = real_returns - counterfactual_returns
                causal_advantages = causal_traj_advantage.view(1, -1).expand_as(rewards_t)
                causal_margin_mean = causal_traj_advantage.mean()
                causal_positive_fraction = (causal_traj_advantage > float(config.causal_policy_min_margin)).float().mean()
        if config.causal_shortfall_policy_weight > 0.0 and causal_advantages is not None:
            shortfall_margin = (
                float(config.causal_shortfall_margin)
                if float(config.causal_shortfall_margin) >= 0.0
                else float(config.causal_policy_min_margin)
            )
            causal_shortfall = torch.relu(shortfall_margin - causal_advantages.detach())
            causal_shortfall_fraction = (causal_shortfall > 0).float().mean()
            if causal_shortfall.sum() > 0:
                # Positive log-prob loss lowers probability of samples that fail counterfactual controls.
                causal_shortfall_policy_loss = (log_probs_t * causal_shortfall).sum() / causal_shortfall.sum().clamp_min(1e-6)
    if config.advantage_baseline == "bc_return":
        with torch.no_grad():
            bc_returns = rollout_return(agent, dynamics, batch, dyn_args, config, mode="bc_prior")
            discounts = torch.tensor(
                [config.gamma**i for i in range(rewards_t.shape[0])],
                device=rewards_t.device,
                dtype=rewards_t.dtype,
            ).view(-1, 1)
            policy_returns = (rewards_t * discounts).sum(dim=0)
            trajectory_advantages = policy_returns - bc_returns
        advantages = trajectory_advantages.view(1, -1).expand_as(rewards_t)
    else:
        advantages = returns_t.detach() - values_t.detach()
    if config.causal_policy_mode in {"advantage", "advantage_gate"} and causal_advantages is not None:
        advantages = causal_advantages.detach()
    policy_advantages = policy_advantage(advantages, config)
    policy_mask = torch.ones_like(policy_advantages, dtype=torch.bool)
    if config.causal_policy_mode in {"gate", "advantage_gate"} and causal_advantages is not None:
        policy_mask &= causal_advantages.detach().abs() >= float(config.causal_policy_min_margin)
    if config.policy_loss_min_advantage_abs > 0:
        policy_mask &= policy_advantages.detach().abs() >= float(config.policy_loss_min_advantage_abs)
    if config.policy_loss_max_prior_mse > 0:
        policy_mask &= prior_mse_t.detach() <= float(config.policy_loss_max_prior_mse)
    positive = (policy_advantages >= 0) & policy_mask
    negative = (policy_advantages < 0) & policy_mask
    if config.advantage_mode == "weighted":
        weights = policy_mask.float()
        policy_loss = -((log_probs_t * policy_advantages.detach()) * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        policy_terms = []
        if positive.any():
            policy_terms.append(-0.5 * log_probs_t[positive].mean())
        if negative.any():
            policy_terms.append(0.5 * log_probs_t[negative].mean())
        policy_loss = sum(policy_terms) if policy_terms else torch.zeros((), device=values_t.device)
    value_loss = F.mse_loss(values_t, returns_t.detach())
    prior_loss = prior_mse_t.mean()
    prior_hinge_loss = torch.relu(prior_loss - float(config.prior_hinge_target)).pow(2)
    mean_prior_loss = mean_prior_mse_t.mean()
    mean_prior_hinge_loss = torch.relu(mean_prior_loss - float(config.mean_prior_hinge_target)).pow(2)
    entropy = entropy_t.mean()
    loss = (
        policy_loss
        + config.prior_weight * prior_loss
        + config.prior_hinge_weight * prior_hinge_loss
        + config.mean_prior_weight * mean_prior_loss
        + config.mean_prior_hinge_weight * mean_prior_hinge_loss
        + config.causal_shortfall_policy_weight * causal_shortfall_policy_loss
        - config.entropy_weight * entropy
    )
    if not config.freeze_value_during_imagination:
        loss = loss + config.value_loss_weight * value_loss
    return {
        "loss_tensor": loss,
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "prior_mse": float(prior_loss.detach().item()),
        "prior_hinge_loss": float(prior_hinge_loss.detach().item()),
        "mean_prior_mse": float(mean_prior_loss.detach().item()),
        "mean_prior_hinge_loss": float(mean_prior_hinge_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
        "imagined_reward": float(rewards_t.mean().detach().item()),
        "imagined_return0": float(returns_t[0].mean().detach().item()),
        "bc_relative_advantage_mean": float(advantages.mean().detach().item()),
        "raw_positive_advantage_fraction": float((advantages >= 0).float().mean().detach().item()),
        "positive_advantage_fraction": float(positive.float().mean().detach().item()),
        "policy_loss_sample_fraction": float(policy_mask.float().mean().detach().item()),
        "advantage_mean": float(advantages.mean().detach().item()),
        "advantage_std": float(advantages.std(unbiased=False).detach().item()),
        "causal_advantage_mean": float(causal_margin_mean.detach().item()),
        "causal_positive_fraction": float(causal_positive_fraction.detach().item()),
        "causal_shortfall_policy_loss": float(causal_shortfall_policy_loss.detach().item()),
        "causal_shortfall_fraction": float(causal_shortfall_fraction.detach().item()),
        "action_mean_abs": float(torch.stack(action_mags, dim=0).mean().detach().item()),
    }


def corrupt_imagination_dynamics_action(
    action: torch.Tensor,
    mask: torch.Tensor,
    config: NativeImaginationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    return corrupt_action_for_mode(action, mask, config.imagination_dynamics_action_mode)


def corrupt_action_for_mode(
    action: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "policy":
        return action, mask
    if mode == "zero":
        return torch.zeros_like(action), mask
    if mode == "shuffle":
        if action.shape[0] <= 1:
            return action, mask
        perm = torch.randperm(action.shape[0], device=action.device)
        return action[perm], mask[perm]
    if mode == "far_shuffle":
        if action.shape[0] <= 1:
            return action, mask
        perm = torch.roll(torch.arange(action.shape[0], device=action.device), shifts=action.shape[0] // 2)
        return action[perm], mask[perm]
    raise ValueError(f"unknown action corruption mode: {mode}")


def corrupt_imagination_agent_action_context(
    action: torch.Tensor,
    mask: torch.Tensor,
    config: NativeImaginationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.imagination_agent_action_context_mode == "policy":
        return action, mask
    if config.imagination_agent_action_context_mode == "zero":
        return torch.zeros_like(action), mask
    if config.imagination_agent_action_context_mode == "shuffle":
        if action.shape[0] <= 1:
            return action, mask
        perm = torch.randperm(action.shape[0], device=action.device)
        return action[perm] * mask, mask
    raise ValueError(
        f"unknown imagination_agent_action_context_mode: {config.imagination_agent_action_context_mode}"
    )


@torch.no_grad()
def evaluate_policies(
    *,
    agent: AgentHeads,
    encoder: Any,
    dynamics: Dynamics,
    loader: Any,
    tok_args: dict[str, Any],
    dyn_args: dict[str, Any],
    config: NativeImaginationConfig,
    device: torch.device,
    task_names: list[str] | None = None,
) -> dict[str, Any]:
    agent.eval()
    totals = {
        "zero": [],
        "bc_prior": [],
        "policy": [],
        "policy_prior_mse": [],
        "policy_action_abs": [],
        "bc_action_abs": [],
        "policy_dyn_zero": [],
        "policy_dyn_shuffle": [],
        "reward_contrast_true_minus_zero": [],
        "reward_contrast_true_minus_shuffle": [],
        "reward_contrast_positive_fraction": [],
    }
    do_causal_eval = should_eval_causal_dynamics(config)
    per_task: dict[int, dict[str, list[float]]] = {}
    for i, raw_batch in enumerate(loader):
        if i >= config.eval_batches:
            break
        batch = prepare_batch(raw_batch, encoder, tok_args, dyn_args, config, device)
        _, reward_cf = reward_counterfactual_loss(
            agent=agent,
            dynamics=dynamics,
            batch=batch,
            dyn_args=dyn_args,
            config=config,
        )
        for key in [
            "reward_contrast_true_minus_zero",
            "reward_contrast_true_minus_shuffle",
            "reward_contrast_positive_fraction",
        ]:
            totals[key].append(float(reward_cf.get(key, 0.0)))
        zero_return = rollout_return(agent, dynamics, batch, dyn_args, config, mode="zero")
        totals["zero"].append(float(zero_return.mean().item()))
        bc_return, bc_abs = rollout_return(
            agent, dynamics, batch, dyn_args, config, mode="bc_prior", return_action_abs=True
        )
        pol_return, pol_abs, prior_mse = rollout_return(
            agent, dynamics, batch, dyn_args, config, mode="policy", return_action_abs=True, return_prior_mse=True
        )
        totals["bc_prior"].append(float(bc_return.mean().item()))
        totals["policy"].append(float(pol_return.mean().item()))
        totals["bc_action_abs"].append(float(bc_abs.mean().item()))
        totals["policy_action_abs"].append(float(pol_abs.mean().item()))
        totals["policy_prior_mse"].append(float(prior_mse.mean().item()))
        if do_causal_eval:
            pol_dyn_zero = rollout_return(
                agent, dynamics, batch, dyn_args, config, mode="policy", dynamics_action_mode="zero"
            )
            pol_dyn_shuffle = rollout_return(
                agent, dynamics, batch, dyn_args, config, mode="policy", dynamics_action_mode="shuffle"
            )
            totals["policy_dyn_zero"].append(float(pol_dyn_zero.mean().item()))
            totals["policy_dyn_shuffle"].append(float(pol_dyn_shuffle.mean().item()))
        task_ids = batch["task_ids"].detach().cpu().tolist()
        for j, task_id in enumerate(task_ids):
            task_bucket = per_task.setdefault(int(task_id), {"zero": [], "bc_prior": [], "policy": []})
            task_bucket["zero"].append(float(zero_return[j].item()))
            task_bucket["bc_prior"].append(float(bc_return[j].item()))
            task_bucket["policy"].append(float(pol_return[j].item()))
    out = {key: mean(vals) for key, vals in totals.items()}
    out["policy_minus_bc"] = out["policy"] - out["bc_prior"]
    out["policy_minus_zero"] = out["policy"] - out["zero"]
    if do_causal_eval:
        out["policy_minus_dyn_zero"] = out["policy"] - out["policy_dyn_zero"]
        out["policy_minus_dyn_shuffle"] = out["policy"] - out["policy_dyn_shuffle"]
        out["causal_policy_gain"] = out["policy"] - max(out["policy_dyn_zero"], out["policy_dyn_shuffle"])
    else:
        out["policy_dyn_zero"] = 0.0
        out["policy_dyn_shuffle"] = 0.0
        out["policy_minus_dyn_zero"] = 0.0
        out["policy_minus_dyn_shuffle"] = 0.0
        out["causal_policy_gain"] = 0.0
    out["per_task"] = summarize_per_task(per_task, task_names)
    return out


def should_eval_causal_dynamics(config: NativeImaginationConfig) -> bool:
    return bool(config.eval_causal_dynamics) or config.causal_policy_mode != "off" or config.best_imagination_metric in {
        "causal_policy_gain",
        "policy_minus_dyn_zero",
        "policy_minus_dyn_shuffle",
        "policy_minus_bc_plus_dyn_shuffle",
        "policy_minus_bc_causal_gate",
        "policy_minus_bc_zero_causal_gate",
        "policy_minus_bc_zero_causal_gate_source_aware",
    }


@torch.no_grad()
def rollout_return(
    agent: AgentHeads,
    dynamics: Dynamics,
    batch: dict[str, torch.Tensor],
    dyn_args: dict[str, Any],
    config: NativeImaginationConfig,
    *,
    mode: str,
    return_action_abs: bool = False,
    return_prior_mse: bool = False,
    dynamics_action_mode: str = "policy",
) -> Any:
    z_seq = batch["z"][:, : config.ctx_len].detach()
    dyn_action_seq = batch["actions"][:, : config.ctx_len].detach()
    dyn_mask_seq = batch["mask"][:, : config.ctx_len].detach()
    policy_action_seq = policy_context_actions(batch, config)[:, : config.ctx_len].detach()
    policy_mask_seq = policy_context_masks(batch, config)[:, : config.ctx_len].detach()
    current_mask = policy_mask_seq[:, -1]
    rewards = []
    action_abs = []
    prior_mses = []
    for _ in range(config.imagination_horizon):
        features = state_features(z_seq, policy_action_seq, max_context=config.ctx_len)
        prior_flat = agent.prior_mean(features)
        prior, _ = first_chunk_action(prior_flat, current_mask, config)
        if mode == "zero":
            action = torch.zeros_like(prior)
        elif mode == "bc_prior":
            action = prior
        elif mode == "policy":
            policy_flat = agent.policy_mean(features)
            action, _ = first_chunk_action(policy_flat, current_mask, config)
        else:
            raise ValueError(f"unknown rollout mode: {mode}")
        if config.policy_action_source == "raw":
            dyn_action, dyn_action_mask = expand_raw_action_step(policy_action_seq, action, current_mask, config)
        else:
            dyn_action, dyn_action_mask = action, current_mask
        if dynamics_action_mode == "time_shift":
            dyn_action_for_model = dyn_action_seq[:, -1]
            dyn_action_mask_for_model = dyn_mask_seq[:, -1]
        else:
            dyn_action_for_model, dyn_action_mask_for_model = corrupt_action_for_mode(
                dyn_action,
                dyn_action_mask,
                dynamics_action_mode,
            )
        next_z = imagine_next(
            dynamics,
            z_seq,
            dyn_action_seq,
            dyn_mask_seq,
            dyn_action_for_model,
            dyn_action_mask_for_model,
            dyn_args,
            config,
        )
        next_dyn_actions = torch.cat([dyn_action_seq, dyn_action_for_model[:, None, :]], dim=1)[:, -config.ctx_len :]
        next_dyn_masks = torch.cat([dyn_mask_seq, dyn_action_mask_for_model[:, None, :]], dim=1)[:, -config.ctx_len :]
        next_policy_actions = torch.cat([policy_action_seq, action[:, None, :]], dim=1)[:, -config.ctx_len :]
        next_policy_masks = torch.cat([policy_mask_seq, current_mask[:, None, :]], dim=1)[:, -config.ctx_len :]
        next_z_seq = torch.cat([z_seq, next_z[:, None, :, :]], dim=1)[:, -config.ctx_len :]
        next_reward_features = reward_value_features(next_z_seq, next_policy_actions, config)
        rewards.append(agent.reward_pred(next_reward_features))
        action_abs.append((action.abs() * current_mask).sum(dim=-1) / current_mask.sum(dim=-1).clamp_min(1.0))
        prior_mses.append(masked_mse(action, prior, current_mask, reduce=False))
        z_seq = next_z_seq
        dyn_action_seq, dyn_mask_seq = next_dyn_actions, next_dyn_masks
        policy_action_seq, policy_mask_seq = next_policy_actions, next_policy_masks
    discounts = torch.tensor(
        [config.gamma**i for i in range(len(rewards))],
        device=batch["z"].device,
        dtype=batch["z"].dtype,
    ).view(-1, 1)
    returns = (torch.stack(rewards, dim=0) * discounts).sum(dim=0)
    outputs: list[Any] = [returns]
    if return_action_abs:
        outputs.append(torch.stack(action_abs, dim=0).mean(dim=0))
    if return_prior_mse:
        outputs.append(torch.stack(prior_mses, dim=0).mean(dim=0))
    return tuple(outputs) if len(outputs) > 1 else returns


@torch.no_grad()
def rollout_rewards_for_action_sequence(
    *,
    agent: AgentHeads,
    dynamics: Dynamics,
    batch: dict[str, torch.Tensor],
    dyn_args: dict[str, Any],
    config: NativeImaginationConfig,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    dynamics_action_mode: str,
) -> torch.Tensor:
    z_seq = batch["z"][:, : config.ctx_len].detach()
    dyn_action_seq = batch["actions"][:, : config.ctx_len].detach()
    dyn_mask_seq = batch["mask"][:, : config.ctx_len].detach()
    policy_action_seq = policy_context_actions(batch, config)[:, : config.ctx_len].detach()
    policy_mask_seq = policy_context_masks(batch, config)[:, : config.ctx_len].detach()
    rewards = []
    for t in range(actions.shape[1]):
        action = actions[:, t]
        action_mask = action_masks[:, t]
        if config.policy_action_source == "raw":
            dyn_action, dyn_action_mask = expand_raw_action_step(policy_action_seq, action, action_mask, config)
        else:
            dyn_action, dyn_action_mask = action, action_mask
        dyn_action_for_model, dyn_action_mask_for_model = corrupt_action_for_mode(
            dyn_action,
            dyn_action_mask,
            dynamics_action_mode,
        )
        next_z = imagine_next(
            dynamics,
            z_seq,
            dyn_action_seq,
            dyn_mask_seq,
            dyn_action_for_model,
            dyn_action_mask_for_model,
            dyn_args,
            config,
        )
        next_z_seq = torch.cat([z_seq, next_z[:, None, :, :]], dim=1)[:, -config.ctx_len :]
        next_dyn_actions = torch.cat([dyn_action_seq, dyn_action_for_model[:, None, :]], dim=1)[:, -config.ctx_len :]
        next_dyn_masks = torch.cat([dyn_mask_seq, dyn_action_mask_for_model[:, None, :]], dim=1)[:, -config.ctx_len :]
        next_policy_actions = torch.cat([policy_action_seq, action[:, None, :]], dim=1)[:, -config.ctx_len :]
        next_policy_masks = torch.cat([policy_mask_seq, action_mask[:, None, :]], dim=1)[:, -config.ctx_len :]
        rewards.append(agent.reward_pred(reward_value_features(next_z_seq, next_policy_actions, config)))
        z_seq = next_z_seq
        dyn_action_seq, dyn_mask_seq = next_dyn_actions, next_dyn_masks
        policy_action_seq, policy_mask_seq = next_policy_actions, next_policy_masks
    return torch.stack(rewards, dim=0)


@torch.no_grad()
def prepare_batch(
    raw_batch: dict[str, torch.Tensor],
    encoder: Any,
    tok_args: dict[str, Any],
    dyn_args: dict[str, Any],
    config: NativeImaginationConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    obs_u8 = raw_batch["obs"].to(device, non_blocking=True)
    act = raw_batch["act"].to(device, non_blocking=True)
    mask = raw_batch["act_mask"].to(device, non_blocking=True)
    raw_act = raw_batch.get("raw_act", raw_batch["act"]).to(device, non_blocking=True)
    raw_mask = raw_batch.get("raw_act_mask", raw_batch["act_mask"]).to(device, non_blocking=True)
    rew = raw_batch["rew"].to(device, non_blocking=True).float()
    task_ids = raw_batch["emb_id"].to(device, non_blocking=True).long()
    frames = obs_u8[:, :-1].float() / 255.0
    act = act.clamp(-1, 1) * mask
    raw_act = raw_act.clamp(-1, 1) * raw_mask
    actions, act_mask = align_actions_to_frames(
        act,
        mask,
        frame_count=frames.shape[1],
        action_frame_offset=config.action_frame_offset,
    )
    raw_aligned, raw_aligned_mask = align_actions_to_frames(
        raw_act,
        raw_mask,
        frame_count=frames.shape[1],
        action_frame_offset=config.action_frame_offset,
    )
    patch = int(tok_args.get("patch", 8))
    packing_factor = int(dyn_args.get("packing_factor", 2))
    n_latents = int(tok_args.get("n_latents", 16))
    n_spatial = n_latents // packing_factor
    patches = temporal_patchify(frames, patch)
    z_btld, _ = encoder(patches)
    z = pack_bottleneck_to_spatial(z_btld, n_spatial=n_spatial, k=packing_factor)
    return {
        "z": z,
        "actions": actions,
        "mask": act_mask,
        "raw_actions": raw_aligned,
        "raw_mask": raw_aligned_mask,
        "raw_targets": raw_act,
        "raw_target_mask": raw_mask,
        "rewards": rew,
        "task_ids": task_ids,
    }


def supervised_targets(
    batch: dict[str, torch.Tensor],
    config: NativeImaginationConfig,
    target_stats: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    z = batch["z"]
    context_actions = policy_context_actions(batch, config)
    target_actions = policy_target_actions(batch, config)
    target_masks = policy_target_masks(batch, config)
    rewards = batch["rewards"]
    bsz, timesteps = target_actions.shape[:2]
    z_flat = z.reshape(bsz, timesteps, -1)
    z_mean = z_flat.cumsum(dim=1) / torch.arange(1, timesteps + 1, device=z.device, dtype=z.dtype).view(1, -1, 1)
    if config.policy_action_source == "raw":
        action_context = context_actions
    else:
        action_context = torch.zeros_like(context_actions)
        action_context[:, 1:] = context_actions[:, :-1]
    policy_features = torch.cat([z_flat, z_mean, action_context], dim=-1).reshape(bsz * timesteps, -1)
    reward_value_context = action_context_for_reward_value(action_context, config)
    reward_value_features_out = torch.cat([z_flat, z_mean, reward_value_context], dim=-1).reshape(bsz * timesteps, -1)
    target_chunks, target_chunk_masks = make_action_chunks(
        target_actions,
        target_masks,
        chunk_len=config.action_chunk_len,
    )
    target_actions = target_chunks.reshape(bsz * timesteps, -1)
    target_masks = target_chunk_masks.reshape(bsz * timesteps, -1)
    value_targets_seq = discounted_returns(rewards, config.gamma)
    reward_targets, value_targets_seq = normalize_targets(
        rewards,
        value_targets_seq,
        batch["task_ids"],
        config,
        target_stats,
    )
    reward_targets = reward_targets.reshape(bsz * timesteps)
    value_targets = value_targets_seq.reshape(bsz * timesteps)
    return policy_features, reward_value_features_out, target_actions, target_masks, reward_targets, value_targets


def state_features(z_seq: torch.Tensor, action_seq: torch.Tensor, *, max_context: int) -> torch.Tensor:
    z_recent = z_seq[:, -max_context:]
    a_recent = action_seq[:, -max_context:]
    z_flat = z_recent.reshape(z_recent.shape[0], z_recent.shape[1], -1)
    z_last = z_flat[:, -1]
    z_mean = z_flat.mean(dim=1)
    a_last = a_recent[:, -1]
    return torch.cat([z_last, z_mean, a_last], dim=-1)


def reward_value_features(z_seq: torch.Tensor, action_seq: torch.Tensor, config: NativeImaginationConfig) -> torch.Tensor:
    z_recent = z_seq[:, -config.ctx_len :]
    a_recent = action_seq[:, -config.ctx_len :]
    z_flat = z_recent.reshape(z_recent.shape[0], z_recent.shape[1], -1)
    z_last = z_flat[:, -1]
    z_mean = z_flat.mean(dim=1)
    a_last = action_context_for_reward_value(a_recent[:, -1], config)
    return torch.cat([z_last, z_mean, a_last], dim=-1)


def action_context_for_reward_value(action_context: torch.Tensor, config: NativeImaginationConfig) -> torch.Tensor:
    if config.reward_value_action_context_mode == "policy":
        return action_context
    if config.reward_value_action_context_mode == "zero":
        return torch.zeros_like(action_context)
    raise ValueError(f"unknown reward_value_action_context_mode: {config.reward_value_action_context_mode}")


def policy_context_actions(batch: dict[str, torch.Tensor], config: NativeImaginationConfig) -> torch.Tensor:
    if config.policy_action_source == "raw":
        return batch["raw_actions"]
    return batch["actions"]


def policy_context_masks(batch: dict[str, torch.Tensor], config: NativeImaginationConfig) -> torch.Tensor:
    if config.policy_action_source == "raw":
        return batch["raw_mask"]
    return batch["mask"]


def policy_target_actions(batch: dict[str, torch.Tensor], config: NativeImaginationConfig) -> torch.Tensor:
    if config.policy_action_source == "raw":
        return batch["raw_targets"]
    return batch["actions"]


def policy_target_masks(batch: dict[str, torch.Tensor], config: NativeImaginationConfig) -> torch.Tensor:
    if config.policy_action_source == "raw":
        return batch["raw_target_mask"]
    return batch["mask"]


def make_action_chunks(
    actions: torch.Tensor,
    masks: torch.Tensor,
    *,
    chunk_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_len = max(1, int(chunk_len))
    if chunk_len == 1:
        return actions, masks

    bsz, timesteps, action_dim = actions.shape
    chunks = []
    chunk_masks = []
    for offset in range(chunk_len):
        shifted = torch.zeros_like(actions)
        shifted_mask = torch.zeros_like(masks)
        if offset < timesteps:
            shifted[:, : timesteps - offset] = actions[:, offset:]
            shifted_mask[:, : timesteps - offset] = masks[:, offset:]
        chunks.append(shifted)
        chunk_masks.append(shifted_mask)
    return torch.cat(chunks, dim=-1), torch.cat(chunk_masks, dim=-1)


def first_chunk_action(
    action_flat: torch.Tensor,
    mask: torch.Tensor,
    config: NativeImaginationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_dim = policy_single_action_dim(config)
    action = action_flat[..., :action_dim]
    return action * mask, mask


def first_chunk_log_prob(
    values_flat: torch.Tensor,
    mask: torch.Tensor,
    config: NativeImaginationConfig,
) -> torch.Tensor:
    action_dim = policy_single_action_dim(config)
    values = values_flat[..., :action_dim]
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def expand_raw_action_step(
    raw_history: torch.Tensor,
    raw_action: torch.Tensor,
    raw_mask: torch.Tensor,
    config: NativeImaginationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_action = raw_action * raw_mask
    prev = raw_history[:, -1] * raw_mask
    history_tail = raw_history[:, -3:] if raw_history.shape[1] >= 3 else raw_history
    mean4 = torch.cat([history_tail, raw_action[:, None, :]], dim=1).mean(dim=1) * raw_mask

    parts = []
    masks = []
    scalar_mask = (raw_mask.sum(dim=-1, keepdim=True) > 0).to(raw_mask.dtype)
    for name in normalize_action_features(config.action_features):
        if name == "current":
            parts.append(raw_action)
            masks.append(raw_mask)
        elif name == "prev":
            parts.append(prev)
            masks.append(raw_mask)
        elif name == "delta":
            parts.append((raw_action - prev) * raw_mask)
            masks.append(raw_mask)
        elif name == "mean4":
            parts.append(mean4)
            masks.append(raw_mask)
        elif name == "norm":
            parts.append(raw_action.abs().mean(dim=-1, keepdim=True))
            masks.append(scalar_mask)
        else:
            raise RuntimeError(f"Unhandled action feature: {name}")

    features = torch.cat(parts, dim=-1) if parts else raw_action.new_zeros(raw_action.shape[0], 0)
    feature_mask = torch.cat(masks, dim=-1) if masks else raw_mask.new_zeros(raw_mask.shape[0], 0)
    if features.shape[-1] > config.action_dim:
        raise RuntimeError(
            f"expanded raw action has dim {features.shape[-1]} but dynamics action_dim is {config.action_dim}"
        )
    padded = raw_action.new_zeros(raw_action.shape[0], config.action_dim)
    padded_mask = raw_mask.new_zeros(raw_mask.shape[0], config.action_dim)
    if features.shape[-1] > 0:
        padded[:, : features.shape[-1]] = features
        padded_mask[:, : feature_mask.shape[-1]] = feature_mask
    return padded, padded_mask


@torch.no_grad()
def imagine_next(
    dynamics: Dynamics,
    z_seq: torch.Tensor,
    action_seq: torch.Tensor,
    mask_seq: torch.Tensor,
    action: torch.Tensor,
    action_mask: torch.Tensor,
    dyn_args: dict[str, Any],
    config: NativeImaginationConfig,
) -> torch.Tensor:
    sched = make_tau_schedule(k_max=int(dyn_args.get("k_max", 8)), schedule="shortcut", d=config.eval_d)
    past_z = z_seq[:, -config.ctx_len :]
    past_actions = action_seq[:, -config.ctx_len :]
    past_masks = mask_seq[:, -config.ctx_len :]
    actions_for_dyn = torch.cat([past_actions, action[:, None, :]], dim=1)
    masks_for_dyn = torch.cat([past_masks, action_mask[:, None, :]], dim=1)
    return sample_one_timestep_packed(
        dynamics,
        past_packed=past_z,
        k_max=int(dyn_args.get("k_max", 8)),
        sched=sched,
        actions=actions_for_dyn,
        act_mask=masks_for_dyn,
    )


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, reduce: bool = True) -> torch.Tensor:
    per = ((pred - target).float().pow(2) * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1).clamp_min(1.0)
    return per.mean() if reduce else per


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    running = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    outs = []
    for t in reversed(range(rewards.shape[1])):
        running = rewards[:, t] + float(gamma) * running
        outs.append(running)
    return torch.stack(list(reversed(outs)), dim=1)


def lambda_returns(rewards_t: torch.Tensor, values_t: torch.Tensor, gamma: float) -> torch.Tensor:
    running = values_t[-1].detach()
    outs = []
    for t in reversed(range(rewards_t.shape[0])):
        running = rewards_t[t] + float(gamma) * running
        outs.append(running)
    return torch.stack(list(reversed(outs)), dim=0)


def build_target_stats(dataset: WMDataset, config: NativeImaginationConfig, device: torch.device) -> dict[str, torch.Tensor]:
    n_tasks = int(dataset.num_tasks)
    reward_means = torch.zeros(n_tasks, dtype=torch.float32)
    reward_stds = torch.ones(n_tasks, dtype=torch.float32)
    value_means = torch.zeros(n_tasks, dtype=torch.float32)
    value_stds = torch.ones(n_tasks, dtype=torch.float32)

    if config.target_normalization == "none":
        mode = torch.tensor(0)
        return {
            "mode": mode,
            "reward_mean": reward_means.to(device),
            "reward_std": reward_stds.to(device),
            "value_mean": value_means.to(device),
            "value_std": value_stds.to(device),
        }

    all_rewards = []
    all_values = []
    per_task_rewards = []
    per_task_values = []
    for rew in dataset.rew:
        rew_f = torch.nan_to_num(rew.float(), nan=0.0, posinf=0.0, neginf=0.0)
        values = discounted_returns(rew_f.view(1, -1), config.gamma).view(-1)
        per_task_rewards.append(rew_f)
        per_task_values.append(values)
        all_rewards.append(rew_f)
        all_values.append(values)

    if config.target_normalization == "global":
        reward_all = torch.cat(all_rewards)
        value_all = torch.cat(all_values)
        reward_mean = reward_all.mean()
        reward_std = reward_all.std(unbiased=False).clamp_min(float(config.min_target_std))
        value_mean = value_all.mean()
        value_std = value_all.std(unbiased=False).clamp_min(float(config.min_target_std))
        reward_means.fill_(float(reward_mean.item()))
        reward_stds.fill_(float(reward_std.item()))
        value_means.fill_(float(value_mean.item()))
        value_stds.fill_(float(value_std.item()))
    elif config.target_normalization == "per_task":
        for idx, (rew_f, values) in enumerate(zip(per_task_rewards, per_task_values)):
            reward_means[idx] = rew_f.mean()
            reward_stds[idx] = rew_f.std(unbiased=False).clamp_min(float(config.min_target_std))
            value_means[idx] = values.mean()
            value_stds[idx] = values.std(unbiased=False).clamp_min(float(config.min_target_std))
    else:
        raise ValueError(f"unknown target_normalization: {config.target_normalization}")

    return {
        "mode": torch.tensor(1),
        "reward_mean": reward_means.to(device),
        "reward_std": reward_stds.to(device),
        "value_mean": value_means.to(device),
        "value_std": value_stds.to(device),
    }


def normalize_targets(
    rewards: torch.Tensor,
    values: torch.Tensor,
    task_ids: torch.Tensor,
    config: NativeImaginationConfig,
    target_stats: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.target_normalization == "none":
        reward_targets = rewards
        value_targets = values
    else:
        reward_mean = target_stats["reward_mean"][task_ids].view(-1, 1)
        reward_std = target_stats["reward_std"][task_ids].view(-1, 1).clamp_min(float(config.min_target_std))
        value_mean = target_stats["value_mean"][task_ids].view(-1, 1)
        value_std = target_stats["value_std"][task_ids].view(-1, 1).clamp_min(float(config.min_target_std))
        reward_targets = (rewards - reward_mean) / reward_std
        value_targets = (values - value_mean) / value_std

    if config.reward_clip > 0:
        reward_targets = reward_targets.clamp(-float(config.reward_clip), float(config.reward_clip))
    if config.value_clip > 0:
        value_targets = value_targets.clamp(-float(config.value_clip), float(config.value_clip))
    return reward_targets, value_targets


def policy_advantage(advantages: torch.Tensor, config: NativeImaginationConfig) -> torch.Tensor:
    if config.advantage_mode == "raw_sign":
        return advantages
    centered = advantages - advantages.mean()
    scaled = centered / advantages.std(unbiased=False).clamp_min(1e-6)
    if config.advantage_clip > 0:
        scaled = scaled.clamp(-float(config.advantage_clip), float(config.advantage_clip))
    if config.advantage_mode in {"centered_sign", "weighted"}:
        return scaled
    raise ValueError(f"unknown advantage_mode: {config.advantage_mode}")


def parse_modes(value: str, *, default: tuple[str, ...]) -> list[str]:
    modes = [mode.strip() for mode in str(value).replace("+", ",").split(",") if mode.strip()]
    return modes or list(default)


def target_stats_for_json(target_stats: dict[str, torch.Tensor], task_names: list[str]) -> dict[str, Any]:
    out = {"tasks": []}
    for idx, name in enumerate(task_names):
        out["tasks"].append(
            {
                "id": idx,
                "name": name,
                "reward_mean": float(target_stats["reward_mean"][idx].detach().cpu().item()),
                "reward_std": float(target_stats["reward_std"][idx].detach().cpu().item()),
                "value_mean": float(target_stats["value_mean"][idx].detach().cpu().item()),
                "value_std": float(target_stats["value_std"][idx].detach().cpu().item()),
            }
        )
    return out


def summarize_per_task(
    per_task: dict[int, dict[str, list[float]]],
    task_names: list[str] | None,
    *,
    max_tasks: int = 20,
) -> dict[str, Any]:
    rows = []
    for task_id, values in sorted(per_task.items()):
        zero = mean(values["zero"])
        bc = mean(values["bc_prior"])
        policy = mean(values["policy"])
        rows.append(
            {
                "task_id": task_id,
                "task": task_names[task_id] if task_names and task_id < len(task_names) else str(task_id),
                "n": len(values["policy"]),
                "zero": zero,
                "bc_prior": bc,
                "policy": policy,
                "policy_minus_bc": policy - bc,
                "policy_minus_zero": policy - zero,
            }
        )
    rows.sort(key=lambda row: row["policy_minus_bc"])
    return {
        "num_tasks_seen": len(rows),
        "mean_policy_minus_bc": mean([row["policy_minus_bc"] for row in rows]),
        "worst_policy_minus_bc": rows[:max_tasks],
        "best_policy_minus_bc": list(reversed(rows[-max_tasks:])),
    }


def cycle_loader(loader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def set_eval_seed(config: NativeImaginationConfig) -> None:
    seed = int(config.eval_seed if config.eval_seed else config.seed + 7919)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def unfreeze_module(module: nn.Module) -> None:
    module.train()
    for param in module.parameters():
        param.requires_grad_(True)


def compare_eval(before: dict[str, float], after: dict[str, float]) -> dict[str, float | bool]:
    delta_bc = after["bc_prior"] - before["bc_prior"]
    delta_policy = after["policy"] - before["policy"]
    return {
        "bc_prior_return_delta": delta_bc,
        "policy_return_delta": delta_policy,
        "after_policy_minus_bc": after["policy"] - after["bc_prior"],
        "after_policy_minus_zero": after["policy"] - after["zero"],
        "after_policy_minus_dyn_zero": after.get("policy_minus_dyn_zero", 0.0),
        "after_policy_minus_dyn_shuffle": after.get("policy_minus_dyn_shuffle", 0.0),
        "after_causal_policy_gain": after.get("causal_policy_gain", 0.0),
        "policy_prior_mse_delta": after["policy_prior_mse"] - before["policy_prior_mse"],
        "policy_action_abs_delta": after["policy_action_abs"] - before["policy_action_abs"],
        "after_reward_contrast_true_minus_zero": after.get("reward_contrast_true_minus_zero", 0.0),
        "after_reward_contrast_true_minus_shuffle": after.get("reward_contrast_true_minus_shuffle", 0.0),
        "reward_contrast_true_minus_zero_delta": after.get("reward_contrast_true_minus_zero", 0.0)
        - before.get("reward_contrast_true_minus_zero", 0.0),
        "reward_contrast_true_minus_shuffle_delta": after.get("reward_contrast_true_minus_shuffle", 0.0)
        - before.get("reward_contrast_true_minus_shuffle", 0.0),
        "policy_improved_over_bc": bool(after["policy"] > after["bc_prior"]),
        "policy_improved_over_zero": bool(after["policy"] > after["zero"]),
    }


def state_dict_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def strip_large_eval(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "per_task"}


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    def compact_eval(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if key != "per_task"}

    return {
        "phase": summary["phase"],
        "imagination_mode": summary["config"].get("imagination_mode"),
        "policy_action_source": summary["config"].get("policy_action_source"),
        "action_chunk_len": summary["config"].get("action_chunk_len"),
        "reward_filter_mode": summary["config"].get("reward_filter_mode"),
        "before": compact_eval(summary["before_imagination"]),
        "after": compact_eval(summary["after_imagination"]),
        "comparison": summary["comparison"],
        "best_imagination_selection": None
        if summary.get("best_imagination_selection") is None
        else {
            "selected_update": summary["best_imagination_selection"].get("selected_update"),
            "metric": summary["best_imagination_selection"].get("metric"),
            "metric_value": summary["best_imagination_selection"].get("metric_value"),
        },
        "split": {
            "mode": summary.get("split_info", {}).get("mode"),
            "train_windows": summary.get("split_info", {}).get("train_windows"),
            "eval_windows": summary.get("split_info", {}).get("eval_windows"),
        },
        "train_sampling": {
            "mode": summary.get("train_sampling_info", {}).get("mode"),
            "sample_count": summary.get("train_sampling_info", {}).get("sample_count"),
        },
        "eval_sampling": {
            "mode": summary.get("eval_sampling_info", {}).get("mode"),
            "num_tasks_sampled": summary.get("eval_sampling_info", {}).get("num_tasks_sampled"),
            "sample_count": summary.get("eval_sampling_info", {}).get("sample_count"),
        },
        "out_dir": summary["config"]["out_dir"],
        "elapsed_s": summary["elapsed_s"],
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    before = summary["before_imagination"]
    after = summary["after_imagination"]
    comp = summary["comparison"]
    best = summary.get("best_imagination_selection")
    lines = [
        "# Native Dreamer4 Imagination Test",
        "",
        "## Claim Boundary",
        summary["claim_boundary"],
        "",
        "## Calibration",
        f"- Imagination mode: `{summary['config'].get('imagination_mode', 'train')}`",
        f"- Eval split: `{summary.get('split_info', {}).get('mode', 'unknown')}`",
        f"- Eval holdout fraction: `{summary['config'].get('eval_holdout_fraction', 0.0)}`",
        f"- Train windows: `{summary.get('split_info', {}).get('train_windows', 'n/a')}`",
        f"- Eval windows: `{summary.get('split_info', {}).get('eval_windows', 'n/a')}`",
        f"- Eval sampling: `{summary.get('eval_sampling_info', {}).get('mode', 'unknown')}`",
        f"- Eval tasks sampled: `{summary.get('eval_sampling_info', {}).get('num_tasks_sampled', 'n/a')}`",
        f"- Eval samples: `{summary.get('eval_sampling_info', {}).get('sample_count', 'n/a')}`",
        f"- Policy action source: `{summary['config'].get('policy_action_source', 'expanded')}`",
        f"- Dynamics action dim: `{summary['config'].get('action_dim', 'n/a')}`",
        f"- Raw action dim: `{summary['config'].get('raw_action_dim', 'n/a')}`",
        f"- Action chunk len: `{summary['config'].get('action_chunk_len', 1)}`",
        f"- Action features for dynamics: `{summary['config'].get('action_features', 'current')}`",
        f"- Select best imagination checkpoint: `{summary['config'].get('select_best_imagination', False)}`",
        f"- Imagination eval every: `{summary['config'].get('imagination_eval_every', 0)}`",
        f"- Min imagination selection update: `{summary['config'].get('min_imagination_selection_update', 0)}`",
        f"- Best imagination metric: `{summary['config'].get('best_imagination_metric', 'policy_minus_bc')}`",
        f"- Detach policy log-prob action: `{summary['config'].get('detach_policy_log_prob', True)}`",
        f"- Imagination dynamics action mode: `{summary['config'].get('imagination_dynamics_action_mode', 'policy')}`",
        f"- Imagination agent action context mode: `{summary['config'].get('imagination_agent_action_context_mode', 'policy')}`",
        f"- Reward/value action context mode: `{summary['config'].get('reward_value_action_context_mode', 'policy')}`",
        f"- Reward contrast weight: `{summary['config'].get('reward_contrast_weight', 0.0)}`",
        f"- Reward contrast negative modes: `{summary['config'].get('reward_contrast_negative_modes', 'zero,shuffle')}`",
        f"- Reward contrast horizon: `{summary['config'].get('reward_contrast_horizon', 1)}`",
        f"- Causal policy mode: `{summary['config'].get('causal_policy_mode', 'off')}`",
        f"- Causal policy negative modes: `{summary['config'].get('causal_policy_negative_modes', 'zero,shuffle')}`",
        f"- Causal policy min margin: `{summary['config'].get('causal_policy_min_margin', 0.0)}`",
        f"- Causal shortfall policy weight: `{summary['config'].get('causal_shortfall_policy_weight', 0.0)}`",
        f"- Source eval sources: `{summary['config'].get('source_eval_sources', '')}`",
        f"- Source gate hard sources: `{summary['config'].get('source_gate_hard_sources', 'all,soar')}`",
        f"- Source gate soft sources: `{summary['config'].get('source_gate_soft_sources', 'droid')}`",
        f"- Eval causal dynamics: `{summary['config'].get('eval_causal_dynamics', False)}`",
        f"- Aux inverse weight: `{summary['config'].get('aux_inverse_weight', 0.0)}`",
        f"- Aux effect weight: `{summary['config'].get('aux_effect_weight', 0.0)}`",
        f"- Require non-noop windows: `{summary['config'].get('require_non_noop', False)}`",
        f"- Reward filter mode: `{summary['config'].get('reward_filter_mode', 'none')}`",
        f"- Reward signal threshold: `{summary['config'].get('reward_signal_threshold', 0.0)}`",
        f"- Min reward signal steps: `{summary['config'].get('min_reward_signal_steps', 1)}`",
        f"- Train sampling mode: `{summary['config'].get('train_sampling_mode', 'shuffle')}`",
        f"- Train sampling samples: `{summary.get('train_sampling_info', {}).get('sample_count', 'n/a')}`",
        f"- Train balance spec: `{summary['config'].get('train_balance_spec', 'n/a')}`",
        f"- Train action active threshold: `{summary['config'].get('train_action_active_threshold', 0.0)}`",
        f"- Train min action active steps: `{summary['config'].get('train_min_action_active_steps', 1)}`",
        f"- Target normalization: `{summary['config']['target_normalization']}`",
        f"- Advantage mode: `{summary['config']['advantage_mode']}`",
        f"- Advantage baseline: `{summary['config'].get('advantage_baseline', 'value')}`",
        f"- Advantage clip: `{summary['config']['advantage_clip']}`",
        f"- Policy loss min advantage abs: `{summary['config'].get('policy_loss_min_advantage_abs', 0.0)}`",
        f"- Policy loss max prior MSE: `{summary['config'].get('policy_loss_max_prior_mse', 0.0)}`",
        f"- Prior weight: `{summary['config']['prior_weight']}`",
        f"- Prior hinge weight: `{summary['config']['prior_hinge_weight']}`",
        f"- Prior hinge target: `{summary['config']['prior_hinge_target']}`",
        f"- Mean-prior weight: `{summary['config']['mean_prior_weight']}`",
        f"- Mean-prior hinge weight: `{summary['config']['mean_prior_hinge_weight']}`",
        f"- Mean-prior hinge target: `{summary['config']['mean_prior_hinge_target']}`",
        "",
        "## Before Imagination",
        f"- Zero-action learned return: `{before['zero']:.4f}`",
        f"- BC-prior learned return: `{before['bc_prior']:.4f}`",
        f"- Policy learned return: `{before['policy']:.4f}`",
        "",
        "## After Imagination",
        f"- Zero-action learned return: `{after['zero']:.4f}`",
        f"- BC-prior learned return: `{after['bc_prior']:.4f}`",
        f"- Policy learned return: `{after['policy']:.4f}`",
        f"- Policy dyn-zero return: `{after.get('policy_dyn_zero', 0.0):.4f}`",
        f"- Policy dyn-shuffle return: `{after.get('policy_dyn_shuffle', 0.0):.4f}`",
        f"- Causal policy gain: `{after.get('causal_policy_gain', 0.0):+.4f}`",
        f"- Policy prior MSE: `{after['policy_prior_mse']:.4f}`",
        f"- Policy action abs: `{after['policy_action_abs']:.4f}`",
        f"- Reward true-minus-zero: `{after.get('reward_contrast_true_minus_zero', 0.0):+.4f}`",
        f"- Reward true-minus-shuffle: `{after.get('reward_contrast_true_minus_shuffle', 0.0):+.4f}`",
        "",
        "## Best-Checkpoint Selection",
        f"- Enabled: `{bool(best)}`",
        f"- Selected update: `{best.get('selected_update') if best else 'n/a'}`",
        f"- Metric: `{best.get('metric') if best else 'n/a'}`",
        f"- Metric value: `{best.get('metric_value') if best else 'n/a'}`",
        "",
        "## Decision",
        f"- Policy improved over BC prior: `{comp['policy_improved_over_bc']}`",
        f"- Policy improved over zero action: `{comp['policy_improved_over_zero']}`",
        f"- After policy minus BC: `{comp['after_policy_minus_bc']:.4f}`",
        f"- After policy minus zero: `{comp['after_policy_minus_zero']:.4f}`",
        f"- After policy minus dyn-zero: `{comp.get('after_policy_minus_dyn_zero', 0.0):+.4f}`",
        f"- After policy minus dyn-shuffle: `{comp.get('after_policy_minus_dyn_shuffle', 0.0):+.4f}`",
        f"- After causal policy gain: `{comp.get('after_causal_policy_gain', 0.0):+.4f}`",
        f"- After reward true-minus-zero: `{comp.get('after_reward_contrast_true_minus_zero', 0.0):+.4f}`",
        f"- After reward true-minus-shuffle: `{comp.get('after_reward_contrast_true_minus_shuffle', 0.0):+.4f}`",
        "",
        "## Drift Diagnostics",
        f"- Policy prior MSE delta: `{comp['policy_prior_mse_delta']:.4f}`",
        f"- Policy action abs delta: `{comp['policy_action_abs_delta']:.4f}`",
        "",
        "## Recommended Reading",
        "- If policy return improves while prior MSE stays bounded, the imagination objective is promising.",
        "- If policy drifts from the BC prior and return decreases, fix reward/value calibration or policy constraints before scaling.",
    ]
    if "per_task" in after:
        lines.extend(
            [
                "",
                "## Per-Task Summary",
                f"- Tasks evaluated: `{after['per_task']['num_tasks_seen']}`",
                f"- Mean policy-minus-BC across tasks: `{after['per_task']['mean_policy_minus_bc']:.4f}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
