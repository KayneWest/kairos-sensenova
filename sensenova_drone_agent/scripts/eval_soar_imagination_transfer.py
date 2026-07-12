#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
for item in [str(SCRIPT_ROOT), str(SRC_ROOT)]:
    if item not in sys.path:
        sys.path.insert(0, item)

import train_soar_dreamer_lite as trainlib  # noqa: E402

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("torch is required for SOAR imagination transfer eval.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether SOAR imagination training improved the policy under the learned "
            "action-grounded latent dynamics and calibrated reward model."
        )
    )
    parser.add_argument(
        "--run-dir",
        default="sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_long_ctx8_v2",
    )
    parser.add_argument("--sequence-cache", default="")
    parser.add_argument("--bc-checkpoint", default="")
    parser.add_argument("--imagination-checkpoint", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--rollout-horizon", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    config = load_training_config(run_dir / "config.json")
    if args.sequence_cache:
        config.sequence_cache = str(resolve_path(args.sequence_cache))
    config.device = args.device
    config.batch_size = int(args.batch_size)
    config.seed = int(args.seed)

    out_dir = resolve_path(args.out_dir) if args.out_dir else run_dir / "soar_transfer_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    bc_checkpoint = resolve_path(args.bc_checkpoint) if args.bc_checkpoint else run_dir / "best_agent_bc.pt"
    imagination_checkpoint = (
        resolve_path(args.imagination_checkpoint) if args.imagination_checkpoint else run_dir / "best_imagination.pt"
    )
    if not bc_checkpoint.exists():
        raise FileNotFoundError(f"BC checkpoint not found: {bc_checkpoint}")
    if not imagination_checkpoint.exists():
        raise FileNotFoundError(f"Imagination checkpoint not found: {imagination_checkpoint}")

    started = time.time()
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    cache = trainlib.load_sequence_cache(config.sequence_cache)
    anchors = trainlib.build_dreamer_anchors(
        cache,
        context_len=config.context_len,
        prediction_horizon=config.prediction_horizon,
        mtp_horizon=config.mtp_horizon,
        future_action_offset=config.future_action_offset,
    )
    anchors, motion_filter = trainlib.filter_motion_anchors(cache, anchors, config)
    train_anchors, val_anchors = trainlib.split_for_config(cache, anchors, config)
    if val_anchors.size <= 0:
        raise RuntimeError("No validation anchors available for SOAR transfer eval.")

    normalizer = trainlib.compute_normalizer(cache)
    val_ds = trainlib.SoarDreamerLiteDataset(cache, val_anchors, normalizer, config)
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = trainlib.resolve_device(args.device)

    bc_model = load_model(config, cache, bc_checkpoint, device)
    imagination_model = load_model(config, cache, imagination_checkpoint, device)

    dynamics_gate = trainlib.evaluate_dynamics_controls(
        model=imagination_model,
        loader=val_loader,
        device=device,
    )
    bc_open_loop = evaluate_open_loop(
        policy_model=bc_model,
        scorer_model=bc_model,
        loader=val_loader,
        config=config,
        device=device,
        max_batches=int(args.max_batches),
    )
    imagination_open_loop = evaluate_open_loop(
        policy_model=imagination_model,
        scorer_model=bc_model,
        loader=val_loader,
        config=config,
        device=device,
        max_batches=int(args.max_batches),
    )
    bc_rollout = evaluate_closed_loop_rollout(
        policy_name="bc_prior",
        policy_model=bc_model,
        scorer_model=bc_model,
        loader=val_loader,
        config=config,
        device=device,
        horizon=int(args.rollout_horizon),
        max_batches=int(args.max_batches),
    )
    imagination_rollout = evaluate_closed_loop_rollout(
        policy_name="after_imagination",
        policy_model=imagination_model,
        scorer_model=bc_model,
        loader=val_loader,
        config=config,
        device=device,
        horizon=int(args.rollout_horizon),
        max_batches=int(args.max_batches),
    )
    zero_rollout = evaluate_zero_action_rollout(
        scorer_model=bc_model,
        loader=val_loader,
        config=config,
        device=device,
        horizon=int(args.rollout_horizon),
        max_batches=int(args.max_batches),
    )

    comparison = compare_policies(bc_rollout, imagination_rollout, bc_open_loop, imagination_open_loop)
    summary = {
        "phase": "soar_imagination_transfer_eval",
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "sequence_cache": config.sequence_cache,
        "bc_checkpoint": str(bc_checkpoint),
        "imagination_checkpoint": str(imagination_checkpoint),
        "config": {
            "context_len": config.context_len,
            "prediction_horizon": config.prediction_horizon,
            "mtp_horizon": config.mtp_horizon,
            "motion_filter_quantile": config.motion_filter_quantile,
            "split_mode": config.split_mode,
            "reward_target_mode": config.reward_target_mode,
            "reward_loss_type": config.reward_loss_type,
            "value_target_mode": config.value_target_mode,
            "value_loss_type": config.value_loss_type,
            "gamma": config.gamma,
            "rollout_horizon": int(args.rollout_horizon),
        },
        "anchors": {
            "total": int(anchors.size),
            "train": int(train_anchors.size),
            "val": int(val_anchors.size),
        },
        "motion_filter": motion_filter,
        "dynamics_gate": dynamics_gate,
        "open_loop": {
            "bc_prior": bc_open_loop,
            "after_imagination": imagination_open_loop,
        },
        "closed_loop_model_rollout": {
            "zero_action": zero_rollout,
            "bc_prior": bc_rollout,
            "after_imagination": imagination_rollout,
        },
        "comparison": comparison,
        "elapsed_s": float(time.time() - started),
    }
    write_json(out_dir / "summary.json", summary)
    write_report(summary, out_dir / "report.md")
    print(json.dumps(compact_summary(summary), indent=2))
    return 0


def load_training_config(path: Path) -> trainlib.SoarDreamerLiteConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(trainlib.SoarDreamerLiteConfig)}
    filtered = {key: value for key, value in data.items() if key in allowed}
    return trainlib.SoarDreamerLiteConfig(**filtered)


def load_model(config: Any, cache: Any, checkpoint: Path, device: Any) -> Any:
    model = trainlib.SoarDreamerLiteModel(config, cache).to(device)
    trainlib.load_checkpoint(model, checkpoint, device)
    model.eval()
    return model


def evaluate_open_loop(
    *,
    policy_model: Any,
    scorer_model: Any,
    loader: Any,
    config: Any,
    device: Any,
    max_batches: int,
) -> dict[str, Any]:
    policy_model.eval()
    scorer_model.eval()
    totals: dict[str, float] = {}
    count = 0
    reward_probs: list[Any] = []
    reward_targets: list[Any] = []
    values: list[Any] = []
    value_targets: list[Any] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = trainlib.move_batch(batch, device)
            batch_size = int(batch["z_context"].shape[0])
            policy_out = policy_model.agent(batch["z_context"], batch["action_context"], batch["task_id"])
            scorer_out = scorer_model.agent(batch["z_context"], batch["action_context"], batch["task_id"])
            reward_target = trainlib.reward_targets(batch, config)
            reward_pred = trainlib.reward_predictions_for_metrics(scorer_out["reward_pred"], config)
            value_target = trainlib.value_targets(batch, config)
            first_action_mse = F.mse_loss(policy_out["action_pred"][:, 0], batch["target_action"][:, 0])
            action_mse = F.mse_loss(policy_out["action_pred"], batch["target_action"])
            prior_plan = scorer_out["action_pred"][:, : config.prediction_horizon]
            policy_plan = policy_out["action_pred"][:, : config.prediction_horizon]
            prior_mse = F.mse_loss(policy_plan, prior_plan)
            success_context = batch["target_reward_raw"].max(dim=1).values >= 0.5
            add_total(totals, "action_mse", action_mse, batch_size)
            add_total(totals, "first_action_mse", first_action_mse, batch_size)
            add_total(totals, "prior_plan_mse", prior_mse, batch_size)
            add_total(totals, "initial_policy_value_mean", policy_out["value"].mean(), batch_size)
            add_total(totals, "success_context_fraction", success_context.float().mean(), batch_size)
            reward_probs.append(reward_pred.reshape(-1).detach().cpu())
            reward_targets.append(reward_target.reshape(-1).detach().cpu())
            values.append(policy_out["value"].reshape(-1).detach().cpu())
            value_targets.append(value_target.reshape(-1).detach().cpu())
            count += batch_size
    metrics = normalize_totals(totals, count)
    if reward_probs:
        reward_prob = torch.cat(reward_probs)
        reward_target = torch.cat(reward_targets)
        value = torch.cat(values)
        value_target = torch.cat(value_targets)
        metrics.update(
            {
                "reward_brier_scorer": float(F.mse_loss(reward_prob, reward_target).item()),
                "reward_accuracy_scorer": float(
                    ((reward_prob >= 0.5) == (reward_target >= 0.5)).float().mean().item()
                ),
                "reward_pred_mean_scorer": float(reward_prob.mean().item()),
                "reward_target_mean": float(reward_target.mean().item()),
                "value_mse": float(F.mse_loss(value, value_target).item()),
                "value_mae": float(F.l1_loss(value, value_target).item()),
                "value_corr": trainlib.pearson_corr(value, value_target),
                "value_target_mean": float(value_target.mean().item()),
                "value_pred_mean": float(value.mean().item()),
            }
        )
    metrics["num_samples"] = int(count)
    return metrics


def evaluate_closed_loop_rollout(
    *,
    policy_name: str,
    policy_model: Any,
    scorer_model: Any,
    loader: Any,
    config: Any,
    device: Any,
    horizon: int,
    max_batches: int,
) -> dict[str, Any]:
    policy_model.eval()
    scorer_model.eval()
    totals: dict[str, float] = {}
    success_totals: dict[str, float] = {}
    failure_totals: dict[str, float] = {}
    count = 0
    success_count = 0
    failure_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = trainlib.move_batch(batch, device)
            batch_size = int(batch["z_context"].shape[0])
            z_context = batch["z_context"]
            action_context = batch["action_context"]
            task_id = batch["task_id"]
            success_context = batch["target_reward_raw"].max(dim=1).values >= 0.5
            returns = torch.zeros(batch_size, device=device)
            reward_sum = torch.zeros(batch_size, device=device)
            prior_penalty_sum = torch.zeros(batch_size, device=device)
            latent_motion_sum = torch.zeros(batch_size, device=device)
            action_norm_sum = torch.zeros(batch_size, device=device)
            discount = 1.0
            initial_out = policy_model.agent(z_context, action_context, task_id)
            initial_value = initial_out["value"]
            for _ in range(int(horizon)):
                policy_out = policy_model.agent(z_context, action_context, task_id)
                scorer_out = scorer_model.agent(z_context, action_context, task_id)
                action_plan = policy_out["action_pred"][:, : config.prediction_horizon, :]
                prior_plan = scorer_out["action_pred"][:, : config.prediction_horizon, :]
                dynamics_out = policy_model.dynamics(z_context, action_context, action_plan, task_id)
                first_action = action_plan[:, 0, :]
                next_z = dynamics_out["predicted_z"][:, 0, :]
                next_z_context = torch.cat([z_context[:, 1:, :], next_z[:, None, :]], dim=1)
                next_action_context = torch.cat([action_context[:, 1:, :], first_action[:, None, :]], dim=1)
                reward_logits = scorer_model.agent(next_z_context, next_action_context, task_id)["reward_pred"][:, 0]
                reward = trainlib.reward_predictions_for_imagination(reward_logits, config)
                returns = returns + discount * reward
                reward_sum = reward_sum + reward
                prior_penalty_sum = prior_penalty_sum + F.mse_loss(action_plan, prior_plan, reduction="none").mean(
                    dim=(1, 2)
                )
                latent_motion_sum = latent_motion_sum + torch.linalg.vector_norm(next_z - z_context[:, -1, :], dim=1)
                action_norm_sum = action_norm_sum + torch.linalg.vector_norm(first_action, dim=1)
                z_context = next_z_context
                action_context = next_action_context
                discount *= float(config.gamma)
            final_value = policy_model.agent(z_context, action_context, task_id)["value"]
            sample_metrics = {
                "model_return": returns,
                "mean_reward": reward_sum / max(int(horizon), 1),
                "prior_plan_mse": prior_penalty_sum / max(int(horizon), 1),
                "latent_motion_norm": latent_motion_sum / max(int(horizon), 1),
                "action_norm": action_norm_sum / max(int(horizon), 1),
                "initial_value": initial_value,
                "final_value": final_value,
                "success_context": success_context.float(),
            }
            for key, value in sample_metrics.items():
                add_total(totals, key, value.mean(), batch_size)
            count += batch_size
            success_mask = success_context
            failure_mask = ~success_context
            if bool(success_mask.any()):
                n = int(success_mask.sum().item())
                success_count += n
                for key, value in sample_metrics.items():
                    add_total(success_totals, key, value[success_mask].mean(), n)
            if bool(failure_mask.any()):
                n = int(failure_mask.sum().item())
                failure_count += n
                for key, value in sample_metrics.items():
                    add_total(failure_totals, key, value[failure_mask].mean(), n)
    metrics = normalize_totals(totals, count)
    metrics["policy_name"] = policy_name
    metrics["num_samples"] = int(count)
    metrics["success_context_metrics"] = normalize_totals(success_totals, success_count)
    metrics["failure_context_metrics"] = normalize_totals(failure_totals, failure_count)
    metrics["success_context_samples"] = int(success_count)
    metrics["failure_context_samples"] = int(failure_count)
    return metrics


def evaluate_zero_action_rollout(
    *,
    scorer_model: Any,
    loader: Any,
    config: Any,
    device: Any,
    horizon: int,
    max_batches: int,
) -> dict[str, Any]:
    scorer_model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = trainlib.move_batch(batch, device)
            batch_size = int(batch["z_context"].shape[0])
            z_context = batch["z_context"]
            action_context = batch["action_context"]
            task_id = batch["task_id"]
            returns = torch.zeros(batch_size, device=device)
            reward_sum = torch.zeros(batch_size, device=device)
            latent_motion_sum = torch.zeros(batch_size, device=device)
            discount = 1.0
            for _ in range(int(horizon)):
                action_plan = torch.zeros(
                    batch_size,
                    config.prediction_horizon,
                    action_context.shape[-1],
                    device=device,
                    dtype=action_context.dtype,
                )
                dynamics_out = scorer_model.dynamics(z_context, action_context, action_plan, task_id)
                first_action = action_plan[:, 0, :]
                next_z = dynamics_out["predicted_z"][:, 0, :]
                next_z_context = torch.cat([z_context[:, 1:, :], next_z[:, None, :]], dim=1)
                next_action_context = torch.cat([action_context[:, 1:, :], first_action[:, None, :]], dim=1)
                reward_logits = scorer_model.agent(next_z_context, next_action_context, task_id)["reward_pred"][:, 0]
                reward = trainlib.reward_predictions_for_imagination(reward_logits, config)
                returns = returns + discount * reward
                reward_sum = reward_sum + reward
                latent_motion_sum = latent_motion_sum + torch.linalg.vector_norm(next_z - z_context[:, -1, :], dim=1)
                z_context = next_z_context
                action_context = next_action_context
                discount *= float(config.gamma)
            add_total(totals, "model_return", returns.mean(), batch_size)
            add_total(totals, "mean_reward", (reward_sum / max(int(horizon), 1)).mean(), batch_size)
            add_total(totals, "latent_motion_norm", (latent_motion_sum / max(int(horizon), 1)).mean(), batch_size)
            count += batch_size
    metrics = normalize_totals(totals, count)
    metrics["policy_name"] = "zero_action"
    metrics["num_samples"] = int(count)
    return metrics


def compare_policies(
    bc_rollout: dict[str, Any],
    imagination_rollout: dict[str, Any],
    bc_open_loop: dict[str, Any],
    imagination_open_loop: dict[str, Any],
) -> dict[str, Any]:
    bc_return = float(bc_rollout.get("model_return", 0.0))
    imagination_return = float(imagination_rollout.get("model_return", 0.0))
    bc_reward = float(bc_rollout.get("mean_reward", 0.0))
    imagination_reward = float(imagination_rollout.get("mean_reward", 0.0))
    prior_mse = float(imagination_rollout.get("prior_plan_mse", 0.0))
    action_mse_delta = float(imagination_open_loop.get("action_mse", 0.0)) - float(bc_open_loop.get("action_mse", 0.0))
    return {
        "return_delta_after_minus_bc": imagination_return - bc_return,
        "return_ratio_after_over_bc": safe_ratio(imagination_return, bc_return),
        "mean_reward_delta_after_minus_bc": imagination_reward - bc_reward,
        "after_prior_plan_mse": prior_mse,
        "open_loop_action_mse_delta_after_minus_bc": action_mse_delta,
        "model_transfer_improved": bool(imagination_return > bc_return and imagination_reward > bc_reward),
        "prior_constrained": bool(prior_mse <= 0.05),
        "interpretation": (
            "after_imagination improves learned-model rollout return over BC prior"
            if imagination_return > bc_return
            else "after_imagination does not improve learned-model rollout return over BC prior"
        ),
    }


def add_total(totals: dict[str, float], key: str, value: Any, count: int) -> None:
    if hasattr(value, "detach"):
        scalar = float(value.detach().cpu())
    else:
        scalar = float(value)
    totals[key] = totals.get(key, 0.0) + scalar * int(count)


def normalize_totals(totals: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(int(count), 1) for key, value in totals.items()}


def safe_ratio(num: float, denom: float) -> float:
    if abs(denom) <= 1e-12:
        return 0.0
    return float(num / denom)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_report(summary: dict[str, Any], path: Path) -> None:
    bc_rollout = summary["closed_loop_model_rollout"]["bc_prior"]
    after_rollout = summary["closed_loop_model_rollout"]["after_imagination"]
    zero_rollout = summary["closed_loop_model_rollout"]["zero_action"]
    bc_open = summary["open_loop"]["bc_prior"]
    after_open = summary["open_loop"]["after_imagination"]
    comparison = summary["comparison"]
    gate = summary["dynamics_gate"]
    lines = [
        "# SOAR Imagination Transfer Eval",
        "",
        "This is a SOAR-only learned-model transfer check. It does not use the drone simulator.",
        "",
        "## Setup",
        "",
        f"- Run dir: `{summary['run_dir']}`",
        f"- Validation anchors: `{summary['anchors']['val']}`",
        f"- Rollout horizon: `{summary['config']['rollout_horizon']}`",
        f"- Reward mode: `{summary['config']['reward_target_mode']} + {summary['config']['reward_loss_type']}`",
        f"- Value mode: `{summary['config']['value_target_mode']} + {summary['config']['value_loss_type']}`",
        "",
        "## Dynamics Gate",
        "",
        f"- Strict gate passed: `{gate.get('strict_gate_passed')}`",
        f"- Normal/persistence: `{gate.get('normal_over_persistence'):.4f}`",
        f"- Shuffle/normal: `{gate.get('shuffle_over_normal'):.4f}`",
        f"- Zero/normal: `{gate.get('zero_over_normal'):.4f}`",
        "",
        "## Closed-Loop Learned-Model Rollout",
        "",
        "| Policy | Model Return | Mean Reward | Prior MSE | Latent Motion | Action Norm |",
        "|---|---:|---:|---:|---:|---:|",
        format_rollout_row("zero_action", zero_rollout),
        format_rollout_row("bc_prior", bc_rollout),
        format_rollout_row("after_imagination", after_rollout),
        "",
        "## Open-Loop Held-Out Action Fit",
        "",
        "| Policy | Action MSE | First Action MSE | Value MSE | Value Corr |",
        "|---|---:|---:|---:|---:|",
        format_open_loop_row("bc_prior", bc_open),
        format_open_loop_row("after_imagination", after_open),
        "",
        "## Decision",
        "",
        f"- Return delta after - BC: `{comparison.get('return_delta_after_minus_bc'):.4f}`",
        f"- Mean reward delta after - BC: `{comparison.get('mean_reward_delta_after_minus_bc'):.4f}`",
        f"- Prior constrained: `{comparison.get('prior_constrained')}`",
        f"- Model-transfer improved: `{comparison.get('model_transfer_improved')}`",
        "",
        "Interpretation:",
        "",
        comparison.get("interpretation", ""),
        "",
        "Caveat: this is still learned-model transfer, not real SOAR environment replay.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_rollout_row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {float(metrics.get('model_return', 0.0)):.4f} | "
        f"{float(metrics.get('mean_reward', 0.0)):.4f} | "
        f"{float(metrics.get('prior_plan_mse', 0.0)):.4f} | "
        f"{float(metrics.get('latent_motion_norm', 0.0)):.4f} | "
        f"{float(metrics.get('action_norm', 0.0)):.4f} |"
    )


def format_open_loop_row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {float(metrics.get('action_mse', 0.0)):.4f} | "
        f"{float(metrics.get('first_action_mse', 0.0)):.4f} | "
        f"{float(metrics.get('value_mse', 0.0)):.4f} | "
        f"{float(metrics.get('value_corr', 0.0)):.4f} |"
    )


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": summary["phase"],
        "out_dir": summary["out_dir"],
        "anchors": summary["anchors"],
        "dynamics_gate": summary["dynamics_gate"],
        "closed_loop_model_rollout": summary["closed_loop_model_rollout"],
        "open_loop": summary["open_loop"],
        "comparison": summary["comparison"],
        "elapsed_s": summary["elapsed_s"],
    }


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
