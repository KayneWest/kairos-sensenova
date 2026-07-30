#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterator

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
    raise RuntimeError("torch is required for SOAR learned-dynamics eval.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the learned SOAR latent dynamics directly on held-out anchors, including "
            "true-action, shuffled-action, zero-action, and persistence controls."
        )
    )
    parser.add_argument(
        "--run-dir",
        default="sensenova_drone_agent/output/soar_dreamer_lite_reward_calibrated_freeze_value_ctx8_v1",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--sequence-cache", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--rollout-horizon", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    run_dir = resolve_path(args.run_dir)
    config = load_training_config(run_dir / "config.json")
    if args.sequence_cache:
        config.sequence_cache = str(resolve_path(args.sequence_cache))
    config.batch_size = int(args.batch_size)
    config.device = args.device
    config.seed = int(args.seed)

    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else run_dir / "best_imagination.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    out_dir = resolve_path(args.out_dir) if args.out_dir else run_dir / "learned_dynamics_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        future_action_window=config.future_action_window,
    )
    anchors, motion_filter = trainlib.filter_motion_anchors(cache, anchors, config)
    train_anchors, val_anchors = trainlib.split_for_config(cache, anchors, config)
    rollout_val_anchors = filter_rollout_anchors(
        cache,
        val_anchors,
        context_len=config.context_len,
        prediction_horizon=config.prediction_horizon,
        rollout_horizon=int(args.rollout_horizon),
        future_action_offset=config.future_action_offset,
        future_action_window=config.future_action_window,
    )
    if val_anchors.size <= 0:
        raise RuntimeError("No validation anchors available.")
    if rollout_val_anchors.size <= 0:
        raise RuntimeError("No validation anchors support the requested rollout horizon.")

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
    model = trainlib.SoarDreamerLiteModel(config, cache).to(device)
    trainlib.load_checkpoint(model, checkpoint, device)
    model.eval()

    single_pass = evaluate_single_pass_controls(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=int(args.max_batches),
    )
    autoregressive = evaluate_autoregressive_controls(
        model=model,
        cache=cache,
        anchors=rollout_val_anchors,
        normalizer=normalizer,
        config=config,
        device=device,
        rollout_horizon=int(args.rollout_horizon),
        batch_size=int(args.batch_size),
        max_batches=int(args.max_batches),
    )
    summary = {
        "phase": "soar_learned_dynamics_eval",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "out_dir": str(out_dir),
        "sequence_cache": config.sequence_cache,
        "config": {
            "context_len": config.context_len,
            "prediction_horizon": config.prediction_horizon,
            "rollout_horizon": int(args.rollout_horizon),
            "future_action_offset": config.future_action_offset,
            "future_action_window": config.future_action_window,
            "future_action_reduce": config.future_action_reduce,
            "motion_filter_quantile": config.motion_filter_quantile,
            "split_mode": config.split_mode,
        },
        "anchors": {
            "total": int(anchors.size),
            "train": int(train_anchors.size),
            "val": int(val_anchors.size),
            "rollout_val": int(rollout_val_anchors.size),
        },
        "motion_filter": motion_filter,
        "single_pass_controls": single_pass,
        "autoregressive_controls": autoregressive,
        "decision": make_decision(single_pass, autoregressive),
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


def filter_rollout_anchors(
    cache: Any,
    anchors: np.ndarray,
    *,
    context_len: int,
    prediction_horizon: int,
    rollout_horizon: int,
    future_action_offset: int,
    future_action_window: int,
) -> np.ndarray:
    valid: list[int] = []
    n = int(cache.num_steps)
    for anchor_raw in np.asarray(anchors, dtype=np.int64):
        anchor = int(anchor_raw)
        context_start = anchor - int(context_len) + 1
        action_start = anchor + int(future_action_offset)
        action_end = action_start + int(rollout_horizon) + int(prediction_horizon) + max(1, int(future_action_window)) - 3
        z_end = anchor + int(rollout_horizon)
        span_start = min(context_start, action_start)
        span_end = max(action_end, z_end)
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
        valid.append(anchor)
    return np.asarray(valid, dtype=np.int64)


def evaluate_single_pass_controls(
    *,
    model: Any,
    loader: Any,
    device: Any,
    max_batches: int,
) -> dict[str, Any]:
    totals: dict[str, torch.Tensor] = {}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = trainlib.move_batch(batch, device)
            batch_size = int(batch["z_context"].shape[0])
            normal = model.dynamics(
                batch["z_context"],
                batch["action_context"],
                batch["future_action"],
                batch["task_id"],
            )["predicted_z"]
            if batch_size > 1:
                perm = torch.randperm(batch_size, device=device)
                shuffled_actions = batch["future_action"][perm]
            else:
                shuffled_actions = torch.roll(batch["future_action"], shifts=1, dims=1)
            shuffled = model.dynamics(
                batch["z_context"],
                batch["action_context"],
                shuffled_actions,
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
            add_step_mse(totals, "normal", normal, target, batch_size)
            add_step_mse(totals, "shuffle", shuffled, target, batch_size)
            add_step_mse(totals, "zero", zeroed, target, batch_size)
            add_step_mse(totals, "persistence", persistence, target, batch_size)
            add_step_mse(totals, "normal_vs_zero_prediction_delta", normal, zeroed, batch_size)
            add_step_mse(totals, "normal_vs_shuffle_prediction_delta", normal, shuffled, batch_size)
            count += batch_size
    return summarize_control_totals(totals, count)


def evaluate_autoregressive_controls(
    *,
    model: Any,
    cache: Any,
    anchors: np.ndarray,
    normalizer: Any,
    config: Any,
    device: Any,
    rollout_horizon: int,
    batch_size: int,
    max_batches: int,
) -> dict[str, Any]:
    totals: dict[str, torch.Tensor] = {}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(
            iter_rollout_batches(
                cache,
                anchors,
                normalizer=normalizer,
                config=config,
                rollout_horizon=rollout_horizon,
                batch_size=batch_size,
                device=device,
            )
        ):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch_size_actual = int(batch["z_context"].shape[0])
            if batch_size_actual > 1:
                perm = torch.randperm(batch_size_actual, device=device)
                shuffled_actions = batch["future_actions"][perm]
            else:
                shuffled_actions = torch.roll(batch["future_actions"], shifts=1, dims=1)
            modes = {
                "normal": {
                    "z_context": batch["z_context"].clone(),
                    "action_context": batch["action_context"].clone(),
                    "future_actions": batch["future_actions"],
                },
                "shuffle": {
                    "z_context": batch["z_context"].clone(),
                    "action_context": batch["action_context"].clone(),
                    "future_actions": shuffled_actions,
                },
                "zero": {
                    "z_context": batch["z_context"].clone(),
                    "action_context": batch["action_context"].clone(),
                    "future_actions": torch.zeros_like(batch["future_actions"]),
                },
            }
            predictions: dict[str, list[Any]] = {key: [] for key in modes}
            for step in range(int(rollout_horizon)):
                for mode_name, state in modes.items():
                    action_plan = state["future_actions"][:, step : step + config.prediction_horizon, :]
                    dynamics_out = model.dynamics(
                        state["z_context"],
                        state["action_context"],
                        action_plan,
                        batch["task_id"],
                    )
                    next_z = dynamics_out["predicted_z"][:, 0, :]
                    first_action = action_plan[:, 0, :]
                    predictions[mode_name].append(next_z)
                    state["z_context"] = torch.cat([state["z_context"][:, 1:, :], next_z[:, None, :]], dim=1)
                    state["action_context"] = torch.cat(
                        [state["action_context"][:, 1:, :], first_action[:, None, :]],
                        dim=1,
                    )
            target = batch["target_z"]
            persistence = batch["z_context"][:, -1:, :].expand_as(target)
            normal_pred = torch.stack(predictions["normal"], dim=1)
            shuffle_pred = torch.stack(predictions["shuffle"], dim=1)
            zero_pred = torch.stack(predictions["zero"], dim=1)
            add_step_mse(totals, "normal", normal_pred, target, batch_size_actual)
            add_step_mse(totals, "shuffle", shuffle_pred, target, batch_size_actual)
            add_step_mse(totals, "zero", zero_pred, target, batch_size_actual)
            add_step_mse(totals, "persistence", persistence, target, batch_size_actual)
            add_step_mse(totals, "normal_vs_zero_prediction_delta", normal_pred, zero_pred, batch_size_actual)
            add_step_mse(totals, "normal_vs_shuffle_prediction_delta", normal_pred, shuffle_pred, batch_size_actual)
            count += batch_size_actual
    return summarize_control_totals(totals, count)


def iter_rollout_batches(
    cache: Any,
    anchors: np.ndarray,
    *,
    normalizer: Any,
    config: Any,
    rollout_horizon: int,
    batch_size: int,
    device: Any,
) -> Iterator[dict[str, Any]]:
    z = torch.from_numpy(trainlib.normalize_np(cache.z, normalizer.z_mean, normalizer.z_std)).float()
    action = torch.from_numpy(trainlib.normalize_np(cache.action, normalizer.action_mean, normalizer.action_std)).float()
    task_id = torch.from_numpy(cache.task_id.astype(np.int64))
    anchors = np.asarray(anchors, dtype=np.int64)
    action_window = int(rollout_horizon) + int(config.prediction_horizon) - 1
    future_action_window = max(1, int(config.future_action_window))
    for start in range(0, anchors.size, int(batch_size)):
        chunk = anchors[start : start + int(batch_size)]
        z_context = []
        action_context = []
        future_actions = []
        target_z = []
        task_ids = []
        for anchor_raw in chunk:
            anchor = int(anchor_raw)
            context_start = anchor - int(config.context_len) + 1
            action_start = anchor + int(config.future_action_offset)
            z_context.append(z[context_start : anchor + 1])
            action_context.append(action[context_start : anchor + 1])
            future_actions.append(
                future_action_sequence(
                    action,
                    action_start,
                    action_window,
                    window=future_action_window,
                    reduce=config.future_action_reduce,
                )
            )
            target_z.append(z[anchor + 1 : anchor + int(rollout_horizon) + 1])
            task_ids.append(task_id[anchor])
        yield {
            "z_context": torch.stack(z_context).to(device, non_blocking=True),
            "action_context": torch.stack(action_context).to(device, non_blocking=True),
            "future_actions": torch.stack(future_actions).to(device, non_blocking=True),
            "target_z": torch.stack(target_z).to(device, non_blocking=True),
            "task_id": torch.stack(task_ids).to(device, non_blocking=True),
        }


def future_action_sequence(
    action: Any,
    start: int,
    horizon: int,
    *,
    window: int,
    reduce: str,
) -> Any:
    values = []
    for offset in range(int(horizon)):
        chunk = action[start + offset : start + offset + int(window)]
        if reduce == "sum":
            values.append(chunk.sum(dim=0))
        elif reduce == "first":
            values.append(chunk[0])
        else:
            values.append(chunk.mean(dim=0))
    return torch.stack(values, dim=0)


def add_step_mse(totals: dict[str, Any], key: str, prediction: Any, target: Any, batch_size: int) -> None:
    mse_by_step = F.mse_loss(prediction, target, reduction="none").mean(dim=(0, 2)).detach()
    totals[key] = totals.get(key, torch.zeros_like(mse_by_step)) + mse_by_step * int(batch_size)


def summarize_control_totals(totals: dict[str, Any], count: int) -> dict[str, Any]:
    if count <= 0:
        return {}
    metrics: dict[str, Any] = {"num_samples": int(count)}
    averaged = {key: value / int(count) for key, value in totals.items()}
    for key, value in averaged.items():
        metrics[f"{key}_mse_by_step"] = [float(item) for item in value.detach().cpu()]
        metrics[f"{key}_mean_mse"] = float(value.mean().detach().cpu())
        metrics[f"{key}_final_mse"] = float(value[-1].detach().cpu())
    normal = max(metrics.get("normal_mean_mse", 0.0), 1e-12)
    persistence = max(metrics.get("persistence_mean_mse", 0.0), 1e-12)
    metrics["normal_over_persistence"] = metrics.get("normal_mean_mse", 0.0) / persistence
    metrics["shuffle_over_normal"] = metrics.get("shuffle_mean_mse", 0.0) / normal
    metrics["zero_over_normal"] = metrics.get("zero_mean_mse", 0.0) / normal
    metrics["action_conditioning_detected"] = bool(
        metrics.get("normal_mean_mse", 0.0) < metrics.get("persistence_mean_mse", 0.0)
        and metrics.get("shuffle_mean_mse", 0.0) > metrics.get("normal_mean_mse", 0.0)
        and metrics.get("zero_mean_mse", 0.0) > metrics.get("normal_mean_mse", 0.0)
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


def make_decision(single_pass: dict[str, Any], autoregressive: dict[str, Any]) -> dict[str, Any]:
    single_pass_strong = bool(single_pass.get("strict_gate_passed"))
    autoregressive_detected = bool(autoregressive.get("action_conditioning_detected"))
    autoregressive_strong = bool(autoregressive.get("strict_gate_passed"))
    if single_pass_strong and autoregressive_strong:
        recommendation = "Dynamics are action-conditioned at direct and autoregressive horizons; proceed to external policy eval."
    elif single_pass_strong and autoregressive_detected:
        recommendation = "Dynamics are action-conditioned but weaken over rollout; keep horizons short and evaluate externally."
    elif single_pass_strong:
        recommendation = "Dynamics are short-horizon action-conditioned only; use for local planning/eval, not long imagination claims."
    else:
        recommendation = "Dynamics gate is weak; return to action-conditioned dynamics training."
    return {
        "single_pass_strong": single_pass_strong,
        "autoregressive_action_conditioning_detected": autoregressive_detected,
        "autoregressive_strong": autoregressive_strong,
        "recommendation": recommendation,
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    single = summary.get("single_pass_controls", {})
    auto = summary.get("autoregressive_controls", {})
    return {
        "phase": summary.get("phase"),
        "out_dir": summary.get("out_dir"),
        "checkpoint": summary.get("checkpoint"),
        "anchors": summary.get("anchors"),
        "single_pass": {
            "normal_over_persistence": single.get("normal_over_persistence"),
            "shuffle_over_normal": single.get("shuffle_over_normal"),
            "zero_over_normal": single.get("zero_over_normal"),
            "strict_gate_passed": single.get("strict_gate_passed"),
        },
        "autoregressive": {
            "normal_over_persistence": auto.get("normal_over_persistence"),
            "shuffle_over_normal": auto.get("shuffle_over_normal"),
            "zero_over_normal": auto.get("zero_over_normal"),
            "strict_gate_passed": auto.get("strict_gate_passed"),
            "action_conditioning_detected": auto.get("action_conditioning_detected"),
        },
        "decision": summary.get("decision"),
        "elapsed_s": summary.get("elapsed_s"),
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    single = summary.get("single_pass_controls", {})
    auto = summary.get("autoregressive_controls", {})
    decision = summary.get("decision", {})
    lines = [
        "# SOAR Learned Dynamics Eval",
        "",
        "This evaluates the learned latent dynamics directly on held-out SOAR anchors.",
        "",
        "## Inputs",
        "",
        f"- Run dir: `{summary.get('run_dir')}`",
        f"- Checkpoint: `{summary.get('checkpoint')}`",
        f"- Sequence cache: `{summary.get('sequence_cache')}`",
        f"- Anchors: `{summary.get('anchors')}`",
        "",
        "## Single-Pass Controls",
        "",
        f"- Normal/persistence: `{single.get('normal_over_persistence')}`",
        f"- Shuffled/normal: `{single.get('shuffle_over_normal')}`",
        f"- Zero/normal: `{single.get('zero_over_normal')}`",
        f"- Strict gate passed: `{single.get('strict_gate_passed')}`",
        f"- Strength: `{single.get('action_conditioning_strength')}`",
        "",
        "## Autoregressive Controls",
        "",
        f"- Rollout horizon: `{summary.get('config', {}).get('rollout_horizon')}`",
        f"- Normal/persistence: `{auto.get('normal_over_persistence')}`",
        f"- Shuffled/normal: `{auto.get('shuffle_over_normal')}`",
        f"- Zero/normal: `{auto.get('zero_over_normal')}`",
        f"- Strict gate passed: `{auto.get('strict_gate_passed')}`",
        f"- Action conditioning detected: `{auto.get('action_conditioning_detected')}`",
        f"- Strength: `{auto.get('action_conditioning_strength')}`",
        "",
        "## Decision",
        "",
        f"- Recommendation: {decision.get('recommendation')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
