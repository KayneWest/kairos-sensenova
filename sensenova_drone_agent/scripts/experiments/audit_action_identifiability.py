#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
DREAMER4_ROOT = REPO_ROOT / "dreamer4" / "dreamer4"
if str(DREAMER4_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMER4_ROOT))

from wm_dataset import WMDataset, collate_batch  # noqa: E402


@dataclass(frozen=True)
class SourceSpec:
    name: str
    raw_dirs: list[Path]
    frame_dirs: list[Path]
    tasks_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether local action-labeled video data exposes a learnable "
            "action signal beyond scene-history prediction."
        )
    )
    parser.add_argument("--out", default="sensenova_drone_agent/output/action_identifiability_audit_v1")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--horizons", default="1,4,8")
    parser.add_argument("--samples-per-source", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--probe-size", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=49)
    parser.add_argument("--action-features", default="current,prev,delta,mean4,norm")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--source",
        action="append",
        choices=[
            "dreamer4_expert",
            "dreamer4_mixed_small",
            "dreamer4_mixed_large",
            "soar",
            "droid",
            "robonet",
            "game_actions_v1",
            "game_actions_blocks_v1",
            "all",
        ],
        help="Source(s) to audit. Defaults to all named sources separately.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(item.strip()) for item in str(args.horizons).split(",") if item.strip()]

    selected = args.source or ["dreamer4_expert", "dreamer4_mixed_small", "dreamer4_mixed_large", "soar", "droid", "robonet"]
    specs = build_sources()
    if "all" in selected:
        selected = list(specs)

    rows = []
    for source_name in selected:
        spec = specs[source_name]
        print(f"[audit] source={source_name}")
        rows.append(audit_source(spec, args=args, horizons=horizons))

    payload = {
        "phase": "action_identifiability_audit",
        "out": str(out_dir),
        "config": {
            "seq_len": int(args.seq_len),
            "ctx_len": int(args.ctx_len),
            "horizons": horizons,
            "samples_per_source": int(args.samples_per_source),
            "batch_size": int(args.batch_size),
            "image_size": int(args.image_size),
            "probe_size": int(args.probe_size),
            "action_dim": int(args.action_dim),
            "action_features": str(args.action_features),
            "ridge": float(args.ridge),
            "train_frac": float(args.train_frac),
            "seed": int(args.seed),
        },
        "sources": rows,
        "elapsed_s": time.time() - started,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def build_sources() -> dict[str, SourceSpec]:
    dreamer_raw = REPO_ROOT / "sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4"
    dreamer_frames = REPO_ROOT / "sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4_shards_full"
    soar = REPO_ROOT / "sensenova_drone_agent/data/robotics/soar/dreamer4_soar_native_v2_action_contrast"
    robonet = REPO_ROOT / "sensenova_drone_agent/data/robotics/robonet/dreamer4_robonet_sample_64"
    droid = REPO_ROOT / "sensenova_drone_agent/data/robotics/hf_action_exports/droid_lerobot_dreamer4"
    dreamer_tasks = REPO_ROOT / "dreamer4/tasks.json"
    game_actions = REPO_ROOT / "sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_v1"
    game_actions_blocks = REPO_ROOT / "sensenova_drone_agent/data/game_action_sources/dreamer4_game_actions_blocks_v1"
    return {
        "dreamer4_expert": SourceSpec(
            "dreamer4_expert",
            [dreamer_raw / "expert"],
            [dreamer_frames / "expert"],
            dreamer_tasks,
        ),
        "dreamer4_mixed_small": SourceSpec(
            "dreamer4_mixed_small",
            [dreamer_raw / "mixed-small"],
            [dreamer_frames / "mixed-small"],
            dreamer_tasks,
        ),
        "dreamer4_mixed_large": SourceSpec(
            "dreamer4_mixed_large",
            [dreamer_raw / "mixed-large"],
            [dreamer_frames / "mixed-large"],
            dreamer_tasks,
        ),
        "soar": SourceSpec(
            "soar",
            [soar / "raw"],
            [soar / "frames"],
            soar / "tasks.json",
        ),
        "droid": SourceSpec(
            "droid",
            [droid / "raw"],
            [droid / "frames"],
            droid / "tasks.json",
        ),
        "robonet": SourceSpec(
            "robonet",
            [robonet / "raw"],
            [robonet / "frames"],
            robonet / "tasks.json",
        ),
        "game_actions_v1": SourceSpec(
            "game_actions_v1",
            [game_actions / "raw"],
            [game_actions / "frames"],
            game_actions / "tasks.json",
        ),
        "game_actions_blocks_v1": SourceSpec(
            "game_actions_blocks_v1",
            [game_actions_blocks / "raw"],
            [game_actions_blocks / "frames"],
            game_actions_blocks / "tasks.json",
        ),
    }


def audit_source(spec: SourceSpec, *, args: argparse.Namespace, horizons: list[int]) -> dict[str, Any]:
    dataset = WMDataset(
        data_dir=[str(path) for path in spec.raw_dirs],
        frames_dir=[str(path) for path in spec.frame_dirs],
        seq_len=int(args.seq_len),
        img_size=int(args.image_size),
        action_dim=int(args.action_dim),
        tasks_json=str(spec.tasks_json),
        tasks=None,
        strict_tasks=False,
        action_features=str(args.action_features),
        verbose=False,
    )
    if len(dataset) == 0:
        return {"name": spec.name, "error": "empty dataset"}

    rng = random.Random(int(args.seed) + stable_int(spec.name))
    n = min(int(args.samples_per_source), len(dataset))
    indices = rng.sample(range(len(dataset)), n) if n < len(dataset) else list(range(len(dataset)))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_batch,
    )

    frame_batches: list[torch.Tensor] = []
    action_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    reward_batches: list[torch.Tensor] = []
    for batch in loader:
        obs = batch["obs"].float() / 255.0
        emb = frame_embed(obs, size=int(args.probe_size))
        frame_batches.append(emb)
        action_batches.append(batch["act"].float())
        mask_batches.append(batch["act_mask"].float())
        reward_batches.append(batch["rew"].float())

    frames = torch.cat(frame_batches, dim=0)
    actions = torch.cat(action_batches, dim=0)
    masks = torch.cat(mask_batches, dim=0)
    rewards = torch.cat(reward_batches, dim=0)
    actions = torch.nan_to_num(actions, nan=0.0) * masks

    ctx = int(args.ctx_len)
    if ctx < 2 or ctx >= frames.shape[1]:
        raise ValueError(f"ctx_len={ctx} incompatible with frame length={frames.shape[1]}")
    scene = torch.cat(
        [
            frames[:, ctx],
            frames[:, ctx] - frames[:, ctx - 1],
            frames[:, ctx - 1] - frames[:, ctx - 2],
        ],
        dim=-1,
    )

    action_stats = compute_action_stats(actions, masks)
    reward_stats = compute_reward_stats(rewards)
    horizon_rows = []
    for horizon in horizons:
        if ctx + horizon >= frames.shape[1]:
            continue
        action_feat = actions[:, ctx : ctx + horizon].reshape(actions.shape[0], -1)
        target = frames[:, ctx + horizon] - frames[:, ctx]
        horizon_rows.append(
            audit_horizon(
                scene=scene,
                action=action_feat,
                target=target,
                train_frac=float(args.train_frac),
                ridge=float(args.ridge),
                seed=int(args.seed) + stable_int(spec.name) + horizon,
                horizon=horizon,
            )
        )

    return {
        "name": spec.name,
        "num_windows": int(len(dataset)),
        "sampled_windows": int(frames.shape[0]),
        "num_tasks": int(getattr(dataset, "num_tasks", len(getattr(dataset, "tasks", [])))),
        "action_stats": action_stats,
        "reward_stats": reward_stats,
        "horizons": horizon_rows,
        "decision": decide_source(action_stats, horizon_rows),
    }


def frame_embed(obs: torch.Tensor, *, size: int) -> torch.Tensor:
    # obs: (B,T,C,H,W) in [0,1]. Use a tiny grayscale pixel probe so the audit
    # measures data identifiability rather than capacity of a deep visual model.
    gray = obs.mean(dim=2)
    small = F.interpolate(
        gray.reshape(-1, 1, gray.shape[-2], gray.shape[-1]),
        size=(size, size),
        mode="area",
    )
    return small.reshape(obs.shape[0], obs.shape[1], size * size)


def compute_action_stats(actions: torch.Tensor, masks: torch.Tensor) -> dict[str, float]:
    active = masks > 0.5
    vals = actions[active]
    if vals.numel() == 0:
        return {
            "active_dims": 0.0,
            "abs_mean": 0.0,
            "std": 0.0,
            "nonzero_fraction": 0.0,
            "per_step_norm_mean": 0.0,
            "per_step_norm_std": 0.0,
        }
    per_step_norm = actions.abs().mean(dim=-1)
    active_dims = (masks.amax(dim=(0, 1)) > 0.5).sum()
    return {
        "active_dims": float(active_dims.item()),
        "abs_mean": float(vals.abs().mean().item()),
        "std": float(vals.std(unbiased=False).item()),
        "nonzero_fraction": float((vals.abs() > 1e-6).float().mean().item()),
        "per_step_norm_mean": float(per_step_norm.mean().item()),
        "per_step_norm_std": float(per_step_norm.std(unbiased=False).item()),
    }


def compute_reward_stats(rewards: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(rewards.mean().item()),
        "std": float(rewards.std(unbiased=False).item()),
        "nonzero_fraction": float((rewards.abs() > 1e-6).float().mean().item()),
        "positive_fraction": float((rewards > 1e-6).float().mean().item()),
    }


def audit_horizon(
    *,
    scene: torch.Tensor,
    action: torch.Tensor,
    target: torch.Tensor,
    train_frac: float,
    ridge: float,
    seed: int,
    horizon: int,
) -> dict[str, Any]:
    n = scene.shape[0]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_train = max(16, min(n - 1, int(round(n * train_frac))))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    if test_idx.numel() == 0:
        test_idx = train_idx

    y_train = target[train_idx]
    y_test = target[test_idx]
    mean_pred = y_train.mean(dim=0, keepdim=True).expand_as(y_test)
    mean_mse = mse(mean_pred, y_test)
    target_energy = mse(torch.zeros_like(y_test), y_test)

    x_scene = scene
    x_action = action
    x_both = torch.cat([scene, action], dim=-1)

    scene_mse = ridge_mse(x_scene[train_idx], y_train, x_scene[test_idx], y_test, ridge=ridge)
    action_mse = ridge_mse(x_action[train_idx], y_train, x_action[test_idx], y_test, ridge=ridge)
    both_mse = ridge_mse(x_both[train_idx], y_train, x_both[test_idx], y_test, ridge=ridge)

    action_norm = action.abs().mean(dim=-1)
    target_norm = target.pow(2).mean(dim=-1).sqrt()
    corr = pearson(action_norm, target_norm)
    action_energy = float(action.pow(2).mean().item())

    return {
        "horizon": int(horizon),
        "train_windows": int(train_idx.numel()),
        "test_windows": int(test_idx.numel()),
        "target_mse_to_zero": target_energy,
        "target_mse_to_mean": mean_mse,
        "scene_mse": scene_mse,
        "action_mse": action_mse,
        "scene_action_mse": both_mse,
        "scene_r2_vs_mean": r2(mean_mse, scene_mse),
        "action_r2_vs_mean": r2(mean_mse, action_mse),
        "scene_action_r2_vs_mean": r2(mean_mse, both_mse),
        "action_incremental_r2_vs_scene": r2(scene_mse, both_mse),
        "action_norm_target_norm_corr": corr,
        "action_energy": action_energy,
    }


def ridge_mse(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    ridge: float,
) -> float:
    x_train, x_mean, x_std = standardize_features(x_train.float())
    x_test = apply_standardize(x_test.float(), x_mean, x_std)
    y_train = y_train.float()
    y_test = y_test.float()

    ones_train = torch.ones((x_train.shape[0], 1), dtype=x_train.dtype)
    ones_test = torch.ones((x_test.shape[0], 1), dtype=x_test.dtype)
    x_train_aug = torch.cat([x_train, ones_train], dim=-1)
    x_test_aug = torch.cat([x_test, ones_test], dim=-1)

    xtx = x_train_aug.T @ x_train_aug
    eye = torch.eye(xtx.shape[0], dtype=xtx.dtype)
    eye[-1, -1] = 0.0
    xty = x_train_aug.T @ y_train
    try:
        weights = torch.linalg.solve(xtx + float(ridge) * eye, xty)
    except RuntimeError:
        weights = torch.linalg.pinv(xtx + float(ridge) * eye) @ xty
    pred = x_test_aug @ weights
    return mse(pred, y_test)


def standardize_features(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (x - mean) / std, mean, std


def apply_standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred.float() - target.float()).pow(2).mean().item())


def r2(baseline_mse: float, model_mse: float) -> float:
    if not math.isfinite(baseline_mse) or abs(baseline_mse) < 1e-12:
        return 0.0
    return float(1.0 - (model_mse / baseline_mse))


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.pow(2).mean().sqrt() * y.pow(2).mean().sqrt()
    if float(denom.item()) < 1e-12:
        return 0.0
    return float((x * y).mean().div(denom).item())


def decide_source(action_stats: dict[str, float], horizons: list[dict[str, Any]]) -> dict[str, Any]:
    best_inc = max((float(row["action_incremental_r2_vs_scene"]) for row in horizons), default=0.0)
    best_action_only = max((float(row["action_r2_vs_mean"]) for row in horizons), default=0.0)
    usable = (
        action_stats["active_dims"] > 0
        and action_stats["std"] > 1e-4
        and action_stats["nonzero_fraction"] > 0.05
    )
    return {
        "actions_have_variance": bool(usable),
        "best_action_incremental_r2_vs_scene": best_inc,
        "best_action_only_r2_vs_mean": best_action_only,
        "data_action_signal_detected": bool(usable and (best_inc > 0.01 or best_action_only > 0.05)),
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Action Identifiability Audit",
        "",
        "This audit asks whether actions in the local datasets predict future visual change beyond scene-history features.",
        "",
        "| Source | Windows | Active action dims | Nonzero action frac | Best action incremental R2 vs scene | Best action-only R2 vs mean | Signal detected |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["sources"]:
        if "error" in row:
            lines.append(f"| {row['name']} | 0 | 0 | 0 | 0 | 0 | error: {row['error']} |")
            continue
        stats = row["action_stats"]
        dec = row["decision"]
        lines.append(
            "| {name} | {windows} | {dims:.0f} | {nz:.3f} | {inc:.4f} | {only:.4f} | {signal} |".format(
                name=row["name"],
                windows=row["sampled_windows"],
                dims=stats["active_dims"],
                nz=stats["nonzero_fraction"],
                inc=dec["best_action_incremental_r2_vs_scene"],
                only=dec["best_action_only_r2_vs_mean"],
                signal=dec["data_action_signal_detected"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `action_incremental_r2_vs_scene` is the important number: positive means scene+action predicts future visual delta better than scene-only.",
            "- Small positive values mean action signal exists but is weak relative to scene persistence/history.",
            "- Negative or near-zero values mean this probe cannot find useful action conditioning in the sampled pixels.",
        ]
    )
    return "\n".join(lines) + "\n"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def stable_int(text: str) -> int:
    total = 0
    for idx, char in enumerate(text):
        total += (idx + 1) * ord(char)
    return total


if __name__ == "__main__":
    raise SystemExit(main())
