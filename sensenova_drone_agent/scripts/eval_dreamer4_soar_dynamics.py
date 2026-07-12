#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DREAMER4_ROOT = REPO_ROOT / "dreamer4" / "dreamer4"
if str(DREAMER4_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMER4_ROOT))

from train_dynamics import (  # noqa: E402
    DEFAULT_ACTION_DIM,
    Dynamics,
    align_actions_to_frames,
    dynamics_pretrain_loss,
    load_frozen_tokenizer_from_pt_ckpt,
    make_tau_schedule,
    pack_bottleneck_to_spatial,
    sample_autoregressive_packed_sequence,
    temporal_patchify,
)
from wm_dataset import WMDataset, collate_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate native Dreamer4-style SOAR dynamics action grounding.")
    parser.add_argument("--data-dir", action="append", required=True)
    parser.add_argument("--frames-dir", action="append", required=True)
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--tokenizer-ckpt", required=True)
    parser.add_argument("--dynamics-ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--eval-d", type=float, default=0.25)
    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="Override padded action width. Defaults to dynamics checkpoint args, then 16 for old checkpoints.",
    )
    parser.add_argument(
        "--action-features",
        default=None,
        help="Override comma-separated action features. Defaults to dynamics checkpoint args, then current.",
    )
    parser.add_argument("--action-frame-offset", type=int, default=0)
    parser.add_argument("--require-non-noop", action="store_true")
    parser.add_argument("--no-op-threshold", type=float, default=0.0)
    parser.add_argument("--min-non-noop-steps", type=int, default=1)
    parser.add_argument(
        "--reward-filter-mode",
        default="none",
        choices=["none", "positive_sum", "abs_sum", "any_positive", "any_abs"],
    )
    parser.add_argument("--reward-signal-threshold", type=float, default=0.0)
    parser.add_argument("--min-reward-signal-steps", type=int, default=1)
    parser.add_argument("--require-visual-delta", action="store_true")
    parser.add_argument("--visual-delta-threshold", type=float, default=0.0)
    parser.add_argument("--min-visual-delta-steps", type=int, default=1)
    parser.add_argument("--visual-delta-stride", type=int, default=4)
    parser.add_argument("--causal-min-ratio", type=float, default=1.02)
    parser.add_argument(
        "--negative-modes",
        default="shuffle,zero",
        help=(
            "Comma-separated action counterfactuals to evaluate. Supported examples: "
            "shuffle, zero, time_shift, time_shift2, time_shift4, time_reverse, time_perm."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    tokenizer_ckpt = resolve_path(args.tokenizer_ckpt)
    dynamics_ckpt = resolve_path(args.dynamics_ckpt)
    encoder, _decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(tokenizer_ckpt), device=device)

    dyn_ckpt = torch.load(dynamics_ckpt, map_location="cpu", weights_only=False)
    dyn_args = dict(dyn_ckpt.get("args", {}))
    action_dim = int(args.action_dim if args.action_dim is not None else dyn_args.get("action_dim", DEFAULT_ACTION_DIM))
    action_features = str(args.action_features if args.action_features is not None else dyn_args.get("action_features", "current"))
    dyn_args["action_dim"] = action_dim
    dyn_args["action_features"] = action_features
    model = build_dynamics(dyn_args, tok_args).to(device)
    model.load_state_dict(dyn_ckpt["dynamics"], strict=True)
    model.eval()

    dataset = WMDataset(
        data_dir=[str(resolve_path(path)) for path in args.data_dir],
        frames_dir=[str(resolve_path(path)) for path in args.frames_dir],
        seq_len=int(args.seq_len),
        img_size=128,
        action_dim=action_dim,
        tasks_json=str(resolve_path(args.tasks_json)),
        tasks=None,
        strict_tasks=False,
        action_features=action_features,
        require_non_noop=bool(args.require_non_noop),
        no_op_threshold=float(args.no_op_threshold),
        min_non_noop_steps=int(args.min_non_noop_steps),
        reward_filter_mode=str(args.reward_filter_mode),
        reward_signal_threshold=float(args.reward_signal_threshold),
        min_reward_signal_steps=int(args.min_reward_signal_steps),
        require_visual_delta=bool(args.require_visual_delta),
        visual_delta_threshold=float(args.visual_delta_threshold),
        min_visual_delta_steps=int(args.min_visual_delta_steps),
        visual_delta_stride=int(args.visual_delta_stride),
        verbose=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_batch,
    )

    metrics = evaluate(
        model=model,
        encoder=encoder,
        loader=loader,
        tok_args=tok_args,
        dyn_args=dyn_args,
        device=device,
        max_batches=int(args.max_batches),
        rollout_horizon=int(args.rollout_horizon),
        ctx_len=int(args.ctx_len),
        eval_d=float(args.eval_d),
        action_frame_offset=int(args.action_frame_offset),
        seed=int(args.seed),
        causal_min_ratio=float(args.causal_min_ratio),
        negative_modes=parse_negative_modes(args.negative_modes),
    )
    payload = {
        "phase": "dreamer4_soar_native_dynamics_eval",
        "data_dir": [str(resolve_path(path)) for path in args.data_dir],
        "frames_dir": [str(resolve_path(path)) for path in args.frames_dir],
        "tokenizer_ckpt": str(tokenizer_ckpt),
        "dynamics_ckpt": str(dynamics_ckpt),
        "out": str(out_path),
        "config": {
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
            "max_batches": int(args.max_batches),
            "rollout_horizon": int(args.rollout_horizon),
            "ctx_len": int(args.ctx_len),
            "eval_d": float(args.eval_d),
            "action_dim": action_dim,
            "action_features": action_features,
            "action_frame_offset": int(args.action_frame_offset),
            "require_non_noop": bool(args.require_non_noop),
            "no_op_threshold": float(args.no_op_threshold),
            "min_non_noop_steps": int(args.min_non_noop_steps),
            "reward_filter_mode": str(args.reward_filter_mode),
            "reward_signal_threshold": float(args.reward_signal_threshold),
            "min_reward_signal_steps": int(args.min_reward_signal_steps),
            "require_visual_delta": bool(args.require_visual_delta),
            "visual_delta_threshold": float(args.visual_delta_threshold),
            "min_visual_delta_steps": int(args.min_visual_delta_steps),
            "visual_delta_stride": int(args.visual_delta_stride),
            "causal_min_ratio": float(args.causal_min_ratio),
            "negative_modes": parse_negative_modes(args.negative_modes),
        },
        "metrics": metrics,
        "decision": decide(
            metrics,
            causal_min_ratio=float(args.causal_min_ratio),
            negative_modes=parse_negative_modes(args.negative_modes),
        ),
        "elapsed_s": time.time() - started,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def build_dynamics(dyn_args: dict[str, Any], tok_args: dict[str, Any]) -> Any:
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


@torch.no_grad()
def evaluate(
    *,
    model: Any,
    encoder: Any,
    loader: Any,
    tok_args: dict[str, Any],
    dyn_args: dict[str, Any],
    device: torch.device,
    max_batches: int,
    rollout_horizon: int,
    ctx_len: int,
    eval_d: float,
    action_frame_offset: int,
    seed: int,
    causal_min_ratio: float,
    negative_modes: list[str],
) -> dict[str, Any]:
    direct = {"normal": [], **{mode: [] for mode in negative_modes}}
    rollout = {"normal": [], **{mode: [] for mode in negative_modes}, "persistence": []}
    direct_pair_pass = {mode: [] for mode in negative_modes}
    rollout_pair_pass = {mode: [] for mode in negative_modes}
    k_max = int(dyn_args.get("k_max", 8))
    packing_factor = int(dyn_args.get("packing_factor", 2))
    n_latents = int(tok_args.get("n_latents", 16))
    n_spatial = n_latents // packing_factor
    patch = int(tok_args.get("patch", 8))
    sched = make_tau_schedule(k_max=k_max, schedule="shortcut", d=eval_d)
    batches_seen = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batches_seen += 1
        frames, actions, act_mask = prepare_batch(batch, device=device, action_frame_offset=action_frame_offset)
        z_gt = encode_frames(encoder, frames, patch=patch, n_spatial=n_spatial, packing_factor=packing_factor)
        action_variants, mask_variants = make_action_variants(
            actions,
            act_mask,
            z_gt=z_gt,
            batch_idx=batch_idx,
            negative_modes=negative_modes,
        )

        direct_batch = {}
        for name, action_tensor in action_variants.items():
            torch.manual_seed(seed + batch_idx)
            loss, aux = dynamics_pretrain_loss(
                model,
                z1=z_gt,
                actions=action_tensor,
                act_mask=mask_variants[name],
                k_max=k_max,
                B_self=0,
                step=0,
                bootstrap_start=10**9,
                agent_tokens=None,
            )
            direct_batch[name] = float(aux["flow_mse"].detach().cpu().item())
            direct[name].append(direct_batch[name])
        for mode in negative_modes:
            direct_pair_pass[mode].append(float(direct_batch[mode] > direct_batch["normal"] * float(causal_min_ratio)))

        horizon = min(int(rollout_horizon), max(1, z_gt.shape[1] - int(ctx_len)))
        if horizon > 0:
            rollout_batch = {}
            for name, action_tensor in action_variants.items():
                torch.manual_seed(seed + 10_000 + batch_idx)
                pred = sample_autoregressive_packed_sequence(
                    model,
                    z_gt_packed=z_gt,
                    ctx_length=int(ctx_len),
                    horizon=horizon,
                    k_max=k_max,
                    sched=sched,
                    actions=action_tensor,
                    act_mask=mask_variants[name],
                )
                rollout_batch[name] = horizon_mse(pred, z_gt, ctx_len=int(ctx_len), horizon=horizon)
                rollout[name].append(rollout_batch[name])
            floor = z_gt.clone()
            floor[:, int(ctx_len) : int(ctx_len) + horizon] = z_gt[:, int(ctx_len) - 1 : int(ctx_len)].expand(
                -1, horizon, -1, -1
            )
            rollout["persistence"].append(horizon_mse(floor, z_gt, ctx_len=int(ctx_len), horizon=horizon))
            for mode in negative_modes:
                rollout_pair_pass[mode].append(
                    float(rollout_batch[mode] > rollout_batch["normal"] * float(causal_min_ratio))
                )

    direct_mean = {key: mean(values) for key, values in direct.items()}
    rollout_mean = {key: mean(values) for key, values in rollout.items()}
    return {
        "batches": int(batches_seen),
        "direct": {
            **direct_mean,
            **{
                f"{mode}_over_normal": safe_ratio(direct_mean[mode], direct_mean["normal"])
                for mode in negative_modes
            },
            **{f"{mode}_pair_pass_fraction": mean(direct_pair_pass[mode]) for mode in negative_modes},
        },
        "autoregressive": {
            **rollout_mean,
            "normal_over_persistence": safe_ratio(rollout_mean["normal"], rollout_mean["persistence"]),
            **{
                f"{mode}_over_normal": safe_ratio(rollout_mean[mode], rollout_mean["normal"])
                for mode in negative_modes
            },
            **{f"{mode}_pair_pass_fraction": mean(rollout_pair_pass[mode]) for mode in negative_modes},
        },
    }


def prepare_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
    action_frame_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_u8 = batch["obs"].to(device, non_blocking=True)
    act = batch["act"].to(device, non_blocking=True)
    mask = batch["act_mask"].to(device, non_blocking=True)
    act = act.clamp(-1, 1) * mask
    frames = obs_u8[:, :-1].float() / 255.0
    actions, act_mask = align_actions_to_frames(
        act,
        mask,
        frame_count=frames.shape[1],
        action_frame_offset=action_frame_offset,
    )
    return frames, actions, act_mask


def encode_frames(
    encoder: Any,
    frames: torch.Tensor,
    *,
    patch: int,
    n_spatial: int,
    packing_factor: int,
) -> torch.Tensor:
    patches = temporal_patchify(frames, patch)
    z_btld, _ = encoder(patches)
    return pack_bottleneck_to_spatial(z_btld, n_spatial=n_spatial, k=packing_factor)


def make_action_variants(
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    *,
    z_gt: torch.Tensor | None = None,
    batch_idx: int,
    negative_modes: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    generator = torch.Generator(device=actions.device)
    generator.manual_seed(1234 + int(batch_idx))
    perm = torch.randperm(actions.shape[0], device=actions.device, generator=generator)
    variants = {"normal": actions}
    masks = {"normal": act_mask}
    for mode in negative_modes:
        if mode == "shuffle":
            variants[mode] = actions[perm]
            masks[mode] = act_mask[perm]
        elif mode in {"far_shuffle", "action_far_shuffle", "distance_shuffle"}:
            far_perm = farthest_action_permutation(actions, act_mask)
            variants[mode] = actions[far_perm]
            masks[mode] = act_mask[far_perm]
        elif mode in {"effect_far_shuffle", "action_effect_shuffle", "far_effect_shuffle"}:
            if z_gt is None:
                effect_perm = farthest_action_permutation(actions, act_mask)
            else:
                effect_perm = farthest_action_effect_permutation(actions, act_mask, z_gt)
            variants[mode] = actions[effect_perm]
            masks[mode] = act_mask[effect_perm]
        elif mode == "zero":
            variants[mode] = torch.zeros_like(actions)
            masks[mode] = torch.zeros_like(act_mask)
        elif (shift := parse_temporal_shift(mode)) is not None:
            variants[mode] = torch.roll(actions, shifts=shift, dims=1)
            masks[mode] = torch.roll(act_mask, shifts=shift, dims=1)
        elif mode in {"time_reverse", "reverse", "temporal_reverse"}:
            variants[mode] = actions.flip(dims=[1])
            masks[mode] = act_mask.flip(dims=[1])
        elif mode in {"time_perm", "temporal_perm", "same_traj_shuffle", "window_shuffle"}:
            idx = torch.stack(
                [torch.randperm(actions.shape[1], device=actions.device, generator=generator) for _ in range(actions.shape[0])],
                dim=0,
            )
            gather_idx = idx[:, :, None].expand(-1, -1, actions.shape[-1])
            variants[mode] = actions.gather(1, gather_idx)
            masks[mode] = act_mask.gather(1, gather_idx)
        else:
            raise ValueError(f"unknown negative mode: {mode}")
    return variants, masks


def farthest_action_permutation(actions: torch.Tensor, act_mask: torch.Tensor) -> torch.Tensor:
    flat = (actions.float() * act_mask.float()).flatten(1)
    if flat.shape[0] <= 1:
        return torch.arange(flat.shape[0], device=actions.device)
    dist = torch.cdist(flat, flat, p=2)
    dist.fill_diagonal_(-1.0)
    return dist.argmax(dim=1)


def farthest_action_effect_permutation(
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    z_gt: torch.Tensor,
) -> torch.Tensor:
    if actions.shape[0] <= 1:
        return torch.arange(actions.shape[0], device=actions.device)
    action_flat = (actions.float() * act_mask.float()).flatten(1)
    action_dist = torch.cdist(action_flat, action_flat, p=2)
    if z_gt.shape[1] > 1:
        effect_flat = (z_gt[:, 1:].float() - z_gt[:, :-1].float()).flatten(1)
    else:
        effect_flat = z_gt.float().flatten(1)
    effect_dist = torch.cdist(effect_flat, effect_flat, p=2)
    score = normalize_pairwise(action_dist) + normalize_pairwise(effect_dist)
    score.fill_diagonal_(-1.0)
    return score.argmax(dim=1)


def normalize_pairwise(dist: torch.Tensor) -> torch.Tensor:
    return dist / dist.detach().mean().clamp_min(1e-6)


def horizon_mse(pred: torch.Tensor, target: torch.Tensor, *, ctx_len: int, horizon: int) -> float:
    diff = pred[:, ctx_len : ctx_len + horizon].float() - target[:, ctx_len : ctx_len + horizon].float()
    return float(diff.pow(2).mean().detach().cpu().item())


def decide(
    metrics: dict[str, Any],
    *,
    causal_min_ratio: float = 1.02,
    negative_modes: list[str] | None = None,
) -> dict[str, Any]:
    direct = metrics["direct"]
    auto = metrics["autoregressive"]
    modes = negative_modes or ["shuffle", "zero"]
    min_ratio = float(causal_min_ratio)
    direct_mode_pass = {mode: direct[f"{mode}_over_normal"] > min_ratio for mode in modes}
    auto_mode_pass = {mode: auto[f"{mode}_over_normal"] > min_ratio for mode in modes}
    direct_action = all(direct_mode_pass.values())
    auto_action = all(auto_mode_pass.values())
    beats_persistence = auto["normal_over_persistence"] < 1.0
    decision = {
        "direct_action_conditioning_detected": bool(direct_action),
        "autoregressive_action_conditioning_detected": bool(auto_action),
        "autoregressive_beats_persistence": bool(beats_persistence),
        "native_dynamics_ready_for_imagination": bool(auto_action and beats_persistence),
        "strict_gate_passed": bool(direct_action and auto_action and beats_persistence),
        "causal_min_ratio": min_ratio,
        "negative_modes": modes,
    }
    for mode in modes:
        decision[f"direct_{mode}_detected"] = bool(direct_mode_pass[mode])
        decision[f"autoregressive_{mode}_detected"] = bool(auto_mode_pass[mode])
    return decision


def parse_negative_modes(value: str) -> list[str]:
    modes = [item.strip() for item in str(value).replace("+", ",").split(",") if item.strip()]
    modes = modes or ["shuffle", "zero"]
    unknown = sorted(
        mode
        for mode in set(modes)
        if mode not in {"shuffle", "far_shuffle", "action_far_shuffle", "distance_shuffle", "effect_far_shuffle", "action_effect_shuffle", "far_effect_shuffle", "zero", "time_reverse", "reverse", "temporal_reverse", "time_perm", "temporal_perm", "same_traj_shuffle", "window_shuffle"}
        and parse_temporal_shift(mode) is None
    )
    if unknown:
        raise ValueError(f"unknown negative modes: {unknown}")
    return modes


def parse_temporal_shift(mode: str) -> int | None:
    aliases = {"time", "temporal", "time_shift", "timeshift", "shift"}
    if mode in aliases:
        return 1
    for prefix in ("time_shift", "timeshift", "shift", "temporal"):
        if mode.startswith(prefix):
            suffix = mode[len(prefix):].lstrip("_-")
            if suffix.isdigit():
                return max(1, int(suffix))
    return None


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(1e-12, denominator))


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
