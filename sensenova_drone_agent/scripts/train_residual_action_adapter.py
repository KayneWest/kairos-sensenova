#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DREAMER4_ROOT = REPO_ROOT / "dreamer4" / "dreamer4"
if str(DREAMER4_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMER4_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_dreamer4_soar_dynamics import (  # noqa: E402
    DEFAULT_ACTION_DIM,
    build_dynamics,
    evaluate,
    parse_negative_modes,
)
from train_dynamics import (  # noqa: E402
    align_actions_to_frames,
    load_frozen_tokenizer_from_pt_ckpt,
    pack_bottleneck_to_spatial,
    temporal_patchify,
)
from wm_dataset import WMDataset, collate_batch  # noqa: E402


class ResidualActionAdapter(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        d_spatial: int,
        n_spatial: int,
        k_max: int,
        hidden: int = 256,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.d_spatial = int(d_spatial)
        self.n_spatial = int(n_spatial)
        self.k_max = int(k_max)
        self.hidden = int(hidden)

        self.action_proj = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.z_pool_proj = nn.Linear(self.d_spatial, self.hidden)
        self.step_embed = nn.Embedding(int(math.log2(self.k_max)) + 1, self.hidden)
        self.signal_embed = nn.Embedding(self.k_max + 1, self.hidden)
        self.temporal = nn.GRU(self.hidden, self.hidden, batch_first=True)
        self.residual = nn.Sequential(
            nn.LayerNorm(self.d_spatial + self.hidden),
            nn.Linear(self.d_spatial + self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.d_spatial),
        )
        # Start as exactly the frozen base dynamics.
        nn.init.zeros_(self.step_embed.weight)
        nn.init.zeros_(self.signal_embed.weight)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        *,
        z_tilde: torch.Tensor,
        actions: torch.Tensor | None,
        act_mask: torch.Tensor | None,
        step_idxs: torch.Tensor,
        signal_idxs: torch.Tensor,
    ) -> torch.Tensor:
        B, T, S, D = z_tilde.shape
        if actions is None:
            action = torch.zeros((B, T, self.action_dim), device=z_tilde.device, dtype=z_tilde.dtype)
        else:
            action = actions
            if act_mask is not None:
                action = action * act_mask
            action = action.clamp(-1, 1)

        pooled = z_tilde.mean(dim=2)
        h = (
            self.action_proj(action.float())
            + self.z_pool_proj(pooled.float())
            + self.step_embed(step_idxs.long())
            + self.signal_embed(signal_idxs.long())
        )
        h, _ = self.temporal(h)
        h_spatial = h[:, :, None, :].expand(B, T, S, self.hidden)
        return self.residual(torch.cat([z_tilde.float(), h_spatial], dim=-1)).to(dtype=z_tilde.dtype)


class ResidualDynamicsWrapper(nn.Module):
    def __init__(self, base: nn.Module, adapter: ResidualActionAdapter, scale: float = 1.0):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.scale = float(scale)

    def forward(
        self,
        actions: torch.Tensor | None,
        step_idxs: torch.Tensor,
        signal_idxs: torch.Tensor,
        packed_enc_tokens: torch.Tensor,
        *,
        act_mask: torch.Tensor | None = None,
        agent_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        with torch.no_grad():
            base_pred, h = self.base(
                actions,
                step_idxs,
                signal_idxs,
                packed_enc_tokens,
                act_mask=act_mask,
                agent_tokens=agent_tokens,
            )
        residual = self.adapter(
            z_tilde=packed_enc_tokens,
            actions=actions,
            act_mask=act_mask,
            step_idxs=step_idxs,
            signal_idxs=signal_idxs,
        )
        return base_pred + self.scale * residual, h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an explicit residual action-dynamics adapter.")
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--tokenizer-ckpt", required=True)
    parser.add_argument("--dynamics-ckpt", required=True)
    parser.add_argument("--tasks-json", default=None)
    parser.add_argument("--source-names", default="soar_native_v2,hf_robot_droid_lerobot_dreamer4,dreamer4_hf_mixed_large")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=20000)
    parser.add_argument("--eval-batches", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--contrast-margin", type=float, default=0.02)
    parser.add_argument("--contrast-modes", default="shuffle,zero,time_shift,time_shift2,time_shift4,time_shift8,time_perm,time_reverse")
    parser.add_argument("--contrast-action-norm-weight", type=float, default=0.0)
    parser.add_argument("--contrast-latent-delta-weight", type=float, default=0.0)
    parser.add_argument("--contrast-weight-clip", type=float, default=10.0)
    parser.add_argument("--signal-level", type=float, default=0.1)
    parser.add_argument("--random-signal", action="store_true")
    parser.add_argument("--action-frame-offset", type=int, default=-1)
    parser.add_argument("--action-dim", type=int, default=49)
    parser.add_argument("--action-features", default="current,prev,delta,mean4,norm")
    parser.add_argument("--require-non-noop", action="store_true")
    parser.add_argument("--no-op-threshold", type=float, default=0.0)
    parser.add_argument("--min-non-noop-steps", type=int, default=1)
    parser.add_argument("--require-visual-delta", action="store_true")
    parser.add_argument("--visual-delta-threshold", type=float, default=0.0)
    parser.add_argument("--min-visual-delta-steps", type=int, default=1)
    parser.add_argument("--visual-delta-stride", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(resolve_path(args.manifest_json).read_text(encoding="utf-8"))
    tasks_json = resolve_path(args.tasks_json or manifest.get("tasks_json", ""))
    sources = select_sources(manifest, args.source_names)

    dataset = WMDataset(
        data_dir=[str(resolve_path(path)) for source in sources for path in source["raw"]],
        frames_dir=[str(resolve_path(path)) for source in sources for path in source["frames"]],
        seq_len=int(args.seq_len),
        img_size=128,
        action_dim=int(args.action_dim),
        tasks_json=str(tasks_json),
        tasks=None,
        strict_tasks=False,
        action_features=str(args.action_features),
        require_non_noop=bool(args.require_non_noop),
        no_op_threshold=float(args.no_op_threshold),
        min_non_noop_steps=int(args.min_non_noop_steps),
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
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=int(args.num_workers) > 0,
        collate_fn=collate_batch,
    )

    encoder, _decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(resolve_path(args.tokenizer_ckpt)), device=device)
    dyn_ckpt = torch.load(resolve_path(args.dynamics_ckpt), map_location="cpu", weights_only=False)
    dyn_args = dict(dyn_ckpt.get("args", {}))
    dyn_args["action_dim"] = int(args.action_dim)
    dyn_args["action_features"] = str(args.action_features)
    base = build_dynamics(dyn_args, tok_args).to(device)
    base.load_state_dict(dyn_ckpt["dynamics"], strict=True)
    base.eval()
    for param in base.parameters():
        param.requires_grad_(False)

    packing_factor = int(dyn_args.get("packing_factor", 2))
    n_latents = int(tok_args.get("n_latents", 16))
    n_spatial = n_latents // packing_factor
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    d_spatial = d_bottleneck * packing_factor
    k_max = int(dyn_args.get("k_max", 8))
    patch = int(tok_args.get("patch", 8))
    adapter = ResidualActionAdapter(
        action_dim=int(args.action_dim),
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        k_max=k_max,
        hidden=int(args.hidden),
    ).to(device)
    wrapped = ResidualDynamicsWrapper(base=base, adapter=adapter, scale=float(args.residual_scale))
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scaler = GradScaler(device="cuda", enabled=torch.cuda.is_available())
    use_amp = torch.cuda.is_available()

    iterator = iter(loader)
    train_logs = []
    for step in range(int(args.train_steps)):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        frames, actions, act_mask, z1 = prepare_latent_batch(
            batch=batch,
            encoder=encoder,
            patch=patch,
            n_spatial=n_spatial,
            packing_factor=packing_factor,
            action_frame_offset=int(args.action_frame_offset),
            device=device,
        )
        step_idxs, signal_idxs, z_tilde = corrupt_latents(
            z1,
            k_max=k_max,
            signal_level=float(args.signal_level),
            random_signal=bool(args.random_signal),
        )
        modes = parse_negative_modes(str(args.contrast_modes))

        with autocast(device_type="cuda", enabled=use_amp):
            pred, _ = wrapped(actions, step_idxs, signal_idxs, z_tilde, act_mask=act_mask)
            normal_per = per_timestep_mse(pred, z1)
            recon_loss = normal_per.mean()
            contrast_loss, ratios = contrastive_action_loss(
                wrapped=wrapped,
                z1=z1,
                z_tilde=z_tilde,
                actions=actions,
                act_mask=act_mask,
                step_idxs=step_idxs,
                signal_idxs=signal_idxs,
                normal_per=normal_per.detach(),
                modes=modes,
                margin=float(args.contrast_margin),
                action_norm_weight=float(args.contrast_action_norm_weight),
                latent_delta_weight=float(args.contrast_latent_delta_weight),
                weight_clip=float(args.contrast_weight_clip),
            )
            loss = recon_loss + float(args.contrast_weight) * contrast_loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if float(args.grad_clip) > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
        scaler.step(opt)
        scaler.update()

        if step % int(args.log_every) == 0:
            log = {
                "step": int(step),
                "loss": float(loss.detach().cpu().item()),
                "recon": float(recon_loss.detach().cpu().item()),
                "contrast": float(contrast_loss.detach().cpu().item()),
                **{f"ratio_{k}": float(v) for k, v in ratios.items()},
            }
            train_logs.append(log)
            print(json.dumps(log), flush=True)
        if int(args.save_every) > 0 and step > 0 and step % int(args.save_every) == 0:
            save_adapter(out_dir / f"adapter_step_{step:07d}.pt", adapter=adapter, args=args, step=step)

    save_adapter(out_dir / "adapter_latest.pt", adapter=adapter, args=args, step=int(args.train_steps))

    eval_loader = DataLoader(
        dataset,
        batch_size=max(1, min(4, int(args.batch_size))),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_batch,
    )
    metrics = evaluate(
        model=wrapped,
        encoder=encoder,
        loader=eval_loader,
        tok_args=tok_args,
        dyn_args=dyn_args,
        device=device,
        max_batches=int(args.eval_batches),
        rollout_horizon=8,
        ctx_len=8,
        eval_d=0.25,
        action_frame_offset=int(args.action_frame_offset),
        seed=int(args.seed) + 999,
        causal_min_ratio=1.02,
        negative_modes=parse_negative_modes(str(args.contrast_modes)),
    )
    payload = {
        "phase": "residual_action_adapter",
        "sources": [source["name"] for source in sources],
        "config": vars(args),
        "metrics": metrics,
        "train_logs_tail": train_logs[-20:],
        "elapsed_s": time.time() - started,
    }
    (out_dir / "residual_adapter_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "metrics": metrics}, indent=2), flush=True)
    return 0


def prepare_latent_batch(
    *,
    batch: dict[str, torch.Tensor],
    encoder: nn.Module,
    patch: int,
    n_spatial: int,
    packing_factor: int,
    action_frame_offset: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = batch["obs"].to(device, non_blocking=True)
    act = batch["act"].to(device, non_blocking=True)
    mask = batch["act_mask"].to(device, non_blocking=True)
    act = act.clamp(-1, 1) * mask
    frames = obs[:, :-1].float() / 255.0
    actions, act_mask = align_actions_to_frames(
        act,
        mask,
        frame_count=frames.shape[1],
        action_frame_offset=action_frame_offset,
    )
    with torch.no_grad():
        patches = temporal_patchify(frames, patch)
        z_btld, _ = encoder(patches)
        z1 = pack_bottleneck_to_spatial(z_btld, n_spatial=n_spatial, k=packing_factor)
    return frames, actions, act_mask, z1


def corrupt_latents(
    z1: torch.Tensor,
    *,
    k_max: int,
    signal_level: float,
    random_signal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T = z1.shape[:2]
    device = z1.device
    emax = int(round(math.log2(k_max)))
    step_idxs = torch.full((B, T), emax, device=device, dtype=torch.long)
    if random_signal:
        signal_idxs = torch.randint(0, k_max + 1, (B, T), device=device, dtype=torch.long)
        sigma = signal_idxs.float() / float(k_max)
    else:
        idx = max(0, min(k_max, int(round(float(signal_level) * k_max))))
        signal_idxs = torch.full((B, T), idx, device=device, dtype=torch.long)
        sigma = torch.full((B, T), idx / float(k_max), device=device, dtype=torch.float32)
    z0 = torch.randn_like(z1)
    z_tilde = (1.0 - sigma)[..., None, None] * z0 + sigma[..., None, None] * z1
    return step_idxs, signal_idxs, z_tilde


def per_timestep_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred.float() - target.float()).pow(2).mean(dim=(2, 3))


def contrastive_action_loss(
    *,
    wrapped: ResidualDynamicsWrapper,
    z1: torch.Tensor,
    z_tilde: torch.Tensor,
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    step_idxs: torch.Tensor,
    signal_idxs: torch.Tensor,
    normal_per: torch.Tensor,
    modes: list[str],
    margin: float,
    action_norm_weight: float,
    latent_delta_weight: float,
    weight_clip: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = build_contrast_weights(
        z1=z1,
        actions=actions,
        act_mask=act_mask,
        action_norm_weight=float(action_norm_weight),
        latent_delta_weight=float(latent_delta_weight),
        weight_clip=float(weight_clip),
    )

    losses = []
    ratios = {}
    for mode in modes:
        if mode == "normal":
            continue
        neg_actions, neg_mask = make_variant(actions, act_mask, mode, z1=z1)
        pred, _ = wrapped(neg_actions, step_idxs, signal_idxs, z_tilde, act_mask=neg_mask)
        neg_per = per_timestep_mse(pred, z1)
        gap = torch.relu(float(margin) + normal_per - neg_per)
        losses.append(masked_mean(gap, weights))
        ratios[mode] = float((masked_mean(neg_per, weights) / masked_mean(normal_per, weights).clamp_min(1e-12)).detach().cpu().item())
    if not losses:
        return normal_per.mean() * 0.0, ratios
    return torch.stack(losses).mean(), ratios


def build_contrast_weights(
    *,
    z1: torch.Tensor,
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    action_norm_weight: float,
    latent_delta_weight: float,
    weight_clip: float,
) -> torch.Tensor:
    active = (act_mask.float().sum(dim=-1) > 0).float()
    weights = active.clone()

    if float(action_norm_weight) > 0.0:
        denom = act_mask.float().sum(dim=-1).clamp_min(1.0)
        action_norm = ((actions.float() * act_mask.float()).pow(2).sum(dim=-1) / denom).sqrt()
        scale = (action_norm * active).sum() / active.sum().clamp_min(1.0)
        weights = weights * (1.0 + float(action_norm_weight) * action_norm / scale.detach().clamp_min(1e-6))

    if float(latent_delta_weight) > 0.0:
        latent_delta = torch.zeros(z1.shape[:2], device=z1.device, dtype=torch.float32)
        if z1.shape[1] > 1:
            latent_delta[:, 1:] = (z1[:, 1:].float() - z1[:, :-1].float()).pow(2).mean(dim=(2, 3)).sqrt()
        scale = (latent_delta * active).sum() / active.sum().clamp_min(1.0)
        weights = weights * (1.0 + float(latent_delta_weight) * latent_delta / scale.detach().clamp_min(1e-6))

    if float(weight_clip) > 0.0:
        weights = weights.clamp(max=float(weight_clip))
    return weights * active


def masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def make_variant(
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    mode: str,
    *,
    z1: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(mode)
    if mode == "shuffle":
        perm = torch.randperm(actions.shape[0], device=actions.device)
        return actions[perm], act_mask[perm]
    if mode in {"far_shuffle", "action_far_shuffle", "distance_shuffle"}:
        perm = farthest_action_permutation(actions, act_mask)
        return actions[perm], act_mask[perm]
    if mode in {"effect_far_shuffle", "action_effect_shuffle", "far_effect_shuffle"}:
        if z1 is None:
            perm = farthest_action_permutation(actions, act_mask)
        else:
            perm = farthest_action_effect_permutation(actions, act_mask, z1)
        return actions[perm], act_mask[perm]
    if mode == "zero":
        return torch.zeros_like(actions), torch.zeros_like(act_mask)
    if mode in {"time_reverse", "reverse"}:
        return actions.flip(dims=[1]), act_mask.flip(dims=[1])
    if mode in {"time_perm", "same_traj_shuffle", "window_shuffle"}:
        idx = torch.stack([torch.randperm(actions.shape[1], device=actions.device) for _ in range(actions.shape[0])])
        gather_idx = idx[:, :, None].expand(-1, -1, actions.shape[-1])
        return actions.gather(1, gather_idx), act_mask.gather(1, gather_idx)
    shift = parse_temporal_shift(mode)
    if shift is not None:
        return torch.roll(actions, shifts=shift, dims=1), torch.roll(act_mask, shifts=shift, dims=1)
    raise ValueError(f"unknown contrast mode: {mode}")


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
    z1: torch.Tensor,
) -> torch.Tensor:
    if actions.shape[0] <= 1:
        return torch.arange(actions.shape[0], device=actions.device)
    action_flat = (actions.float() * act_mask.float()).flatten(1)
    action_dist = torch.cdist(action_flat, action_flat, p=2)
    if z1.shape[1] > 1:
        effect_flat = (z1[:, 1:].float() - z1[:, :-1].float()).flatten(1)
    else:
        effect_flat = z1.float().flatten(1)
    effect_dist = torch.cdist(effect_flat, effect_flat, p=2)
    score = normalize_pairwise(action_dist) + normalize_pairwise(effect_dist)
    score.fill_diagonal_(-1.0)
    return score.argmax(dim=1)


def normalize_pairwise(dist: torch.Tensor) -> torch.Tensor:
    return dist / dist.detach().mean().clamp_min(1e-6)


def parse_temporal_shift(mode: str) -> int | None:
    if mode in {"time_shift", "timeshift", "shift", "time"}:
        return 1
    for prefix in ("time_shift", "timeshift", "shift"):
        if mode.startswith(prefix):
            suffix = mode[len(prefix):].lstrip("_-")
            if suffix.isdigit():
                return max(1, int(suffix))
    return None


def select_sources(manifest: dict[str, Any], source_names: str) -> list[dict[str, Any]]:
    raw_sources = manifest.get("sources", [])
    requested = [name.strip() for name in str(source_names).split(",") if name.strip()]
    specs = []
    for name in requested:
        matched = [source for source in raw_sources if source.get("name") == name]
        if not matched:
            raise ValueError(f"source not found in manifest: {name}")
        source = matched[0]
        specs.append({"name": source["name"], "raw": [source["raw"]], "frames": [source["frames"]]})
    return specs


def save_adapter(path: Path, *, adapter: ResidualActionAdapter, args: argparse.Namespace, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter.state_dict(), "args": vars(args), "step": int(step)}, path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
