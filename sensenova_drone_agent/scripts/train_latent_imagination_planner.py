#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DREAMER4_ROOT = REPO_ROOT / "dreamer4" / "dreamer4"
for item in (str(DREAMER4_ROOT), str(PROJECT_ROOT / "scripts")):
    if item not in sys.path:
        sys.path.insert(0, item)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("torch is required for latent imagination planner training.") from exc

from train_dynamics import (  # noqa: E402
    DEFAULT_ACTION_DIM,
    align_actions_to_frames,
    load_frozen_tokenizer_from_pt_ckpt,
    pack_bottleneck_to_spatial,
    temporal_patchify,
)
from wm_dataset import WMDataset, collate_batch  # noqa: E402


@dataclass
class PlannerConfig:
    data_dirs: list[str]
    frame_dirs: list[str]
    tasks_json: str
    tokenizer_ckpt: str
    out_dir: str
    resume_ckpt: str = ""
    manifest_json: str = ""
    source_names: str = ""
    use_manifest_weights: bool = True
    seq_len: int = 24
    ctx_len: int = 8
    horizon: int = 8
    img_size: int = 128
    batch_size: int = 8
    num_workers: int = 2
    max_steps: int = 500000
    eval_every: int = 1000
    eval_batches: int = 64
    save_every: int = 10000
    trace_every: int = 5000
    action_dim: int = DEFAULT_ACTION_DIM
    raw_action_dim: int = DEFAULT_ACTION_DIM
    action_features: str = "current"
    action_frame_offset: int = -1
    hidden_dim: int = 1024
    plan_dim: int = 128
    num_candidates: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    gamma: float = 0.997
    future_loss_weight: float = 1.0
    reward_loss_weight: float = 0.25
    inverse_loss_weight: float = 0.25
    contrast_weight: float = 1.0
    contrast_margin: float = 0.02
    contrast_relative_margin: float = 0.0
    plan_l2_weight: float = 1e-4
    effect_loss_weight: float = 0.10
    plan_unit_norm: bool = False
    plan_step_conditioning: bool = False
    rank_loss_weight: float = 0.0
    rank_num_bank: int = 4
    rank_num_matched: int = 4
    rank_margin: float = 0.05
    rank_mse_gap: float = 1.1
    inverse_plan_dropout: float = 0.0
    inverse_imagined_weight: float = 0.0
    inverse_cross_weight: float = 0.0
    score_plan_dropout: float = 0.0
    bc_head_weight: float = 0.0
    bc_encoder_grad: bool = False
    horizon_curriculum_max: int = 0
    horizon_curriculum_weight: float = 0.5
    contrast_modes: str = "shuffle,zero,time_shift,time_shift2,time_perm,time_reverse"
    require_non_noop: bool = False
    no_op_threshold: float = 0.0
    min_non_noop_steps: int = 1
    reward_filter_mode: str = "none"
    reward_signal_threshold: float = 0.0
    min_reward_signal_steps: int = 1
    require_visual_delta: bool = False
    visual_delta_threshold: float = 0.0
    min_visual_delta_steps: int = 1
    visual_delta_stride: int = 4
    device: str = "cuda"
    seed: int = 20260607
    amp: bool = True


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        cur = in_dim
        for _ in range(max(1, depth)):
            layers += [nn.Linear(cur, hidden_dim), nn.SiLU()]
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentImaginationPlanner(nn.Module):
    def __init__(
        self,
        *,
        z_dim: int,
        action_dim: int,
        hidden_dim: int,
        plan_dim: int,
        horizon: int,
        plan_unit_norm: bool = False,
        plan_step_conditioning: bool = False,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.plan_dim = int(plan_dim)
        self.horizon = int(horizon)
        self.plan_unit_norm = bool(plan_unit_norm)
        self.plan_step_conditioning = bool(plan_step_conditioning)

        self.ctx_rnn = nn.GRU(self.z_dim + self.action_dim, self.hidden_dim, batch_first=True)
        self.action_rnn = nn.GRU(self.action_dim, self.hidden_dim, batch_first=True)
        self.plan_head = MLP(self.hidden_dim * 2, self.plan_dim, self.hidden_dim, depth=2)
        self.z_in = nn.Linear(self.z_dim, self.hidden_dim)
        self.plan_in = nn.Linear(self.plan_dim, self.hidden_dim)
        self.time_embed = nn.Embedding(self.horizon, self.hidden_dim)
        self.future_cell = nn.GRUCell(self.hidden_dim * 3, self.hidden_dim)
        self.future_delta = nn.Linear(self.hidden_dim, self.z_dim)
        self.scorer = MLP(self.hidden_dim + self.z_dim + self.plan_dim, 1, self.hidden_dim, depth=2)
        self.inverse = MLP(self.hidden_dim + self.z_dim + self.plan_dim, self.horizon * self.action_dim, self.hidden_dim, depth=2)
        self.effect = MLP(self.hidden_dim, self.z_dim, self.hidden_dim, depth=2)
        if self.plan_step_conditioning:
            # per-step readout of the plan token: gives rollout step t a
            # pathway to "the action at step t" so exact timing is learnable
            # while the plan stays a single samplable token
            self.plan_step_head = nn.Linear(self.plan_dim, self.horizon * self.hidden_dim)
        # optional BC chunk head (control-purposed encoder training + act-time
        # behavior prior); created lazily via enable_bc_head
        self.bc_head = None

    def enable_bc_head(self) -> None:
        if self.bc_head is None:
            self.bc_head = MLP(self.hidden_dim, self.horizon * self.action_dim, self.hidden_dim, depth=2)

    def bc_logits(self, ctx_h: torch.Tensor) -> torch.Tensor:
        return self.bc_head(ctx_h).view(ctx_h.shape[0], self.horizon, self.action_dim)

    def encode_context(self, ctx_z: torch.Tensor, ctx_actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ctx_z, ctx_actions], dim=-1)
        _, h = self.ctx_rnn(x)
        return h[-1]

    def encode_plan(self, ctx_h: torch.Tensor, future_actions: torch.Tensor) -> torch.Tensor:
        _, h = self.action_rnn(future_actions)
        plan = self.plan_head(torch.cat([ctx_h, h[-1]], dim=-1))
        return self.normalize_plan(plan)

    def normalize_plan(self, plan: torch.Tensor) -> torch.Tensor:
        if not self.plan_unit_norm:
            return plan
        return F.normalize(plan, dim=-1) * math.sqrt(self.plan_dim)

    def propose_future(self, ctx_z: torch.Tensor, ctx_h: torch.Tensor, plan: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
        horizon = self.horizon if horizon is None else int(horizon)
        prev_z = ctx_z[:, -1]
        h = ctx_h
        outs = []
        plan_h = self.plan_in(plan)
        step_plan = None
        if self.plan_step_conditioning:
            step_plan = self.plan_step_head(plan).view(plan.shape[0], self.horizon, self.hidden_dim)
        for t in range(horizon):
            step = self.time_embed.weight[min(t, self.horizon - 1)][None, :].expand(ctx_z.shape[0], -1)
            plan_t = plan_h if step_plan is None else plan_h + step_plan[:, min(t, self.horizon - 1)]
            x = torch.cat([self.z_in(prev_z), plan_t, step], dim=-1)
            h = self.future_cell(x, h)
            delta = self.future_delta(h)
            prev_z = prev_z + delta
            outs.append(prev_z)
        return torch.stack(outs, dim=1)

    def score_future(self, ctx_h: torch.Tensor, future_z: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        pooled = future_z.mean(dim=1)
        return self.scorer(torch.cat([ctx_h, pooled, plan], dim=-1)).squeeze(-1)

    def inverse_actions(self, ctx_h: torch.Tensor, future_z: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        pooled = future_z.mean(dim=1)
        pred = self.inverse(torch.cat([ctx_h, pooled, plan], dim=-1))
        return pred.view(future_z.shape[0], self.horizon, self.action_dim)

    def action_effect(self, future_actions: torch.Tensor) -> torch.Tensor:
        _, h = self.action_rnn(future_actions)
        return self.effect(h[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a latent imagination planner over frozen Kairos/Dreamer4 tokenizer latents.")
    parser.add_argument("--manifest-json", default="")
    parser.add_argument("--source-names", default="")
    parser.add_argument("--no-manifest-weights", action="store_true")
    parser.add_argument("--data-dir", action="append", default=[])
    parser.add_argument("--frames-dir", action="append", default=[])
    parser.add_argument("--tasks-json", default="")
    parser.add_argument("--tokenizer-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume-ckpt", default="")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--trace-every", type=int, default=5000)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--raw-action-dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--action-features", default="current")
    parser.add_argument("--action-frame-offset", type=int, default=-1)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--plan-dim", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--future-loss-weight", type=float, default=1.0)
    parser.add_argument("--reward-loss-weight", type=float, default=0.25)
    parser.add_argument("--inverse-loss-weight", type=float, default=0.25)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--contrast-margin", type=float, default=0.02)
    parser.add_argument("--contrast-relative-margin", type=float, default=0.0)
    parser.add_argument("--plan-l2-weight", type=float, default=1e-4)
    parser.add_argument("--effect-loss-weight", type=float, default=0.10)
    parser.add_argument("--plan-unit-norm", action="store_true")
    parser.add_argument("--plan-step-conditioning", action="store_true")
    parser.add_argument("--rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--rank-num-bank", type=int, default=4)
    parser.add_argument("--rank-num-matched", type=int, default=4)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank-mse-gap", type=float, default=1.1)
    parser.add_argument("--inverse-plan-dropout", type=float, default=0.0)
    parser.add_argument("--inverse-imagined-weight", type=float, default=0.0)
    parser.add_argument("--inverse-cross-weight", type=float, default=0.0)
    parser.add_argument("--score-plan-dropout", type=float, default=0.0)
    parser.add_argument("--bc-head-weight", type=float, default=0.0)
    parser.add_argument("--bc-encoder-grad", action="store_true")
    parser.add_argument("--horizon-curriculum-max", type=int, default=0)
    parser.add_argument("--horizon-curriculum-weight", type=float, default=0.5)
    parser.add_argument("--contrast-modes", default="shuffle,zero,time_shift,time_shift2,time_perm,time_reverse")
    parser.add_argument("--require-non-noop", action="store_true")
    parser.add_argument("--no-op-threshold", type=float, default=0.0)
    parser.add_argument("--min-non-noop-steps", type=int, default=1)
    parser.add_argument("--reward-filter-mode", default="none")
    parser.add_argument("--reward-signal-threshold", type=float, default=0.0)
    parser.add_argument("--min-reward-signal-steps", type=int, default=1)
    parser.add_argument("--require-visual-delta", action="store_true")
    parser.add_argument("--visual-delta-threshold", type=float, default=0.0)
    parser.add_argument("--min-visual-delta-steps", type=int, default=1)
    parser.add_argument("--visual-delta-stride", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(args)
    seed_everything(config.seed)
    device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = resolve_path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "planner_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "config.json", asdict(config))
    write_json(out_dir / "data_sources.json", {"data_dirs": config.data_dirs, "frame_dirs": config.frame_dirs})

    dataset = WMDataset(
        data_dir=[str(resolve_path(p)) for p in config.data_dirs],
        frames_dir=[str(resolve_path(p)) for p in config.frame_dirs],
        seq_len=config.seq_len,
        img_size=config.img_size,
        action_dim=config.action_dim,
        raw_action_dim=config.raw_action_dim,
        tasks_json=str(resolve_path(config.tasks_json)) if config.tasks_json else "",
        tasks=None,
        strict_tasks=False,
        action_features=config.action_features,
        require_non_noop=config.require_non_noop,
        no_op_threshold=config.no_op_threshold,
        min_non_noop_steps=config.min_non_noop_steps,
        reward_filter_mode=config.reward_filter_mode,
        reward_signal_threshold=config.reward_signal_threshold,
        min_reward_signal_steps=config.min_reward_signal_steps,
        require_visual_delta=config.require_visual_delta,
        visual_delta_threshold=config.visual_delta_threshold,
        min_visual_delta_steps=config.min_visual_delta_steps,
        visual_delta_stride=config.visual_delta_stride,
        verbose=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=True,
        collate_fn=collate_batch,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=max(1, min(config.batch_size, 4)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_batch,
    )

    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(resolve_path(config.tokenizer_ckpt)), device=device)
    patch = int(tok_args.get("patch", 8))
    packing_factor = int(tok_args.get("packing_factor", 2))
    n_latents = int(tok_args.get("n_latents", 16))
    n_spatial = n_latents // packing_factor
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    z_dim = n_spatial * d_bottleneck * packing_factor

    model = LatentImaginationPlanner(
        z_dim=z_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        plan_dim=config.plan_dim,
        horizon=config.horizon,
        plan_unit_norm=config.plan_unit_norm,
        plan_step_conditioning=config.plan_step_conditioning,
    )
    if config.bc_head_weight > 0:
        model.enable_bc_head()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler(device="cuda", enabled=(device.type == "cuda" and config.amp))
    use_amp = device.type == "cuda" and config.amp
    metrics_path = out_dir / "metrics.jsonl"
    start_step = 0
    if config.resume_ckpt:
        start_step = load_checkpoint(
            resolve_path(config.resume_ckpt),
            model=model,
            opt=opt,
            scaler=scaler,
            device=device,
        )
        print(json.dumps({"phase": "resume", "resume_ckpt": config.resume_ckpt, "start_step": start_step}), flush=True)
    started = time.time()

    print(
        json.dumps(
            {
                "phase": "latent_imagination_planner",
                "dataset_sequences": len(dataset),
                "tasks": dataset.num_tasks,
                "z_dim": z_dim,
                "device": str(device),
                "start_step": start_step,
                "max_steps": config.max_steps,
            },
            indent=2,
        ),
        flush=True,
    )

    iterator = iter(loader)
    last_eval: dict[str, float] = {}
    for step in range(start_step + 1, config.max_steps + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_batch = next(iterator)

        batch = encode_batch(
            raw_batch=raw_batch,
            encoder=encoder,
            patch=patch,
            n_spatial=n_spatial,
            packing_factor=packing_factor,
            action_frame_offset=config.action_frame_offset,
            device=device,
        )

        with autocast(device_type="cuda", enabled=use_amp):
            loss, train_metrics = compute_losses(model, batch, config)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if config.grad_clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % 100 == 0:
            row = {
                "step": step,
                "phase": "train",
                "elapsed_s": time.time() - started,
                **train_metrics,
            }
            append_jsonl(metrics_path, row)
            print(json.dumps(row), flush=True)

        if config.eval_every > 0 and step % config.eval_every == 0:
            last_eval = evaluate(
                model=model,
                loader=eval_loader,
                encoder=encoder,
                patch=patch,
                n_spatial=n_spatial,
                packing_factor=packing_factor,
                config=config,
                device=device,
                max_batches=config.eval_batches,
            )
            row = {"step": step, "phase": "eval", "elapsed_s": time.time() - started, **last_eval}
            append_jsonl(metrics_path, row)
            print(json.dumps(row), flush=True)
            write_json(out_dir / "latest_eval.json", row)
            write_report(out_dir / "report.md", config=config, latest=row)

        if config.trace_every > 0 and step % config.trace_every == 0:
            write_trace(
                model=model,
                loader=eval_loader,
                encoder=encoder,
                patch=patch,
                n_spatial=n_spatial,
                packing_factor=packing_factor,
                config=config,
                device=device,
                path=trace_dir / f"trace_step_{step:07d}.json",
            )

        if config.save_every > 0 and step % config.save_every == 0:
            save_checkpoint(ckpt_dir / f"step_{step:07d}.pt", model=model, opt=opt, scaler=scaler, config=config, step=step)
            save_checkpoint(ckpt_dir / "latest.pt", model=model, opt=opt, scaler=scaler, config=config, step=step)

    save_checkpoint(ckpt_dir / "final.pt", model=model, opt=opt, scaler=scaler, config=config, step=config.max_steps)
    final = {
        "step": config.max_steps,
        "phase": "complete",
        "elapsed_s": time.time() - started,
        **last_eval,
    }
    write_json(out_dir / "summary.json", final)
    write_report(out_dir / "report.md", config=config, latest=final)
    print(json.dumps({"out_dir": str(out_dir), "summary": final}, indent=2), flush=True)
    return 0


def compute_losses(
    model: LatentImaginationPlanner,
    batch: dict[str, torch.Tensor],
    config: PlannerConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    ctx_z, ctx_actions, future_z, future_actions, future_mask, rewards = split_batch(batch, config)
    ctx_h = model.encode_context(ctx_z, ctx_actions)
    plan = model.encode_plan(ctx_h, future_actions)
    pred_future = model.propose_future(ctx_z, ctx_h, plan, horizon=config.horizon)
    # return regression must not shape the imagined future: once the scorer is
    # fidelity-sensitive (rank loss), an undetached path lets reward gradients
    # drag pred_future away from the real future (observed 5-10x future_loss
    # regression in the v2 rankfix arms A/B)
    # score plan-dropout: like the inverse head, a scorer given the plan
    # token shortcuts through it (Q(ctx, plan)) and ignores the imagined
    # future - which makes candidate selection blind to what imagination
    # already predicts (e.g. a crash rendered as frozen frames). Dropout
    # forces value to be read from the future content.
    score_plan = plan.detach()
    if config.score_plan_dropout > 0:
        keep = (torch.rand(plan.shape[0], 1, device=plan.device) >= config.score_plan_dropout).float()
        score_plan = score_plan * keep
    pred_score = model.score_future(ctx_h.detach(), pred_future.detach(), score_plan)
    # plan dropout forces the inverse head to decode actions from the future
    # latents instead of reading them back out of the plan token (the act-time
    # path feeds candidate plans whose action content is wrong for this
    # context, so a plan-decoding inverse head emits the wrong actions)
    inv_plan = plan
    if config.inverse_plan_dropout > 0:
        keep = (torch.rand(plan.shape[0], 1, device=plan.device) >= config.inverse_plan_dropout).float()
        inv_plan = plan * keep
    inv_pred = model.inverse_actions(ctx_h, future_z, inv_plan)
    effect_pred = model.action_effect(future_actions)

    future_loss = F.mse_loss(pred_future.float(), future_z.float())
    # long-horizon curriculum: extend the rollout beyond the trained horizon
    # (time embeddings clamp to the last step) and supervise the tail so
    # h16 open-loop imagination stops diverging
    long_h = int(config.horizon_curriculum_max)
    if long_h > config.horizon and config.horizon_curriculum_weight > 0:
        avail = int(config.seq_len - config.ctx_len)
        long_h = min(long_h, avail)
        if long_h > config.horizon:
            z_flat_all = batch["z"].flatten(2)
            long_target = z_flat_all[:, config.ctx_len : config.ctx_len + long_h]
            long_pred = model.propose_future(ctx_z, ctx_h, plan, horizon=long_h)
            tail_loss = F.mse_loss(long_pred[:, config.horizon :].float(), long_target[:, config.horizon :].float())
            future_loss = future_loss + float(config.horizon_curriculum_weight) * tail_loss
    returns = discounted_returns(rewards, gamma=config.gamma)
    reward_loss = F.mse_loss(pred_score.float(), returns.float())
    inverse_loss = masked_mse(inv_pred.float(), future_actions.float(), future_mask.float())
    # act-time path supervision: decode actions from the IMAGINED future with
    # no plan input; target is the true actions the plan was encoded from
    inverse_imagined_loss = ctx_z.new_zeros(())
    if config.inverse_imagined_weight > 0:
        inv_img = model.inverse_actions(ctx_h.detach(), pred_future.detach(), torch.zeros_like(plan))
        inverse_imagined_loss = masked_mse(inv_img.float(), future_actions.float(), future_mask.float())
    # cross-context imagined-inverse: hold the context fixed, imagine the
    # future implied by ANOTHER context's action chunk, and require the
    # decoder to recover those foreign actions from the future alone. The
    # context cannot explain the target, so the head must read the future -
    # this is the pathway that lets candidate selection change emitted actions.
    inverse_cross_loss = ctx_z.new_zeros(())
    if config.inverse_cross_weight > 0:
        cross_actions = future_actions.roll(1, dims=0)
        cross_mask = future_mask.roll(1, dims=0)
        with torch.no_grad():
            cross_plan = model.encode_plan(ctx_h, cross_actions * cross_mask)
            cross_future = model.propose_future(ctx_z, ctx_h, cross_plan, horizon=config.horizon)
        inv_cross = model.inverse_actions(ctx_h.detach(), cross_future, torch.zeros_like(plan))
        inverse_cross_loss = masked_mse(inv_cross.float(), cross_actions.float(), cross_mask.float())
    target_effect = future_z[:, -1].float() - ctx_z[:, -1].float()
    effect_loss = F.mse_loss(effect_pred.float(), target_effect.float())
    plan_l2 = plan.float().pow(2).mean()
    contrast_loss, contrast_metrics = plan_contrast_loss(
        model=model,
        ctx_z=ctx_z,
        ctx_h=ctx_h,
        future_z=future_z,
        future_actions=future_actions,
        future_mask=future_mask,
        normal_pred=pred_future,
        config=config,
    )
    # BC head: cross-entropy over one-hot action targets; gradient into the
    # context encoder when bc_encoder_grad (control-purposed representation)
    bc_loss = ctx_z.new_zeros(())
    if config.bc_head_weight > 0:
        bc_in = ctx_h if config.bc_encoder_grad else ctx_h.detach()
        logits = model.bc_logits(bc_in)
        target = future_actions.argmax(-1)
        bc_loss = F.cross_entropy(logits.reshape(-1, config.action_dim), target.reshape(-1))
    rank_loss = ctx_z.new_zeros(())
    rank_metrics: dict[str, float] = {}
    if config.rank_loss_weight > 0:
        rank_loss, rank_metrics = candidate_rank_loss(
            model=model,
            ctx_z=ctx_z,
            ctx_h=ctx_h,
            future_z=future_z,
            future_actions=future_actions,
            future_mask=future_mask,
            true_plan=plan,
            config=config,
        )
    loss = (
        config.future_loss_weight * future_loss
        + config.reward_loss_weight * reward_loss
        + config.inverse_loss_weight * inverse_loss
        + config.contrast_weight * contrast_loss
        + config.effect_loss_weight * effect_loss
        + config.plan_l2_weight * plan_l2
        + config.rank_loss_weight * rank_loss
        + config.inverse_imagined_weight * inverse_imagined_loss
        + config.inverse_cross_weight * inverse_cross_loss
        + config.bc_head_weight * bc_loss
    )
    metrics = {
        "loss": item(loss),
        "future_loss": item(future_loss),
        "reward_loss": item(reward_loss),
        "inverse_loss": item(inverse_loss),
        "inverse_imagined_loss": item(inverse_imagined_loss),
        "inverse_cross_loss": item(inverse_cross_loss),
        "bc_loss": item(bc_loss),
        "contrast_loss": item(contrast_loss),
        "effect_loss": item(effect_loss),
        "plan_l2": item(plan_l2),
        "return_mean": item(returns.mean()),
        "score_mean": item(pred_score.mean()),
        **contrast_metrics,
        **rank_metrics,
    }
    return loss, metrics


def candidate_rank_loss(
    *,
    model: LatentImaginationPlanner,
    ctx_z: torch.Tensor,
    ctx_h: torch.Tensor,
    future_z: torch.Tensor,
    future_actions: torch.Tensor,
    future_mask: torch.Tensor,
    true_plan: torch.Tensor,
    config: PlannerConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train the scorer to rank candidate plans by imagined-future fidelity.

    The pool per item mixes the true plan, wrong-action control plans, bank
    plans from other batch items, and Gaussian samples matched to the batch
    plan statistics. Rollouts and fidelity targets are computed without grad;
    the pairwise hinge only updates the scorer head, so this cannot fight the
    future/contrast losses through the proposer. Works without reward labels.
    """
    B = ctx_z.shape[0]
    with torch.no_grad():
        pool = [true_plan]
        for mode in ("zero", "time_shift"):
            neg_actions, neg_mask = make_action_variant(future_actions, future_mask, mode)
            pool.append(model.encode_plan(ctx_h, neg_actions * neg_mask))
        pool.append(model.encode_plan(ctx_h, future_actions.roll(1, dims=0) * future_mask.roll(1, dims=0)))
        for k in range(1, max(0, int(config.rank_num_bank)) + 1):
            pool.append(true_plan.roll(k, dims=0))
        if config.rank_num_matched > 0:
            mu = true_plan.mean(dim=0, keepdim=True)
            sigma = true_plan.std(dim=0, keepdim=True).clamp_min(1e-6)
            for _ in range(int(config.rank_num_matched)):
                pool.append(model.normalize_plan(mu + sigma * torch.randn_like(true_plan)))
        plans = torch.stack(pool, dim=1)  # (B, P, plan_dim)
        P = plans.shape[1]
        ctx_z_rep = ctx_z[:, None].expand(B, P, *ctx_z.shape[1:]).reshape(B * P, *ctx_z.shape[1:])
        ctx_h_rep = ctx_h[:, None].expand(B, P, ctx_h.shape[-1]).reshape(B * P, -1)
        plans_flat = plans.reshape(B * P, -1)
        futures = model.propose_future(ctx_z_rep, ctx_h_rep, plans_flat, horizon=config.horizon)
        fid = (futures.view(B, P, *futures.shape[1:]).float() - future_z[:, None].float()).pow(2).mean(dim=(2, 3))
    rank_plans = plans_flat
    if config.score_plan_dropout > 0:
        keep = (torch.rand(plans_flat.shape[0], 1, device=plans_flat.device) >= config.score_plan_dropout).float()
        rank_plans = plans_flat * keep
    scores = model.score_future(ctx_h_rep.detach(), futures, rank_plans).view(B, P)
    # pair (i, j) where candidate i is clearly more faithful than j
    better = fid[:, None, :] > float(config.rank_mse_gap) * fid[:, :, None]  # (B, i, j)
    hinge = torch.relu(float(config.rank_margin) + scores[:, None, :] - scores[:, :, None])
    valid = better.float()
    loss = (hinge * valid).sum() / valid.sum().clamp_min(1.0)
    s = scores.detach().float()
    f = -fid.detach().float()
    s = s - s.mean(dim=1, keepdim=True)
    f = f - f.mean(dim=1, keepdim=True)
    denom = (s.norm(dim=1) * f.norm(dim=1)).clamp_min(1e-12)
    fid_corr = ((s * f).sum(dim=1) / denom).mean()
    top1 = (scores.detach().argmax(dim=1) == fid.detach().argmin(dim=1)).float().mean()
    metrics = {
        "rank_loss": item(loss),
        "rank_fid_corr": item(fid_corr),
        "rank_top1_acc": item(top1),
        "rank_pool_size": float(P),
    }
    return loss, metrics


def plan_contrast_loss(
    *,
    model: LatentImaginationPlanner,
    ctx_z: torch.Tensor,
    ctx_h: torch.Tensor,
    future_z: torch.Tensor,
    future_actions: torch.Tensor,
    future_mask: torch.Tensor,
    normal_pred: torch.Tensor,
    config: PlannerConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    normal_per = (normal_pred.float() - future_z.float()).pow(2).mean(dim=(1, 2))
    modes = [m.strip() for m in config.contrast_modes.replace("+", ",").split(",") if m.strip()]
    losses = []
    metrics: dict[str, float] = {"normal_future_mse": item(normal_per.mean())}
    for mode in modes:
        neg_actions, neg_mask = make_action_variant(future_actions, future_mask, mode)
        neg_plan = model.encode_plan(ctx_h, neg_actions * neg_mask)
        neg_pred = model.propose_future(ctx_z, ctx_h, neg_plan, horizon=config.horizon)
        neg_per = (neg_pred.float() - future_z.float()).pow(2).mean(dim=(1, 2))
        if config.contrast_relative_margin > 0:
            # ratio margin: keep gradient on hard (timing) negatives after the
            # easy zero/shuffle negatives clear the absolute margin
            losses.append(torch.relu(float(config.contrast_relative_margin) * normal_per - neg_per).mean())
        else:
            losses.append(torch.relu(float(config.contrast_margin) + normal_per - neg_per).mean())
        metrics[f"{metric_key(mode)}_future_mse"] = item(neg_per.mean())
        metrics[f"{metric_key(mode)}_over_normal"] = item(neg_per.mean() / normal_per.mean().clamp_min(1e-12))
    if not losses:
        return normal_per.mean() * 0.0, metrics
    return torch.stack(losses).mean(), metrics


@torch.no_grad()
def evaluate(
    *,
    model: LatentImaginationPlanner,
    loader: DataLoader,
    encoder: nn.Module,
    patch: int,
    n_spatial: int,
    packing_factor: int,
    config: PlannerConfig,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    iterator = iter(loader)
    for _ in range(max_batches):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            break
        batch = encode_batch(
            raw_batch=raw_batch,
            encoder=encoder,
            patch=patch,
            n_spatial=n_spatial,
            packing_factor=packing_factor,
            action_frame_offset=config.action_frame_offset,
            device=device,
        )
        ctx_z, ctx_actions, future_z, future_actions, future_mask, rewards = split_batch(batch, config)
        ctx_h = model.encode_context(ctx_z, ctx_actions)
        plan = model.encode_plan(ctx_h, future_actions)
        pred = model.propose_future(ctx_z, ctx_h, plan, horizon=config.horizon)
        score = model.score_future(ctx_h, pred, plan)
        inv = model.inverse_actions(ctx_h, future_z, plan)
        returns = discounted_returns(rewards, gamma=config.gamma)
        normal_mse = (pred.float() - future_z.float()).pow(2).mean()
        inv_mse = masked_mse(inv.float(), future_actions.float(), future_mask.float())
        candidates = sample_candidate_plans(model, ctx_z, ctx_h, config.num_candidates)
        cand_scores = candidates["scores"]
        selected = cand_scores.argmax(dim=1)
        rand_idx = torch.randint(0, config.num_candidates, (cand_scores.shape[0],), device=device)
        selected_scores = cand_scores.gather(1, selected[:, None]).squeeze(1)
        random_scores = cand_scores.gather(1, rand_idx[:, None]).squeeze(1)
        row = {
            "future_mse": item(normal_mse),
            "inverse_mse": item(inv_mse),
            "score_mse": item(F.mse_loss(score.float(), returns.float())),
            "score_return_corr": corr_item(score.float(), returns.float()),
            "candidate_selected_score": item(selected_scores.mean()),
            "candidate_random_score": item(random_scores.mean()),
            "candidate_selected_minus_random": item((selected_scores - random_scores).mean()),
        }
        _, contrast_metrics = plan_contrast_loss(
            model=model,
            ctx_z=ctx_z,
            ctx_h=ctx_h,
            future_z=future_z,
            future_actions=future_actions,
            future_mask=future_mask,
            normal_pred=pred,
            config=config,
        )
        row.update(contrast_metrics)
        if config.rank_loss_weight > 0:
            _, rank_metrics = candidate_rank_loss(
                model=model,
                ctx_z=ctx_z,
                ctx_h=ctx_h,
                future_z=future_z,
                future_actions=future_actions,
                future_mask=future_mask,
                true_plan=plan,
                config=config,
            )
            row.update(rank_metrics)
        rows.append(row)
    model.train()
    return average_rows(rows)


@torch.no_grad()
def write_trace(
    *,
    model: LatentImaginationPlanner,
    loader: DataLoader,
    encoder: nn.Module,
    patch: int,
    n_spatial: int,
    packing_factor: int,
    config: PlannerConfig,
    device: torch.device,
    path: Path,
) -> None:
    model.eval()
    raw_batch = next(iter(loader))
    batch = encode_batch(
        raw_batch=raw_batch,
        encoder=encoder,
        patch=patch,
        n_spatial=n_spatial,
        packing_factor=packing_factor,
        action_frame_offset=config.action_frame_offset,
        device=device,
    )
    ctx_z, ctx_actions, future_z, future_actions, _future_mask, rewards = split_batch(batch, config)
    ctx_h = model.encode_context(ctx_z, ctx_actions)
    candidates = sample_candidate_plans(model, ctx_z[:1], ctx_h[:1], config.num_candidates)
    scores = candidates["scores"][0]
    selected = int(scores.argmax().item())
    payload = {
        "scores": [float(x) for x in scores.detach().cpu().tolist()],
        "selected": selected,
        "reward_sum": float(rewards[:1].sum().detach().cpu().item()),
        "target_action_norm": float(future_actions[:1].norm(dim=-1).mean().detach().cpu().item()),
        "candidate_future_norms": [
            float(x) for x in candidates["futures"][0].flatten(1).norm(dim=-1).detach().cpu().tolist()
        ],
        "note": "Trace stores candidate scores and latent norms. Decode previews can be added once this scorer is useful.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    model.train()


def sample_candidate_plans(
    model: LatentImaginationPlanner,
    ctx_z: torch.Tensor,
    ctx_h: torch.Tensor,
    num_candidates: int,
) -> dict[str, torch.Tensor]:
    B = ctx_z.shape[0]
    K = int(num_candidates)
    plans = model.normalize_plan(torch.randn((B * K, model.plan_dim), device=ctx_z.device, dtype=ctx_z.dtype))
    ctx_z_rep = ctx_z[:, None].expand(B, K, *ctx_z.shape[1:]).reshape(B * K, *ctx_z.shape[1:])
    ctx_h_rep = ctx_h[:, None].expand(B, K, ctx_h.shape[-1]).reshape(B * K, ctx_h.shape[-1])
    futures = model.propose_future(ctx_z_rep, ctx_h_rep, plans, horizon=model.horizon)
    scores = model.score_future(ctx_h_rep, futures, plans)
    return {
        "plans": plans.view(B, K, -1),
        "futures": futures.view(B, K, *futures.shape[1:]),
        "scores": scores.view(B, K),
    }


def split_batch(
    batch: dict[str, torch.Tensor],
    config: PlannerConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ctx = int(config.ctx_len)
    horizon = int(config.horizon)
    end = ctx + horizon
    z_flat = batch["z"].flatten(2)
    ctx_z = z_flat[:, :ctx]
    future_z = z_flat[:, ctx:end]
    ctx_actions = batch["actions"][:, :ctx]
    future_actions = batch["transition_actions"][:, ctx:end]
    future_mask = batch["transition_mask"][:, ctx:end]
    rewards = batch["rewards"][:, ctx:end]
    return ctx_z, ctx_actions, future_z, future_actions, future_mask, rewards


@torch.no_grad()
def encode_batch(
    *,
    raw_batch: dict[str, torch.Tensor],
    encoder: nn.Module,
    patch: int,
    n_spatial: int,
    packing_factor: int,
    action_frame_offset: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    obs_u8 = raw_batch["obs"].to(device, non_blocking=True)
    act = raw_batch["act"].to(device, non_blocking=True).float()
    act_mask = raw_batch["act_mask"].to(device, non_blocking=True).float()
    rewards = raw_batch["rew"].to(device, non_blocking=True).float()
    act = act.clamp(-1, 1) * act_mask
    frames = obs_u8[:, :-1].float() / 255.0
    actions, mask = align_actions_to_frames(
        act,
        act_mask,
        frame_count=frames.shape[1],
        action_frame_offset=action_frame_offset,
    )
    patches = temporal_patchify(frames, patch)
    z_btld, _ = encoder(patches)
    z = pack_bottleneck_to_spatial(z_btld, n_spatial=n_spatial, k=packing_factor)
    return {
        "z": z,
        "actions": actions.clamp(-1, 1) * mask,
        "mask": mask,
        "transition_actions": act,
        "transition_mask": act_mask,
        "rewards": rewards,
    }


def discounted_returns(rewards: torch.Tensor, *, gamma: float) -> torch.Tensor:
    weights = torch.pow(
        torch.full((rewards.shape[1],), float(gamma), device=rewards.device, dtype=rewards.dtype),
        torch.arange(rewards.shape[1], device=rewards.device, dtype=rewards.dtype),
    )
    return (rewards * weights[None, :]).sum(dim=1)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - target).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)


def make_action_variant(actions: torch.Tensor, mask: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(mode)
    if mode == "shuffle":
        perm = torch.randperm(actions.shape[0], device=actions.device)
        return actions[perm], mask[perm]
    if mode == "zero":
        return torch.zeros_like(actions), torch.zeros_like(mask)
    if mode in {"time_reverse", "reverse"}:
        return actions.flip(1), mask.flip(1)
    if mode in {"time_perm", "same_traj_shuffle", "window_shuffle"}:
        idx = torch.stack([torch.randperm(actions.shape[1], device=actions.device) for _ in range(actions.shape[0])], dim=0)
        gather = idx[:, :, None].expand(-1, -1, actions.shape[-1])
        return actions.gather(1, gather), mask.gather(1, gather)
    shift = parse_temporal_shift(mode)
    if shift is not None:
        return torch.roll(actions, shifts=shift, dims=1), torch.roll(mask, shifts=shift, dims=1)
    raise ValueError(f"Unsupported contrast mode: {mode}")


def parse_temporal_shift(mode: str) -> int | None:
    if mode in {"time", "time_shift", "timeshift", "shift"}:
        return 1
    for prefix in ("time_shift", "timeshift", "shift"):
        if mode.startswith(prefix):
            suffix = mode[len(prefix):].lstrip("_-")
            if suffix.isdigit():
                return max(1, int(suffix))
    return None


def build_config(args: argparse.Namespace) -> PlannerConfig:
    data_dirs = list(args.data_dir)
    frame_dirs = list(args.frames_dir)
    tasks_json = str(args.tasks_json)
    source_rows: list[dict[str, Any]] = []
    if args.manifest_json:
        manifest_path = resolve_path(args.manifest_json)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = select_manifest_sources(manifest, args.source_names)
        for source in selected:
            weight = int(source.get("weight", 1))
            repeats = 1 if args.no_manifest_weights else max(0, weight)
            if repeats == 0:
                continue
            for _ in range(repeats):
                data_dirs.append(str(source["raw"]))
                frame_dirs.append(str(source["frames"]))
            source_rows.append({"name": source.get("name", ""), "weight": weight, "repeats": repeats})
        tasks_json = tasks_json or str(manifest.get("tasks_json", ""))
        if not args.action_features or args.action_features == "current":
            args.action_features = str(manifest.get("action_features", args.action_features))
        if args.action_dim == DEFAULT_ACTION_DIM and "action_dim" in manifest:
            args.action_dim = int(manifest["action_dim"])
        if args.action_frame_offset == -1 and "action_frame_offset" in manifest:
            args.action_frame_offset = int(manifest["action_frame_offset"])
    if not data_dirs or not frame_dirs:
        raise ValueError("Provide --data-dir/--frames-dir or --manifest-json with positive source weights.")
    if len(data_dirs) != len(frame_dirs):
        raise ValueError(f"data-dir and frames-dir counts differ: {len(data_dirs)} vs {len(frame_dirs)}")
    cfg = PlannerConfig(
        data_dirs=data_dirs,
        frame_dirs=frame_dirs,
        tasks_json=tasks_json,
        tokenizer_ckpt=str(args.tokenizer_ckpt),
        out_dir=str(args.out_dir),
        resume_ckpt=str(args.resume_ckpt),
        manifest_json=str(args.manifest_json),
        source_names=str(args.source_names),
        use_manifest_weights=not bool(args.no_manifest_weights),
        seq_len=int(args.seq_len),
        ctx_len=int(args.ctx_len),
        horizon=int(args.horizon),
        img_size=int(args.img_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_steps=int(args.max_steps),
        eval_every=int(args.eval_every),
        eval_batches=int(args.eval_batches),
        save_every=int(args.save_every),
        trace_every=int(args.trace_every),
        action_dim=int(args.action_dim),
        raw_action_dim=int(args.raw_action_dim),
        action_features=str(args.action_features),
        action_frame_offset=int(args.action_frame_offset),
        hidden_dim=int(args.hidden_dim),
        plan_dim=int(args.plan_dim),
        num_candidates=int(args.num_candidates),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        gamma=float(args.gamma),
        future_loss_weight=float(args.future_loss_weight),
        reward_loss_weight=float(args.reward_loss_weight),
        inverse_loss_weight=float(args.inverse_loss_weight),
        contrast_weight=float(args.contrast_weight),
        contrast_margin=float(args.contrast_margin),
        contrast_relative_margin=float(args.contrast_relative_margin),
        plan_l2_weight=float(args.plan_l2_weight),
        effect_loss_weight=float(args.effect_loss_weight),
        plan_unit_norm=bool(args.plan_unit_norm),
        plan_step_conditioning=bool(args.plan_step_conditioning),
        rank_loss_weight=float(args.rank_loss_weight),
        rank_num_bank=int(args.rank_num_bank),
        rank_num_matched=int(args.rank_num_matched),
        rank_margin=float(args.rank_margin),
        rank_mse_gap=float(args.rank_mse_gap),
        inverse_plan_dropout=float(args.inverse_plan_dropout),
        inverse_imagined_weight=float(args.inverse_imagined_weight),
        inverse_cross_weight=float(args.inverse_cross_weight),
        score_plan_dropout=float(args.score_plan_dropout),
        bc_head_weight=float(args.bc_head_weight),
        bc_encoder_grad=bool(args.bc_encoder_grad),
        horizon_curriculum_max=int(args.horizon_curriculum_max),
        horizon_curriculum_weight=float(args.horizon_curriculum_weight),
        contrast_modes=str(args.contrast_modes),
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
        device=str(args.device),
        seed=int(args.seed),
        amp=not bool(args.no_amp),
    )
    if cfg.ctx_len + cfg.horizon > cfg.seq_len:
        raise ValueError(f"ctx_len + horizon must be <= seq_len, got {cfg.ctx_len} + {cfg.horizon} > {cfg.seq_len}")
    return cfg


def select_manifest_sources(manifest: dict[str, Any], source_names: str) -> list[dict[str, Any]]:
    sources = list(manifest.get("sources", []))
    requested = [name.strip() for name in str(source_names).split(",") if name.strip()]
    if not requested:
        return sources
    out = []
    for name in requested:
        matches = [source for source in sources if source.get("name") == name]
        if not matches:
            raise ValueError(f"source not found in manifest: {name}")
        out.append(matches[0])
    return out


def average_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in keys
    }


def metric_key(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def corr_item(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return 0.0
    x = x.flatten().float()
    y = y.flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.detach().cpu().item()) <= 1e-12:
        return 0.0
    return item((x * y).sum() / denom)


def item(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_report(path: Path, *, config: PlannerConfig, latest: dict[str, Any]) -> None:
    lines = [
        "# Latent Imagination Planner Run",
        "",
        "## Config",
        "",
        f"- `max_steps`: {config.max_steps}",
        f"- `seq_len`: {config.seq_len}",
        f"- `ctx_len`: {config.ctx_len}",
        f"- `horizon`: {config.horizon}",
        f"- `action_features`: {config.action_features}",
        f"- `contrast_modes`: {config.contrast_modes}",
        f"- `manifest_json`: {config.manifest_json}",
        "",
        "## Latest Metrics",
        "",
    ]
    for key in sorted(latest):
        lines.append(f"- `{key}`: {latest[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    config: PlannerConfig,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "planner": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(config),
            "step": int(step),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["planner"], strict=False)
    if missing or unexpected:
        print(json.dumps({"phase": "resume_warning", "missing_keys": list(missing), "unexpected_keys": list(unexpected)}), flush=True)
    if "opt" in ckpt and ckpt["opt"] is not None:
        try:
            opt.load_state_dict(ckpt["opt"])
        except Exception as exc:
            print(json.dumps({"phase": "resume_warning", "message": f"failed to restore optimizer (fresh opt state): {exc}"}), flush=True)
    if "scaler" in ckpt and ckpt["scaler"] is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as exc:
            print(json.dumps({"phase": "resume_warning", "message": f"failed to restore scaler: {exc}"}), flush=True)
    return int(ckpt.get("step", 0))


def seed_everything(seed: int) -> None:
    seed = int(seed) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
