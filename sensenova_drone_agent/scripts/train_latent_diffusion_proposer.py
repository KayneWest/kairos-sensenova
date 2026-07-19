#!/usr/bin/env python3
"""Train a conditional latent-diffusion proposer over frozen tokenizer latents.

Replaces the deterministic GRU rollout as the SAMPLER of imagined futures:
p(future_z | ctx_h, plan) with plan-dropout, so futures can be sampled
plan-conditioned (candidate-style thinking) or plan-free (guided-refinement
thinking, where the value head's gradient steers the denoising trajectory —
"forcing z down a thinking trajectory before releasing it").

Conditioning uses a FROZEN trained planner's context/plan encoders, so the
proposer plugs into the existing think-then-act stack unchanged. Saves
{"model", "config", "norm", "planner_ckpt", "step"}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DREAMER4_ROOT = REPO_ROOT / "dreamer4" / "dreamer4"
for item in (str(DREAMER4_ROOT), str(PROJECT_ROOT / "scripts")):
    if item not in sys.path:
        sys.path.insert(0, item)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_dynamics import load_frozen_tokenizer_from_pt_ckpt  # noqa: E402
from wm_dataset import WMDataset, collate_batch  # noqa: E402
from train_latent_imagination_planner import (  # noqa: E402
    LatentImaginationPlanner,
    PlannerConfig,
    encode_batch,
    make_action_variant,
    resolve_path,
    seed_everything,
    split_batch,
)


def cosine_alpha_bar(t: torch.Tensor, T: int) -> torch.Tensor:
    s = 0.008
    f = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    f0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
    return (f / f0).clamp(1e-5, 1.0)


class FiLMBlock(nn.Module):
    def __init__(self, width: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(cond_dim, width * 2)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        h = self.fc2(torch.nn.functional.silu(self.fc1(torch.nn.functional.silu(h))))
        return x + h


class DiffusionProposer(nn.Module):
    """Epsilon-predictor over the flattened future chunk (horizon * z_dim)."""

    def __init__(self, *, x_dim: int, ctx_dim: int, plan_dim: int, z_dim: int, width: int = 2048, depth: int = 4, t_dim: int = 256):
        super().__init__()
        self.x_dim, self.ctx_dim, self.plan_dim = int(x_dim), int(ctx_dim), int(plan_dim)
        self.z_dim = int(z_dim)
        self.width, self.depth, self.t_dim = int(width), int(depth), int(t_dim)
        cond_dim = 512
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.ctx_proj = nn.Linear(ctx_dim, cond_dim)
        self.plan_proj = nn.Linear(plan_dim, cond_dim)
        self.z_proj = nn.Linear(z_dim, cond_dim)  # last ctx frame latent — the anchor the delta continues from
        self.in_proj = nn.Linear(x_dim, width)
        self.blocks = nn.ModuleList([FiLMBlock(width, cond_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, x_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def time_embed(self, t: torch.Tensor) -> torch.Tensor:
        half = self.t_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(1, half - 1))
        ang = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, ctx_h: torch.Tensor, plan: torch.Tensor, last_z: torch.Tensor) -> torch.Tensor:
        cond = self.t_mlp(self.time_embed(t)) + self.ctx_proj(ctx_h) + self.plan_proj(plan) + self.z_proj(last_z)
        h = self.in_proj(x_t)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_proj(torch.nn.functional.silu(self.out_norm(h)))


@torch.no_grad()
def ddim_sample(model, *, ctx_h, plan, last_z, steps, T, shape, device, generator=None,
                guidance_fn=None, guidance_scale=0.0):
    """DDIM sampling; optional value guidance shifts x_t toward higher score.

    guidance_fn(x0_flat) -> scalar score per batch row, computed on the
    UNNORMALIZED future (caller closes over norm stats + judge). Gradients
    are taken wrt x_t through the x0 estimate (grad enabled locally).
    """
    x = torch.randn(shape, device=device, generator=generator)
    ts = torch.linspace(T - 1, 0, steps, device=device).long()
    for i, t in enumerate(ts):
        tb = t.expand(shape[0])
        ab_t = cosine_alpha_bar(tb.float(), T)[:, None]
        if guidance_fn is not None and guidance_scale > 0:
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(True)
                eps = model(x_req, tb, ctx_h, plan, last_z)
                x0 = (x_req - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
                score = guidance_fn(x0).sum()
                grad = torch.autograd.grad(score, x_req)[0]
            eps = eps.detach() - guidance_scale * (1 - ab_t).sqrt() * grad
        else:
            eps = model(x, tb, ctx_h, plan, last_z)
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        x0 = x0.clamp(-4.0, 4.0)
        if i == len(ts) - 1:
            x = x0
        else:
            ab_prev = cosine_alpha_bar(ts[i + 1].float().expand(shape[0]), T)[:, None]
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return x


def load_planner(path: str, device: torch.device):
    ckpt = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    cfg = PlannerConfig(**ckpt["config"])
    encoder, _d, tok = load_frozen_tokenizer_from_pt_ckpt(str(resolve_path(cfg.tokenizer_ckpt)), device=device)
    patch, pk = int(tok.get("patch", 8)), int(tok.get("packing_factor", 2))
    ns = int(tok.get("n_latents", 16)) // pk
    z_dim = ns * int(tok.get("d_bottleneck", 32)) * pk
    planner = LatentImaginationPlanner(
        z_dim=z_dim, action_dim=cfg.action_dim, hidden_dim=cfg.hidden_dim, plan_dim=cfg.plan_dim,
        horizon=cfg.horizon, plan_unit_norm=cfg.plan_unit_norm,
        plan_step_conditioning=getattr(cfg, "plan_step_conditioning", False))
    if getattr(cfg, "bc_head_weight", 0.0) > 0:
        planner.enable_bc_head()
    planner = planner.to(device)
    planner.load_state_dict(ckpt["planner"], strict=True)
    planner.eval()
    for p_ in planner.parameters():
        p_.requires_grad_(False)
    return planner, cfg, encoder, patch, pk, ns, z_dim


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--planner-ckpt", required=True, help="Frozen planner providing ctx/plan encoders + tokenizer path.")
    p.add_argument("--manifest-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--diffusion-T", type=int, default=1000)
    p.add_argument("--plan-dropout", type=float, default=0.5)
    p.add_argument("--contrast-weight", type=float, default=0.0,
                   help="Absolute-margin hinge: wrong-plan x0 reconstructions must be worse than true-plan by the margin.")
    p.add_argument("--contrast-margin", type=float, default=0.05)
    p.add_argument("--contrast-modes", default="shuffle,zero,time_shift,time_shift2,time_perm,time_reverse")
    p.add_argument("--contrast-per-step", type=int, default=2, help="Contrast modes sampled per training step.")
    p.add_argument("--norm-batches", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-samples", type=int, default=64)
    p.add_argument("--ddim-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260710)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (out_dir / "metrics.jsonl").open("a")

    planner, cfg, encoder, patch, pk, ns, z_dim = load_planner(args.planner_ckpt, device)

    manifest = json.loads(resolve_path(args.manifest_json).read_text())
    data_dirs, frame_dirs = [], []
    for src in manifest["sources"]:
        for _ in range(int(src.get("weight", 1))):
            data_dirs.append(src["raw"])
            frame_dirs.append(src["frames"])
    dataset = WMDataset(
        data_dir=[str(resolve_path(d)) for d in data_dirs],
        frames_dir=[str(resolve_path(d)) for d in frame_dirs],
        seq_len=cfg.seq_len, img_size=cfg.img_size, action_dim=cfg.action_dim,
        raw_action_dim=cfg.raw_action_dim,
        tasks_json=str(resolve_path(manifest["tasks_json"])),
        tasks=None, strict_tasks=False, action_features=cfg.action_features, verbose=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        drop_last=True, collate_fn=collate_batch, persistent_workers=args.num_workers > 0)
    x_dim = cfg.horizon * z_dim
    model = DiffusionProposer(x_dim=x_dim, ctx_dim=cfg.hidden_dim, plan_dim=cfg.plan_dim,
                              z_dim=z_dim, width=args.width, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    n_params = sum(q.numel() for q in model.parameters())
    print(json.dumps({"phase": "diffusion_train", "windows": len(dataset), "params": n_params,
                      "x_dim": x_dim, "steps": args.steps}), flush=True)

    def batches():
        while True:
            for raw in loader:
                yield raw

    it = batches()

    def prep(raw):
        with torch.no_grad():
            batch = encode_batch(raw_batch=raw, encoder=encoder, patch=patch, n_spatial=ns,
                                 packing_factor=pk, action_frame_offset=cfg.action_frame_offset, device=device)
            ctx_z, ctx_actions, future_z, future_actions, _fm, _rw = split_batch(batch, cfg)
            ctx_h = planner.encode_context(ctx_z, ctx_actions)
            plan = planner.encode_plan(ctx_h, future_actions)
            last_z = ctx_z[:, -1]
            delta = (future_z - last_z[:, None, :]).flatten(1)
        return ctx_h, plan, last_z, delta, future_z, future_actions

    # --- per-dim normalization stats over the persistence DELTA ---
    print(json.dumps({"phase": "norm_stats", "batches": args.norm_batches}), flush=True)
    acc = torch.zeros(x_dim, device=device)
    acc2 = torch.zeros(x_dim, device=device)
    n_acc = 0
    for _ in range(args.norm_batches):
        _, _, _, delta, _, _ = prep(next(it))
        acc += delta.sum(dim=0)
        acc2 += (delta ** 2).sum(dim=0)
        n_acc += delta.shape[0]
    mu = acc / n_acc
    sigma = (acc2 / n_acc - mu ** 2).clamp_min(1e-8).sqrt().clamp_min(1e-3)
    print(json.dumps({"phase": "norm_stats_done", "mu_mean": float(mu.mean()), "sigma_mean": float(sigma.mean())}), flush=True)

    # fixed eval batch (held aside from the stream, not strictly held out —
    # generative MSE here is a training diagnostic; decision-quality is
    # measured downstream)
    eval_raw = next(it)
    eval_ctx_h, eval_plan, eval_last_z, eval_delta, eval_future, _efa = prep(eval_raw)
    ns_eval = min(args.eval_samples, eval_ctx_h.shape[0])
    eval_ctx_h, eval_plan = eval_ctx_h[:ns_eval], eval_plan[:ns_eval]
    eval_last_z, eval_future = eval_last_z[:ns_eval], eval_future[:ns_eval].flatten(1)
    persist_mse = torch.nn.functional.mse_loss(
        eval_last_z[:, None, :].expand(-1, cfg.horizon, -1).flatten(1), eval_future).item()

    def denorm_to_future(x_norm):
        delta = x_norm * sigma[None, :] + mu[None, :]
        return delta + eval_last_z[:, None, :].expand(-1, cfg.horizon, -1).flatten(1)

    started = time.time()
    T = args.diffusion_T
    contrast_modes = [m for m in args.contrast_modes.split(",") if m]
    mode_rng = torch.Generator().manual_seed(args.seed + 7)
    for step in range(1, args.steps + 1):
        ctx_h, plan_true, last_z, delta, _, future_actions = prep(next(it))
        x0 = (delta - mu[None, :]) / sigma[None, :]
        plan = plan_true
        if args.plan_dropout > 0:
            drop = (torch.rand(plan.shape[0], 1, device=device) < args.plan_dropout).float()
            plan = plan * (1 - drop)
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        ab = cosine_alpha_bar(t.float(), T)[:, None]
        noise = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        with torch.amp.autocast("cuda"):
            eps = model(x_t, t, ctx_h, plan, last_z)
            loss_eps = torch.nn.functional.mse_loss(eps.float(), noise)
            loss = loss_eps
            contrast_val, contrast_ratio = 0.0, 1.0
            if args.contrast_weight > 0 and contrast_modes:
                # Contrast at low-noise t where reconstruction is feasible:
                # given the TRUE plan the model must reconstruct better than
                # given a wrong-action plan, by an ABSOLUTE margin (relative
                # margins collapse plan-sensitivity — arm C).
                t_c = torch.randint(0, T // 2, (x0.shape[0],), device=device)
                ab_c = cosine_alpha_bar(t_c.float(), T)[:, None]
                noise_c = torch.randn_like(x0)
                x_tc = ab_c.sqrt() * x0 + (1 - ab_c).sqrt() * noise_c
                eps_true = model(x_tc, t_c, ctx_h, plan_true, last_z)
                x0_true = (x_tc - (1 - ab_c).sqrt() * eps_true) / ab_c.sqrt()
                mse_true = (x0_true.float() - x0).pow(2).mean(dim=1)
                loss_c = x0.new_zeros(())
                wrong_mses = []
                idx = torch.randperm(len(contrast_modes), generator=mode_rng)[: args.contrast_per_step]
                for mi in idx.tolist():
                    wa, _wm = make_action_variant(future_actions, torch.ones_like(future_actions), contrast_modes[mi])
                    with torch.no_grad():
                        plan_w = planner.encode_plan(ctx_h, wa)
                    eps_w = model(x_tc, t_c, ctx_h, plan_w, last_z)
                    x0_w = (x_tc - (1 - ab_c).sqrt() * eps_w) / ab_c.sqrt()
                    mse_w = (x0_w.float() - x0).pow(2).mean(dim=1)
                    wrong_mses.append(mse_w.mean().item())
                    loss_c = loss_c + torch.relu(mse_true + args.contrast_margin - mse_w).mean()
                loss_c = loss_c / max(1, len(idx))
                loss = loss + args.contrast_weight * loss_c
                contrast_val = float(loss_c.item())
                contrast_ratio = float(sum(wrong_mses) / max(1, len(wrong_mses)) / max(1e-8, mse_true.mean().item()))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % 200 == 0:
            row = {"step": step, "loss": float(loss.item()), "loss_eps": float(loss_eps.item()),
                   "contrast": contrast_val, "wrong_over_true": round(contrast_ratio, 4),
                   "elapsed_s": time.time() - started}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            gen = torch.Generator(device=device.type).manual_seed(args.seed + step)
            with torch.no_grad():
                s_plan = ddim_sample(model, ctx_h=eval_ctx_h, plan=eval_plan, last_z=eval_last_z,
                                     steps=args.ddim_steps, T=T, shape=(ns_eval, x_dim), device=device, generator=gen)
                s_free = ddim_sample(model, ctx_h=eval_ctx_h, plan=torch.zeros_like(eval_plan), last_z=eval_last_z,
                                     steps=args.ddim_steps, T=T, shape=(ns_eval, x_dim),
                                     device=device, generator=gen)
            mse_plan = torch.nn.functional.mse_loss(denorm_to_future(s_plan), eval_future).item()
            mse_free = torch.nn.functional.mse_loss(denorm_to_future(s_free), eval_future).item()
            row = {"step": step, "eval": True, "sample_mse_plan": mse_plan, "sample_mse_planfree": mse_free,
                   "persist_mse": persist_mse, "plan_over_free": mse_plan / max(1e-9, mse_free)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            model.train()

    ckpt = {"model": model.state_dict(),
            "config": {"x_dim": x_dim, "ctx_dim": cfg.hidden_dim, "plan_dim": cfg.plan_dim,
                       "width": args.width, "depth": args.depth, "diffusion_T": T,
                       "horizon": cfg.horizon, "z_dim": z_dim, "ddim_steps": args.ddim_steps},
            "norm": {"mu": mu.cpu(), "sigma": sigma.cpu(), "parameterization": "persistence_delta_perdim"},
            "planner_ckpt": str(args.planner_ckpt), "step": args.steps}
    torch.save(ckpt, out_dir / "final.pt")
    print(json.dumps({"phase": "done", "out": str(out_dir / "final.pt")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
