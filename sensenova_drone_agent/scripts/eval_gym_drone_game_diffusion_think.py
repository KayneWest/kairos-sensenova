#!/usr/bin/env python3
"""Closed-loop eval: diffusion-proposer thinking vs the judge-amplifier.

Policies (matched episode seeds across all):
  bc           BC chunk head argmax (no thinking; control)
  bc_random    random BC candidate executed (thinking without a judge; control)
  diff_prior   plan-free diffusion sample -> plan-free inverse decode
               (guided with lambda=0; generative-BC control)
  diff_argmax  K BC candidates -> plan-conditioned diffusion sample each ->
               judge scores plan-free -> argmax candidate's actions
               (hard optimizer against the judge)
  diff_guided  ONE plan-free diffusion sample, value-guided at every DDIM
               step by the judge's gradient -> plan-free inverse decode
               (soft, prior-anchored optimizer)

The PROPOSER stack (ctx/plan encoders, BC head, inverse head) is held fixed
(--planner-ckpt); the JUDGE (--judge-ckpt: its own ctx encoder + scorer) is
swappable — pass the certified-good or certified-inverted planner to stress
the amplifier hypothesis. Writes summary.json with paired bootstrap gates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for item in (str(REPO_ROOT / "dreamer4" / "dreamer4"), str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT / "src")):
    if item not in sys.path:
        sys.path.insert(0, item)

import numpy as np
import torch

from sensenova_drone.gym_drone_game import DroneGameConfig, DroneMazeEnv
from train_dynamics import align_actions_to_frames, pack_bottleneck_to_spatial, temporal_patchify  # noqa: E402
from train_drone_bc_chunk_head import BCChunkHead  # noqa: E402
from train_latent_diffusion_proposer import DiffusionProposer, ddim_sample, load_planner  # noqa: E402
from train_latent_imagination_planner import resolve_path, seed_everything  # noqa: E402
import eval_gym_drone_game_act_by_imagination as ev  # noqa: E402

NUM_ACTIONS = 9
POLICIES = ["bc", "bc_random", "diff_prior", "diff_argmax", "diff_guided"]


class Stack:
    """Everything needed at act time, bundled."""

    def __init__(self, args, device):
        self.device = device
        self.planner, self.cfg, self.encoder, self.patch, self.pk, self.ns, self.z_dim = \
            load_planner(args.planner_ckpt, device)
        self.judge, self.judge_cfg, _enc2, _p2, _k2, _n2, _z2 = load_planner(args.judge_ckpt, device)
        hck = torch.load(resolve_path(args.bc_head), map_location="cpu", weights_only=False)
        self.bc_head = BCChunkHead(**hck["config"]).to(device)
        self.bc_head.load_state_dict(hck["head"])
        self.bc_head.eval()
        dck = torch.load(resolve_path(args.diffusion_ckpt), map_location="cpu", weights_only=False)
        dc = dck["config"]
        self.diff = DiffusionProposer(x_dim=dc["x_dim"], ctx_dim=dc["ctx_dim"], plan_dim=dc["plan_dim"],
                                      z_dim=dc["z_dim"], width=dc["width"], depth=dc["depth"],
                                      plan_in_trunk=bool(dc.get("plan_in_trunk", False))).to(device)
        self.diff.load_state_dict(dck["model"])
        self.diff.eval()
        self.T = int(dc["diffusion_T"])
        self.ddim_steps = int(args.ddim_steps or dc["ddim_steps"])
        self.mu = dck["norm"]["mu"].to(device)        # per-dim, (h*z_dim,)
        self.sigma = dck["norm"]["sigma"].to(device)  # per-dim, (h*z_dim,)
        self.h = int(self.cfg.horizon)

    @torch.no_grad()
    def encode_ctx(self, frames, act_hist):
        """Returns (ctx_z_flat, ctx_h_proposer, ctx_h_judge)."""
        obs = torch.from_numpy(np.stack(frames)).to(self.device)[None].float() / 255.0
        patches = temporal_patchify(obs, self.patch)
        z, _ = self.encoder(patches)
        ctx_z = pack_bottleneck_to_spatial(z, n_spatial=self.ns, k=self.pk).flatten(2)
        act = torch.from_numpy(np.stack(act_hist)).to(self.device)[None].float()
        mask = torch.ones_like(act)
        actions, amask = align_actions_to_frames(act, mask, frame_count=int(self.cfg.ctx_len),
                                                 action_frame_offset=self.cfg.action_frame_offset)
        acts = actions.clamp(-1, 1) * amask
        return ctx_z, self.planner.encode_context(ctx_z, acts), self.judge.encode_context(ctx_z, acts)

    def to_future(self, x_norm_flat, last_z):
        """Normalized persistence-delta -> absolute future latents (B, h, z)."""
        delta = x_norm_flat * self.sigma[None, :] + self.mu[None, :]
        fut = delta.view(-1, self.h, self.z_dim) + last_z[:, None, :]
        return fut

    def judge_score_fn(self, ctx_h_judge, last_z):
        """Score futures (from normalized deltas) with the judge, plan-free."""
        zero_plan = torch.zeros((1, self.judge.plan_dim), device=self.device)

        def fn(x0_norm_flat):
            fut = self.to_future(x0_norm_flat, last_z.expand(x0_norm_flat.shape[0], -1))
            chj = ctx_h_judge.expand(fut.shape[0], -1)
            zp = zero_plan.expand(fut.shape[0], -1)
            return self.judge.score_future(chj, fut, zp)
        return fn


def plan_chunk_diffusion(stack: Stack, *, policy, frames, act_hist, K, rng, lam, gen):
    ctx_z, ctx_h, ctx_h_j = stack.encode_ctx(frames, act_hist)
    last_z = ctx_z[:, -1]  # (1, z_dim)
    h, dev = stack.h, stack.device
    with torch.no_grad():
        logits = stack.bc_head(ctx_h)[0] / 0.8  # bc_temperature matched to prior evals

    if policy == "bc":
        return [int(i) for i in logits.argmax(dim=-1).tolist()]

    if policy in ("bc_random", "diff_argmax", "gru_argmax"):
        probs = torch.softmax(logits, dim=-1)
        with torch.no_grad():
            samples = torch.multinomial(probs, K - 1, replacement=True, generator=None).T
        chunks = [[int(i) for i in logits.argmax(dim=-1).tolist()]]
        chunks += [[int(i) for i in row.tolist()] for row in samples]
        if policy == "bc_random":
            return list(chunks[int(rng.integers(0, len(chunks)))])
        onehot = torch.zeros((len(chunks), h, NUM_ACTIONS), device=dev)
        for i, chunk in enumerate(chunks):
            for t, a in enumerate(chunk):
                onehot[i, t, a] = 1.0
        with torch.no_grad():
            chr_ = ctx_h.expand(len(chunks), -1)
            plans = stack.planner.encode_plan(chr_, onehot)
            if policy == "gru_argmax":
                # act_bc_think with a swappable judge: the ACTION-CONDITIONED
                # GRU proposer imagines each candidate's future; the judge
                # (its own ctx encoder + scorer, plan-free) picks. This is
                # the true hard-optimizer/amplifier arm.
                czr = ctx_z.expand(len(chunks), *ctx_z.shape[1:])
                futures = stack.planner.propose_future(czr, chr_, plans, horizon=h)
                chj = ctx_h_j.expand(len(chunks), -1)
                zp = torch.zeros((len(chunks), stack.judge.plan_dim), device=dev)
                scores = stack.judge.score_future(chj, futures, zp)
            else:
                lzr = last_z.expand(len(chunks), -1)
                x = ddim_sample(stack.diff, ctx_h=chr_, plan=plans, last_z=lzr, steps=stack.ddim_steps, T=stack.T,
                                shape=(len(chunks), stack.h * stack.z_dim), device=dev, generator=gen)
                scores = stack.judge_score_fn(ctx_h_j, last_z)(x)
        pick = int(scores.argmax())
        return list(chunks[pick])

    if policy == "plan_grad":
        # Soft thinking on the plan manifold: gradient-ascend the judge's
        # score THROUGH the action-conditioned GRU proposer, re-projecting
        # onto the unit-norm plan sphere each step; decode actions plan-free
        # from the final imagined future. "Force z down a thinking
        # trajectory" where actions have causal grip.
        best = [int(i) for i in logits.argmax(dim=-1).tolist()]
        onehot = torch.zeros((1, h, NUM_ACTIONS), device=dev)
        for t, a in enumerate(best):
            onehot[0, t, a] = 1.0
        zero_plan_j = torch.zeros((1, stack.judge.plan_dim), device=dev)
        zero_plan_p = torch.zeros((1, stack.planner.plan_dim), device=dev)
        with torch.no_grad():
            plan = stack.planner.encode_plan(ctx_h, onehot)
        for _ in range(int(lam_steps := 10)):
            with torch.enable_grad():
                pl = plan.detach().requires_grad_(True)
                fut = stack.planner.propose_future(ctx_z, ctx_h, pl, horizon=h)
                sc = stack.judge.score_future(ctx_h_j, fut, zero_plan_j).sum()
                g = torch.autograd.grad(sc, pl)[0]
            plan = plan + 0.5 * g
            plan = torch.nn.functional.normalize(plan, dim=-1) * (stack.planner.plan_dim ** 0.5)
        with torch.no_grad():
            fut = stack.planner.propose_future(ctx_z, ctx_h, plan, horizon=h)
            act_logits = stack.planner.inverse_actions(ctx_h, fut, zero_plan_p)[0]
        return [int(i) for i in act_logits.argmax(dim=-1).tolist()]

    if policy in ("diff_prior", "diff_guided"):
        scale = 0.0 if policy == "diff_prior" else float(lam)
        zero_plan = torch.zeros((1, stack.planner.plan_dim), device=dev)
        x = ddim_sample(stack.diff, ctx_h=ctx_h, plan=zero_plan, last_z=last_z, steps=stack.ddim_steps, T=stack.T,
                        shape=(1, stack.h * stack.z_dim), device=dev, generator=gen,
                        guidance_fn=stack.judge_score_fn(ctx_h_j, last_z) if scale > 0 else None,
                        guidance_scale=scale)
        with torch.no_grad():
            fut = stack.to_future(x, last_z)
            act_logits = stack.planner.inverse_actions(ctx_h, fut, zero_plan)[0]
        return [int(i) for i in act_logits.argmax(dim=-1).tolist()]

    raise ValueError(policy)


def paired_ci(a: np.ndarray, b: np.ndarray, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {"mean": float(d.mean()), "ci_lo": float(np.quantile(means, 0.025)),
            "ci_hi": float(np.quantile(means, 0.975))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--planner-ckpt", required=True, help="Proposer stack (ctx/plan encoders + inverse head).")
    p.add_argument("--judge-ckpt", required=True, help="Judge stack (its ctx encoder + scorer).")
    p.add_argument("--bc-head", required=True)
    p.add_argument("--diffusion-ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--num-candidates", type=int, default=32)
    p.add_argument("--replan-every", type=int, default=4)
    p.add_argument("--ddim-steps", type=int, default=0)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--policies", default=",".join(POLICIES))
    p.add_argument("--seed", type=int, default=20260710)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = Stack(args, device)
    policies = [s for s in args.policies.split(",") if s]
    jsonl = (out_dir / "episodes.jsonl").open("w")

    per_policy, returns, succ = {}, {}, {}
    for policy in policies:
        env = DroneMazeEnv(DroneGameConfig(max_episode_steps=args.max_steps))
        rng = np.random.default_rng(args.seed + 17)
        gen = torch.Generator(device=device.type).manual_seed(args.seed + 23)
        rows = []
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep)  # matched seeds across policies
            frame = ev.resize_chw_uint8(env.render(), int(stack.cfg.img_size))
            frames = [frame] * int(stack.cfg.ctx_len)
            act_hist = [np.zeros(NUM_ACTIONS, dtype=np.float32)] * int(stack.cfg.ctx_len)
            ep_return, steps, term_r, pending, done = 0.0, 0, 0.0, [], False
            while not done and steps < args.max_steps:
                if not pending:
                    pending = plan_chunk_diffusion(stack, policy=policy, frames=frames,
                                                   act_hist=act_hist, K=args.num_candidates,
                                                   rng=rng, lam=args.guidance_scale,
                                                   gen=gen)[: args.replan_every]
                a = pending.pop(0)
                _o, r, t1, t2, _i = env.step(a)
                ep_return += float(r)
                term_r = float(r)
                steps += 1
                frame = ev.resize_chw_uint8(env.render(), int(stack.cfg.img_size))
                frames = frames[1:] + [frame]
                v = np.zeros(NUM_ACTIONS, dtype=np.float32)
                v[a] = 1.0
                act_hist = act_hist[1:] + [v]
                done = t1 or t2
            outcome = "success" if (done and term_r > 5.0) else ("collision_or_oob" if (done and term_r < -4.0) else "timeout")
            row = {"policy": policy, "episode": ep, "return": ep_return, "steps": steps, "outcome": outcome}
            rows.append(row)
            jsonl.write(json.dumps(row) + "\n")
            if (ep + 1) % 100 == 0:
                sr = sum(1 for x in rows if x["outcome"] == "success") / len(rows)
                print(json.dumps({"policy": policy, "ep": ep + 1, "success_rate": round(sr, 4)}), flush=True)
        returns[policy] = np.array([x["return"] for x in rows])
        succ[policy] = np.array([1.0 if x["outcome"] == "success" else 0.0 for x in rows])
        per_policy[policy] = {
            "mean_return": float(returns[policy].mean()),
            "success_rate": float(succ[policy].mean()),
            "collision_or_oob_rate": float(np.mean([x["outcome"] == "collision_or_oob" for x in rows])),
            "timeout_rate": float(np.mean([x["outcome"] == "timeout" for x in rows])),
            "mean_steps": float(np.mean([x["steps"] for x in rows])),
        }
        print(json.dumps({"policy": policy, **per_policy[policy]}), flush=True)

    gates = {}
    for think in ("diff_argmax", "diff_guided", "diff_prior", "gru_argmax", "plan_grad"):
        if think not in per_policy:
            continue
        for ctrl in ("bc", "bc_random", "diff_prior"):
            if ctrl not in per_policy or ctrl == think:
                continue
            gates[f"{think}_vs_{ctrl}_return"] = paired_ci(returns[think], returns[ctrl])
            gates[f"{think}_vs_{ctrl}_success"] = paired_ci(succ[think], succ[ctrl])
    for think in ("diff_argmax", "diff_guided", "gru_argmax", "plan_grad"):
        if think in per_policy and "bc" in per_policy and "bc_random" in per_policy:
            gates[f"{think}_wins"] = bool(
                gates[f"{think}_vs_bc_success"]["ci_lo"] > 0
                and gates[f"{think}_vs_bc_random_success"]["ci_lo"] > 0
                and gates[f"{think}_vs_bc_return"]["ci_lo"] > 0
                and gates[f"{think}_vs_bc_random_return"]["ci_lo"] > 0)

    summary = {"meta": {"phase": "closed_loop_diffusion_think", "episodes": args.episodes,
                        "num_candidates": args.num_candidates, "replan_every": args.replan_every,
                        "guidance_scale": args.guidance_scale, "ddim_steps": stack.ddim_steps,
                        "planner_ckpt": args.planner_ckpt, "judge_ckpt": args.judge_ckpt,
                        "diffusion_ckpt": args.diffusion_ckpt, "seed": args.seed,
                        "policies": policies},
               "per_policy": per_policy, "gates": gates}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"phase": "done", "gates": {k: v for k, v in gates.items() if k.endswith("_wins")}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
