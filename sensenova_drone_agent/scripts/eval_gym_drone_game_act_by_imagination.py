#!/usr/bin/env python3
"""Closed-loop act-by-imagination evaluation in the gym drone game.

The behavioral verdict for the think-then-act loop: run real episodes where
the agent, every `replan_every` steps, imagines K candidate futures from its
recent frames, scores them, selects one, and decodes it (plan-free) into an
action chunk that is actually executed. Compared against matched-seed
controls:

  act_mpc         action-chunk candidates -> encode_plan -> imagine -> score
                  -> execute the WINNING CHUNK'S OWN ACTIONS (think-then-act;
                  no inverse head at act time)
  act_mpc_random  identical candidate set, uniform random selection
                  (imagine but do not evaluate - the exact causal control)
  act_selected    argmax-score sphere-plan candidate decoded via inverse head
  act_random      random sphere-plan candidate via inverse head
  act_zero        zero plan token via inverse head
  heuristic       env expert action every step    (reference ceiling)
  random_action   uniform random actions          (floor)

Every policy sees the same episode seeds (same layouts/goals). Metrics are
the env's real outcomes: success rate, collision rate, timeout rate, mean
return, mean steps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DREAMER4_ROOT = REPO_ROOT / "dreamer4" / "dreamer4"
for item in (str(DREAMER4_ROOT), str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT / "src")):
    if item not in sys.path:
        sys.path.insert(0, item)

import torch

from collect_game_action_dreamer4_dataset import resize_chw_uint8  # noqa: E402
from sensenova_drone.gym_drone_game import DroneGameConfig, DroneMazeEnv  # noqa: E402
from train_dynamics import (  # noqa: E402
    align_actions_to_frames,
    load_frozen_tokenizer_from_pt_ckpt,
    pack_bottleneck_to_spatial,
    temporal_patchify,
)
from train_latent_imagination_planner import (  # noqa: E402
    LatentImaginationPlanner,
    PlannerConfig,
    resolve_path,
    seed_everything,
)

NUM_ACTIONS = 9
POLICIES = ("act_mpc", "act_mpc_random", "act_selected", "act_random", "act_zero", "heuristic", "random_action")
BC_POLICIES = ("act_bc", "act_bc_think", "act_bc_random")
POLICY_POLICIES = ("act_policy", "act_policy_think")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Closed-loop drone-game act-by-imagination eval.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tokenizer-ckpt", default="")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--num-candidates", type=int, default=32)
    p.add_argument("--replan-every", type=int, default=4)
    p.add_argument("--policies", default=",".join(POLICIES))
    p.add_argument("--score-plan-mode", default="plan", choices=["plan", "zero"],
                   help="Plan input when scoring candidates. 'zero' = plan-free scoring (for checkpoints trained with score plan dropout).")
    p.add_argument("--bc-head", default="", help="BCChunkHead checkpoint (required for act_bc* policies).")
    p.add_argument("--bc-temperature", type=float, default=1.0)
    p.add_argument("--policy-head", default="", help="PMPO policy checkpoint (required for act_policy* policies).")
    p.add_argument("--use-builtin-bc", action="store_true", help="Use the planner's built-in BC head for act_bc* policies.")
    p.add_argument("--seed", type=int, default=20260710)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(resolve_path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = PlannerConfig(**ckpt["config"])
    tokenizer_ckpt = args.tokenizer_ckpt or cfg.tokenizer_ckpt
    encoder, _decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(resolve_path(tokenizer_ckpt)), device=device)
    patch = int(tok_args.get("patch", 8))
    packing_factor = int(tok_args.get("packing_factor", 2))
    n_spatial = int(tok_args.get("n_latents", 16)) // packing_factor
    z_dim = n_spatial * int(tok_args.get("d_bottleneck", 32)) * packing_factor

    model = LatentImaginationPlanner(
        z_dim=z_dim,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.hidden_dim,
        plan_dim=cfg.plan_dim,
        horizon=cfg.horizon,
        plan_unit_norm=getattr(cfg, "plan_unit_norm", False),
        plan_step_conditioning=getattr(cfg, "plan_step_conditioning", False),
    )
    if getattr(cfg, "bc_head_weight", 0.0) > 0:
        model.enable_bc_head()
    model = model.to(device)
    model.load_state_dict(ckpt["planner"], strict=True)
    model.eval()
    if cfg.action_dim != NUM_ACTIONS or cfg.action_features != "current":
        raise ValueError(f"Expected drone-game planner (action_dim=9, features=current); got {cfg.action_dim}, {cfg.action_features}")

    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    bc_head = None
    if any(p_ in BC_POLICIES for p_ in policies):
        if args.use_builtin_bc:
            if model.bc_head is None:
                raise ValueError("--use-builtin-bc requires a planner trained with --bc-head-weight")
            bc_head = model.bc_logits
        elif args.bc_head:
            from train_drone_bc_chunk_head import BCChunkHead
            hck = torch.load(resolve_path(args.bc_head), map_location="cpu", weights_only=False)
            head = BCChunkHead(**hck["config"]).to(device)
            head.load_state_dict(hck["head"])
            head.eval()
            bc_head = head
        else:
            raise ValueError("act_bc* policies require --bc-head or --use-builtin-bc")
    policy_head = None
    if any(p_ in POLICY_POLICIES for p_ in policies):
        if not args.policy_head:
            raise ValueError("act_policy* policies require --policy-head")
        from train_drone_bc_chunk_head import BCChunkHead
        pck = torch.load(resolve_path(args.policy_head), map_location="cpu", weights_only=False)
        policy_head = BCChunkHead(**pck["config"]).to(device)
        policy_head.load_state_dict(pck["head"])
        policy_head.eval()
    meta = {
        "phase": "closed_loop_drone_game",
        "ckpt": str(args.ckpt),
        "ckpt_step": int(ckpt.get("step", -1)),
        "episodes": args.episodes,
        "num_candidates": args.num_candidates,
        "replan_every": args.replan_every,
        "policies": policies,
        "seed": args.seed,
    }
    print(json.dumps(meta, indent=2), flush=True)
    (out_dir / "closed_loop_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    results = {}
    jsonl = (out_dir / "episodes.jsonl").open("w", encoding="utf-8")
    for policy in policies:
        started = time.time()
        rows = run_policy(
            policy=policy,
            model=model,
            encoder=encoder,
            patch=patch,
            n_spatial=n_spatial,
            packing_factor=packing_factor,
            cfg=cfg,
            args=args,
            device=device,
            jsonl=jsonl,
            bc_head=bc_head,
            policy_head=policy_head,
        )
        results[policy] = summarize(rows)
        results[policy]["elapsed_s"] = time.time() - started
        print(json.dumps({"phase": "policy_done", "policy": policy, **{k: v for k, v in results[policy].items() if k != 'elapsed_s'}}, indent=2), flush=True)
    jsonl.close()

    gates = compute_gates(results, out_dir=out_dir, seed=args.seed)
    payload = {"meta": meta, "per_policy": results, "gates": gates}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"phase": "closed_loop_complete", "gates": gates}, indent=2), flush=True)
    return 0


@torch.no_grad()
def run_policy(*, policy, model, encoder, patch, n_spatial, packing_factor, cfg, args, device, jsonl, bc_head=None, policy_head=None):
    env = DroneMazeEnv(DroneGameConfig(max_episode_steps=args.max_steps))
    ctx = int(cfg.ctx_len)
    h = int(cfg.horizon)
    K = int(args.num_candidates)
    rng = np.random.default_rng(args.seed + 17)
    rows = []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)  # matched seeds across policies
        frame = resize_chw_uint8(env.render(), 128)
        frames = [frame] * ctx  # pad history with the first frame
        act_hist = [np.zeros(NUM_ACTIONS, dtype=np.float32)] * ctx
        ep_return, steps, terminal_reward = 0.0, 0, 0.0
        pending: list[int] = []
        done = False
        replans = 0
        while not done and steps < args.max_steps:
            if policy == "heuristic":
                action = int(env.expert_action_index())
            elif policy == "random_action":
                action = int(rng.integers(0, NUM_ACTIONS))
            else:
                if not pending:
                    pending = plan_chunk(
                        policy=policy, model=model, encoder=encoder, patch=patch,
                        n_spatial=n_spatial, packing_factor=packing_factor, cfg=cfg,
                        frames=frames, act_hist=act_hist, K=K, device=device, rng=rng,
                        score_plan_mode=args.score_plan_mode,
                        bc_head=bc_head, bc_temperature=args.bc_temperature,
                        policy_head=policy_head,
                    )[: args.replan_every]
                    replans += 1
                action = pending.pop(0)
            _obs, reward, terminated, truncated, _info = env.step(action)
            ep_return += float(reward)
            terminal_reward = float(reward)
            steps += 1
            frame = resize_chw_uint8(env.render(), 128)
            frames = frames[1:] + [frame]
            act_hist = act_hist[1:] + [one_hot_np(action)]
            done = terminated or truncated
        outcome = "timeout"
        if done and terminal_reward > 5.0:
            outcome = "success"
        elif done and terminal_reward < -4.0:
            outcome = "collision_or_oob"
        row = {"policy": policy, "episode": ep, "seed": args.seed + ep, "return": ep_return,
               "steps": steps, "outcome": outcome, "replans": replans}
        rows.append(row)
        jsonl.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def one_hot_np(index: int) -> np.ndarray:
    v = np.zeros(NUM_ACTIONS, dtype=np.float32)
    v[int(index)] = 1.0
    return v


@torch.no_grad()
def plan_chunk(*, policy, model, encoder, patch, n_spatial, packing_factor, cfg, frames, act_hist, K, device, rng, score_plan_mode="plan", bc_head=None, bc_temperature=1.0, policy_head=None):
    ctx = int(cfg.ctx_len)
    h = int(cfg.horizon)
    obs = torch.from_numpy(np.stack(frames)).to(device)[None].float() / 255.0  # (1, ctx, C, H, W)
    patches = temporal_patchify(obs, patch)
    z, _ = encoder(patches)
    ctx_z = pack_bottleneck_to_spatial(z, n_spatial=n_spatial, k=packing_factor).flatten(2)
    act = torch.from_numpy(np.stack(act_hist)).to(device)[None].float()  # (1, ctx, A)
    mask = torch.ones_like(act)
    actions, amask = align_actions_to_frames(act, mask, frame_count=ctx, action_frame_offset=cfg.action_frame_offset)
    ctx_h = model.encode_context(ctx_z, actions.clamp(-1, 1) * amask)

    if policy == "act_policy":
        return [int(i) for i in policy_head(ctx_h)[0].argmax(dim=-1).tolist()]
    if policy == "act_policy_think":
        logits = policy_head(ctx_h)[0]
        probs = torch.softmax(logits, dim=-1)
        samples = torch.multinomial(probs, K - 1, replacement=True).T
        chunks = [[int(i) for i in logits.argmax(dim=-1).tolist()]]
        chunks += [[int(i) for i in row.tolist()] for row in samples]
        chunk_onehot = torch.zeros((len(chunks), h, NUM_ACTIONS), device=device)
        for i, chunk in enumerate(chunks):
            for t, a in enumerate(chunk):
                chunk_onehot[i, t, a] = 1.0
        chr_ = ctx_h.expand(len(chunks), ctx_h.shape[-1])
        plans = model.encode_plan(chr_, chunk_onehot)
        futures = model.propose_future(ctx_z.expand(len(chunks), *ctx_z.shape[1:]), chr_, plans, horizon=h)
        score_plans = torch.zeros_like(plans) if score_plan_mode == "zero" else plans
        pick = int(model.score_future(chr_, futures, score_plans).argmax())
        return list(chunks[pick])
    if policy in ("act_bc", "act_bc_think", "act_bc_random"):
        # BC-anchored thinking: candidates sampled from the behavior prior's
        # support; imagination + value only choose AMONG plausible behavior.
        logits = bc_head(ctx_h)[0] / max(1e-6, bc_temperature)  # (h, A)
        if policy == "act_bc":
            return [int(i) for i in logits.argmax(dim=-1).tolist()]
        probs = torch.softmax(logits, dim=-1)
        samples = torch.multinomial(probs, K - 1, replacement=True).T  # (K-1, h)
        chunks = [ [int(i) for i in logits.argmax(dim=-1).tolist()] ]  # argmax always candidate 0
        chunks += [[int(i) for i in row.tolist()] for row in samples]
        chunk_onehot = torch.zeros((len(chunks), h, NUM_ACTIONS), device=device)
        for i, chunk in enumerate(chunks):
            for t, a in enumerate(chunk):
                chunk_onehot[i, t, a] = 1.0
        chr_ = ctx_h.expand(len(chunks), ctx_h.shape[-1])
        plans = model.encode_plan(chr_, chunk_onehot)
        czr = ctx_z.expand(len(chunks), *ctx_z.shape[1:])
        futures = model.propose_future(czr, chr_, plans, horizon=h)
        if policy == "act_bc_think":
            score_plans = torch.zeros_like(plans) if score_plan_mode == "zero" else plans
            scores = model.score_future(chr_, futures, score_plans)
            pick = int(scores.argmax())
        else:
            pick = int(rng.integers(0, len(chunks)))
        return list(chunks[pick])

    if policy in ("act_mpc", "act_mpc_random"):
        # action-native MPC: candidates are concrete action chunks; the plan
        # encoder is the trained action->plan interface, and the winning
        # chunk's own actions are executed (no inverse head).
        chunks = candidate_action_chunks(h, K, rng)  # (K, h) ints
        chunk_onehot = torch.zeros((len(chunks), h, NUM_ACTIONS), device=device)
        for i, chunk in enumerate(chunks):
            for t, a in enumerate(chunk):
                chunk_onehot[i, t, a] = 1.0
        chr_ = ctx_h.expand(len(chunks), ctx_h.shape[-1])
        plans = model.encode_plan(chr_, chunk_onehot)
        czr = ctx_z.expand(len(chunks), *ctx_z.shape[1:])
        futures = model.propose_future(czr, chr_, plans, horizon=h)
        if policy == "act_mpc":
            score_plans = torch.zeros_like(plans) if score_plan_mode == "zero" else plans
            scores = model.score_future(chr_, futures, score_plans)
            pick = int(scores.argmax())
        else:
            pick = int(rng.integers(0, len(chunks)))
        return list(chunks[pick])

    if policy == "act_zero":
        plans = torch.zeros((1, model.plan_dim), device=device)
    else:
        plans = model.normalize_plan(torch.from_numpy(rng.standard_normal((K, model.plan_dim)).astype(np.float32)).to(device))
    reps = plans.shape[0]
    czr = ctx_z.expand(reps, *ctx_z.shape[1:])
    chr_ = ctx_h.expand(reps, ctx_h.shape[-1])
    futures = model.propose_future(czr, chr_, plans, horizon=h)
    if policy == "act_selected":
        scores = model.score_future(chr_, futures, plans)
        pick = int(scores.argmax())
    elif policy == "act_random":
        pick = int(rng.integers(0, reps))
    else:
        pick = 0
    acts = model.inverse_actions(chr_[pick : pick + 1], futures[pick : pick + 1], torch.zeros_like(plans[pick : pick + 1]))
    return [int(i) for i in acts[0].argmax(dim=-1).tolist()]


def candidate_action_chunks(h: int, K: int, rng) -> list[list[int]]:
    """9 pure single-action blocks + two-phase blocks to fill K candidates."""
    chunks = [[a] * h for a in range(NUM_ACTIONS)]
    while len(chunks) < K:
        a, b = int(rng.integers(0, NUM_ACTIONS)), int(rng.integers(0, NUM_ACTIONS))
        split = int(rng.integers(1, h))
        chunks.append([a] * split + [b] * (h - split))
    return chunks[:K]


def summarize(rows):
    n = len(rows)
    out = {"episodes": n}
    for key in ("success", "collision_or_oob", "timeout"):
        out[f"{key}_rate"] = sum(1 for r in rows if r["outcome"] == key) / max(1, n)
    out["mean_return"] = float(np.mean([r["return"] for r in rows]))
    out["mean_steps"] = float(np.mean([r["steps"] for r in rows]))
    return out


def compute_gates(results, *, out_dir, seed):
    gates = {}
    rows = [json.loads(l) for l in (out_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["policy"], {})[r["episode"]] = r
    rng = np.random.default_rng(seed)

    def paired_delta(a, b, key):  # a-b on matched episodes, bootstrap CI
        eps = sorted(set(by.get(a, {})) & set(by.get(b, {})))
        if not eps:
            return None
        va = np.array([by[a][e][key] if key != "success" else float(by[a][e]["outcome"] == "success") for e in eps])
        vb = np.array([by[b][e][key] if key != "success" else float(by[b][e]["outcome"] == "success") for e in eps])
        d = va - vb
        idx = rng.integers(0, len(d), size=(1000, len(d)))
        means = d[idx].mean(axis=1)
        return {"mean": float(d.mean()), "ci_lo": float(np.percentile(means, 2.5)), "ci_hi": float(np.percentile(means, 97.5))}

    for main, base in (("act_policy", "act_bc"), ("act_policy_think", "act_policy"),
                       ("act_policy_think", "act_bc"),
                       ("act_bc_think", "act_bc"), ("act_bc_think", "act_bc_random"),
                       ("act_mpc", "act_mpc_random"), ("act_mpc", "random_action"),
                       ("act_selected", "act_random"), ("act_selected", "act_zero"),
                       ("act_selected", "random_action")):
        for key in ("success", "return"):
            delta = paired_delta(main, base, key)
            if delta is not None:
                gates[f"{main}_vs_{base}_{key}"] = delta
    gates["mpc_thinking_wins"] = bool(
        gates.get("act_mpc_vs_act_mpc_random_return", {}).get("ci_lo", -1) > 0
        and gates.get("act_mpc_vs_act_mpc_random_success", {}).get("ci_lo", -1) > 0
    )
    gates["bc_thinking_wins"] = bool(
        gates.get("act_bc_think_vs_act_bc_success", {}).get("ci_lo", -1) > 0
        and gates.get("act_bc_think_vs_act_bc_random_success", {}).get("ci_lo", -1) > 0
    )
    return gates


if __name__ == "__main__":
    raise SystemExit(main())
