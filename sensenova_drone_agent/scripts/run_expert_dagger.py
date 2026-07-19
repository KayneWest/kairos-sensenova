#!/usr/bin/env python3
"""TRUE DAgger: expert corrections on the agent's visited states, iterated.

The campaign's "DAgger" was self-imitation (outcome-labeled own episodes) —
it repaired coverage once, then poisoned the imagination. This is the real
thing: roll out the think-then-act agent, and at EVERY visited state record
what the scripted expert would do for the next horizon (env snapshot ->
expert rollout -> restore). The healthy cycle-1 imagination is HELD FIXED;
only the BC candidate head retrains each round on the aggregated
expert-labeled data. If corrective data is the missing capacity, success
ladders round over round toward the 41.5% expert ceiling.

Round 0 base data: episodes where the expert itself acts (its states, its
labels). Rounds 1..R: the current think-agent acts; expert labels only.
Heads are trained on cached ctx_h (frozen encoder) — minutes per round.
"""
from __future__ import annotations

import argparse
import json
import subprocess
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
from train_latent_diffusion_proposer import load_planner  # noqa: E402
from train_latent_imagination_planner import resolve_path, seed_everything  # noqa: E402
import eval_gym_drone_game_act_by_imagination as ev  # noqa: E402

NUM_ACTIONS = 9


@torch.no_grad()
def encode_ctx_h(planner, encoder, patch, ns, pk, cfg, frames, act_hist, device):
    obs = torch.from_numpy(np.stack(frames)).to(device)[None].float() / 255.0
    patches = temporal_patchify(obs, patch)
    z, _ = encoder(patches)
    ctx_z = pack_bottleneck_to_spatial(z, n_spatial=ns, k=pk).flatten(2)
    act = torch.from_numpy(np.stack(act_hist)).to(device)[None].float()
    mask = torch.ones_like(act)
    actions, amask = align_actions_to_frames(act, mask, frame_count=int(cfg.ctx_len),
                                             action_frame_offset=cfg.action_frame_offset)
    return planner.encode_context(ctx_z, actions.clamp(-1, 1) * amask)


def expert_chunk(env, horizon: int) -> list[int]:
    """Expert's next-h actions from the current state (snapshot/restore)."""
    snap = env.snapshot()
    chunk = []
    for _ in range(horizon):
        a = int(env.expert_action_index())
        chunk.append(a)
        _o, _r, t1, t2, _i = env.step(a)
        if t1 or t2:
            break
    while len(chunk) < horizon:
        chunk.append(0)  # absorbing convention: hover
    env.restore(snap)
    return chunk


def collect_round(*, planner, encoder, patch, ns, pk, cfg, bc_head, episodes, max_steps,
                  seed, device, num_candidates, replan_every, bc_temperature, expert_acts):
    env = DroneMazeEnv(DroneGameConfig(max_episode_steps=max_steps))
    rng = np.random.default_rng(seed)
    h = int(cfg.horizon)
    xs, ys = [], []
    outcomes = {"success": 0, "collision_or_oob": 0, "timeout": 0}
    for ep in range(episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        frame = ev.resize_chw_uint8(env.render(), int(cfg.img_size))
        frames = [frame] * int(cfg.ctx_len)
        act_hist = [np.zeros(NUM_ACTIONS, dtype=np.float32)] * int(cfg.ctx_len)
        pending, done, term_r, steps = [], False, 0.0, 0
        while not done and steps < max_steps:
            ctx_h = encode_ctx_h(planner, encoder, patch, ns, pk, cfg, frames, act_hist, device)
            xs.append(ctx_h[0].cpu().numpy().astype(np.float32))
            ys.append(np.array(expert_chunk(env, h), dtype=np.int64))
            if expert_acts:
                a = int(env.expert_action_index())
            else:
                if not pending:
                    pending = ev.plan_chunk(policy="act_bc_think", model=planner, encoder=encoder,
                                            patch=patch, n_spatial=ns, packing_factor=pk, cfg=cfg,
                                            frames=frames, act_hist=act_hist, K=num_candidates,
                                            device=device, rng=rng, score_plan_mode="zero",
                                            bc_head=bc_head, bc_temperature=bc_temperature)[:replan_every]
                a = pending.pop(0)
            _o, r, t1, t2, _i = env.step(a)
            frame = ev.resize_chw_uint8(env.render(), int(cfg.img_size))
            frames = frames[1:] + [frame]
            v = np.zeros(NUM_ACTIONS, dtype=np.float32)
            v[a] = 1.0
            act_hist = act_hist[1:] + [v]
            term_r, steps, done = float(r), steps + 1, (t1 or t2)
        outcomes["success" if (done and term_r > 5.0) else ("collision_or_oob" if done else "timeout")] += 1
    return np.stack(xs), np.stack(ys), outcomes


def train_head(xs: np.ndarray, ys: np.ndarray, *, ctx_dim, horizon, steps, seed, device):
    torch.manual_seed(seed)
    head = BCChunkHead(ctx_dim=ctx_dim, horizon=horizon, hidden_dim=1024).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-5)
    x_t = torch.from_numpy(xs).to(device)
    y_t = torch.from_numpy(ys).to(device)
    n = x_t.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    accs = []
    for step in range(steps):
        idx = torch.randint(0, n, (256,), generator=gen).to(device)
        logits = head(x_t[idx])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, NUM_ACTIONS), y_t[idx].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y_t[idx]).float().mean().item()
            accs.append(acc)
    head.eval()
    return head, accs[-1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--planner-ckpt", required=True)
    p.add_argument("--init-bc-head", required=True)
    p.add_argument("--out-prefix", required=True, help="e.g. expert_dagger_s1")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--expert-episodes", type=int, default=200)
    p.add_argument("--episodes-per-round", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--num-candidates", type=int, default=32)
    p.add_argument("--replan-every", type=int, default=4)
    p.add_argument("--bc-temperature", type=float, default=0.8)
    p.add_argument("--head-train-steps", type=int, default=8000)
    p.add_argument("--eval-episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260710)
    p.add_argument("--eval-seed", type=int, default=20260710)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = PROJECT_ROOT / "output"

    planner, cfg, encoder, patch, pk, ns, _z = load_planner(args.planner_ckpt, device)
    hck = torch.load(resolve_path(args.init_bc_head), map_location="cpu", weights_only=False)
    bc_head = BCChunkHead(**hck["config"]).to(device)
    bc_head.load_state_dict(hck["head"])
    bc_head.eval()

    all_x, all_y = [], []

    def log(obj):
        print(json.dumps(obj), flush=True)

    # Round 0: expert acts on its own states (base aggregate).
    xs, ys, outc = collect_round(planner=planner, encoder=encoder, patch=patch, ns=ns, pk=pk, cfg=cfg,
                                 bc_head=bc_head, episodes=args.expert_episodes, max_steps=args.max_steps,
                                 seed=args.seed, device=device, num_candidates=args.num_candidates,
                                 replan_every=args.replan_every, bc_temperature=args.bc_temperature,
                                 expert_acts=True)
    all_x.append(xs)
    all_y.append(ys)
    log({"round": 0, "mode": "expert", "states": int(xs.shape[0]), "outcomes": outc})

    for rnd in range(1, args.rounds + 1):
        # Collect with the CURRENT agent, expert labels on its visited states.
        xs, ys, outc = collect_round(planner=planner, encoder=encoder, patch=patch, ns=ns, pk=pk, cfg=cfg,
                                     bc_head=bc_head, episodes=args.episodes_per_round,
                                     max_steps=args.max_steps, seed=args.seed + rnd * 1000, device=device,
                                     num_candidates=args.num_candidates, replan_every=args.replan_every,
                                     bc_temperature=args.bc_temperature, expert_acts=False)
        all_x.append(xs)
        all_y.append(ys)
        log({"round": rnd, "mode": "agent_collect", "states": int(xs.shape[0]), "outcomes": outc})

        X = np.concatenate(all_x)
        Y = np.concatenate(all_y)
        np.savez_compressed(out_dir / f"{args.out_prefix}_r{rnd}_data.npz", x_last=xs, y_last=ys)
        bc_head, acc = train_head(X, Y, ctx_dim=cfg.hidden_dim, horizon=int(cfg.horizon),
                                  steps=args.head_train_steps, seed=args.seed + rnd, device=device)
        head_path = out_dir / f"{args.out_prefix}_head_r{rnd}.pt"
        torch.save({"head": bc_head.state_dict(),
                    "config": {"ctx_dim": cfg.hidden_dim, "horizon": int(cfg.horizon), "hidden_dim": 1024},
                    "planner_ckpt": str(args.planner_ckpt), "round": rnd}, head_path)
        log({"round": rnd, "mode": "train", "aggregate_states": int(X.shape[0]), "train_acc": round(acc, 4)})

        eval_out = out_dir / f"closed_loop_drone_game_v20_{args.out_prefix}_r{rnd}"
        r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "eval_gym_drone_game_diffusion_think.py"),
                            "--planner-ckpt", args.planner_ckpt, "--judge-ckpt", args.planner_ckpt,
                            "--bc-head", str(head_path),
                            "--diffusion-ckpt", str(PROJECT_ROOT / "output" / "latent_diffusion_proposer_v2" / "final.pt"),
                            "--out-dir", str(eval_out), "--episodes", str(args.eval_episodes),
                            "--seed", str(args.eval_seed), "--policies", "bc,bc_random,gru_argmax"])
        if r.returncode != 0:
            log({"round": rnd, "mode": "eval", "error": r.returncode})
            return 1
        summ = json.loads((eval_out / "summary.json").read_text())
        log({"round": rnd, "mode": "eval",
             "success": {k: summ["per_policy"][k]["success_rate"] for k in summ["per_policy"]},
             "gru_argmax_wins": summ["gates"].get("gru_argmax_wins")})

    log({"phase": "done", "rounds": args.rounds})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
