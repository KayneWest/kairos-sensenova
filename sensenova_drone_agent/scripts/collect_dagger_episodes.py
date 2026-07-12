#!/usr/bin/env python3
"""DAgger collection: run the act_bc_think agent and record ITS episodes in
WMDataset layout, so the agent's own visited states enter the training
distribution (the identified root cause of act-time seed fragility)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for item in (str(REPO_ROOT / "dreamer4" / "dreamer4"), str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT / "src")):
    if item not in sys.path:
        sys.path.insert(0, item)

import torch

from collect_game_action_dreamer4_dataset import TaskWriter, one_hot
from sensenova_drone.gym_drone_game import DroneGameConfig, DroneMazeEnv
from train_dynamics import load_frozen_tokenizer_from_pt_ckpt
from train_drone_bc_chunk_head import BCChunkHead
from train_latent_imagination_planner import LatentImaginationPlanner, PlannerConfig, resolve_path, seed_everything
import eval_gym_drone_game_act_by_imagination as ev

NUM_ACTIONS = 9
ACTION_LABELS = ["hover", "yaw_left", "yaw_right", "ascend", "descend", "forward", "backward", "strafe_left", "strafe_right"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--bc-head", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--pad-terminal", type=int, default=12)
    p.add_argument("--num-candidates", type=int, default=32)
    p.add_argument("--replan-every", type=int, default=4)
    p.add_argument("--bc-temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=20270100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = resolve_path(args.out)
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} exists")
    raw_dir, frames_dir = out_dir / "raw", out_dir / "frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(resolve_path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = PlannerConfig(**ckpt["config"])
    encoder, _d, tok = load_frozen_tokenizer_from_pt_ckpt(str(resolve_path(cfg.tokenizer_ckpt)), device=device)
    patch, pk = int(tok.get("patch", 8)), int(tok.get("packing_factor", 2))
    ns = int(tok.get("n_latents", 16)) // pk
    z_dim = ns * int(tok.get("d_bottleneck", 32)) * pk
    model = LatentImaginationPlanner(z_dim=z_dim, action_dim=cfg.action_dim, hidden_dim=cfg.hidden_dim,
                                     plan_dim=cfg.plan_dim, horizon=cfg.horizon, plan_unit_norm=cfg.plan_unit_norm,
                                     plan_step_conditioning=getattr(cfg, "plan_step_conditioning", False))
    if getattr(cfg, "bc_head_weight", 0.0) > 0:
        model.enable_bc_head()
    model = model.to(device)
    model.load_state_dict(ckpt["planner"], strict=True)
    model.eval()
    hck = torch.load(resolve_path(args.bc_head), map_location="cpu", weights_only=False)
    bc_head = BCChunkHead(**hck["config"]).to(device)
    bc_head.load_state_dict(hck["head"])
    bc_head.eval()

    env = DroneMazeEnv(DroneGameConfig(max_episode_steps=args.max_steps))
    writer = TaskWriter(task_name="drone_maze_dagger", text="dagger episodes from the act_bc_think agent",
                        raw_dir=raw_dir, frames_dir=frames_dir, frame_size=cfg.img_size, shard_size=2048,
                        action_dim=NUM_ACTIONS, action_labels=ACTION_LABELS, source="gym_drone_game_dagger",
                        metadata={"agent": "act_bc_think", "planner": str(args.ckpt)})
    rng = np.random.default_rng(args.seed)
    outcomes = {"success": 0, "collision_or_oob": 0, "timeout": 0}
    with torch.no_grad():
        for ep in range(args.episodes):
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            frame = ev.resize_chw_uint8(env.render(), cfg.img_size)
            frames = [frame] * cfg.ctx_len
            act_hist = [np.zeros(NUM_ACTIONS, dtype=np.float32)] * cfg.ctx_len
            writer.append(episode=ep, frame_rgb=env.render(), action=np.zeros(NUM_ACTIONS, dtype=np.float32), reward=0.0)
            pending, done, term_r, steps = [], False, 0.0, 0
            while not done and steps < args.max_steps:
                if not pending:
                    pending = ev.plan_chunk(policy="act_bc_think", model=model, encoder=encoder, patch=patch,
                                            n_spatial=ns, packing_factor=pk, cfg=cfg, frames=frames,
                                            act_hist=act_hist, K=args.num_candidates, device=device, rng=rng,
                                            score_plan_mode="zero", bc_head=bc_head,
                                            bc_temperature=args.bc_temperature)[: args.replan_every]
                a = pending.pop(0)
                _o, r, t1, t2, _i = env.step(a)
                fr = env.render()
                writer.append(episode=ep, frame_rgb=fr, action=one_hot(a, NUM_ACTIONS), reward=float(r))
                frame = ev.resize_chw_uint8(fr, cfg.img_size)
                frames = frames[1:] + [frame]
                act_hist = act_hist[1:] + [one_hot(a, NUM_ACTIONS)]
                term_r, steps, done = float(r), steps + 1, (t1 or t2)
                if done:
                    for _ in range(args.pad_terminal):
                        writer.append(episode=ep, frame_rgb=fr, action=one_hot(0, NUM_ACTIONS), reward=0.0)
            outcomes["success" if (done and term_r > 5) else ("collision_or_oob" if done else "timeout")] += 1
            writer.record_episode(episode_return=0.0, length=steps)
            if (ep + 1) % 50 == 0:
                print(json.dumps({"ep": ep + 1, "frames": writer.total_frames, **outcomes}), flush=True)
    info = writer.finalize()
    (out_dir / "tasks.json").write_text(json.dumps({writer.task_name: {"action_dim": NUM_ACTIONS, "text": writer.text,
        "source": writer.source, "action_labels": ACTION_LABELS}}, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"episodes": args.episodes, "outcomes": outcomes, "task": info}, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "outcomes": outcomes, "frames": info["frames"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
