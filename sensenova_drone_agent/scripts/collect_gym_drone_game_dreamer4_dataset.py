#!/usr/bin/env python3
"""Collect gym drone game episodes into the native WMDataset layout.

Domain data for the closed-loop act-by-imagination experiment
(docs/ACT_BY_IMAGINATION_HARNESS.md). Mixed behavior policies give both
action identifiability and reward variance:

  expert       env.expert_action_index() every step
  eps_expert   expert with epsilon-random actions
  blocks       one random action repeated for a block of steps

Actions are one-hot over the 9 discrete drone actions; row t+1 stores the
action/reward for the transition frame t -> t+1 (WMDataset alignment, zero
first row per episode) - the same convention as
collect_game_action_dreamer4_dataset.py, whose TaskWriter this reuses.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for item in (str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT / "src")):
    if item not in sys.path:
        sys.path.insert(0, item)

from collect_game_action_dreamer4_dataset import TaskWriter, one_hot, validate_wm_dataset  # noqa: E402
from sensenova_drone.gym_drone_game import DroneGameConfig, DroneMazeEnv  # noqa: E402

NUM_ACTIONS = 9
ACTION_LABELS = ["hover", "yaw_left", "yaw_right", "ascend", "descend", "forward", "backward", "strafe_left", "strafe_right"]
POLICIES = ("expert", "eps_expert", "blocks")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect drone game data in WMDataset layout.")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=600, help="Total episodes across all policies.")
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--frame-size", type=int, default=128)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--epsilon", type=float, default=0.25)
    p.add_argument("--block-min", type=int, default=3)
    p.add_argument("--block-max", type=int, default=8)
    p.add_argument("--policies", default=",".join(POLICIES),
                   help="Comma list of behavior policies to cycle through (expert, eps_expert, blocks).")
    p.add_argument("--pad-terminal", type=int, default=0,
                   help="Absorbing-state padding: append N rows after a terminal step (repeated final frame, hover action, 0 reward) so terminal transitions appear inside scored window futures.")
    p.add_argument("--seed", type=int, default=20260709)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--validate-seq-len", type=int, default=24)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT.parent / out_dir).resolve()
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} exists; pass --overwrite")
    raw_dir = out_dir / "raw"
    frames_dir = out_dir / "frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    env = DroneMazeEnv(DroneGameConfig(max_episode_steps=args.max_steps))
    writer = TaskWriter(
        task_name="drone_maze",
        text="navigate the drone to the goal without hitting trees",
        raw_dir=raw_dir,
        frames_dir=frames_dir,
        frame_size=int(args.frame_size),
        shard_size=int(args.shard_size),
        action_dim=NUM_ACTIONS,
        action_labels=ACTION_LABELS,
        source="gym_drone_game",
        metadata={"policies": list(POLICIES), "epsilon": args.epsilon},
    )

    policies = [p_ for p_ in args.policies.split(",") if p_.strip() in POLICIES]
    if not policies:
        raise SystemExit(f"no valid policies in {args.policies!r}")
    policy_counts = {k: 0 for k in policies}
    outcome_counts = {"success": 0, "collision_or_oob": 0, "timeout": 0}
    for ep in range(args.episodes):
        policy = policies[ep % len(policies)]
        policy_counts[policy] += 1
        _obs, _info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        writer.append(episode=ep, frame_rgb=env.render(), action=np.zeros(NUM_ACTIONS, dtype=np.float32), reward=0.0)
        block_action, block_left = 0, 0
        ep_return, length, terminated = 0.0, 0, False
        for _t in range(args.max_steps):
            if policy == "expert":
                action = int(env.expert_action_index())
            elif policy == "eps_expert":
                action = int(rng.integers(0, NUM_ACTIONS)) if rng.random() < args.epsilon else int(env.expert_action_index())
            else:
                if block_left <= 0:
                    block_action = int(rng.integers(0, NUM_ACTIONS))
                    block_left = int(rng.integers(args.block_min, args.block_max + 1))
                action = block_action
                block_left -= 1
            _obs, reward, terminated, truncated, info = env.step(action)
            frame_rgb = env.render()
            writer.append(episode=ep, frame_rgb=frame_rgb, action=one_hot(action, NUM_ACTIONS), reward=float(reward))
            ep_return += float(reward)
            length += 1
            if terminated or truncated:
                for _pad in range(args.pad_terminal):
                    writer.append(episode=ep, frame_rgb=frame_rgb, action=one_hot(0, NUM_ACTIONS), reward=0.0)
                break
        writer.record_episode(episode_return=ep_return, length=length)
        if terminated and ep_return > 0:
            outcome_counts["success"] += 1
        elif terminated:
            outcome_counts["collision_or_oob"] += 1
        else:
            outcome_counts["timeout"] += 1
        if (ep + 1) % 50 == 0:
            print(json.dumps({"phase": "collect", "episodes": ep + 1, "frames": writer.total_frames}), flush=True)

    task_summary = writer.finalize()
    preview = writer.write_preview(out_dir)
    task_meta = {
        writer.task_name: {
            "action_dim": int(writer.action_dim),
            "text": writer.text,
            "source": writer.source,
            "action_labels": writer.action_labels,
            **writer.metadata,
        }
    }
    tasks_json = out_dir / "tasks.json"
    tasks_json.write_text(json.dumps(task_meta, indent=2, sort_keys=True), encoding="utf-8")

    validation = None
    if args.validate:
        validation = validate_wm_dataset(
            raw_dir=raw_dir,
            frames_dir=frames_dir,
            tasks_json=tasks_json,
            seq_len=int(args.validate_seq_len),
            frame_size=int(args.frame_size),
            action_dim=NUM_ACTIONS,
            shard_size=int(args.shard_size),
        )

    summary = {
        "phase": "gym_drone_game_dreamer4_collection",
        "created_unix_s": started,
        "completed_unix_s": time.time(),
        "out": str(out_dir),
        "episodes": args.episodes,
        "policy_counts": policy_counts,
        "outcome_counts": outcome_counts,
        "action_dim": NUM_ACTIONS,
        "frame_size": args.frame_size,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "task": task_summary,
        "preview": preview,
        "validation": validation,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("task",)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
