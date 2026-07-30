#!/usr/bin/env python3
"""Collect TRUE counterfactual branch data from the gym drone game.

The contrastive "thinking data" the original handoff called for: from the
SAME state, every discrete action is rolled out for a full horizon, so
"same context + action A -> survives" and "same context + action B ->
crashes" become directly supervisable. Branch states are concentrated where
decisions bind (front clearance below a threshold) plus a background rate.

Each branch is written as its own WMDataset episode:

  rows 0..8    trunk history (9 frames: 8-frame context + branch state)
  rows 9..16   branch future (action a repeated; frozen-frame absorbing
               padding with hover/0-reward after a terminal)

With seq_len 16 (ctx 8 + horizon 8) each branch episode yields exactly one
window whose context is the trunk and whose scored future is the branch.
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
CTX_ROWS = 9  # 8-frame context + the branch-point frame
HORIZON = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect counterfactual branch data from the drone game.")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=80, help="Trunk episodes.")
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--frame-size", type=int, default=128)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--epsilon", type=float, default=0.25)
    p.add_argument("--front-threshold-m", type=float, default=3.0, help="Branch whenever front clearance drops below this.")
    p.add_argument("--background-every", type=int, default=10, help="Also branch every Nth step regardless of clearance.")
    p.add_argument("--max-branch-states", type=int, default=6, help="Branch states per trunk episode.")
    p.add_argument("--seed", type=int, default=20260711)
    p.add_argument("--validate", action="store_true")
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
    env = DroneMazeEnv(DroneGameConfig(max_episode_steps=args.max_steps + HORIZON + 2))
    writer = TaskWriter(
        task_name="drone_maze_branch",
        text="counterfactual action branches near obstacles in the drone maze",
        raw_dir=raw_dir,
        frames_dir=frames_dir,
        frame_size=int(args.frame_size),
        shard_size=int(args.shard_size),
        action_dim=NUM_ACTIONS,
        action_labels=ACTION_LABELS,
        source="gym_drone_game_branch",
        metadata={"front_threshold_m": args.front_threshold_m, "horizon": HORIZON},
    )

    branch_episode_id = 0
    branch_state_count = {"blocked": 0, "background": 0}
    outcome_count = {"terminated": 0, "alive": 0}
    for ep in range(args.episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        # rolling trunk history: (frame_rgb, action_onehot, reward)
        hist = [(env.render(), np.zeros(NUM_ACTIONS, dtype=np.float32), 0.0)]
        branches_done = 0
        for t in range(args.max_steps):
            front_m = float(env._compute_clearances()["front_m"])
            blocked = front_m < args.front_threshold_m
            background = args.background_every > 0 and t % args.background_every == 0
            if len(hist) >= CTX_ROWS and branches_done < args.max_branch_states and (blocked or background):
                branch_state_count["blocked" if blocked else "background"] += 1
                branches_done += 1
                snap = env.snapshot()
                trunk = hist[-CTX_ROWS:]
                for a in range(NUM_ACTIONS):
                    env.restore(snap)
                    for frame_rgb, act_vec, rew in trunk:
                        writer.append(episode=branch_episode_id, frame_rgb=frame_rgb, action=act_vec, reward=rew)
                    frozen, done_b = None, False
                    for _k in range(HORIZON):
                        if done_b:
                            writer.append(episode=branch_episode_id, frame_rgb=frozen, action=one_hot(0, NUM_ACTIONS), reward=0.0)
                            continue
                        _o, r_b, term_b, trunc_b, _i = env.step(a)
                        frozen = env.render()
                        writer.append(episode=branch_episode_id, frame_rgb=frozen, action=one_hot(a, NUM_ACTIONS), reward=float(r_b))
                        if term_b or trunc_b:
                            done_b = True
                    outcome_count["terminated" if done_b else "alive"] += 1
                    branch_episode_id += 1
                env.restore(snap)
            # trunk behavior: eps-expert
            action = int(rng.integers(0, NUM_ACTIONS)) if rng.random() < args.epsilon else int(env.expert_action_index())
            _o, reward, terminated, truncated, _i = env.step(action)
            hist.append((env.render(), one_hot(action, NUM_ACTIONS), float(reward)))
            if len(hist) > CTX_ROWS + 2:
                hist.pop(0)
            if terminated or truncated:
                break
        writer.record_episode(episode_return=0.0, length=branches_done)
        if (ep + 1) % 10 == 0:
            print(json.dumps({"phase": "collect", "trunks": ep + 1, "branch_episodes": branch_episode_id, "frames": writer.total_frames}), flush=True)

    task_summary = writer.finalize()
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
            raw_dir=raw_dir, frames_dir=frames_dir, tasks_json=tasks_json,
            seq_len=CTX_ROWS + HORIZON - 1, frame_size=int(args.frame_size),
            action_dim=NUM_ACTIONS, shard_size=int(args.shard_size),
        )
    summary = {
        "phase": "gym_drone_game_branch_collection",
        "created_unix_s": started,
        "completed_unix_s": time.time(),
        "out": str(out_dir),
        "trunk_episodes": args.episodes,
        "branch_episodes": branch_episode_id,
        "branch_state_count": branch_state_count,
        "branch_outcomes": outcome_count,
        "task": task_summary,
        "validation": validation,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "task"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
