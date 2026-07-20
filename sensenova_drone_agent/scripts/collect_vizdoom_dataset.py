#!/usr/bin/env python3
"""Base dataset for the ViZDoom domain: oracle + epsilon-noise policies,
WMDataset layout with absorbing-state padding (RIGHT_DATA_SPEC compliant:
teacher exists, terminals in-window, success density via oracle mix)."""
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for item in (str(REPO_ROOT / "dreamer4" / "dreamer4"), str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT / "src")):
    sys.path.insert(0, item)
import numpy as np
from collect_game_action_dreamer4_dataset import TaskWriter, one_hot
from sensenova_drone.vizdoom_game import VizdoomCorridorEnv, VizdoomGameConfig, NUM_ACTIONS, ACTION_LABELS
from train_latent_imagination_planner import resolve_path

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--episodes", type=int, default=600)
p.add_argument("--max-steps", type=int, default=160)
p.add_argument("--pad-terminal", type=int, default=12)
p.add_argument("--seed", type=int, default=20270400)
p.add_argument("--overwrite", action="store_true")
p.add_argument("--random-policy", action="store_true")
args = p.parse_args()
out_dir = resolve_path(args.out)
if out_dir.exists() and not args.overwrite:
    raise SystemExit(f"{out_dir} exists")
raw_dir, frames_dir = out_dir / "raw", out_dir / "frames"
raw_dir.mkdir(parents=True, exist_ok=True); frames_dir.mkdir(parents=True, exist_ok=True)
env = VizdoomCorridorEnv(VizdoomGameConfig(max_episode_steps=args.max_steps))
writer = TaskWriter(task_name="doom_health", text="vizdoom health_gathering oracle+noise",
                    raw_dir=raw_dir, frames_dir=frames_dir, frame_size=128, shard_size=2048,
                    action_dim=NUM_ACTIONS, action_labels=ACTION_LABELS, source="vizdoom_health",
                    metadata={"scenario": "health_gathering"})
rng = np.random.default_rng(args.seed)
outcomes = {"success": 0, "collision_or_oob": 0, "timeout": 0}
for ep in range(args.episodes):
    eps = 1.0 if args.random_policy else [0.0, 0.3, 0.6][ep % 3]
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    writer.append(episode=ep, frame_rgb=env.render(), action=np.zeros(NUM_ACTIONS, dtype=np.float32), reward=0.0)
    done, term_r, steps = False, 0.0, 0
    while not done and steps < args.max_steps:
        a = int(rng.integers(0, NUM_ACTIONS)) if rng.random() < eps else env.expert_action_index()
        _o, r, t1, t2, _i = env.step(a)
        fr = env.render()
        writer.append(episode=ep, frame_rgb=fr, action=one_hot(a, NUM_ACTIONS), reward=float(r))
        term_r, steps, done = float(r), steps + 1, (t1 or t2)
        if done:
            for _ in range(args.pad_terminal):
                writer.append(episode=ep, frame_rgb=fr, action=one_hot(0, NUM_ACTIONS), reward=0.0)
    outcomes["success" if (done and term_r > 5) else ("collision_or_oob" if done else "timeout")] += 1
    writer.record_episode(episode_return=0.0, length=steps)
    if (ep + 1) % 100 == 0:
        print(json.dumps({"ep": ep + 1, **outcomes}), flush=True)
info = writer.finalize()
(out_dir / "tasks.json").write_text(json.dumps({writer.task_name: {"action_dim": NUM_ACTIONS,
    "text": writer.text, "source": writer.source, "action_labels": ACTION_LABELS}}, indent=2))
(out_dir / "summary.json").write_text(json.dumps({"episodes": args.episodes, "outcomes": outcomes, "task": info}, indent=2))
print(json.dumps({"phase": "done", "outcomes": outcomes, "frames": info["frames"]}), flush=True)
