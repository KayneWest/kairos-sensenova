#!/usr/bin/env python3
"""Verify the ViZDoom adapter against RIGHT_DATA_SPEC's sim checklist:
stepping speed, oracle success rate, snapshot/restore determinism,
action identifiability (shift-1 cosine), modal-action share."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from sensenova_drone.vizdoom_game import VizdoomCorridorEnv, VizdoomGameConfig, NUM_ACTIONS

env = VizdoomCorridorEnv(VizdoomGameConfig())
rng = np.random.default_rng(0)

# 1) speed
env.reset(seed=1)
t0, n = time.time(), 0
for _ in range(300):
    _, _, t1, t2, _ = env.step(int(rng.integers(0, NUM_ACTIONS)))
    n += 1
    if t1 or t2:
        env.reset(seed=int(rng.integers(0, 1 << 30)))
print(f"speed: {n / (time.time() - t0):.1f} env-steps/s")

# 2) oracle success + identifiability
succ, acts, lens = 0, [], []
for ep in range(30):
    env.reset(seed=100 + ep)
    done, steps, term_r = False, 0, 0.0
    while not done and steps < 160:
        a = env.expert_action_index()
        acts.append(a)
        _, r, t1, t2, _ = env.step(a)
        term_r, steps, done = r, steps + 1, (t1 or t2)
    lens.append(steps)
    succ += int(done and term_r > 5.0)
acts_np = np.array(acts)
onehot = np.eye(NUM_ACTIONS)[acts_np]
cos = float((onehot[1:] * onehot[:-1]).sum(1).mean())
counts = np.bincount(acts_np, minlength=NUM_ACTIONS)
print(f"oracle: success {succ}/30, mean len {np.mean(lens):.1f}, "
      f"shift1_cos {cos:.2f}, modal_share {counts.max() / counts.sum():.2f}, dist {counts.tolist()}")

# 3) snapshot determinism
env.reset(seed=777)
for _ in range(10):
    env.step(5)
snap = env.snapshot()
seq = [5, 1, 5, 3, 5]
import vizdoom as vzd
def pos(e):
    return (e.game.get_game_variable(vzd.GameVariable.POSITION_X),
            e.game.get_game_variable(vzd.GameVariable.POSITION_Y),
            e.game.get_game_variable(vzd.GameVariable.HEALTH))
run1 = [(env.step(a)[0].copy(), pos(env)) for a in seq]
env.restore(snap)
run2 = [(env.step(a)[0].copy(), pos(env)) for a in seq]
fdiffs = [round(float(np.abs(a[0].astype(int) - b[0].astype(int)).mean()), 2) for a, b in zip(run1, run2)]
pdiffs = [round(max(abs(x1 - x2), abs(y1 - y2), abs(h1 - h2)), 2)
          for (_, (x1, y1, h1)), (_, (x2, y2, h2)) in zip(run1, run2)]
print(f"snapshot determinism: frame diff {fdiffs}; pos/health max-diff {pdiffs} (0.0 = exact)")
env.close()
print("SMOKE DONE")
