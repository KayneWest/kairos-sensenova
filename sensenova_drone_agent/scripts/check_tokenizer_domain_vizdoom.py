#!/usr/bin/env python3
"""RIGHT_DATA_SPEC pre-check: can the frozen drone tokenizer reconstruct Doom?"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for item in (str(REPO_ROOT / "dreamer4" / "dreamer4"), str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT / "src")):
    sys.path.insert(0, item)
import numpy as np, torch, json
from train_dynamics import load_frozen_tokenizer_from_pt_ckpt, temporal_patchify
from sensenova_drone.vizdoom_game import VizdoomCorridorEnv, VizdoomGameConfig
from sensenova_drone.gym_drone_game import DroneGameConfig, DroneMazeEnv
import eval_gym_drone_game_act_by_imagination as ev

dev = torch.device("cuda")
enc, dec, tok = load_frozen_tokenizer_from_pt_ckpt(str(PROJECT_ROOT / "output/drone_game_tokenizer_v1/latest.pt"), device=dev)
patch = int(tok.get("patch", 8))

def clip_frames(env, n, expert):
    frames = []
    env.reset(seed=42)
    for _ in range(n):
        a = env.expert_action_index() if expert else 5
        env.step(a)
        frames.append(ev.resize_chw_uint8(env.render(), 128))
    return np.stack(frames)  # (n, C, H, W)

def recon_mse(frames):
    x = torch.from_numpy(frames).to(dev)[None].float() / 255.0  # (1,T,C,H,W)
    with torch.no_grad():
        z, _ = enc(temporal_patchify(x, patch))
        y = dec(z)
    # decoder returns patches; measure in patch space against input patches
    xp = temporal_patchify(x, patch)
    mse = float(torch.nn.functional.mse_loss(y, xp).item())
    var = float(xp.var().item())
    return mse, var

doom = clip_frames(VizdoomCorridorEnv(VizdoomGameConfig()), 16, True)
drone = clip_frames(DroneMazeEnv(DroneGameConfig(max_episode_steps=80)), 16, True)
md, vd = recon_mse(doom)
mr, vr = recon_mse(drone)
print(json.dumps({"doom": {"mse": md, "var": vd, "rel": md/vd}, "drone": {"mse": mr, "var": vr, "rel": mr/vr},
                  "doom_over_drone_rel": (md/vd)/(mr/vr)}))
