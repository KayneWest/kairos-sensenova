#!/usr/bin/env python3
"""Relabel a WMDataset raw file's rewards as per-step discounted return-to-go.

Fixes scorer myopia in the closed-loop drone experiment: the planner's scorer
regresses the window "return", which with raw shaped rewards over horizon 8
never sees the eventual collision/success. With RTG labels and --gamma 0 in
the trainer, the scorer becomes an episode value predictor (Dreamer-style),
so imagined futures heading into a tree score low even when the crash is
beyond the imagination horizon.

Writes a sibling dataset dir with a transformed raw/, symlinked frames/, and
copied tasks.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Relabel rewards as discounted return-to-go.")
    p.add_argument("--src", required=True, help="Source dataset dir (with raw/, frames/, tasks.json).")
    p.add_argument("--out", required=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite")
    (out / "raw").mkdir(parents=True, exist_ok=True)

    stats = {}
    for raw_path in sorted((src / "raw").glob("*.pt")):
        data = torch.load(raw_path, map_location="cpu")
        episode = data["episode"].long()
        reward = data["reward"].float()
        rtg = torch.zeros_like(reward)
        # per-episode backward pass; row t stores reward for transition t-1 -> t
        for ep in episode.unique().tolist():
            idx = (episode == ep).nonzero(as_tuple=True)[0]
            acc = 0.0
            for i in reversed(idx.tolist()):
                acc = float(reward[i]) + args.gamma * acc
                rtg[i] = acc
        data["reward_original"] = reward
        data["reward"] = rtg
        torch.save(data, out / "raw" / raw_path.name)
        stats[raw_path.stem] = {
            "rows": int(reward.numel()),
            "episodes": int(episode.unique().numel()),
            "reward_mean": float(reward.mean()),
            "rtg_mean": float(rtg.mean()),
            "rtg_std": float(rtg.std()),
            "rtg_min": float(rtg.min()),
            "rtg_max": float(rtg.max()),
        }

    frames_link = out / "frames"
    if not frames_link.exists():
        os.symlink(src / "frames", frames_link)
    tasks_src = src / "tasks.json"
    if tasks_src.exists():
        (out / "tasks.json").write_text(tasks_src.read_text(encoding="utf-8"), encoding="utf-8")
    (out / "rtg_summary.json").write_text(json.dumps({"gamma": args.gamma, "src": str(src), "tasks": stats}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"phase": "rtg_relabel_done", "out": str(out), "gamma": args.gamma, **stats}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
