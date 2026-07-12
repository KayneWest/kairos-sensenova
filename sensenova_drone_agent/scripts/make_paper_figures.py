#!/usr/bin/env python3
"""Generate paper figures (PNG) for the thinking-in-frames campaign.

Fig 1  training dynamics: scorer fidelity corr + time-shift ratio vs steps
       (arms D/E), timing phase transition annotated.
Fig 2  closed-loop drone game: success rate and mean return per policy
       (n=1000 powered eval; heuristic reference n=200).
Fig 3  two-seed robot-scorer trajectory across checkpoints (SOAR + held-out
       bridge): the transient inversion and recovery.

Colors: validated categorical palette (blue #2a78d6, red #e34948); controls
in neutral grays; direct labels everywhere (contrast relief rule).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/workspace/sensenova_drone_agent")
OUT = ROOT / "output" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, RED, GRAY, LGRAY, INK, MUTED = "#2a78d6", "#e34948", "#6b6a66", "#b6b5af", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": LGRAY, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
})


def eval_rows(run: str):
    p = ROOT / "output" / run / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if '"eval"' in l]


def smooth(x, k=5):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def fig1():
    arms = {"arm D (objective only)": ("latent_imagination_planner_all_data_v3_rankfix_armD", BLUE),
            "arm E (+ per-step plan)": ("latent_imagination_planner_all_data_v3_rankfix_armE", RED)}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), constrained_layout=True)
    for name, (run, color) in arms.items():
        rows = eval_rows(run)
        steps = np.array([r["step"] for r in rows]) / 1000.0
        fid = smooth([r.get("rank_fid_corr", np.nan) for r in rows])
        tsh = smooth([r.get("time_shift_over_normal", np.nan) for r in rows])
        axes[0].plot(steps, fid, color=color, lw=2)
        axes[1].plot(steps, tsh, color=color, lw=2)
        axes[0].annotate(name, (steps[-1], fid[-1]), xytext=(-4, 8 if color == BLUE else -14),
                         textcoords="offset points", ha="right", color=color, fontsize=8.5)
    axes[0].axhline(0, color=LGRAY, lw=0.8)
    axes[0].set_title("Scorer fidelity correlation (training eval)")
    axes[0].set_xlabel("training steps (thousands)")
    axes[0].set_ylim(-0.2, 1.0)
    axes[1].set_yscale("log")
    axes[1].axhline(1.0, color=LGRAY, lw=0.8)
    axes[1].axvspan(144, 156, color="#eda100", alpha=0.12, lw=0)
    axes[1].annotate("timing phase transition", (150, 1.15), ha="center", color=MUTED, fontsize=8)
    axes[1].set_title("Time-shift ratio (wrong-timing MSE / true, log)")
    axes[1].set_xlabel("training steps (thousands)")
    fig.savefig(OUT / "fig1_training_dynamics.png", dpi=200)
    plt.close(fig)


def fig2():
    # Two-seed closed-loop result: the sign reversal IS the finding.
    runs = [("training seed 1", "closed_loop_drone_game_v9_power"),
            ("training seed 2 (repeat)", "closed_loop_drone_game_v10_power_seed2")]
    order = [("think, then act\n(select by imagined value)", "act_bc_think", BLUE),
             ("act without thinking\n(BC argmax)", "act_bc", GRAY),
             ("imagine, pick at random", "act_bc_random", LGRAY)]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.8), constrained_layout=True, sharex=True)
    for ax, (seed_label, run) in zip(axes, runs):
        s = json.loads((ROOT / "output" / run / "summary.json").read_text())
        vals = [s["per_policy"][o[1]]["mean_return"] for o in order]
        colors = [o[2] for o in order]
        y = np.arange(len(order))[::-1]
        ax.barh(y, vals, height=0.55, color=colors, edgecolor="white", linewidth=1)
        for yi, v, o in zip(y, vals, order):
            succ = s["per_policy"][o[1]]["success_rate"]
            ax.annotate(f"{v:+.2f}  ({succ:.1%} succ)", (max(v, 0), yi), xytext=(4, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=8.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([o[0] for o in order] if ax is axes[0] else ["", "", ""], fontsize=8.5)
        ax.axvline(0, color=LGRAY, lw=0.8)
        ax.set_title(f"Mean return — {seed_label} (n=1000)")
        ax.set_xlim(-3.2, 1.6)
    fig.suptitle("Closed-loop value-selection reverses sign across training seeds (offline metrics equal or better in seed 2)",
                 fontsize=9.5)
    fig.savefig(OUT / "fig2_closed_loop.png", dpi=200)
    plt.close(fig)


def fig3():
    runs = {
        "seed 1": {150: "decision_quality_audit_armE_150k", 170: "decision_quality_audit_armE_0170000_robot",
                   190: "decision_quality_audit_armE_0190000_robot", 210: "decision_quality_audit_armE_210k_full"},
        "seed 2": {150: "decision_quality_audit_armE_seed2_150k", 170: "decision_quality_audit_armE_seed2_0170000_robot",
                   190: "decision_quality_audit_armE_seed2_0190000_robot", 210: "decision_quality_audit_armE_seed2_210k"},
    }
    srcs = [("soar_native_v2", "SOAR"), ("hf_robot_bridge_orig_lerobot_dreamer4", "bridge (held out)")]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.8), constrained_layout=True, sharey=True)
    for ax, (src, title) in zip(axes, srcs):
        for seed, color, ls in (("seed 1", BLUE, "-"), ("seed 2", RED, "-")):
            xs, ys = [], []
            for step, run in sorted(runs[seed].items()):
                p = ROOT / "output" / run / "summary.json"
                if not p.exists():
                    continue
                r = json.loads(p.read_text())["per_source"].get(src)
                if r is None:
                    continue
                xs.append(step)
                ys.append(r["selection"]["bank"]["fidelity_corr"]["mean"])
            ax.plot(xs, ys, color=color, lw=2, ls=ls, marker="o", ms=5, mec="white", mew=1)
            ax.annotate(seed, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=color, fontsize=8.5)
        ax.axhline(0, color=LGRAY, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("checkpoint (k steps)")
        ax.set_xticks([150, 170, 190, 210])
        ax.set_xlim(145, 228)
    axes[0].set_ylabel("scorer fidelity corr")
    axes[0].set_ylim(-1.05, 1.05)
    fig.suptitle("Robot-scorer transient: inversion tracks the timing transition, then recovers", fontsize=10)
    fig.savefig(OUT / "fig3_two_seed_scorer.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print(json.dumps({"phase": "figures_done", "out": str(OUT), "files": sorted(p.name for p in OUT.glob("*.png"))}))
