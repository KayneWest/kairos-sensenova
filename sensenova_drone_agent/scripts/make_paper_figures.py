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
    # The whole paper in one row: value-selection wins on seed 1, reverses on
    # seed 2 (offline metrics equal or better), and one DAgger iteration makes
    # the win consistent across both seeds.
    runs = [("before DAgger\nseed 1", "closed_loop_drone_game_v9_power"),
            ("before DAgger\nseed 2 (repeat)", "closed_loop_drone_game_v10_power_seed2"),
            ("after one DAgger iter.\nseed 1", "closed_loop_drone_game_v12_dagger_c1"),
            ("after one DAgger iter.\nseed 2 (repeat)", "closed_loop_drone_game_v12_dagger_c1_seed2")]
    order = [("think, then act\n(select by imagined value)", "act_bc_think", BLUE),
             ("act without thinking\n(BC argmax)", "act_bc", GRAY),
             ("imagine, pick at random", "act_bc_random", LGRAY)]
    verdicts = ["win", "REVERSAL", "win", "win"]
    fig, axes = plt.subplots(1, 4, figsize=(11.4, 2.9), constrained_layout=True, sharex=True)
    for ax, (panel_label, run), verdict in zip(axes, runs, verdicts):
        s = json.loads((ROOT / "output" / run / "summary.json").read_text())
        vals = [s["per_policy"][o[1]]["mean_return"] for o in order]
        colors = [o[2] for o in order]
        y = np.arange(len(order))[::-1]
        ax.barh(y, vals, height=0.55, color=colors, edgecolor="white", linewidth=1)
        for yi, v, o in zip(y, vals, order):
            succ = s["per_policy"][o[1]]["success_rate"]
            ax.annotate(f"{v:+.2f} ({succ:.1%})", (max(v, 0), yi), xytext=(3, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=7.8, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([o[0] for o in order] if ax is axes[0] else ["", "", ""], fontsize=8.5)
        ax.axvline(0, color=LGRAY, lw=0.8)
        ax.set_title(panel_label, fontsize=9,
                     color=(RED if verdict == "REVERSAL" else INK))
        ax.set_xlim(-6.6, 3.6)
        ax.set_xlabel("mean return")
    fig.suptitle("One DAgger iteration turns seed-fragile act-time selection into a consistent win "
                 "(n=1000 per panel; labels: mean return (success rate))", fontsize=9.5)
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


def fig0_architecture():
    # Think-then-act loop schematic. Boxes drawn manually; annotations mark
    # the four fixes (rank loss, unit-norm plans, reward-detach, per-step
    # plan readout) and the plan-free act-time heads (shortcut cure).
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, label, fc="#f4f3f0", ec=LGRAY, color=INK, fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=color, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color=GRAY, ls="-", lw=1.4):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=11, color=color, ls=ls, lw=lw))

    def note(x, y, text, color=MUTED, fs=7.4, ha="center"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, style="italic")

    # ----- perception (shared) -----
    box(1, 26, 12, 7, "context\nframes", fc="#eceff5")
    box(17, 26, 13, 7, "frozen\ntokenizer ❄")
    box(34, 26, 12, 7, "context\nencoder GRU")
    arrow(13, 29.5, 17, 29.5)
    arrow(30, 29.5, 34, 29.5)
    # ctx_h feeds proposer and scorer (elbow line, then drops)
    ax.plot([46, 80], [27, 27], color=LGRAY, lw=1.4)
    note(63, 28.2, "ctx_h", fs=8)

    # ----- think lane -----
    box(1, 8, 15, 7, "K candidate\naction chunks", fc="#eceff5")
    box(20, 8, 13, 7, "plan encoder\n(unit-norm)")
    box(37, 8, 17, 7, "future proposer GRU\n(per-step plan readout)", fc="#e8f0fb", ec=BLUE)
    box(58, 8, 14, 7, "K imagined\nfutures ẑ⁽ᵏ⁾", fc="#e8f0fb", ec=BLUE)
    box(76, 8, 13, 7, "scorer\nS(ctx, ẑ, ∅)")
    arrow(16, 11.5, 20, 11.5)
    arrow(33, 11.5, 37, 11.5)
    arrow(54, 11.5, 58, 11.5)
    arrow(72, 11.5, 76, 11.5)
    arrow(40, 26, 40, 15.4, color=LGRAY)          # ctx_h -> proposer
    arrow(80, 27, 80, 15.4, color=LGRAY)          # ctx_h -> scorer (from elbow)
    note(8.5, 5.6, "BC-anchored samples\n(+ dropout-diverse)")
    note(26.5, 5.6, "‖p‖ = √d  (fix 2)")
    note(45.5, 5.6, "plan token → per-step\nembeddings (fix 4)")
    note(82.5, 5.6, "plan-free at act time\n(shortcut cure)")

    # ----- act lane -----
    box(93, 8, 12, 7, "select best\nfuture ẑ*", fc="#fdeeee", ec=RED)
    box(93, 26, 12, 7, "inverse dyn.\nI(ctx, ẑ*, ∅)", fc="#fdeeee", ec=RED)
    arrow(89, 11.5, 93, 11.5)
    arrow(99, 15.4, 99, 26, color=RED)
    arrow(93, 29.5, 89, 29.5, color=RED)
    ax.text(84, 31.5, "action chunk → env", fontsize=8.5, color=RED,
            ha="center", fontweight="bold")

    # ----- training-only heads -----
    box(37, 37, 21, 6, "rank hinge on fidelity-ranked\ncandidates (fix 1)", fc="#fbf6e9", ec="#c9a227", fs=7.8)
    box(62, 37, 21, 6, "reward / RTG head\n(inputs detached — fix 3)", fc="#fbf6e9", ec="#c9a227", fs=7.8)
    note(31, 40, "training-only:", ha="right", fs=8)
    arrow(65, 15.4, 47, 37, color="#c9a227", ls=":")
    arrow(80, 15.4, 72, 37, color="#c9a227", ls=":")

    ax.text(1, 36, "PERCEIVE", fontsize=9, color=MUTED, fontweight="bold")
    ax.text(1, 18.5, "THINK  (imagine K futures, one per candidate action chunk)",
            fontsize=9, color=BLUE, fontweight="bold")
    ax.text(102.5, 20.5, "ACT", fontsize=9, color=RED, fontweight="bold",
            ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig0_architecture.png", dpi=200)
    plt.close(fig)


def fig4_trace_grid():
    # Visible thinking: real rollout vs decoded imagined futures. Expert ctx 3
    # is both visually legible (walker discernible in decoded rows) and
    # discriminative: selected ~ true plan (0.0034 vs 0.0028), random 5.8x
    # worse, zero-plan 16x worse.
    tr_dir = ROOT / "output" / "imagination_traces_armE_latest_v2dec"
    meta = next(t for t in json.loads((tr_dir / "traces.json").read_text())
                if t["source"] == "dreamer4_hf_expert" and t["context_id"] == 3)
    img = plt.imread(tr_dir / "dreamer4_hf_expert" / "ctx_03_grid.png")
    cell = img.shape[0] // len(meta["row_order"])   # square cells, rows stacked
    n_ctx = img.shape[1] // cell - 8                 # columns = ctx + horizon(8)
    rows = [("real rollout", "real", None),
            ("imagined — true plan", "true_plan", BLUE),
            ("imagined — selected plan", "selected", BLUE),
            ("imagined — random plan", "random", GRAY),
            ("imagined — no plan (zeroed)", "zero_plan", RED)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10.2, 4.8), constrained_layout=True)
    for ax, (label, key, color) in zip(axes, rows):
        r = meta["row_order"].index(key)
        ax.imshow(img[r * cell:(r + 1) * cell, :, :])
        ax.axvline(n_ctx * cell - 0.5, color="white", ls="--", lw=1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if key == "real":
            tag = label
        else:
            tag = f"{label}   (future MSE {meta['future_mse'][key]:.4f})"
        ax.set_ylabel(tag, rotation=0, ha="right", va="center", fontsize=8.5,
                      color=color or INK)
    axes[0].annotate("context", (n_ctx * cell / 2, -10), ha="center", fontsize=8,
                     color=MUTED, annotation_clip=False)
    axes[0].annotate("imagined / real future (horizon 8)",
                     ((n_ctx + 4) * cell, -10), ha="center", fontsize=8,
                     color=MUTED, annotation_clip=False)
    fig.suptitle("Thinking in frames: decoded imagined futures are action-conditioned "
                 "(expert walker, held-out context)", fontsize=9.5)
    fig.savefig(OUT / "fig4_trace_grid.png", dpi=200)
    plt.close(fig)


def fig5_dagger_cycles():
    # The iteration arc: think-success across DAgger data recipes, two seeds
    # each. Cycle 1 passes the strict gate twice; every recipe containing
    # cycle-2 data fails, and pure replacement re-inverts selection.
    recipes = [
        ("cycle 1\nbase + c1 (1/3)", ["closed_loop_drone_game_v12_dagger_c1",
                                      "closed_loop_drone_game_v12_dagger_c1_seed2"], True),
        ("cycle 2\n+ c2 (1/2)", ["closed_loop_drone_game_v13_dagger_c2",
                                 "closed_loop_drone_game_v13_dagger_c2_seed2"], False),
        ("cycle 2b\n+ c2 (1/3)", ["closed_loop_drone_game_v14_dagger_c2_rebal",
                                  "closed_loop_drone_game_v14_dagger_c2_rebal_seed2"], False),
        ("cycle 2c\nbase + c2 only (1/3)", ["closed_loop_drone_game_v15_dagger_c2_only",
                                            "closed_loop_drone_game_v15_dagger_c2_only_seed2"], False),
    ]
    series = [("act_bc_think", "think, then act", BLUE, "o", 62),
              ("act_bc", "act without thinking", GRAY, "s", 30),
              ("act_bc_random", "imagine, pick at random", LGRAY, "^", 34)]
    fig, ax = plt.subplots(figsize=(7.6, 3.2), constrained_layout=True)
    for xi, (label, runs, gate) in enumerate(recipes):
        for si, run in enumerate(runs):
            p = ROOT / "output" / run / "summary.json"
            if not p.exists():
                ax.annotate("seed 2\nrunning", (xi + 0.13, 0.4), ha="center", fontsize=7,
                            color=MUTED, style="italic")
                continue
            s = json.loads(p.read_text())["per_policy"]
            x = xi + (-0.13 if si == 0 else 0.13)
            for key, _, color, marker, size in series:
                ax.scatter(x, s[key]["success_rate"] * 100, color=color, marker=marker,
                           s=size, zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate("gate " + ("PASS x2" if gate else "fail"),
                    (xi, -1.55), ha="center", fontsize=8,
                    color=(BLUE if gate else RED), fontweight="bold",
                    annotation_clip=False)
    for key, name, color, marker, size in series:
        ax.scatter([], [], color=color, marker=marker, s=size, label=name,
                   edgecolor="white", linewidth=0.8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xticks(range(len(recipes)))
    ax.set_xticklabels([r[0] for r in recipes], fontsize=8.5)
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(-0.3, 7.2)
    ax.set_xlim(-0.5, len(recipes) - 0.5)
    ax.set_title("Self-improvement is a repair, not a ladder (two seeds per recipe, n=1000 per point)",
                 fontsize=9.5)
    fig.savefig(OUT / "fig5_dagger_cycles.png", dpi=200)
    plt.close(fig)


def fig6_decomposition():
    # The judge/imagination exchange: think-success follows the proposer in
    # all 8 cells; the judge is irrelevant everywhere. Right panel: the
    # value-guided diffusion test — guidance on an action-blind prior
    # collapses into passivity under either judge.
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.3), width_ratios=[1.15, 1],
                             constrained_layout=True)

    ax = axes[0]
    cells = [
        ("healthy imagination\n(cycle-1 proposer)", 0, [
            ("honest judge", ["closed_loop_drone_game_v16_diffthink_goodjudge_gruarg",
                              "closed_loop_drone_game_v17_decomp_c1prop-goodjudge_seed2"]),
            ('"inverted" judge', ["closed_loop_drone_game_v16_diffthink_badjudge_gruarg",
                                  "closed_loop_drone_game_v17_decomp_c1prop-badjudge_seed2"])]),
        ("poisoned imagination\n(cycle-2c proposer)", 1, [
            ("honest judge", ["closed_loop_drone_game_v17_decomp_c2cprop-goodjudge",
                              "closed_loop_drone_game_v17_decomp_c2cprop-goodjudge_seed2"]),
            ('"inverted" judge', ["closed_loop_drone_game_v17_decomp_c2cprop-badjudge",
                                  "closed_loop_drone_game_v17_decomp_c2cprop-badjudge_seed2"])]),
    ]
    for label, gi, judges in cells:
        for ji, (jlabel, runs) in enumerate(judges):
            x = gi * 2 + ji * 0.7
            for si, run in enumerate(runs):
                s = json.loads((ROOT / "output" / run / "summary.json").read_text())
                y = s["per_policy"]["gru_argmax"]["success_rate"] * 100
                ax.scatter(x + (-0.1 if si == 0 else 0.1), y, color=BLUE, s=60,
                           zorder=3, edgecolor="white", linewidth=0.8)
            ax.annotate(jlabel, (x, -0.75), ha="center", fontsize=7.6, color=MUTED,
                        annotation_clip=False)
        ax.annotate(label, (gi * 2 + 0.35, -1.7), ha="center", fontsize=8.5,
                    color=INK, fontweight="bold", annotation_clip=False)
    ax.axhline(2.0, color=LGRAY, lw=1, ls="--")
    ax.annotate("random-selection band (~1.5-2.5%)", (1.35, 2.25), fontsize=7.4, color=MUTED, ha="center")
    ax.set_xticks([])
    ax.set_ylabel("think success (%)")
    ax.set_ylim(-0.3, 7.2)
    ax.set_xlim(-0.5, 3.2)
    ax.set_title("Exchange test: success follows the imagination,\nnever the judge (2 seeds x 4 cells, n=1000)", fontsize=9)

    ax = axes[1]
    pols = [("bc", "BC\n(no think)", GRAY), ("bc_random", "random\nselect", LGRAY),
            ("diff_argmax", "diffusion\nargmax", "#7ca6d8"), ("diff_prior", "diffusion\nprior", "#b9cfe8"),
            ("diff_guided", "diffusion\nguided", BLUE)]
    runs = ["closed_loop_drone_game_v16_diffthink_goodjudge", "closed_loop_drone_game_v16_diffthink_goodjudge_seed2"]
    for pi, (key, label, color) in enumerate(pols):
        for si, run in enumerate(runs):
            s = json.loads((ROOT / "output" / run / "summary.json").read_text())
            y = s["per_policy"][key]["success_rate"] * 100
            ax.scatter(pi + (-0.12 if si == 0 else 0.12), y, color=color, s=55,
                       zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(label, (pi, -0.75), ha="center", fontsize=7.6, color=MUTED,
                    annotation_clip=False)
    ax.axhline(6.1, color=RED, lw=1, ls=":")
    ax.annotate("GRU argmax reference (6.1%)", (2.0, 6.35), fontsize=7.4, color=RED, ha="center")
    ax.set_xticks([])
    ax.set_ylim(-0.3, 7.2)
    ax.set_title("Value-guided diffusion: an action-blind prior\ngives guidance nothing to steer (good judge)", fontsize=9)
    fig.savefig(OUT / "fig6_decomposition.png", dpi=200)
    plt.close(fig)


def fig7_expert_dagger():
    # True DAgger (expert corrections on visited states, imagination fixed):
    # corrective data doubles the campaign ceiling in 1-2 rounds, then
    # plateaus ~10% — the next binding constraint (frozen encoder).
    runs = {
        "seed 1": ("closed_loop_drone_game_v20_expert_dagger_s1_r{r}", (0.061, 0.005)),
        "seed 2": ("closed_loop_drone_game_v20_expert_dagger_s2_r{r}", (0.057, 0.001)),
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    for (label, (tpl, (think0, bc0))), color in zip(runs.items(), (BLUE, RED)):
        xs, think, bc = [0], [think0 * 100], [bc0 * 100]
        for r in (1, 2, 3):
            p_ = ROOT / "output" / tpl.format(r=r) / "summary.json"
            d = json.loads(p_.read_text())["per_policy"]
            xs.append(r)
            think.append(d["gru_argmax"]["success_rate"] * 100)
            bc.append(d["bc"]["success_rate"] * 100)
        ax.plot(xs, think, color=color, lw=2, marker="o", ms=5, mec="white", label=f"{label} — think")
        ax.plot(xs, bc, color=color, lw=1.6, ls="--", marker="s", ms=4, mec="white", alpha=0.65,
                label=f"{label} — BC alone")
    ax.axhline(6.1, color=GRAY, lw=1, ls=":")
    ax.annotate("previous campaign ceiling (6.1%)", (0.05, 6.35), fontsize=7.6, color=MUTED)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0\n(self-imitation\nDAgger)", "1", "2", "3"], fontsize=8)
    ax.set_xlabel("expert-correction round")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 12.5)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right", ncols=2)
    ax.set_title("Corrective data ladders where self-imitation could not, then plateaus\n"
                 "(expert labels on visited states; imagination and judge held fixed; n=1000/point; expert ceiling 41.5%)",
                 fontsize=9)
    fig.savefig(OUT / "fig7_expert_dagger.png", dpi=200)
    plt.close(fig)


def fig8_doom_expert_dagger():
    # Second domain (ViZDoom health-gathering, borrowed drone tokenizer,
    # 100% scripted teacher): clean corrective labels jump selection to a
    # ~78-84% ceiling in round 1 and saturate (two seeds); the drift-noised
    # teacher tops out below half that; BC degrades under aggregation while
    # selection holds the ceiling.
    runs = [
        ("clean teacher — seed 1", "closed_loop_drone_game_v20_doom_expert_dagger_s1fix_r{r}", BLUE, "-"),
        ("clean teacher — seed 2", "closed_loop_drone_game_v20_doom_expert_dagger_s2fix_r{r}", RED, "-"),
        ("noisy teacher (drifted labels)", "closed_loop_drone_game_v20_doom_expert_dagger_s1_r{r}", GRAY, "-"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    for label, tpl, color, ls in runs:
        xs, think, bc = [], [], []
        for r in (1, 2, 3):
            p_ = ROOT / "output" / tpl.format(r=r) / "summary.json"
            d = json.loads(p_.read_text())["per_policy"]
            xs.append(r)
            think.append(d["gru_argmax"]["success_rate"] * 100)
            bc.append(d["bc"]["success_rate"] * 100)
        ax.plot(xs, think, color=color, lw=2, ls=ls, marker="o", ms=5, mec="white",
                label=f"{label} — think")
        ax.plot(xs, bc, color=color, lw=1.6, ls="--", marker="s", ms=4, mec="white",
                alpha=0.6, label=f"{label} — BC alone")
    ax.axhline(100, color=GRAY, lw=1, ls=":")
    ax.annotate("teacher (100%)", (1.02, 95.0), fontsize=7.6, color=MUTED)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("expert-correction round")
    ax.set_ylabel("survival rate (%)")
    ax.set_ylim(0, 104)
    ax.legend(frameon=False, fontsize=7.2, loc="lower left", ncols=2)
    ax.set_title("Doom, borrowed tokenizer: clean corrective labels hit a ~80% ceiling immediately;\n"
                 "BC degrades under aggregation while selection holds (n=500/point; teacher 100%)",
                 fontsize=9)
    fig.savefig(OUT / "fig8_doom_expert_dagger.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig0_architecture(); fig1(); fig2(); fig3(); fig4_trace_grid(); fig5_dagger_cycles(); fig6_decomposition(); fig7_expert_dagger(); fig8_doom_expert_dagger()
    print(json.dumps({"phase": "figures_done", "out": str(OUT), "files": sorted(p.name for p in OUT.glob("*.png"))}))
