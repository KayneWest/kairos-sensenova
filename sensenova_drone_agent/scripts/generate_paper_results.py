#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


@dataclass
class PaperResultRow:
    model: str
    family: str
    source_run: str
    episodes: int | None
    eval_seed_start: int | None
    success_rate: float
    success_ci95_low: float | None
    success_ci95_high: float | None
    collision_rate: float
    timeout_rate: float
    mean_return: float
    mean_length: float | None
    deployment_score: float
    training_budget: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile existing drone-game results into paper-facing tables."
    )
    parser.add_argument(
        "--out-dir",
        default="sensenova_drone_agent/output/paper_results_v1",
        help="Directory for generated JSON/CSV/Markdown tables.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def infer_episodes(result: dict[str, Any], fallback: int | None = None) -> int | None:
    for key in ("num_episodes", "episodes"):
        value = result.get(key)
        if isinstance(value, int):
            return value
    return fallback


def deployment_score(success_rate: float, collision_rate: float, timeout_rate: float, mean_return: float) -> float:
    """Single sortable score for paper tables; not a training objective."""
    return 100.0 * success_rate + mean_return - 50.0 * collision_rate - 10.0 * timeout_rate


def wilson_ci(rate: float, episodes: int | None, z: float = 1.96) -> tuple[float | None, float | None]:
    if episodes is None or episodes <= 0:
        return None, None
    successes = int(round(rate * episodes))
    n = float(episodes)
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def row_from_result(
    result: dict[str, Any],
    *,
    model: str,
    family: str,
    source_run: str,
    seed_start: int | None,
    fallback_episodes: int | None,
    training_budget: str,
    notes: str,
) -> PaperResultRow:
    episodes = infer_episodes(result, fallback_episodes)
    success = safe_float(result.get("success_rate"))
    collision = safe_float(result.get("collision_rate"))
    timeout = safe_float(result.get("timeout_rate"))
    mean_return = safe_float(result.get("mean_return"))
    ci_low, ci_high = wilson_ci(success, episodes)
    return PaperResultRow(
        model=model,
        family=family,
        source_run=source_run,
        episodes=episodes,
        eval_seed_start=seed_start,
        success_rate=success,
        success_ci95_low=ci_low,
        success_ci95_high=ci_high,
        collision_rate=collision,
        timeout_rate=timeout,
        mean_return=mean_return,
        mean_length=result.get("mean_length"),
        deployment_score=deployment_score(success, collision, timeout, mean_return),
        training_budget=training_budget,
        notes=notes,
    )


def find_model(summary: dict[str, Any], model_name: str, encoder_source: str | None = None) -> dict[str, Any]:
    for result in summary.get("results", []):
        if result.get("model") != model_name:
            continue
        if encoder_source is not None and result.get("encoder_source") != encoder_source:
            continue
        return result
    raise KeyError(f"Could not find model={model_name!r} encoder_source={encoder_source!r}")


def compile_rows() -> tuple[list[PaperResultRow], dict[str, Any]]:
    v14_path = PROJECT_ROOT / "output/gym_drone_game_model_benchmark_v14_cnn_vs_world_model_dqn/summary.json"
    v13_path = PROJECT_ROOT / "output/gym_drone_game_model_benchmark_v13_random_encoder_dqn_shield_in_loop_10/summary.json"
    dreamer_v2_path = PROJECT_ROOT / "output/dreamer4_lite_v2_conservative/summary.json"
    dreamer_v1_path = PROJECT_ROOT / "output/dreamer4_lite_v1/summary.json"
    action_condition_path = PROJECT_ROOT / "output/action_conditioned_rollouts_v1/summary.json"

    v14 = load_json(v14_path)
    v13 = load_json(v13_path)
    dreamer_v2 = load_json(dreamer_v2_path)
    dreamer_v1 = load_json(dreamer_v1_path)
    action_condition = load_json(action_condition_path) if action_condition_path.exists() else {}

    rows = [
        row_from_result(
            find_model(v14, "heuristic"),
            model="Geometric heuristic",
            family="hand-coded",
            source_run=rel(v14_path),
            seed_start=v14.get("seed_start"),
            fallback_episodes=v14.get("episodes"),
            training_budget="none",
            notes="Safe hand-coded baseline; often stalls/timeouts.",
        ),
        row_from_result(
            find_model(v14, "image_bc"),
            model="Image BC from DQN teacher",
            family="pixel imitation",
            source_run=rel(v14_path),
            seed_start=v14.get("seed_start"),
            fallback_episodes=v14.get("episodes"),
            training_budget="offline BC",
            notes="Closed-loop pixel imitation from DQN episodes.",
        ),
        row_from_result(
            find_model(v14, "cnn_dqn"),
            model="CNN DQN from scratch",
            family="pixel RL baseline",
            source_run=rel(v14_path),
            seed_start=v14.get("seed_start"),
            fallback_episodes=v14.get("episodes"),
            training_budget="6000 env steps",
            notes="Under small budget, slower than frozen world-model features; needs longer run before final paper claim.",
        ),
        row_from_result(
            find_model(v13, "world_model_dqn", encoder_source="random"),
            model="Frozen random encoder DQN",
            family="representation ablation",
            source_run=rel(v13_path),
            seed_start=v13.get("seed_start"),
            fallback_episodes=v13.get("episodes"),
            training_budget="12000 env steps",
            notes="Same architecture/shield path with frozen random visual features.",
        ),
        row_from_result(
            find_model(v14, "world_model_dqn", encoder_source="pretrained"),
            model="Frozen world-model encoder DQN",
            family="world-model features + RL",
            source_run=rel(v14_path),
            seed_start=v14.get("seed_start"),
            fallback_episodes=v14.get("episodes"),
            training_budget="12000 env steps",
            notes="Current strongest paper result in matched drone-game benchmark.",
        ),
        row_from_result(
            dreamer_v2["supervised_eval"],
            model="Dreamer4-lite supervised",
            family="world-model features + BC/reward heads",
            source_run=rel(dreamer_v2_path),
            seed_start=None,
            fallback_episodes=dreamer_v2["supervised_eval"].get("episodes"),
            training_budget="3 supervised epochs",
            notes="Policy/reward/value heads on frozen world model before imagination updates.",
        ),
        row_from_result(
            dreamer_v2["final_eval"],
            model="Dreamer4-lite KL imagination",
            family="world-model features + imagination update",
            source_run=rel(dreamer_v2_path),
            seed_start=None,
            fallback_episodes=dreamer_v2["final_eval"].get("episodes"),
            training_budget="3 epochs + 80 imagination updates",
            notes="Conservative KL-constrained imagination improved slightly but remains below DQN.",
        ),
        row_from_result(
            dreamer_v1["final_eval"],
            model="Dreamer4-lite weak-KL failure",
            family="negative result",
            source_run=rel(dreamer_v1_path),
            seed_start=None,
            fallback_episodes=dreamer_v1["final_eval"].get("episodes"),
            training_budget="weakly constrained imagination",
            notes="Reward-model exploitation/collapse case; useful limitation evidence.",
        ),
    ]

    context = {
        "claim_boundary": (
            "The table supports claims about a learned action-conditioned drone-game "
            "world-model encoder. It does not yet prove that the full Kairos/Sensenova "
            "foundation model can directly produce robust drone policies."
        ),
        "primary_result": "Frozen world-model encoder DQN",
        "action_conditioned_kairos_rollouts": {
            "source": rel(action_condition_path),
            "summary": action_condition,
        },
        "paper_ready": False,
        "paper_ready_blockers": [
            "Run all trainable agents across multiple training seeds.",
            "Evaluate every agent on one larger matched seed suite, preferably 256-1000 episodes.",
            "Add a longer/fairer CNN baseline and preferably an off-the-shelf pretrained visual encoder baseline.",
            "Separate toy drone-game claims from PX4/Gazebo transfer claims.",
            "Decide whether to include Kairos prompt/action-conditioning as a negative result or defer it.",
        ],
    }
    return rows, context


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def ci_text(row: PaperResultRow) -> str:
    if row.success_ci95_low is None or row.success_ci95_high is None:
        return pct(row.success_rate)
    return f"{pct(row.success_rate)} [{pct(row.success_ci95_low)}, {pct(row.success_ci95_high)}]"


def write_csv(rows: list[PaperResultRow], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(rows: list[PaperResultRow], context: dict[str, Any], path: Path) -> None:
    ranked = sorted(rows, key=lambda row: row.deployment_score, reverse=True)
    lines = [
        "# Paper Results V1",
        "",
        "This file is generated by `sensenova_drone_agent/scripts/generate_paper_results.py`.",
        "",
        "## Claim Boundary",
        "",
        context["claim_boundary"],
        "",
        "## Main Table",
        "",
        "| Rank | Model | Family | Episodes | Success 95% CI | Collision | Timeout | Mean Return | Deployment Score | Training Budget |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row.model,
                    row.family,
                    str(row.episodes or "n/a"),
                    ci_text(row),
                    pct(row.collision_rate),
                    pct(row.timeout_rate),
                    number(row.mean_return, 2),
                    number(row.deployment_score, 2),
                    row.training_budget,
                ]
            )
            + " |"
        )

    lines += [
        "",
        "Deployment score is `100*success_rate + mean_return - 50*collision_rate - 10*timeout_rate`.",
        "It is a compact reporting score, not a training objective.",
        "",
        "## Notes",
        "",
    ]
    for row in ranked:
        lines.append(f"- `{row.model}`: {row.notes} Source: `{row.source_run}`.")

    lines += [
        "",
        "## Current Publication Status",
        "",
        "- `PAPER_READY=false`",
        "- `WORKSHOP_SUBMISSION_CANDIDATE=true` after reproducing the table with more seeds and a stronger baseline.",
        "",
        "## Required Before Submission",
        "",
    ]
    for blocker in context["paper_ready_blockers"]:
        lines.append(f"- {blocker}")

    lines += [
        "",
        "## Recommended Conservative Claim",
        "",
        (
            "In a lightweight first-person drone navigation game, an action-conditioned "
            "visual world-model encoder provides a more sample-efficient representation "
            "for shielded reward-driven control than a frozen random encoder and a small-budget "
            "CNN trained from scratch. KL-constrained imagination updates are feasible but "
            "currently weaker than direct DQN on frozen world-model features."
        ),
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, context = compile_rows()
    rows_for_json = [asdict(row) for row in rows]
    (out_dir / "paper_results.json").write_text(
        json.dumps({"rows": rows_for_json, "context": context}, indent=2, sort_keys=True)
    )
    write_csv(rows, out_dir / "paper_results.csv")
    write_markdown(rows, context, out_dir / "paper_results.md")
    print(f"Wrote {len(rows)} rows to {rel(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
