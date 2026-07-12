#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready SOAR/DROID imagination result tables.")
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--run-glob", action="append", default=[])
    parser.add_argument("--repeat-run-dir", action="append", default=[])
    parser.add_argument("--repeat-run-glob", action="append", default=[])
    parser.add_argument("--repeat-out-json", default="")
    parser.add_argument("--repeatability-json", default="")
    parser.add_argument("--breakdown-json", action="append", default=[])
    parser.add_argument("--out-md", default="sensenova_drone_agent/paper/soar_imagination_results.md")
    parser.add_argument("--title", default="SOAR/DROID Imagination Results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_md = resolve_path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    run_dirs = collect_paths(args.run_dir, args.run_glob)
    for run in run_dirs:
        run_dir = resolve_path(run)
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summaries.append((run_dir.name, json.loads(summary_path.read_text(encoding="utf-8"))))

    lines = [
        f"# {args.title}",
        "",
        "Claim boundary: learned-simulator evaluation only; no real-world robot/drone control claim.",
        "",
    ]
    if summaries:
        lines.extend(
            [
                "## Main Runs",
                "",
                "| Run | Selection metric | Selected update | Source gate | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle |",
                "|---|---|---:|:---:|---:|---:|---:|---:|",
            ]
        )
        for name, payload in summaries:
            after = payload["after_imagination"]
            best = payload.get("best_imagination_selection") or {}
            gate_pass = source_gate_pass(best)
            lines.append(
                "| {name} | `{metric}` | {update} | {gate} | {bc:+.4f} | {zero:+.4f} | {dyn_zero:+.4f} | {dyn_shuffle:+.4f} |".format(
                    name=name,
                    metric=best.get("metric", "n/a"),
                    update=best.get("selected_update", "n/a"),
                    gate="yes" if gate_pass else "no",
                    bc=float(after.get("policy_minus_bc", 0.0)),
                    zero=float(after.get("policy_minus_zero", 0.0)),
                    dyn_zero=float(after.get("policy_minus_dyn_zero", 0.0)),
                    dyn_shuffle=float(after.get("policy_minus_dyn_shuffle", 0.0)),
                )
            )
        lines.append("")

        source_rows = []
        for name, payload in summaries:
            best = payload.get("best_imagination_selection") or {}
            eval_payload = best.get("eval") or {}
            for source, source_payload in sorted((eval_payload.get("source_eval") or {}).items()):
                source_rows.append((name, source, source_payload))
        if source_rows:
            lines.extend(
                [
                    "## Source Gates",
                    "",
                    "| Run | Source | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for name, source, source_payload in source_rows:
                lines.append(
                    "| {name} | {source} | {bc:+.4f} | {zero:+.4f} | {dyn_zero:+.4f} | {dyn_shuffle:+.4f} |".format(
                        name=name,
                        source=source,
                        bc=float(source_payload.get("policy_minus_bc", 0.0)),
                        zero=float(source_payload.get("policy_minus_zero", 0.0)),
                        dyn_zero=float(source_payload.get("policy_minus_dyn_zero", 0.0)),
                        dyn_shuffle=float(source_payload.get("policy_minus_dyn_shuffle", 0.0)),
                    )
                )
            lines.append("")

    repeat = None
    repeat_dirs = collect_paths(args.repeat_run_dir, args.repeat_run_glob)
    if repeat_dirs:
        repeat = summarize_repeatability([resolve_path(path) for path in repeat_dirs])
        if args.repeat_out_json:
            repeat_out = resolve_path(args.repeat_out_json)
            repeat_out.parent.mkdir(parents=True, exist_ok=True)
            repeat_out.write_text(json.dumps(repeat, indent=2), encoding="utf-8")
    elif args.repeatability_json:
        repeat = json.loads(resolve_path(args.repeatability_json).read_text(encoding="utf-8"))
    if repeat:
        lines.extend(
            [
                "## Repeatability",
                "",
                "| Run | Pass | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Selected update |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in repeat.get("runs", []):
            lines.append(
                "| {run} | {passed} | {bc:+.4f} | {zero:+.4f} | {dyn_zero:+.4f} | {dyn_shuffle:+.4f} | {update} |".format(
                    run=row["run"],
                    passed="yes" if row.get("passed") else "no",
                    bc=float(row.get("policy_minus_bc", 0.0)),
                    zero=float(row.get("policy_minus_zero", 0.0)),
                    dyn_zero=float(row.get("policy_minus_dyn_zero", 0.0)),
                    dyn_shuffle=float(row.get("policy_minus_dyn_shuffle", 0.0)),
                    update=row.get("selected_update", "n/a"),
                )
            )
        aggregate = repeat.get("aggregate", {})
        if any(row.get("selected_update") == 0 for row in repeat.get("runs", [])):
            lines.extend(["", "Note: selected update `0` denotes the post-BC/pre-imagination checkpoint."])
        lines.extend(
            [
                "",
                f"Pass count: `{aggregate.get('pass_count', 0)}/{aggregate.get('run_count', 0)}`",
                f"Mean policy-minus-BC: `{aggregate.get('mean_policy_minus_bc', 0.0):+.4f}`",
                f"Mean policy-minus-dyn-shuffle: `{aggregate.get('mean_policy_minus_dyn_shuffle', 0.0):+.4f}`",
                "",
            ]
        )

    for breakdown_arg in args.breakdown_json:
        breakdown_path = resolve_path(breakdown_arg)
        breakdown = json.loads(breakdown_path.read_text(encoding="utf-8"))
        breakdown_label = breakdown_path.parent.parent.name if breakdown_path.parent.name == "breakdown_eval" else breakdown_path.stem
        lines.extend(
            [
                f"## Breakdown: `{breakdown_label}`",
                "",
                "| Source | Horizon | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Time-shift margin | Far-shuffle margin |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in breakdown.get("rows", []):
            margins = row.get("control_margins", {})
            lines.append(
                "| {source} | {horizon} | {bc:+.4f} | {zero:+.4f} | {dyn_zero:+.4f} | {dyn_shuffle:+.4f} | {time_shift:+.4f} | {far_shuffle:+.4f} |".format(
                    source=row["source"],
                    horizon=int(row["horizon"]),
                    bc=float(row["policy_minus_bc"]),
                    zero=float(row["policy_minus_zero"]),
                    dyn_zero=float(row["policy_minus_dyn_zero"]),
                    dyn_shuffle=float(row["policy_minus_dyn_shuffle"]),
                    time_shift=float(margins.get("time_shift", 0.0)),
                    far_shuffle=float(margins.get("far_shuffle", 0.0)),
                )
            )
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_md": str(out_md),
                "main_runs": len(summaries),
                "repeat_runs": len(repeat.get("runs", [])) if repeat else 0,
                "breakdowns": len(args.breakdown_json),
            },
            indent=2,
        )
    )
    return 0


def summarize_repeatability(run_dirs: list[Path], min_causal_margin: float = 0.002) -> dict[str, Any]:
    rows = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        after = payload["after_imagination"]
        best = payload.get("best_imagination_selection") or {}
        policy_minus_bc = float(after.get("policy_minus_bc", 0.0))
        policy_minus_zero = float(after.get("policy_minus_zero", 0.0))
        policy_minus_dyn_zero = float(after.get("policy_minus_dyn_zero", 0.0))
        policy_minus_dyn_shuffle = float(after.get("policy_minus_dyn_shuffle", 0.0))
        rows.append(
            {
                "run": run_dir.name,
                "selected_update": best.get("selected_update"),
                "policy_minus_bc": policy_minus_bc,
                "policy_minus_zero": policy_minus_zero,
                "policy_minus_dyn_zero": policy_minus_dyn_zero,
                "policy_minus_dyn_shuffle": policy_minus_dyn_shuffle,
                "passed": (
                    policy_minus_bc > 0.0
                    and policy_minus_zero > 0.0
                    and policy_minus_dyn_zero >= min_causal_margin
                    and policy_minus_dyn_shuffle >= min_causal_margin
                ),
            }
        )
    return {
        "runs": rows,
        "aggregate": {
            "run_count": len(rows),
            "pass_count": sum(1 for row in rows if row["passed"]),
            "mean_policy_minus_bc": mean([row["policy_minus_bc"] for row in rows]) if rows else 0.0,
            "mean_policy_minus_dyn_shuffle": mean([row["policy_minus_dyn_shuffle"] for row in rows]) if rows else 0.0,
        },
    }


def source_gate_pass(best_selection: dict[str, Any], min_causal_margin: float = 0.002) -> bool:
    metric_value = best_selection.get("metric_value")
    if metric_value is not None:
        return float(metric_value) > -1e5
    eval_payload = best_selection.get("eval") or {}
    return (
        float(eval_payload.get("policy_minus_bc", 0.0)) > 0.0
        and float(eval_payload.get("policy_minus_zero", 0.0)) > 0.0
        and float(eval_payload.get("policy_minus_dyn_zero", 0.0)) >= min_causal_margin
        and float(eval_payload.get("policy_minus_dyn_shuffle", 0.0)) >= min_causal_margin
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def collect_paths(paths: list[str], patterns: list[str]) -> list[Path]:
    collected = [resolve_path(path) for path in paths]
    for pattern in patterns:
        glob_path = resolve_path(pattern)
        parent = glob_path.parent
        name = glob_path.name
        collected.extend(sorted(parent.glob(name)))
    deduped = []
    seen = set()
    for path in collected:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


if __name__ == "__main__":
    raise SystemExit(main())
