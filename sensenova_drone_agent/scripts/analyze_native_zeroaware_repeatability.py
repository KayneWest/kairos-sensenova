#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze native zero-aware imagination repeat seeds.")
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory. Repeat for each seed.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--margin", type=float, default=0.002)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw in args.run_dir:
        run_dir = resolve_path(raw)
        best_path = run_dir / "best_imagination_selection.json"
        if not best_path.exists():
            raise FileNotFoundError(best_path)
        best = json.loads(best_path.read_text(encoding="utf-8"))
        ev = best.get("eval") or {}
        row = {
            "seed": infer_seed(run_dir),
            "run_dir": str(run_dir),
            "selected_update": best.get("selected_update"),
            "metric": best.get("metric"),
            "metric_value": best.get("metric_value"),
            "policy_minus_bc": f(ev, "policy_minus_bc"),
            "policy_minus_zero": f(ev, "policy_minus_zero"),
            "policy_minus_dyn_zero": f(ev, "policy_minus_dyn_zero"),
            "policy_minus_dyn_shuffle": f(ev, "policy_minus_dyn_shuffle"),
            "causal_policy_gain": f(ev, "causal_policy_gain"),
        }
        row["strict_gate_pass"] = strict_gate(row, margin=float(args.margin))
        row["failure_reasons"] = failure_reasons(row, margin=float(args.margin))
        row["source_h8"] = load_source_h8(run_dir)
        rows.append(row)

    rows.sort(key=lambda item: str(item["seed"]))
    pass_rows = [row for row in rows if row["strict_gate_pass"]]
    fail_rows = [row for row in rows if not row["strict_gate_pass"]]
    summary = {
        "margin": float(args.margin),
        "num_runs": len(rows),
        "num_pass": len(pass_rows),
        "num_fail": len(fail_rows),
        "pass_rate": len(pass_rows) / max(1, len(rows)),
        "mean_policy_minus_bc": mean([row["policy_minus_bc"] for row in rows]) if rows else 0.0,
        "mean_pass_policy_minus_bc": mean([row["policy_minus_bc"] for row in pass_rows]) if pass_rows else 0.0,
        "mean_fail_policy_minus_bc": mean([row["policy_minus_bc"] for row in fail_rows]) if fail_rows else 0.0,
        "rows": rows,
        "claim_boundary": "Learned-simulator repeatability analysis only; does not prove real-world robot control.",
    }
    (out_dir / "native_zeroaware_repeatability_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, out_dir / "native_zeroaware_repeatability_analysis.md")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


def f(payload: dict[str, Any], key: str) -> float:
    return float(payload.get(key, 0.0))


def strict_gate(row: dict[str, Any], *, margin: float) -> bool:
    return (
        row["policy_minus_bc"] > 0.0
        and row["policy_minus_zero"] >= 0.0
        and row["policy_minus_dyn_zero"] >= margin
        and row["policy_minus_dyn_shuffle"] >= margin
    )


def failure_reasons(row: dict[str, Any], *, margin: float) -> list[str]:
    reasons = []
    if row["policy_minus_bc"] <= 0.0:
        reasons.append("bc")
    if row["policy_minus_zero"] < 0.0:
        reasons.append("zero")
    if row["policy_minus_dyn_zero"] < margin:
        reasons.append("dyn_zero")
    if row["policy_minus_dyn_shuffle"] < margin:
        reasons.append("dyn_shuffle")
    return reasons


def load_source_h8(run_dir: Path) -> dict[str, dict[str, float]]:
    path = run_dir / "breakdown_eval" / "breakdown_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in payload.get("rows", []):
        if int(row.get("horizon", -1)) != 8:
            continue
        out[str(row.get("source", ""))] = {
            "policy_minus_bc": f(row, "policy_minus_bc"),
            "policy_minus_zero": f(row, "policy_minus_zero"),
            "policy_minus_dyn_zero": f(row, "policy_minus_dyn_zero"),
            "policy_minus_dyn_shuffle": f(row, "policy_minus_dyn_shuffle"),
        }
    return out


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Native Zero-Aware Repeatability Analysis",
        "",
        f"Pass count: `{summary['num_pass']}/{summary['num_runs']}`",
        f"Mean policy-minus-BC: `{summary['mean_policy_minus_bc']:+.4f}`",
        "",
        "| Seed | Selected update | Pass | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Failure reasons |",
        "|---:|---:|:---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {seed} | {update} | {passed} | {bc:+.4f} | {zero:+.4f} | {dyn_zero:+.4f} | {dyn_shuffle:+.4f} | {reasons} |".format(
                seed=row["seed"],
                update=row["selected_update"],
                passed="yes" if row["strict_gate_pass"] else "no",
                bc=row["policy_minus_bc"],
                zero=row["policy_minus_zero"],
                dyn_zero=row["policy_minus_dyn_zero"],
                dyn_shuffle=row["policy_minus_dyn_shuffle"],
                reasons=", ".join(row["failure_reasons"]) or "-",
            )
        )
    lines.extend(["", "## Source H8 Rows", ""])
    lines.extend(
        [
            "| Seed | Source | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        for source, vals in sorted(row.get("source_h8", {}).items()):
            lines.append(
                "| {seed} | {source} | {bc:+.4f} | {zero:+.4f} | {dyn_zero:+.4f} | {dyn_shuffle:+.4f} |".format(
                    seed=row["seed"],
                    source=source,
                    bc=vals["policy_minus_bc"],
                    zero=vals["policy_minus_zero"],
                    dyn_zero=vals["policy_minus_dyn_zero"],
                    dyn_shuffle=vals["policy_minus_dyn_shuffle"],
                )
            )
    lines.extend(["", summary["claim_boundary"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def infer_seed(path: Path) -> str:
    match = re.search(r"seed_(\d+)", path.name)
    if match:
        return match.group(1)
    match = re.search(r"(\d{8})", path.name)
    return match.group(1) if match else path.name


def resolve_path(value: str | Path) -> Path:
    text = str(value)
    repo = Path(__file__).resolve().parents[2]
    if text.startswith("/workspace/"):
        return (repo / text[len("/workspace/") :]).resolve()
    path = Path(text)
    if path.is_absolute():
        return path
    return (repo / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
