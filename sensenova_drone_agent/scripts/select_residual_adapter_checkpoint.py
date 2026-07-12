#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


DEFAULT_EVALS = {
    "random_final": "sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_m1_v2/residual_adapter_eval.json",
    "random_final_far": "sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_m1_v2/adapter_latest_farshuffle_eval256.json",
    "far_step5000": "sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_farshuffle_m1_v1/adapter_step_0005000_eval128.json",
    "far_final": "sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_farshuffle_m1_v1/residual_adapter_eval.json",
    "effect_final": "sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_effect_farshuffle_m1_v1/residual_adapter_eval.json",
}

ADAPTER_CKPTS = {
    "random_final": "sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_m1_v2/adapter_latest.pt",
    "random_final_far": "sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_m1_v2/adapter_latest.pt",
    "far_step5000": "sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_farshuffle_m1_v1/adapter_step_0005000.pt",
    "far_final": "sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_farshuffle_m1_v1/adapter_latest.pt",
    "effect_final": "sensenova_drone_agent/output/residual_action_adapter_soar_droid_random_signal_effect_farshuffle_m1_v1/adapter_latest.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank residual adapter checkpoints from held-out causal eval JSON files.")
    parser.add_argument("--eval", action="append", default=[], help="label:path to an eval JSON file.")
    parser.add_argument("--out-dir", default="sensenova_drone_agent/output/residual_action_adapter_selection_v1")
    parser.add_argument("--causal-min-ratio", type=float, default=1.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evals = parse_eval_specs(args.eval) if args.eval else DEFAULT_EVALS
    rows = []
    for label, path_str in evals.items():
        path = resolve_path(path_str)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(summarize_eval(label=label, path=path, payload=payload, causal_min_ratio=args.causal_min_ratio))
    if not rows:
        raise SystemExit("No eval files found.")
    rows.sort(key=lambda row: row["score"], reverse=True)
    selected = rows[0]
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "residual_adapter_checkpoint_selection",
        "causal_min_ratio": float(args.causal_min_ratio),
        "selected": selected,
        "ranked": rows,
        "selection_rule": (
            "Prefer checkpoints that beat persistence and pass AR cross/no-op/temporal controls; "
            "score balances rollout quality with causal ratios and penalizes weak direct cross-trajectory sensitivity."
        ),
    }
    (out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, out_dir / "selection_summary.md")
    print(json.dumps({"selected": selected, "out_dir": str(out_dir)}, indent=2))
    return 0


def summarize_eval(*, label: str, path: Path, payload: dict[str, Any], causal_min_ratio: float) -> dict[str, Any]:
    metrics = payload["metrics"]
    ar = metrics["autoregressive"]
    direct = metrics["direct"]
    cross_key = first_ratio_key(ar, ["effect_far_shuffle_over_normal", "far_shuffle_over_normal", "shuffle_over_normal"])
    direct_cross_key = first_ratio_key(direct, ["effect_far_shuffle_over_normal", "far_shuffle_over_normal", "shuffle_over_normal"])
    temporal_keys = [
        key
        for key in [
            "time_shift_over_normal",
            "time_shift2_over_normal",
            "time_shift4_over_normal",
            "time_shift8_over_normal",
            "time_perm_over_normal",
            "time_reverse_over_normal",
        ]
        if key in ar
    ]
    temporal_min = min(float(ar[key]) for key in temporal_keys) if temporal_keys else 0.0
    cross = float(ar[cross_key])
    direct_cross = float(direct[direct_cross_key])
    zero = float(ar.get("zero_over_normal", 0.0))
    normal_over_persistence = float(ar["normal_over_persistence"])
    beats_persistence = normal_over_persistence < 1.0
    ar_gate = beats_persistence and cross > causal_min_ratio and zero > causal_min_ratio and temporal_min > causal_min_ratio
    direct_cross_gate = direct_cross > causal_min_ratio
    score = (
        -math.log(max(normal_over_persistence, 1e-9))
        + 0.75 * math.log(max(cross, 1e-9))
        + 0.25 * math.log(max(zero, 1e-9))
        + 0.25 * math.log(max(temporal_min, 1e-9))
    )
    if not ar_gate:
        score -= 10.0
    if not direct_cross_gate:
        score -= 0.5
    if "effect_far_shuffle" in cross_key:
        score += 0.1
    return {
        "label": label,
        "eval_json": str(path),
        "adapter_ckpt": str(resolve_path(ADAPTER_CKPTS.get(label, ""))) if label in ADAPTER_CKPTS else "",
        "score": float(score),
        "ar_gate": bool(ar_gate),
        "direct_cross_gate": bool(direct_cross_gate),
        "ar_normal": float(ar["normal"]),
        "ar_persistence": float(ar["persistence"]),
        "ar_normal_over_persistence": normal_over_persistence,
        "ar_cross_key": cross_key,
        "ar_cross_over_normal": cross,
        "direct_cross_key": direct_cross_key,
        "direct_cross_over_normal": direct_cross,
        "ar_zero_over_normal": zero,
        "ar_temporal_min_over_normal": temporal_min,
    }


def first_ratio_key(row: dict[str, Any], candidates: list[str]) -> str:
    for key in candidates:
        if key in row:
            return key
    raise KeyError(f"none of {candidates} found in {sorted(row)}")


def parse_eval_specs(items: list[str]) -> dict[str, str]:
    specs = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"--eval expects label:path, got {item!r}")
        label, path = item.split(":", 1)
        specs[label.strip()] = path.strip()
    return specs


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Residual Adapter Checkpoint Selection",
        "",
        f"Selected: `{summary['selected']['label']}`",
        "",
        "| Rank | Label | Score | AR Gate | Direct Cross | AR/Persist | Cross | Zero | Temporal Min |",
        "|---:|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(summary["ranked"], start=1):
        lines.append(
            "| {idx} | `{label}` | {score:.3f} | {ar_gate} | {direct_gate} | {persist:.4f} | {cross:.4f} | {zero:.4f} | {temporal:.4f} |".format(
                idx=idx,
                label=row["label"],
                score=float(row["score"]),
                ar_gate="yes" if row["ar_gate"] else "no",
                direct_gate="yes" if row["direct_cross_gate"] else "no",
                persist=float(row["ar_normal_over_persistence"]),
                cross=float(row["ar_cross_over_normal"]),
                zero=float(row["ar_zero_over_normal"]),
                temporal=float(row["ar_temporal_min_over_normal"]),
            )
        )
    lines.extend(["", summary["selection_rule"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
