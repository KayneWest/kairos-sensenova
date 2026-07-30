#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


DEFAULT_SOURCE_RUN = (
    "sensenova_drone_agent/output/"
    "dreamer4_all_data_native_continued_action_wm_hf_robot_source_weighted_m1_50k_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a selected residual adapter into a stable simulator artifact.")
    parser.add_argument(
        "--selection-json",
        default="sensenova_drone_agent/output/residual_action_adapter_selection_v1/selection_summary.json",
    )
    parser.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--out-dir", default="sensenova_drone_agent/output/controllable_soar_simulator_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selection_path = resolve_path(args.selection_json)
    source_run = resolve_path(args.source_run)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["selected"]
    adapter_ckpt = resolve_path(selected["adapter_ckpt"])
    eval_json = resolve_path(selected["eval_json"])
    eval_payload = json.loads(eval_json.read_text(encoding="utf-8"))
    eval_config = dict(eval_payload.get("config", {}))

    source_manifest = source_run / "all_data_manifest.json"
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_names = set(eval_payload.get("sources", []))
    sources = [
        normalize_source_paths(source)
        for source in source_payload.get("sources", [])
        if not source_names or source.get("name") in source_names
    ]

    tokenizer_ckpt = source_run / "tokenizer_ckpts" / "latest.pt"
    dynamics_ckpt = source_run / "dynamics_ckpts" / "final_step_0275000.pt"
    tasks_json = source_run / "tasks_all_data.json"

    manifest: dict[str, Any] = {
        "phase": "controllable_soar_simulator_artifact",
        "artifact": "controllable_soar_simulator_v1",
        "selected_label": selected["label"],
        "created_from": {
            "selection_json": dual_path(selection_path),
            "source_run": dual_path(source_run),
            "source_manifest": dual_path(source_manifest),
            "eval_json": dual_path(eval_json),
        },
        "components": {
            "tokenizer_ckpt": dual_path(tokenizer_ckpt),
            "dynamics_ckpt": dual_path(dynamics_ckpt),
            "residual_adapter_ckpt": dual_path(adapter_ckpt),
            "tasks_json": dual_path(tasks_json),
        },
        "runtime": {
            "loader": "sensenova_drone_agent.scripts.residual_adapter_runtime.wrap_dynamics_with_residual_adapter",
            "training_entrypoint": "sensenova_drone_agent/scripts/train_native_dreamer4_imagination.py",
            "action_frame_offset": int(eval_config.get("action_frame_offset", -1)),
            "action_dim": int(eval_config.get("action_dim", 49)),
            "action_features": str(eval_config.get("action_features", "current,prev,delta,mean4,norm")),
            "residual_scale": float(eval_config.get("residual_scale", 1.0)),
        },
        "data_sources": sources,
        "selection_metrics": selected,
        "heldout_eval": eval_payload.get("metrics", {}),
        "claim_boundary": (
            "Promoted artifact is a frozen learned SOAR/DROID latent simulator with a residual action adapter. "
            "It supports imagination-training experiments, but it does not prove real-world control."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "loader_config.json").write_text(json.dumps(manifest["components"] | manifest["runtime"], indent=2), encoding="utf-8")
    write_readme(manifest, out_dir / "README.md")
    print(json.dumps({"out_dir": str(out_dir), "selected": selected["label"], "manifest": str(out_dir / "manifest.json")}, indent=2))
    return 0


def normalize_source_paths(source: dict[str, Any]) -> dict[str, Any]:
    out = dict(source)
    for key in ("raw", "frames", "tasks_json"):
        if key in out:
            out[key] = dual_path(resolve_path(out[key]))
    return out


def write_readme(manifest: dict[str, Any], path: Path) -> None:
    metrics = manifest["selection_metrics"]
    lines = [
        "# Controllable SOAR Simulator v1",
        "",
        f"Selected adapter: `{manifest['selected_label']}`",
        "",
        "This artifact freezes the tokenizer and continued Dreamer4-style dynamics, then wraps the dynamics with the selected residual action adapter.",
        "",
        "## Causal Eval Summary",
        "",
        f"- AR normal/persistence: `{metrics['ar_normal_over_persistence']:.4f}`",
        f"- AR {metrics['ar_cross_key']}: `{metrics['ar_cross_over_normal']:.4f}`",
        f"- Direct {metrics['direct_cross_key']}: `{metrics['direct_cross_over_normal']:.4f}`",
        f"- AR zero/normal: `{metrics['ar_zero_over_normal']:.4f}`",
        f"- AR temporal-min/normal: `{metrics['ar_temporal_min_over_normal']:.4f}`",
        "",
        "Known limitation: direct one-step cross-trajectory action identity is still weak; use this as the best current retrofit simulator, not as a native Dreamer4 dynamics proof.",
        "",
        "## Primary Components",
        "",
    ]
    for key, value in manifest["components"].items():
        lines.append(f"- `{key}`: `{value['host']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if str(path).startswith("/workspace/"):
        return (REPO_ROOT / str(path)[len("/workspace/") :]).resolve()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def dual_path(path: Path) -> dict[str, str]:
    path = path.resolve()
    try:
        rel = path.relative_to(REPO_ROOT)
        workspace = Path("/workspace") / rel
    except ValueError:
        workspace = path
    return {"host": str(path), "workspace": str(workspace)}


if __name__ == "__main__":
    raise SystemExit(main())
