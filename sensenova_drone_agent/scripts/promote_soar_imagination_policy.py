#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a validated SOAR/DROID imagination policy into a stable artifact manifest.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--breakdown-json", default="")
    parser.add_argument("--repeatability-json", default="")
    parser.add_argument("--out-dir", default="sensenova_drone_agent/output/controllable_soar_imagination_policy_v1")
    parser.add_argument("--artifact-name", default="controllable_soar_imagination_policy_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    best_path = run_dir / "best_imagination_selection.json"
    policy_ckpt = run_dir / "after_imagination.pt"
    bc_ckpt = run_dir / "bc_prior.pt"
    config_path = run_dir / "config.json"
    for path in [summary_path, best_path, policy_ckpt, bc_ckpt, config_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = json.loads(best_path.read_text(encoding="utf-8"))
    config = summary["config"]
    after = summary["after_imagination"]
    breakdown_path = resolve_path(args.breakdown_json) if args.breakdown_json else None
    repeatability_path = resolve_path(args.repeatability_json) if args.repeatability_json else None

    manifest: dict[str, Any] = {
        "phase": "controllable_soar_imagination_policy_artifact",
        "artifact": args.artifact_name,
        "created_from": {
            "run_dir": dual_path(run_dir),
            "summary": dual_path(summary_path),
            "report": dual_path(report_path) if report_path.exists() else None,
            "best_selection": dual_path(best_path),
            "breakdown": dual_path(breakdown_path) if breakdown_path and breakdown_path.exists() else None,
            "repeatability": dual_path(repeatability_path) if repeatability_path and repeatability_path.exists() else None,
        },
        "components": {
            "policy_ckpt": dual_path(policy_ckpt),
            "bc_prior_ckpt": dual_path(bc_ckpt),
            "tokenizer_ckpt": dual_path(resolve_path(config["tokenizer_ckpt"])),
            "dynamics_ckpt": dual_path(resolve_path(config["dynamics_ckpt"])),
            "residual_adapter_ckpt": dual_path(resolve_path(config["residual_adapter_ckpt"])),
            "tasks_json": dual_path(resolve_path(config["tasks_json"])),
        },
        "runtime": {
            "entrypoint": "sensenova_drone_agent/scripts/train_native_dreamer4_imagination.py",
            "agent_class": "AgentHeads",
            "policy_action_source": config.get("policy_action_source"),
            "action_dim": config.get("action_dim"),
            "raw_action_dim": config.get("raw_action_dim"),
            "action_features": config.get("action_features"),
            "action_chunk_len": config.get("action_chunk_len"),
            "action_frame_offset": config.get("action_frame_offset"),
            "ctx_len": config.get("ctx_len"),
            "imagination_horizon": config.get("imagination_horizon"),
        },
        "selection": {
            "metric": best.get("metric"),
            "selected_update": best.get("selected_update"),
            "min_selection_update": best.get("min_selection_update", config.get("min_imagination_selection_update")),
            "metric_value": best.get("metric_value"),
        },
        "heldout_metrics": {
            "policy_minus_bc": after.get("policy_minus_bc"),
            "policy_minus_zero": after.get("policy_minus_zero"),
            "policy_minus_dyn_zero": after.get("policy_minus_dyn_zero"),
            "policy_minus_dyn_shuffle": after.get("policy_minus_dyn_shuffle"),
            "causal_policy_gain": after.get("causal_policy_gain"),
            "policy_prior_mse": after.get("policy_prior_mse"),
        },
        "claim_boundary": (
            "This artifact is a learned-simulator imagination policy over SOAR/DROID latent rollouts. "
            "It supports offline simulator analyses only and does not prove real-world robot or drone control."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(manifest, out_dir / "README.md")
    print(json.dumps({"out_dir": str(out_dir), "manifest": str(out_dir / "manifest.json")}, indent=2))
    return 0


def write_readme(manifest: dict[str, Any], path: Path) -> None:
    metrics = manifest["heldout_metrics"]
    lines = [
        f"# {manifest['artifact']}",
        "",
        "Validated SOAR/DROID learned-simulator imagination policy.",
        "",
        "## Selection",
        "",
        f"- Metric: `{manifest['selection']['metric']}`",
        f"- Selected update: `{manifest['selection']['selected_update']}`",
        f"- Metric value: `{manifest['selection']['metric_value']}`",
        "",
        "## Held-Out Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: `{float(value):+.6f}`" if isinstance(value, (int, float)) else f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Components",
            "",
        ]
    )
    for key, value in manifest["components"].items():
        lines.append(f"- `{key}`: `{value['host']}`")
    lines.extend(["", manifest["claim_boundary"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    text = str(value)
    if text.startswith("/workspace/"):
        return (REPO_ROOT / text[len("/workspace/") :]).resolve()
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def dual_path(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    path = path.resolve()
    try:
        rel = path.relative_to(REPO_ROOT)
        workspace = Path("/workspace") / rel
    except ValueError:
        workspace = path
    return {"host": str(path), "workspace": str(workspace)}


if __name__ == "__main__":
    raise SystemExit(main())
