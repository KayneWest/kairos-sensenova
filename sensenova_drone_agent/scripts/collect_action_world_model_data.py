#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_OUT = "sensenova_drone_agent/data/action_world_model_continue_v1"
DEFAULT_DREAMER_RAW = "sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4"
DEFAULT_DREAMER_SHARDS = "sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4_shards_full"
DEFAULT_SOAR_ZIP = "sensenova_drone_agent/data/robotics/soar/soar-dataset-numpy.zip"
DEFAULT_SOAR_ROOT = "sensenova_drone_agent/data/robotics/soar/dreamer4_soar_native_v2_action_contrast"
DEFAULT_ROBONET_ARCHIVE = "sensenova_drone_agent/data/robotics/robonet/raw/robonet_sampler.tar.gz"
DEFAULT_ROBONET_ROOT = "sensenova_drone_agent/data/robotics/robonet/dreamer4_robonet_sample_64"
DEFAULT_HF_ACTION_EXPORT_ROOT = "sensenova_drone_agent/data/robotics/hf_action_exports"
DEFAULT_HF_ACTION_DATASETS = (
    "droid_lerobot_dreamer4,"
    "fractal20220817_data_lerobot_dreamer4,"
    "bridge_orig_lerobot_dreamer4"
)
DEFAULT_DREAMER4_HF_SPLITS = ("expert", "mixed-small", "mixed-large")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    family: str
    raw: str
    frames: str
    tasks_json: str
    role: str
    action_signal: str
    reward_signal: str
    recommended_weight: float
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect and audit action-labeled data for continued action-conditioned "
            "world-model pretraining. This is a manifest builder around the existing "
            "SOAR, Dreamer4-HF, and RoboNet download/export tools."
        )
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--sources",
        default="dreamer4-hf,soar,robonet",
        help="Comma-separated source families to include: dreamer4-hf,soar,robonet,hf-robot.",
    )
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--report-name", default="report.md")
    parser.add_argument("--download", action="store_true", help="Download missing raw archives/snapshots.")
    parser.add_argument("--export", action="store_true", help="Export missing Dreamer4-format raw/frame datasets.")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess Dreamer4-HF PNG strips into frame shards if missing.")
    parser.add_argument("--force-export", action="store_true", help="Rebuild exported SOAR/RoboNet datasets.")
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild Dreamer4-HF frame shards.")
    parser.add_argument("--dry-run", action="store_true", help="Print intended commands and write an audit-only manifest.")
    parser.add_argument("--scan-bytes", action="store_true", help="Compute source byte sizes. This can take a few seconds.")

    parser.add_argument("--dreamer-raw", default=DEFAULT_DREAMER_RAW)
    parser.add_argument("--dreamer-shards", default=DEFAULT_DREAMER_SHARDS)
    parser.add_argument("--dreamer-hf-id", default="nicklashansen/dreamer4")
    parser.add_argument("--dreamer-hf-splits", default=",".join(DEFAULT_DREAMER4_HF_SPLITS))
    parser.add_argument("--dreamer-hf-max-gb", type=float, default=40.0)
    parser.add_argument("--dreamer-target-size", type=int, default=128)
    parser.add_argument("--dreamer-shard-size", type=int, default=2048)

    parser.add_argument("--soar-zip", default=DEFAULT_SOAR_ZIP)
    parser.add_argument("--soar-root", default=DEFAULT_SOAR_ROOT)
    parser.add_argument("--soar-max-trajectories", type=int, default=1024)
    parser.add_argument("--soar-target-task-count", type=int, default=64)
    parser.add_argument("--soar-max-steps-per-trajectory", type=int, default=128)
    parser.add_argument("--soar-frame-stride", type=int, default=2)
    parser.add_argument("--soar-seed", type=int, default=17)

    parser.add_argument("--robonet-archive", default=DEFAULT_ROBONET_ARCHIVE)
    parser.add_argument("--robonet-root", default=DEFAULT_ROBONET_ROOT)
    parser.add_argument("--robonet-max-trajectories", type=int, default=700)
    parser.add_argument("--robonet-frame-stride", type=int, default=1)
    parser.add_argument("--robonet-seed", type=int, default=0)

    parser.add_argument("--hf-action-export-root", default=DEFAULT_HF_ACTION_EXPORT_ROOT)
    parser.add_argument("--hf-action-datasets", default=DEFAULT_HF_ACTION_DATASETS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = set(parse_csv(args.sources))
    unknown = selected - {"dreamer4-hf", "soar", "robonet", "hf-robot"}
    if unknown:
        raise ValueError(f"Unknown source family: {sorted(unknown)}")

    if "dreamer4-hf" in selected and args.download:
        download_dreamer4_hf(args)
    if "soar" in selected and args.download:
        download_soar(args)
    if "robonet" in selected and args.download:
        download_robonet(args)

    if "dreamer4-hf" in selected and args.preprocess:
        preprocess_dreamer4_hf(args)
    if "soar" in selected and args.export:
        export_soar(args)
    if "robonet" in selected and args.export:
        export_robonet(args)

    specs = build_source_specs(args, selected)
    manifest = build_manifest(args, specs)
    manifest_path = out_dir / args.manifest_name
    report_path = out_dir / args.report_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path.write_text(render_report(manifest), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "report": str(report_path), "ready": manifest["ready"]}, indent=2))
    return 0


def build_source_specs(args: argparse.Namespace, selected: set[str]) -> list[SourceSpec]:
    specs: list[SourceSpec] = []
    dreamer_raw = resolve_path(args.dreamer_raw)
    dreamer_shards = resolve_path(args.dreamer_shards)
    if "dreamer4-hf" in selected:
        for split, weight in (("expert", 0.20), ("mixed-small", 0.20), ("mixed-large", 0.20)):
            specs.append(
                SourceSpec(
                    name=f"dreamer4_hf_{safe_name(split)}",
                    family="dreamer4-hf",
                    raw=str(dreamer_raw / split),
                    frames=str(dreamer_shards / split),
                    tasks_json=str(REPO_ROOT / "dreamer4/tasks.json"),
                    role="control_pretraining",
                    action_signal="dense simulator actions",
                    reward_signal="task rewards from Dreamer4-HF tensors",
                    recommended_weight=weight,
                    notes="Best current source for clear action-conditioned dynamics; keep in the mix as the causality anchor.",
                )
            )
    if "soar" in selected:
        soar_root = resolve_path(args.soar_root)
        specs.append(
            SourceSpec(
                name="soar_robotics_task_balanced",
                family="soar",
                raw=str(soar_root / "raw"),
                frames=str(soar_root / "frames"),
                tasks_json=str(soar_root / "tasks.json"),
                role="robotics_success_failure",
                action_signal="7D end-effector action intervals aggregated to video frames",
                reward_signal="trajectory success/failure labels projected onto steps",
                recommended_weight=0.25,
                notes="Robotics source closest to Dreamer4 paper SOAR setting; sparse reward but real manipulation videos.",
            )
        )
    if "robonet" in selected:
        robonet_root = resolve_path(args.robonet_root)
        specs.append(
            SourceSpec(
                name="robonet_sample_64",
                family="robonet",
                raw=str(robonet_root / "raw"),
                frames=str(robonet_root / "frames"),
                tasks_json=str(robonet_root / "tasks.json"),
                role="robotics_replay",
                action_signal="5D robot action trajectories",
                reward_signal="none; zero reward placeholder",
                recommended_weight=0.15,
                notes="Useful for action-conditioned video dynamics and anti-forgetting, not for reward learning.",
            )
        )
    if "hf-robot" in selected:
        hf_root = resolve_path(args.hf_action_export_root)
        for dataset, weight in (
            ("droid_lerobot_dreamer4", 0.10),
            ("fractal20220817_data_lerobot_dreamer4", 0.10),
            ("bridge_orig_lerobot_dreamer4", 0.10),
        ):
            if dataset not in set(parse_csv(args.hf_action_datasets)):
                continue
            root = hf_root / dataset
            specs.append(
                SourceSpec(
                    name=f"hf_robot_{safe_name(dataset)}",
                    family="hf-robot",
                    raw=str(root / "raw"),
                    frames=str(root / "frames"),
                    tasks_json=str(root / "tasks.json"),
                    role="robotics_action_video_scaling",
                    action_signal="real robot low-level actions paired with MP4 observations",
                    reward_signal="none; zero reward placeholder",
                    recommended_weight=weight,
                    notes="LeRobot/OXE-style paired video-action corpus for continued action-conditioned dynamics.",
                )
            )
    return specs


def build_manifest(args: argparse.Namespace, specs: list[SourceSpec]) -> dict[str, Any]:
    audited = [audit_source(spec, scan_bytes=bool(args.scan_bytes)) for spec in specs]
    ready_sources = [row for row in audited if row["ready"]]
    raw_dirs = [row["raw"] for row in ready_sources]
    frame_dirs = [row["frames"] for row in ready_sources]
    weights = {row["name"]: row["recommended_weight"] for row in ready_sources}
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {key: value / total_weight for key, value in weights.items()}
    train_env = {
        "DREAMER_RAW": str(resolve_path(args.dreamer_raw)),
        "DREAMER_SHARDS": str(resolve_path(args.dreamer_shards)),
        "SOAR_ROOT": str(resolve_path(args.soar_root)),
        "ROBONET_ROOT": str(resolve_path(args.robonet_root)),
        "HF_ACTION_EXPORT_ROOT": str(resolve_path(args.hf_action_export_root)),
        "HF_ACTION_DATASETS": args.hf_action_datasets,
        "ACTION_DIM": "49",
        "ACTION_FEATURES": "current,prev,delta,mean4,norm",
        "SKIP_TOKENIZER": "1",
        "SELF_FRACTION": "0.25",
        "ACTION_CONTRAST_NEGATIVE_MODES": "shuffle,zero,time_shift",
        "ACTION_CONTRAST_MIN_ACTION_NORM": "0.001",
        "ACTION_CONTRAST_TEMPORAL_START": "1",
    }
    return {
        "phase": "continued_action_conditioned_world_model_data",
        "created_unix_s": time.time(),
        "repo_root": str(REPO_ROOT),
        "ready": len(ready_sources) == len(audited) and len(ready_sources) > 0,
        "sources_requested": parse_csv(args.sources),
        "sources": audited,
        "ready_source_count": len(ready_sources),
        "source_count": len(audited),
        "total_usable_task_count": sum(int(row["usable_task_count"]) for row in ready_sources),
        "total_raw_task_count": sum(int(row["raw_task_count"]) for row in ready_sources),
        "total_frame_task_count": sum(int(row["frame_task_count"]) for row in ready_sources),
        "recommended_sampling_weights": weights,
        "train_roots": {
            "raw_dirs": raw_dirs,
            "frame_dirs": frame_dirs,
            "tasks_json_strategy": "merge per-source tasks into the all-data native training manifest before launch",
        },
        "train_env": train_env,
        "next_command": (
            "RUN_ID=continued_action_wm_v1 BASE_DYNAMICS_CKPT=/workspace/sensenova_drone_agent/output/"
            "dreamer4_all_data_native_causal_ident_gate20k_continue_25k_v1/dynamics_ckpts/latest.pt "
            "SKIP_TOKENIZER=1 DYNAMICS_STEPS=50000 "
            "./sensenova_drone_agent/scripts/experiments/launch_all_data_native_dreamer4.sh"
        ),
        "claim_gate": {
            "normal_vs_shuffle_zero_time_shift": "strict ratios > 1.02 on direct and autoregressive eval",
            "normal_vs_persistence": "normal autoregressive latent MSE should beat persistence",
            "reward_counterfactual": "true-action reward return should exceed zero/shuffle on positive windows",
            "imagination": "selected policy should have policy_minus_bc > 0 and causal_policy_gain > 0 with nontrivial margin",
        },
    }


def audit_source(spec: SourceSpec, *, scan_bytes: bool) -> dict[str, Any]:
    raw = Path(spec.raw)
    frames = Path(spec.frames)
    tasks_json = Path(spec.tasks_json)
    raw_tasks = {path.stem for path in raw.glob("*.pt")} if raw.exists() else set()
    frame_tasks = set()
    frame_shard_count = 0
    if frames.exists():
        for task_dir in frames.iterdir():
            if not task_dir.is_dir():
                continue
            shards = list(task_dir.glob("*shard*.pt")) + list(task_dir.glob("frames_shard_*.pt"))
            if shards:
                frame_tasks.add(task_dir.name)
                frame_shard_count += len(shards)
    usable_tasks = sorted(raw_tasks & frame_tasks)
    summary_path = raw.parent / "summary.json" if raw.name == "raw" else raw.parent / "summary.json"
    if spec.family == "dreamer4-hf":
        summary_path = raw.parent / ".download.json"
    summary = load_json_if_exists(summary_path)
    tasks_meta = load_json_if_exists(tasks_json)
    return {
        **asdict(spec),
        "raw": str(raw),
        "frames": str(frames),
        "tasks_json": str(tasks_json),
        "raw_exists": raw.exists(),
        "frames_exists": frames.exists(),
        "tasks_json_exists": tasks_json.exists(),
        "raw_task_count": len(raw_tasks),
        "frame_task_count": len(frame_tasks),
        "usable_task_count": len(usable_tasks),
        "frame_shard_count": frame_shard_count,
        "usable_task_preview": usable_tasks[:20],
        "ready": bool(raw_tasks) and bool(frame_tasks) and bool(usable_tasks),
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "summary_preview": summarize_existing_summary(summary),
        "tasks_meta_count": len(tasks_meta) if isinstance(tasks_meta, dict) else None,
        "bytes": tree_bytes(raw.parent if spec.family == "dreamer4-hf" else raw.parent.parent) if scan_bytes else None,
    }


def summarize_existing_summary(summary: Any) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    keys = [
        "source",
        "dataset_id",
        "completed",
        "local_total_human",
        "selected_count",
        "exported_trajectories",
        "trajectory_count",
        "archive_trajectories",
        "selected_trajectories",
        "frame_stride",
        "reward_mode",
        "action_aggregation",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Action-Conditioned World-Model Data Collection",
        "",
        f"Ready: `{manifest['ready']}`",
        "",
        "This corpus is for continued world-model pretraining with action tokens, not policy/reward/value midtraining.",
        "",
        "## Sources",
        "",
    ]
    for row in manifest["sources"]:
        status = "ready" if row["ready"] else "missing"
        lines.extend(
            [
                f"### {row['name']}",
                "",
                f"- status: `{status}`",
                f"- role: `{row['role']}`",
                f"- raw tasks: `{row['raw_task_count']}`",
                f"- frame tasks: `{row['frame_task_count']}`",
                f"- usable tasks: `{row['usable_task_count']}`",
                f"- frame shards: `{row['frame_shard_count']}`",
                f"- action signal: {row['action_signal']}",
                f"- reward signal: {row['reward_signal']}",
                f"- notes: {row['notes']}",
                f"- raw: `{row['raw']}`",
                f"- frames: `{row['frames']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Suggested Training Environment",
            "",
            "```bash",
        ]
    )
    for key, value in manifest["train_env"].items():
        lines.append(f"export {key}={shell_quote(str(value))}")
    lines.extend(
        [
            "```",
            "",
            "## Next Command",
            "",
            "```bash",
            manifest["next_command"],
            "```",
            "",
            "## Gate",
            "",
        ]
    )
    for key, value in manifest["claim_gate"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def download_dreamer4_hf(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "sensenova_drone_agent/scripts/download_soar_numpy_dataset.py",
        "--source",
        "dreamer4-hf",
        "--hf-dataset-id",
        args.dreamer_hf_id,
        "--hf-dir",
        args.dreamer_raw,
        "--hf-splits",
        args.dreamer_hf_splits,
        "--max-download-gb",
        str(args.dreamer_hf_max_gb),
    ]
    run_cmd(cmd, dry_run=args.dry_run)


def download_soar(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "sensenova_drone_agent/scripts/download_soar_numpy_dataset.py",
        "--source",
        "soar",
        "--dest",
        args.soar_zip,
    ]
    run_cmd(cmd, dry_run=args.dry_run)


def download_robonet(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "sensenova_drone_agent/scripts/download_soar_numpy_dataset.py",
        "--source",
        "robonet-gdrive",
        "--robonet-archive",
        args.robonet_archive,
    ]
    run_cmd(cmd, dry_run=args.dry_run)


def preprocess_dreamer4_hf(args: argparse.Namespace) -> None:
    raw_root = resolve_path(args.dreamer_raw)
    shard_root = resolve_path(args.dreamer_shards)
    for split in parse_csv(args.dreamer_hf_splits):
        raw_split = raw_root / split
        shard_split = shard_root / split
        if not raw_split.exists():
            continue
        if shard_split.exists() and any(shard_split.rglob("*shard*.pt")) and not args.force_preprocess:
            continue
        if args.force_preprocess and shard_split.exists() and not args.dry_run:
            shutil.rmtree(shard_split)
        cmd = [
            sys.executable,
            "dreamer4/dreamer4/preprocess_dataset.py",
            "--filedir",
            str(raw_split),
            "--outdir",
            str(shard_split),
            "--target-size",
            str(args.dreamer_target_size),
            "--shard-size",
            str(args.dreamer_shard_size),
            "--tasks-from-data",
        ]
        run_cmd(cmd, dry_run=args.dry_run)


def export_soar(args: argparse.Namespace) -> None:
    out = resolve_path(args.soar_root)
    if out.exists() and any((out / "raw").glob("*.pt")) and any((out / "frames").rglob("*shard*.pt")) and not args.force_export:
        return
    if args.force_export and out.exists() and not args.dry_run:
        shutil.rmtree(out)
    cmd = [
        sys.executable,
        "sensenova_drone_agent/scripts/export_soar_dreamer4_dataset.py",
        "--zip",
        args.soar_zip,
        "--out",
        args.soar_root,
        "--max-trajectories",
        str(args.soar_max_trajectories),
        "--target-task-count",
        str(args.soar_target_task_count),
        "--min-trajectories-per-task",
        "4",
        "--max-trajectories-per-task",
        "16",
        "--require-both-outcomes-per-task",
        "--max-steps-per-trajectory",
        str(args.soar_max_steps_per_trajectory),
        "--frame-stride",
        str(args.soar_frame_stride),
        "--frame-size",
        "128",
        "--shard-size",
        "2048",
        "--selection-mode",
        "task_balanced",
        "--action-aggregation",
        "sum",
        "--reward-mode",
        "trajectory_success",
        "--task-name-mode",
        "language",
        "--seed",
        str(args.soar_seed),
    ]
    run_cmd(cmd, dry_run=args.dry_run)


def export_robonet(args: argparse.Namespace) -> None:
    out = resolve_path(args.robonet_root)
    if out.exists() and any((out / "raw").glob("*.pt")) and any((out / "frames").rglob("*shard*.pt")) and not args.force_export:
        return
    if args.force_export and out.exists() and not args.dry_run:
        shutil.rmtree(out)
    cmd = [
        sys.executable,
        "sensenova_drone_agent/scripts/export_robonet_dreamer4_dataset.py",
        "--source",
        "tar",
        "--tar",
        args.robonet_archive,
        "--out",
        args.robonet_root,
        "--max-trajectories",
        str(args.robonet_max_trajectories),
        "--frame-size",
        "128",
        "--frame-stride",
        str(args.robonet_frame_stride),
        "--task-mode",
        "robot_name",
        "--reward-mode",
        "zero",
        "--seed",
        str(args.robonet_seed),
    ]
    run_cmd(cmd, dry_run=args.dry_run)


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(shell_quote(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_name(value: str) -> str:
    return value.replace("-", "_").replace("/", "_")


def load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def tree_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=/:.,")
    if all(ch in safe for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
