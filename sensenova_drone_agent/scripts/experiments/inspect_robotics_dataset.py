#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent


FRAME_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ARRAY_EXTS = {".npz", ".npy", ".pt", ".pth", ".h5", ".hdf5", ".tfrecord"}
TABLE_EXTS = {".json", ".jsonl", ".parquet", ".csv", ".yaml", ".yml"}
METADATA_NAMES = {
    "readme.md",
    "dataset_info.json",
    "info.json",
    "modality.json",
    "stats.json",
    "episodes.jsonl",
    "tasks.json",
}


@dataclass
class FileEntry:
    path: str
    size: int | None
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a robotics/world-model dataset layout before downloading or converting it. "
            "This is intentionally schema-first: it lists files and checks whether frames, actions, "
            "rewards, and episode boundaries appear to be available."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-id", help="Hugging Face dataset repo id, e.g. nicklashansen/dreamer4")
    source.add_argument("--local-dir", help="Already downloaded dataset directory")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", default="sensenova_drone_agent/logs/robotics_data/inspection")
    parser.add_argument("--download-sample", action="store_true", help="Download a small set of likely schema/sample files.")
    parser.add_argument("--sample-dir", default="", help="Where sample files should be downloaded.")
    parser.add_argument("--max-sample-files", type=int, default=32)
    parser.add_argument("--max-sample-file-mb", type=float, default=64.0)
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Optional fnmatch pattern to restrict sample downloads. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_id:
        entries = list_hf_dataset(args.dataset_id, args.revision)
        source_name = args.dataset_id.replace("/", "__")
        source_kind = "huggingface"
    else:
        local_dir = resolve_path(args.local_dir)
        entries = list_local_dataset(local_dir)
        source_name = local_dir.name
        source_kind = "local"

    summary = summarize_entries(entries)
    diagnosis = diagnose_layout(entries)
    payload = {
        "source_kind": source_kind,
        "dataset_id": args.dataset_id,
        "local_dir": args.local_dir,
        "revision": args.revision,
        "file_count": len(entries),
        "summary": summary,
        "diagnosis": diagnosis,
        "entries": [asdict(entry) for entry in entries],
    }

    safe_name = safe_slug(source_name)
    manifest_path = out_dir / f"{safe_name}_manifest.json"
    report_path = out_dir / f"{safe_name}_report.md"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")

    downloaded: list[str] = []
    if args.dataset_id and args.download_sample:
        sample_dir = resolve_path(args.sample_dir) if args.sample_dir else out_dir / f"{safe_name}_sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        downloaded = download_hf_sample(
            dataset_id=args.dataset_id,
            revision=args.revision,
            entries=entries,
            sample_dir=sample_dir,
            max_files=args.max_sample_files,
            max_file_bytes=int(args.max_sample_file_mb * 1024 * 1024),
            allow_patterns=args.allow_pattern,
        )
        (out_dir / f"{safe_name}_sample_files.json").write_text(
            json.dumps({"sample_dir": str(sample_dir), "files": downloaded}, indent=2),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "report": str(report_path),
                "file_count": len(entries),
                "detected_layouts": diagnosis["detected_layouts"],
                "readiness": diagnosis["readiness"],
                "downloaded_sample_files": downloaded,
            },
            indent=2,
        )
    )
    return 0


def list_hf_dataset(dataset_id: str, revision: str) -> list[FileEntry]:
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub is required for --dataset-id") from exc

    api = HfApi()
    entries: list[FileEntry] = []
    for item in api.list_repo_tree(
        dataset_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        expand=True,
    ):
        if item.__class__.__name__ != "RepoFile":
            continue
        path = str(item.path)
        entries.append(FileEntry(path=path, size=getattr(item, "size", None), kind=classify_path(path)))
    return sorted(entries, key=lambda entry: entry.path)


def list_local_dataset(local_dir: Path) -> list[FileEntry]:
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)
    entries: list[FileEntry] = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        entries.append(FileEntry(path=rel, size=path.stat().st_size, kind=classify_path(rel)))
    return entries


def classify_path(path: str) -> str:
    ext = Path(path).suffix.lower()
    name = Path(path).name.lower()
    if name in METADATA_NAMES:
        return "metadata"
    if ext in FRAME_EXTS:
        return "frame"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in ARRAY_EXTS:
        return "array_or_tensor"
    if ext in TABLE_EXTS:
        return "table_or_metadata"
    return "other"


def summarize_entries(entries: list[FileEntry]) -> dict[str, Any]:
    by_kind = Counter(entry.kind for entry in entries)
    by_ext = Counter(Path(entry.path).suffix.lower() or "<none>" for entry in entries)
    by_root = Counter(entry.path.split("/", 1)[0] for entry in entries)
    total_bytes = sum(entry.size or 0 for entry in entries)
    known_size_count = sum(1 for entry in entries if entry.size is not None)
    return {
        "by_kind": dict(by_kind.most_common()),
        "by_extension": dict(by_ext.most_common(24)),
        "top_level": dict(by_root.most_common(40)),
        "known_size_count": known_size_count,
        "known_total_gb": round(total_bytes / (1024**3), 3),
        "largest_files": [
            {"path": entry.path, "size_mb": round((entry.size or 0) / (1024**2), 3)}
            for entry in sorted(entries, key=lambda item: item.size or 0, reverse=True)[:20]
        ],
    }


def diagnose_layout(entries: list[FileEntry]) -> dict[str, Any]:
    paths = [entry.path for entry in entries]
    lower = [path.lower() for path in paths]
    roots = {path.split("/", 1)[0] for path in paths if "/" in path}

    detected: list[str] = []
    if {"expert", "mixed-small", "mixed-large"} & roots and any(path.endswith(".pt") for path in lower):
        detected.append("dreamer4_filetree")
    if any(path.startswith("data/") and path.endswith(".parquet") for path in lower) or any(
        path.startswith("videos/") for path in lower
    ):
        detected.append("lerobot_like")
    if any(path.endswith(".tfrecord") for path in lower):
        detected.append("rlds_like")
    if any(path.endswith(".h5") or path.endswith(".hdf5") for path in lower):
        detected.append("hdf5_robotics")
    if not detected:
        detected.append("unknown")

    kind_counts = Counter(entry.kind for entry in entries)
    has_visuals = bool(kind_counts["frame"] or kind_counts["video"])
    has_actions = any(
        token in path
        for path in lower
        for token in ("action", "actions", ".pt", ".npz", ".h5", ".hdf5", ".parquet", ".tfrecord")
    )
    has_rewards = any("reward" in path or "success" in path or path.endswith(".pt") for path in lower)
    has_episode_metadata = any(
        token in path for path in lower for token in ("episode", "episodes", "traj", "trajectory", "meta/")
    )

    readiness = "not_ready"
    if has_visuals and has_actions and has_episode_metadata:
        readiness = "schema_mapping_needed"
    if "dreamer4_filetree" in detected:
        readiness = "dreamer4_preprocess_ready"
    if "lerobot_like" in detected:
        readiness = "lerobot_loader_needed"

    return {
        "detected_layouts": detected,
        "signals": {
            "has_visual_frames_or_video": has_visuals,
            "has_action_candidates": has_actions,
            "has_reward_or_success_candidates": has_rewards,
            "has_episode_boundary_candidates": has_episode_metadata,
        },
        "readiness": readiness,
        "likely_frame_examples": [entry.path for entry in entries if entry.kind in {"frame", "video"}][:12],
        "likely_action_metadata_examples": [
            entry.path
            for entry in entries
            if entry.kind in {"array_or_tensor", "table_or_metadata", "metadata"}
        ][:24],
    }


def download_hf_sample(
    *,
    dataset_id: str,
    revision: str,
    entries: list[FileEntry],
    sample_dir: Path,
    max_files: int,
    max_file_bytes: int,
    allow_patterns: list[str],
) -> list[str]:
    from huggingface_hub import hf_hub_download

    selected = select_sample_files(entries, max_files=max_files, max_file_bytes=max_file_bytes, allow_patterns=allow_patterns)
    downloaded: list[str] = []
    for entry in selected:
        try:
            local = hf_hub_download(
                repo_id=dataset_id,
                repo_type="dataset",
                revision=revision,
                filename=entry.path,
                local_dir=sample_dir,
            )
        except Exception as exc:  # keep the report useful if one large/symlinked file fails
            downloaded.append(f"FAILED {entry.path}: {exc}")
            continue
        downloaded.append(str(Path(local).relative_to(sample_dir)))
        maybe_write_image_probe(Path(local))
    return downloaded


def select_sample_files(
    entries: list[FileEntry],
    *,
    max_files: int,
    max_file_bytes: int,
    allow_patterns: list[str],
) -> list[FileEntry]:
    def allowed(entry: FileEntry) -> bool:
        if entry.size is not None and entry.size > max_file_bytes:
            return False
        if allow_patterns and not any(fnmatch.fnmatch(entry.path, pattern) for pattern in allow_patterns):
            return False
        return True

    buckets: dict[str, list[FileEntry]] = defaultdict(list)
    for entry in entries:
        if allowed(entry):
            buckets[entry.kind].append(entry)

    selected: list[FileEntry] = []
    for kind in ("metadata", "table_or_metadata", "array_or_tensor", "frame", "video", "other"):
        for entry in buckets.get(kind, [])[: max(1, max_files // 4)]:
            if entry not in selected:
                selected.append(entry)
            if len(selected) >= max_files:
                return selected
    return selected[:max_files]


def maybe_write_image_probe(path: Path) -> None:
    if path.suffix.lower() not in FRAME_EXTS:
        return
    try:
        with Image.open(path) as image:
            probe = {
                "path": str(path),
                "format": image.format,
                "mode": image.mode,
                "size": list(image.size),
            }
    except Exception as exc:
        probe = {"path": str(path), "error": str(exc)}
    path.with_suffix(path.suffix + ".probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    diagnosis = payload["diagnosis"]
    signals = diagnosis["signals"]
    lines = [
        "# Robotics Dataset Inspection",
        "",
        f"- Source kind: `{payload['source_kind']}`",
        f"- Dataset id: `{payload.get('dataset_id')}`",
        f"- Local dir: `{payload.get('local_dir')}`",
        f"- Revision: `{payload.get('revision')}`",
        f"- File count: `{payload['file_count']}`",
        f"- Known total size: `{summary['known_total_gb']} GB`",
        f"- Detected layout: `{', '.join(diagnosis['detected_layouts'])}`",
        f"- Readiness: `{diagnosis['readiness']}`",
        "",
        "## Required Signals",
        "",
        f"- Visual frames/video: `{signals['has_visual_frames_or_video']}`",
        f"- Action candidates: `{signals['has_action_candidates']}`",
        f"- Reward/success candidates: `{signals['has_reward_or_success_candidates']}`",
        f"- Episode boundary candidates: `{signals['has_episode_boundary_candidates']}`",
        "",
        "## Top-Level Layout",
        "",
    ]
    for name, count in summary["top_level"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## File Kinds", ""])
    for name, count in summary["by_kind"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Likely Visual Examples", ""])
    for path in diagnosis["likely_frame_examples"]:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Likely Action/Metadata Examples", ""])
    for path in diagnosis["likely_action_metadata_examples"]:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Largest Files", ""])
    for item in summary["largest_files"]:
        lines.append(f"- `{item['path']}`: {item['size_mb']} MB")
    lines.extend(
        [
            "",
            "## Next Mapping",
            "",
            "To use this dataset for our world-model control experiments, map it into sequences:",
            "",
            "```text",
            "obs[t] image or video frame",
            "action[t] continuous or discrete robot action",
            "reward[t] scalar reward or success/hindsight score",
            "episode[t] trajectory id",
            "step[t] timestep within trajectory",
            "```",
            "",
            "The first acceptance test is not policy success. It is whether action-conditioned prediction beats action-shuffled prediction.",
        ]
    )
    return "\n".join(lines) + "\n"


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "dataset"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
