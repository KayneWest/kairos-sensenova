#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_OUT = "sensenova_drone_agent/data/robotics/hf_action_sources"


@dataclass(frozen=True)
class Candidate:
    repo_id: str
    role: str
    default_weight: float
    notes: str


PROFILES: dict[str, list[Candidate]] = {
    "oxe-compact": [
        Candidate(
            repo_id="IPEC-COMMUNITY/droid_lerobot",
            role="real_robot_action_video",
            default_weight=0.35,
            notes="Compact DROID LeRobot conversion; useful as a real-world manipulation action anchor.",
        ),
        Candidate(
            repo_id="IPEC-COMMUNITY/fractal20220817_data_lerobot",
            role="real_robot_action_video",
            default_weight=0.25,
            notes="Google Robot / Fractal-style manipulation trajectories in compact LeRobot layout.",
        ),
        Candidate(
            repo_id="IPEC-COMMUNITY/bridge_orig_lerobot",
            role="real_robot_action_video",
            default_weight=0.25,
            notes="BridgeData-style WidowX manipulation trajectories in compact LeRobot layout.",
        ),
        Candidate(
            repo_id="IPEC-COMMUNITY/language_table_lerobot",
            role="real_robot_action_tabletop",
            default_weight=0.15,
            notes="Large action/proprio table manipulation source; may not contain videos in this mirror.",
        ),
    ],
    "droid-compact": [
        Candidate(
            repo_id="cadene/droid_1.0.1",
            role="real_robot_action_video",
            default_weight=1.0,
            notes="DROID LeRobot conversion with per-episode parquets and available exterior videos.",
        ),
    ],
    "droid-full": [
        Candidate(
            repo_id="lerobot/droid_1.0.1",
            role="real_robot_action_video_full",
            default_weight=1.0,
            notes="Full LeRobot DROID mirror; much larger because videos are stored in chunked MP4 files.",
        ),
    ],
    "berkeley": [
        Candidate(
            repo_id="lerobot/berkeley_autolab_ur5",
            role="real_robot_action_video",
            default_weight=0.5,
            notes="Berkeley UR5 manipulation data.",
        ),
        Candidate(
            repo_id="lerobot/berkeley_cable_routing",
            role="real_robot_action_video",
            default_weight=0.5,
            notes="Berkeley cable routing manipulation data.",
        ),
    ],
    "libero-sim": [
        Candidate(
            repo_id="openvla/modified_libero_rlds",
            role="sim_robot_action_video",
            default_weight=1.0,
            notes="LIBERO no-noop RLDS data used by OpenVLA/Dream-VLA fine-tuning; simulator, not real robot.",
        ),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and resumably download public action-labeled robot datasets from Hugging Face. "
            "These snapshots are source material for later Dreamer4-format export."
        )
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(PROFILES),
        help="Named corpus profile. Can be repeated. Defaults to oxe-compact.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Additional HF dataset repo_id to include. Can be repeated.",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--max-download-gb",
        type=float,
        default=64.0,
        help="Per-repo safety guard. <=0 disables.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help=(
            "Override snapshot allow patterns. Can be repeated. Defaults to README/meta/data/videos "
            "for LeRobot-style datasets."
        ),
    )
    parser.add_argument(
        "--paired-video-parquets-only",
        action="store_true",
        help=(
            "For per-episode LeRobot mirrors, download metadata, MP4s, and only the parquet episodes "
            "that have a matching MP4. This avoids wasting requests on action-only episodes that cannot "
            "train pixel/latent dynamics."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--repo-retries",
        type=int,
        default=8,
        help="Retry each repo snapshot after transient HF/network failures.",
    )
    parser.add_argument(
        "--retry-sleep-s",
        type=float,
        default=120.0,
        help="Initial sleep between repo retry attempts. Later retries use linear backoff.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub is required") from exc

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidates(args)
    api = HfApi(token=args.token)
    allow_patterns = args.allow_pattern or default_allow_patterns()
    rows = []
    for candidate in candidates:
        local_dir = out / safe_name(candidate.repo_id)
        state_path = local_dir / ".download.json"
        row = audit_repo(
            api=api,
            candidate=candidate,
            revision=args.revision,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            max_download_gb=float(args.max_download_gb),
            paired_video_parquets_only=bool(args.paired_video_parquets_only),
        )
        download_paths = list(row.pop("_download_paths"))
        download_patterns = list(row.pop("_download_patterns"))
        row["download_file_count"] = len(download_paths)
        row["download_pattern_count"] = len(download_patterns)
        rows.append(row)
        print(json.dumps({"remote": row}, indent=2))
        if args.dry_run:
            continue
        if not row["within_size_guard"]:
            raise RuntimeError(
                f"Refusing to download {row['total_human']} from {candidate.repo_id}; "
                "increase --max-download-gb or set it <=0."
            )
        local_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        repo_retries = max(0, int(args.repo_retries))
        for attempt in range(repo_retries + 1):
            try:
                if download_paths:
                    snapshot_path = download_exact_files(
                        hf_hub_download=hf_hub_download,
                        repo_id=candidate.repo_id,
                        revision=args.revision,
                        local_dir=local_dir,
                        paths=download_paths,
                        token=args.token,
                        force_download=bool(args.force_download),
                        max_workers=max(1, int(args.max_workers)),
                    )
                else:
                    snapshot_path = snapshot_download(
                        repo_id=candidate.repo_id,
                        repo_type="dataset",
                        revision=args.revision,
                        local_dir=local_dir,
                        allow_patterns=download_patterns,
                        token=args.token,
                        force_download=bool(args.force_download),
                        max_workers=max(1, int(args.max_workers)),
                    )
                local_files = list_local_files(local_dir)
                payload = {
                    **row,
                    "completed": True,
                    "snapshot_path": str(snapshot_path),
                    "elapsed_s": time.time() - started,
                    "attempt": attempt + 1,
                    "local_file_count": len(local_files),
                    "local_total_bytes": sum(path.stat().st_size for path in local_files),
                    "local_total_human": human_bytes(sum(path.stat().st_size for path in local_files)),
                    "updated_unix_s": time.time(),
                }
                break
            except Exception as exc:
                payload = {
                    **row,
                    "completed": False,
                    "attempt": attempt + 1,
                    "max_attempts": repo_retries + 1,
                    "message": str(exc),
                    "updated_unix_s": time.time(),
                }
                state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                if attempt >= repo_retries:
                    raise
                sleep_s = max(0.0, float(args.retry_sleep_s)) * (attempt + 1)
                print(
                    json.dumps(
                        {
                            "retrying": candidate.repo_id,
                            "attempt": attempt + 1,
                            "max_attempts": repo_retries + 1,
                            "sleep_s": sleep_s,
                            "message": str(exc),
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                time.sleep(sleep_s)
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"downloaded": payload}, indent=2))

    manifest = {
        "phase": "hf_robot_action_source_download",
        "created_unix_s": time.time(),
        "out": str(out),
        "dry_run": bool(args.dry_run),
        "profiles": args.profile or ["oxe-compact"],
        "repos": [candidate.repo_id for candidate in candidates],
        "allow_patterns": allow_patterns,
        "total_remote_bytes": sum(int(row["total_bytes"]) for row in rows),
        "total_remote_human": human_bytes(sum(int(row["total_bytes"]) for row in rows)),
        "sources": rows,
    }
    manifest_path = out / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "total_remote_human": manifest["total_remote_human"]}, indent=2))
    return 0


def collect_candidates(args: argparse.Namespace) -> list[Candidate]:
    profiles = args.profile or ["oxe-compact"]
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for profile in profiles:
        for candidate in PROFILES[profile]:
            if candidate.repo_id in seen:
                continue
            candidates.append(candidate)
            seen.add(candidate.repo_id)
    for repo_id in args.repo:
        if repo_id in seen:
            continue
        candidates.append(
            Candidate(
                repo_id=repo_id,
                role="custom_hf_robot_action_source",
                default_weight=1.0,
                notes="User-specified Hugging Face dataset.",
            )
        )
        seen.add(repo_id)
    return candidates


def audit_repo(
    *,
    api: Any,
    candidate: Candidate,
    revision: str,
    local_dir: Path,
    allow_patterns: list[str],
    max_download_gb: float,
    paired_video_parquets_only: bool,
) -> dict[str, Any]:
    info = api.dataset_info(candidate.repo_id, revision=revision, files_metadata=True)
    siblings = list(info.siblings)
    if paired_video_parquets_only:
        matched_paths = paired_video_parquet_paths(siblings)
        matched = [item for item in siblings if item.rfilename in matched_paths]
        download_patterns = sorted(matched_paths)
        download_paths = sorted(matched_paths)
    else:
        matched = [item for item in siblings if pattern_match_any(item.rfilename, allow_patterns)]
        download_patterns = list(allow_patterns)
        download_paths = []
    total_bytes = sum(int(item.size or 0) for item in matched)
    counts = {
        "files": len(matched),
        "parquet_files": sum(1 for item in matched if item.rfilename.endswith(".parquet")),
        "video_files": sum(1 for item in matched if item.rfilename.endswith(".mp4")),
        "json_files": sum(1 for item in matched if item.rfilename.endswith((".json", ".jsonl"))),
    }
    bytes_by_kind = {
        "parquet": sum(int(item.size or 0) for item in matched if item.rfilename.endswith(".parquet")),
        "mp4": sum(int(item.size or 0) for item in matched if item.rfilename.endswith(".mp4")),
        "json": sum(int(item.size or 0) for item in matched if item.rfilename.endswith((".json", ".jsonl"))),
    }
    largest = sorted(matched, key=lambda item: int(item.size or 0), reverse=True)[:20]
    return {
        **asdict(candidate),
        "revision": revision,
        "local_dir": str(local_dir),
        "allow_patterns": allow_patterns,
        "paired_video_parquets_only": paired_video_parquets_only,
        "repo_file_count": len(siblings),
        "matched_file_count": len(matched),
        "counts": counts,
        "bytes_by_kind": {key: {"bytes": value, "human": human_bytes(value)} for key, value in bytes_by_kind.items()},
        "total_bytes": total_bytes,
        "total_human": human_bytes(total_bytes),
        "max_download_gb": max_download_gb,
        "within_size_guard": within_size_guard(total_bytes, max_download_gb),
        "tags": getattr(info, "tags", None) or [],
        "largest_files": [
            {"path": item.rfilename, "bytes": int(item.size or 0), "human": human_bytes(int(item.size or 0))}
            for item in largest
        ],
        "_download_patterns": download_patterns,
        "_download_paths": download_paths,
    }


def download_exact_files(
    *,
    hf_hub_download: Any,
    repo_id: str,
    revision: str,
    local_dir: Path,
    paths: list[str],
    token: str | None,
    force_download: bool,
    max_workers: int,
) -> str:
    """Download an audited exact file list without snapshot pattern matching.

    Passing thousands of exact paths as snapshot allow patterns can become CPU-bound
    inside fnmatch. Direct file downloads avoid that O(repo_files * patterns) path
    and still resume from the local Hugging Face cache/local_dir.
    """

    total = len(paths)
    if total == 0:
        return str(local_dir)
    started = time.time()

    def fetch(path: str) -> str:
        return hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=path,
            local_dir=local_dir,
            token=token,
            force_download=force_download,
        )

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {executor.submit(fetch, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed to download {repo_id}:{path}: {exc}") from exc
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == total:
                print(
                    json.dumps(
                        {
                            "download_progress": repo_id,
                            "completed": completed,
                            "total": total,
                            "elapsed_s": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
    return str(local_dir)


def default_allow_patterns() -> list[str]:
    return [
        "README.md",
        ".gitattributes",
        "meta/*",
        "data/*",
        "data/**/*",
        "videos/*",
        "videos/**/*",
    ]


def paired_video_parquet_paths(siblings: list[Any]) -> set[str]:
    paths = {str(item.rfilename) for item in siblings}
    out = {
        path
        for path in paths
        if path in {"README.md", ".gitattributes"}
        or path.startswith("meta/")
        or path.endswith(".md")
    }
    for path in paths:
        if not path.startswith("videos/") or not path.endswith(".mp4"):
            continue
        out.add(path)
        parts = path.split("/")
        if len(parts) >= 4 and parts[1].startswith("chunk-") and parts[-1].startswith("episode_"):
            parquet = f"data/{parts[1]}/{Path(parts[-1]).with_suffix('.parquet').name}"
            if parquet in paths:
                out.add(parquet)
    return out


def pattern_match_any(path: str, patterns: list[str]) -> bool:
    import fnmatch

    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def list_local_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [item for item in path.rglob("*") if item.is_file()]


def within_size_guard(num_bytes: int, max_gb: float) -> bool:
    if max_gb <= 0:
        return True
    return num_bytes <= max_gb * (1024**3)


def human_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def safe_name(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum():
            out.append(ch)
        elif ch in {"-", "_", ".", "/"}:
            out.append("_")
    name = "".join(out).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name or "hf_dataset"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
