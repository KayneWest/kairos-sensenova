#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_URL = "https://rail.eecs.berkeley.edu/datasets/soar_release/soar-dataset-numpy.zip"
DEFAULT_DEST = "sensenova_drone_agent/data/robotics/soar/soar-dataset-numpy.zip"
DEFAULT_DREAMER4_HF_ID = "nicklashansen/dreamer4"
DEFAULT_DREAMER4_HF_DIR = "sensenova_drone_agent/data/dreamer4/nicklashansen_dreamer4"
DEFAULT_ROBONET_TFDS_NAME = "robonet/robonet_sample_64"
DEFAULT_ROBONET_TFDS_DIR = "sensenova_drone_agent/data/robotics/robonet/tfds"
DEFAULT_ROBONET_GDRIVE_ID = "1YX2TgT8IKSn9V4wGCwdzbRnS53yicV2P"
DEFAULT_ROBONET_ARCHIVE = "sensenova_drone_agent/data/robotics/robonet/raw/robonet_sampler.tar.gz"
REQUEST_HEADERS = {
    "Accept-Encoding": "identity",
    "User-Agent": "sensenova-drone-agent-soar-downloader/1.0",
}


@dataclass
class RemoteMetadata:
    url: str
    final_url: str
    status_code: int
    content_length: int | None
    accept_ranges: str | None
    etag: str | None
    last_modified: str | None
    content_type: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable downloader for action-labeled world-model datasets. "
            "Default source is the SOAR-Data numpy release; use --source dreamer4-hf "
            "for the Hugging Face dataset used by the local Dreamer4 reproduction, or "
            "--source robonet-tfds for the TFDS RoboNet sample."
        )
    )
    parser.add_argument("--source", choices=["soar", "dreamer4-hf", "robonet-tfds", "robonet-gdrive"], default="soar")

    # SOAR ZIP source.
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--dest", default=DEFAULT_DEST)

    # Dreamer4 Hugging Face dataset source.
    parser.add_argument("--hf-dataset-id", default=DEFAULT_DREAMER4_HF_ID)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-dir", default=DEFAULT_DREAMER4_HF_DIR)
    parser.add_argument(
        "--hf-splits",
        default="expert,mixed-small,mixed-large",
        help="Comma-separated Dreamer4 dataset splits to download.",
    )
    parser.add_argument(
        "--max-download-gb",
        type=float,
        default=40.0,
        help="Safety guard for Hugging Face downloads; <=0 disables.",
    )
    parser.add_argument("--hf-max-workers", type=int, default=8)
    parser.add_argument("--hf-token", default=None, help="Optional HF token; otherwise uses local HF auth/env.")

    # RoboNet TensorFlow Datasets source.
    parser.add_argument("--tfds-name", default=DEFAULT_ROBONET_TFDS_NAME)
    parser.add_argument("--tfds-data-dir", default=DEFAULT_ROBONET_TFDS_DIR)
    parser.add_argument("--tfds-download-dir", default="")
    parser.add_argument("--tfds-max-download-gb", type=float, default=2.0)
    parser.add_argument("--robonet-gdrive-id", default=DEFAULT_ROBONET_GDRIVE_ID)
    parser.add_argument("--robonet-archive", default=DEFAULT_ROBONET_ARCHIVE)

    # Shared controls.
    parser.add_argument("--dry-run", action="store_true", help="Only fetch metadata; do not download.")
    parser.add_argument("--force", action="store_true", help="Delete existing final/partial file and restart.")
    parser.add_argument("--force-lock", action="store_true", help="Remove a stale lock file before starting.")
    parser.add_argument("--max-attempts", type=int, default=0, help="0 means retry forever.")
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--retry-initial-s", type=float, default=5.0)
    parser.add_argument("--retry-max-s", type=float, default=300.0)
    parser.add_argument("--state-every-mb", type=int, default=128)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--quick-zip-check", action="store_true", default=True)
    parser.add_argument("--no-quick-zip-check", action="store_false", dest="quick_zip_check")
    parser.add_argument("--verify-zip", action="store_true", help="Run full zip CRC test after download; can be slow.")
    parser.add_argument("--extract-dir", default="", help="Optional extraction directory after successful download.")
    parser.add_argument("--extract", action="store_true", help="Extract zip into --extract-dir after successful download.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source == "dreamer4-hf":
        return download_dreamer4_hf(args)
    if args.source == "robonet-tfds":
        return download_robonet_tfds(args)
    if args.source == "robonet-gdrive":
        return download_robonet_gdrive(args)

    dest = resolve_path(args.dest)
    part_path = Path(str(dest) + ".part")
    state_path = Path(str(dest) + ".download.json")
    lock_path = Path(str(dest) + ".lock")

    metadata = fetch_remote_metadata(args.url, timeout_s=args.timeout_s)
    print(json.dumps({"remote": asdict(metadata)}, indent=2))
    if args.dry_run:
        return 0

    if args.force:
        for path in (dest, part_path, state_path):
            if path.exists():
                path.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    with download_lock(lock_path, force_lock=args.force_lock):
        download_with_resume(
            args=args,
            metadata=metadata,
            dest=dest,
            part_path=part_path,
            state_path=state_path,
        )
        if args.quick_zip_check:
            quick_zip_check(dest)
        if args.verify_zip:
            verify_zip(dest)
        if args.extract:
            extract_dir = resolve_path(args.extract_dir) if args.extract_dir else dest.with_suffix("")
            extract_zip(dest, extract_dir)
    return 0


def download_robonet_gdrive(args: argparse.Namespace) -> int:
    archive_path = resolve_path(args.robonet_archive)
    state_path = Path(str(archive_path) + ".download.json")
    lock_path = Path(str(archive_path) + ".lock")
    manifest = {
        "source": "robonet-gdrive",
        "gdrive_id": args.robonet_gdrive_id,
        "url": f"https://drive.google.com/uc?id={args.robonet_gdrive_id}",
        "dest": str(archive_path),
        "expected_bytes": 125618217,
        "expected_human": "119.80 MiB",
        "expected_sha256": "33367bb81c85a98630d0610c425d9cb33dc1652be57c98ce0ac239d12168d671",
        "note": "Fallback for TFDS RoboNet when TensorFlow Datasets receives a Google Drive confirmation page.",
    }
    print(json.dumps({"remote": manifest}, indent=2))
    if args.dry_run:
        return 0

    try:
        import gdown  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("gdown is required for --source robonet-gdrive") from exc

    if args.force:
        archive_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with download_lock(lock_path, force_lock=args.force_lock):
        started = time.time()
        cmd = [
            sys.executable,
            "-m",
            "gdown",
            f"https://drive.google.com/uc?id={args.robonet_gdrive_id}",
            "-O",
            str(archive_path),
            "--continue",
        ]
        subprocess_run(cmd)
        sha256 = file_sha256(archive_path)
        payload = {
            **manifest,
            "completed": True,
            "elapsed_s": time.time() - started,
            "bytes": archive_path.stat().st_size,
            "human": human_bytes(archive_path.stat().st_size),
            "sha256": sha256,
            "sha256_matches": sha256 == manifest["expected_sha256"],
            "updated_unix_s": time.time(),
        }
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not payload["sha256_matches"]:
            raise RuntimeError(f"RoboNet archive checksum mismatch: {sha256}")
        print(f"RoboNet archive download complete: {archive_path} ({payload['human']})")
    return 0


def download_robonet_tfds(args: argparse.Namespace) -> int:
    tfds_dir = resolve_path(args.tfds_data_dir)
    download_dir = resolve_path(args.tfds_download_dir) if args.tfds_download_dir else None
    state_path = tfds_dir / ".download.json"
    lock_path = tfds_dir / ".lock"

    try:
        import tensorflow_datasets as tfds
    except ModuleNotFoundError:
        manifest = robonet_tfds_dependency_manifest(args, tfds_dir, download_dir)
        print(json.dumps({"remote": manifest}, indent=2))
        if args.dry_run:
            return 0
        raise RuntimeError(
            "tensorflow_datasets is required for --source robonet-tfds. "
            "Install tensorflow-datasets and tensorflow, or run inside a data-prep image."
        )

    builder = tfds.builder(args.tfds_name, data_dir=str(tfds_dir))
    info = builder.info
    download_bytes = tfds_size_to_bytes(getattr(info, "download_size", None))
    dataset_bytes = tfds_size_to_bytes(getattr(info, "dataset_size", None))
    manifest = {
        "source": "robonet-tfds",
        "tfds_name": args.tfds_name,
        "data_dir": str(tfds_dir),
        "download_dir": str(download_dir) if download_dir else None,
        "download_bytes": download_bytes,
        "download_human": human_bytes(download_bytes),
        "dataset_bytes": dataset_bytes,
        "dataset_human": human_bytes(dataset_bytes),
        "version": str(info.version),
        "description": str(info.description).strip(),
        "features": repr(info.features),
        "max_download_gb": args.tfds_max_download_gb,
        "within_size_guard": within_size_guard(download_bytes or 0, args.tfds_max_download_gb),
    }
    print(json.dumps({"remote": manifest}, indent=2))
    if args.dry_run:
        return 0
    if download_bytes is not None and not manifest["within_size_guard"]:
        raise RuntimeError(
            f"Refusing to download {human_bytes(download_bytes)} from {args.tfds_name}; "
            f"raise --tfds-max-download-gb or set it <=0 to disable the guard."
        )

    if args.force and tfds_dir.exists():
        shutil.rmtree(tfds_dir)
    tfds_dir.mkdir(parents=True, exist_ok=True)
    if download_dir:
        download_dir.mkdir(parents=True, exist_ok=True)

    with download_lock(lock_path, force_lock=args.force_lock):
        started = time.time()
        builder.download_and_prepare(download_dir=str(download_dir) if download_dir else None)
        files = list_local_files(tfds_dir)
        payload = {
            **manifest,
            "completed": True,
            "elapsed_s": time.time() - started,
            "local_file_count": len(files),
            "local_total_bytes": sum(path.stat().st_size for path in files),
            "local_total_human": human_bytes(sum(path.stat().st_size for path in files)),
            "updated_unix_s": time.time(),
        }
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"RoboNet TFDS download complete: {tfds_dir} ({payload['local_total_human']})")
    return 0


def robonet_tfds_dependency_manifest(
    args: argparse.Namespace,
    tfds_dir: Path,
    download_dir: Path | None,
) -> dict[str, Any]:
    return {
        "source": "robonet-tfds",
        "tfds_name": args.tfds_name,
        "data_dir": str(tfds_dir),
        "download_dir": str(download_dir) if download_dir else None,
        "dependency_missing": "tensorflow_datasets",
        "known_default": {
            "name": DEFAULT_ROBONET_TFDS_NAME,
            "download_size": "119.80 MiB",
            "dataset_size": "183.04 MiB",
            "examples": {"train": 700},
            "features": {
                "video": "(None, 64, 64, 3) uint8",
                "actions": "(None, 5) float32",
                "states": "(None, 5) float32",
                "filename": "string",
            },
        },
        "install_hint": "pip install tensorflow-cpu tensorflow-datasets",
    }


def download_dreamer4_hf(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub is required for --source dreamer4-hf") from exc

    hf_dir = resolve_path(args.hf_dir)
    state_path = hf_dir / ".download.json"
    lock_path = hf_dir / ".lock"
    splits = parse_csv(args.hf_splits)
    allow_patterns = ["README.md", ".gitattributes"] + [f"{split}/*" for split in splits]

    api = HfApi(token=args.hf_token)
    entries = []
    for item in api.list_repo_tree(
        args.hf_dataset_id,
        repo_type="dataset",
        revision=args.hf_revision,
        recursive=True,
        expand=True,
    ):
        if item.__class__.__name__ != "RepoFile":
            continue
        path = str(item.path)
        if path in {"README.md", ".gitattributes"} or any(path.startswith(f"{split}/") for split in splits):
            entries.append({"path": path, "size": getattr(item, "size", None)})

    total_bytes = sum(entry["size"] or 0 for entry in entries)
    by_root: dict[str, dict[str, Any]] = {}
    for entry in entries:
        root = entry["path"].split("/", 1)[0]
        bucket = by_root.setdefault(root, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += entry["size"] or 0

    manifest = {
        "source": "dreamer4-hf",
        "dataset_id": args.hf_dataset_id,
        "revision": args.hf_revision,
        "local_dir": str(hf_dir),
        "splits": splits,
        "allow_patterns": allow_patterns,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "total_human": human_bytes(total_bytes),
        "by_root": {
            key: {"files": value["files"], "bytes": value["bytes"], "human": human_bytes(value["bytes"])}
            for key, value in sorted(by_root.items())
        },
        "largest_files": [
            {"path": entry["path"], "size": entry["size"], "human": human_bytes(entry["size"])}
            for entry in sorted(entries, key=lambda item: item["size"] or 0, reverse=True)[:20]
        ],
        "max_download_gb": args.max_download_gb,
        "within_size_guard": within_size_guard(total_bytes, args.max_download_gb),
    }
    print(json.dumps({"remote": manifest}, indent=2))
    if args.dry_run:
        return 0
    if not manifest["within_size_guard"]:
        raise RuntimeError(
            f"Refusing to download {human_bytes(total_bytes)} from {args.hf_dataset_id}; "
            f"raise --max-download-gb or set it <=0 to disable the guard."
        )

    if args.force and hf_dir.exists():
        shutil.rmtree(hf_dir)
    hf_dir.mkdir(parents=True, exist_ok=True)

    with download_lock(lock_path, force_lock=args.force_lock):
        started = time.time()
        try:
            local_path = snapshot_download(
                repo_id=args.hf_dataset_id,
                repo_type="dataset",
                revision=args.hf_revision,
                local_dir=hf_dir,
                allow_patterns=allow_patterns,
                force_download=args.force,
                token=args.hf_token,
                max_workers=max(1, int(args.hf_max_workers)),
            )
            downloaded = list_local_files(hf_dir)
            payload = {
                **manifest,
                "snapshot_path": str(local_path),
                "completed": True,
                "elapsed_s": time.time() - started,
                "local_file_count": len(downloaded),
                "local_total_bytes": sum(path.stat().st_size for path in downloaded),
                "local_total_human": human_bytes(sum(path.stat().st_size for path in downloaded)),
                "updated_unix_s": time.time(),
            }
            state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Dreamer4 HF download complete: {hf_dir} ({payload['local_total_human']})")
        except Exception as exc:
            state_path.write_text(
                json.dumps(
                    {
                        **manifest,
                        "completed": False,
                        "message": str(exc),
                        "updated_unix_s": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise
    return 0


def fetch_remote_metadata(url: str, timeout_s: float) -> RemoteMetadata:
    response = requests.head(url, headers=REQUEST_HEADERS, allow_redirects=True, timeout=timeout_s)
    response.raise_for_status()
    headers = response.headers
    content_length = headers.get("Content-Length")
    return RemoteMetadata(
        url=url,
        final_url=response.url,
        status_code=response.status_code,
        content_length=int(content_length) if content_length and content_length.isdigit() else None,
        accept_ranges=headers.get("Accept-Ranges"),
        etag=headers.get("ETag"),
        last_modified=headers.get("Last-Modified"),
        content_type=headers.get("Content-Type"),
    )


def download_with_resume(
    *,
    args: argparse.Namespace,
    metadata: RemoteMetadata,
    dest: Path,
    part_path: Path,
    state_path: Path,
) -> None:
    expected_size = metadata.content_length
    if dest.exists():
        final_size = dest.stat().st_size
        if expected_size is None or final_size == expected_size:
            write_state(
                state_path,
                args=args,
                metadata=metadata,
                dest=dest,
                part_path=part_path,
                bytes_downloaded=final_size,
                completed=True,
                message="final file already exists",
            )
            print(f"Final file already exists: {dest} ({human_bytes(final_size)})")
            return
        raise RuntimeError(
            f"Existing final file has unexpected size: {dest} is {final_size}, expected {expected_size}. "
            "Use --force to restart."
        )

    if expected_size is not None and part_path.exists() and part_path.stat().st_size > expected_size:
        raise RuntimeError(
            f"Partial file is larger than expected remote size: {part_path}. "
            "Use --force to restart."
        )

    attempt = 0
    retry_sleep = float(args.retry_initial_s)
    started = time.time()
    while True:
        attempt += 1
        if args.max_attempts > 0 and attempt > args.max_attempts:
            raise RuntimeError(f"Download failed after {args.max_attempts} attempts.")
        try:
            offset = part_path.stat().st_size if part_path.exists() else 0
            if expected_size is not None and offset == expected_size:
                finalize_download(part_path, dest, state_path, args, metadata, started)
                return
            run_download_attempt(
                args=args,
                metadata=metadata,
                part_path=part_path,
                state_path=state_path,
                dest=dest,
                offset=offset,
                attempt=attempt,
                started=started,
            )
            final_size = part_path.stat().st_size
            if expected_size is not None and final_size != expected_size:
                raise RuntimeError(
                    f"Connection ended early at {human_bytes(final_size)}; expected {human_bytes(expected_size)}."
                )
            finalize_download(part_path, dest, state_path, args, metadata, started)
            return
        except KeyboardInterrupt:
            offset = part_path.stat().st_size if part_path.exists() else 0
            write_state(
                state_path,
                args=args,
                metadata=metadata,
                dest=dest,
                part_path=part_path,
                bytes_downloaded=offset,
                completed=False,
                message="interrupted by user",
            )
            raise
        except Exception as exc:
            offset = part_path.stat().st_size if part_path.exists() else 0
            write_state(
                state_path,
                args=args,
                metadata=metadata,
                dest=dest,
                part_path=part_path,
                bytes_downloaded=offset,
                completed=False,
                message=f"attempt {attempt} failed: {exc}",
            )
            print(f"Attempt {attempt} failed at {human_bytes(offset)}: {exc}", file=sys.stderr)
            print(f"Retrying in {retry_sleep:.1f}s", file=sys.stderr)
            time.sleep(retry_sleep)
            retry_sleep = min(float(args.retry_max_s), retry_sleep * 1.7)


def run_download_attempt(
    *,
    args: argparse.Namespace,
    metadata: RemoteMetadata,
    part_path: Path,
    state_path: Path,
    dest: Path,
    offset: int,
    attempt: int,
    started: float,
) -> None:
    headers: dict[str, str] = dict(REQUEST_HEADERS)
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"

    with requests.get(metadata.url, headers=headers, stream=True, timeout=args.timeout_s, allow_redirects=True) as response:
        if offset > 0 and response.status_code == 200:
            print("Server ignored Range request; restarting partial file from zero.", file=sys.stderr)
            part_path.unlink(missing_ok=True)
            offset = 0
        elif offset > 0 and response.status_code == 416:
            expected_size = metadata.content_length
            if expected_size is not None and offset == expected_size:
                return
            response.raise_for_status()
        elif response.status_code not in {200, 206}:
            response.raise_for_status()

        mode = "ab" if offset > 0 else "wb"
        chunk_size = max(1, int(args.chunk_mb)) * 1024 * 1024
        state_every = max(1, int(args.state_every_mb)) * 1024 * 1024
        since_state = 0
        expected_size = metadata.content_length
        progress = make_progress_bar(
            enabled=not args.no_progress,
            total=expected_size,
            initial=offset,
            desc=f"SOAR download attempt {attempt}",
        )
        downloaded = offset
        with open(part_path, mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                since_state += len(chunk)
                if progress:
                    progress.update(len(chunk))
                if since_state >= state_every:
                    handle.flush()
                    write_state(
                        state_path,
                        args=args,
                        metadata=metadata,
                        dest=dest,
                        part_path=part_path,
                        bytes_downloaded=downloaded,
                        completed=False,
                        message="downloading",
                    )
                    since_state = 0
        if progress:
            progress.close()


def finalize_download(
    part_path: Path,
    dest: Path,
    state_path: Path,
    args: argparse.Namespace,
    metadata: RemoteMetadata,
    started: float,
) -> None:
    part_path.replace(dest)
    write_state(
        state_path,
        args=args,
        metadata=metadata,
        dest=dest,
        part_path=part_path,
        bytes_downloaded=dest.stat().st_size,
        completed=True,
        message="download complete",
        elapsed_s=time.time() - started,
    )
    print(f"Download complete: {dest} ({human_bytes(dest.stat().st_size)})")


def quick_zip_check(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not names:
                raise RuntimeError("zip contains no entries")
            print(f"Quick ZIP check passed: {len(names)} entries; first entry={names[0]}")
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Downloaded file is not a valid zip: {path}") from exc


def verify_zip(path: Path) -> None:
    print("Running full ZIP CRC verification. This may take a while.")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"ZIP verification failed at member: {bad}")
    print("Full ZIP verification passed.")


def extract_zip(path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {path} -> {extract_dir}")
    with zipfile.ZipFile(path) as archive:
        archive.extractall(extract_dir)
    print(f"Extraction complete: {extract_dir}")


def write_state(
    path: Path,
    *,
    args: argparse.Namespace,
    metadata: RemoteMetadata,
    dest: Path,
    part_path: Path,
    bytes_downloaded: int,
    completed: bool,
    message: str,
    elapsed_s: float | None = None,
) -> None:
    expected = metadata.content_length
    payload: dict[str, Any] = {
        "url": metadata.url,
        "final_url": metadata.final_url,
        "dest": str(dest),
        "part_path": str(part_path),
        "expected_bytes": expected,
        "downloaded_bytes": int(bytes_downloaded),
        "downloaded_human": human_bytes(bytes_downloaded),
        "percent": round(100.0 * bytes_downloaded / expected, 4) if expected else None,
        "completed": completed,
        "message": message,
        "updated_unix_s": time.time(),
        "remote": asdict(metadata),
        "args": vars(args),
    }
    if elapsed_s is not None:
        payload["elapsed_s"] = elapsed_s
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def make_progress_bar(*, enabled: bool, total: int | None, initial: int, desc: str):
    if not enabled:
        return None
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return None
    return tqdm(
        total=total,
        initial=initial,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
        dynamic_ncols=True,
    )


@contextmanager
def download_lock(lock_path: Path, *, force_lock: bool):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if force_lock and lock_path.exists():
        lock_path.unlink()
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Lock exists: {lock_path}. If stale, rerun with --force-lock.") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "created_unix_s": time.time()}, indent=2))

    def cleanup(*_):
        lock_path.unlink(missing_ok=True)

    old_int = signal.signal(signal.SIGINT, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    old_term = signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        cleanup()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def within_size_guard(total_bytes: int, max_download_gb: float) -> bool:
    if max_download_gb <= 0:
        return True
    return total_bytes <= int(max_download_gb * (1024**3))


def tfds_size_to_bytes(value: Any) -> int | None:
    if value is None:
        return None
    for attr in ("bytes", "_bytes"):
        raw = getattr(value, attr, None)
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def list_local_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file() and path.name != ".lock"]


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def subprocess_run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{value} B"


if __name__ == "__main__":
    raise SystemExit(main())
