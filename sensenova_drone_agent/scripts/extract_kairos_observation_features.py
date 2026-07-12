#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensenova_drone.kairos_features import (  # noqa: E402
    KairosVAEFeatureExtractor,
    audit_kairos_feature_access,
    save_feature_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit/extract native Kairos/Sensenova observation features from one RGB frame."
    )
    parser.add_argument(
        "--input-frame",
        default="sensenova_drone_agent/sim_assets/sample_frames/gazebo_rgb_000001.png",
    )
    parser.add_argument("--config", default="kairos/configs/kairos_4b_config_DMD.py")
    parser.add_argument("--out-dir", default="sensenova_drone_agent/output/kairos_feature_audit_v1")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--dtype", default="bfloat16", help="bfloat16 | float16 | float32")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--no-tiled", action="store_true")
    parser.add_argument("--tile-size", default="30,52")
    parser.add_argument("--tile-stride", default="15,26")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only inspect Kairos paths/config; do not load the VAE checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = audit_kairos_feature_access(config_file=args.config, repo_root=REPO_ROOT)
    audit_path = out_dir / "feature_access_audit.json"
    audit_path.write_text(json.dumps(audit.to_dict(), indent=2), encoding="utf-8")
    print(f"Wrote audit: {audit_path}")

    if args.metadata_only:
        print("metadata_only=true; skipped VAE feature extraction.")
        return 0

    device = resolve_device(args.device)
    extractor = KairosVAEFeatureExtractor(
        config_file=args.config,
        repo_root=REPO_ROOT,
        device=device,
        dtype=args.dtype,
        height=args.height,
        width=args.width,
        tiled=not args.no_tiled,
        tile_size=parse_pair(args.tile_size),
        tile_stride=parse_pair(args.tile_stride),
    )
    payload = extractor.encode_image(Path(args.input_frame))
    summary_path = out_dir / "kairos_vae_feature_summary.json"
    save_feature_summary(payload, summary_path)
    print(f"Wrote feature summary: {summary_path}")
    print(f"Wrote latent tensor: {summary_path.with_suffix('.latent.pt')}")
    print(f"latent_shape={payload['metadata']['latent_shape']}")
    print(f"feature_dim={payload['metadata']['feature_dim']}")
    return 0


def parse_pair(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected pair H,W, got {value!r}")
    return int(parts[0]), int(parts[1])


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
