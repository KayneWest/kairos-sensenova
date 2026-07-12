#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_dreamer4_soar_dynamics import (  # noqa: E402
    DEFAULT_ACTION_DIM,
    build_dynamics,
    collate_batch,
    evaluate,
    load_frozen_tokenizer_from_pt_ckpt,
    parse_negative_modes,
)
from wm_dataset import WMDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run action-frame offset and per-source causal evals for a native "
            "Dreamer4 dynamics checkpoint."
        )
    )
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--tokenizer-ckpt", required=True)
    parser.add_argument("--dynamics-ckpt", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--tasks-json", default=None)
    parser.add_argument("--offsets", default="-3,-2,-1,0,1,2,3")
    parser.add_argument("--negative-modes", default="shuffle,zero,time_shift")
    parser.add_argument("--source-names", default="all,*")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches-all", type=int, default=128)
    parser.add_argument("--max-batches-source", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--eval-d", type=float, default=0.25)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--action-features", default=None)
    parser.add_argument("--require-non-noop", action="store_true")
    parser.add_argument("--no-op-threshold", type=float, default=0.0)
    parser.add_argument("--min-non-noop-steps", type=int, default=1)
    parser.add_argument("--require-visual-delta", action="store_true")
    parser.add_argument("--visual-delta-threshold", type=float, default=0.0)
    parser.add_argument("--min-visual-delta-steps", type=int, default=1)
    parser.add_argument("--visual-delta-stride", type=int, default=4)
    parser.add_argument("--causal-min-ratio", type=float, default=1.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--verbose-dataset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    manifest_path = resolve_path(args.manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    offsets = parse_offsets(args.offsets)
    negative_modes = parse_negative_modes(args.negative_modes)
    out_json = resolve_path(args.out_json)
    out_csv = resolve_path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    tokenizer_ckpt = resolve_path(args.tokenizer_ckpt)
    dynamics_ckpt = resolve_path(args.dynamics_ckpt)
    encoder, _decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(tokenizer_ckpt), device=device)

    dyn_ckpt = torch.load(dynamics_ckpt, map_location="cpu", weights_only=False)
    dyn_args = dict(dyn_ckpt.get("args", {}))
    action_dim = int(args.action_dim if args.action_dim is not None else dyn_args.get("action_dim", DEFAULT_ACTION_DIM))
    action_features = str(args.action_features if args.action_features is not None else dyn_args.get("action_features", "current"))
    dyn_args["action_dim"] = action_dim
    dyn_args["action_features"] = action_features
    model = build_dynamics(dyn_args, tok_args).to(device)
    model.load_state_dict(dyn_ckpt["dynamics"], strict=True)
    model.eval()

    source_specs = select_sources(manifest, args.source_names)
    tasks_json = resolve_path(args.tasks_json or manifest.get("tasks_json", ""))
    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for source_index, source in enumerate(source_specs):
        max_batches = int(args.max_batches_all if source["name"] == "all" else args.max_batches_source)
        print(
            f"[sweep] source={source['name']} offsets={offsets} "
            f"max_batches={max_batches} data_dirs={len(source['raw'])}",
            flush=True,
        )
        dataset = WMDataset(
            data_dir=[str(resolve_path(path)) for path in source["raw"]],
            frames_dir=[str(resolve_path(path)) for path in source["frames"]],
            seq_len=int(args.seq_len),
            img_size=128,
            action_dim=action_dim,
            tasks_json=str(tasks_json),
            tasks=None,
            strict_tasks=False,
            action_features=action_features,
            require_non_noop=bool(args.require_non_noop),
            no_op_threshold=float(args.no_op_threshold),
            min_non_noop_steps=int(args.min_non_noop_steps),
            require_visual_delta=bool(args.require_visual_delta),
            visual_delta_threshold=float(args.visual_delta_threshold),
            min_visual_delta_steps=int(args.min_visual_delta_steps),
            visual_delta_stride=int(args.visual_delta_stride),
            verbose=bool(args.verbose_dataset),
        )
        for offset in offsets:
            generator = torch.Generator()
            generator.manual_seed(int(args.seed) + source_index * 1009)
            loader = DataLoader(
                dataset,
                batch_size=int(args.batch_size),
                shuffle=True,
                generator=generator,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
                collate_fn=collate_batch,
            )
            metrics = evaluate(
                model=model,
                encoder=encoder,
                loader=loader,
                tok_args=tok_args,
                dyn_args=dyn_args,
                device=device,
                max_batches=max_batches,
                rollout_horizon=int(args.rollout_horizon),
                ctx_len=int(args.ctx_len),
                eval_d=float(args.eval_d),
                action_frame_offset=int(offset),
                seed=int(args.seed) + source_index * 1009,
                causal_min_ratio=float(args.causal_min_ratio),
                negative_modes=negative_modes,
            )
            result = {
                "source": source["name"],
                "source_index": source_index,
                "offset": int(offset),
                "max_batches": max_batches,
                "metrics": metrics,
            }
            results.append(result)
            row = flatten_result(result, negative_modes)
            rows.append(row)
            print(
                "[sweep] "
                f"source={source['name']} offset={offset:+d} "
                f"batches={metrics['batches']} "
                f"direct_normal={metrics['direct']['normal']:.6f} "
                f"ar_normal={metrics['autoregressive']['normal']:.6f} "
                f"ar_time={metrics['autoregressive'].get('time_shift_over_normal', float('nan')):.4f} "
                f"ar_persist={metrics['autoregressive']['normal_over_persistence']:.4f}",
                flush=True,
            )

    summary = summarize(results, negative_modes)
    payload = {
        "phase": "dreamer4_action_alignment_source_sweep",
        "manifest_json": str(manifest_path),
        "tokenizer_ckpt": str(tokenizer_ckpt),
        "dynamics_ckpt": str(dynamics_ckpt),
        "tasks_json": str(tasks_json),
        "config": {
            "offsets": offsets,
            "negative_modes": negative_modes,
            "source_names": args.source_names,
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
            "max_batches_all": int(args.max_batches_all),
            "max_batches_source": int(args.max_batches_source),
            "rollout_horizon": int(args.rollout_horizon),
            "ctx_len": int(args.ctx_len),
            "eval_d": float(args.eval_d),
            "action_dim": action_dim,
            "action_features": action_features,
            "require_non_noop": bool(args.require_non_noop),
            "no_op_threshold": float(args.no_op_threshold),
            "min_non_noop_steps": int(args.min_non_noop_steps),
            "require_visual_delta": bool(args.require_visual_delta),
            "visual_delta_threshold": float(args.visual_delta_threshold),
            "min_visual_delta_steps": int(args.min_visual_delta_steps),
            "visual_delta_stride": int(args.visual_delta_stride),
            "causal_min_ratio": float(args.causal_min_ratio),
            "seed": int(args.seed),
        },
        "summary": summary,
        "results": results,
        "elapsed_s": time.time() - started,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out_csv, rows)
    print(json.dumps({"out_json": str(out_json), "out_csv": str(out_csv), "summary": summary}, indent=2))
    return 0


def select_sources(manifest: dict[str, Any], source_names: str) -> list[dict[str, Any]]:
    raw_sources = manifest.get("sources", [])
    all_source = {
        "name": "all",
        "raw": [source["raw"] for source in raw_sources],
        "frames": [source["frames"] for source in raw_sources],
    }
    requested = [name.strip() for name in str(source_names).split(",") if name.strip()]
    if not requested:
        requested = ["all", "*"]
    specs: list[dict[str, Any]] = []
    for name in requested:
        if name == "all":
            specs.append(all_source)
        elif name == "*":
            for source in raw_sources:
                specs.append({"name": source["name"], "raw": [source["raw"]], "frames": [source["frames"]]})
        else:
            matched = [source for source in raw_sources if source.get("name") == name]
            if not matched:
                raise ValueError(f"source not found in manifest: {name}")
            source = matched[0]
            specs.append({"name": source["name"], "raw": [source["raw"]], "frames": [source["frames"]]})
    seen = set()
    deduped = []
    for spec in specs:
        if spec["name"] in seen:
            continue
        seen.add(spec["name"])
        deduped.append(spec)
    return deduped


def summarize(results: list[dict[str, Any]], negative_modes: list[str]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_source.setdefault(result["source"], []).append(result)

    summary: dict[str, Any] = {}
    for source, items in by_source.items():
        best_direct = min(items, key=lambda item: item["metrics"]["direct"]["normal"])
        best_ar = min(items, key=lambda item: item["metrics"]["autoregressive"]["normal"])
        best_temporal = max(
            items,
            key=lambda item: max(
                item["metrics"]["autoregressive"].get(f"{mode}_over_normal", float("-inf"))
                for mode in negative_modes
                if mode not in {"shuffle", "zero"}
            ),
        )
        summary[source] = {
            "best_direct_normal_offset": best_direct["offset"],
            "best_direct_normal": best_direct["metrics"]["direct"]["normal"],
            "best_ar_normal_offset": best_ar["offset"],
            "best_ar_normal": best_ar["metrics"]["autoregressive"]["normal"],
            "best_ar_temporal_offset": best_temporal["offset"],
            "best_ar_temporal_ratio": max(
                best_temporal["metrics"]["autoregressive"].get(f"{mode}_over_normal", float("-inf"))
                for mode in negative_modes
                if mode not in {"shuffle", "zero"}
            ),
            "offsets": {
                str(item["offset"]): {
                    "direct_normal": item["metrics"]["direct"]["normal"],
                    "ar_normal": item["metrics"]["autoregressive"]["normal"],
                    "ar_normal_over_persistence": item["metrics"]["autoregressive"]["normal_over_persistence"],
                    **{
                        f"ar_{mode}_over_normal": item["metrics"]["autoregressive"].get(f"{mode}_over_normal")
                        for mode in negative_modes
                    },
                }
                for item in sorted(items, key=lambda item: item["offset"])
            },
        }
    return summary


def flatten_result(result: dict[str, Any], negative_modes: list[str]) -> dict[str, Any]:
    metrics = result["metrics"]
    row: dict[str, Any] = {
        "source": result["source"],
        "offset": result["offset"],
        "batches": metrics["batches"],
        "direct_normal": metrics["direct"]["normal"],
        "ar_normal": metrics["autoregressive"]["normal"],
        "ar_persistence": metrics["autoregressive"]["persistence"],
        "ar_normal_over_persistence": metrics["autoregressive"]["normal_over_persistence"],
    }
    for mode in negative_modes:
        row[f"direct_{mode}_over_normal"] = metrics["direct"].get(f"{mode}_over_normal")
        row[f"direct_{mode}_pair_pass_fraction"] = metrics["direct"].get(f"{mode}_pair_pass_fraction")
        row[f"ar_{mode}_over_normal"] = metrics["autoregressive"].get(f"{mode}_over_normal")
        row[f"ar_{mode}_pair_pass_fraction"] = metrics["autoregressive"].get(f"{mode}_pair_pass_fraction")
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_offsets(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).replace(";", ",").split(",") if item.strip()]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
