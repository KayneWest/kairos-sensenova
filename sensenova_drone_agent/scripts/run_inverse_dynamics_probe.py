#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DREAMER4_ROOT = REPO_ROOT / "dreamer4" / "dreamer4"
if str(DREAMER4_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMER4_ROOT))

from train_dynamics import load_frozen_tokenizer_from_pt_ckpt, temporal_patchify  # noqa: E402
from wm_dataset import WMDataset, collate_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small inverse-dynamics probe: tokenizer latents "
            "(z_t, z_{t+1}, delta) -> action_t. This tests whether the "
            "latent space preserves action-identifiable transitions."
        )
    )
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--tokenizer-ckpt", required=True)
    parser.add_argument("--tasks-json", default=None)
    parser.add_argument("--source-names", default="soar_native_v2,hf_robot_droid_lerobot_dreamer4,dreamer4_hf_mixed_large")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--action-dim", type=int, default=49)
    parser.add_argument("--action-features", default="current,prev,delta,mean4,norm")
    parser.add_argument(
        "--target-offsets",
        default="0,-1,-2",
        help="Comma-separated action target offsets relative to transition t. "
        "Offset 0 predicts action for obs[t] -> obs[t+1].",
    )
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--verbose-dataset", action="store_true")
    return parser.parse_args()


class Probe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> int:
    args = parse_args()
    started = time.time()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_path = resolve_path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = resolve_path(args.manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks_json = resolve_path(args.tasks_json or manifest.get("tasks_json", ""))
    source_specs = select_sources(manifest, args.source_names)
    if int(args.max_sources) > 0:
        source_specs = source_specs[: int(args.max_sources)]
    offsets = parse_offsets(args.target_offsets)

    encoder, _decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(resolve_path(args.tokenizer_ckpt)), device=device)
    patch = int(tok_args.get("patch", 8))

    results = []
    for source_index, source in enumerate(source_specs):
        print(f"[probe] source={source['name']} offsets={offsets}", flush=True)
        dataset = WMDataset(
            data_dir=[str(resolve_path(path)) for path in source["raw"]],
            frames_dir=[str(resolve_path(path)) for path in source["frames"]],
            seq_len=int(args.seq_len),
            img_size=128,
            action_dim=int(args.action_dim),
            raw_action_dim=int(args.action_dim),
            tasks_json=str(tasks_json),
            tasks=None,
            strict_tasks=False,
            action_features=str(args.action_features),
            verbose=bool(args.verbose_dataset),
        )
        for offset in offsets:
            result = run_probe_for_source_offset(
                args=args,
                source=source,
                source_index=source_index,
                dataset=dataset,
                encoder=encoder,
                patch=patch,
                target_offset=offset,
                device=device,
            )
            results.append(result)
            print(
                "[probe] "
                f"source={source['name']} offset={offset:+d} "
                f"eval_mse={result['metrics']['eval_mse']:.6g} "
                f"baseline_mse={result['metrics']['baseline_mse']:.6g} "
                f"r2={result['metrics']['r2']:.4f} "
                f"shuffle_ratio={result['metrics']['shuffle_over_eval']:.4f}",
                flush=True,
            )

    payload = {
        "phase": "inverse_dynamics_probe",
        "manifest_json": str(manifest_path),
        "tokenizer_ckpt": str(resolve_path(args.tokenizer_ckpt)),
        "tasks_json": str(tasks_json),
        "config": {
            "source_names": args.source_names,
            "offsets": offsets,
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
            "train_steps": int(args.train_steps),
            "eval_batches": int(args.eval_batches),
            "lr": float(args.lr),
            "hidden": int(args.hidden),
            "action_dim": int(args.action_dim),
            "action_features": str(args.action_features),
            "seed": int(args.seed),
        },
        "summary": summarize(results),
        "results": results,
        "elapsed_s": time.time() - started,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": str(out_path), "summary": payload["summary"]}, indent=2))
    return 0


def run_probe_for_source_offset(
    *,
    args: argparse.Namespace,
    source: dict[str, Any],
    source_index: int,
    dataset: WMDataset,
    encoder: nn.Module,
    patch: int,
    target_offset: int,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + source_index * 1009 + (target_offset + 128) * 17)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_batch,
    )
    iterator = iter(loader)

    # Initialize dimensions from one batch.
    batch = next(iterator)
    x, y, mask, valid = encode_batch(
        batch=batch,
        encoder=encoder,
        patch=patch,
        target_offset=target_offset,
        device=device,
    )
    in_dim = int(x.shape[-1])
    out_dim = int(y.shape[-1])
    probe = Probe(in_dim, out_dim, int(args.hidden)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=float(args.lr), weight_decay=1e-4)

    train_loss_ema = None
    target_sum = torch.zeros(out_dim, device=device)
    target_count = torch.zeros((), device=device)

    for step in range(int(args.train_steps)):
        if step > 0:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            x, y, mask, valid = encode_batch(
                batch=batch,
                encoder=encoder,
                patch=patch,
                target_offset=target_offset,
                device=device,
            )

        pred = probe(x)
        loss = masked_mse(pred, y, mask, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            weights = (mask * valid[:, None]).float()
            target_sum += (y * weights).sum(dim=0)
            target_count += weights.sum().clamp_min(0.0)
        value = float(loss.detach().item())
        train_loss_ema = value if train_loss_ema is None else 0.98 * train_loss_ema + 0.02 * value

    target_mean = target_sum / target_count.clamp_min(1.0)
    metrics = evaluate_probe(
        probe=probe,
        loader=loader,
        encoder=encoder,
        patch=patch,
        target_offset=target_offset,
        target_mean=target_mean,
        eval_batches=int(args.eval_batches),
        device=device,
    )
    metrics["train_loss_ema"] = float(train_loss_ema or 0.0)

    return {
        "source": source["name"],
        "offset": int(target_offset),
        "metrics": metrics,
        "dataset_len": len(dataset),
    }


@torch.no_grad()
def encode_batch(
    *,
    batch: dict[str, torch.Tensor],
    encoder: nn.Module,
    patch: int,
    target_offset: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = batch["obs"].to(device, non_blocking=True).float() / 255.0  # (B,T+1,3,H,W)
    raw_act = batch["raw_act"].to(device, non_blocking=True).float()
    raw_mask = batch["raw_act_mask"].to(device, non_blocking=True).float()
    patches = temporal_patchify(obs, patch)
    z, _ = encoder(patches)  # (B,T+1,L,D)
    z = z.flatten(2)  # (B,T+1,F)
    z0 = z[:, :-1]
    z1 = z[:, 1:]
    x = torch.cat([z0, z1, z1 - z0], dim=-1)

    B, T, A = raw_act.shape
    offset = int(target_offset)
    if offset == 0:
        y = raw_act
        mask = raw_mask
        valid = torch.ones((B, T), device=device, dtype=torch.bool)
    else:
        y = torch.zeros_like(raw_act)
        mask = torch.zeros_like(raw_mask)
        valid = torch.zeros((B, T), device=device, dtype=torch.bool)
        for t in range(T):
            src = t + offset
            if 0 <= src < T:
                y[:, t] = raw_act[:, src]
                mask[:, t] = raw_mask[:, src]
                valid[:, t] = True

    return x.reshape(B * T, -1), y.reshape(B * T, A), mask.reshape(B * T, A), valid.reshape(B * T)


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = mask.float() * valid[:, None].float()
    return ((pred - y).pow(2) * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_probe(
    *,
    probe: nn.Module,
    loader: DataLoader,
    encoder: nn.Module,
    patch: int,
    target_offset: int,
    target_mean: torch.Tensor,
    eval_batches: int,
    device: torch.device,
) -> dict[str, float]:
    probe.eval()
    total_loss = torch.zeros((), device=device)
    total_base = torch.zeros((), device=device)
    total_shuffle = torch.zeros((), device=device)
    total_weight = torch.zeros((), device=device)
    total_norm = torch.zeros((), device=device)
    total_abs = torch.zeros((), device=device)
    batches = 0

    for batch in loader:
        x, y, mask, valid = encode_batch(
            batch=batch,
            encoder=encoder,
            patch=patch,
            target_offset=target_offset,
            device=device,
        )
        pred = probe(x)
        weights = mask.float() * valid[:, None].float()
        weight = weights.sum().clamp_min(1.0)
        base = target_mean[None, :].expand_as(y)
        perm = torch.randperm(y.shape[0], device=device)
        shuffled = y[perm]
        total_loss += ((pred - y).pow(2) * weights).sum()
        total_base += ((base - y).pow(2) * weights).sum()
        total_shuffle += ((pred - shuffled).pow(2) * weights).sum()
        total_weight += weight
        total_norm += ((y.pow(2) * weights).sum(dim=-1).sqrt()).sum()
        total_abs += (y.abs() * weights).sum()
        batches += 1
        if batches >= int(eval_batches):
            break

    eval_mse = total_loss / total_weight.clamp_min(1.0)
    baseline_mse = total_base / total_weight.clamp_min(1.0)
    shuffle_mse = total_shuffle / total_weight.clamp_min(1.0)
    r2 = 1.0 - eval_mse / baseline_mse.clamp_min(1e-12)
    probe.train()
    return {
        "eval_mse": float(eval_mse.item()),
        "baseline_mse": float(baseline_mse.item()),
        "shuffle_mse": float(shuffle_mse.item()),
        "r2": float(r2.item()),
        "shuffle_over_eval": float((shuffle_mse / eval_mse.clamp_min(1e-12)).item()),
        "mean_target_l2": float((total_norm / max(1, batches)).item()),
        "mean_target_abs": float((total_abs / total_weight.clamp_min(1.0)).item()),
        "eval_batches": int(batches),
    }


def select_sources(manifest: dict[str, Any], source_names: str) -> list[dict[str, Any]]:
    raw_sources = manifest.get("sources", [])
    requested = [name.strip() for name in str(source_names).split(",") if name.strip()]
    if not requested:
        requested = ["*"]
    specs: list[dict[str, Any]] = []
    for name in requested:
        if name == "*":
            for source in raw_sources:
                specs.append({"name": source["name"], "raw": [source["raw"]], "frames": [source["frames"]]})
        else:
            matched = [source for source in raw_sources if source.get("name") == name]
            if not matched:
                raise ValueError(f"source not found in manifest: {name}")
            source = matched[0]
            specs.append({"name": source["name"], "raw": [source["raw"]], "frames": [source["frames"]]})
    seen = set()
    out = []
    for spec in specs:
        if spec["name"] in seen:
            continue
        seen.add(spec["name"])
        out.append(spec)
    return out


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_source.setdefault(result["source"], []).append(result)
    out = {}
    for source, items in by_source.items():
        best = max(items, key=lambda item: item["metrics"]["r2"])
        out[source] = {
            "best_offset": best["offset"],
            "best_r2": best["metrics"]["r2"],
            "best_shuffle_over_eval": best["metrics"]["shuffle_over_eval"],
            "offsets": {
                str(item["offset"]): {
                    "r2": item["metrics"]["r2"],
                    "eval_mse": item["metrics"]["eval_mse"],
                    "baseline_mse": item["metrics"]["baseline_mse"],
                    "shuffle_over_eval": item["metrics"]["shuffle_over_eval"],
                }
                for item in items
            },
        }
    return out


def parse_offsets(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
