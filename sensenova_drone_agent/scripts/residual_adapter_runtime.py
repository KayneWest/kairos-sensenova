from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_residual_action_adapter import ResidualActionAdapter, ResidualDynamicsWrapper  # noqa: E402


def wrap_dynamics_with_residual_adapter(
    *,
    base: nn.Module,
    adapter_ckpt: str | Path,
    dyn_args: dict[str, Any],
    tok_args: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a trained residual action adapter and wrap a frozen dynamics model."""

    ckpt_path = resolve_path(adapter_ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = dict(ckpt.get("args", {}))

    action_dim = int(saved_args.get("action_dim", dyn_args.get("action_dim", 49)))
    packing_factor = int(dyn_args.get("packing_factor", 2))
    n_latents = int(tok_args.get("n_latents", 16))
    if n_latents % packing_factor != 0:
        raise ValueError(f"n_latents={n_latents} must be divisible by packing_factor={packing_factor}")
    n_spatial = n_latents // packing_factor
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    d_spatial = d_bottleneck * packing_factor
    k_max = int(dyn_args.get("k_max", 8))

    adapter = ResidualActionAdapter(
        action_dim=action_dim,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        k_max=k_max,
        hidden=int(saved_args.get("hidden", 256)),
    ).to(device)
    adapter.load_state_dict(ckpt["adapter"], strict=True)
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad_(False)

    wrapped = ResidualDynamicsWrapper(
        base=base,
        adapter=adapter,
        scale=float(saved_args.get("residual_scale", 1.0)),
    ).to(device)
    wrapped.eval()
    for param in wrapped.parameters():
        param.requires_grad_(False)

    info = {
        "adapter_ckpt": str(ckpt_path),
        "adapter_step": int(ckpt.get("step", -1)),
        "residual_scale": float(saved_args.get("residual_scale", 1.0)),
        "action_dim": action_dim,
        "action_features": str(saved_args.get("action_features", dyn_args.get("action_features", ""))),
        "source_names": str(saved_args.get("source_names", "")),
        "contrast_modes": str(saved_args.get("contrast_modes", "")),
        "saved_args": saved_args,
    }
    return wrapped, info


def infer_adapter_action_overrides(adapter_ckpt: str | Path | None) -> dict[str, Any]:
    if not adapter_ckpt:
        return {}
    ckpt = torch.load(resolve_path(adapter_ckpt), map_location="cpu", weights_only=False)
    saved_args = dict(ckpt.get("args", {}))
    overrides: dict[str, Any] = {}
    if "action_dim" in saved_args:
        overrides["action_dim"] = int(saved_args["action_dim"])
    if "action_features" in saved_args:
        overrides["action_features"] = str(saved_args["action_features"])
    return overrides


def write_residual_adapter_info(path: str | Path, info: dict[str, Any]) -> None:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()
