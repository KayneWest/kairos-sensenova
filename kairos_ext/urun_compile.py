"""Opt-in ``urun.compile`` catalog for Kairos-owned CUDA kernels.

This integration deliberately does not change Kairos's existing inference or
engine-patching defaults. Callers first wrap the eager VAE RMSNorm+SiLU pairs
without changing their math, then pass the returned catalog to
``urun.compile`` or ``urun.compile.bench``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

VAE_RMS_SILU_OP = "app:kairos/vae-rms-silu"
VAE_RMS_SILU_FLOOR = "app:kairos/vae-rms-silu:eager"
VAE_RMS_SILU_KERNEL = "byo:kairos/vae-rms-silu-cf"
VAE_CUDA_FLAGS = ("-O3", "--use_fast_math", "-arch=sm_120")

_FLOOR_SEMANTICS = "F.silu(RMS_norm(x));channel-first;v1"
_FLOOR_IDENTITY = (
    "kairos:vae-rms-silu-floor:"
    + hashlib.sha256(_FLOOR_SEMANTICS.encode()).hexdigest()
)


class VaeRmsSiluFloor(nn.Module):
    """Eager-equivalent wrapper that gives the fused pair one module boundary."""

    __urun_kernel_floor__ = _FLOOR_IDENTITY

    def __init__(self, rms_norm: nn.Module) -> None:
        super().__init__()
        self.rms_norm = rms_norm

    @property
    def gamma(self) -> torch.Tensor:
        return self.rms_norm.gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.rms_norm(x))


def _vae_model(vae_or_model: Any) -> Any:
    model = vae_or_model.model if hasattr(vae_or_model, "model") else vae_or_model
    if not hasattr(model, "decoder"):
        raise RuntimeError(f"Cannot find decoder in {type(model).__name__}")
    return model


def _wrap_sequential(seq: nn.Sequential) -> tuple[nn.Sequential, int]:
    from kairos.modules.vaes.wan_video_vae import RMS_norm

    layers = list(seq.children())
    wrapped: list[nn.Module] = []
    count = 0
    index = 0
    while index < len(layers):
        if (
            index + 1 < len(layers)
            and isinstance(layers[index], RMS_norm)
            and isinstance(layers[index + 1], nn.SiLU)
        ):
            wrapped.append(VaeRmsSiluFloor(layers[index]))
            index += 2
            count += 1
        else:
            wrapped.append(layers[index])
            index += 1
    if count == 0:
        return seq, 0
    return nn.Sequential(*wrapped), count


def prepare_vae_targets(vae_or_model: Any) -> int:
    """Expose eager RMSNorm+SiLU pairs as app-owned structural targets.

    This is opt-in and mathematically neutral: no extension is loaded and no
    fused kernel is installed. Calling it twice is idempotent.
    """
    from kairos.modules.vaes.wan_video_vae import ResidualBlock

    decoder = _vae_model(vae_or_model).decoder
    wrapped = 0
    for module in decoder.modules():
        if isinstance(module, ResidualBlock):
            module.residual, count = _wrap_sequential(module.residual)
            wrapped += count
    if hasattr(decoder, "head"):
        decoder.head, count = _wrap_sequential(decoder.head)
        wrapped += count
    return wrapped


def _find_targets(root: Any) -> Iterator[tuple[str, VaeRmsSiluFloor]]:
    named_modules = getattr(root, "named_modules", None)
    if not callable(named_modules):
        return
    for name, module in named_modules():
        if isinstance(module, VaeRmsSiluFloor):
            yield name, module


def _require_targets(vae_or_model: Any) -> None:
    decoder = _vae_model(vae_or_model).decoder
    if not any(True for _name, _module in _find_targets(decoder)):
        raise RuntimeError(
            "Kairos VAE has no eager RMSNorm+SiLU targets; it may already be "
            "patched with a different fused path"
        )


def _dtype_probe(module: VaeRmsSiluFloor) -> str | None:
    return {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
    }.get(module.gamma.dtype)


def _representative_shapes(channels: int) -> tuple[tuple[int, int, int, int, int], ...]:
    known = {
        512: ((1, 512, 1, 60, 80), (1, 512, 1, 30, 40)),
        384: ((1, 384, 1, 60, 80),),
        256: ((1, 256, 2, 120, 160),),
        128: ((1, 128, 4, 240, 320),),
    }
    return known.get(channels, ((1, channels, 1, 60, 80),))


def _bench_cases(module: VaeRmsSiluFloor):
    import urun

    dtype = module.gamma.dtype
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            "Kairos VAE RMSNorm+SiLU bench requires the loaded VAE in bf16 or fp16"
        )
    device = module.gamma.device
    channels = int(module.gamma.shape[0])
    return tuple(
        urun.KernelBenchCase(
            args=(torch.randn(shape, device=device, dtype=dtype),),
            label="x".join(str(value) for value in shape),
        )
        for shape in _representative_shapes(channels)
    )


def _artifact_digest() -> str:
    source = Path(__file__).with_name("csrc") / "vae_fused_kernels.cu"
    payload = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cuda_flags": VAE_CUDA_FLAGS,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _load_kernel():
    from kairos_ext.vae_patch import _get_ext

    return _get_ext().rms_silu_fused_cf


def _adapt_kernel(module: VaeRmsSiluFloor, kernel):
    eager = module.forward

    def fused(x: torch.Tensor) -> torch.Tensor:
        eligible = (
            x.dim() == 5
            and module.rms_norm.channel_first
            and x.is_contiguous()
            and x.dtype in (torch.bfloat16, torch.float16)
        )
        if not eligible:
            return eager(x)
        return kernel(x, module.gamma)

    return fused


def build_vae_catalog(*, promotions_dir: str | Path | None = None):
    """Build the app-scoped catalog; registration alone never engages it."""
    import urun

    catalog = urun.KernelCatalog()
    catalog.define_op(
        op=VAE_RMS_SILU_OP,
        floor_id=VAE_RMS_SILU_FLOOR,
        finder=_find_targets,
        dtype_probe=_dtype_probe,
        bench_cases=_bench_cases,
    )
    digest = _artifact_digest()
    for dtype in ("bf16", "fp16"):
        catalog.register(
            op=VAE_RMS_SILU_OP,
            arch="sm120",
            dtype=dtype,
            kernel_id=f"{VAE_RMS_SILU_KERNEL}-{dtype}",
            artifact_digest=digest,
            loader=_load_kernel,
            adapter=_adapt_kernel,
            note="Kairos channels-first 5D fused RMSNorm+SiLU CUDA kernel",
        )
    if promotions_dir is not None:
        catalog.load_promotions(promotions_dir)
    return catalog


def bench_vae(
    vae_or_model: Any,
    *,
    out_dir: str | Path,
    iters: int = 50,
    warmup: int = 10,
):
    """Prepare and benchmark Kairos's VAE fusion on the current GPU."""
    import urun

    prepare_vae_targets(vae_or_model)
    _require_targets(vae_or_model)
    catalog = build_vae_catalog()
    return urun.compile.bench(
        vae_or_model,
        ops=(VAE_RMS_SILU_OP,),
        iters=iters,
        warmup=warmup,
        out_dir=out_dir,
        catalog=catalog,
    )


def enable_vae(
    vae_or_model: Any,
    *,
    promotions_dir: str | Path,
):
    """Install only a matching measured promotion; otherwise stay eager."""
    import urun

    prepare_vae_targets(vae_or_model)
    _require_targets(vae_or_model)
    catalog = build_vae_catalog(promotions_dir=promotions_dir)
    return urun.compile(vae_or_model, mode=None, catalog=catalog)
