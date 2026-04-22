"""
Monkey-patch the Kairos/WAN VAE decoder with fused RMSNorm+SiLU CUDA kernels.

Replaces consecutive (RMS_norm, SiLU) pairs in ResidualBlocks and the decoder
head with a single fused kernel call. The kernel operates directly on
channels-first [B,C,T,H,W] tensors with coalesced spatial reads — no permutes.

Usage:
    from kairos_ext.vae_patch import patch_vae
    patch_vae(pipeline.vae)
"""
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load as cpp_load

_ext = None


def _get_ext():
    """JIT-compile the VAE fused kernels."""
    global _ext
    if _ext is not None:
        return _ext

    csrc = os.path.join(os.path.dirname(__file__), "csrc", "vae_fused_kernels.cu")
    _ext = cpp_load(
        name="vae_fused_kernels",
        sources=[csrc],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_120"],
        verbose=True,
    )
    return _ext


class FusedRMSSiLU(nn.Module):
    """Drop-in replacement for (RMS_norm, SiLU) pair using fused CUDA kernel.

    Works directly on channels-first 5D tensors — no layout conversion needed.
    """

    def __init__(self, rms_norm):
        super().__init__()
        self.gamma = rms_norm.gamma
        self.channel_first = rms_norm.channel_first
        self.scale = rms_norm.scale
        self.bias = rms_norm.bias

    def forward(self, x):
        ext = _get_ext()

        if (x.dim() == 5
                and self.channel_first
                and x.is_contiguous()
                and x.dtype in (torch.bfloat16, torch.float16)):
            return ext.rms_silu_fused_cf(x, self.gamma.data)
        else:
            import torch.nn.functional as F
            dim = 1 if self.channel_first else -1
            normed = F.normalize(x, dim=dim) * self.scale * self.gamma + self.bias
            return torch.nn.functional.silu(normed)


def _patch_sequential(seq):
    """Replace (RMS_norm, SiLU) pairs in an nn.Sequential with FusedRMSSiLU."""
    from kairos.modules.vaes.wan_video_vae import RMS_norm

    layers = list(seq.children())
    new_layers = []
    i = 0
    fused_count = 0
    while i < len(layers):
        if (i + 1 < len(layers)
                and isinstance(layers[i], RMS_norm)
                and isinstance(layers[i + 1], nn.SiLU)):
            new_layers.append(FusedRMSSiLU(layers[i]))
            i += 2
            fused_count += 1
        else:
            new_layers.append(layers[i])
            i += 1
    if fused_count > 0:
        return nn.Sequential(*new_layers), fused_count
    return seq, 0


def patch_vae(vae_or_model, verbose=True):
    """Patch a WanVideoVAE or its inner model with fused kernels.

    Args:
        vae_or_model: WanVideoVAE (has .model) or VideoVAE_ (has .decoder)
        verbose: print patching info

    Returns:
        total number of fused pairs
    """
    from kairos.modules.vaes.wan_video_vae import ResidualBlock

    model = vae_or_model
    if hasattr(model, 'model'):
        model = model.model
    if not hasattr(model, 'decoder'):
        raise RuntimeError(f"Cannot find decoder in {type(model).__name__}")

    decoder = model.decoder
    total_fused = 0

    if verbose:
        print("  Compiling VAE fused kernels...")
    _get_ext()

    for name, module in decoder.named_modules():
        if isinstance(module, ResidualBlock):
            new_seq, count = _patch_sequential(module.residual)
            if count > 0:
                module.residual = new_seq
                total_fused += count
                if verbose:
                    print(f"    Fused {count} norm+silu in {name}.residual")

    if hasattr(decoder, 'head'):
        new_head, count = _patch_sequential(decoder.head)
        if count > 0:
            decoder.head = new_head
            total_fused += count
            if verbose:
                print(f"    Fused {count} norm+silu in head")

    if verbose:
        print(f"  VAE patched: {total_fused} RMSNorm+SiLU pairs fused")

    return total_fused
