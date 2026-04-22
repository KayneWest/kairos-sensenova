"""Shim for apex.normalization.fused_layer_norm when NVIDIA apex is not installed.

Import this module BEFORE importing kairos to provide a drop-in FusedRMSNorm.
"""
import sys
import types
import torch
import torch.nn as nn


class FusedRMSNorm(nn.Module):
    """Drop-in replacement for apex FusedRMSNorm using pure PyTorch."""
    def __init__(self, normalized_shape, eps=1e-6, **kwargs):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f * rms).to(dtype) * self.weight


def install():
    """Install the apex shim into sys.modules if apex is not available."""
    try:
        from apex.normalization.fused_layer_norm import FusedRMSNorm as _
        return  # apex already available
    except (ImportError, ModuleNotFoundError):
        pass

    import importlib
    apex = types.ModuleType('apex')
    apex.__spec__ = importlib.machinery.ModuleSpec('apex', None)
    apex.normalization = types.ModuleType('apex.normalization')
    apex.normalization.__spec__ = importlib.machinery.ModuleSpec('apex.normalization', None)
    fln = types.ModuleType('apex.normalization.fused_layer_norm')
    fln.__spec__ = importlib.machinery.ModuleSpec('apex.normalization.fused_layer_norm', None)
    fln.FusedRMSNorm = FusedRMSNorm
    apex.normalization.fused_layer_norm = fln
    sys.modules['apex'] = apex
    sys.modules['apex.normalization'] = apex.normalization
    sys.modules['apex.normalization.fused_layer_norm'] = fln


# Auto-install on import
install()
