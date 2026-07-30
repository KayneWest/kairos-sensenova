"""CPU conformance tests for the optional urun.compile Kairos catalog."""

from __future__ import annotations

import torch
import torch.nn as nn
from kairos.modules.vaes.wan_video_vae import ResidualBlock, RMS_norm

from kairos_ext.urun_compile import (
    VAE_RMS_SILU_FLOOR,
    VAE_RMS_SILU_KERNEL,
    VAE_RMS_SILU_OP,
    VaeRmsSiluFloor,
    _wrap_sequential,
    build_vae_catalog,
    prepare_vae_targets,
)


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = ResidualBlock(8, 8)
        self.head = nn.Sequential(RMS_norm(8, images=False), nn.SiLU())


class _VaeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _Decoder()


def test_target_wrapper_preserves_the_eager_math():
    seq = nn.Sequential(RMS_norm(8, images=False), nn.SiLU())
    x = torch.randn(1, 8, 1, 2, 2)
    expected = seq(x)

    wrapped, count = _wrap_sequential(seq)

    assert count == 1
    assert isinstance(wrapped[0], VaeRmsSiluFloor)
    torch.testing.assert_close(wrapped(x), expected)


def test_target_preparation_is_opt_in_and_idempotent():
    model = _VaeModel()
    assert not any(isinstance(module, VaeRmsSiluFloor) for module in model.modules())

    first = prepare_vae_targets(model)
    second = prepare_vae_targets(model)

    assert first == 3
    assert second == 0
    assert sum(isinstance(module, VaeRmsSiluFloor) for module in model.modules()) == 3


def test_catalog_carries_sm120_bf16_and_fp16_candidates():
    catalog = build_vae_catalog()

    assert catalog.floor_for(VAE_RMS_SILU_OP).kernel_id == VAE_RMS_SILU_FLOOR
    assert [candidate.kernel_id for candidate in catalog.candidates_for(
        VAE_RMS_SILU_OP, "sm120", "bf16"
    )] == [f"{VAE_RMS_SILU_KERNEL}-bf16"]
    assert [candidate.kernel_id for candidate in catalog.candidates_for(
        VAE_RMS_SILU_OP, "sm120", "fp16"
    )] == [f"{VAE_RMS_SILU_KERNEL}-fp16"]
    assert catalog.op_definition(VAE_RMS_SILU_OP).bench_cases is not None


def test_no_promotion_keeps_the_eager_floor_and_never_loads_cuda(monkeypatch):
    import urun
    from kairos_ext import urun_compile

    monkeypatch.setattr("urun.core.accel.detect_arch", lambda: "sm120")
    monkeypatch.setattr(
        urun_compile,
        "_load_kernel",
        lambda: (_ for _ in ()).throw(AssertionError("loader must stay lazy")),
    )
    floor = VaeRmsSiluFloor(RMS_norm(8, images=False)).to(torch.bfloat16)
    root = nn.Sequential(floor)
    x = torch.randn(1, 8, 1, 2, 2, dtype=torch.bfloat16)
    expected = root(x)
    catalog = build_vae_catalog()

    out = urun.compile(root, mode=None, catalog=catalog)

    torch.testing.assert_close(out(x), expected)
    assert out.__urun_accel__.kernels == {}
    assert out.__urun_accel__.provenance[VAE_RMS_SILU_OP]["kernel"] == VAE_RMS_SILU_FLOOR
