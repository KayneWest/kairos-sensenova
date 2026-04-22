#!/usr/bin/env python3
"""Quick test for VAE fused RMSNorm+SiLU kernels (channels-first, no permute)."""
import torch
import torch.nn.functional as F
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_rms_silu_fused():
    from torch.utils.cpp_extension import load as cpp_load

    csrc = os.path.join(os.path.dirname(__file__), "csrc", "vae_fused_kernels.cu")
    print("Compiling vae_fused_kernels.cu ...")
    ext = cpp_load(
        name="vae_fused_kernels",
        sources=[csrc],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_120"],
        verbose=True,
    )
    print("Compiled OK\n")

    # Test shapes matching VAE decoder stages
    test_cases = [
        (1, 512, 1, 60, 80),    # middle blocks  (N=4800)
        (1, 512, 1, 30, 40),    # smaller middle  (N=1200)
        (1, 256, 2, 120, 160),  # stage 2         (N=38400)
        (1, 128, 4, 240, 320),  # stage 3         (N=307200)
        (1, 384, 1, 60, 80),    # C=384 variant   (N=4800)
    ]

    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)
    for shape in test_cases:
        B, C, T, H, W = shape
        N = B * T * H * W
        print(f"\nShape {shape} (N={N}, C={C})")

        x = torch.randn(B, C, T, H, W, device="cuda", dtype=torch.bfloat16)
        gamma = torch.randn(C, 1, 1, 1, device="cuda", dtype=torch.float32) * 0.3 + 1.0

        # Stock path
        scale = C ** 0.5
        normed = F.normalize(x, dim=1) * scale * gamma
        stock = F.silu(normed)

        # Fused
        fused = ext.rms_silu_fused_cf(x.contiguous(), gamma)

        diff = (stock.float() - fused.float()).abs()
        max_err = diff.max().item()
        cos_sim = F.cosine_similarity(stock.float().flatten(), fused.float().flatten(), dim=0).item()
        status = "PASS" if cos_sim > 0.999 else "FAIL"
        print(f"  max_err={max_err:.6f}  cos_sim={cos_sim:.6f}  [{status}]")

    # add_rms_silu correctness
    print(f"\nadd_rms_silu_fused_cf test:")
    B, C, T, H, W = 1, 512, 1, 60, 80
    xa = torch.randn(B, C, T, H, W, device="cuda", dtype=torch.bfloat16)
    xb = torch.randn(B, C, T, H, W, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(C, 1, 1, 1, device="cuda", dtype=torch.float32) * 0.3 + 1.0
    scale = C ** 0.5
    added = xa + xb
    stock = F.silu(F.normalize(added, dim=1) * scale * gamma)
    fused = ext.add_rms_silu_fused_cf(xa.contiguous(), xb.contiguous(), gamma)
    cos_sim = F.cosine_similarity(stock.float().flatten(), fused.float().flatten(), dim=0).item()
    print(f"  cos_sim={cos_sim:.6f}  [{'PASS' if cos_sim > 0.999 else 'FAIL'}]")

    # Benchmarks
    print("\n" + "=" * 60)
    print("BENCHMARKS")
    print("=" * 60)

    for shape in test_cases:
        B, C, T, H, W = shape
        print(f"\nShape {shape}, bf16")
        x = torch.randn(B, C, T, H, W, device="cuda", dtype=torch.bfloat16)
        gamma = torch.ones(C, 1, 1, 1, device="cuda", dtype=torch.float32)
        scale = C ** 0.5

        # Warmup
        for _ in range(20):
            _ = ext.rms_silu_fused_cf(x, gamma)
            normed = F.normalize(x, dim=1) * scale * gamma
            _ = F.silu(normed)
        torch.cuda.synchronize()

        N = 200
        # Stock: F.normalize + multiply + F.silu  (3 kernels)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N):
            normed = F.normalize(x, dim=1) * scale * gamma
            _ = F.silu(normed)
        torch.cuda.synchronize()
        stock_us = (time.time() - t0) / N * 1e6

        # Fused: 1 kernel, no permutes
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N):
            _ = ext.rms_silu_fused_cf(x, gamma)
        torch.cuda.synchronize()
        fused_us = (time.time() - t0) / N * 1e6

        speedup = stock_us / fused_us if fused_us > 0 else 0
        print(f"  Stock:  {stock_us:8.1f} us")
        print(f"  Fused:  {fused_us:8.1f} us")
        print(f"  Speedup: {speedup:.2f}x")

    print("\nAll tests done!")


if __name__ == "__main__":
    test_rms_silu_fused()
