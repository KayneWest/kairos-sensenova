#!/usr/bin/env python3
"""Quick CuTe-DSL smoke test on SM120."""
import torch
import cutlass
import cutlass.cute as cute


@cute.kernel
def vec_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    tid = cute.arch.threadIdx.x + cute.arch.blockIdx.x * 256
    if tid < 256:
        c[tid] = a[tid] + b[tid]


def main():
    a = torch.randn(256, device='cuda', dtype=torch.float32)
    b = torch.randn(256, device='cuda', dtype=torch.float32)
    c = torch.zeros(256, device='cuda', dtype=torch.float32)
    cfg = cutlass.LaunchConfig(grid=[1], block=[256])
    vec_add(a, b, c, launch_config=cfg)
    ref = a + b
    print(f"Max diff: {(c - ref).abs().max().item():.2e}")
    print("CuTe-DSL kernel compilation + execution OK on SM120")


if __name__ == "__main__":
    main()
