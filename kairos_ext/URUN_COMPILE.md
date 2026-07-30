# Kairos × `urun.compile` native-kernel gate

This is an opt-in conformance application for the first-class BYO-kernel path.
It does not replace Kairos's existing `patch_vae()` or full-DiT engine path,
and importing it changes nothing.

The first target is the existing channels-first CUDA fusion in
`csrc/vae_fused_kernels.cu`:

```text
eager RMS_norm → SiLU
        versus
rms_silu_fused_cf
```

It is a useful gate because it is a real app-owned composite operation rather
than one of `urun.compile`'s built-in op families. The source, CUDA flags,
PyTorch version, and PyTorch CUDA version form the candidate artifact digest;
a changed build cannot consume an old promotion.

## Why RTX PRO 6000 is a valid first target

NVIDIA lists both the GeForce RTX 5090 and RTX PRO 6000 Blackwell as compute
capability 12.0. The Kairos kernel targets `sm_120`, while its original 5090
development environment has 32 GB and the RTX PRO 6000 has 96 GB. Memory
capacity should therefore not be the constraint, although the full validation
must still run because clocks, thermals, drivers, CUDA, and PyTorch can change
the winning verdict.

- Compute capability table:
  https://developer.nvidia.com/cuda/gpus
- RTX PRO 6000 memory:
  https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/
- RTX 5090 memory:
  https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/

## Scratch-GPU benchmark

Use the stacked `urun-python` draft ending at PR `#1473`, load Kairos normally
in bf16 on the target GPU, then:

```python
from kairos_ext.urun_compile import bench_vae

records = bench_vae(
    pipeline.vae,
    out_dir="kairos_ext/kernel_promotions/rtx-pro-6000",
    iters=50,
    warmup=10,
)
print(records)
```

Registration alone never installs the CUDA kernel. `urun.compile.bench` first
compares every representative 5D call against the eager pair with bf16/fp16
correctness tolerances, then requires at least a 1.05× speedup. If either gate
fails, it emits no promotion record.

To exercise a generated record:

```python
from kairos_ext.urun_compile import enable_vae

enable_vae(
    pipeline.vae,
    promotions_dir="kairos_ext/kernel_promotions/rtx-pro-6000",
)
```

The record is bound to:

- `sm120`;
- input dtype;
- the exact eager-floor identity;
- the module parameter/config site;
- the candidate source/build digest.

A mismatch stays on the eager floor.

## Merge go/no-go

Do not treat successful compilation as success. The gate is:

1. The canonical bench emits a measured promotion (correctness passes and the
   minimum representative-shape speedup is at least 1.05×).
2. Run the same fixed-seed 480p Kairos demo with an eager pipeline and a fresh
   promoted pipeline.
3. The existing video quality gate passes (`SSIM >= 0.90`, `PSNR >= 28`).
4. Median VAE decode time across at least three warmed runs improves; median
   end-to-end generation time must not regress.
5. The attached `AccelReport` names the measured Kairos promotion rather than
   a floor or fallback.

Until all five are recorded on the RTX PRO 6000, this remains a draft
integration and no promotion JSON should be committed.

## Scope boundary

This validates the “`nn.Module` containing an app-owned C++/CUDA operation”
case. Kairos's whole-DiT engine mutates model topology and owns buffers,
caches, weight synchronization, training fallback, and teardown. That is an
engine-installation lifecycle, not a pure per-module call adapter, and should
get a separate explicit integration rather than being disguised as a kernel
swap.
