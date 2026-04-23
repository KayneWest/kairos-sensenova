# Kairos FP8 CUDA Engine — Status Report

**Date:** April 21, 2026
**Hardware:** NVIDIA RTX 5090 (SM120, 32GB, BF16 1.7 PFLOPS, FP8 3.4 PFLOPS)
**Model:** Kairos 4B DiT — 32 layers (24 quadratic attention + 8 GatedDeltaNet), D=2560, NH=20, HD=128, FFN=10240

## Current Performance

| Configuration | FPS (480p, 41 frames) | vs Stock+Compile |
|---|---|---|
| Stock (PyTorch + Triton) | 6.0 | — |
| Stock + torch.compile | 7.2 | 1.00x baseline |
| **Engine (FP8 + CUDA)** | **8.5** | **1.18x** |
| Engine + torch.compile | 8.5 | 1.18x (no benefit) |

SDPA microbenchmark at seq=13200:
| Method | Latency | Speedup |
|---|---|---|
| BF16 cuDNN SDPA | 9.6ms | 1.0x |
| SageAttention3 (INT8 QK, FP4 PV) | 3.2ms | 3.0x |

## Architecture

The engine replaces the stock PyTorch DiT blocks with a single fused CUDA forward pass. All 32 layers (both quadratic attention and GatedDeltaNet) run through one `ext.forward()` call.

**Data flow per layer:**
```
Input x [seq, D] bf16
  → LN + AdaLN (fused kernel: k_ln_adaln_ssts)
  → Q/K/V projections (FP8 cuBLASLt GEMMs)
  → RMSNorm + RoPE + FP8 cast (fused: k_rmsnorm_rope_fp8)
  → Self-Attention SDPA (cuDNN BF16)
  → Output projection (FP8 GEMM, beta=0)
  → Gate residual (k_gate_res_ssts)
  → Cross-Attention (CA K/V built and cached in-engine, reused across denoise steps, cuDNN SDPA)
  → CA output projection + ungated residual
  → FFN LN + AdaLN → SiLU → FP8 GEMM up → FP8 GEMM down
  → FFN gate residual
Output x [seq, D] bf16
```

For GatedDeltaNet layers (3, 7, 11, 15, 19, 23, 27, 31), the self-attention is replaced by the GDN recurrence.

## Files

### Core Engine
| File | Purpose |
|---|---|
| `kairos_ext/csrc/kairos_engine.cu` | Main CUDA engine — all kernels, buffer management, pybind interface |
| `kairos_ext/kairos_engine_patch.py` | Python patcher — extracts weights, replaces DiT blocks with Runner stubs, JIT-compiles embedded SageAttention3 |
| `kairos_ext/generate_comparison.py` | Side-by-side video generation (stock vs engine) |
| `kairos_ext/_apex_shim.py` | Drop-in replacement for NVIDIA apex FusedRMSNorm |

### GDN Chunk Kernels
| File | Purpose | Status |
|---|---|---|
| `kairos_ext/csrc/gdn_chunk.cuh` | Prepare kernel (A matrix, w/u computation) + state_output kernel | Prepare: **FIXED** (forward sub sign bug). State_output: correct for state, wrong for output |
| `kairos_ext/csrc/gdn_chunk_h.cuh` | MMA-based state propagation kernel (tensor cores) | **WORKS** with fixed prepare |
| `kairos_ext/csrc/gdn_chunk_h_simple.cuh` | Simple warp-register state propagation (no MMA) | **WORKS**, slower than MMA version |
| `kairos_ext/csrc/gdn_chunk_o.cuh` | Output kernel (Q@H + causal attention) | **CORRECT**, bottleneck — needs MMA for speed |

### VAE Kernels
| File | Purpose |
|---|---|
| `kairos_ext/csrc/vae_fused_kernels.cu` | Fused RMSNorm+SiLU for VAE decoder |
| `kairos_ext/vae_patch.py` | Monkey-patches VAE with fused kernels |

### SageAttention3
| File | Purpose |
|---|---|
| `kairos/third_party/SageAttention/sageattention3_blackwell/` | FP4 attention for Blackwell — 3x faster than BF16 SDPA |
| `kairos/third_party/SageAttention/sageattention3_blackwell/sageattn3/blackwell/api.cu` | Vendored Sage3 Blackwell attention kernel entrypoint (`mha_fwd`) |
| `kairos/third_party/SageAttention/sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu` | Vendored FP4 quantization kernels used by Sage3 |
| `kairos_ext/csrc/kairos_engine.cu` | Engine-side Sage3 backend selector plus `sage3_py` and fully native `sage3_cpp` paths |

## Bugs Found and Fixed

### Bug 1: GDN Missing Output Kernel
**Symptom:** Engine produced "city lights" instead of waterfall — text conditioning was dead.
**Root cause:** The cuBLASLt chunk GDN pipeline called `k_gdn_chunk_h` which only propagates recurrent state but never computes the output tensor `o[t]`. The output buffer contained uninitialized memory.
**Fix:** Wrote `k_gdn_chunk_fwd_o` (gdn_chunk_o.cuh) — a separate output kernel matching the stock FLA Triton's `chunk_fwd_o`.

### Bug 2: GDN State Decay Ordering
**Symptom:** Even with the output kernel, the chunk pipeline produced near-black video.
**Root cause:** `k_gdn_chunk_state_output` applied `h *= exp(g_last)` decay BEFORE computing `v_new = u - w@h`. The Triton computes v_new using the UNDECAYED h, then decays. Wrong ordering meant v_new used the decayed (near-zero) state.
**Fix:** Wrote `k_gdn_chunk_h_simple` with correct ordering: save h_old → decay h → compute v_new from h_old → gate → update h.

### Bug 3: Forward Substitution Sign Convention
**Symptom:** Prepare kernel's w/u diverged from Triton (cos=0.58, 2x magnitude) causing state explosion across chunks.
**Root cause:** Our `(I-A)^{-1}` forward substitution used the wrong sign convention vs Triton's `solve_tril`. Triton negates A first, does forward substitution on -A, then adds identity. Our code used +A directly, producing A_inv with flipped signs for off-diagonal elements.
**Fix:** In `k_gdn_chunk_prepare`, negate A before forward substitution, matching Triton:
```c
// Negate lower triangle
for (i) for (j<i) s_A[i*BT+j] = -s_A[i*BT+j];
// Forward sub on -A
for (i>=2) for (j<i) s_A[i,j] += sum(s_A[i,k] * s_A[k,j])
// Add identity
for (i) s_A[i,i] = 1.0f;
```

### Bug 4: CA K/V Cache Dimension Mismatch (earlier)
**Symptom:** Cross-attention SDPA ran with wrong K/V sequence length.
**Root cause:** `prepare_ca()` was called before `set_ca_kv_cache_valid()`, so the SDPA was built for the default ctx=512 instead of actual ctx (7-117 tokens).
**Fix:** Reordered calls: set_ca_kv_cache_valid → prepare_ca.

### Bug 5: Context Length / Uninitialized Memory (earlier)
**Symptom:** Engine read garbage beyond actual encoder token count.
**Fix:** Added `g_actual_ctx` tracking and dynamic SDPA rebuilding.

### Bug 6: Embedded Sage3 JIT Targeted `sm_120` Instead Of `sm_120a`
**Symptom:** Native SageAttention3 embedding failed to compile inside `torch.utils.cpp_extension.load()` with PTX errors like block-scaled MMA / FP4 instructions unsupported on `.target sm_120`.
**Root cause:** PyTorch's default JIT arch detection picked plain `12.0` (`sm_120`) for the 5090, but SageAttention3 Blackwell kernels require the `a`-feature target (`sm_120a`).
**Fix:** In `kairos_engine_patch.py`, force `TORCH_CUDA_ARCH_LIST=12.0a` (or matching Blackwell `a` arch) during the engine JIT build, and guard the vendored Sage3 `PYBIND11_MODULE` blocks so the sources can be linked directly into `kairos_engine`.

## What Was Tried But Wasn't The Issue
- `--use_fast_math` compilation flag — removed and restored, no effect
- FP8 quantization precision — roundtrip cos=0.999+, not the cause
- SiLU activation in GDN conv — stock model DOES apply SiLU (confirmed at runtime)
- Fused gate+LN kernel — disabled, no improvement
- exp(gcum) underflow in prepare — the stock Triton handles gcum=-677 correctly, and our prepare matches (cos=0.999997 after sign fix)

## How to Run

### Prerequisites
```bash
# Environment
conda activate blackwell-prod
cd /home/mkrzus/kairos-sensenova

# Required env vars for CUDA includes
export CUDNN_INC=/home/mkrzus/Miniforge3/envs/blackwell-prod/lib/python3.11/site-packages/nvidia/cudnn/include
export CUDNN_LIB=/home/mkrzus/Miniforge3/envs/blackwell-prod/lib/python3.11/site-packages/nvidia/cudnn/lib
export NCCL_HOME=/home/mkrzus/Miniforge3/envs/blackwell-prod/lib/python3.11/site-packages/nvidia/nccl
export PYTHONPATH=kairos/third_party:$PYTHONPATH
export TORCHDYNAMO_DISABLE=1
```

### Generate Side-by-Side Comparison Video
```bash
python kairos_ext/generate_comparison.py \
    --config kairos/configs/kairos_4b_config_DMD.py \
    --input <(echo '{"prompt":"A beautiful waterfall","height":480,"width":640,"num_frames":41,"cfg_scale":5}') \
    --output output/comparison \
    --seed 42
```
Produces `stock.mp4`, `engine.mp4`, and `comparison.mp4` (side-by-side with FPS overlay).

### Engine-Only Inference
```python
from kairos_ext.generate_comparison import build_pipeline, patch_dit_engine
pipe, cfg = build_pipeline('kairos/configs/kairos_4b_config_DMD.py')
patch_dit_engine(pipe, max_seq=13200, ctx_len=512)
result = pipe(prompt="...", height=480, width=640, num_frames=41, cfg_scale=5, seed=42, tiled=True)
```

### Enable Chunk GDN (experimental, slower but correct)
```python
ext.set_gdn_cublas_chunk(True)  # after patch_engine
```

### Enable FP8 SDPA (requires cuDNN SM120 support — not yet available)
```bash
ENGINE_FP8_SDPA=1 python ...
```

### Install SageAttention3 Blackwell In `blackwell-prod`
Use the vendored tree that Kairos imports from:
```bash
conda activate blackwell-prod
cd /home/mkrzus/kairos-sensenova/kairos/third_party/SageAttention/sageattention3_blackwell
export SAGEATTN3_SKIP_TORCH_CUDA_CHECK=TRUE
python setup.py install
```

Why the env var is needed:
- PyTorch in `blackwell-prod` is `2.10.0+cu128`
- System `nvcc` is CUDA `13.1`
- `torch.utils.cpp_extension` rejects that mismatch by default even though the Blackwell build works in practice here

Important:
- Kairos prepends `kairos/third_party/SageAttention` on `sys.path`, so this vendored tree is the one that must be kept in sync with the compiled `fp4attn_cuda` / `fp4quant_cuda` extensions
- Installing from the separate top-level clone at `sageattention/sageattention3_blackwell` can produce a different Python/C++ ABI pairing than Kairos expects

### Engine SDPA Backend Selection
The engine now supports these self-attention backends for quadratic layers:

| Backend | Meaning |
|---|---|
| `cudnn` | Current production path: cuDNN BF16 SDPA |
| `torch` | Debug path: PyTorch `scaled_dot_product_attention` |
| `sage3_py` | Engine C++ calls Python `sageattn3.api.sageattn3_blackwell` directly |
| `sage3_cpp` | Engine C++ reproduces Sage3 preprocessing/quantization and calls embedded Sage3 entrypoints (`scaled_fp4_quant*`, `mha_fwd`) compiled directly into `kairos_engine` |

Select it with:
```bash
export KAIROS_ENGINE_SDPA_BACKEND=sage3_cpp
```

Or in the profiler:
```bash
python kairos_ext/profile_kairos.py --mode engine --engine-sdpa-backend sage3_cpp
```

Current smoke test at `seq=64`, `ctx=16`, synthetic weights:

| Engine SA backend | Full engine time |
|---|---|
| `cudnn` | 390.6 ms |
| `sage3_cpp` | 376.4 ms |
| `sage3_py` | 983.2 ms |

Correctness at the same small test size:

| Engine SA backend | Cosine vs stock |
|---|---|
| `sage3_cpp` | 0.998834 |
| `sage3_py` | 0.998820 |

Interpretation:
- `sage3_py` is functionally correct but too slow inside the engine loop because it pays Python-wrapper overhead on every quadratic layer
- `sage3_cpp` is now the useful path: it is fully embedded into `kairos_engine`, stays numerically good, and is already slightly faster than the cuDNN backend on this small full-engine test
- The remaining gap is now ordinary kernel/plumbing optimization, not a Python module boundary inside the quadratic self-attention path

Larger synthetic smoke test at `seq=2048`, `ctx=128`:

| Engine SA backend | Full engine time |
|---|---|
| `cudnn` | 513.8 ms |
| `sage3_cpp` | 429.9 ms |
| `sage3_py` | 672.4 ms |

At `seq=2048`, the fully native `sage3_cpp` path is now a clear win over the engine's cuDNN BF16 backend: `429.9 ms` vs `513.8 ms` (`1.20x`). The Python-wrapper path remains too slow to keep.

CA cache persistence is now enabled in the runner. For a real 4-step generation with a fixed prompt/context, the engine now rebuilds CA K/V once and prepares CA descriptors once, then reuses both for the remaining denoise steps.

## Optimization Targets (Priority Order)

### 1. Productionize Native SageAttention3
**Impact:** Highest near-term payoff on quadratic self-attention. Synthetic engine tests now show a real backend win (`1.20x` at `seq=2048`).
**Status:** Done for native engine integration; next step is end-to-end 480p/video benchmarking and rollout gating.
**Work needed:** Benchmark full inference/video with `KAIROS_ENGINE_SDPA_BACKEND=sage3_cpp`, validate on real prompts, and decide whether to switch the quadratic SA production default from `cudnn` to `sage3_cpp`.

### 2. MMA-Based chunk_fwd_o (estimated: chunk path from 0.83x → ~1.3x)
**Impact:** Would make chunk GDN competitive with or faster than recurrent, enabling further GDN optimizations.
**Status:** The warp-reduce based fwd_o is the bottleneck (O(BT²) per-pair score computations). MMA would be ~13x faster: 832 MMAs vs ~43K warp-reduce cycles per chunk.
**Work needed:** Port the Triton `chunk_fwd_kernel_o` pattern to CUDA MMA — compute Q@K^T and Q@H as block matmuls using mma.m16n8k16, then apply gating/mask and do A@v_new as a final matmul.

### 3. CUDA Graphs (estimated: 5-10% improvement)
**Impact:** Eliminates kernel launch overhead across the 32-layer forward pass.
**Status:** Infrastructure exists (`set_use_graph` flag) but not tested with the current pipeline.
**Work needed:** Capture the full forward pass as a CUDA graph. Requires fixed buffer addresses (already the case) and no Python in the loop during graph capture.

### 4. cuDNN FP8 SDPA (when available)
**Impact:** 2x on SDPA (42% of compute) → ~1.4x total.
**Status:** cuDNN 9.10.2 doesn't have FP8 SDPA engines for SM120 (GeForce Blackwell). Expected in a future cuDNN release.

### 5. Fuse Adjacent Elementwise Kernels
**Impact:** 5-10% from reduced memory traffic.
**Status:** Some kernels already fused (e.g., k_rmsnorm_rope_fp8, k_gate_res_ssts). More opportunities: fuse SiLU+FP8 cast with FFN up GEMM epilogue.

## Theoretical Performance Ceiling

At 480p 41 frames (534 TFLOPS total compute):
- **30% utilization:** ~70 FPS
- **50% utilization:** ~116 FPS
- Current (8.5 FPS) = ~4% utilization

The gap is dominated by:
1. Self-attention O(seq²) — 42% of compute, BF16 only → SageAttn3 would help
2. Kernel launch overhead — hundreds of small kernels → CUDA graphs would help
3. Python orchestration — Runner/pipeline has Python in the hot path
4. VAE decode — ~1.5s of 4.8s total (not optimized yet)
