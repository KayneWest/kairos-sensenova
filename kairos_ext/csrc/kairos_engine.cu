/**
 * kairos_engine.cu — Production CUDA engine for Kairos 4B transformer.
 *
 * Adapted from wan_engine_multi.cu (WAN 14B LingBot engine).
 *
 * Kairos 4B architecture:
 *   D=2560, NH=20, HD=128, FFN=10240, NL=32
 *   Hybrid attention: 24 quadratic + 8 GatedDeltaNet (every 4th layer)
 *   SiLU activation (not GELU)
 *   Affine cross_attn_norm on ALL layers
 *   CA applied to ALL tokens (no guidance split)
 *   CA residual is ungated (x += ca_out)
 *   Modulation order: [scale_sa, shift_sa, gate_sa, scale_ffn, shift_ffn, gate_ffn]
 *
 * Stripped from WAN: camera injection, Ulysses, TPP, SA KV cache.
 * Kept: FP8 SDPA, fused kernels, CUDA graphs, TP (dual-5090).
 *
 * forward() runs a RANGE of layers [start_layer, end_layer), skipping
 * GatedDeltaNet layers (handled in Python).
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_bf16.h>
#include <cudnn_frontend.h>
#include <nccl.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/util/host_tensor.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace fe = cudnn_frontend;
namespace py = pybind11;

// Embedded SageAttention3 entrypoints, compiled into this extension.
std::vector<at::Tensor>
mha_fwd(at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
        const at::Tensor &sfq, const at::Tensor &sfk, const at::Tensor &sfv,
        const at::Tensor &delta_s, int unpadded_k,
        c10::optional<at::Tensor> &out_,
        const float softmax_scale, bool is_causal,
        bool per_block_mean, bool is_bf16,
        int window_size_left, int window_size_right);

void mha_fwd_contiguous_bf16_raw(
        void *q_ptr,
        void *k_ptr,
        void *v_ptr,
        void *sfq_ptr,
        void *sfk_ptr,
        void *sfv_ptr,
        void *delta_s_ptr,
        void *out_ptr,
        float *softmax_lse_ptr,
        int *tile_count_semaphore_ptr,
        int batch_size,
        int num_heads,
        int seqlen_q,
        int seqlen_k,
        int unpadded_k,
        int unpacked_head_size,
        float softmax_scale,
        bool is_causal,
        bool per_block_mean,
        int window_size_left,
        int window_size_right,
        cudaStream_t stream);
void mha_fwd_strided_bf16_raw(
        void *q_ptr,
        void *k_ptr,
        void *v_ptr,
        void *sfq_ptr,
        void *sfk_ptr,
        void *sfv_ptr,
        void *delta_s_ptr,
        void *out_ptr,
        float *softmax_lse_ptr,
        int *tile_count_semaphore_ptr,
        int batch_size,
        int num_heads,
        int seqlen_q,
        int seqlen_k,
        int unpadded_k,
        int unpacked_head_size,
        int64_t out_batch_stride,
        int64_t out_head_stride,
        int64_t out_row_stride,
        float softmax_scale,
        bool is_causal,
        bool per_block_mean,
        int window_size_left,
        int window_size_right,
        cudaStream_t stream);

void scaled_fp4_quant(torch::Tensor const& input,
                      torch::Tensor const& output,
                      torch::Tensor const& output_sf,
                      int tensor_layout);
void scaled_fp4_quant_permute(torch::Tensor const& input,
                              torch::Tensor const& output,
                              torch::Tensor const& output_sf,
                              int tensor_layout);
void scaled_fp4_quant_trans(torch::Tensor const& input,
                            torch::Tensor const& output,
                            torch::Tensor const& output_sf,
                            int tensor_layout);

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
quantize_fp4_rowwise_bf16(torch::Tensor in_t);

void scaled_fp4_quant_bf16_raw_contig(
    const nv_bfloat16* input,
    uint8_t* output,
    uint8_t* output_sf,
    int batch_size,
    int num_heads,
    int num_tokens,
    int head_dim,
    cudaStream_t stream);
void scaled_fp4_quant_bf16_raw_strided_groupmean(
    const nv_bfloat16* input,
    uint8_t* output,
    uint8_t* output_sf,
    nv_bfloat16* output_mean,
    int batch_size,
    int num_heads,
    int num_tokens,
    int head_dim,
    int stride_bz_input,
    int stride_h_input,
    int stride_seq_input,
    cudaStream_t stream);
void scaled_fp4_quant_permute_bf16_raw_contig(
    const nv_bfloat16* input,
    uint8_t* output,
    uint8_t* output_sf,
    int batch_size,
    int num_heads,
    int num_tokens,
    int head_dim,
    cudaStream_t stream);
void scaled_fp4_quant_permute_bf16_raw_strided_centered(
    const nv_bfloat16* input,
    uint8_t* output,
    uint8_t* output_sf,
    const nv_bfloat16* mean,
    nv_bfloat16* centered,
    int batch_size,
    int num_heads,
    int num_tokens,
    int head_dim,
    int stride_bz_input,
    int stride_h_input,
    int stride_seq_input,
    cudaStream_t stream);
void scaled_fp4_quant_trans_bf16_raw_contig(
    const nv_bfloat16* input,
    uint8_t* output,
    uint8_t* output_sf,
    int batch_size,
    int num_heads,
    int num_tokens,
    int head_dim,
    cudaStream_t stream);
void scaled_fp4_quant_trans_bf16_raw_strided(
    const nv_bfloat16* input,
    uint8_t* output,
    uint8_t* output_sf,
    int batch_size,
    int num_heads,
    int num_tokens,
    int head_dim,
    int stride_bz_input,
    int stride_h_input,
    int stride_seq_input,
    cudaStream_t stream);

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { TORCH_CHECK(false, "CUDA ", __LINE__, ": ", cudaGetErrorString(e)); } } while(0)
#define CKBL(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { TORCH_CHECK(false, "cublasLt ", __LINE__, ": status=", (int)s); } } while(0)
#define CKNCCL(x) do { ncclResult_t r = (x); if (r != ncclSuccess) { TORCH_CHECK(false, "NCCL ", __LINE__, ": ", ncclGetErrorString(r)); } } while(0)

// ============================================================
// CUDA kernels (architecture-agnostic)
// ============================================================

// Proper BF16 -> FP8 E4M3 conversion
__device__ __forceinline__ unsigned char bf16_to_fp8_byte(__nv_bfloat16 v) {
    __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(__bfloat162float(v));
    return *reinterpret_cast<unsigned char*>(&fp8);
}

__device__ __forceinline__ unsigned char float_to_ue4m3_byte(float v) {
    cutlass::float_ue4m3_t fp8(v);
    return fp8.raw();
}

__device__ __forceinline__ float ue4m3_byte_to_float(unsigned char v) {
    return static_cast<float>(cutlass::float_ue4m3_t::bitcast(v));
}

__device__ __forceinline__ float warp_reduce(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

static inline int round_up_int(int x, int m) {
    return ((x + m - 1) / m) * m;
}

static inline int fp4_scale_rows(int rows) {
    return round_up_int(rows, 128);
}

static inline int fp4_scale_cols(int cols) {
    TORCH_CHECK(cols % 16 == 0, "FP4 requires inner dimension divisible by 16, got ", cols);
    return round_up_int(cols / 16, 4);
}

static inline unsigned char fp4_scale_one_byte() {
    return cutlass::float_ue4m3_t(1.0f).raw();
}

static inline torch::Tensor swizzle_fp4_scales_rowwise(const torch::Tensor& scales) {
    TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
    TORCH_CHECK(scales.scalar_type() == torch::kUInt8, "scales must be uint8/E4M3 bytes");
    TORCH_CHECK(scales.dim() == 2, "scales must be rank-2");
    int rows = (int)scales.size(0);
    int cols = (int)scales.size(1);
    TORCH_CHECK(rows % 128 == 0, "FP4 scale rows must be padded to 128, got ", rows);
    TORCH_CHECK(cols % 4 == 0, "FP4 scale cols must be padded to 4, got ", cols);
    return scales.view({rows / 128, 4, 32, cols / 4, 4})
        .permute({0, 3, 2, 1, 4})
        .contiguous()
        .view_as(scales);
}

__device__ __forceinline__ size_t fp4_swizzled_scale_index(int row, int scale_col, int scale_cols) {
    int a = row >> 7;
    int b = (row >> 5) & 3;
    int c = row & 31;
    int d = scale_col >> 2;
    int e = scale_col & 3;
    return ((((size_t)a * (size_t)(scale_cols >> 2) + d) * 32 + c) * 4 + b) * 4 + e;
}

__global__ void k_quantize_rowwise_bf16_to_fp4(
    const __nv_bfloat16* __restrict__ in,
    uint8_t* __restrict__ out_packed,
    uint8_t* __restrict__ out_scales,
    int rows,
    int cols,
    int scales_ld)
{
    constexpr float FP4_MAX = 6.0f;
    int row = blockIdx.y;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    if (row >= rows) return;

    int group32 = blockIdx.x * warps_per_block + warp;
    int col_base = group32 * 32;
    if (col_base >= cols) return;
    int block16_0 = group32 * 2;
    int block16_1 = block16_0 + 1;
    bool valid0 = (col_base + 15) < cols;
    bool valid1 = (col_base + 31) < cols;

    float v = 0.0f;
    int col = col_base + lane;
    if (col < cols) {
        v = __bfloat162float(in[(size_t)row * cols + col]);
    }
    float amax = fabsf(v);
    for (int off = 8; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    }
    float amax0 = __shfl_sync(0xffffffff, amax, 0);
    float amax1 = __shfl_sync(0xffffffff, amax, 16);

    float decode_scale0 = (amax0 > 0.0f) ? (amax0 / FP4_MAX) : 1.0f;
    float decode_scale1 = (amax1 > 0.0f) ? (amax1 / FP4_MAX) : 1.0f;
    float encode_scale0 = 1.0f / decode_scale0;
    float encode_scale1 = 1.0f / decode_scale1;

    if (lane == 0) {
        out_scales[(size_t)row * scales_ld + block16_0] = float_to_ue4m3_byte(decode_scale0);
    }
    if (lane == 16 && valid1) {
        out_scales[(size_t)row * scales_ld + block16_1] = float_to_ue4m3_byte(decode_scale1);
    }

    int pair_lane = lane & 7;
    float g0x = __shfl_sync(0xffffffff, v, pair_lane * 2 + 0) * encode_scale0;
    float g0y = __shfl_sync(0xffffffff, v, pair_lane * 2 + 1) * encode_scale0;
    float g1x = __shfl_sync(0xffffffff, v, 16 + pair_lane * 2 + 0) * encode_scale1;
    float g1y = __shfl_sync(0xffffffff, v, 16 + pair_lane * 2 + 1) * encode_scale1;

    if (lane < 8) {
        float2 in_pair{g0x, g0y};
        __nv_fp4x2_storage_t packed = __nv_cvt_float2_to_fp4x2(in_pair, __NV_E2M1, cudaRoundNearest);
        out_packed[(size_t)row * (cols / 2) + block16_0 * 8 + lane] = packed;
    } else if (lane < 16 && valid1) {
        float2 in_pair{g1x, g1y};
        __nv_fp4x2_storage_t packed = __nv_cvt_float2_to_fp4x2(in_pair, __NV_E2M1, cudaRoundNearest);
        out_packed[(size_t)row * (cols / 2) + block16_1 * 8 + (lane - 8)] = packed;
    }
}

// Quantize a logical [rows, cols] row-major BF16 matrix into packed FP4 bytes
// laid out as a ColumnMajor operand. Unlike the older 2-pass path, this computes
// the per-row block scale and the packed bytes from the same exact amax so it
// matches CUTLASS host packing.
__global__ void k_quantize_colmajor_bf16_to_fp4(
    const __nv_bfloat16* __restrict__ in,
    uint8_t* __restrict__ out_packed,
    uint8_t* __restrict__ out_scales,
    int rows,
    int cols,
    int scales_ld)
{
    constexpr float FP4_MAX = 6.0f;
    int row_pair = blockIdx.y;
    int block16 = blockIdx.x;
    int lane = threadIdx.x & 31;
    int row0 = row_pair * 2;
    int row1 = row0 + 1;
    if (row1 >= rows) return;

    int col0 = block16 * 16;
    if (col0 >= cols || lane >= 16) return;

    float v0 = __bfloat162float(in[(size_t)row0 * cols + col0 + lane]);
    float v1 = __bfloat162float(in[(size_t)row1 * cols + col0 + lane]);
    float amax0 = fabsf(v0);
    float amax1 = fabsf(v1);
    for (int off = 8; off > 0; off >>= 1) {
        amax0 = fmaxf(amax0, __shfl_xor_sync(0xFFFF, amax0, off, 16));
        amax1 = fmaxf(amax1, __shfl_xor_sync(0xFFFF, amax1, off, 16));
    }
    amax0 = __shfl_sync(0xFFFF, amax0, 0, 16);
    amax1 = __shfl_sync(0xFFFF, amax1, 0, 16);

    float scale0 = (amax0 > 0.0f) ? (amax0 / FP4_MAX) : 1.0f;
    float scale1 = (amax1 > 0.0f) ? (amax1 / FP4_MAX) : 1.0f;
    float inv0 = 1.0f / scale0;
    float inv1 = 1.0f / scale1;

    if (lane == 0) {
        out_scales[(size_t)row0 * scales_ld + block16] = float_to_ue4m3_byte(scale0);
        out_scales[(size_t)row1 * scales_ld + block16] = float_to_ue4m3_byte(scale1);
    }

    int col = col0 + lane;
    float2 in_pair;
    in_pair.x = v0 * inv0;
    in_pair.y = v1 * inv1;
    __nv_fp4x2_storage_t packed = __nv_cvt_float2_to_fp4x2(in_pair, __NV_E2M1, cudaRoundNearest);
    out_packed[(size_t)col * (rows / 2) + row_pair] = packed;
}

// On-the-fly AdaLN LN kernel. 256 threads/block.
// Launch with D*sizeof(__nv_bfloat16) smem for single-HBM-read staging.
// Pass out=nullptr to skip bf16 writeback.
__global__ void k_ln_adaln_ssts(
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
    __nv_fp8_e4m3* __restrict__ out_fp8,
    const float* __restrict__ ssts_scale, const float* __restrict__ ssts_shift,
    const float* __restrict__ temb_scale, const float* __restrict__ temb_shift,
    int D, int temb_row_stride) {
    extern __shared__ __nv_bfloat16 srow_ln_bf[];
    __shared__ float sm[16];
    int r = blockIdx.x, t = threadIdx.x, w = t >> 5, l = t & 31;
    const __nv_bfloat16* xi = x + (size_t)r * D;
    const float* tsc = temb_scale + (size_t)r * temb_row_stride;
    const float* tsh = temb_shift + (size_t)r * temb_row_stride;
    // Pass 1: HBM->SMEM + compute sum & sumsq
    float s1 = 0, s2 = 0;
    for (int i = t; i < D; i += 256) {
        __nv_bfloat16 bv = xi[i];
        srow_ln_bf[i] = bv;
        float v = __bfloat162float(bv);
        s1 += v; s2 += v * v;
    }
    s1 = warp_reduce(s1); s2 = warp_reduce(s2);
    if (l == 0) { sm[w] = s1; sm[w + 8] = s2; }
    __syncthreads();
    if (t == 0) { float a = 0, b = 0; for (int i = 0; i < 8; i++) { a += sm[i]; b += sm[i + 8]; } sm[0] = a; sm[1] = b; }
    __syncthreads();
    float mean = sm[0] / D, rstd = rsqrtf(sm[1] / D - mean * mean + 1e-6f);
    unsigned char* fi = (unsigned char*)(out_fp8 + (size_t)r * D);
    // Pass 2: apply LN+AdaLN from SMEM
    if (out) {
        __nv_bfloat16* oi = out + (size_t)r * D;
        for (int i = t; i < D; i += 256) {
            float sc = ssts_scale[i] + tsc[i];
            float sh = ssts_shift[i] + tsh[i];
            float v = (__bfloat162float(srow_ln_bf[i]) - mean) * rstd;
            __nv_bfloat16 bf = __float2bfloat16(v * (1.f + sc) + sh);
            oi[i] = bf;
            fi[i] = bf16_to_fp8_byte(bf);
        }
    } else {
        for (int i = t; i < D; i += 256) {
            float sc = ssts_scale[i] + tsc[i];
            float sh = ssts_shift[i] + tsh[i];
            float v = (__bfloat162float(srow_ln_bf[i]) - mean) * rstd;
            __nv_bfloat16 bf = __float2bfloat16(v * (1.f + sc) + sh);
            fi[i] = bf16_to_fp8_byte(bf);
        }
    }
}

// Gated residual: x += y * (ssts_gate + temb_gate)
__global__ void k_gate_res_ssts(
    __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ y,
    const float* __restrict__ ssts_gate, const float* __restrict__ temb_gate,
    int N, int D, int temb_row_stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total8 = (N * D) >> 3;
    for (int i8 = tid; i8 < total8; i8 += stride) {
        int base = i8 << 3;
        int r = base / D, c = base - r * D;
        uint4 xv = *reinterpret_cast<const uint4*>(x + base);
        uint4 yv = *reinterpret_cast<const uint4*>(y + base);
        const float* tg = temb_gate + (size_t)r * temb_row_stride + c;
        const float* sg = ssts_gate + c;
        __nv_bfloat16* xp = (__nv_bfloat16*)&xv;
        __nv_bfloat16* yp = (__nv_bfloat16*)&yv;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float g = sg[k] + tg[k];
            float v = __bfloat162float(xp[k]) + __bfloat162float(yp[k]) * g;
            xp[k] = __float2bfloat16(v);
        }
        *reinterpret_cast<uint4*>(x + base) = xv;
    }
}

// Gated residual + FP8 output: x += y * (ssts_gate + temb_gate); also write fp8
__global__ void k_gate_res_ssts_fp8(
    __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ y,
    const float* __restrict__ ssts_gate, const float* __restrict__ temb_gate,
    __nv_fp8_e4m3* __restrict__ out_fp8,
    int N, int D, int temb_row_stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total8 = (N * D) >> 3;
    for (int i8 = tid; i8 < total8; i8 += stride) {
        int base = i8 << 3;
        int r = base / D, c = base - r * D;
        uint4 xv = *reinterpret_cast<const uint4*>(x + base);
        uint4 yv = *reinterpret_cast<const uint4*>(y + base);
        const float* tg = temb_gate + (size_t)r * temb_row_stride + c;
        const float* sg = ssts_gate + c;
        __nv_bfloat16* xp = (__nv_bfloat16*)&xv;
        __nv_bfloat16* yp = (__nv_bfloat16*)&yv;
        unsigned char fp8[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float g = sg[k] + tg[k];
            float v = __bfloat162float(xp[k]) + __bfloat162float(yp[k]) * g;
            __nv_bfloat16 bf = __float2bfloat16(v);
            xp[k] = bf;
            fp8[k] = bf16_to_fp8_byte(bf);
        }
        *reinterpret_cast<uint4*>(x + base) = xv;
        *reinterpret_cast<uint2*>((unsigned char*)out_fp8 + base) = *reinterpret_cast<uint2*>(fp8);
    }
}

// Fused gated residual + LayerNorm + AdaLN (Optimization 3)
// Launch: <<<seq, 256, (D + 16) * sizeof(float)>>>
__global__ void k_gate_res_ln_adaln_ssts(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    const float* __restrict__ ssts_gate,
    const float* __restrict__ temb_gate,
    const float* __restrict__ ssts_scale,
    const float* __restrict__ ssts_shift,
    const float* __restrict__ temb_scale,
    const float* __restrict__ temb_shift,
    __nv_bfloat16* __restrict__ ln_out,
    __nv_fp8_e4m3* __restrict__ ln_out_fp8,
    int D, int temb_row_stride) {
    extern __shared__ float smem[];
    float* row = smem;
    float* sm = smem + D;
    int r = blockIdx.x, t = threadIdx.x, w = t >> 5, l = t & 31;
    __nv_bfloat16* xi = x + (size_t)r * D;
    const __nv_bfloat16* yi = y + (size_t)r * D;
    int temb_off = r * temb_row_stride;
    // Pass 1: gated residual + cache in SMEM + compute sum/sumsq for LN
    float s1 = 0, s2 = 0;
    for (int i = t; i < D; i += 256) {
        float g = ssts_gate[i] + temb_gate[temb_off + i];
        float v = __bfloat162float(xi[i]) + __bfloat162float(yi[i]) * g;
        row[i] = v;
        xi[i] = __float2bfloat16(v);
        s1 += v; s2 += v * v;
    }
    s1 = warp_reduce(s1); s2 = warp_reduce(s2);
    if (l == 0) { sm[w] = s1; sm[w + 8] = s2; }
    __syncthreads();
    if (t == 0) { float a = 0, b = 0; for (int i = 0; i < 8; i++) { a += sm[i]; b += sm[i + 8]; } sm[0] = a; sm[1] = b; }
    __syncthreads();
    float mean = sm[0] / D, rstd = rsqrtf(sm[1] / D - mean * mean + 1e-6f);
    // Pass 2: LN + AdaLN + write bf16/fp8 outputs
    __nv_bfloat16* oi = ln_out ? (ln_out + (size_t)r * D) : nullptr;
    __nv_fp8_e4m3* fi = ln_out_fp8 + (size_t)r * D;
    for (int i = t; i < D; i += 256) {
        float v = (row[i] - mean) * rstd;
        float sc = ssts_scale[i] + temb_scale[temb_off + i];
        float sh = ssts_shift[i] + temb_shift[temb_off + i];
        float out = v * (1.f + sc) + sh;
        __nv_bfloat16 bf = __float2bfloat16(out);
        if (oi) oi[i] = bf;
        fi[i] = __nv_fp8_e4m3(__bfloat162float(bf));
    }
}

// Affine LN (has weight + bias) -> BF16 + FP8 output. Used for cross_attn_norm.
// Pass out=nullptr to skip bf16 writeback.
__global__ void k_ln_affine(
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
    __nv_fp8_e4m3* __restrict__ out_fp8,
    const float* __restrict__ w, const float* __restrict__ b, int D) {
    __shared__ float sm[16];
    int r = blockIdx.x, t = threadIdx.x, ww = t >> 5, l = t & 31;
    const __nv_bfloat16* xi = x + (size_t)r * D;
    float s1 = 0, s2 = 0;
    for (int i = t; i < D; i += 256) { float v = __bfloat162float(xi[i]); s1 += v; s2 += v * v; }
    s1 = warp_reduce(s1); s2 = warp_reduce(s2);
    if (l == 0) { sm[ww] = s1; sm[ww + 8] = s2; }
    __syncthreads();
    if (t == 0) { float a = 0, c = 0; for (int i = 0; i < 8; i++) { a += sm[i]; c += sm[i + 8]; } sm[0] = a; sm[1] = c; }
    __syncthreads();
    float mean = sm[0] / D, rstd = rsqrtf(sm[1] / D - mean * mean + 1e-6f);
    __nv_bfloat16* oi = out ? (out + (size_t)r * D) : nullptr;
    unsigned char* fi = (unsigned char*)(out_fp8 + (size_t)r * D);
    for (int i = t; i < D; i += 256) {
        __nv_bfloat16 bf = __float2bfloat16((__bfloat162float(xi[i]) - mean) * rstd * w[i] + b[i]);
        if (oi) oi[i] = bf;
        fi[i] = bf16_to_fp8_byte(bf);
    }
}

// Whole-D RMSNorm. Launch as <<<seq, 256>>>.
__global__ void k_rmsnorm(
    __nv_bfloat16* __restrict__ io, const float* __restrict__ w, int D) {
    __shared__ float sm[8];
    int r = blockIdx.x, t = threadIdx.x, ww = t >> 5, l = t & 31;
    __nv_bfloat16* xi = io + (size_t)r * D;
    float s = 0;
    for (int i = t; i < D; i += 256) { float v = __bfloat162float(xi[i]); s += v * v; }
    s = warp_reduce(s);
    if (l == 0) sm[ww] = s;
    __syncthreads();
    if (t == 0) { float a = 0; for (int i = 0; i < 8; i++) a += sm[i]; sm[0] = a; }
    __syncthreads();
    float rstd = rsqrtf(sm[0] / D + 1e-6f);
    for (int i = t; i < D; i += 256)
        xi[i] = __float2bfloat16(__bfloat162float(xi[i]) * rstd * w[i]);
}

// Fused RMSNorm + RoPE. Launch as <<<seq, 256, D*sizeof(bf16)>>>.
__global__ void k_rmsnorm_rope(
    __nv_bfloat16* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ freqs_cis, int nh, int hd) {
    extern __shared__ __nv_bfloat16 srow_rr[];
    int D = nh * hd;
    __shared__ float sm[8];
    int r = blockIdx.x, t = threadIdx.x, ww = t >> 5, l = t & 31;
    __nv_bfloat16* xi = x + (size_t)r * D;
    float s = 0;
    for (int i = t; i < D; i += 256) {
        __nv_bfloat16 bv = xi[i];
        srow_rr[i] = bv;
        float v = __bfloat162float(bv);
        s += v * v;
    }
    s = warp_reduce(s);
    if (l == 0) sm[ww] = s;
    __syncthreads();
    if (t == 0) { float a = 0; for (int i = 0; i < 8; i++) a += sm[i]; sm[0] = a; }
    __syncthreads();
    float rstd = rsqrtf(sm[0] / D + 1e-6f);
    for (int i = t; i < D; i += 256) {
        srow_rr[i] = __float2bfloat16(__bfloat162float(srow_rr[i]) * rstd * w[i]);
    }
    __syncthreads();
    const float* fc = freqs_cis + (size_t)r * 2 * hd;
    int half_hd = hd >> 1;
    int pairs = D >> 1;
    for (int i = t; i < pairs; i += 256) {
        int h = i / half_hd;
        int kk = i % half_hd;
        int base = h * hd + (kk << 1);
        float c = fc[kk << 1];
        float sinv = fc[hd + (kk << 1) + 1];
        float a = __bfloat162float(srow_rr[base]);
        float b = __bfloat162float(srow_rr[base + 1]);
        xi[base]     = __float2bfloat16(a * c - b * sinv);
        xi[base + 1] = __float2bfloat16(a * sinv + b * c);
    }
}

// Fused RMSNorm + RoPE + FP8 output. Writes BOTH bf16 and fp8.
// Launch: <<<seq, 256, D * sizeof(bf16)>>>
__global__ void k_rmsnorm_rope_fp8(
    __nv_bfloat16* __restrict__ x,
    __nv_fp8_e4m3* __restrict__ x_fp8,
    const float* __restrict__ w,
    const float* __restrict__ freqs_cis,
    int nh, int hd) {
    extern __shared__ __nv_bfloat16 srow[];
    int D = nh * hd;
    __shared__ float sm[8];
    int r = blockIdx.x, t = threadIdx.x, ww = t >> 5, l = t & 31;
    __nv_bfloat16* xi = x + (size_t)r * D;
    __nv_fp8_e4m3* xi_fp8 = x_fp8 + (size_t)r * D;
    float s = 0;
    for (int i = t; i < D; i += 256) {
        __nv_bfloat16 bv = xi[i];
        srow[i] = bv;
        float v = __bfloat162float(bv);
        s += v * v;
    }
    s = warp_reduce(s);
    if (l == 0) sm[ww] = s;
    __syncthreads();
    if (t == 0) { float a = 0; for (int i = 0; i < 8; i++) a += sm[i]; sm[0] = a; }
    __syncthreads();
    float rstd = rsqrtf(sm[0] / D + 1e-6f);
    for (int i = t; i < D; i += 256) {
        srow[i] = __float2bfloat16(__bfloat162float(srow[i]) * rstd * w[i]);
    }
    __syncthreads();
    const float* fc = freqs_cis + (size_t)r * 2 * hd;
    int half_hd = hd >> 1;
    int pairs = D >> 1;
    for (int i = t; i < pairs; i += 256) {
        int h = i / half_hd;
        int kk = i % half_hd;
        int base = h * hd + (kk << 1);
        float c = fc[kk << 1];
        float sinv = fc[hd + (kk << 1) + 1];
        float a = __bfloat162float(srow[base]);
        float b = __bfloat162float(srow[base + 1]);
        float out0 = a * c - b * sinv;
        float out1 = a * sinv + b * c;
        xi[base]     = __float2bfloat16(out0);
        xi[base + 1] = __float2bfloat16(out1);
        xi_fp8[base]     = __nv_fp8_e4m3(out0);
        xi_fp8[base + 1] = __nv_fp8_e4m3(out1);
    }
}

// Vectorized BF16 -> FP8 conversion
__global__ void k_bf16_to_fp8(__nv_fp8_e4m3* out, const __nv_bfloat16* in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n8 = n / 8;
    const int4* in_v = reinterpret_cast<const int4*>(in);
    uint2* out_v = reinterpret_cast<uint2*>(out);
    for (int i = tid; i < n8; i += stride) {
        int4 iv = in_v[i];
        const __nv_bfloat16* bb = reinterpret_cast<const __nv_bfloat16*>(&iv);
        unsigned char c[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            __nv_fp8_e4m3 f(__bfloat162float(bb[k]));
            c[k] = *reinterpret_cast<unsigned char*>(&f);
        }
        uint2 ou;
        ou.x = (uint32_t)c[0] | ((uint32_t)c[1] << 8) | ((uint32_t)c[2] << 16) | ((uint32_t)c[3] << 24);
        ou.y = (uint32_t)c[4] | ((uint32_t)c[5] << 8) | ((uint32_t)c[6] << 16) | ((uint32_t)c[7] << 24);
        out_v[i] = ou;
    }
    int start = n8 * 8;
    for (int i = start + tid; i < n; i += stride) {
        out[i] = __nv_fp8_e4m3(__bfloat162float(in[i]));
    }
}

// Fused SiLU activation + BF16->FP8 conversion (for Kairos FFN activation)
// Reads bf16 input, writes fp8 output only (bf16 input NOT written back).
__global__ void k_silu_to_fp8(const __nv_bfloat16* __restrict__ io,
                               __nv_fp8_e4m3* __restrict__ out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n8 = n >> 3;
    for (int i8 = tid; i8 < n8; i8 += stride) {
        int base = i8 << 3;
        uint4 xv = *reinterpret_cast<const uint4*>(io + base);
        const __nv_bfloat16* p = (const __nv_bfloat16*)&xv;
        unsigned char fp8[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float v = __bfloat162float(p[k]);
            float s = __fdividef(v, 1.f + __expf(-v));  // silu = x * sigmoid(x)
            fp8[k] = bf16_to_fp8_byte(__float2bfloat16(s));
        }
        *reinterpret_cast<uint2*>((unsigned char*)out + base) = *reinterpret_cast<uint2*>(fp8);
    }
    // Tail
    for (int i = (n8 << 3) + tid; i < n; i += stride) {
        float v = __bfloat162float(io[i]);
        float s = __fdividef(v, 1.f + __expf(-v));
        ((unsigned char*)out)[i] = bf16_to_fp8_byte(__float2bfloat16(s));
    }
}

// Fused SiLU activation + BF16->FP4 conversion with direct write into the
// CUTLASS SM120 rowwise packed-data + swizzled-scale layout expected by FFN2.
// Grid: dim3(cols/16, rows), block: 32 threads (one warp per 16-wide block).
__global__ void k_silu_to_fp4_rowwise_swizzled(
    const __nv_bfloat16* __restrict__ io,
    uint8_t* __restrict__ out_packed,
    uint8_t* __restrict__ out_scales_swizzled,
    int rows,
    int cols,
    int scale_cols)
{
    constexpr float FP4_MAX = 6.0f;
    int row = blockIdx.y;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    if (row >= rows) return;

    int group32 = blockIdx.x * warps_per_block + warp;
    int col_base = group32 * 32;
    if (col_base >= cols) return;
    int block16_0 = group32 * 2;
    int block16_1 = block16_0 + 1;
    bool valid0 = (col_base + 15) < cols;
    bool valid1 = (col_base + 31) < cols;

    float s = 0.0f;
    int col = col_base + lane;
    if (col < cols) {
        float v = __bfloat162float(io[(size_t)row * cols + col]);
        s = __fdividef(v, 1.f + __expf(-v));
    }
    float amax = fabsf(s);
    for (int off = 8; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    }
    float amax0 = __shfl_sync(0xffffffff, amax, 0);
    float amax1 = __shfl_sync(0xffffffff, amax, 16);

    float decode_scale0 = (amax0 > 0.0f) ? (amax0 / FP4_MAX) : 1.0f;
    float decode_scale1 = (amax1 > 0.0f) ? (amax1 / FP4_MAX) : 1.0f;
    float encode_scale0 = 1.0f / decode_scale0;
    float encode_scale1 = 1.0f / decode_scale1;

    if (lane == 0) {
        size_t scale_idx = fp4_swizzled_scale_index(row, block16_0, scale_cols);
        out_scales_swizzled[scale_idx] = float_to_ue4m3_byte(decode_scale0);
    }
    if (lane == 16 && valid1) {
        size_t scale_idx = fp4_swizzled_scale_index(row, block16_1, scale_cols);
        out_scales_swizzled[scale_idx] = float_to_ue4m3_byte(decode_scale1);
    }

    int pair_lane = lane & 7;
    float s0x = __shfl_sync(0xffffffff, s, pair_lane * 2 + 0) * encode_scale0;
    float s0y = __shfl_sync(0xffffffff, s, pair_lane * 2 + 1) * encode_scale0;
    float s1x = __shfl_sync(0xffffffff, s, 16 + pair_lane * 2 + 0) * encode_scale1;
    float s1y = __shfl_sync(0xffffffff, s, 16 + pair_lane * 2 + 1) * encode_scale1;

    if (lane < 8) {
        float2 pair{s0x, s0y};
        __nv_fp4x2_storage_t packed = __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
        out_packed[(size_t)row * (cols / 2) + block16_0 * 8 + lane] = packed;
    } else if (lane < 16 && valid1) {
        float2 pair{s1x, s1y};
        __nv_fp4x2_storage_t packed = __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
        out_packed[(size_t)row * (cols / 2) + block16_1 * 8 + (lane - 8)] = packed;
    }
}

// In-place SiLU on BF16 buffer (for force_bf16_gemms debugging mode)
__global__ void k_silu_bf16(__nv_bfloat16* io, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __bfloat162float(io[i]);
        v = v / (1.f + expf(-v));
        io[i] = __float2bfloat16(v);
    }
}

// Vectorized elementwise add in-place: x[i] += y[i]
__global__ void k_add_vec(__nv_bfloat16* __restrict__ x,
                          const __nv_bfloat16* __restrict__ y, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total8 = total >> 3;
    for (int i8 = tid; i8 < total8; i8 += stride) {
        int base = i8 << 3;
        uint4 xv = *reinterpret_cast<const uint4*>(x + base);
        uint4 yv = *reinterpret_cast<const uint4*>(y + base);
        __nv_bfloat16* xp = (__nv_bfloat16*)&xv;
        __nv_bfloat16* yp = (__nv_bfloat16*)&yv;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            xp[k] = __float2bfloat16(__bfloat162float(xp[k]) + __bfloat162float(yp[k]));
        }
        *reinterpret_cast<uint4*>(x + base) = xv;
    }
    for (int i = (total8 << 3) + tid; i < total; i += stride) {
        x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(y[i]));
    }
}

// Vectorized add bias in-place: x[r, :] += bias[:]
__global__ void k_add_bias_vec(__nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ bias,
                                int N, int D) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total8 = (N * D) >> 3;
    for (int i8 = tid; i8 < total8; i8 += stride) {
        int base = i8 << 3;
        int c = base - (base / D) * D;
        uint4 xv = *reinterpret_cast<const uint4*>(x + base);
        uint4 bv = *reinterpret_cast<const uint4*>(bias + c);
        __nv_bfloat16* xp = (__nv_bfloat16*)&xv;
        __nv_bfloat16* bp = (__nv_bfloat16*)&bv;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            xp[k] = __float2bfloat16(__bfloat162float(xp[k]) + __bfloat162float(bp[k]));
        }
        *reinterpret_cast<uint4*>(x + base) = xv;
    }
    for (int i = (total8 << 3) + tid; i < N * D; i += stride) {
        int c = i - (i / D) * D;
        x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(bias[c]));
    }
}

// Fused (bias + residual): x[r, c] += y[r, c] + bias[c]
__global__ void k_add_bias_res(__nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ y,
                                const __nv_bfloat16* __restrict__ bias,
                                int N, int D) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total8 = (N * D) >> 3;
    for (int i8 = tid; i8 < total8; i8 += stride) {
        int base = i8 << 3;
        int c = base - (base / D) * D;
        uint4 xv = *reinterpret_cast<const uint4*>(x + base);
        uint4 yv = *reinterpret_cast<const uint4*>(y + base);
        uint4 bv = *reinterpret_cast<const uint4*>(bias + c);
        __nv_bfloat16* xp = (__nv_bfloat16*)&xv;
        __nv_bfloat16* yp = (__nv_bfloat16*)&yv;
        __nv_bfloat16* bp = (__nv_bfloat16*)&bv;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            xp[k] = __float2bfloat16(__bfloat162float(xp[k])
                                    + __bfloat162float(yp[k])
                                    + __bfloat162float(bp[k]));
        }
        *reinterpret_cast<uint4*>(x + base) = xv;
    }
}

// TP RMSNorm phase 1: compute partial sum(x^2) per row -> [seq] floats
__global__ void k_partial_sumsq(const __nv_bfloat16* __restrict__ x,
                                 float* __restrict__ sum_sq_out, int Dpr) {
    __shared__ float sm[8];
    int r = blockIdx.x, t = threadIdx.x, ww = t >> 5, l = t & 31;
    const __nv_bfloat16* xi = x + (size_t)r * Dpr;
    float s = 0;
    for (int i = t; i < Dpr; i += 256) { float v = __bfloat162float(xi[i]); s += v * v; }
    s = warp_reduce(s);
    if (l == 0) sm[ww] = s;
    __syncthreads();
    if (t == 0) { float a = 0; for (int i = 0; i < 8; i++) a += sm[i]; sum_sq_out[r] = a; }
}

// TP RMSNorm phase 2: apply rstd*w then RoPE. freqs_cis can be nullptr to skip RoPE.
__global__ void k_apply_rmsnorm_rope(__nv_bfloat16* __restrict__ x,
                                      const float* __restrict__ w,
                                      const float* __restrict__ freqs_cis,
                                      const float* __restrict__ sum_sq_global,
                                      int full_D, int nh_pr, int hd) {
    extern __shared__ __nv_bfloat16 srow_tp[];
    int Dpr = nh_pr * hd;
    int r = blockIdx.x, t = threadIdx.x;
    __nv_bfloat16* xi = x + (size_t)r * Dpr;
    float rstd = rsqrtf(sum_sq_global[r] / full_D + 1e-6f);
    for (int i = t; i < Dpr; i += 256) {
        srow_tp[i] = __float2bfloat16(__bfloat162float(xi[i]) * rstd * w[i]);
    }
    __syncthreads();
    if (freqs_cis == nullptr) {
        for (int i = t; i < Dpr; i += 256) xi[i] = srow_tp[i];
        return;
    }
    const float* fc = freqs_cis + (size_t)r * 2 * hd;
    int half_hd = hd >> 1;
    int pairs = Dpr >> 1;
    for (int i = t; i < pairs; i += 256) {
        int h = i / half_hd;
        int kk = i % half_hd;
        int base = h * hd + (kk << 1);
        float c = fc[kk << 1];
        float sinv = fc[hd + (kk << 1) + 1];
        float a = __bfloat162float(srow_tp[base]);
        float b = __bfloat162float(srow_tp[base + 1]);
        xi[base]     = __float2bfloat16(a * c - b * sinv);
        xi[base + 1] = __float2bfloat16(a * sinv + b * c);
    }
}

// TP fused RMSNorm + RoPE + FP8 output
__global__ void k_apply_rmsnorm_rope_fp8(__nv_bfloat16* __restrict__ x,
                                          __nv_fp8_e4m3* __restrict__ x_fp8,
                                          const float* __restrict__ w,
                                          const float* __restrict__ freqs_cis,
                                          const float* __restrict__ sum_sq_global,
                                          int full_D, int nh_pr, int hd) {
    extern __shared__ __nv_bfloat16 srow_tp2[];
    int Dpr = nh_pr * hd;
    int r = blockIdx.x, t = threadIdx.x;
    __nv_bfloat16* xi = x + (size_t)r * Dpr;
    __nv_fp8_e4m3* xi_fp8 = x_fp8 + (size_t)r * Dpr;
    float rstd = rsqrtf(sum_sq_global[r] / full_D + 1e-6f);
    for (int i = t; i < Dpr; i += 256) {
        srow_tp2[i] = __float2bfloat16(__bfloat162float(xi[i]) * rstd * w[i]);
    }
    __syncthreads();
    if (freqs_cis == nullptr) {
        for (int i = t; i < Dpr; i += 256) {
            float v = __bfloat162float(srow_tp2[i]);
            xi[i] = srow_tp2[i];
            xi_fp8[i] = __nv_fp8_e4m3(v);
        }
        return;
    }
    const float* fc = freqs_cis + (size_t)r * 2 * hd;
    int half_hd = hd >> 1;
    int pairs = Dpr >> 1;
    for (int i = t; i < pairs; i += 256) {
        int h = i / half_hd;
        int kk = i % half_hd;
        int base = h * hd + (kk << 1);
        float c = fc[kk << 1];
        float sinv = fc[hd + (kk << 1) + 1];
        float a = __bfloat162float(srow_tp2[base]);
        float b = __bfloat162float(srow_tp2[base + 1]);
        float out0 = a * c - b * sinv;
        float out1 = a * sinv + b * c;
        xi[base]     = __float2bfloat16(out0);
        xi[base + 1] = __float2bfloat16(out1);
        xi_fp8[base]     = __nv_fp8_e4m3(out0);
        xi_fp8[base + 1] = __nv_fp8_e4m3(out1);
    }
}

// ============================================================
// GatedDeltaNet (GDN) CUDA kernels
// ============================================================

// Causal depthwise conv1d + SiLU. Reads from in[], writes to out[].
// Weight layout: [C, 1, K] where K=kernel_size (4 for Kairos GDN).
// Input/output layout: [T, C] bf16.
__global__ void k_causal_dw_conv_silu(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ w,  // [C, K] float32
    int T, int C, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int c = idx % C;
    int t = idx / C;
    float acc = 0;
    for (int k = 0; k < K; k++) {
        int src_t = t - (K - 1) + k;
        if (src_t >= 0) {
            acc += __bfloat162float(in[src_t * C + c]) * w[c * K + k];
        }
    }
    float s = acc / (1.f + expf(-acc));  // SiLU (stock ShortConvolution uses activation='silu')
    out[t * C + c] = __float2bfloat16(s);
}

__global__ void k_gdn_compute_gates(
    const __nv_bfloat16* __restrict__ a_proj,  // [T, NH]
    const __nv_bfloat16* __restrict__ b_proj,  // [T, NH]
    const float* __restrict__ A_log,           // [NH]
    const float* __restrict__ dt_bias,         // [NH]
    float* __restrict__ g_out,                 // [T, NH]
    float* __restrict__ beta_out,              // [T, NH]
    int T, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * NH) return;
    int h = idx % NH;
    float a_val = __bfloat162float(a_proj[idx]) + dt_bias[h];
    float b_val = __bfloat162float(b_proj[idx]);
    // beta = sigmoid(b)
    beta_out[idx] = 1.f / (1.f + expf(-b_val));
    // g = -exp(A_log) * softplus(a + dt_bias)
    float sp = (a_val > 20.f) ? a_val : logf(1.f + expf(a_val));  // softplus
    g_out[idx] = -expf(A_log[h]) * sp;
}

// Pre-normalize Q and K with L2 norm + scale, in-place.
// q[t,h,k] = q[t,h,k] / ||q[t,h,:]|| * scale
// k[t,h,k] = k[t,h,k] / ||k[t,h,:]||
// Grid: (T, NH), Block: 256 threads (reduce over K=256)
__global__ void k_gdn_l2norm_scale(
    __nv_bfloat16* __restrict__ q,   // [T, NH, K] in-place
    __nv_bfloat16* __restrict__ k,   // [T, NH, K] in-place
    int K, float scale) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    size_t base = ((size_t)t * gridDim.y + h) * K;

    // Compute L2 norms (parallel reduce over K)
    float q_sq = 0, k_sq = 0;
    for (int i = tid; i < K; i += blockDim.x) {
        float qv = __bfloat162float(q[base + i]);
        float kv = __bfloat162float(k[base + i]);
        q_sq += qv * qv;
        k_sq += kv * kv;
    }
    // Warp reduce
    for (int off = 16; off > 0; off >>= 1) {
        q_sq += __shfl_down_sync(0xffffffff, q_sq, off);
        k_sq += __shfl_down_sync(0xffffffff, k_sq, off);
    }
    __shared__ float sm[16];
    int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) { sm[wid] = q_sq; sm[8 + wid] = k_sq; }
    __syncthreads();
    if (tid == 0) {
        float sq = 0, sk = 0;
        for (int i = 0; i < (blockDim.x + 31) / 32; i++) { sq += sm[i]; sk += sm[8 + i]; }
        sm[0] = sq; sm[1] = sk;
    }
    __syncthreads();
    float q_sc = rsqrtf(sm[0] + 1e-6f) * scale;
    float k_sc = rsqrtf(sm[1] + 1e-6f);

    // Apply in-place
    for (int i = tid; i < K; i += blockDim.x) {
        q[base + i] = __float2bfloat16(__bfloat162float(q[base + i]) * q_sc);
        k[base + i] = __float2bfloat16(__bfloat162float(k[base + i]) * k_sc);
    }
}

// Persistent GDN recurrent kernel — Triton-style tile scheduling.
//
// Instead of 1 block per (V_tile, head), launches NUM_SMS * warps_per_sm
// persistent blocks. Each block loops through multiple (V_tile, head) pairs
// via an atomic work counter. This keeps warps resident on SMs, giving:
//  - Better L1 cache reuse (same SM processes consecutive tokens for same head)
//  - Higher effective occupancy (all SMs fully loaded)
//  - Temporal locality: q/k/v data for nearby tokens stays in L1
//
// Q and K are pre-normalized. No L2 norm in inner loop.
//
// Grid: dim3(N_PERSISTENT_BLOCKS), Block: 32
#define GDN_BV 8
#define GDN_BK 8
// Global atomic counter for persistent work distribution
static __device__ int g_gdn_work_counter;

__device__ __forceinline__ void load_bf16x8_to_f32(const __nv_bfloat16* src, float* dst) {
    int4 packed = *reinterpret_cast<const int4*>(src);
    const __nv_bfloat162* vals = reinterpret_cast<const __nv_bfloat162*>(&packed);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 pair = __bfloat1622float2(vals[i]);
        dst[2 * i] = pair.x;
        dst[2 * i + 1] = pair.y;
    }
}

__device__ __forceinline__ void load_bf16x4_to_f32(const __nv_bfloat16* src, float* dst) {
    int2 packed = *reinterpret_cast<const int2*>(src);
    const __nv_bfloat162* vals = reinterpret_cast<const __nv_bfloat162*>(&packed);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        float2 pair = __bfloat1622float2(vals[i]);
        dst[2 * i] = pair.x;
        dst[2 * i + 1] = pair.y;
    }
}

__global__ void __launch_bounds__(32) k_gdn_recurrent(
    const __nv_bfloat16* __restrict__ q,   // [T, NH, K]
    const __nv_bfloat16* __restrict__ k,   // [T, NH, K]
    const __nv_bfloat16* __restrict__ v,   // [T, NH, V]
    const float* __restrict__ g,           // [T, NH]
    const float* __restrict__ beta,        // [T, NH]
    __nv_bfloat16* __restrict__ o,         // [T, NH, V]
    float* __restrict__ state,             // [NH, K, V]
    int T, int NH, int K, int V, float scale,
    int total_tiles) {                     // NH * (V/BV)
    int lane = threadIdx.x;  // 0..31
    int k_off = lane * GDN_BK;

    // Persistent work loop: each block grabs the next (V_tile, head) pair
    while (true) {
        // Atomically grab next tile
        int tile_id;
        if (lane == 0) tile_id = atomicAdd(&g_gdn_work_counter, 1);
        tile_id = __shfl_sync(0xffffffff, tile_id, 0);
        if (tile_id >= total_tiles) break;

        // Decode tile_id → (head, V_tile)
        int NV = V / GDN_BV;
        int h = tile_id / NV;
        int bv = tile_id % NV;
        int v_off = bv * GDN_BV;

        // Load state for this (head, V_tile)
        float hr[GDN_BK][GDN_BV];
        float* st = state + (size_t)h * K * V;
        #pragma unroll
        for (int ki = 0; ki < GDN_BK; ki++)
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                hr[ki][vi] = st[(size_t)(k_off + ki) * V + v_off + vi];

        // Process ALL T tokens for this tile
        for (int t = 0; t < T; t++) {
            size_t qk_base = ((size_t)t * NH + h) * K;
            size_t v_base = ((size_t)t * NH + h) * V;

            // Load K once up front. Load Q later, just before the output
            // reduction, to shorten its live range and ease register pressure.
            float my_k[GDN_BK];
            load_bf16x8_to_f32(k + qk_base + k_off, my_k);
            float gt = (lane == 0) ? g[(size_t)t * NH + h] : 0.0f;
            float bt = (lane == 0) ? beta[(size_t)t * NH + h] : 0.0f;
            gt = __shfl_sync(0xffffffff, gt, 0);
            bt = __shfl_sync(0xffffffff, bt, 0);

            // Decay
            float decay = __expf(gt);
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++)
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    hr[ki][vi] *= decay;

            // h@k reduction
            float hk[GDN_BV];
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++) {
                float acc = 0;
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++)
                    acc += hr[ki][vi] * my_k[ki];
                hk[vi] = acc;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    hk[vi] += __shfl_down_sync(0xffffffff, hk[vi], off);
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                hk[vi] = __shfl_sync(0xffffffff, hk[vi], 0);

            // Delta rule + state update
            float v_vals[GDN_BV];
            load_bf16x8_to_f32(v + v_base + v_off, v_vals);
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++) {
                float vn = bt * (v_vals[vi] - hk[vi]);
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++)
                    hr[ki][vi] += my_k[ki] * vn;
            }

            // Output
            float my_q[GDN_BK];
            load_bf16x8_to_f32(q + qk_base + k_off, my_q);
            float ov[GDN_BV];
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++) {
                float acc = 0;
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++)
                    acc += hr[ki][vi] * my_q[ki];
                ov[vi] = acc;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    ov[vi] += __shfl_down_sync(0xffffffff, ov[vi], off);
            if (lane == 0) {
                size_t o_base = ((size_t)t * NH + h) * V;
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    o[o_base + v_off + vi] = __float2bfloat16(ov[vi]);
            }
        }

        // Store final state
        #pragma unroll
        for (int ki = 0; ki < GDN_BK; ki++)
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                st[(size_t)(k_off + ki) * V + v_off + vi] = hr[ki][vi];
    }
}


// GDN output norm + gate: out[t,v] = rmsnorm(recurrent_out, weight) * silu(gate[t,v])
// recurrent_out is [T, NH, Vhd], gate is [T, NH*Vhd], weight is [Vhd]
// Output is [T, NH*Vhd] bf16.
__global__ void k_gdn_rmsnorm_silu_gate(
    const __nv_bfloat16* __restrict__ rec_out,  // [T, NH, Vhd]
    const __nv_bfloat16* __restrict__ gate,      // [T, NH*Vhd]
    const float* __restrict__ weight,             // [Vhd]
    __nv_bfloat16* __restrict__ out,             // [T, NH*Vhd]
    int T, int NH, int Vhd) {
    // One block per (t, h) pair. 256 threads reduce over Vhd.
    int th = blockIdx.x;
    if (th >= T * NH) return;
    int t = th / NH, h = th % NH;
    int tid = threadIdx.x;
    int total_v = NH * Vhd;

    const __nv_bfloat16* ri = rec_out + (size_t)th * Vhd;
    const __nv_bfloat16* gi = gate + (size_t)t * total_v + (size_t)h * Vhd;
    __nv_bfloat16* oi = out + (size_t)t * total_v + (size_t)h * Vhd;

    // Compute sum of squares for RMSNorm
    __shared__ float sm[8];
    float ss = 0;
    for (int i = tid; i < Vhd; i += 256) {
        float v = __bfloat162float(ri[i]);
        ss += v * v;
    }
    // Warp reduce
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
    int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) sm[wid] = ss;
    __syncthreads();
    if (tid == 0) { float a = 0; for (int i = 0; i < 8; i++) a += sm[i]; sm[0] = a; }
    __syncthreads();
    float rstd = rsqrtf(sm[0] / Vhd + 1e-6f);

    // Apply: out = (rec * rstd * weight) * silu(gate)
    for (int i = tid; i < Vhd; i += 256) {
        float rv = __bfloat162float(ri[i]) * rstd * weight[i];
        float gv = __bfloat162float(gi[i]);
        float sg = gv / (1.f + expf(-gv));  // silu(gate)
        oi[i] = __float2bfloat16(rv * sg);
    }
}

// ============================================================
// Chunk-parallel GatedDeltaNet kernels (included from separate files)
// ============================================================
#include "gdn_chunk.cuh"
#include "gdn_chunk_h.cuh"
#include "gdn_chunk_o.cuh"
#include "gdn_chunk_h_simple.cuh"

// ============================================================
// Batched strided BF16 GEMM for cuBLASLt chunk GDN
// ============================================================
// Computes D = alpha * op(A) @ op(B) + beta * C for each batch element.
// Supports flexible transpose modes via opA/opB parameters.
static size_t g_gemm_ws_bytes = 128 * 1024 * 1024;

struct BatchedBF16Gemm {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr, Dd = nullptr;
    cublasLtMatmulHeuristicResult_t heur;

    void setup(cublasLtHandle_t h, int M, int N, int K,
               int batch, long long strideA, long long strideB, long long strideC,
               void* ws, size_t wss,
               cublasOperation_t opA = CUBLAS_OP_T,
               cublasOperation_t opB = CUBLAS_OP_N) {
        CKBL(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

        int ldA, rowA, colA;
        if (opA == CUBLAS_OP_T) { ldA = K; rowA = K; colA = N; }
        else                    { ldA = N; rowA = N; colA = K; }

        int ldB, rowB, colB;
        if (opB == CUBLAS_OP_T) { ldB = M; rowB = M; colB = K; }
        else                    { ldB = K; rowB = K; colB = M; }

        CKBL(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_16BF, rowA, colA, ldA));
        CKBL(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_16BF, rowB, colB, ldB));
        CKBL(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, N, M, N));
        CKBL(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_16BF, N, M, N));

        CKBL(cublasLtMatrixLayoutSetAttribute(Ad, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CKBL(cublasLtMatrixLayoutSetAttribute(Bd, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CKBL(cublasLtMatrixLayoutSetAttribute(Cd, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CKBL(cublasLtMatrixLayoutSetAttribute(Dd, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CKBL(cublasLtMatrixLayoutSetAttribute(Ad, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        CKBL(cublasLtMatrixLayoutSetAttribute(Bd, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        CKBL(cublasLtMatrixLayoutSetAttribute(Cd, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
        CKBL(cublasLtMatrixLayoutSetAttribute(Dd, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));

        cublasLtMatmulPreference_t pref;
        CKBL(cublasLtMatmulPreferenceCreate(&pref));
        size_t mws = std::min(wss, g_gemm_ws_bytes);
        CKBL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &mws, sizeof(mws)));
        int ret = 0;
        CKBL(cublasLtMatmulAlgoGetHeuristic(h, desc, Ad, Bd, Cd, Dd, pref, 1, &heur, &ret));
        TORCH_CHECK(ret > 0, "BatchedBF16Gemm: no algo for M=", M, " N=", N, " K=", K, " batch=", batch);
        CKBL(cublasLtMatmulPreferenceDestroy(pref));
    }

    void run(cublasLtHandle_t h, void* A, void* B, void* D, void* ws, size_t wss,
             float alpha, float beta, cudaStream_t st) {
        size_t use_wss = std::min(wss, g_gemm_ws_bytes);
        CKBL(cublasLtMatmul(h, desc, &alpha, A, Ad, B, Bd, &beta, D, Cd, D, Dd,
                            &heur.algo, ws, use_wss, st));
    }

    void destroy() {
        if (Dd) cublasLtMatrixLayoutDestroy(Dd);
        if (Cd) cublasLtMatrixLayoutDestroy(Cd);
        if (Bd) cublasLtMatrixLayoutDestroy(Bd);
        if (Ad) cublasLtMatrixLayoutDestroy(Ad);
        if (desc) cublasLtMatmulDescDestroy(desc);
        desc = nullptr; Ad = Bd = Cd = Dd = nullptr;
    }
};

// ============================================================
// cuBLASLt chunk GDN: custom kernels for phases 2 and 3
// ============================================================

// Phase 2a kernel: Apply gating/beta/norm to A_raw (K@K^T) and solve A_inv.
// Also computes coeff[j] = beta[j] * exp(gcum[j]) * krnorm[j] for w computation,
// and stores gcum and beta for later use.
//
// Grid: dim3(NC, NH), Block: 64 threads (one per token in chunk)
// SMEM: BT*BT*4 (s_A) + BT*3*4 (gcum, beta, krnorm) = ~17 KB
__global__ void __launch_bounds__(64)
k_gdn_chunk_gate_solve(
    const __nv_bfloat16* __restrict__ k_in,      // [T, NH, K]
    const float*         __restrict__ g_in,      // [T, NH]
    const float*         __restrict__ beta_in,   // [T, NH]
    __nv_bfloat16*       __restrict__ kkt_inout, // [NC*NH, BT, BT] bf16 — gated A_inv output
    float*               __restrict__ gcum_out,  // [NC, NH, BT]
    float*               __restrict__ coeff_out, // [NC*NH, BT] — beta*exp(gcum)*krnorm per token
    float*               __restrict__ beta_out,  // [NC*NH, BT] — beta per token (for u computation)
    int T, int NH, int K)
{
    const int BT = GDN_CHUNK_BT;  // 64
    const int chunk = blockIdx.x;
    const int head  = blockIdx.y;
    const int tid   = threadIdx.x;
    const int t_base = chunk * BT;
    const int chunk_len = min(BT, T - t_base);
    const int batch_idx = chunk * NH + head;

    extern __shared__ char _smem_gs[];
    char* sp = _smem_gs;
    float* s_A      = (float*)sp;  sp += BT * BT * sizeof(float);
    float* s_gcum   = (float*)sp;  sp += BT * sizeof(float);
    float* s_beta   = (float*)sp;  sp += BT * sizeof(float);
    float* s_krnorm = (float*)sp;  sp += BT * sizeof(float);

    // Load g, beta
    float my_g = 0.0f, my_beta = 0.0f;
    if (tid < chunk_len) {
        int tg = t_base + tid;
        my_g    = g_in[(size_t)tg * NH + head];
        my_beta = beta_in[(size_t)tg * NH + head];
    }
    s_gcum[tid] = (tid < chunk_len) ? my_g : 0.0f;
    s_beta[tid] = (tid < chunk_len) ? my_beta : 0.0f;
    __syncthreads();

    // Prefix sum of gates
    if (tid == 0) _chunk_prefix_sum(s_gcum, chunk_len);
    __syncthreads();

    // K L2 norm
    float k_sq = 0.0f;
    if (tid < chunk_len) {
        int tg = t_base + tid;
        for (int kk = 0; kk < K; kk++) {
            float val = __bfloat162float(k_in[((size_t)tg * NH + head) * K + kk]);
            k_sq += val * val;
        }
    }
    s_krnorm[tid] = (tid < chunk_len) ? rsqrtf(k_sq + 1e-6f) : 0.0f;
    __syncthreads();

    // Store gcum, coeff, beta to global
    {
        size_t gcum_base = ((size_t)chunk * NH + head) * BT;
        gcum_out[gcum_base + tid] = s_gcum[tid];
    }
    {
        size_t c_base = (size_t)batch_idx * BT;
        coeff_out[c_base + tid] = (tid < chunk_len) ?
            s_beta[tid] * expf(s_gcum[tid]) * s_krnorm[tid] : 0.0f;
        beta_out[c_base + tid] = s_beta[tid];
    }

    // Load A_raw from kkt_inout (bf16) into s_A (fp32) and apply gating
    {
        __nv_bfloat16* A_bf16 = kkt_inout + (size_t)batch_idx * BT * BT;
        // A_raw[i,j] is the raw K@K^T dot product (from cuBLASLt phase 1)
        // Apply: A[i,j] = beta[i] * exp(gcum[i]-gcum[j]) * krnorm[i] * krnorm[j] * A_raw[i,j]  for j < i
        // and A[i,j] = 0 for j >= i
        if (tid < chunk_len) {
            float my_gcum = s_gcum[tid];
            float my_beta_val = s_beta[tid];
            float my_knorm = s_krnorm[tid];
            for (int j = 0; j < tid; j++) {
                float raw = __bfloat162float(A_bf16[tid * BT + j]);
                s_A[tid * BT + j] = my_beta_val * expf(my_gcum - s_gcum[j])
                                   * my_knorm * s_krnorm[j] * raw;
            }
        }
        // Zero out diagonal and upper triangular
        for (int j = tid; j < BT; j++)
            s_A[tid * BT + j] = 0.0f;
    }
    __syncthreads();

    // Forward substitution: (I - A)^{-1}
    if (tid == 0) {
        float A_row[GDN_CHUNK_BT];
        for (int i = 0; i < chunk_len; i++) {
            for (int j = 0; j < i; j++) A_row[j] = s_A[i * BT + j];
            s_A[i * BT + i] = 1.0f;
            for (int j = 0; j < i; j++) {
                float acc = 0.0f;
                for (int kk = j; kk < i; kk++)
                    acc += A_row[kk] * s_A[kk * BT + j];
                s_A[i * BT + j] = acc;
            }
        }
    }
    __syncthreads();

    // Write A_inv back to kkt_inout as bf16 for subsequent batched GEMMs
    {
        __nv_bfloat16* A_bf16 = kkt_inout + (size_t)batch_idx * BT * BT;
        for (int j = 0; j < BT; j++)
            A_bf16[tid * BT + j] = __float2bfloat16(s_A[tid * BT + j]);
    }
}

// Phase 2b kernel: Prepare scaled K for w GEMM.
// Writes: scaled_K[batch, BT, K] bf16 = K_normed[t,h,k] * coeff[t] (per-token scalar)
// where coeff[t] = beta[t] * exp(gcum[t]) * krnorm[t].
// Also writes: scaled_V[batch, BT, V] bf16 = V[t,h,v] * beta[t].
//
// Grid: dim3(NC, NH), Block: 256 threads
__global__ void k_gdn_chunk_scale_kv(
    const __nv_bfloat16* __restrict__ k_in,   // [T, NH, K]
    const __nv_bfloat16* __restrict__ v_in,   // [T, NH, V]
    const float*         __restrict__ coeff,  // [NC*NH, BT] — beta*exp(gcum)*krnorm
    const float*         __restrict__ beta,   // [NC*NH, BT]
    __nv_bfloat16*       __restrict__ sk_out, // [NC*NH, BT, K] — scaled K_normed
    __nv_bfloat16*       __restrict__ sv_out, // [NC*NH, BT, V] — scaled V
    int T, int NH, int K, int V)
{
    const int BT = GDN_CHUNK_BT;
    const int chunk = blockIdx.x;
    const int head  = blockIdx.y;
    const int tid   = threadIdx.x;
    const int t_base = chunk * BT;
    const int chunk_len = min(BT, T - t_base);
    const int batch_idx = chunk * NH + head;

    // Process K: each thread handles multiple (token, k) pairs
    size_t sk_base = (size_t)batch_idx * BT * K;
    for (int idx = tid; idx < BT * K; idx += blockDim.x) {
        int i = idx / K;
        int kk = idx % K;
        if (i < chunk_len) {
            int tg = t_base + i;
            float c = coeff[(size_t)batch_idx * BT + i];
            float val = __bfloat162float(k_in[((size_t)tg * NH + head) * K + kk]);
            sk_out[sk_base + idx] = __float2bfloat16(val * c);
        } else {
            sk_out[sk_base + idx] = __float2bfloat16(0.0f);
        }
    }

    // Process V: each thread handles multiple (token, v) pairs
    size_t sv_base = (size_t)batch_idx * BT * V;
    for (int idx = tid; idx < BT * V; idx += blockDim.x) {
        int i = idx / V;
        int vv = idx % V;
        if (i < chunk_len) {
            int tg = t_base + i;
            float b = beta[(size_t)batch_idx * BT + i];
            float val = __bfloat162float(v_in[((size_t)tg * NH + head) * V + vv]);
            sv_out[sv_base + idx] = __float2bfloat16(val * b);
        } else {
            sv_out[sv_base + idx] = __float2bfloat16(0.0f);
        }
    }
}

// Phase 3 helper kernel: State propagation step for one chunk.
// Computes v_new = u_bf16 - w@h_correction (both bf16), gates it,
// computes output = scale * Q_normed @ h, updates h.
//
// This is the same as k_gdn_chunk_state_output but for a SINGLE chunk,
// and reads w/u from bf16 buffers (cuBLASLt output) instead of fp32.
// Also, the w@h and output Q@h products are pre-computed by cuBLASLt,
// so this kernel only needs to do the gating, state update, and output write.
//
// Actually, we keep the warp-register-tiled approach for the sequential state
// loop since it's already fast (the w@h product involves state that changes
// per-token within the chunk). cuBLASLt can't help here because of the
// sequential dependency. So we keep the existing k_gdn_chunk_state_output
// kernel for phase 3. The big win is phases 1-2 (prepare) being much faster.

// ============================================================
// gdn_cublas_chunk_forward: cuBLASLt-accelerated chunk GDN
// ============================================================
// Replaces k_gdn_recurrent with a 4-phase chunk algorithm:
//   Phase 1: K@K^T via cuBLASLt batched GEMM (parallel, ~0.16ms)
//   Phase 2: Gating + A_inv solve + scale K/V (custom kernels, ~1ms)
//   Phase 3: A_inv @ scaled_K → w, A_inv @ scaled_V → u via cuBLASLt (~1.5ms)
//   Phase 4: Sequential state propagation (existing warp kernel, ~2.3ms)
// Total: ~5ms vs 11ms for fused recurrent, vs 8.2ms for pure-custom prepare
//
// Scratch buffers (bf16, allocated in engine_init):
//   b_gdn_kkt_bf16:  [NC*NH, BT, BT] bf16 — K@K^T then A_inv
//   b_gdn_sk_bf16:   [NC*NH, BT, K]  bf16 — scaled K_normed
//   b_gdn_sv_bf16:   [NC*NH, BT, V]  bf16 — scaled V
//   b_gdn_w_bf16:    [NC*NH, BT, K]  bf16 — w output from A_inv @ scaled_K
//   b_gdn_u_bf16:    [NC*NH, BT, V]  bf16 — u output from A_inv @ scaled_V
//   b_gdn_coeff:     [NC*NH, BT]     f32  — beta*exp(gcum)*krnorm
//   b_gdn_beta_f32:  [NC*NH, BT]     f32  — beta per token

// Static scratch buffers (allocated in engine_init, declared near other GDN buffers)
static __nv_bfloat16 *b_gdn_kkt_bf16 = nullptr;   // [NC*NH, BT, BT]
static __nv_bfloat16 *b_gdn_sk_bf16  = nullptr;   // [NC*NH, BT, K]
static __nv_bfloat16 *b_gdn_sv_bf16  = nullptr;   // [NC*NH, BT, V]
static __nv_bfloat16 *b_gdn_w_bf16   = nullptr;   // [NC*NH, BT, K]
static __nv_bfloat16 *b_gdn_u_bf16   = nullptr;   // [NC*NH, BT, V]
static float         *b_gdn_coeff    = nullptr;    // [NC*NH, BT]
static float         *b_gdn_beta_f32 = nullptr;    // [NC*NH, BT]
static __nv_bfloat16 *b_gdn_hout    = nullptr;     // [(NC+1)*NH, K, V] state snapshots bf16

// Per-seq cuBLASLt descriptors for chunk GDN batched GEMMs
struct ChunkGdnGemms {
    BatchedBF16Gemm kkt;     // K @ K^T: [batch, BT, K] @ [batch, K, BT] → [batch, BT, BT]
    BatchedBF16Gemm ainv_w;  // A_inv @ scaled_K: [batch, BT, BT] @ [batch, BT, K] → [batch, BT, K]
    BatchedBF16Gemm ainv_u;  // A_inv @ scaled_V: [batch, BT, BT] @ [batch, BT, V] → [batch, BT, V]
    bool built = false;

    void build(cublasLtHandle_t h, int seq, int NH, int K, int V, void* ws, size_t wss) {
        const int BT = GDN_CHUNK_BT;
        int NC = (seq + BT - 1) / BT;
        int batch = NC * NH;

        // K@K^T: row-major [BT, K] @ [BT, K]^T → [BT, BT]
        // In cuBLAS col-major: C[BT,BT] = A[K,BT]^T @ B[K,BT], TRANSA=T TRANSB=N
        // But we want K @ K^T. In col-major, K[BT,K] stored row-major is K^T[K,BT] col-major.
        // So K@K^T (row) = K_col^T @ K_col = C[BT,BT] with A=K_col[K,BT], B=K_col[K,BT], TRANSA=T, TRANSB=N.
        kkt.setup(h, BT, BT, K, batch,
                  (long long)BT * K, (long long)BT * K, (long long)BT * BT,
                  ws, wss, CUBLAS_OP_T, CUBLAS_OP_N);

        // A_inv @ scaled_K: row-major [BT, BT] @ [BT, K] → [BT, K]
        // Col-major: C[K,BT] = A_col[BT,BT]^T @ B_col[K,BT]... no.
        // Row-major A[BT,BT] @ B[BT,K] → C[BT,K].
        // In col-major: A_row[BT,BT] = A_col^T[BT,BT], B_row[BT,K] = B_col^T[K,BT].
        // C_row[BT,K] = A_row @ B_row = A_col^T @ B_col^T = (B_col @ A_col)^T.
        // cuBLAS: D_col[K,BT] = B_col[K,BT] @ A_col[BT,BT] where TRANSA=N, TRANSB=N.
        // But that gives us D_col which is C_row transposed... hmm.
        //
        // Simpler: use the identity C_row = (B_col @ A_col)^T.
        // We want C_row stored in row-major → same as C_col^T stored contiguously.
        // cuBLAS D = alpha * opA(A) @ opB(B). We want D = C_col = C_row^T.
        // C_row = A_row @ B_row. C_col = C_row^T = B_row^T @ A_row^T = B_col @ A_col.
        // So: D[K,BT] = B_col[K,BT] @ A_col[BT,BT]. opA=N, opB=N.
        // This means: A_ptr = B_col (scaled_K in row-major = stored as is),
        //             B_ptr = A_col (A_inv in row-major = stored as is).
        // The "A" in cuBLAS is scaled_K[K,BT] (col-major view of row-major [BT,K]),
        // the "B" in cuBLAS is A_inv[BT,BT] (col-major view of row-major [BT,BT]).
        // D[K,BT] col-major = C[BT,K] row-major. Leading dim of D is K.
        //
        // M=BT, N=K (output is [BT,K] row-major = [K,BT] col-major)
        // Actually let me re-derive. cuBLAS sees everything col-major.
        // We store A_inv as row-major [BT,BT]. Col-major interpretation: [BT,BT]^T.
        // We store scaled_K as row-major [BT,K]. Col-major interpretation: [K,BT]^T... no.
        // Row-major [R,C] with ld=C is the same bits as col-major [C,R] with ld=C.
        //
        // So A_inv row[BT,BT] ld=BT  ↔  col[BT,BT] ld=BT (symmetric shape, but it's transposed!)
        // scaled_K row[BT,K] ld=K    ↔  col[K,BT] ld=K
        // w_out row[BT,K] ld=K       ↔  col[K,BT] ld=K
        //
        // We want w = A_inv_row @ sk_row = (A_inv_col)^T @ (sk_col)^T
        // Using cuBLAS: D = opA(A) @ opB(B) where A,B,D are col-major.
        // Set opA=N on sk_col[K,BT] and opB=T on ainv_col[BT,BT]:
        //   D[K,BT] = sk_col[K,BT] @ (ainv_col[BT,BT])^T
        //   = sk_col @ ainv_col^T
        // In row-major this is D_row[BT,K] = (sk_col @ ainv_col^T)^T_row... complicated.
        //
        // Easiest approach: use the row-major trick.
        // For row-major C = A @ B with dims [M,K_inner] @ [K_inner,N]:
        //   cuBLAS: D_col[N,M] = B_col[N,K_inner]^? @ A_col[K_inner,M]^?
        //   With everything stored row-major:
        //   B_row[K_inner,N] ↔ B_col[N,K_inner], A_row[M,K_inner] ↔ A_col[K_inner,M]
        //   D = B_col @ A_col = [N,K_inner] @ [K_inner,M] → [N,M] col → [M,N] row ✓
        //   So: opA=N, opB=N, first_ptr=B_row, second_ptr=A_row,
        //   cuBLAS M_cublas=N, N_cublas=M, K_cublas=K_inner.
        //
        // For A_inv[BT,BT] @ sk[BT,K] → w[BT,K]:
        //   "A_row" = A_inv[BT,BT], "B_row" = sk[BT,K], M=BT, K_inner=BT, N=K
        //   cuBLAS: ptr_A=sk, ptr_B=A_inv, M_cb=K, N_cb=BT, K_cb=BT, opA=N, opB=N
        ainv_w.setup(h, K, BT, BT, batch,
                     (long long)BT * K, (long long)BT * BT, (long long)BT * K,
                     ws, wss, CUBLAS_OP_N, CUBLAS_OP_N);
        // ptr_A = sk, ptr_B = ainv → D = w  (row-major [BT,K])

        // Similarly for A_inv[BT,BT] @ sv[BT,V] → u[BT,V]:
        //   cuBLAS: ptr_A=sv, ptr_B=A_inv, M_cb=V, N_cb=BT, K_cb=BT, opA=N, opB=N
        ainv_u.setup(h, V, BT, BT, batch,
                     (long long)BT * V, (long long)BT * BT, (long long)BT * V,
                     ws, wss, CUBLAS_OP_N, CUBLAS_OP_N);

        built = true;
    }
};
// g_chunk_gdn_gemms, get_chunk_gdn_gemms, and gdn_cublas_chunk_forward
// are defined later (after globals) since they reference g_ltH, g_gemm_ws, etc.

// ============================================================
// FP8 GEMM descriptor (one per shape)
// ============================================================
static bool g_lt_autotune_enabled = true;
static int g_lt_autotune_topk = 32;
static bool g_lt_fast_accum_fp8 = false;

struct FP8Gemm {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr, Dd = nullptr;
    cublasLtMatmulHeuristicResult_t heur{};
    std::vector<cublasLtMatmulHeuristicResult_t> heurs;
    bool has_bias_epi = false;
    bool tuned = false;
    int M = 0, N = 0, K = 0;

    void setup(cublasLtHandle_t h, int M, int N, int K, float* sA, float* sB,
               void* ws, size_t wss, int epi = 0) {
        this->M = M; this->N = N; this->K = K;
        CKBL(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        if (g_lt_fast_accum_fp8) {
            int32_t fast_accum = 1;
            CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));
        }
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sA, sizeof(sA)));
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sB, sizeof(sB)));
        cublasLtMatmulMatrixScale_t sm_scalar = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm_scalar, sizeof(sm_scalar)));
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm_scalar, sizeof(sm_scalar)));
        if (epi > 0) {
            has_bias_epi = true;
            cublasLtEpilogue_t e = (epi == 2) ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
            CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
            cudaDataType_t bt = CUDA_R_16BF;
            CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
        }
        CKBL(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, K, N, K));
        CKBL(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, K, M, K));
        CKBL(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, N, M, N));
        CKBL(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_16BF, N, M, N));
        cublasLtMatmulPreference_t pref; CKBL(cublasLtMatmulPreferenceCreate(&pref));
        size_t mws = g_gemm_ws_bytes;
        CKBL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &mws, sizeof(mws)));
        int max_results = std::max(1, g_lt_autotune_topk);
        heurs.resize(max_results);
        int ret = 0;
        CKBL(cublasLtMatmulAlgoGetHeuristic(h, desc, Ad, Bd, Cd, Dd, pref, max_results, heurs.data(), &ret));
        TORCH_CHECK(ret > 0, "no algo for [", M, ",", N, ",", K, "] epi=", epi);
        heurs.resize(ret);
        heur = heurs[0];
        tuned = (ret == 1) || !g_lt_autotune_enabled;
        cublasLtMatmulPreferenceDestroy(pref);
    }

    void run(cublasLtHandle_t h, void* wt_fp8, void* act_fp8, void* out,
             void* ws, size_t wss, float alpha, float beta, cudaStream_t st,
             const void* bias = nullptr, void* c_ptr = nullptr) {
        size_t use_wss = std::min(wss, g_gemm_ws_bytes);
        if (has_bias_epi && bias) {
            CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        }
        void* c = c_ptr ? c_ptr : out;
        if (!tuned && beta == 0.0f && heurs.size() > 1) {
            cudaEvent_t ev0, ev1;
            CK(cudaEventCreate(&ev0));
            CK(cudaEventCreate(&ev1));
            float best_ms = 1.0e30f;
            int best_idx = 0;
            for (size_t i = 0; i < heurs.size(); ++i) {
                auto& cand = heurs[i];
                cublasStatus_t warm = cublasLtMatmul(h, desc, &alpha, wt_fp8, Ad, act_fp8, Bd, &beta,
                                                     c, Cd, out, Dd, &cand.algo, ws, use_wss, st);
                if (warm != CUBLAS_STATUS_SUCCESS) continue;
                CK(cudaEventRecord(ev0, st));
                cublasStatus_t timed = cublasLtMatmul(h, desc, &alpha, wt_fp8, Ad, act_fp8, Bd, &beta,
                                                      c, Cd, out, Dd, &cand.algo, ws, use_wss, st);
                if (timed != CUBLAS_STATUS_SUCCESS) continue;
                CK(cudaEventRecord(ev1, st));
                CK(cudaEventSynchronize(ev1));
                float ms = 0.0f;
                CK(cudaEventElapsedTime(&ms, ev0, ev1));
                if (ms < best_ms) {
                    best_ms = ms;
                    best_idx = (int)i;
                }
            }
            CK(cudaEventDestroy(ev0));
            CK(cudaEventDestroy(ev1));
            heur = heurs[best_idx];
            tuned = true;
        }
        CKBL(cublasLtMatmul(h, desc, &alpha, wt_fp8, Ad, act_fp8, Bd, &beta,
                            c, Cd, out, Dd, &heur.algo, ws, use_wss, st));
    }
};

struct FP4Gemm {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr, Dd = nullptr;
    cublasLtMatmulHeuristicResult_t heur{};
    std::vector<cublasLtMatmulHeuristicResult_t> heurs;
    bool tuned = false;
    int M = 0, N = 0, K = 0;

    void setup(cublasLtHandle_t h, int M, int N, int K, uint8_t* sA, uint8_t* sB,
               void* ws, size_t wss) {
        this->M = M; this->N = N; this->K = K;
        CKBL(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        cudaDataType_t scale_type = CUDA_R_32F;
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));
        cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sA, sizeof(sA)));
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sB, sizeof(sB)));
        cublasLtMatmulMatrixScale_t sm_vec16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm_vec16, sizeof(sm_vec16)));
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm_vec16, sizeof(sm_vec16)));

        CKBL(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_4F_E2M1, K, N, K));
        CKBL(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_4F_E2M1, K, M, K));
        CKBL(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, N, M, N));
        CKBL(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_16BF, N, M, N));

        cublasLtMatmulPreference_t pref;
        CKBL(cublasLtMatmulPreferenceCreate(&pref));
        size_t mws = std::min(wss, g_gemm_ws_bytes);
        CKBL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &mws, sizeof(mws)));
        int max_results = std::max(1, g_lt_autotune_topk);
        heurs.resize(max_results);
        int nres = 0;
        CKBL(cublasLtMatmulAlgoGetHeuristic(h, desc, Ad, Bd, Cd, Dd, pref, max_results, heurs.data(), &nres));
        TORCH_CHECK(nres > 0, "FP4Gemm: no heuristic for M=", M, " N=", N, " K=", K);
        heurs.resize(nres);
        heur = heurs[0];
        tuned = (nres == 1) || !g_lt_autotune_enabled;
        CKBL(cublasLtMatmulPreferenceDestroy(pref));
    }

    void run(cublasLtHandle_t h, void* wt_fp4, void* act_fp4, void* out,
             void* ws, size_t wss, float* alpha_dev, float* beta_dev, cudaStream_t st) {
        size_t use_wss = std::min(wss, g_gemm_ws_bytes);
        if (!tuned && heurs.size() > 1) {
            cudaEvent_t ev0, ev1;
            CK(cudaEventCreate(&ev0));
            CK(cudaEventCreate(&ev1));
            float best_ms = 1.0e30f;
            int best_idx = 0;
            for (size_t i = 0; i < heurs.size(); ++i) {
                auto& cand = heurs[i];
                cublasStatus_t warm = cublasLtMatmul(h, desc, alpha_dev, wt_fp4, Ad, act_fp4, Bd, beta_dev,
                                                     out, Cd, out, Dd, &cand.algo, ws, use_wss, st);
                if (warm != CUBLAS_STATUS_SUCCESS) continue;
                CK(cudaEventRecord(ev0, st));
                cublasStatus_t timed = cublasLtMatmul(h, desc, alpha_dev, wt_fp4, Ad, act_fp4, Bd, beta_dev,
                                                      out, Cd, out, Dd, &cand.algo, ws, use_wss, st);
                if (timed != CUBLAS_STATUS_SUCCESS) continue;
                CK(cudaEventRecord(ev1, st));
                CK(cudaEventSynchronize(ev1));
                float ms = 0.0f;
                CK(cudaEventElapsedTime(&ms, ev0, ev1));
                if (ms < best_ms) {
                    best_ms = ms;
                    best_idx = (int)i;
                }
            }
            CK(cudaEventDestroy(ev0));
            CK(cudaEventDestroy(ev1));
            heur = heurs[best_idx];
            tuned = true;
        }
        CKBL(cublasLtMatmul(h, desc, alpha_dev, wt_fp4, Ad, act_fp4, Bd, beta_dev,
                            out, Cd, out, Dd, &heur.algo, ws, use_wss, st));
    }
};

// CUTLASS fused FFN1 path: FP8 GEMM + bias + SiLU -> FP8 output.
// This replaces the standalone k_silu_to_fp8 pass when enabled.
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
struct CutlassFusedFfn1 {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::float_e4m3_t;
    using ElementD = cutlass::float_e4m3_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementBias = cutlass::bfloat16_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    static constexpr int AlignA = 16;
    static constexpr int AlignB = 16;
    static constexpr int AlignC = 16;
    static constexpr int AlignD = 16;

    using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerColBiasEltAct<
        cutlass::epilogue::thread::SiLu,
        ElementD,
        ElementCompute,
        ElementBias,
        ElementC>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignC,
        ElementD,
        LayoutC,
        AlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation
    >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        AlignA,
        ElementB,
        LayoutB,
        AlignB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using Arguments = typename Gemm::Arguments;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    Gemm gemm{};
    StrideA stride_A{};
    StrideB stride_B{};
    StrideC stride_C{};
    StrideD stride_D{};
    void* workspace = nullptr;
    size_t workspace_size = 0;
    int M = 0, N = 0, K = 0;
    bool built = false;
    bool initialized = false;

    void setup(int M_, int N_, int K_) {
        if (built && M == M_ && N == N_ && K == K_) return;
        M = M_;
        N = N_;
        K = K_;
        stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {nullptr, stride_A, nullptr, stride_B},
            {{}, nullptr, stride_C, nullptr, stride_D}
        };
        args.scheduler.max_swizzle_size = 1;
        workspace_size = Gemm::get_workspace_size(args);
        if (workspace != nullptr) {
            CK(cudaFree(workspace));
            workspace = nullptr;
        }
        if (workspace_size > 0) {
            CK(cudaMalloc(&workspace, workspace_size));
        }
        built = true;
        initialized = false;
    }

    Arguments make_args(void* wt_fp8, void* act_fp8, void* out_fp8, const __nv_bfloat16* bias) const {
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {reinterpret_cast<ElementA const*>(act_fp8), stride_A,
             reinterpret_cast<ElementB const*>(wt_fp8), stride_B},
            {{}, reinterpret_cast<ElementC const*>(out_fp8), stride_C,
             reinterpret_cast<ElementD*>(out_fp8), stride_D}
        };
        auto& fusion = args.epilogue.thread;
        fusion.alpha = 1.0f;
        fusion.beta = 0.0f;
        fusion.scale_a = 1.0f;
        fusion.scale_b = 1.0f;
        fusion.scale_c = 1.0f;
        fusion.scale_d = 1.0f;
        fusion.bias_ptr = reinterpret_cast<ElementBias const*>(bias);
        args.scheduler.max_swizzle_size = 1;
        return args;
    }

    void run(void* wt_fp8, void* act_fp8, void* out_fp8, const __nv_bfloat16* bias, cudaStream_t stream) {
        auto args = make_args(wt_fp8, act_fp8, out_fp8, bias);
        cutlass::Status st = Gemm::can_implement(args);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN1 unsupported for current shape");
        if (!initialized) {
            st = gemm.initialize(args, workspace, stream);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN1 initialize failed");
            initialized = true;
        } else {
            st = gemm.update(args);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN1 update failed");
        }
        st = gemm.run(stream);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN1 launch failed");
    }
};

struct CutlassFusedFfn1Fp4 {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::float_e2m1_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementBias = cutlass::bfloat16_t;
    using ElementBlockScaleFactor = cutlass::float_ue4m3_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignA = 16;
    static constexpr int AlignB = 16;
    static constexpr int AlignC = 8;
    static constexpr int AlignD = 32;
    static constexpr int OutputSFVectorSize = 16;

    using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
        cutlass::epilogue::thread::SiLu,
        OutputSFVectorSize,
        ElementD,
        ElementCompute,
        ElementBlockScaleFactor,
        cutlass::layout::RowMajor,
        ElementBias,
        ElementC>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignC,
        ElementD,
        LayoutD,
        AlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation
    >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        AlignA,
        ElementB,
        LayoutB,
        AlignB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using Arguments = typename Gemm::Arguments;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
    using LayoutSFD = typename SfdOutputCfg::LayoutSF;

    Gemm gemm{};
    StrideA stride_A{};
    StrideB stride_B{};
    StrideC stride_C{};
    StrideD stride_D{};
    LayoutSFD layout_SFD{};
    void* workspace = nullptr;
    size_t workspace_size = 0;
    int M = 0, N = 0, K = 0;
    bool built = false;
    bool initialized = false;

    void setup(int M_, int N_, int K_) {
        if (built && M == M_ && N == N_ && K == K_) return;
        M = M_;
        N = N_;
        K = K_;
        stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(cute::make_shape(M, N, K, 1));
        size_t expected_scale_bytes = (size_t)fp4_scale_rows(M) * (size_t)fp4_scale_cols(N);
        TORCH_CHECK(
            cute::size(cute::filter_zeros(layout_SFD)) == expected_scale_bytes,
            "CUTLASS fused FFN1 FP4 scale layout size mismatch: got ",
            cute::size(cute::filter_zeros(layout_SFD)),
            " expected ", expected_scale_bytes);
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {nullptr, stride_A, nullptr, stride_B},
            {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}
        };
        args.scheduler.max_swizzle_size = 1;
        args.epilogue.thread.block_scale_factor_ptr = nullptr;
        args.epilogue.thread.norm_constant_ptr = nullptr;
        workspace_size = Gemm::get_workspace_size(args);
        if (workspace != nullptr) {
            CK(cudaFree(workspace));
            workspace = nullptr;
        }
        if (workspace_size > 0) {
            CK(cudaMalloc(&workspace, workspace_size));
        }
        built = true;
        initialized = false;
    }

    Arguments make_args(
        void* wt_fp8,
        void* act_fp8,
        const __nv_bfloat16* src_bf16,
        void* out_fp4,
        void* out_scales,
        const __nv_bfloat16* bias,
        const float* norm_constant_dev) const {
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {reinterpret_cast<ElementA const*>(act_fp8), stride_A,
             reinterpret_cast<ElementB const*>(wt_fp8), stride_B},
            {{1.0f, 0.0f},
             reinterpret_cast<ElementC const*>(src_bf16), stride_C,
             reinterpret_cast<ElementD*>(out_fp4), stride_D}
        };
        auto& fusion = args.epilogue.thread;
        fusion.alpha = 1.0f;
        fusion.beta = 0.0f;
        fusion.bias_ptr = reinterpret_cast<ElementBias const*>(bias);
        fusion.block_scale_factor_ptr = reinterpret_cast<ElementBlockScaleFactor*>(out_scales);
        fusion.norm_constant_ptr = norm_constant_dev;
        args.scheduler.max_swizzle_size = 1;
        return args;
    }

    void run(
        void* wt_fp8,
        void* act_fp8,
        const __nv_bfloat16* src_bf16,
        void* out_fp4,
        void* out_scales,
        const __nv_bfloat16* bias,
        const float* norm_constant_dev,
        cudaStream_t stream) {
        auto args = make_args(wt_fp8, act_fp8, src_bf16, out_fp4, out_scales, bias, norm_constant_dev);
        cutlass::Status st = Gemm::can_implement(args);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN1 FP4 unsupported for current shape");
        if (!initialized) {
            st = gemm.initialize(args, workspace, stream);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN1 FP4 initialize failed");
            initialized = true;
        } else {
            st = gemm.update(args);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN1 FP4 update failed");
        }
        st = gemm.run(stream);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN1 FP4 launch failed");
    }
};

struct CutlassFusedFfn1Fp4Identity {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::float_e2m1_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementBias = cutlass::bfloat16_t;
    using ElementBlockScaleFactor = cutlass::float_ue4m3_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignA = 16;
    static constexpr int AlignB = 16;
    static constexpr int AlignC = 8;
    static constexpr int AlignD = 32;
    static constexpr int OutputSFVectorSize = 16;

    using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
        cutlass::epilogue::thread::Identity,
        OutputSFVectorSize,
        ElementD,
        ElementCompute,
        ElementBlockScaleFactor,
        cutlass::layout::RowMajor,
        ElementBias,
        ElementC>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignC,
        ElementD,
        LayoutD,
        AlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation
    >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        AlignA,
        ElementB,
        LayoutB,
        AlignB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using Arguments = typename Gemm::Arguments;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
    using LayoutSFD = typename SfdOutputCfg::LayoutSF;

    Gemm gemm{};
    StrideA stride_A{};
    StrideB stride_B{};
    StrideC stride_C{};
    StrideD stride_D{};
    LayoutSFD layout_SFD{};
    void* workspace = nullptr;
    size_t workspace_size = 0;
    int M = 0, N = 0, K = 0;
    bool built = false;
    bool initialized = false;

    void setup(int M_, int N_, int K_) {
        if (built && M == M_ && N == N_ && K == K_) return;
        M = M_;
        N = N_;
        K = K_;
        stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(cute::make_shape(M, N, K, 1));
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {nullptr, stride_A, nullptr, stride_B},
            {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}
        };
        args.scheduler.max_swizzle_size = 1;
        args.epilogue.thread.block_scale_factor_ptr = nullptr;
        args.epilogue.thread.norm_constant_ptr = nullptr;
        workspace_size = Gemm::get_workspace_size(args);
        if (workspace != nullptr) {
            CK(cudaFree(workspace));
            workspace = nullptr;
        }
        if (workspace_size > 0) {
            CK(cudaMalloc(&workspace, workspace_size));
        }
        built = true;
        initialized = false;
    }

    Arguments make_args(
        void* wt_fp8,
        void* act_fp8,
        const __nv_bfloat16* src_bf16,
        void* out_fp4,
        void* out_scales,
        const __nv_bfloat16* bias,
        const float* norm_constant_dev) const {
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {reinterpret_cast<ElementA const*>(act_fp8), stride_A,
             reinterpret_cast<ElementB const*>(wt_fp8), stride_B},
            {{1.0f, 0.0f},
             reinterpret_cast<ElementC const*>(src_bf16), stride_C,
             reinterpret_cast<ElementD*>(out_fp4), stride_D}
        };
        auto& fusion = args.epilogue.thread;
        fusion.alpha = 1.0f;
        fusion.beta = 0.0f;
        fusion.bias_ptr = reinterpret_cast<ElementBias const*>(bias);
        fusion.block_scale_factor_ptr = reinterpret_cast<ElementBlockScaleFactor*>(out_scales);
        fusion.norm_constant_ptr = norm_constant_dev;
        args.scheduler.max_swizzle_size = 1;
        return args;
    }

    void run(
        void* wt_fp8,
        void* act_fp8,
        const __nv_bfloat16* src_bf16,
        void* out_fp4,
        void* out_scales,
        const __nv_bfloat16* bias,
        const float* norm_constant_dev,
        cudaStream_t stream) {
        auto args = make_args(wt_fp8, act_fp8, src_bf16, out_fp4, out_scales, bias, norm_constant_dev);
        cutlass::Status st = Gemm::can_implement(args);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN1 FP4 identity unsupported for current shape");
        if (!initialized) {
            st = gemm.initialize(args, workspace, stream);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN1 FP4 identity initialize failed");
            initialized = true;
        } else {
            st = gemm.update(args);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN1 FP4 identity update failed");
        }
        st = gemm.run(stream);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN1 FP4 identity launch failed");
    }
};

struct CutlassFfn1Fp4BlockScaleOnly {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::float_e2m1_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementBlockScaleFactor = cutlass::float_ue4m3_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignA = 16;
    static constexpr int AlignB = 16;
    static constexpr int AlignC = 8;
    static constexpr int AlignD = 32;
    static constexpr int OutputSFVectorSize = 16;

    using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
        OutputSFVectorSize,
        ElementD,
        ElementCompute,
        ElementBlockScaleFactor,
        cutlass::layout::RowMajor,
        ElementC>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignC,
        ElementD,
        LayoutD,
        AlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation
    >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        AlignA,
        ElementB,
        LayoutB,
        AlignB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using Arguments = typename Gemm::Arguments;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
    using LayoutSFD = typename SfdOutputCfg::LayoutSF;

    Gemm gemm{};
    StrideA stride_A{};
    StrideB stride_B{};
    StrideC stride_C{};
    StrideD stride_D{};
    LayoutSFD layout_SFD{};
    void* workspace = nullptr;
    size_t workspace_size = 0;
    int M = 0, N = 0, K = 0;
    bool built = false;
    bool initialized = false;

    void setup(int M_, int N_, int K_) {
        if (built && M == M_ && N == N_ && K == K_) return;
        M = M_;
        N = N_;
        K = K_;
        stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(cute::make_shape(M, N, K, 1));
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {nullptr, stride_A, nullptr, stride_B},
            {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}
        };
        args.scheduler.max_swizzle_size = 1;
        args.epilogue.thread.block_scale_factor_ptr = nullptr;
        args.epilogue.thread.norm_constant_ptr = nullptr;
        workspace_size = Gemm::get_workspace_size(args);
        if (workspace != nullptr) {
            CK(cudaFree(workspace));
            workspace = nullptr;
        }
        if (workspace_size > 0) {
            CK(cudaMalloc(&workspace, workspace_size));
        }
        built = true;
        initialized = false;
    }

    Arguments make_args(
        void* wt_fp8,
        void* act_fp8,
        const __nv_bfloat16* src_bf16,
        void* out_fp4,
        void* out_scales,
        const float* norm_constant_dev) const {
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {reinterpret_cast<ElementA const*>(act_fp8), stride_A,
             reinterpret_cast<ElementB const*>(wt_fp8), stride_B},
            {{1.0f, 0.0f},
             reinterpret_cast<ElementC const*>(src_bf16), stride_C,
             reinterpret_cast<ElementD*>(out_fp4), stride_D}
        };
        auto& fusion = args.epilogue.thread;
        fusion.alpha = 1.0f;
        fusion.beta = 0.0f;
        fusion.block_scale_factor_ptr = reinterpret_cast<ElementBlockScaleFactor*>(out_scales);
        fusion.norm_constant_ptr = norm_constant_dev;
        args.scheduler.max_swizzle_size = 1;
        return args;
    }

    void run(
        void* wt_fp8,
        void* act_fp8,
        const __nv_bfloat16* src_bf16,
        void* out_fp4,
        void* out_scales,
        const float* norm_constant_dev,
        cudaStream_t stream) {
        auto args = make_args(wt_fp8, act_fp8, src_bf16, out_fp4, out_scales, norm_constant_dev);
        cutlass::Status st = Gemm::can_implement(args);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS FP4 blockscale-only unsupported for current shape");
        if (!initialized) {
            st = gemm.initialize(args, workspace, stream);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS FP4 blockscale-only initialize failed");
            initialized = true;
        } else {
            st = gemm.update(args);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS FP4 blockscale-only update failed");
        }
        st = gemm.run(stream);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS FP4 blockscale-only launch failed");
    }
};

struct CutlassFfn1Fp4BlockScaleOnlyE8M0 {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::float_e2m1_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementBlockScaleFactor = cutlass::float_ue8m0_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignA = 16;
    static constexpr int AlignB = 16;
    static constexpr int AlignC = 8;
    static constexpr int AlignD = 32;
    static constexpr int OutputSFVectorSize = 16;

    using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
        OutputSFVectorSize,
        ElementD,
        ElementCompute,
        ElementBlockScaleFactor,
        cutlass::layout::RowMajor,
        ElementC>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignC,
        ElementD,
        LayoutD,
        AlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation
    >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        AlignA,
        ElementB,
        LayoutB,
        AlignB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using Arguments = typename Gemm::Arguments;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
    using LayoutSFD = typename SfdOutputCfg::LayoutSF;

    Gemm gemm{};
    StrideA stride_A{};
    StrideB stride_B{};
    StrideC stride_C{};
    StrideD stride_D{};
    LayoutSFD layout_SFD{};
    void* workspace = nullptr;
    size_t workspace_size = 0;
    int M = 0, N = 0, K = 0;
    bool built = false;
    bool initialized = false;

    void setup(int M_, int N_, int K_) {
        if (built && M == M_ && N == N_ && K == K_) return;
        M = M_;
        N = N_;
        K = K_;
        stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(cute::make_shape(M, N, K, 1));
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {nullptr, stride_A, nullptr, stride_B},
            {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}
        };
        args.scheduler.max_swizzle_size = 1;
        args.epilogue.thread.block_scale_factor_ptr = nullptr;
        args.epilogue.thread.norm_constant_ptr = nullptr;
        workspace_size = Gemm::get_workspace_size(args);
        if (workspace != nullptr) {
            CK(cudaFree(workspace));
            workspace = nullptr;
        }
        if (workspace_size > 0) {
            CK(cudaMalloc(&workspace, workspace_size));
        }
        built = true;
        initialized = false;
    }

    Arguments make_args(
        void* wt_fp8,
        void* act_fp8,
        const __nv_bfloat16* src_bf16,
        void* out_fp4,
        void* out_scales,
        const float* norm_constant_dev) const {
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {reinterpret_cast<ElementA const*>(act_fp8), stride_A,
             reinterpret_cast<ElementB const*>(wt_fp8), stride_B},
            {{1.0f, 0.0f},
             reinterpret_cast<ElementC const*>(src_bf16), stride_C,
             reinterpret_cast<ElementD*>(out_fp4), stride_D}
        };
        auto& fusion = args.epilogue.thread;
        fusion.alpha = 1.0f;
        fusion.beta = 0.0f;
        fusion.block_scale_factor_ptr = reinterpret_cast<ElementBlockScaleFactor*>(out_scales);
        fusion.norm_constant_ptr = norm_constant_dev;
        args.scheduler.max_swizzle_size = 1;
        return args;
    }

    void run(
        void* wt_fp8,
        void* act_fp8,
        const __nv_bfloat16* src_bf16,
        void* out_fp4,
        void* out_scales,
        const float* norm_constant_dev,
        cudaStream_t stream) {
        auto args = make_args(wt_fp8, act_fp8, src_bf16, out_fp4, out_scales, norm_constant_dev);
        cutlass::Status st = Gemm::can_implement(args);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS FP4 blockscale-only E8M0 unsupported for current shape");
        if (!initialized) {
            st = gemm.initialize(args, workspace, stream);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS FP4 blockscale-only E8M0 initialize failed");
            initialized = true;
        } else {
            st = gemm.update(args);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS FP4 blockscale-only E8M0 update failed");
        }
        st = gemm.run(stream);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS FP4 blockscale-only E8M0 launch failed");
    }
};

struct CutlassFusedFfn2 {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementBias = cutlass::bfloat16_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    static constexpr int AlignA = 16;
    static constexpr int AlignB = 16;
    static constexpr int AlignC = 8;
    static constexpr int AlignD = 8;

    using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBias<
        ElementD,
        ElementCompute,
        ElementBias,
        ElementC>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignC,
        ElementD,
        LayoutC,
        AlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation
    >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        AlignA,
        ElementB,
        LayoutB,
        AlignB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using Arguments = typename Gemm::Arguments;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    Gemm gemm{};
    StrideA stride_A{};
    StrideB stride_B{};
    StrideC stride_C{};
    StrideD stride_D{};
    void* workspace = nullptr;
    size_t workspace_size = 0;
    int M = 0, N = 0, K = 0;
    bool built = false;
    bool initialized = false;

    void setup(int M_, int N_, int K_) {
        if (built && M == M_ && N == N_ && K == K_) return;
        M = M_;
        N = N_;
        K = K_;
        stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {nullptr, stride_A, nullptr, stride_B},
            {{}, nullptr, stride_C, nullptr, stride_D}
        };
        args.scheduler.max_swizzle_size = 1;
        workspace_size = Gemm::get_workspace_size(args);
        if (workspace != nullptr) {
            CK(cudaFree(workspace));
            workspace = nullptr;
        }
        if (workspace_size > 0) {
            CK(cudaMalloc(&workspace, workspace_size));
        }
        built = true;
        initialized = false;
    }

    Arguments make_args(void* wt_fp8, void* act_fp8, void* out_bf16, const __nv_bfloat16* bias) const {
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {reinterpret_cast<ElementA const*>(act_fp8), stride_A,
             reinterpret_cast<ElementB const*>(wt_fp8), stride_B},
            {{}, reinterpret_cast<ElementC const*>(out_bf16), stride_C,
             reinterpret_cast<ElementD*>(out_bf16), stride_D}
        };
        auto& fusion = args.epilogue.thread;
        fusion.alpha = 1.0f;
        fusion.beta = 0.0f;
        fusion.bias_ptr = reinterpret_cast<ElementBias const*>(bias);
        args.scheduler.max_swizzle_size = 1;
        return args;
    }

    void run(void* wt_fp8, void* act_fp8, void* out_bf16, const __nv_bfloat16* bias, cudaStream_t stream) {
        auto args = make_args(wt_fp8, act_fp8, out_bf16, bias);
        cutlass::Status st = Gemm::can_implement(args);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN2 unsupported for current shape");
        if (!initialized) {
            st = gemm.initialize(args, workspace, stream);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN2 initialize failed");
            initialized = true;
        } else {
            st = gemm.update(args);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN2 update failed");
        }
        st = gemm.run(stream);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS fused FFN2 launch failed");
    }
};

struct CutlassFp4RowwiseGemm {
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using ArchTag = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    static constexpr int AlignA = 32;
    static constexpr int AlignB = 32;
    static constexpr int AlignC = 8;
    static constexpr int AlignD = 8;

    using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignC,
        ElementD,
        LayoutD,
        AlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        ElementA,
        LayoutA,
        AlignA,
        ElementB,
        LayoutB,
        AlignB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using Arguments = typename Gemm::Arguments;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using TensorLayoutA = decltype(cute::make_layout(cute::make_shape(0, 0, 0), StrideA{}));
    using TensorLayoutB = decltype(cute::make_layout(cute::make_shape(0, 0, 0), StrideB{}));
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    Gemm gemm{};
    StrideA stride_A{};
    StrideB stride_B{};
    StrideC stride_C{};
    StrideD stride_D{};
    TensorLayoutA layout_A{};
    TensorLayoutB layout_B{};
    LayoutSFA layout_SFA{};
    LayoutSFB layout_SFB{};
    void* workspace = nullptr;
    size_t workspace_size = 0;
    int M = 0, N = 0, K = 0;
    bool built = false;
    bool initialized = false;

    void setup(int M_, int N_, int K_) {
        if (built && M == M_ && N == N_ && K == K_) return;
        M = M_;
        N = N_;
        K = K_;
        stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        layout_A = cute::make_layout(cute::make_shape(M, K, 1), stride_A);
        layout_B = cute::make_layout(cute::make_shape(N, K, 1), stride_B);
        layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
        layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
            {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}
        };
        args.scheduler.max_swizzle_size = 1;
        workspace_size = Gemm::get_workspace_size(args);
        if (workspace != nullptr) {
            CK(cudaFree(workspace));
            workspace = nullptr;
        }
        if (workspace_size > 0) {
            CK(cudaMalloc(&workspace, workspace_size));
        }
        built = true;
        initialized = false;
    }

    Arguments make_args(
        void* wt_fp4,
        void* wt_scales,
        void* act_fp4,
        void* act_scales,
        void* out_bf16) const {
        Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {
                reinterpret_cast<typename ElementA::DataType const*>(act_fp4), stride_A,
                reinterpret_cast<typename ElementB::DataType const*>(wt_fp4), stride_B,
                reinterpret_cast<typename ElementA::ScaleFactorType const*>(act_scales), layout_SFA,
                reinterpret_cast<typename ElementB::ScaleFactorType const*>(wt_scales), layout_SFB
            },
            {
                {1.0f, 0.0f},
                reinterpret_cast<ElementC const*>(out_bf16), stride_C,
                reinterpret_cast<ElementD*>(out_bf16), stride_D
            }
        };
        args.scheduler.max_swizzle_size = 1;
        return args;
    }

    void run(
        void* wt_fp4,
        void* wt_scales,
        void* act_fp4,
        void* act_scales,
        void* out_bf16,
        cudaStream_t stream) {
        auto args = make_args(wt_fp4, wt_scales, act_fp4, act_scales, out_bf16);
        cutlass::Status st = Gemm::can_implement(args);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS FP4 rowwise GEMM unsupported for current shape");
        if (!initialized) {
            st = gemm.initialize(args, workspace, stream);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS FP4 rowwise GEMM initialize failed");
            initialized = true;
        } else {
            st = gemm.update(args);
            TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS FP4 rowwise GEMM update failed");
        }
        st = gemm.run(stream);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS FP4 rowwise GEMM launch failed");
    }
};

template <class LayoutSF>
static torch::Tensor pack_cutlass_scales_from_rowwise_host(
    const torch::Tensor& rowwise_scales_cuda,
    LayoutSF layout_sf,
    int logical_rows,
    int logical_cols) {
    TORCH_CHECK(rowwise_scales_cuda.is_cuda(), "rowwise scales must be CUDA");
    TORCH_CHECK(rowwise_scales_cuda.scalar_type() == torch::kUInt8, "rowwise scales must be uint8");
    TORCH_CHECK(rowwise_scales_cuda.dim() == 2, "rowwise scales must be rank-2");
    TORCH_CHECK(logical_cols % 16 == 0, "logical cols must be divisible by 16");

    auto rowwise_cpu = rowwise_scales_cuda.to(torch::kCPU);
    auto packed_cpu = torch::zeros(
        {(int64_t)cute::size(cute::filter_zeros(layout_sf))},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));

    auto* rowwise_ptr = rowwise_cpu.data_ptr<uint8_t>();
    int rowwise_ld = (int)rowwise_cpu.size(1);
    auto tensor_sf = cute::make_tensor(
        reinterpret_cast<cutlass::float_ue4m3_t*>(packed_cpu.data_ptr<uint8_t>()),
        layout_sf);

    for (int row = 0; row < logical_rows; ++row) {
        for (int block16 = 0; block16 < logical_cols / 16; ++block16) {
            cutlass::float_ue4m3_t scale = cutlass::float_ue4m3_t::bitcast(
                rowwise_ptr[(size_t)row * rowwise_ld + block16]);
            for (int kk = 0; kk < 16; ++kk) {
                tensor_sf(cute::make_coord(row, block16 * 16 + kk, 0)) = scale;
            }
        }
    }
    return packed_cpu.to(torch::kCUDA);
}

template <class Element, class LayoutData, class LayoutSF>
static void pack_cutlass_operand_host(
    cutlass::HostTensor<typename Element::DataType, cutlass::layout::PackedVectorLayout>& block_data,
    cutlass::HostTensor<typename Element::ScaleFactorType, cutlass::layout::PackedVectorLayout>& block_sf,
    LayoutData layout_data,
    LayoutSF layout_sf,
    const at::BFloat16* src,
    int rows,
    int cols,
    int ld) {
    constexpr float FP4_MAX = 6.0f;
    block_data.reset(cutlass::make_Coord(cute::size(layout_data)));
    block_sf.reset(cutlass::make_Coord(cute::size(cute::filter_zeros(layout_sf))));

    auto tensor_data = cute::make_tensor(cute::recast_ptr<typename Element::DataType>(block_data.host_data()), layout_data);
    auto tensor_sf = cute::make_tensor(block_sf.host_data(), layout_sf);

    for (int row = 0; row < rows; ++row) {
        for (int block16 = 0; block16 < cols / 16; ++block16) {
            float amax = 0.0f;
            for (int kk = 0; kk < 16; ++kk) {
                float v = (float)src[(size_t)row * ld + block16 * 16 + kk];
                amax = std::max(amax, std::abs(v));
            }
            float scale = (amax > 0.0f) ? (amax / FP4_MAX) : 1.0f;
            float inv_scale = 1.0f / scale;
            typename Element::ScaleFactorType sf(scale);
            for (int kk = 0; kk < 16; ++kk) {
                int k = block16 * 16 + kk;
                float v = (float)src[(size_t)row * ld + k] * inv_scale;
                tensor_data(cute::make_coord(row, k, 0)) = typename Element::DataType(v);
                tensor_sf(cute::make_coord(row, k, 0)) = sf;
            }
        }
    }

    block_data.sync_device();
    block_sf.sync_device();
}

template <class Element, class LayoutData, class LayoutSF>
static std::tuple<torch::Tensor, torch::Tensor> pack_cutlass_operand_host_tensors(
    torch::Tensor in_t,
    LayoutData layout_data,
    LayoutSF layout_sf,
    int rows,
    int cols) {
    auto in_cpu = in_t.contiguous().to(torch::kCPU);
    cutlass::HostTensor<typename Element::DataType, cutlass::layout::PackedVectorLayout> block_data;
    cutlass::HostTensor<typename Element::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_sf;
    pack_cutlass_operand_host<Element>(
        block_data, block_sf, layout_data, layout_sf,
        in_cpu.data_ptr<at::BFloat16>(), rows, cols, cols);

    size_t data_bytes = (block_data.size() * cutlass::sizeof_bits<typename Element::DataType>::value + 7) / 8;
    size_t scale_bytes = block_sf.size() * sizeof(typename Element::ScaleFactorType);
    auto opts_cpu = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto packed = torch::empty({(long)data_bytes}, opts_cpu);
    auto scales = torch::empty({(long)scale_bytes}, opts_cpu);
    std::memcpy(packed.data_ptr(), reinterpret_cast<uint8_t*>(block_data.host_data()), data_bytes);
    std::memcpy(scales.data_ptr(), reinterpret_cast<uint8_t*>(block_sf.host_data()), scale_bytes);
    return {packed, scales};
}
#endif

// BF16 x BF16 GEMM (for ENGINE_BF16_ATTN=1 mode)
struct BF16Gemm {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr, Dd = nullptr;
    cublasLtMatmulHeuristicResult_t heur{};
    std::vector<cublasLtMatmulHeuristicResult_t> heurs;
    bool has_bias_epi = false;
    bool tuned = false;
    int M = 0, N = 0, K = 0;

    void setup(cublasLtHandle_t h, int M, int N, int K,
               void* ws, size_t wss, int epi = 0) {
        this->M = M; this->N = N; this->K = K;
        CKBL(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
        CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        if (epi > 0) {
            has_bias_epi = true;
            cublasLtEpilogue_t e = (epi == 2) ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
            CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
            cudaDataType_t bt = CUDA_R_16BF;
            CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
        }
        CKBL(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_16BF, K, N, K));
        CKBL(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_16BF, K, M, K));
        CKBL(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, N, M, N));
        CKBL(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_16BF, N, M, N));
        cublasLtMatmulPreference_t pref; CKBL(cublasLtMatmulPreferenceCreate(&pref));
        size_t mws = std::min(wss, g_gemm_ws_bytes);
        CKBL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &mws, sizeof(mws)));
        int max_results = std::max(1, g_lt_autotune_topk);
        heurs.resize(max_results);
        int nres = 0;
        CKBL(cublasLtMatmulAlgoGetHeuristic(h, desc, Ad, Bd, Cd, Dd, pref, max_results, heurs.data(), &nres));
        TORCH_CHECK(nres > 0, "BF16Gemm: no heuristic for M=", M, " N=", N, " K=", K);
        heurs.resize(nres);
        heur = heurs[0];
        tuned = (nres == 1) || !g_lt_autotune_enabled;
        CKBL(cublasLtMatmulPreferenceDestroy(pref));
    }

    void run(cublasLtHandle_t h, void* wt_bf16, void* act_bf16, void* out,
             void* ws, size_t wss, float alpha, float beta, cudaStream_t st,
             const void* bias = nullptr) {
        size_t use_wss = std::min(wss, g_gemm_ws_bytes);
        if (has_bias_epi && bias) {
            CKBL(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        }
        if (!tuned && beta == 0.0f && heurs.size() > 1) {
            cudaEvent_t ev0, ev1;
            CK(cudaEventCreate(&ev0));
            CK(cudaEventCreate(&ev1));
            float best_ms = 1.0e30f;
            int best_idx = 0;
            for (size_t i = 0; i < heurs.size(); ++i) {
                auto& cand = heurs[i];
                cublasStatus_t warm = cublasLtMatmul(h, desc, &alpha, wt_bf16, Ad, act_bf16, Bd, &beta,
                                                     out, Cd, out, Dd, &cand.algo, ws, use_wss, st);
                if (warm != CUBLAS_STATUS_SUCCESS) continue;
                CK(cudaEventRecord(ev0, st));
                cublasStatus_t timed = cublasLtMatmul(h, desc, &alpha, wt_bf16, Ad, act_bf16, Bd, &beta,
                                                      out, Cd, out, Dd, &cand.algo, ws, use_wss, st);
                if (timed != CUBLAS_STATUS_SUCCESS) continue;
                CK(cudaEventRecord(ev1, st));
                CK(cudaEventSynchronize(ev1));
                float ms = 0.0f;
                CK(cudaEventElapsedTime(&ms, ev0, ev1));
                if (ms < best_ms) {
                    best_ms = ms;
                    best_idx = (int)i;
                }
            }
            CK(cudaEventDestroy(ev0));
            CK(cudaEventDestroy(ev1));
            heur = heurs[best_idx];
            tuned = true;
        }
        CKBL(cublasLtMatmul(h, desc, &alpha, wt_bf16, Ad, act_bf16, Bd, &beta,
                            out, Cd, out, Dd, &heur.algo, ws, use_wss, st));
    }
};

// ============================================================
// cuDNN SDPA (BF16, BSHD layout via custom strides)
// ============================================================
enum UIDS : int64_t { Q_UID = 1, K_UID = 2, V_UID = 3, O_UID = 4 };
struct CudnnSDPA {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t ws_size = 0; void* d_ws = nullptr;

    void build(cudnnHandle_t h, int64_t nh, int64_t sq, int64_t sk, int64_t hd, float sc) {
        graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::BFLOAT16)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);
        auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q").set_uid(Q_UID)
            .set_dim({1, nh, sq, hd}).set_stride({sq*nh*hd, hd, nh*hd, 1}));
        auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K").set_uid(K_UID)
            .set_dim({1, nh, sk, hd}).set_stride({sk*nh*hd, hd, nh*hd, 1}));
        auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V").set_uid(V_UID)
            .set_dim({1, nh, sk, hd}).set_stride({sk*nh*hd, hd, nh*hd, 1}));
        auto [O, S] = graph->sdpa(Q, K, V,
            fe::graph::SDPA_attributes().set_name("sdpa").set_attn_scale(sc));
        O->set_output(true).set_dim({1, nh, sq, hd}).set_stride({sq*nh*hd, hd, nh*hd, 1}).set_uid(O_UID);
        auto st = graph->build(h, {fe::HeurMode_t::B, fe::HeurMode_t::A});
        TORCH_CHECK(st.is_good(), "cuDNN SDPA build: ", st.get_message());
        (void)graph->get_workspace_size(ws_size);
        if (ws_size > 0) CK(cudaMalloc(&d_ws, ws_size));
    }

    void run(cudnnHandle_t h, void* q, void* k, void* v, void* o, cudaStream_t s) {
        cudnnSetStream(h, s);
        std::unordered_map<int64_t, void*> p = {{Q_UID, q}, {K_UID, k}, {V_UID, v}, {O_UID, o}};
        graph->execute(h, p, d_ws);
    }
};

// ============================================================
// FP8 SDPA graph (Q/K/V/O all FP8_E4M3)
// ============================================================
enum UIDS_FP8 : int64_t {
    Q_UID_FP8 = 11, K_UID_FP8 = 12, V_UID_FP8 = 13, O_UID_FP8 = 14,
    DQ_UID = 21, DK_UID = 22, DV_UID = 23, DS_UID = 24,
    SS_UID = 25, SO_UID = 26, AS_UID = 27, AO_UID = 28,
};
struct CudnnSDPA_FP8 {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t ws_size = 0; void* d_ws = nullptr;

    void build(cudnnHandle_t h, int64_t nh, int64_t sq, int64_t sk, int64_t hd, float sc) {
        graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::FP8_E4M3)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);
        auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q").set_uid(Q_UID_FP8)
            .set_dim({1, nh, sq, hd}).set_stride({sq*nh*hd, hd, nh*hd, 1}));
        auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K").set_uid(K_UID_FP8)
            .set_dim({1, nh, sk, hd}).set_stride({sk*nh*hd, hd, nh*hd, 1}));
        auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V").set_uid(V_UID_FP8)
            .set_dim({1, nh, sk, hd}).set_stride({sk*nh*hd, hd, nh*hd, 1}));
        auto mk_scale = [&](const char* n, int64_t uid) {
            return graph->tensor(fe::graph::Tensor_attributes()
                .set_name(n).set_uid(uid)
                .set_dim({1,1,1,1}).set_stride({1,1,1,1})
                .set_data_type(fe::DataType_t::FLOAT));
        };
        auto descale_q = mk_scale("Descale_Q", DQ_UID);
        auto descale_k = mk_scale("Descale_K", DK_UID);
        auto descale_v = mk_scale("Descale_V", DV_UID);
        auto descale_s = mk_scale("Descale_S", DS_UID);
        auto scale_s   = mk_scale("Scale_S",   SS_UID);
        auto scale_o   = mk_scale("Scale_O",   SO_UID);
        auto sdpa_opts = fe::graph::SDPA_fp8_attributes()
            .set_name("sdpa_fp8")
            .set_generate_stats(false)
            .set_causal_mask(false)
            .set_attn_scale(sc);
        auto [O, Stats, Amax_S, Amax_O] = graph->sdpa_fp8(
            Q, K, V, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o, sdpa_opts);
        O->set_output(true).set_dim({1, nh, sq, hd}).set_stride({sq*nh*hd, hd, nh*hd, 1}).set_uid(O_UID_FP8);
        Amax_S->set_output(true).set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::FLOAT).set_uid(AS_UID);
        Amax_O->set_output(true).set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::FLOAT).set_uid(AO_UID);
        auto st = graph->validate();
        TORCH_CHECK(st.is_good(), "cuDNN FP8 SDPA validate: ", st.get_message());
        st = graph->build_operation_graph(h);
        TORCH_CHECK(st.is_good(), "cuDNN FP8 SDPA build_operation_graph: ", st.get_message());
        auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
        TORCH_CHECK(plans.is_good(), "cuDNN FP8 SDPA create_execution_plans: ", plans.get_message());
        st = graph->check_support(h);
        TORCH_CHECK(st.is_good(), "cuDNN FP8 SDPA check_support: ", st.get_message());
        st = graph->build_plans(h);
        TORCH_CHECK(st.is_good(), "cuDNN FP8 SDPA build_plans: ", st.get_message());
        (void)graph->get_workspace_size(ws_size);
        if (ws_size > 0) CK(cudaMalloc(&d_ws, ws_size));
    }

    void run(cudnnHandle_t h, void* q_fp8, void* k_fp8, void* v_fp8, void* o_fp8,
             void* dq, void* dk, void* dv, void* ds, void* ss, void* so,
             void* amax_s, void* amax_o, cudaStream_t s) {
        cudnnSetStream(h, s);
        std::unordered_map<int64_t, void*> p = {
            {Q_UID_FP8, q_fp8}, {K_UID_FP8, k_fp8}, {V_UID_FP8, v_fp8}, {O_UID_FP8, o_fp8},
            {DQ_UID, dq}, {DK_UID, dk}, {DV_UID, dv}, {DS_UID, ds},
            {SS_UID, ss}, {SO_UID, so}, {AS_UID, amax_s}, {AO_UID, amax_o},
        };
        graph->execute(h, p, d_ws);
    }
};

// ============================================================
// Per-layer state
// ============================================================
struct Layer {
    // FP8 weights (device memory, owned)
    void *sq_w8 = nullptr, *sk_w8 = nullptr, *sv_w8 = nullptr, *so_w8 = nullptr;
    void *cq_w8 = nullptr, *ck_w8 = nullptr, *cv_w8 = nullptr, *co_w8 = nullptr;
    void *f1_w8 = nullptr, *f2_w8 = nullptr;
    uint8_t *f2_w4 = nullptr, *f2_s4 = nullptr;
    // BF16 bias pointers (persistent copies)
    __nv_bfloat16 *sqb = nullptr, *skb = nullptr, *svb = nullptr, *sob = nullptr;
    __nv_bfloat16 *cqb = nullptr, *ckb = nullptr, *cvb = nullptr, *cob = nullptr;
    __nv_bfloat16 *f1b = nullptr, *f2b = nullptr;
    // Float32 norm weights (persistent copies)
    float *rms_q = nullptr, *rms_k = nullptr;
    float *ca_rms_q = nullptr, *ca_rms_k = nullptr;
    float *n2w = nullptr, *n2b = nullptr;  // cross_attn_norm weight+bias (always present in Kairos)
    float *sst = nullptr;                   // modulation data
    bool has_norm2 = false;
    int ffn_dim = 0;
    int ca_k_dim = 0;

    // ---- BF16 weight copies (for force_bf16_gemms debugging mode) ----
    __nv_bfloat16 *sq_w16=nullptr, *sk_w16=nullptr, *sv_w16=nullptr, *so_w16=nullptr;
    __nv_bfloat16 *cq_w16=nullptr, *ck_w16=nullptr, *cv_w16=nullptr, *co_w16=nullptr;
    __nv_bfloat16 *f1_w16=nullptr, *f2_w16=nullptr;
    // GDN BF16 weights (q/k/v/g/o only — a/b already have bf16)
    __nv_bfloat16 *gdn_q_w16=nullptr, *gdn_k_w16=nullptr, *gdn_v_w16=nullptr;
    __nv_bfloat16 *gdn_g_w16=nullptr, *gdn_o_w16=nullptr;

    // ---- GatedDeltaNet (GDN) weights (for linear attention layers) ----
    bool is_gdn = false;
    // Projection weights (FP8)
    void *gdn_q_w8 = nullptr, *gdn_k_w8 = nullptr, *gdn_v_w8 = nullptr;
    void *gdn_a_w8 = nullptr, *gdn_b_w8 = nullptr;  // FP8 (unused, kept for compat)
    __nv_bfloat16 *gdn_a_w_bf16 = nullptr, *gdn_b_w_bf16 = nullptr;  // BF16 for small GEMMs
    void *gdn_g_w8 = nullptr, *gdn_o_w8 = nullptr;
    // Projection biases (BF16, nullptr if no bias)
    __nv_bfloat16 *gdn_qb = nullptr, *gdn_kb = nullptr, *gdn_vb = nullptr;
    __nv_bfloat16 *gdn_ab = nullptr, *gdn_bb = nullptr;
    __nv_bfloat16 *gdn_gb = nullptr, *gdn_ob = nullptr;
    // Short conv weights: [channels, kernel_size] float32
    float *gdn_conv_q_w = nullptr, *gdn_conv_k_w = nullptr, *gdn_conv_v_w = nullptr;
    // Gate parameters
    float *gdn_A_log = nullptr;   // [NH] float32
    float *gdn_dt_bias = nullptr; // [NH] float32
    // Output norm weight
    float *gdn_o_norm_w = nullptr; // [head_v_dim] float32
    // Dimensions
    int gdn_key_dim = 0, gdn_value_dim = 0;
    int gdn_head_k = 0, gdn_head_v = 0;
};

// ============================================================
// Global engine state
// ============================================================
static cublasHandle_t g_blasH = nullptr;
static cublasLtHandle_t g_ltH = nullptr;
static cudnnHandle_t g_cudnnH = nullptr;
static float *g_scaleA = nullptr, *g_scaleB = nullptr, *g_scaleZero = nullptr;
static void *g_gemm_ws = nullptr;
static int g_D, g_NH, g_HD, g_NL, g_FFN, g_CA_K_DIM, g_nsm;
static int g_max_seq = 0, g_ctx = 0;
static int g_actual_ctx = 0;  // actual encoder token count (may be < g_ctx)
static bool g_ready = false;

static void ensure_gemm_runtime() {
    if (!g_blasH) CKBL(cublasCreate(&g_blasH));
    if (!g_ltH) CKBL(cublasLtCreate(&g_ltH));
    if (!g_gemm_ws) {
        CK(cudaMalloc(&g_gemm_ws, g_gemm_ws_bytes));
    }
    float one = 1.f, zero = 0.f;
    if (!g_scaleA) {
        CK(cudaMalloc(&g_scaleA, sizeof(float)));
        CK(cudaMemcpy(g_scaleA, &one, sizeof(float), cudaMemcpyHostToDevice));
    }
    if (!g_scaleB) {
        CK(cudaMalloc(&g_scaleB, sizeof(float)));
        CK(cudaMemcpy(g_scaleB, &one, sizeof(float), cudaMemcpyHostToDevice));
    }
    if (!g_scaleZero) {
        CK(cudaMalloc(&g_scaleZero, sizeof(float)));
        CK(cudaMemcpy(g_scaleZero, &zero, sizeof(float), cudaMemcpyHostToDevice));
    }
}

// FP8 SDPA feature flag
static bool g_fp8_sdpa_enabled = false;

// BF16 attention mode flag
static bool g_bf16_attn = false;
struct BF16AttnWeights {
    __nv_bfloat16 *sq_w = nullptr, *sk_w = nullptr, *sv_w = nullptr, *so_w = nullptr;
    __nv_bfloat16 *cq_w = nullptr, *ck_w = nullptr, *cv_w = nullptr, *co_w = nullptr;
};
static std::vector<BF16AttnWeights> g_bf16_attn_weights;

// Force ALL GEMMs to BF16 (for correctness debugging — slower but no FP8 quantization)
static bool g_force_bf16_gemms = false;
void set_force_bf16_gemms(bool v) { g_force_bf16_gemms = v; }

// Tensor-parallel state
static int g_tp_rank = 0, g_tp_world = 1;
static ncclComm_t g_nccl_comm = nullptr;
static int g_D_per_rank = 0;
static int g_NH_per_rank = 0;
static int g_FFN_per_rank = 0;

// Per-layer data
static std::vector<Layer> g_layers;

// Profile mode: 0=all, 1=gemms only, 2=ewise only, 3=sdpa only, 4=no sdpa, 5=no gemm, 6=no ewise
static int g_profile_mode = 0;
void set_profile_mode(int64_t m) { g_profile_mode = (int)m; }

enum class SdpaBackend {
    CudnnBf16 = 0,
    TorchBf16 = 1,
    SageAttn3Py = 2,
    SageAttn3Cpp = 3,
};

static SdpaBackend g_sdpa_backend = SdpaBackend::CudnnBf16;
static bool g_use_torch_sdpa = false;

static const char* sdpa_backend_name(SdpaBackend backend) {
    switch (backend) {
        case SdpaBackend::CudnnBf16: return "cudnn";
        case SdpaBackend::TorchBf16: return "torch";
        case SdpaBackend::SageAttn3Py: return "sage3_py";
        case SdpaBackend::SageAttn3Cpp: return "sage3_cpp";
    }
    return "unknown";
}

void set_torch_sdpa(bool v) {
    g_use_torch_sdpa = v;
    if (v) {
        g_sdpa_backend = SdpaBackend::TorchBf16;
    } else if (g_sdpa_backend == SdpaBackend::TorchBf16) {
        g_sdpa_backend = SdpaBackend::CudnnBf16;
    }
}

void set_sdpa_backend(const std::string& name) {
    if (name == "cudnn" || name == "default") {
        g_sdpa_backend = SdpaBackend::CudnnBf16;
    } else if (name == "torch") {
        g_sdpa_backend = SdpaBackend::TorchBf16;
    } else if (name == "sage3_py") {
        g_sdpa_backend = SdpaBackend::SageAttn3Py;
    } else if (name == "sage3_cpp") {
        g_sdpa_backend = SdpaBackend::SageAttn3Cpp;
    } else {
        TORCH_CHECK(false, "Unknown SDPA backend: ", name,
                    " (expected cudnn, torch, sage3_py, or sage3_cpp)");
    }
    g_use_torch_sdpa = (g_sdpa_backend == SdpaBackend::TorchBf16);
}

std::string get_sdpa_backend() {
    return sdpa_backend_name(g_sdpa_backend);
}

// Skip all CA operations — let Python handle cross-attention
static bool g_skip_ca = false;
void set_skip_ca(bool v) { g_skip_ca = v; }

// Helper: run PyTorch SDPA on raw bf16 buffers
// Q/K/V layout: [seq, nh*hd] contiguous (seq-major, heads interleaved)
static void torch_sdpa_bf16(void* q_ptr, void* k_ptr, void* v_ptr, void* o_ptr,
                            int sq, int sk, int nh, int hd, float scale,
                            cudaStream_t stream) {
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    // Wrap raw pointers as [1, seq, nh, hd] then transpose to [1, nh, seq, hd]
    auto Q = torch::from_blob(q_ptr, {1, sq, nh, hd}, opts).permute({0, 2, 1, 3});
    auto K = torch::from_blob(k_ptr, {1, sk, nh, hd}, opts).permute({0, 2, 1, 3});
    auto V = torch::from_blob(v_ptr, {1, sk, nh, hd}, opts).permute({0, 2, 1, 3});
    // at::scaled_dot_product_attention expects [B, NH, S, HD] contiguous
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    auto O = at::scaled_dot_product_attention(Q, K, V,
        /*attn_mask=*/{}, /*dropout_p=*/0.0, /*is_causal=*/false, /*scale=*/scale);
    // O is [1, nh, sq, hd], need to write back as [sq, nh*hd]
    auto O_out = O.permute({0, 2, 1, 3}).contiguous().view({sq, nh * hd});
    // Copy to output buffer
    CK(cudaMemcpyAsync(o_ptr, O_out.data_ptr(), (size_t)sq * nh * hd * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream));
}

static at::Tensor wrap_seqmajor_bf16(void* ptr, int seq, int nh, int hd) {
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    return torch::from_blob(ptr, {1, seq, nh, hd}, opts).permute({0, 2, 1, 3}).contiguous();
}

static at::Tensor wrap_bhsd_bf16(void* ptr, int seq, int nh, int hd) {
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    return torch::from_blob(ptr, {1, nh, seq, hd}, opts);
}

__global__ void k_seqmajor_to_bhsd_pad(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int in_seq, int out_seq, int nh, int hd) {
    size_t total = (size_t)out_seq * nh * hd;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        int d = idx % hd;
        size_t tmp = idx / hd;
        int s = tmp % out_seq;
        int h = tmp / out_seq;
        if (s < in_seq) {
            size_t in_idx = ((size_t)s * nh + h) * hd + d;
            out[idx] = in[in_idx];
        } else {
            out[idx] = __float2bfloat16(0.0f);
        }
    }
}

static void copy_seqmajor_to_bhsd_pad(const __nv_bfloat16* in_ptr, __nv_bfloat16* out_ptr,
                                      int in_seq, int out_seq, int nh, int hd,
                                      cudaStream_t stream) {
    size_t total = (size_t)out_seq * nh * hd;
    int max_blocks = std::max(1, 8 * g_nsm);
    int blocks = std::max(1, (int)std::min<size_t>((total + 255) / 256, (size_t)max_blocks));
    k_seqmajor_to_bhsd_pad<<<blocks, 256, 0, stream>>>(in_ptr, out_ptr, in_seq, out_seq, nh, hd);
}

__global__ void k_center_k_seqmajor_to_bhsd(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int valid_seq, int padded_seq, int nh, int hd) {
    int h = blockIdx.x;
    int d2 = threadIdx.x;
    int hd2 = hd >> 1;
    if (d2 >= hd2) return;
    __nv_bfloat16* head_out = out + (size_t)h * padded_seq * hd;
    int d = d2 << 1;

    float sum0 = 0.0f, sum1 = 0.0f;
    for (int s = 0; s < valid_seq; s++) {
        size_t in_idx = ((size_t)s * nh + h) * hd + d;
        float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(in + in_idx));
        sum0 += v.x;
        sum1 += v.y;
    }
    __nv_bfloat16 mean0_bf = __float2bfloat16(sum0 / valid_seq);
    __nv_bfloat16 mean1_bf = __float2bfloat16(sum1 / valid_seq);
    float mean0 = __bfloat162float(mean0_bf);
    float mean1 = __bfloat162float(mean1_bf);

    for (int s = 0; s < valid_seq; s++) {
        size_t in_idx = ((size_t)s * nh + h) * hd + d;
        float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(in + in_idx));
        *reinterpret_cast<__nv_bfloat162*>(head_out + (size_t)s * hd + d) =
            __floats2bfloat162_rn(v.x - mean0, v.y - mean1);
    }
    for (int s = valid_seq; s < padded_seq; s++) {
        *reinterpret_cast<__nv_bfloat162*>(head_out + (size_t)s * hd + d) =
            __floats2bfloat162_rn(0.0f, 0.0f);
    }
}

__global__ void k_mean_k_seqmajor(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ mean_out,
    int valid_seq, int nh, int hd) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= hd) return;
    float sum = 0.0f;
    for (int s = 0; s < valid_seq; s++) {
        size_t in_idx = ((size_t)s * nh + h) * hd + d;
        sum += __bfloat162float(in[in_idx]);
    }
    mean_out[h * hd + d] = __float2bfloat16(sum / valid_seq);
}

__global__ void k_group_mean_center_q_seqmajor_to_bhsd(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out_centered,
    __nv_bfloat16* __restrict__ out_mean,
    int valid_seq, int groups, int padded_seq, int nh, int hd) {
    int g = blockIdx.x;
    int h = blockIdx.y;
    int d2 = threadIdx.x;
    int hd2 = hd >> 1;
    if (d2 >= hd2) return;
    __nv_bfloat16* head_out = out_centered + (size_t)h * padded_seq * hd;
    __nv_bfloat16* head_mean = out_mean + ((size_t)h * groups + g) * hd;
    int base_s = g * 128;
    int d = d2 << 1;

    float sum0 = 0.0f, sum1 = 0.0f;
    #pragma unroll
    for (int j = 0; j < 128; j++) {
        int s = base_s + j;
        if (s < valid_seq) {
            size_t in_idx = ((size_t)s * nh + h) * hd + d;
            float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(in + in_idx));
            sum0 += v.x;
            sum1 += v.y;
        }
    }
    __nv_bfloat16 mean0_bf = __float2bfloat16(sum0 * (1.0f / 128.0f));
    __nv_bfloat16 mean1_bf = __float2bfloat16(sum1 * (1.0f / 128.0f));
    float mean0 = __bfloat162float(mean0_bf);
    float mean1 = __bfloat162float(mean1_bf);
    *reinterpret_cast<__nv_bfloat162*>(head_mean + d) = __halves2bfloat162(mean0_bf, mean1_bf);

    #pragma unroll
    for (int j = 0; j < 128; j++) {
        int s = base_s + j;
        float out0 = 0.0f, out1 = 0.0f;
        if (s < valid_seq) {
            size_t in_idx = ((size_t)s * nh + h) * hd + d;
            float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(in + in_idx));
            out0 = v.x - mean0;
            out1 = v.y - mean1;
        }
        *reinterpret_cast<__nv_bfloat162*>(head_out + (size_t)s * hd + d) =
            __floats2bfloat162_rn(out0, out1);
    }
}

__global__ void k_bhsd_to_seqmajor_bf16(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int in_seq, int out_seq, int nh, int hd) {
    size_t total = (size_t)out_seq * nh * hd;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        int d = idx % hd;
        size_t tmp = idx / hd;
        int h = tmp % nh;
        int s = tmp / nh;
        size_t in_idx = ((size_t)h * in_seq + s) * hd + d;
        out[idx] = in[in_idx];
    }
}

__global__ void k_bhsd_to_seqmajor_bf16_fp8(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out_bf16,
    __nv_fp8_e4m3* __restrict__ out_fp8,
    int in_seq, int out_seq, int nh, int hd) {
    size_t total = (size_t)out_seq * nh * hd;
    auto* out_fp8_bytes = (unsigned char*)out_fp8;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        int d = idx % hd;
        size_t tmp = idx / hd;
        int h = tmp % nh;
        int s = tmp / nh;
        size_t in_idx = ((size_t)h * in_seq + s) * hd + d;
        __nv_bfloat16 v = in[in_idx];
        if (out_bf16) out_bf16[idx] = v;
        if (out_fp8_bytes) out_fp8_bytes[idx] = bf16_to_fp8_byte(v);
    }
}

static void copy_bhsd_ptr_to_seqmajor_bf16(const __nv_bfloat16* in_ptr, void* o_ptr,
                                           int in_seq, int out_seq, int nh, int hd,
                                           cudaStream_t stream) {
    auto* out_ptr = (__nv_bfloat16*)o_ptr;
    size_t total = (size_t)out_seq * nh * hd;
    int max_blocks = std::max(1, 8 * g_nsm);
    int blocks = std::max(1, (int)std::min<size_t>((total + 255) / 256, (size_t)max_blocks));
    k_bhsd_to_seqmajor_bf16<<<blocks, 256, 0, stream>>>(in_ptr, out_ptr, in_seq, out_seq, nh, hd);
}

static void copy_bhsd_ptr_to_seqmajor_bf16_fp8(const __nv_bfloat16* in_ptr,
                                               void* o_bf16_ptr,
                                               __nv_fp8_e4m3* o_fp8_ptr,
                                               int in_seq, int out_seq, int nh, int hd,
                                               cudaStream_t stream) {
    auto* out_bf16 = (__nv_bfloat16*)o_bf16_ptr;
    size_t total = (size_t)out_seq * nh * hd;
    int max_blocks = std::max(1, 8 * g_nsm);
    int blocks = std::max(1, (int)std::min<size_t>((total + 255) / 256, (size_t)max_blocks));
    k_bhsd_to_seqmajor_bf16_fp8<<<blocks, 256, 0, stream>>>(in_ptr, out_bf16, o_fp8_ptr, in_seq, out_seq, nh, hd);
}

static void copy_bhsd_to_seqmajor_bf16(const at::Tensor& o_bhsd, void* o_ptr,
                                       int sq, int nh, int hd, cudaStream_t stream) {
    auto* in_ptr = (__nv_bfloat16*)o_bhsd.data_ptr();
    copy_bhsd_ptr_to_seqmajor_bf16(in_ptr, o_ptr, sq, sq, nh, hd, stream);
}

static void copy_bhsd_to_seqmajor_bf16_fp8(const at::Tensor& o_bhsd,
                                           void* o_bf16_ptr,
                                           __nv_fp8_e4m3* o_fp8_ptr,
                                           int sq, int nh, int hd,
                                           cudaStream_t stream) {
    auto* in_ptr = (__nv_bfloat16*)o_bhsd.data_ptr();
    copy_bhsd_ptr_to_seqmajor_bf16_fp8(in_ptr, o_bf16_ptr, o_fp8_ptr, sq, sq, nh, hd, stream);
}

static uint8_t *b_sage_q_packed = nullptr, *b_sage_k_packed = nullptr, *b_sage_v_packed = nullptr;
static uint8_t *b_sage_q_sf = nullptr, *b_sage_k_sf = nullptr, *b_sage_v_sf = nullptr;
static __nv_bfloat16 *b_sage_q_bhsd = nullptr, *b_sage_k_bhsd = nullptr, *b_sage_v_bhsd = nullptr;
static __nv_bfloat16 *b_sage_k_centered = nullptr, *b_sage_k_mean = nullptr, *b_sage_q_mean = nullptr;
static __nv_bfloat16 *b_sage_o = nullptr;
static float *b_sage_delta_s = nullptr;
static float *b_sage_softmax_lse = nullptr;
static int *b_sage_tile_count = nullptr;

static void sageattn3_py_bf16(void* q_ptr, void* k_ptr, void* v_ptr,
                              void* o_bf16_ptr, __nv_fp8_e4m3* o_fp8_ptr,
                              int sq, int sk, int nh, int hd, cudaStream_t stream) {
    py::gil_scoped_acquire gil;
    static py::object* sage3_fn = nullptr;
    if (sage3_fn == nullptr) {
        sage3_fn = new py::object(py::module_::import("sageattn3.api").attr("sageattn3_blackwell"));
    }

    auto Q = wrap_seqmajor_bf16(q_ptr, sq, nh, hd);
    auto K = wrap_seqmajor_bf16(k_ptr, sk, nh, hd);
    auto V = wrap_seqmajor_bf16(v_ptr, sk, nh, hd);
    auto O = (*sage3_fn)(Q, K, V).cast<at::Tensor>();
    copy_bhsd_to_seqmajor_bf16_fp8(O.narrow(2, 0, sq).contiguous(), o_bf16_ptr, o_fp8_ptr, sq, nh, hd, stream);
}

static void run_sage_delta_s_gemm(const __nv_bfloat16* qm_ptr,
                                  const __nv_bfloat16* k_centered_ptr,
                                  float* out_ptr,
                                  int nh, int q_groups, int k_seq, int hd,
                                  cudaStream_t stream) {
    float alpha = 1.0f, beta = 0.0f;
    long long stride_k = (long long)k_seq * hd;
    long long stride_qm = (long long)q_groups * hd;
    long long stride_out = (long long)q_groups * k_seq;
    CKBL(cublasSetStream(g_blasH, stream));
    CKBL(cublasGemmStridedBatchedEx(
        g_blasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        k_seq, q_groups, hd,
        &alpha,
        k_centered_ptr, CUDA_R_16BF, hd, stride_k,
        qm_ptr, CUDA_R_16BF, hd, stride_qm,
        &beta,
        out_ptr, CUDA_R_32F, k_seq, stride_out,
        nh,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static at::Tensor pad_seq_to_128(const at::Tensor& x) {
    int64_t seq = x.size(2);
    int64_t pad = (128 - (seq % 128)) % 128;
    if (pad == 0) {
        return x.contiguous();
    }
    auto out = torch::zeros({x.size(0), x.size(1), seq + pad, x.size(3)}, x.options());
    out.narrow(2, 0, seq).copy_(x);
    return out;
}

static void sageattn3_cpp_bf16(void* q_ptr, void* k_ptr, void* v_ptr,
                               void* o_bf16_ptr, __nv_fp8_e4m3* o_fp8_ptr,
                               int sq, int sk, int nh, int hd, cudaStream_t stream) {
    TORCH_CHECK(o_bf16_ptr != nullptr, "sageattn3_cpp_bf16 requires a BF16 output buffer");
    int64_t q_padded = ((int64_t)sq + 127) & ~127LL;
    int64_t k_padded = ((int64_t)sk + 127) & ~127LL;
    int64_t q_groups = q_padded / 128;
    float softmax_scale = 1.0f / sqrtf((float)hd);

    k_group_mean_center_q_seqmajor_to_bhsd<<<dim3((unsigned int)q_groups, (unsigned int)nh), 64, 0, stream>>>(
        (const __nv_bfloat16*)q_ptr, b_sage_q_bhsd, b_sage_q_mean, sq, (int)q_groups, (int)q_padded, nh, hd);
    k_center_k_seqmajor_to_bhsd<<<nh, 64, 0, stream>>>(
        (const __nv_bfloat16*)k_ptr, b_sage_k_centered, sk, (int)k_padded, nh, hd);

    scaled_fp4_quant_bf16_raw_contig(
        reinterpret_cast<const nv_bfloat16*>(b_sage_q_bhsd),
        b_sage_q_packed, b_sage_q_sf,
        1, nh, (int)q_padded, hd, stream);
    scaled_fp4_quant_permute_bf16_raw_contig(
        reinterpret_cast<const nv_bfloat16*>(b_sage_k_centered),
        b_sage_k_packed, b_sage_k_sf,
        1, nh, (int)k_padded, hd, stream);
    scaled_fp4_quant_trans_bf16_raw_strided(
        reinterpret_cast<const nv_bfloat16*>(v_ptr),
        b_sage_v_packed, b_sage_v_sf,
        1, nh, sk, hd,
        sk * nh * hd, hd, nh * hd, stream);

    run_sage_delta_s_gemm(
        b_sage_q_mean,
        b_sage_k_centered,
        b_sage_delta_s,
        nh, (int)q_groups, (int)k_padded, hd, stream);

    CK(cudaMemsetAsync(b_sage_tile_count, 0, sizeof(int), stream));
    mha_fwd_contiguous_bf16_raw(
        b_sage_q_packed, b_sage_k_packed, b_sage_v_packed,
        b_sage_q_sf, b_sage_k_sf, b_sage_v_sf,
        b_sage_delta_s,
        b_sage_o, b_sage_softmax_lse, b_sage_tile_count,
        1, nh, (int)q_padded, (int)k_padded, sk, hd,
        softmax_scale, false, true, -1, -1, stream);
    if (o_fp8_ptr) {
        copy_bhsd_ptr_to_seqmajor_bf16_fp8(b_sage_o, o_bf16_ptr, o_fp8_ptr, (int)q_padded, sq, nh, hd, stream);
    } else {
        copy_bhsd_ptr_to_seqmajor_bf16(b_sage_o, o_bf16_ptr, (int)q_padded, sq, nh, hd, stream);
    }
}
#define DO_GEMM  (g_profile_mode == 0 || g_profile_mode == 1 || g_profile_mode == 4 || g_profile_mode == 6)
#define DO_EWISE (g_profile_mode == 0 || g_profile_mode == 2 || g_profile_mode == 4 || g_profile_mode == 5)
#define DO_SDPA  (g_profile_mode == 0 || g_profile_mode == 3 || g_profile_mode == 5 || g_profile_mode == 6)

// Pre-allocated activation buffers
static __nv_bfloat16 *b_norm, *b_q, *b_k, *b_v, *b_attn, *b_sa_out;
static __nv_bfloat16 *b_ca_norm, *b_ca_q, *b_ca_k, *b_ca_v, *b_ca_out;
static __nv_bfloat16 *b_ffn_norm, *b_ffn_mid, *b_ffn_out;
static __nv_fp8_e4m3 *b_norm_fp8, *b_ca_norm_fp8, *b_ffn_norm_fp8;
static __nv_fp8_e4m3 *b_sa_out_fp8, *b_ca_out_fp8, *b_ffn_mid_fp8;
static __nv_fp8_e4m3 *b_enc_fp8;

// FP8 SDPA buffers
static __nv_fp8_e4m3 *b_q_fp8_attn = nullptr, *b_k_fp8_attn = nullptr, *b_v_fp8_attn = nullptr;
static float *g_fp8_sdpa_descale_q = nullptr, *g_fp8_sdpa_descale_k = nullptr;
static float *g_fp8_sdpa_descale_v = nullptr, *g_fp8_sdpa_descale_s = nullptr;
static float *g_fp8_sdpa_scale_s = nullptr,   *g_fp8_sdpa_scale_o = nullptr;
static float *g_fp8_sdpa_amax_s = nullptr,    *g_fp8_sdpa_amax_o = nullptr;

// CA K/V output cache
static __nv_bfloat16 *b_ca_k_cache = nullptr;
static __nv_bfloat16 *b_ca_v_cache = nullptr;
static bool g_ca_kv_cache_valid = false;

// TP scratch buffers
static float *b_tp_sumsq_qk = nullptr;
static float *b_tp_sumsq_cq = nullptr;
static float *b_tp_sumsq_ck = nullptr;

// GDN (GatedDeltaNet) buffers
static int g_gdn_key_dim = 0, g_gdn_value_dim = 0;
static int g_gdn_head_k = 0, g_gdn_head_v = 0;
static int g_gdn_conv_k = 4;  // conv kernel size
static __nv_bfloat16 *b_gdn_q = nullptr, *b_gdn_k = nullptr;
static __nv_bfloat16 *b_gdn_v = nullptr, *b_gdn_g = nullptr;
static __nv_bfloat16 *b_gdn_a = nullptr, *b_gdn_b = nullptr;
static __nv_bfloat16 *b_gdn_out = nullptr;
static __nv_bfloat16 *b_gdn_conv_scratch = nullptr;
static __nv_fp8_e4m3 *b_gdn_out_fp8 = nullptr;
static float *b_gdn_gates_g = nullptr, *b_gdn_gates_beta = nullptr;
static float *b_gdn_state = nullptr;  // [NL_GDN * NH * K * V]
static int g_n_gdn_layers = 0;
static bool g_gdn_use_cublas_chunk = false; // Recurrent (1.27x). Chunk correct but needs MMA fwd_o to beat recurrent.
static bool g_cutlass_fused_ffn1 = false;
static bool g_cutlass_fused_ffn2 = false;
static bool g_fp4_ffn2 = false;
static bool g_cutlass_fused_ffn1_fp4 = false;
static std::vector<int> g_gdn_layer_indices;  // which layers are GDN
// Chunk algorithm scratch buffers (allocated in engine_init)
static float *b_gdn_chunk_w = nullptr;     // [NC, NH, BT, K]
static float *b_gdn_chunk_u = nullptr;     // [NC, NH, BT, V]
static float *b_gdn_chunk_gcum = nullptr;  // [NC, NH, BT]

// Graph-internal input buffers
static __nv_bfloat16 *g_x_buf = nullptr, *g_enc_buf = nullptr;
static float *g_temb_buf = nullptr, *g_ssts_buf = nullptr, *g_rope_buf = nullptr;
static bool g_use_graph = false;
static cudaStream_t g_graph_stream = nullptr;
static std::unordered_map<uint64_t, cudaGraphExec_t> g_graphs_by_key;
static uint8_t *b_ffn_mid_fp4 = nullptr;
static uint8_t *b_ffn_mid_fp4_scales = nullptr;
static uint8_t *b_ffn_mid_fp4_scales_swizzled = nullptr;
static float *g_fp4_normconst = nullptr;

// ============================================================
// cuBLASLt chunk GDN: helper functions (need globals above)
// ============================================================
static std::unordered_map<int, ChunkGdnGemms> g_chunk_gdn_gemms;

static ChunkGdnGemms* get_chunk_gdn_gemms(int seq) {
    auto it = g_chunk_gdn_gemms.find(seq);
    if (it != g_chunk_gdn_gemms.end()) return &it->second;
    auto& cg = g_chunk_gdn_gemms[seq];
    int NHpr = g_NH_per_rank;
    cg.build(g_ltH, seq, NHpr, g_gdn_head_k, g_gdn_head_v,
             g_gemm_ws, 32 * 1024 * 1024);
    return &cg;
}

static void gdn_cublas_chunk_forward(
    __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
    float* g, float* beta,
    __nv_bfloat16* o, float* state,
    float* w_buf, float* u_buf, float* gcum_buf,
    int T, int NH, int K, int V, float scale,
    cudaStream_t stream)
{
    const int BT = GDN_CHUNK_BT;
    int NC = (T + BT - 1) / BT;

    // Save inputs for debugging (first call only)
    static bool _dbg_saved = false;
    if (!_dbg_saved) {
        _dbg_saved = true;
        CK(cudaStreamSynchronize(stream));
        size_t qk_n = (size_t)T * NH * K, v_n = (size_t)T * NH * V, g_n = (size_t)T * NH;
        auto opts16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
        auto opts32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        // Save as raw binary: q(bf16), k(bf16), v(bf16), g(f32), beta(f32)
        auto q_cpu = torch::from_blob(q, {(long)qk_n}, opts16).clone().cpu();
        auto k_cpu = torch::from_blob(k, {(long)qk_n}, opts16).clone().cpu();
        auto v_cpu = torch::from_blob(v, {(long)v_n}, opts16).clone().cpu();
        auto g_cpu = torch::from_blob(g, {(long)g_n}, opts32).clone().cpu();
        auto b_cpu = torch::from_blob(beta, {(long)g_n}, opts32).clone().cpu();
        FILE* fp = fopen("/tmp/_gdn_inputs.bin", "wb");
        fwrite(q_cpu.data_ptr(), 2, qk_n, fp);
        fwrite(k_cpu.data_ptr(), 2, qk_n, fp);
        fwrite(v_cpu.data_ptr(), 2, v_n, fp);
        fwrite(g_cpu.data_ptr(), 4, g_n, fp);
        fwrite(b_cpu.data_ptr(), 4, g_n, fp);
        fclose(fp);
        printf("  [DBG] Saved chunk inputs T=%d NH=%d K=%d V=%d scale=%.6f\n", T, NH, K, V, scale);
    }

    // 3-kernel chunk pipeline: prepare → chunk_h_simple → chunk_fwd_o
    // Phase 0: L2-normalize Q and K
    k_gdn_l2norm_scale<<<dim3(T, NH), 256, 0, stream>>>(q, k, K, scale);

    // Debug: trace data flow

    // Phase 1: Prepare w, u, gcum (K already L2-normalized, krnorm≈1)
    {
        size_t smem = gdn_chunk_prepare_smem(K, V);
        if (smem > 48 * 1024)
            cudaFuncSetAttribute(k_gdn_chunk_prepare,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        k_gdn_chunk_prepare<<<dim3(NC, NH), 64, smem, stream>>>(
            k, v, g, beta,
            w_buf, u_buf, gcum_buf,
            T, NH, K, V);
    }


    // Phase 2: State propagation → h_snapshots (MMA-based, verified correct)
    gdn_chunk_h_launch(
        w_buf, u_buf, gcum_buf,
        b_gdn_hout,    // [NC+1, NH, K, V] bf16
        state,          // final state
        state,          // h0
        NC, NH, K, V, stream);


    // Phase 3: Output from h_snapshots (parallel across chunks!)
    {
        int NV_blocks = V / GDN_BV;
        size_t fwd_o_smem = BT * GDN_BV * sizeof(float) + BT * sizeof(float);  // v_new + gcum
        k_gdn_chunk_fwd_o<<<dim3(NV_blocks, NC, NH), 256, fwd_o_smem, stream>>>(
            q, k,               // [T, NH, K] bf16 (L2-normalized)
            w_buf, u_buf,       // [NC, NH, BT, K/V] fp32
            b_gdn_hout,        // [NC+1, NH, K, V] bf16
            gcum_buf,           // [NC, NH, BT] fp32
            o,                  // [T, NH, V] bf16 output
            T, NH, K, V, scale);
    }
}

// ============================================================
// NCCL setup
// ============================================================
torch::Tensor engine_nccl_get_unique_id() {
    ncclUniqueId id;
    CKNCCL(ncclGetUniqueId(&id));
    auto out = torch::empty({NCCL_UNIQUE_ID_BYTES}, torch::TensorOptions().dtype(torch::kUInt8));
    std::memcpy(out.data_ptr(), &id, NCCL_UNIQUE_ID_BYTES);
    return out;
}

void engine_nccl_comm_init(torch::Tensor uid_bytes, int64_t rank, int64_t world) {
    TORCH_CHECK(uid_bytes.numel() == NCCL_UNIQUE_ID_BYTES, "bad uid size");
    TORCH_CHECK(uid_bytes.dtype() == torch::kUInt8, "uid must be uint8");
    auto cpu = uid_bytes.to(torch::kCPU).contiguous();
    ncclUniqueId id;
    std::memcpy(&id, cpu.data_ptr(), NCCL_UNIQUE_ID_BYTES);
    g_tp_rank = (int)rank;
    g_tp_world = (int)world;
    CKNCCL(ncclCommInitRank(&g_nccl_comm, (int)world, id, (int)rank));
    printf("  NCCL comm initialized: rank=%d world=%d\n", (int)rank, (int)world);
}

// ============================================================
// Per-seq GEMM descriptors
// ============================================================
struct SeqGemms {
    FP8Gemm sq, sk, sv, so, f1, f2;
    BF16Gemm sq_bf16, sk_bf16, sv_bf16, so_bf16;
    BF16Gemm f1_bf16, f2_bf16;  // for force_bf16_gemms mode
    CudnnSDPA sa_sdpa;
    CudnnSDPA_FP8 sa_sdpa_fp8;
    bool fp8_sdpa_built = false;
    bool bf16_attn_built = false;
    bool bf16_ffn_built = false;
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    CutlassFusedFfn1 f1_fused;
    bool f1_fused_built = false;
    CutlassFusedFfn1Fp4 f1_fused_fp4;
    bool f1_fused_fp4_built = false;
    CutlassFusedFfn2 f2_fused;
    bool f2_fused_built = false;
    CutlassFp4RowwiseGemm f2_fp4;
    bool f2_fp4_built = false;
#endif

    // GatedDeltaNet GEMMs
    FP8Gemm gdn_q, gdn_k, gdn_v, gdn_g, gdn_o;
    BF16Gemm gdn_a, gdn_b;  // a/b have tiny N=20, FP8 not supported
    BF16Gemm gdn_q_bf16, gdn_k_bf16, gdn_v_bf16, gdn_g_bf16, gdn_o_bf16;  // for force mode
    bool gdn_built = false;
    bool gdn_bf16_built = false;
};
static std::unordered_map<int, SeqGemms> g_gemms_by_seq;
static FP8Gemm g_ck, g_cv;  // CA K/V only depend on ctx
static BF16Gemm g_ck_bf16, g_cv_bf16;  // BF16 versions for force_bf16_gemms mode
static bool g_ck_bf16_built = false;

// Per-ca_len GEMM descriptors
struct CaGemms {
    FP8Gemm cq, co;
    BF16Gemm cq_bf16, co_bf16;
    CudnnSDPA ca_sdpa;
    bool bf16_ca_built = false;
};
static std::unordered_map<int, CaGemms> g_ca_gemms_by_len;

// Helper: get or create GEMMs for a specific seq_len
SeqGemms* get_gemms(int seq) {
    auto it = g_gemms_by_seq.find(seq);
    if (it != g_gemms_by_seq.end()) return &it->second;

    auto& g = g_gemms_by_seq[seq];
    int D = g_D, FFN = g_FFN;
    int Dpr = g_D_per_rank, FFNpr = g_FFN_per_rank;
    int row_epi = 1;
    // Column-parallel SA projections
    g.sq.setup(g_ltH, seq, Dpr, D, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, 1);
    g.sk.setup(g_ltH, seq, Dpr, D, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, 1);
    g.sv.setup(g_ltH, seq, Dpr, D, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, 1);
    // FFN1 (column-parallel, SiLU applied separately — no fused epilogue)
    g.f1.setup(g_ltH, seq, FFNpr, D, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, 1);
    // Row-parallel
    g.so.setup(g_ltH, seq, D, Dpr, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, row_epi);
    g.f2.setup(g_ltH, seq, D, FFNpr, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, row_epi);
    // BF16 attention GEMMs
    if ((g_bf16_attn || g_force_bf16_gemms) && !g.bf16_attn_built) {
        g.sq_bf16.setup(g_ltH, seq, Dpr, D, g_gemm_ws, 32*1024*1024, 1);
        g.sk_bf16.setup(g_ltH, seq, Dpr, D, g_gemm_ws, 32*1024*1024, 1);
        g.sv_bf16.setup(g_ltH, seq, Dpr, D, g_gemm_ws, 32*1024*1024, 1);
        g.so_bf16.setup(g_ltH, seq, D, Dpr, g_gemm_ws, 32*1024*1024, row_epi);
        g.bf16_attn_built = true;
    }
    // BF16 FFN GEMMs
    if (g_force_bf16_gemms && !g.bf16_ffn_built) {
        g.f1_bf16.setup(g_ltH, seq, FFNpr, D, g_gemm_ws, 32*1024*1024, 1);
        g.f2_bf16.setup(g_ltH, seq, D, FFNpr, g_gemm_ws, 32*1024*1024, row_epi);
        g.bf16_ffn_built = true;
    }
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    if (g_cutlass_fused_ffn1 && !g.f1_fused_built) {
        g.f1_fused.setup(seq, FFNpr, D);
        g.f1_fused_built = true;
    }
    if (g_fp4_ffn2 && g_cutlass_fused_ffn1_fp4 && !g.f1_fused_fp4_built) {
        g.f1_fused_fp4.setup(seq, FFNpr, D);
        g.f1_fused_fp4_built = true;
    }
    if (g_cutlass_fused_ffn2 && !g.f2_fused_built) {
        g.f2_fused.setup(seq, D, FFNpr);
        g.f2_fused_built = true;
    }
    if (g_fp4_ffn2 && !g.f2_fp4_built) {
        g.f2_fp4.setup(seq, D, FFNpr);
        g.f2_fp4_built = true;
    }
#endif
    float scl = 1.f / sqrtf((float)g_HD);
    g.sa_sdpa.build(g_cudnnH, g_NH_per_rank, seq, seq, g_HD, scl);
    if (g_fp8_sdpa_enabled) {
        g.sa_sdpa_fp8.build(g_cudnnH, g_NH_per_rank, seq, seq, g_HD, scl);
        g.fp8_sdpa_built = true;
    }

    // GDN GEMMs (built lazily on first GDN layer forward)
    // These use global dims since they're the same for all GDN layers
    if (!g.gdn_built && g_gdn_key_dim > 0) {
        int KD = g_gdn_key_dim, VD = g_gdn_value_dim;
        int KDpr = KD / g_tp_world, VDpr = VD / g_tp_world;
        // Col-parallel projections: output sharded
        g.gdn_q.setup(g_ltH, seq, KDpr, D, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, 1);
        g.gdn_k.setup(g_ltH, seq, KDpr, D, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, 1);
        g.gdn_v.setup(g_ltH, seq, VDpr, D, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, 1);
        g.gdn_g.setup(g_ltH, seq, VDpr, D, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, 1);
        // a/b: small output (NH=20), use BF16 GEMMs (FP8 doesn't support tiny N)
        g.gdn_a.setup(g_ltH, seq, g_NH, D, g_gemm_ws, 32*1024*1024, 1);
        g.gdn_b.setup(g_ltH, seq, g_NH, D, g_gemm_ws, 32*1024*1024, 1);
        // Row-parallel output projection
        int row_epi = 1;
        g.gdn_o.setup(g_ltH, seq, D, VDpr, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, row_epi);
        g.gdn_built = true;
    }
    // BF16 GDN GEMMs for force mode
    if (g_force_bf16_gemms && !g.gdn_bf16_built && g_gdn_key_dim > 0) {
        int KD = g_gdn_key_dim, VD = g_gdn_value_dim;
        int KDpr = KD / g_tp_world, VDpr = VD / g_tp_world;
        g.gdn_q_bf16.setup(g_ltH, seq, KDpr, D, g_gemm_ws, 32*1024*1024, 1);
        g.gdn_k_bf16.setup(g_ltH, seq, KDpr, D, g_gemm_ws, 32*1024*1024, 1);
        g.gdn_v_bf16.setup(g_ltH, seq, VDpr, D, g_gemm_ws, 32*1024*1024, 1);
        g.gdn_g_bf16.setup(g_ltH, seq, VDpr, D, g_gemm_ws, 32*1024*1024, 1);
        int row_epi_gdn = 1;
        g.gdn_o_bf16.setup(g_ltH, seq, D, VDpr, g_gemm_ws, 32*1024*1024, row_epi_gdn);
        g.gdn_bf16_built = true;
    }

    return &g;
}

CaGemms* get_ca_gemms(int ca_len) {
    auto it = g_ca_gemms_by_len.find(ca_len);
    if (it != g_ca_gemms_by_len.end()) return &it->second;
    auto& g = g_ca_gemms_by_len[ca_len];
    int D = g_D;
    int Dpr = g_D_per_rank;
    int row_epi = 1;
    g.cq.setup(g_ltH, ca_len, Dpr, D, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, 1);
    g.co.setup(g_ltH, ca_len, D, Dpr, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, row_epi);
    float scl = 1.f / sqrtf((float)g_HD);
    int ca_kv_len = (g_actual_ctx > 0) ? g_actual_ctx : g_ctx;
    g.ca_sdpa.build(g_cudnnH, g_NH_per_rank, ca_len, ca_kv_len, g_HD, scl);
    // BF16 CA GEMMs for force_bf16_gemms mode
    if (g_force_bf16_gemms && !g.bf16_ca_built) {
        g.cq_bf16.setup(g_ltH, ca_len, Dpr, D, g_gemm_ws, 32*1024*1024, 1);
        g.co_bf16.setup(g_ltH, ca_len, D, Dpr, g_gemm_ws, 32*1024*1024, row_epi);
        g.bf16_ca_built = true;
    }
    // BF16 CA K/V GEMMs (global, built once)
    // Always build BF16 CA K/V GEMMs (FP8 has tiling bugs at small ctx)
    if (!g_ck_bf16_built) {
        g_ck_bf16.setup(g_ltH, ca_kv_len, Dpr, g_CA_K_DIM, g_gemm_ws, 32*1024*1024, 1);
        g_cv_bf16.setup(g_ltH, ca_kv_len, Dpr, g_CA_K_DIM, g_gemm_ws, 32*1024*1024, 1);
        g_ck_bf16_built = true;
    }
    return &g;
}

static void run_ffn2_fp4(const Layer& L, SeqGemms* gm, int seq, cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    TORCH_CHECK(L.f2_w4 && L.f2_s4, "FFN2 FP4 weights not loaded");
    gm->f2_fp4.run(
        L.f2_w4,
        L.f2_s4,
        b_ffn_mid_fp4,
        b_ffn_mid_fp4_scales_swizzled,
        b_ffn_out,
        stream);
    k_add_bias_vec<<<8 * g_nsm, 256, 0, stream>>>(b_ffn_out, L.f2b, seq, g_D);
#else
    TORCH_CHECK(false, "FFN2 FP4 path requested without SM120 CUTLASS support");
#endif
}

// ============================================================
// Init: allocate buffers
// ============================================================
void engine_init(int64_t D, int64_t NH, int64_t HD, int64_t NL,
                 int64_t FFN, int64_t CA_K_DIM, int64_t max_seq, int64_t ctx,
                 int64_t gdn_key_dim, int64_t gdn_value_dim) {
    g_D = D; g_NH = NH; g_HD = HD; g_NL = NL; g_FFN = FFN; g_CA_K_DIM = CA_K_DIM;
    g_max_seq = max_seq; g_ctx = ctx;

    TORCH_CHECK(D % g_tp_world == 0, "D not divisible by tp_world");
    TORCH_CHECK(NH % g_tp_world == 0, "NH not divisible by tp_world");
    TORCH_CHECK(FFN % g_tp_world == 0, "FFN not divisible by tp_world");
    g_D_per_rank = D / g_tp_world;
    g_NH_per_rank = NH / g_tp_world;
    g_FFN_per_rank = FFN / g_tp_world;

    cudaDeviceProp prop; int dev; CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    g_nsm = prop.multiProcessorCount;

    const char* lt_autotune_env = getenv("KAIROS_LT_AUTOTUNE");
    if (lt_autotune_env) {
        g_lt_autotune_enabled = atoi(lt_autotune_env) != 0;
    }
    const char* lt_topk_env = getenv("KAIROS_LT_AUTOTUNE_TOPK");
    if (lt_topk_env) {
        g_lt_autotune_topk = std::max(1, atoi(lt_topk_env));
    }
    const char* gemm_ws_env = getenv("KAIROS_GEMM_WS_MB");
    if (gemm_ws_env) {
        g_gemm_ws_bytes = (size_t)std::max(1, atoi(gemm_ws_env)) * 1024 * 1024;
    }
    const char* fast_accum_env = getenv("KAIROS_LT_FAST_ACCUM_FP8");
    if (fast_accum_env) {
        g_lt_fast_accum_fp8 = atoi(fast_accum_env) != 0;
    }
    const char* fused_ffn1_env = getenv("KAIROS_FUSED_FFN1_CUTLASS");
    if (fused_ffn1_env) {
        g_cutlass_fused_ffn1 = atoi(fused_ffn1_env) != 0;
    }
    const char* fused_ffn2_env = getenv("KAIROS_FUSED_FFN2_CUTLASS");
    if (fused_ffn2_env) {
        g_cutlass_fused_ffn2 = atoi(fused_ffn2_env) != 0;
    }
    const char* fp4_ffn2_env = getenv("KAIROS_FP4_FFN2");
    if (fp4_ffn2_env) {
        g_fp4_ffn2 = atoi(fp4_ffn2_env) != 0;
    }
    const char* fused_ffn1_fp4_env = getenv("KAIROS_FUSED_FFN1_FP4");
    if (fused_ffn1_fp4_env) {
        g_cutlass_fused_ffn1_fp4 = atoi(fused_ffn1_fp4_env) != 0;
    }
    ensure_gemm_runtime();
    if (!g_cudnnH) {
        cudnnCreate(&g_cudnnH);
    }
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    if (g_cutlass_fused_ffn1) {
        printf("  [CUTLASS FFN1] disabled: SM120 CUTLASS support not compiled in\n");
        g_cutlass_fused_ffn1 = false;
    }
    if (g_cutlass_fused_ffn2) {
        printf("  [CUTLASS FFN2] disabled: SM120 CUTLASS support not compiled in\n");
        g_cutlass_fused_ffn2 = false;
    }
#endif
    if (g_cutlass_fused_ffn1) {
        printf("  [CUTLASS FFN1] enabled via KAIROS_FUSED_FFN1_CUTLASS=1\n");
    }
    if (g_cutlass_fused_ffn2) {
        printf("  [CUTLASS FFN2] enabled via KAIROS_FUSED_FFN2_CUTLASS=1\n");
    }
    if (g_fp4_ffn2) {
        printf("  [FP4 FFN2] requested via KAIROS_FP4_FFN2=1\n");
    }
    if (g_cutlass_fused_ffn1_fp4) {
        printf("  [CUTLASS FFN1 FP4] enabled via KAIROS_FUSED_FFN1_FP4=1\n");
    }
    if (g_lt_fast_accum_fp8) {
        printf("  [cuBLASLt] FP8 fast accumulation enabled via KAIROS_LT_FAST_ACCUM_FP8=1\n");
    }

#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    if (g_fp4_ffn2) {
        printf("  [FP4 FFN2] disabled: SM120 CUTLASS support not compiled in\n");
        g_fp4_ffn2 = false;
    }
#endif
    if (g_fp4_ffn2) {
        TORCH_CHECK(g_FFN_per_rank % 32 == 0, "FP4 FFN2 requires FFN per rank divisible by 32, got ", g_FFN_per_rank);
    }

    auto abf16 = [](size_t n) { __nv_bfloat16* p; CK(cudaMalloc(&p, n * 2)); return p; };
    auto afp8  = [](size_t n) { __nv_fp8_e4m3* p; CK(cudaMalloc(&p, n)); return p; };
    auto au8   = [](size_t n) { uint8_t* p; CK(cudaMalloc(&p, n)); return p; };
    auto af32n = [](size_t n) { float* p; CK(cudaMalloc(&p, n * sizeof(float))); return p; };

    b_norm = abf16(max_seq * D); b_q = abf16(max_seq * D); b_k = abf16(max_seq * D); b_v = abf16(max_seq * D);
    b_attn = abf16(max_seq * D); b_sa_out = abf16(max_seq * D);
    b_ca_norm = abf16(max_seq * D); b_ca_q = abf16(max_seq * D);
    b_ca_k = abf16(ctx * D); b_ca_v = abf16(ctx * D); b_ca_out = abf16(max_seq * D);
    b_ffn_norm = abf16(max_seq * D); b_ffn_mid = abf16(max_seq * FFN); b_ffn_out = abf16(max_seq * D);
    CK(cudaMemset(b_ffn_mid, 0, max_seq * FFN * sizeof(__nv_bfloat16)));
    b_norm_fp8 = afp8(max_seq * D); b_ca_norm_fp8 = afp8(max_seq * D); b_ffn_norm_fp8 = afp8(max_seq * D);
    b_sa_out_fp8 = afp8(max_seq * D); b_ca_out_fp8 = afp8(max_seq * D); b_ffn_mid_fp8 = afp8(max_seq * FFN);
    b_enc_fp8 = afp8(ctx * CA_K_DIM);
    if (g_fp4_ffn2) {
        size_t rows_padded = (size_t)fp4_scale_rows((int)max_seq);
        size_t scale_cols = (size_t)fp4_scale_cols(g_FFN_per_rank);
        CK(cudaMalloc((void**)&b_ffn_mid_fp4, rows_padded * (size_t)g_FFN_per_rank / 2));
        CK(cudaMalloc((void**)&b_ffn_mid_fp4_scales, rows_padded * scale_cols));
        CK(cudaMalloc((void**)&b_ffn_mid_fp4_scales_swizzled, rows_padded * scale_cols));
        if (g_cutlass_fused_ffn1_fp4 && g_fp4_normconst == nullptr) {
            float h_normconst = 6.0f;
            CK(cudaMalloc((void**)&g_fp4_normconst, sizeof(float)));
            CK(cudaMemcpy(g_fp4_normconst, &h_normconst, sizeof(float), cudaMemcpyHostToDevice));
        }
    }
    {
        size_t sage_seq = (size_t)(((int)max_seq + 127) / 128) * 128;
        size_t nh_hd = (size_t)g_NH_per_rank * (size_t)HD;
        size_t sage_groups = sage_seq / 128;
        b_sage_q_bhsd = abf16(nh_hd * sage_seq);
        b_sage_k_bhsd = abf16(nh_hd * sage_seq);
        b_sage_v_bhsd = abf16(nh_hd * sage_seq);
        b_sage_k_centered = abf16(nh_hd * sage_seq);
        b_sage_o = abf16(nh_hd * sage_seq);
        b_sage_k_mean = abf16((size_t)g_NH_per_rank * (size_t)HD);
        b_sage_q_mean = abf16((size_t)g_NH_per_rank * sage_groups * (size_t)HD);
        b_sage_delta_s = af32n((size_t)g_NH_per_rank * sage_groups * sage_seq);
        b_sage_q_packed = au8(nh_hd * sage_seq / 2);
        b_sage_k_packed = au8(nh_hd * sage_seq / 2);
        b_sage_v_packed = au8(nh_hd * sage_seq / 2);
        b_sage_q_sf = au8(nh_hd * sage_seq / 16);
        b_sage_k_sf = au8(nh_hd * sage_seq / 16);
        b_sage_v_sf = au8(nh_hd * sage_seq / 16);
        b_sage_softmax_lse = af32n((size_t)g_NH_per_rank * sage_seq);
        CK(cudaMalloc(&b_sage_tile_count, sizeof(int)));
    }

    // FP8 SDPA (opt-in via ENGINE_FP8_SDPA=1)
    const char* fp8_sdpa_env = getenv("ENGINE_FP8_SDPA");
    g_fp8_sdpa_enabled = (fp8_sdpa_env && atoi(fp8_sdpa_env) == 1);
    const char* bf16_attn_env = getenv("ENGINE_BF16_ATTN");
    g_bf16_attn = (bf16_attn_env && atoi(bf16_attn_env) == 1);
    if (g_bf16_attn) {
        printf("  [BF16 ATTN] enabled\n");
    }
    if (g_fp8_sdpa_enabled) {
        b_q_fp8_attn = afp8(max_seq * g_D_per_rank);
        b_k_fp8_attn = afp8(max_seq * g_D_per_rank);
        b_v_fp8_attn = afp8(max_seq * g_D_per_rank);
        auto af32 = []() { float* p; CK(cudaMalloc(&p, sizeof(float))); return p; };
        g_fp8_sdpa_descale_q = af32(); g_fp8_sdpa_descale_k = af32();
        g_fp8_sdpa_descale_v = af32(); g_fp8_sdpa_descale_s = af32();
        g_fp8_sdpa_scale_s   = af32(); g_fp8_sdpa_scale_o   = af32();
        g_fp8_sdpa_amax_s    = af32(); g_fp8_sdpa_amax_o    = af32();
        float one_val = 1.f;
        for (float* p : {g_fp8_sdpa_descale_q, g_fp8_sdpa_descale_k, g_fp8_sdpa_descale_v,
                         g_fp8_sdpa_descale_s, g_fp8_sdpa_scale_s, g_fp8_sdpa_scale_o}) {
            CK(cudaMemcpy(p, &one_val, sizeof(float), cudaMemcpyHostToDevice));
        }
        printf("  [FP8 SDPA] enabled via ENGINE_FP8_SDPA=1\n");
    }

    // CA K/V cache
    b_ca_k_cache = abf16((size_t)NL * ctx * g_D_per_rank);
    b_ca_v_cache = abf16((size_t)NL * ctx * g_D_per_rank);
    g_ca_kv_cache_valid = false;

    // Graph-internal I/O buffers
    CK(cudaMalloc(&g_x_buf, (size_t)max_seq * D * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&g_enc_buf, (size_t)ctx * CA_K_DIM * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&g_temb_buf, (size_t)max_seq * 6 * D * sizeof(float)));
    CK(cudaMalloc(&g_ssts_buf, (size_t)NL * 6 * D * sizeof(float)));
    CK(cudaMalloc(&g_rope_buf, (size_t)max_seq * 2 * HD * sizeof(float)));
    CK(cudaStreamCreateWithFlags(&g_graph_stream, cudaStreamNonBlocking));

    // TP RMSNorm scratch buffers
    if (g_tp_world > 1) {
        CK(cudaMalloc(&b_tp_sumsq_qk, (size_t)2 * max_seq * sizeof(float)));
        CK(cudaMalloc(&b_tp_sumsq_cq, (size_t)max_seq * sizeof(float)));
        CK(cudaMalloc(&b_tp_sumsq_ck, (size_t)ctx * sizeof(float)));
    }

    // CA K/V GEMMs (column-parallel)
    g_ck.setup(g_ltH, ctx, g_D_per_rank, CA_K_DIM, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, 1);
    g_cv.setup(g_ltH, ctx, g_D_per_rank, CA_K_DIM, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, 1);

    g_layers.resize(NL);

    // GDN buffers
    g_gdn_key_dim = (int)gdn_key_dim;
    g_gdn_value_dim = (int)gdn_value_dim;
    if (gdn_key_dim > 0 && gdn_value_dim > 0) {
        g_gdn_head_k = gdn_key_dim / NH;
        g_gdn_head_v = gdn_value_dim / NH;
        int KDpr = gdn_key_dim / g_tp_world;
        int VDpr = gdn_value_dim / g_tp_world;
        int NHpr = NH / g_tp_world;
        b_gdn_q = abf16(max_seq * KDpr);
        b_gdn_k = abf16(max_seq * KDpr);
        b_gdn_v = abf16(max_seq * VDpr);
        b_gdn_g = abf16(max_seq * VDpr);
        b_gdn_a = abf16(max_seq * NH);  // replicated
        b_gdn_b = abf16(max_seq * NH);
        b_gdn_out = abf16(max_seq * VDpr);
        b_gdn_conv_scratch = abf16(max_seq * (size_t)std::max(KDpr, VDpr));
        b_gdn_out_fp8 = afp8(max_seq * VDpr);
        b_gdn_gates_g = (float*)nullptr;
        b_gdn_gates_beta = (float*)nullptr;
        CK(cudaMalloc(&b_gdn_gates_g, (size_t)max_seq * NHpr * sizeof(float)));
        CK(cudaMalloc(&b_gdn_gates_beta, (size_t)max_seq * NHpr * sizeof(float)));
        // Count GDN layers
        g_n_gdn_layers = 0;
        g_gdn_layer_indices.clear();
        for (int i = 0; i < NL; i++) {
            if (i % 4 == 3) { g_gdn_layer_indices.push_back(i); g_n_gdn_layers++; }
        }
        // Per-GDN-layer recurrent state: [NH_per_rank, K, V]
        size_t state_per_layer = (size_t)NHpr * g_gdn_head_k * g_gdn_head_v;
        CK(cudaMalloc(&b_gdn_state, (size_t)g_n_gdn_layers * state_per_layer * sizeof(float)));
        CK(cudaMemset(b_gdn_state, 0, (size_t)g_n_gdn_layers * state_per_layer * sizeof(float)));
        printf("  [GDN] %d layers, key_dim=%d value_dim=%d head_k=%d head_v=%d\n",
               g_n_gdn_layers, g_gdn_key_dim, g_gdn_value_dim, g_gdn_head_k, g_gdn_head_v);
        printf("  [GDN] state: %.1f MB\n",
               (double)(g_n_gdn_layers * state_per_layer * sizeof(float)) / (1024.0 * 1024.0));
        // Chunk algorithm scratch buffers — per HEAD dimensions (not per-rank)
        int NC = ((int)max_seq + GDN_CHUNK_BT - 1) / GDN_CHUNK_BT;
        size_t w_elems = (size_t)NC * NHpr * GDN_CHUNK_BT * g_gdn_head_k;
        size_t u_elems = (size_t)NC * NHpr * GDN_CHUNK_BT * g_gdn_head_v;
        size_t gcum_elems = (size_t)NC * NHpr * GDN_CHUNK_BT;
        CK(cudaMalloc(&b_gdn_chunk_w, w_elems * sizeof(float)));
        CK(cudaMalloc(&b_gdn_chunk_u, u_elems * sizeof(float)));
        CK(cudaMalloc(&b_gdn_chunk_gcum, gcum_elems * sizeof(float)));
        double chunk_mb = (double)(w_elems + u_elems + gcum_elems) * sizeof(float) / (1024.0 * 1024.0);
        printf("  [GDN] chunk scratch: %.1f MB (NC=%d)\n", chunk_mb, NC);

        // cuBLASLt chunk GDN bf16 scratch buffers
        size_t kkt_elems   = (size_t)NC * NHpr * GDN_CHUNK_BT * GDN_CHUNK_BT;
        size_t sk_elems    = (size_t)NC * NHpr * GDN_CHUNK_BT * g_gdn_head_k;
        size_t sv_elems    = (size_t)NC * NHpr * GDN_CHUNK_BT * g_gdn_head_v;
        size_t coeff_elems = (size_t)NC * NHpr * GDN_CHUNK_BT;
        CK(cudaMalloc(&b_gdn_kkt_bf16,  kkt_elems * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&b_gdn_sk_bf16,   sk_elems  * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&b_gdn_sv_bf16,   sv_elems  * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&b_gdn_w_bf16,    sk_elems  * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&b_gdn_u_bf16,    sv_elems  * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&b_gdn_coeff,     coeff_elems * sizeof(float)));
        CK(cudaMalloc(&b_gdn_beta_f32,  coeff_elems * sizeof(float)));
        // h_out: state snapshots for output kernel [(NC+1) * NH * K * V] bf16
        size_t hout_elems = (size_t)(NC + 1) * NHpr * g_gdn_head_k * g_gdn_head_v;
        CK(cudaMalloc(&b_gdn_hout, hout_elems * sizeof(__nv_bfloat16)));
        // v_new: delta values for output kernel [NC * NH * BT * V] float32 (reuse u_buf)
        double cublas_mb = (double)((kkt_elems + 2*sk_elems + 2*sv_elems) * sizeof(__nv_bfloat16)
                                    + 2*coeff_elems * sizeof(float)
                                    + hout_elems * sizeof(float)) / (1024.0 * 1024.0);
        printf("  [GDN] cuBLASLt chunk scratch: %.1f MB\n", cublas_mb);
    }

    g_ready = true;
    printf("Kairos Engine: %lld layers, max_seq=%lld, ctx=%lld, dim=%lld, ffn=%lld, ca_k=%lld, tp=(%d/%d) D/r=%d NH/r=%d FFN/r=%d\n",
        NL, max_seq, ctx, D, FFN, CA_K_DIM,
        g_tp_rank, g_tp_world, g_D_per_rank, g_NH_per_rank, g_FFN_per_rank);
}

// Pre-build GEMMs for a specific seq_len
void engine_prepare_seq(int64_t seq) {
    get_gemms((int)seq);
}

void engine_prepare_ca(int64_t ca_len) {
    get_ca_gemms((int)ca_len);
}

static void update_actual_ctx_and_gemms(int actual_ctx) {
    if (actual_ctx <= 0) actual_ctx = g_ctx;
    TORCH_CHECK(actual_ctx <= g_ctx, "actual ctx ", actual_ctx, " > max ctx ", g_ctx);
    if (actual_ctx == g_actual_ctx) {
        return;
    }
    g_actual_ctx = actual_ctx;
    g_ck.setup(g_ltH, actual_ctx, g_D_per_rank, g_CA_K_DIM, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, 1);
    g_cv.setup(g_ltH, actual_ctx, g_D_per_rank, g_CA_K_DIM, g_scaleA, g_scaleB, g_gemm_ws, 32*1024*1024, 1);
    g_ck_bf16.setup(g_ltH, actual_ctx, g_D_per_rank, g_CA_K_DIM, g_gemm_ws, 32*1024*1024, 1);
    g_cv_bf16.setup(g_ltH, actual_ctx, g_D_per_rank, g_CA_K_DIM, g_gemm_ws, 32*1024*1024, 1);
    g_ck_bf16_built = true;
    g_ca_gemms_by_len.clear();
    g_ca_kv_cache_valid = false;
}

void engine_build_ca_kv_cache(torch::Tensor enc_t) {
    TORCH_CHECK(g_ready, "engine not initialized");
    TORCH_CHECK(enc_t.is_cuda(), "enc_t must be CUDA");
    TORCH_CHECK(enc_t.scalar_type() == torch::kBFloat16, "enc_t must be bfloat16");
    TORCH_CHECK(enc_t.dim() == 2, "enc_t must have shape [ctx, D]");
    TORCH_CHECK((int)enc_t.size(1) == g_CA_K_DIM, "enc dim mismatch");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int ctx = (int)enc_t.size(0);
    int Dpr = g_D_per_rank;
    bool TP = (g_tp_world > 1);
    update_actual_ctx_and_gemms(ctx);

    auto* enc_ptr = (__nv_bfloat16*)enc_t.data_ptr();
    if (!g_force_bf16_gemms) {
        k_bf16_to_fp8<<<g_nsm, 256, 0, stream>>>(b_enc_fp8, enc_ptr, ctx * g_CA_K_DIM);
    }

    for (int li = 0; li < g_NL; li++) {
        auto& L = g_layers[li];
        size_t ca_kv_slot = (size_t)li * (size_t)g_ctx * (size_t)Dpr;
        size_t bytes = (size_t)ctx * Dpr * sizeof(__nv_bfloat16);

        if (g_force_bf16_gemms) {
            g_ck_bf16.run(g_ltH, L.ck_w16, enc_ptr, b_ca_k, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.ckb);
            g_cv_bf16.run(g_ltH, L.cv_w16, enc_ptr, b_ca_v, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cvb);
        } else {
            g_ck.run(g_ltH, L.ck_w8, b_enc_fp8, b_ca_k, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.ckb);
            g_cv.run(g_ltH, L.cv_w8, b_enc_fp8, b_ca_v, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cvb);
        }

        if (!TP) {
            k_rmsnorm<<<ctx, 256, 0, stream>>>(b_ca_k, L.ca_rms_k, Dpr);
        } else {
            k_partial_sumsq<<<ctx, 256, 0, stream>>>(b_ca_k, b_tp_sumsq_ck, Dpr);
            CKNCCL(ncclAllReduce(b_tp_sumsq_ck, b_tp_sumsq_ck, (size_t)ctx, ncclFloat32, ncclSum, g_nccl_comm, stream));
            size_t smem_rr = (size_t)Dpr * sizeof(__nv_bfloat16);
            k_apply_rmsnorm_rope<<<ctx, 256, smem_rr, stream>>>(b_ca_k, L.ca_rms_k, nullptr, b_tp_sumsq_ck, g_D, g_NH_per_rank, g_HD);
        }

        CK(cudaMemcpyAsync(b_ca_k_cache + ca_kv_slot, b_ca_k, bytes, cudaMemcpyDeviceToDevice, stream));
        CK(cudaMemcpyAsync(b_ca_v_cache + ca_kv_slot, b_ca_v, bytes, cudaMemcpyDeviceToDevice, stream));
    }

    g_ca_kv_cache_valid = true;
}

// ============================================================
// Load one layer's weights (BF16 -> FP8 conversion)
// ============================================================
void engine_load(int64_t li,
    torch::Tensor sq, torch::Tensor sk, torch::Tensor sv, torch::Tensor so,
    torch::Tensor cq, torch::Tensor ck, torch::Tensor cv, torch::Tensor co,
    torch::Tensor f1, torch::Tensor f2,
    torch::Tensor sqb, torch::Tensor skb, torch::Tensor svb, torch::Tensor sob,
    torch::Tensor cqb, torch::Tensor ckb, torch::Tensor cvb, torch::Tensor cob,
    torch::Tensor f1b, torch::Tensor f2b,
    torch::Tensor rq, torch::Tensor rk, torch::Tensor crq, torch::Tensor crk,
    torch::Tensor n2w, torch::Tensor n2b, torch::Tensor sst) {
    auto& L = g_layers[li];

    // In force_bf16 mode, skip FP8 weights entirely (saves ~50% weight memory)
    if (!g_force_bf16_gemms) {
        auto conv = [](torch::Tensor w, void** dst) {
            auto fp8 = w.to(torch::kFloat8_e4m3fn).contiguous();
            CK(cudaMalloc(dst, fp8.nbytes()));
            CK(cudaMemcpy(*dst, fp8.data_ptr(), fp8.nbytes(), cudaMemcpyDeviceToDevice));
        };
        conv(sq, &L.sq_w8); conv(sk, &L.sk_w8); conv(sv, &L.sv_w8); conv(so, &L.so_w8);
        conv(cq, &L.cq_w8); conv(ck, &L.ck_w8); conv(cv, &L.cv_w8); conv(co, &L.co_w8);
        conv(f1, &L.f1_w8); conv(f2, &L.f2_w8);
    }

    if (g_fp4_ffn2) {
        auto f2_bf16 = f2.to(torch::kBFloat16).contiguous();
        TORCH_CHECK(f2_bf16.is_cuda(), "FP4 FFN2 expects CUDA weights");
        auto [packed, _scales, swizzled] = quantize_fp4_rowwise_bf16(f2_bf16);
        CK(cudaMalloc((void**)&L.f2_w4, packed.nbytes()));
        CK(cudaMalloc((void**)&L.f2_s4, swizzled.nbytes()));
        CK(cudaMemcpy(L.f2_w4, packed.data_ptr(), packed.nbytes(), cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(L.f2_s4, swizzled.data_ptr(), swizzled.nbytes(), cudaMemcpyDeviceToDevice));
    }

    // BF16 weight copies for force_bf16_gemms mode (CA K/V now pre-computed in Python)
    if (g_force_bf16_gemms) {
        auto keep_bf16 = [](torch::Tensor w, __nv_bfloat16** dst) {
            auto tc = w.to(torch::kBFloat16).contiguous();
            size_t nbytes = tc.numel() * sizeof(__nv_bfloat16);
            CK(cudaMalloc((void**)dst, nbytes));
            CK(cudaMemcpy(*dst, tc.data_ptr(), nbytes, cudaMemcpyDeviceToDevice));
        };
        keep_bf16(sq, &L.sq_w16); keep_bf16(sk, &L.sk_w16);
        keep_bf16(sv, &L.sv_w16); keep_bf16(so, &L.so_w16);
        keep_bf16(cq, &L.cq_w16); keep_bf16(ck, &L.ck_w16);
        keep_bf16(cv, &L.cv_w16); keep_bf16(co, &L.co_w16);
        keep_bf16(f1, &L.f1_w16); keep_bf16(f2, &L.f2_w16);
    }

    // Persist biases
    auto persist_bf16 = [](torch::Tensor t, __nv_bfloat16** dst) {
        auto tc = t.contiguous();
        size_t nbytes = tc.numel() * sizeof(__nv_bfloat16);
        CK(cudaMalloc((void**)dst, nbytes));
        CK(cudaMemcpy(*dst, tc.data_ptr(), nbytes, cudaMemcpyDeviceToDevice));
    };
    persist_bf16(sqb, &L.sqb); persist_bf16(skb, &L.skb);
    persist_bf16(svb, &L.svb); persist_bf16(sob, &L.sob);
    persist_bf16(cqb, &L.cqb); persist_bf16(ckb, &L.ckb);
    persist_bf16(cvb, &L.cvb); persist_bf16(cob, &L.cob);
    persist_bf16(f1b, &L.f1b); persist_bf16(f2b, &L.f2b);

    // Persist norm weights
    auto persist_f32 = [](torch::Tensor t, float** dst) {
        TORCH_CHECK(t.scalar_type() == torch::kFloat32, "persist_f32: expected float32");
        auto tc = t.contiguous();
        size_t n = tc.numel();
        CK(cudaMalloc(dst, n * sizeof(float)));
        CK(cudaMemcpy(*dst, tc.data_ptr<float>(), n * sizeof(float), cudaMemcpyDeviceToDevice));
    };
    persist_f32(rq, &L.rms_q); persist_f32(rk, &L.rms_k);
    persist_f32(crq, &L.ca_rms_q); persist_f32(crk, &L.ca_rms_k);
    L.has_norm2 = (n2w.numel() > 0);
    if (L.has_norm2) { persist_f32(n2w, &L.n2w); persist_f32(n2b, &L.n2b); }
    else { L.n2w = nullptr; L.n2b = nullptr; }
    persist_f32(sst, &L.sst);
    L.ffn_dim = f1.size(0);
    L.ca_k_dim = ck.size(1);
}

// Load GDN-specific weights for one linear attention layer.
void engine_load_gdn(int64_t li,
    torch::Tensor q_w, torch::Tensor k_w, torch::Tensor v_w,
    torch::Tensor a_w, torch::Tensor b_w,
    torch::Tensor g_w, torch::Tensor o_w,
    torch::Tensor q_b, torch::Tensor k_b, torch::Tensor v_b,
    torch::Tensor a_b, torch::Tensor b_b,
    torch::Tensor g_b, torch::Tensor o_b,
    torch::Tensor conv_q_w, torch::Tensor conv_k_w, torch::Tensor conv_v_w,
    torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor o_norm_w) {
    auto& L = g_layers[li];
    L.is_gdn = true;
    L.gdn_key_dim = g_gdn_key_dim / g_tp_world;
    L.gdn_value_dim = g_gdn_value_dim / g_tp_world;
    L.gdn_head_k = g_gdn_head_k;
    L.gdn_head_v = g_gdn_head_v;

    if (!g_force_bf16_gemms) {
        auto conv = [](torch::Tensor w, void** dst) {
            auto fp8 = w.to(torch::kFloat8_e4m3fn).contiguous();
            CK(cudaMalloc(dst, fp8.nbytes()));
            CK(cudaMemcpy(*dst, fp8.data_ptr(), fp8.nbytes(), cudaMemcpyDeviceToDevice));
        };
        conv(q_w, &L.gdn_q_w8); conv(k_w, &L.gdn_k_w8); conv(v_w, &L.gdn_v_w8);
        conv(a_w, &L.gdn_a_w8); conv(b_w, &L.gdn_b_w8);
        conv(g_w, &L.gdn_g_w8); conv(o_w, &L.gdn_o_w8);
    }

    // Keep BF16 weight copies for force_bf16_gemms mode (q/k/v/g/o only — a/b already have bf16)
    if (g_force_bf16_gemms) {
        auto keep_bf16 = [](torch::Tensor w, __nv_bfloat16** dst) {
            auto tc = w.to(torch::kBFloat16).contiguous();
            size_t nbytes = tc.numel() * sizeof(__nv_bfloat16);
            CK(cudaMalloc((void**)dst, nbytes));
            CK(cudaMemcpy(*dst, tc.data_ptr(), nbytes, cudaMemcpyDeviceToDevice));
        };
        keep_bf16(q_w, &L.gdn_q_w16); keep_bf16(k_w, &L.gdn_k_w16);
        keep_bf16(v_w, &L.gdn_v_w16); keep_bf16(g_w, &L.gdn_g_w16);
        keep_bf16(o_w, &L.gdn_o_w16);
    }

    auto persist_bf16 = [](torch::Tensor t, __nv_bfloat16** dst) {
        if (t.numel() == 0) { *dst = nullptr; return; }
        auto tc = t.contiguous();
        size_t nbytes = tc.numel() * sizeof(__nv_bfloat16);
        CK(cudaMalloc((void**)dst, nbytes));
        CK(cudaMemcpy(*dst, tc.data_ptr(), nbytes, cudaMemcpyDeviceToDevice));
    };
    persist_bf16(q_b, &L.gdn_qb); persist_bf16(k_b, &L.gdn_kb); persist_bf16(v_b, &L.gdn_vb);
    persist_bf16(a_b, &L.gdn_ab); persist_bf16(b_b, &L.gdn_bb);
    persist_bf16(g_b, &L.gdn_gb); persist_bf16(o_b, &L.gdn_ob);
    // BF16 copies of a/b weights for BF16Gemm path (FP8 doesn't support N=20)
    persist_bf16(a_w.to(torch::kBFloat16).contiguous(), &L.gdn_a_w_bf16);
    persist_bf16(b_w.to(torch::kBFloat16).contiguous(), &L.gdn_b_w_bf16);

    auto persist_f32 = [](torch::Tensor t, float** dst) {
        auto tc = t.contiguous();
        CK(cudaMalloc(dst, tc.numel() * sizeof(float)));
        CK(cudaMemcpy(*dst, tc.data_ptr<float>(), tc.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
    };
    // Conv weights: reshape from [C, 1, K] to [C, K]
    persist_f32(conv_q_w.squeeze(1).to(torch::kFloat32), &L.gdn_conv_q_w);
    persist_f32(conv_k_w.squeeze(1).to(torch::kFloat32), &L.gdn_conv_k_w);
    persist_f32(conv_v_w.squeeze(1).to(torch::kFloat32), &L.gdn_conv_v_w);
    persist_f32(A_log.to(torch::kFloat32), &L.gdn_A_log);
    persist_f32(dt_bias.to(torch::kFloat32), &L.gdn_dt_bias);
    persist_f32(o_norm_w.to(torch::kFloat32), &L.gdn_o_norm_w);
}

void engine_reset_gdn_state() {
    if (b_gdn_state && g_n_gdn_layers > 0) {
        int NHpr = g_NH_per_rank;
        size_t total = (size_t)g_n_gdn_layers * NHpr * g_gdn_head_k * g_gdn_head_v * sizeof(float);
        CK(cudaMemset(b_gdn_state, 0, total));
    }
}

// Write pre-computed CA K/V into the cache for one layer
// This lets Python pre-compute CA K/V using stock PyTorch ops
// and inject them, bypassing the engine's cuBLASLt GEMM + RMSNorm.
void engine_set_ca_kv_cache(int64_t layer, torch::Tensor k_t, torch::Tensor v_t) {
    TORCH_CHECK(g_ready, "engine not initialized");
    int Dpr = g_D_per_rank;
    int ctx_rows = (int)k_t.size(0);
    TORCH_CHECK(ctx_rows <= g_ctx, "ctx_rows > g_ctx");
    TORCH_CHECK(k_t.size(1) == Dpr, "K dim mismatch");
    TORCH_CHECK(v_t.size(0) == ctx_rows && v_t.size(1) == Dpr, "V shape mismatch");
    size_t slot = (size_t)layer * (size_t)g_ctx * (size_t)Dpr;
    size_t bytes = (size_t)ctx_rows * Dpr * sizeof(__nv_bfloat16);
    CK(cudaMemcpy(b_ca_k_cache + slot, k_t.data_ptr(), bytes, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(b_ca_v_cache + slot, v_t.data_ptr(), bytes, cudaMemcpyDeviceToDevice));
}

void engine_set_ca_kv_cache_valid(bool v, int64_t actual_ctx) {
    g_ca_kv_cache_valid = v;
    if (actual_ctx > 0 && (int)actual_ctx != g_actual_ctx) {
        g_actual_ctx = (int)actual_ctx;
        // Must rebuild CA SDPA for new K/V length
        g_ca_gemms_by_len.clear();
    }
}

// Invalidate CA K/V cache
void engine_invalidate_ca_kv_cache() {
    g_ca_kv_cache_valid = false;
}

// Toggle cuBLASLt chunk GDN (true) vs fused recurrent (false)
void engine_set_gdn_cublas_chunk(bool enable) {
    g_gdn_use_cublas_chunk = enable;
    printf("  [GDN] cuBLASLt chunk mode: %s\n", enable ? "ON" : "OFF");
}

// ============================================================
// Forward core: runs layers [start_layer, end_layer)
// Handles both quadratic attention and GatedDeltaNet layers.
// ============================================================
static void forward_core(
    __nv_bfloat16* x, __nv_bfloat16* enc,
    float* temb_base, float* ssts_base, float* rope_freqs,
    int seq, int ca_len, int temb_row_stride,
    int start_layer, int end_layer,
    cudaStream_t stream) {
    int D = g_D, ctx = g_actual_ctx > 0 ? g_actual_ctx : g_ctx;
    int NH = g_NH, HD = g_HD, FFN = g_FFN;
    int Dpr = g_D_per_rank, NHpr = g_NH_per_rank, FFNpr = g_FFN_per_rank;
    int seq_D = seq * D;
    int seq_Dpr = seq * Dpr;
    int seq_FFNpr = seq * FFNpr;
    const bool TP = (g_tp_world > 1);

    // Get GEMMs/SDPA (creates if not cached)
    SeqGemms* gm = get_gemms(seq);
    CaGemms* cg = get_ca_gemms(ca_len);
    int ca_len_Dpr = ca_len * Dpr;

    // Convert encoder to FP8 once (skip in force_bf16 mode — use enc directly)
    if (DO_EWISE && !g_force_bf16_gemms && !g_ca_kv_cache_valid) {
        k_bf16_to_fp8<<<g_nsm, 256, 0, stream>>>(b_enc_fp8, enc, ctx * g_CA_K_DIM);
    }

    size_t ssts_layer_stride = (size_t)6 * D;

    // Kairos modulation order: [scale_sa, shift_sa, gate_sa, scale_ffn, shift_ffn, gate_ffn]
    // temb layout per row follows the same order.

    // GatedDeltaNet layer indices (every 4th starting at 3)
    // These layers are skipped by the engine and handled in Python.
    auto is_gdn_layer = [](int li) -> bool {
        return (li % 4 == 3);
    };

    // Optional per-phase timing
    const char* time_env = getenv("ENGINE_TIME");
    const bool DO_TIME = (time_env && atoi(time_env) == 1);
    const int NPHASES = 11;  // SA_LN, SA_QKV, SA_SDPA, SA_O, CA_LN, CA_QKV, CA_SDPA, CA_O, FFN_LN, FFN1, FFN2
    std::vector<cudaEvent_t> evs;
    int n_quadratic = 0;
    if (DO_TIME) {
        for (int li = start_layer; li < end_layer; li++) {
            if (!is_gdn_layer(li)) n_quadratic++;
        }
        evs.resize((NPHASES + 1) * n_quadratic);
        for (auto& e : evs) cudaEventCreate(&e);
    }
    int quad_idx = 0;
    auto TREC = [&](int idx) {
        if (DO_TIME) cudaEventRecord(evs[quad_idx * (NPHASES + 1) + idx], stream);
    };

    bool sa_ln_fused_from_prev = false;

    for (int li = start_layer; li < end_layer; li++) {
        const auto& L = g_layers[li];
        float* ssts_L = ssts_base + li * ssts_layer_stride;

        // Kairos modulation order: [scale_sa, shift_sa, gate_sa, scale_ffn, shift_ffn, gate_ffn]
        float* ssts_scale_sa  = ssts_L + 0 * D;
        float* ssts_shift_sa  = ssts_L + 1 * D;
        float* ssts_gate_sa   = ssts_L + 2 * D;
        float* ssts_scale_ffn = ssts_L + 3 * D;
        float* ssts_shift_ffn = ssts_L + 4 * D;
        float* ssts_gate_ffn  = ssts_L + 5 * D;

        // temb layout per row: [6, D] matching modulation order
        float* temb_scale_sa  = temb_base + 0 * D;
        float* temb_shift_sa  = temb_base + 1 * D;
        float* temb_gate_sa   = temb_base + 2 * D;
        float* temb_scale_ffn = temb_base + 3 * D;
        float* temb_shift_ffn = temb_base + 4 * D;
        float* temb_gate_ffn  = temb_base + 5 * D;

        if (is_gdn_layer(li) && L.is_gdn) {
            // ---- GatedDeltaNet path ----
            // SA LN + AdaLN — write BOTH bf16 (for a/b BF16 GEMMs) and fp8 (for FP8 GEMMs)
            if (DO_EWISE && !sa_ln_fused_from_prev) {
                k_ln_adaln_ssts<<<seq, 256, (size_t)D*sizeof(__nv_bfloat16), stream>>>(x, b_norm, b_norm_fp8,
                    ssts_scale_sa, ssts_shift_sa, temb_scale_sa, temb_shift_sa, D, temb_row_stride);
            }
            sa_ln_fused_from_prev = false;

            int KDpr = L.gdn_key_dim, VDpr = L.gdn_value_dim;
            int NHpr = g_NH_per_rank;
            int hk = L.gdn_head_k, hv = L.gdn_head_v;

            // Projections: q/k/v/g use FP8, a/b use BF16 (N=20 too small for FP8)
            if (DO_GEMM) {
                if (g_force_bf16_gemms) {
                    gm->gdn_q_bf16.run(g_ltH, L.gdn_q_w16, b_norm, b_gdn_q, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.gdn_qb);
                    gm->gdn_k_bf16.run(g_ltH, L.gdn_k_w16, b_norm, b_gdn_k, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.gdn_kb);
                    gm->gdn_v_bf16.run(g_ltH, L.gdn_v_w16, b_norm, b_gdn_v, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.gdn_vb);
                    gm->gdn_g_bf16.run(g_ltH, L.gdn_g_w16, b_norm, b_gdn_g, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.gdn_gb);
                } else {
                    gm->gdn_q.run(g_ltH, L.gdn_q_w8, b_norm_fp8, b_gdn_q, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.gdn_qb);
                    gm->gdn_k.run(g_ltH, L.gdn_k_w8, b_norm_fp8, b_gdn_k, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.gdn_kb);
                    gm->gdn_v.run(g_ltH, L.gdn_v_w8, b_norm_fp8, b_gdn_v, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.gdn_vb);
                    gm->gdn_g.run(g_ltH, L.gdn_g_w8, b_norm_fp8, b_gdn_g, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.gdn_gb);
                }
                gm->gdn_a.run(g_ltH, L.gdn_a_w_bf16, b_norm, b_gdn_a, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.gdn_ab);
                gm->gdn_b.run(g_ltH, L.gdn_b_w_bf16, b_norm, b_gdn_b, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.gdn_bb);
            }

            // Causal depthwise conv + SiLU on Q, K, V
            if (DO_EWISE) {
                int conv_k = g_gdn_conv_k;
                int blk = (KDpr * seq + 255) / 256;
                k_causal_dw_conv_silu<<<blk, 256, 0, stream>>>(b_gdn_q, b_gdn_conv_scratch, L.gdn_conv_q_w, seq, KDpr, conv_k);
                CK(cudaMemcpyAsync(b_gdn_q, b_gdn_conv_scratch, (size_t)seq * KDpr * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
                k_causal_dw_conv_silu<<<blk, 256, 0, stream>>>(b_gdn_k, b_gdn_conv_scratch, L.gdn_conv_k_w, seq, KDpr, conv_k);
                CK(cudaMemcpyAsync(b_gdn_k, b_gdn_conv_scratch, (size_t)seq * KDpr * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
                int blk_v = (VDpr * seq + 255) / 256;
                k_causal_dw_conv_silu<<<blk_v, 256, 0, stream>>>(b_gdn_v, b_gdn_conv_scratch, L.gdn_conv_v_w, seq, VDpr, conv_k);
                CK(cudaMemcpyAsync(b_gdn_v, b_gdn_conv_scratch, (size_t)seq * VDpr * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            }

            // Compute gates: beta and g
            if (DO_EWISE) {
                int blk_g = (seq * NHpr + 255) / 256;
                k_gdn_compute_gates<<<blk_g, 256, 0, stream>>>(
                    b_gdn_a, b_gdn_b, L.gdn_A_log, L.gdn_dt_bias,
                    b_gdn_gates_g, b_gdn_gates_beta, seq, NHpr);
            }

            // GDN forward: cuBLASLt chunk or fused recurrent
            // Q, K are [seq, KDpr] = [seq, NHpr*hk], need [seq, NHpr, hk] view (already contiguous)
            // V is [seq, VDpr] = [seq, NHpr*hv], need [seq, NHpr, hv] view
            if (DO_EWISE) {
                // Find GDN layer index for state offset
                int gdn_idx = 0;
                for (int gi = 0; gi < (int)g_gdn_layer_indices.size(); gi++) {
                    if (g_gdn_layer_indices[gi] == li) { gdn_idx = gi; break; }
                }
                size_t state_per_layer = (size_t)NHpr * hk * hv;
                float* layer_state = b_gdn_state + gdn_idx * state_per_layer;
                float sc = 1.f / sqrtf((float)hk);

                if (g_gdn_use_cublas_chunk && b_gdn_kkt_bf16 != nullptr) {
                    // cuBLASLt chunk GDN: phases 1-4
                    // Q/K L2-norm is handled inside the chunk kernels
                    // (prepare kernel normalizes K, state kernel normalizes Q)
                    // But we still need to L2-norm Q for the state kernel (it reads raw Q)
                    // The k_gdn_chunk_state_output kernel normalizes Q inline, so no pre-norm needed.
                    gdn_cublas_chunk_forward(
                        b_gdn_q, b_gdn_k, b_gdn_v,
                        b_gdn_gates_g, b_gdn_gates_beta,
                        b_gdn_out, layer_state,
                        b_gdn_chunk_w, b_gdn_chunk_u, b_gdn_chunk_gcum,
                        seq, NHpr, hk, hv, sc, stream);
                } else {
                    // Legacy: pre-normalize Q and K (L2 norm + scale)
                    k_gdn_l2norm_scale<<<dim3(seq, NHpr), 256, 0, stream>>>(
                        b_gdn_q, b_gdn_k, hk, sc);

                    // Persistent GDN recurrent — Triton-style work stealing
                    int NV_blocks = hv / GDN_BV;
                    int total_tiles = NHpr * NV_blocks;
                    int zero = 0;
                    cudaMemcpyToSymbolAsync(g_gdn_work_counter, &zero, sizeof(int), 0,
                                            cudaMemcpyHostToDevice, stream);
                    int n_persistent = std::min(total_tiles, g_nsm * 8);
                    k_gdn_recurrent<<<n_persistent, 32, 0, stream>>>(
                        b_gdn_q, b_gdn_k, b_gdn_v,
                        b_gdn_gates_g, b_gdn_gates_beta,
                        b_gdn_out, layer_state,
                        seq, NHpr, hk, hv, sc, total_tiles);
                }
            }

            // RMSNorm + SiLU gating
            if (DO_EWISE) {
                k_gdn_rmsnorm_silu_gate<<<seq * NHpr, 256, 0, stream>>>(
                    b_gdn_out, b_gdn_g, L.gdn_o_norm_w,
                    b_gdn_out, seq, NHpr, hv);
            }

            // Output projection (row-parallel)
            if (g_force_bf16_gemms) {
                if (DO_GEMM) gm->gdn_o_bf16.run(g_ltH, L.gdn_o_w16, b_gdn_out, b_sa_out, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.gdn_ob);
            } else {
                // BF16->FP8 for output projection
                if (DO_EWISE) k_bf16_to_fp8<<<8*g_nsm, 256, 0, stream>>>(b_gdn_out_fp8, b_gdn_out, seq * VDpr);
                if (DO_GEMM) gm->gdn_o.run(g_ltH, L.gdn_o_w8, b_gdn_out_fp8, b_sa_out, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.gdn_ob);
            }
            if (TP) {
                CKNCCL(ncclAllReduce(b_sa_out, b_sa_out, (size_t)seq_D, ncclBfloat16, ncclSum, g_nccl_comm, stream));
            }

            // Gate residual (same as quadratic)
            if (DO_EWISE) k_gate_res_ssts<<<8*g_nsm, 256, 0, stream>>>(x, b_sa_out, ssts_gate_sa, temb_gate_sa, seq, D, temb_row_stride);

            // ---- CA + FFN (identical to quadratic path) ----
          if (!g_skip_ca) {
            // CA LN (affine)
            if (DO_EWISE) {
                if (L.has_norm2) {
                    k_ln_affine<<<ca_len, 256, 0, stream>>>(x, b_ca_norm, b_ca_norm_fp8, L.n2w, L.n2b, D);
                } else if (!g_force_bf16_gemms) {
                    k_bf16_to_fp8<<<8*g_nsm, 256, 0, stream>>>(b_ca_norm_fp8, x, ca_len * D);
                }
            }
            // CA Q/K/V
            if (DO_GEMM) {
                if (g_force_bf16_gemms) {
                    __nv_bfloat16* ca_norm_bf16 = L.has_norm2 ? b_ca_norm : x;
                    cg->cq_bf16.run(g_ltH, L.cq_w16, ca_norm_bf16, b_ca_q, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cqb);
                } else {
                    cg->cq.run(g_ltH, L.cq_w8, b_ca_norm_fp8, b_ca_q, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cqb);
                }
            }
            __nv_bfloat16* ca_k_ptr;
            __nv_bfloat16* ca_v_ptr;
            size_t ca_kv_slot = (size_t)li * (size_t)g_ctx * (size_t)Dpr;  // use g_ctx for consistent cache slot
            if (g_ca_kv_cache_valid) {
                ca_k_ptr = b_ca_k_cache + ca_kv_slot;
                ca_v_ptr = b_ca_v_cache + ca_kv_slot;
            } else {
                if (DO_GEMM) {
                    if (g_force_bf16_gemms) {
                        g_ck_bf16.run(g_ltH, L.ck_w16, enc, b_ca_k, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.ckb);
                        g_cv_bf16.run(g_ltH, L.cv_w16, enc, b_ca_v, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cvb);
                    } else {
                        g_ck.run(g_ltH, L.ck_w8, b_enc_fp8, b_ca_k, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.ckb);
                        g_cv.run(g_ltH, L.cv_w8, b_enc_fp8, b_ca_v, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cvb);
                    }
                }
                ca_k_ptr = b_ca_k;
                ca_v_ptr = b_ca_v;
            }
            // CA RMSNorm
            if (DO_EWISE) {
                if (!TP) {
                    k_rmsnorm<<<ca_len, 256, 0, stream>>>(b_ca_q, L.ca_rms_q, Dpr);
                    if (!g_ca_kv_cache_valid) k_rmsnorm<<<ctx, 256, 0, stream>>>(b_ca_k, L.ca_rms_k, Dpr);
                } else {
                    k_partial_sumsq<<<ca_len, 256, 0, stream>>>(b_ca_q, b_tp_sumsq_cq, Dpr);
                    if (!g_ca_kv_cache_valid) {
                        k_partial_sumsq<<<ctx, 256, 0, stream>>>(b_ca_k, b_tp_sumsq_ck, Dpr);
                        ncclGroupStart();
                        CKNCCL(ncclAllReduce(b_tp_sumsq_cq, b_tp_sumsq_cq, (size_t)ca_len, ncclFloat32, ncclSum, g_nccl_comm, stream));
                        CKNCCL(ncclAllReduce(b_tp_sumsq_ck, b_tp_sumsq_ck, (size_t)ctx, ncclFloat32, ncclSum, g_nccl_comm, stream));
                        ncclGroupEnd();
                    } else {
                        CKNCCL(ncclAllReduce(b_tp_sumsq_cq, b_tp_sumsq_cq, (size_t)ca_len, ncclFloat32, ncclSum, g_nccl_comm, stream));
                    }
                    size_t smem_rr = (size_t)Dpr * sizeof(__nv_bfloat16);
                    k_apply_rmsnorm_rope<<<ca_len, 256, smem_rr, stream>>>(b_ca_q, L.ca_rms_q, nullptr, b_tp_sumsq_cq, D, NHpr, HD);
                    if (!g_ca_kv_cache_valid)
                        k_apply_rmsnorm_rope<<<ctx, 256, smem_rr, stream>>>(b_ca_k, L.ca_rms_k, nullptr, b_tp_sumsq_ck, D, NHpr, HD);
                }
                if (!g_ca_kv_cache_valid) {
                    CK(cudaMemcpyAsync(b_ca_k_cache + ca_kv_slot, b_ca_k, (size_t)ctx * Dpr * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
                    CK(cudaMemcpyAsync(b_ca_v_cache + ca_kv_slot, b_ca_v, (size_t)ctx * Dpr * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
                }
            }
            // CA SDPA
            if (DO_SDPA) {
                if (li == 0 && start_layer == 0) {
                }
                if (g_use_torch_sdpa) {
                    float ca_scale = 1.0f / sqrtf((float)HD);
                    torch_sdpa_bf16(b_ca_q, ca_k_ptr, ca_v_ptr, b_ca_out, ca_len, ctx, NHpr, HD, ca_scale, stream);
                } else {
                    cg->ca_sdpa.run(g_cudnnH, b_ca_q, ca_k_ptr, ca_v_ptr, b_ca_out, stream);
                }
            }
            // CA_O (ungated residual)
            if (g_force_bf16_gemms) {
                if (!TP) {
                    if (DO_GEMM) cg->co_bf16.run(g_ltH, L.co_w16, b_ca_out, x, g_gemm_ws, 32*1024*1024, 1.f, 1.f, stream, L.cob);
                } else {
                    if (DO_GEMM) cg->co_bf16.run(g_ltH, L.co_w16, b_ca_out, b_sa_out, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cob);
                    CKNCCL(ncclAllReduce(b_sa_out, b_sa_out, (size_t)ca_len * D, ncclBfloat16, ncclSum, g_nccl_comm, stream));
                    if (DO_EWISE) k_add_vec<<<8*g_nsm, 256, 0, stream>>>(x, b_sa_out, ca_len * D);
                }
            } else {
                if (DO_EWISE) k_bf16_to_fp8<<<8*g_nsm, 256, 0, stream>>>(b_ca_out_fp8, b_ca_out, ca_len_Dpr);
                if (!TP) {
                    if (DO_GEMM) cg->co.run(g_ltH, L.co_w8, b_ca_out_fp8, x, g_gemm_ws, 32*1024*1024, 1.f, 1.f, stream, L.cob);
                } else {
                    if (DO_GEMM) cg->co.run(g_ltH, L.co_w8, b_ca_out_fp8, b_sa_out, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cob);
                    CKNCCL(ncclAllReduce(b_sa_out, b_sa_out, (size_t)ca_len * D, ncclBfloat16, ncclSum, g_nccl_comm, stream));
                    if (DO_EWISE) k_add_vec<<<8*g_nsm, 256, 0, stream>>>(x, b_sa_out, ca_len * D);
                }
            }
          } // end if (!g_skip_ca) — GDN CA path
            // FFN LN
            if (DO_EWISE) {
                k_ln_adaln_ssts<<<seq, 256, (size_t)D*sizeof(__nv_bfloat16), stream>>>(x, b_ffn_norm, b_ffn_norm_fp8,
                    ssts_scale_ffn, ssts_shift_ffn, temb_scale_ffn, temb_shift_ffn, D, temb_row_stride);
            }
            // FFN1 + SiLU
            if (g_force_bf16_gemms) {
                if (DO_GEMM) gm->f1_bf16.run(g_ltH, L.f1_w16, b_ffn_norm, b_ffn_mid, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.f1b);
                if (DO_EWISE) k_silu_bf16<<<(seq_FFNpr + 255) / 256, 256, 0, stream>>>(b_ffn_mid, seq_FFNpr);
            } else {
                bool use_fp4_ffn2 = g_fp4_ffn2 && DO_GEMM && DO_EWISE;
                bool use_fused_ffn1_fp4 = use_fp4_ffn2 && g_cutlass_fused_ffn1_fp4;
                bool use_fused_ffn1 = g_cutlass_fused_ffn1 && DO_GEMM && DO_EWISE && !use_fused_ffn1_fp4;
                if (use_fused_ffn1_fp4) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
                    gm->f1_fused_fp4.run(
                        L.f1_w8,
                        b_ffn_norm_fp8,
                        b_ffn_mid,
                        b_ffn_mid_fp4,
                        b_ffn_mid_fp4_scales_swizzled,
                        L.f1b,
                        g_fp4_normconst,
                        stream);
#else
                    TORCH_CHECK(false, "CUTLASS fused FFN1 FP4 requested without SM120 CUTLASS support");
#endif
                } else if (use_fused_ffn1) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
                    gm->f1_fused.run(L.f1_w8, b_ffn_norm_fp8, b_ffn_mid_fp8, L.f1b, stream);
#else
                    TORCH_CHECK(false, "CUTLASS fused FFN1 requested without SM120 CUTLASS support");
#endif
                } else {
                    if (DO_GEMM) gm->f1.run(g_ltH, L.f1_w8, b_ffn_norm_fp8, b_ffn_mid, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.f1b);
                    if (DO_EWISE) {
                        if (g_fp4_ffn2 && DO_GEMM) {
                            int rows_padded = fp4_scale_rows(seq);
                            int scale_cols = fp4_scale_cols(g_FFN_per_rank);
                            size_t packed_bytes = (size_t)rows_padded * (size_t)g_FFN_per_rank / 2;
                            size_t scale_bytes = (size_t)rows_padded * (size_t)scale_cols;
                            if (rows_padded != seq) {
                                CK(cudaMemsetAsync(b_ffn_mid_fp4, 0, packed_bytes, stream));
                            }
                            if (rows_padded != seq || scale_cols != (g_FFN_per_rank / 16)) {
                                CK(cudaMemsetAsync(b_ffn_mid_fp4_scales_swizzled, fp4_scale_one_byte(), scale_bytes, stream));
                            }
                            constexpr int FP4_WARPS_PER_BLOCK = 4;
                            constexpr int FP4_THREADS = FP4_WARPS_PER_BLOCK * 32;
                            int fp4_grid_x = (g_FFN_per_rank + (FP4_WARPS_PER_BLOCK * 32 - 1)) / (FP4_WARPS_PER_BLOCK * 32);
                            k_silu_to_fp4_rowwise_swizzled<<<dim3(fp4_grid_x, seq), FP4_THREADS, 0, stream>>>(
                                b_ffn_mid,
                                b_ffn_mid_fp4,
                                b_ffn_mid_fp4_scales_swizzled,
                                seq,
                                g_FFN_per_rank,
                                scale_cols);
                        } else {
                            k_silu_to_fp8<<<8*g_nsm, 256, 0, stream>>>(b_ffn_mid, b_ffn_mid_fp8, seq_FFNpr);
                        }
                    }
                }
            }
            // FFN2
            if (g_force_bf16_gemms) {
                if (DO_GEMM) gm->f2_bf16.run(g_ltH, L.f2_w16, b_ffn_mid, b_ffn_out, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.f2b);
            } else {
                bool use_fp4_ffn2 = g_fp4_ffn2 && DO_GEMM && DO_EWISE;
                bool use_fused_ffn2 = g_cutlass_fused_ffn2 && DO_GEMM;
                if (use_fp4_ffn2) {
                    run_ffn2_fp4(L, gm, seq, stream);
                } else if (use_fused_ffn2) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
                    gm->f2_fused.run(L.f2_w8, b_ffn_mid_fp8, b_ffn_out, L.f2b, stream);
#else
                    TORCH_CHECK(false, "CUTLASS fused FFN2 requested without SM120 CUTLASS support");
#endif
                } else {
                    if (DO_GEMM) gm->f2.run(g_ltH, L.f2_w8, b_ffn_mid_fp8, b_ffn_out, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.f2b);
                }
            }
            if (TP) CKNCCL(ncclAllReduce(b_ffn_out, b_ffn_out, (size_t)seq_D, ncclBfloat16, ncclSum, g_nccl_comm, stream));
            // FFN gate residual (with optional fusion to next layer)
            if (DO_EWISE) {
                int next_quad = (li + 1 < end_layer && !is_gdn_layer(li + 1)) ? (li + 1) : -1;
                if (next_quad >= 0) {
                    float* ssts_L_next = ssts_base + next_quad * ssts_layer_stride;
                    float* next_ssts_scale_sa = ssts_L_next + 0 * D;
                    float* next_ssts_shift_sa = ssts_L_next + 1 * D;
                    size_t smem_fused = (size_t)(D + 16) * sizeof(float);
                    k_gate_res_ln_adaln_ssts<<<seq, 256, smem_fused, stream>>>(
                        x, b_ffn_out, ssts_gate_ffn, temb_gate_ffn,
                        next_ssts_scale_sa, next_ssts_shift_sa,
                        temb_scale_sa, temb_shift_sa,
                        g_force_bf16_gemms ? b_norm : nullptr, b_norm_fp8, D, temb_row_stride);
                    sa_ln_fused_from_prev = true;
                } else {
                    k_gate_res_ssts<<<8*g_nsm, 256, 0, stream>>>(x, b_ffn_out, ssts_gate_ffn, temb_gate_ffn, seq, D, temb_row_stride);
                }
            }
        } else {
            // ---- Quadratic attention path (existing code) ----
            if (DO_TIME) TREC(0);

            // ---- SA LN + AdaLN ----
            if (DO_EWISE && !sa_ln_fused_from_prev) {
                // Always write both BF16 and FP8 outputs (BF16 needed for force_bf16 mode and a/b GEMMs)
                k_ln_adaln_ssts<<<seq, 256, (size_t)D*sizeof(__nv_bfloat16), stream>>>(x, b_norm, b_norm_fp8,
                    ssts_scale_sa, ssts_shift_sa, temb_scale_sa, temb_shift_sa, D, temb_row_stride);
            }
            sa_ln_fused_from_prev = false;
            if (DO_TIME) TREC(1);

            // ---- SA Q/K/V (bias fused in epilogue) ----
            if (g_force_bf16_gemms) {
                if (DO_GEMM) gm->sq_bf16.run(g_ltH, L.sq_w16, b_norm, b_q, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.sqb);
                if (DO_GEMM) gm->sk_bf16.run(g_ltH, L.sk_w16, b_norm, b_k, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.skb);
                if (DO_GEMM) gm->sv_bf16.run(g_ltH, L.sv_w16, b_norm, b_v, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.svb);
            } else {
                if (DO_GEMM) gm->sq.run(g_ltH, L.sq_w8, b_norm_fp8, b_q, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.sqb);
                if (DO_GEMM) gm->sk.run(g_ltH, L.sk_w8, b_norm_fp8, b_k, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.skb);
                if (DO_GEMM) gm->sv.run(g_ltH, L.sv_w8, b_norm_fp8, b_v, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.svb);
            }

            // ---- RMSNorm + RoPE for Q and K ----
            bool q_fp8_fused = false, k_fp8_fused = false;
            if (DO_EWISE) {
                if (!TP) {
                    if (rope_freqs && g_fp8_sdpa_enabled) {
                        size_t smem_rr = (size_t)Dpr * sizeof(__nv_bfloat16);
                        k_rmsnorm_rope_fp8<<<seq, 256, smem_rr, stream>>>(b_q, b_q_fp8_attn, L.rms_q, rope_freqs, NHpr, HD);
                        k_rmsnorm_rope_fp8<<<seq, 256, smem_rr, stream>>>(b_k, b_k_fp8_attn, L.rms_k, rope_freqs, NHpr, HD);
                        q_fp8_fused = true;
                        k_fp8_fused = true;
                    } else if (rope_freqs) {
                        size_t smem_rr = (size_t)Dpr * sizeof(__nv_bfloat16);
                        k_rmsnorm_rope<<<seq, 256, smem_rr, stream>>>(b_q, L.rms_q, rope_freqs, NHpr, HD);
                        k_rmsnorm_rope<<<seq, 256, smem_rr, stream>>>(b_k, L.rms_k, rope_freqs, NHpr, HD);
                    } else {
                        k_rmsnorm<<<seq, 256, 0, stream>>>(b_q, L.rms_q, Dpr);
                        k_rmsnorm<<<seq, 256, 0, stream>>>(b_k, L.rms_k, Dpr);
                    }
                } else {
                    // TP: 2-phase RMSNorm
                    float* sumsq_q = b_tp_sumsq_qk;
                    float* sumsq_k = b_tp_sumsq_qk + seq;
                    k_partial_sumsq<<<seq, 256, 0, stream>>>(b_q, sumsq_q, Dpr);
                    k_partial_sumsq<<<seq, 256, 0, stream>>>(b_k, sumsq_k, Dpr);
                    CKNCCL(ncclAllReduce(b_tp_sumsq_qk, b_tp_sumsq_qk, (size_t)2 * seq, ncclFloat32, ncclSum, g_nccl_comm, stream));
                    size_t smem_rr = (size_t)Dpr * sizeof(__nv_bfloat16);
                    if (g_fp8_sdpa_enabled) {
                        k_apply_rmsnorm_rope_fp8<<<seq, 256, smem_rr, stream>>>(b_q, b_q_fp8_attn, L.rms_q, rope_freqs, sumsq_q, D, NHpr, HD);
                        k_apply_rmsnorm_rope_fp8<<<seq, 256, smem_rr, stream>>>(b_k, b_k_fp8_attn, L.rms_k, rope_freqs, sumsq_k, D, NHpr, HD);
                        q_fp8_fused = true;
                        k_fp8_fused = true;
                    } else {
                        k_apply_rmsnorm_rope<<<seq, 256, smem_rr, stream>>>(b_q, L.rms_q, rope_freqs, sumsq_q, D, NHpr, HD);
                        k_apply_rmsnorm_rope<<<seq, 256, smem_rr, stream>>>(b_k, L.rms_k, rope_freqs, sumsq_k, D, NHpr, HD);
                    }
                }
            }
            if (DO_TIME) TREC(2);

            // ---- SDPA ----
            if (g_sdpa_backend == SdpaBackend::SageAttn3Py) {
                if (DO_SDPA) {
                    sageattn3_py_bf16(
                        b_q, b_k, b_v,
                        g_force_bf16_gemms ? b_attn : nullptr,
                        (DO_EWISE && !g_force_bf16_gemms) ? b_sa_out_fp8 : nullptr,
                        seq, seq, NHpr, HD, stream);
                }
                if (DO_TIME) TREC(3);
            } else if (g_sdpa_backend == SdpaBackend::SageAttn3Cpp) {
                if (DO_SDPA) {
                    sageattn3_cpp_bf16(
                        b_q, b_k, b_v,
                        b_attn,
                        (DO_EWISE && !g_force_bf16_gemms) ? b_sa_out_fp8 : nullptr,
                        seq, seq, NHpr, HD, stream);
                }
                if (DO_TIME) TREC(3);
            } else if (g_fp8_sdpa_enabled && gm->fp8_sdpa_built && !g_force_bf16_gemms) {
                if (DO_EWISE) {
                    if (!q_fp8_fused) {
                        k_bf16_to_fp8<<<8*g_nsm, 256, 0, stream>>>(b_q_fp8_attn, b_q, seq_Dpr);
                    }
                    if (!k_fp8_fused) {
                        k_bf16_to_fp8<<<8*g_nsm, 256, 0, stream>>>(b_k_fp8_attn, b_k, seq_Dpr);
                    }
                    k_bf16_to_fp8<<<8*g_nsm, 256, 0, stream>>>(b_v_fp8_attn, b_v, seq_Dpr);
                }
                if (DO_SDPA) {
                    gm->sa_sdpa_fp8.run(g_cudnnH,
                        b_q_fp8_attn, b_k_fp8_attn, b_v_fp8_attn, b_sa_out_fp8,
                        g_fp8_sdpa_descale_q, g_fp8_sdpa_descale_k, g_fp8_sdpa_descale_v,
                        g_fp8_sdpa_descale_s, g_fp8_sdpa_scale_s, g_fp8_sdpa_scale_o,
                        g_fp8_sdpa_amax_s, g_fp8_sdpa_amax_o, stream);
                }
                if (DO_TIME) TREC(3);
            } else {
                if (DO_SDPA) {
                    if (g_use_torch_sdpa) {
                        float sa_scale = 1.0f / sqrtf((float)HD);
                        torch_sdpa_bf16(b_q, b_k, b_v, b_attn, seq, seq, NHpr, HD, sa_scale, stream);
                    } else {
                        gm->sa_sdpa.run(g_cudnnH, b_q, b_k, b_v, b_attn, stream);
                    }
                }
                if (DO_TIME) TREC(3);
                if (DO_EWISE && !g_force_bf16_gemms) k_bf16_to_fp8<<<8*g_nsm, 256, 0, stream>>>(b_sa_out_fp8, b_attn, seq_Dpr);
            }

            // ---- SA_O GEMM (row-parallel) ----
            if (g_force_bf16_gemms) {
                // In force BF16 mode: SDPA output is b_attn (BF16), use BF16 GEMM
                if (DO_GEMM) gm->so_bf16.run(g_ltH, L.so_w16, b_attn, b_sa_out, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.sob);
            } else {
                if (DO_GEMM) gm->so.run(g_ltH, L.so_w8, b_sa_out_fp8, b_sa_out, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.sob);
            }

            // TP AllReduce for SA_O
            if (TP) {
                CKNCCL(ncclAllReduce(b_sa_out, b_sa_out, (size_t)seq_D, ncclBfloat16, ncclSum, g_nccl_comm, stream));
            }

            // ---- SA gated residual ----
            if (DO_EWISE) k_gate_res_ssts<<<8*g_nsm, 256, 0, stream>>>(x, b_sa_out, ssts_gate_sa, temb_gate_sa, seq, D, temb_row_stride);
            if (DO_TIME) TREC(4);

          if (!g_skip_ca) {
            // ---- CA norm2 (affine LN, cross_attn_norm) applied to ALL tokens ----
            if (DO_EWISE) {
                if (L.has_norm2) {
                    k_ln_affine<<<ca_len, 256, 0, stream>>>(x, b_ca_norm, b_ca_norm_fp8, L.n2w, L.n2b, D);
                } else if (!g_force_bf16_gemms) {
                    k_bf16_to_fp8<<<8*g_nsm, 256, 0, stream>>>(b_ca_norm_fp8, x, ca_len * D);
                }
            }
            if (DO_TIME) TREC(5);

            // ---- CA Q/K/V ----
            if (DO_GEMM) {
                if (g_force_bf16_gemms) {
                    __nv_bfloat16* ca_norm_bf16 = L.has_norm2 ? b_ca_norm : x;
                    cg->cq_bf16.run(g_ltH, L.cq_w16, ca_norm_bf16, b_ca_q, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cqb);
                } else {
                    cg->cq.run(g_ltH, L.cq_w8, b_ca_norm_fp8, b_ca_q, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cqb);
                }
            }
            __nv_bfloat16* ca_k_ptr;
            __nv_bfloat16* ca_v_ptr;
            size_t ca_kv_slot = (size_t)li * (size_t)g_ctx * (size_t)Dpr;  // use g_ctx for consistent cache slot
            if (g_ca_kv_cache_valid) {
                ca_k_ptr = b_ca_k_cache + ca_kv_slot;
                ca_v_ptr = b_ca_v_cache + ca_kv_slot;
            } else {
                if (DO_GEMM) {
                    if (g_force_bf16_gemms) {
                        g_ck_bf16.run(g_ltH, L.ck_w16, enc, b_ca_k, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.ckb);
                        g_cv_bf16.run(g_ltH, L.cv_w16, enc, b_ca_v, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cvb);
                    } else {
                        g_ck.run(g_ltH, L.ck_w8, b_enc_fp8, b_ca_k, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.ckb);
                        g_cv.run(g_ltH, L.cv_w8, b_enc_fp8, b_ca_v, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cvb);
                    }
                }
                ca_k_ptr = b_ca_k;
                ca_v_ptr = b_ca_v;
            }

            // ---- RMSNorm CA Q/K ----
            if (DO_EWISE) {
                if (!TP) {
                    k_rmsnorm<<<ca_len, 256, 0, stream>>>(b_ca_q, L.ca_rms_q, Dpr);
                    if (!g_ca_kv_cache_valid) {
                        k_rmsnorm<<<ctx, 256, 0, stream>>>(b_ca_k, L.ca_rms_k, Dpr);
                    }
                } else {
                    k_partial_sumsq<<<ca_len, 256, 0, stream>>>(b_ca_q, b_tp_sumsq_cq, Dpr);
                    if (!g_ca_kv_cache_valid) {
                        k_partial_sumsq<<<ctx, 256, 0, stream>>>(b_ca_k, b_tp_sumsq_ck, Dpr);
                        ncclGroupStart();
                        CKNCCL(ncclAllReduce(b_tp_sumsq_cq, b_tp_sumsq_cq, (size_t)ca_len, ncclFloat32, ncclSum, g_nccl_comm, stream));
                        CKNCCL(ncclAllReduce(b_tp_sumsq_ck, b_tp_sumsq_ck, (size_t)ctx,    ncclFloat32, ncclSum, g_nccl_comm, stream));
                        ncclGroupEnd();
                    } else {
                        CKNCCL(ncclAllReduce(b_tp_sumsq_cq, b_tp_sumsq_cq, (size_t)ca_len, ncclFloat32, ncclSum, g_nccl_comm, stream));
                    }
                    size_t smem_rr = (size_t)Dpr * sizeof(__nv_bfloat16);
                    k_apply_rmsnorm_rope<<<ca_len, 256, smem_rr, stream>>>(b_ca_q, L.ca_rms_q, nullptr, b_tp_sumsq_cq, D, NHpr, HD);
                    if (!g_ca_kv_cache_valid) {
                        k_apply_rmsnorm_rope<<<ctx, 256, smem_rr, stream>>>(b_ca_k, L.ca_rms_k, nullptr, b_tp_sumsq_ck, D, NHpr, HD);
                    }
                }
                // Populate CA K/V cache on miss
                if (!g_ca_kv_cache_valid) {
                    CK(cudaMemcpyAsync(b_ca_k_cache + ca_kv_slot, b_ca_k,
                                       (size_t)ctx * Dpr * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToDevice, stream));
                    CK(cudaMemcpyAsync(b_ca_v_cache + ca_kv_slot, b_ca_v,
                                       (size_t)ctx * Dpr * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToDevice, stream));
                }
            }
            if (DO_TIME) TREC(6);

            // ---- CA SDPA ----
            if (DO_SDPA) {
                if (li == 0) {
                }
                if (g_use_torch_sdpa) {
                    float ca_scale = 1.0f / sqrtf((float)HD);
                    torch_sdpa_bf16(b_ca_q, ca_k_ptr, ca_v_ptr, b_ca_out, ca_len, ctx, NHpr, HD, ca_scale, stream);
                } else {
                    cg->ca_sdpa.run(g_cudnnH, b_ca_q, ca_k_ptr, ca_v_ptr, b_ca_out, stream);
                }
            }
            if (DO_TIME) TREC(7);

            // ---- CA_O GEMM + ungated residual ----
            // Kairos: CA residual is ungated (x += ca_out, no gate)
            if (g_force_bf16_gemms) {
                if (!TP) {
                    if (DO_GEMM) cg->co_bf16.run(g_ltH, L.co_w16, b_ca_out, x, g_gemm_ws, 32*1024*1024, 1.f, 1.f, stream, L.cob);
                } else {
                    if (DO_GEMM) cg->co_bf16.run(g_ltH, L.co_w16, b_ca_out, b_sa_out, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cob);
                    CKNCCL(ncclAllReduce(b_sa_out, b_sa_out, (size_t)ca_len * D, ncclBfloat16, ncclSum, g_nccl_comm, stream));
                    if (DO_EWISE) k_add_vec<<<8*g_nsm, 256, 0, stream>>>(x, b_sa_out, ca_len * D);
                }
            } else {
                if (DO_EWISE) k_bf16_to_fp8<<<8*g_nsm, 256, 0, stream>>>(b_ca_out_fp8, b_ca_out, ca_len_Dpr);
                if (!TP) {
                    // Single-GPU: fused bias + residual via beta=1
                    if (DO_GEMM) cg->co.run(g_ltH, L.co_w8, b_ca_out_fp8, x, g_gemm_ws, 32*1024*1024, 1.f, 1.f, stream, L.cob);
                } else {
                    // TP row-parallel: bias/W in epilogue, AllReduce, then simple residual add
                    if (DO_GEMM) cg->co.run(g_ltH, L.co_w8, b_ca_out_fp8, b_sa_out, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.cob);
                    CKNCCL(ncclAllReduce(b_sa_out, b_sa_out, (size_t)ca_len * D, ncclBfloat16, ncclSum, g_nccl_comm, stream));
                    if (DO_EWISE) k_add_vec<<<8*g_nsm, 256, 0, stream>>>(x, b_sa_out, ca_len * D);
                }
            }
            if (DO_TIME) TREC(8);
          } // end if (!g_skip_ca) — quadratic CA path

            // ---- FFN LN on full x ----
            if (DO_EWISE) {
                k_ln_adaln_ssts<<<seq, 256, (size_t)D*sizeof(__nv_bfloat16), stream>>>(x, b_ffn_norm, b_ffn_norm_fp8,
                    ssts_scale_ffn, ssts_shift_ffn, temb_scale_ffn, temb_shift_ffn, D, temb_row_stride);
            }
            if (DO_TIME) TREC(9);

            // ---- FFN1 + SiLU + FP8 ----
            if (g_force_bf16_gemms) {
                if (DO_GEMM) gm->f1_bf16.run(g_ltH, L.f1_w16, b_ffn_norm, b_ffn_mid, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.f1b);
                if (DO_EWISE) k_silu_bf16<<<(seq_FFNpr + 255) / 256, 256, 0, stream>>>(b_ffn_mid, seq_FFNpr);
            } else {
                bool use_fp4_ffn2 = g_fp4_ffn2 && DO_GEMM && DO_EWISE;
                bool use_fused_ffn1_fp4 = use_fp4_ffn2 && g_cutlass_fused_ffn1_fp4;
                bool use_fused_ffn1 = g_cutlass_fused_ffn1 && DO_GEMM && DO_EWISE && !use_fused_ffn1_fp4;
                if (use_fused_ffn1_fp4) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
                    gm->f1_fused_fp4.run(
                        L.f1_w8,
                        b_ffn_norm_fp8,
                        b_ffn_mid,
                        b_ffn_mid_fp4,
                        b_ffn_mid_fp4_scales_swizzled,
                        L.f1b,
                        g_fp4_normconst,
                        stream);
#else
                    TORCH_CHECK(false, "CUTLASS fused FFN1 FP4 requested without SM120 CUTLASS support");
#endif
                } else if (use_fused_ffn1) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
                    gm->f1_fused.run(L.f1_w8, b_ffn_norm_fp8, b_ffn_mid_fp8, L.f1b, stream);
#else
                    TORCH_CHECK(false, "CUTLASS fused FFN1 requested without SM120 CUTLASS support");
#endif
                } else {
                    if (DO_GEMM) gm->f1.run(g_ltH, L.f1_w8, b_ffn_norm_fp8, b_ffn_mid, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.f1b);
                    if (DO_EWISE) {
                        if (g_fp4_ffn2 && DO_GEMM) {
                            int rows_padded = fp4_scale_rows(seq);
                            int scale_cols = fp4_scale_cols(g_FFN_per_rank);
                            size_t packed_bytes = (size_t)rows_padded * (size_t)g_FFN_per_rank / 2;
                            size_t scale_bytes = (size_t)rows_padded * (size_t)scale_cols;
                            if (rows_padded != seq) {
                                CK(cudaMemsetAsync(b_ffn_mid_fp4, 0, packed_bytes, stream));
                            }
                            if (rows_padded != seq || scale_cols != (g_FFN_per_rank / 16)) {
                                CK(cudaMemsetAsync(b_ffn_mid_fp4_scales_swizzled, fp4_scale_one_byte(), scale_bytes, stream));
                            }
                            constexpr int FP4_WARPS_PER_BLOCK = 4;
                            constexpr int FP4_THREADS = FP4_WARPS_PER_BLOCK * 32;
                            int fp4_grid_x = (g_FFN_per_rank + (FP4_WARPS_PER_BLOCK * 32 - 1)) / (FP4_WARPS_PER_BLOCK * 32);
                            k_silu_to_fp4_rowwise_swizzled<<<dim3(fp4_grid_x, seq), FP4_THREADS, 0, stream>>>(
                                b_ffn_mid,
                                b_ffn_mid_fp4,
                                b_ffn_mid_fp4_scales_swizzled,
                                seq,
                                g_FFN_per_rank,
                                scale_cols);
                        } else {
                            k_silu_to_fp8<<<8*g_nsm, 256, 0, stream>>>(b_ffn_mid, b_ffn_mid_fp8, seq_FFNpr);
                        }
                    }
                }
            }
            if (DO_TIME) TREC(10);

            // ---- FFN2 (row-parallel) ----
            if (g_force_bf16_gemms) {
                if (DO_GEMM) gm->f2_bf16.run(g_ltH, L.f2_w16, b_ffn_mid, b_ffn_out, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.f2b);
            } else {
                bool use_fp4_ffn2 = g_fp4_ffn2 && DO_GEMM && DO_EWISE;
                bool use_fused_ffn2 = g_cutlass_fused_ffn2 && DO_GEMM;
                if (use_fp4_ffn2) {
                    run_ffn2_fp4(L, gm, seq, stream);
                } else if (use_fused_ffn2) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
                    gm->f2_fused.run(L.f2_w8, b_ffn_mid_fp8, b_ffn_out, L.f2b, stream);
#else
                    TORCH_CHECK(false, "CUTLASS fused FFN2 requested without SM120 CUTLASS support");
#endif
                } else {
                    if (DO_GEMM) gm->f2.run(g_ltH, L.f2_w8, b_ffn_mid_fp8, b_ffn_out, g_gemm_ws, 32*1024*1024, 1.f, 0.f, stream, L.f2b);
                }
            }
            if (TP) {
                CKNCCL(ncclAllReduce(b_ffn_out, b_ffn_out, (size_t)seq_D, ncclBfloat16, ncclSum, g_nccl_comm, stream));
            }

            // ---- FFN gated residual ----
            // Optimization 3: fuse gate_res + next-layer SA LN when the next layer
            // is also in range AND is a quadratic layer (not GatedDeltaNet).
            if (DO_EWISE) {
                int next_quad = (li + 1 < end_layer && !is_gdn_layer(li + 1)) ? (li + 1) : -1;
                if (next_quad >= 0) {
                    // Fused: gate_res + next layer's SA LN
                    float* ssts_L_next = ssts_base + next_quad * ssts_layer_stride;
                    float* next_ssts_scale_sa = ssts_L_next + 0 * D;
                    float* next_ssts_shift_sa = ssts_L_next + 1 * D;
                    size_t smem_fused = (size_t)(D + 16) * sizeof(float);
                    k_gate_res_ln_adaln_ssts<<<seq, 256, smem_fused, stream>>>(
                        x, b_ffn_out,
                        ssts_gate_ffn, temb_gate_ffn,
                        next_ssts_scale_sa, next_ssts_shift_sa,
                        temb_scale_sa, temb_shift_sa,
                        g_force_bf16_gemms ? b_norm : nullptr, b_norm_fp8,
                        D, temb_row_stride);
                    sa_ln_fused_from_prev = true;
                } else {
                    // Last quadratic layer in range: no fusion
                    k_gate_res_ssts<<<8*g_nsm, 256, 0, stream>>>(x, b_ffn_out, ssts_gate_ffn, temb_gate_ffn, seq, D, temb_row_stride);
                }
            }
            if (DO_TIME) { TREC(11); quad_idx++; }
        } // end if/else GDN vs quadratic
    }

    // Mark CA K/V cache valid after first forward
    // Don't auto-enable cache — CFG needs fresh K/V per call
    // Cache can be explicitly enabled via set_ca_kv_cache_valid()
    // if (!g_ca_kv_cache_valid) g_ca_kv_cache_valid = true;

    // Print per-phase timing
    if (DO_TIME && n_quadratic > 0) {
        cudaStreamSynchronize(stream);
        const char* PHASE_NAMES[NPHASES] = {
            "SA_LN   ", "SA_QKV  ", "SA_SDPA ", "SA_O    ",
            "CA_LN   ", "CA_QKV  ", "CA_SDPA ", "CA_O    ",
            "FFN_LN  ", "FFN1    ", "FFN2    "
        };
        float totals[NPHASES] = {0};
        for (int qi = 0; qi < n_quadratic; qi++) {
            for (int p = 0; p < NPHASES; p++) {
                float dt;
                cudaEventElapsedTime(&dt,
                    evs[qi * (NPHASES + 1) + p],
                    evs[qi * (NPHASES + 1) + p + 1]);
                totals[p] += dt;
            }
        }
        float total_all = 0;
        for (int p = 0; p < NPHASES; p++) total_all += totals[p];
        printf("\n=== KAIROS_ENGINE_TIME: %d quadratic layers [%d,%d), total=%.3f ms ===\n",
               n_quadratic, start_layer, end_layer, total_all);
        for (int p = 0; p < NPHASES; p++) {
            printf("  %s %7.3f ms  (%4.1f%%)   %.4f ms/layer\n",
                PHASE_NAMES[p], totals[p], 100.0f * totals[p] / total_all, totals[p] / n_quadratic);
        }
        for (auto& e : evs) cudaEventDestroy(e);
    }
}

// ============================================================
// engine_forward: Python-callable wrapper
// ============================================================
void set_use_graph(int64_t flag) { g_use_graph = (flag != 0); }

static uint64_t make_graph_key(
    int seq, int actual_ctx, int ca_len, int temb_rows,
    int start_layer, int end_layer, bool has_rope)
{
    uint64_t key = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        key ^= v + 0x9e3779b97f4a7c15ull + (key << 6) + (key >> 2);
    };
    mix((uint64_t)seq);
    mix((uint64_t)actual_ctx);
    mix((uint64_t)ca_len);
    mix((uint64_t)temb_rows);
    mix((uint64_t)start_layer);
    mix((uint64_t)end_layer);
    mix(has_rope ? 1ull : 0ull);
    return key;
}

torch::Tensor engine_forward(
    torch::Tensor x_t,      // [seq, D] bf16
    torch::Tensor enc_t,    // [ctx, CA_K_DIM] bf16
    torch::Tensor temb_t,   // [seq, 6, D] or [1, 6, D] float32
    torch::Tensor ssts_t,   // [NL, 6, D] float32
    torch::Tensor rope_t,   // [seq, 2*HD] float32 or empty
    int64_t ca_len,         // always == seq for Kairos
    int64_t start_layer,    // first layer to run (inclusive)
    int64_t end_layer       // last layer to run (exclusive)
) {
    TORCH_CHECK(g_ready, "engine_init first");
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int seq = x_t.size(0);
    int ca_l = (int)ca_len;
    if (ca_l <= 0 || ca_l > seq) ca_l = seq;
    // Set actual context size from encoder tensor (may be < g_ctx)
    int actual_ctx = (int)enc_t.size(0);
    update_actual_ctx_and_gemms(actual_ctx);
    int temb_rows = temb_t.size(0);
    int temb_row_stride = (temb_rows == 1) ? 0 : (6 * g_D);
    bool has_rope = (rope_t.numel() > 0);
    float* rope_ptr = has_rope ? rope_t.data_ptr<float>() : nullptr;

    int sl = (int)start_layer;
    int el = (int)end_layer;
    if (sl < 0) sl = 0;
    if (el <= 0 || el > g_NL) el = g_NL;

    if (!g_use_graph) {
        forward_core(
            (__nv_bfloat16*)x_t.data_ptr(), (__nv_bfloat16*)enc_t.data_ptr(),
            temb_t.data_ptr<float>(), ssts_t.data_ptr<float>(), rope_ptr,
            seq, ca_l, temb_row_stride,
            sl, el,
            stream);
        return x_t;
    }

    // CUDA graph path
    size_t x_bytes = (size_t)seq * g_D * sizeof(__nv_bfloat16);
    size_t enc_bytes = (size_t)actual_ctx * g_CA_K_DIM * sizeof(__nv_bfloat16);
    size_t temb_bytes = (size_t)temb_rows * 6 * g_D * sizeof(float);
    size_t ssts_bytes = (size_t)g_NL * 6 * g_D * sizeof(float);
    size_t rope_bytes = has_rope ? (size_t)seq * 2 * g_HD * sizeof(float) : 0;

    CK(cudaStreamSynchronize(stream));

    CK(cudaMemcpyAsync(g_x_buf, x_t.data_ptr(), x_bytes, cudaMemcpyDeviceToDevice, g_graph_stream));
    CK(cudaMemcpyAsync(g_enc_buf, enc_t.data_ptr(), enc_bytes, cudaMemcpyDeviceToDevice, g_graph_stream));
    CK(cudaMemcpyAsync(g_temb_buf, temb_t.data_ptr(), temb_bytes, cudaMemcpyDeviceToDevice, g_graph_stream));
    CK(cudaMemcpyAsync(g_ssts_buf, ssts_t.data_ptr(), ssts_bytes, cudaMemcpyDeviceToDevice, g_graph_stream));
    if (has_rope) {
        CK(cudaMemcpyAsync(g_rope_buf, rope_t.data_ptr(), rope_bytes, cudaMemcpyDeviceToDevice, g_graph_stream));
    }

    // Graph key includes the dimensions and branch shape that affect capture.
    uint64_t graph_key = make_graph_key(seq, actual_ctx, ca_l, temb_rows, sl, el, has_rope);
    float* graph_rope_ptr = has_rope ? g_rope_buf : nullptr;
    auto git = g_graphs_by_key.find(graph_key);
    if (git == g_graphs_by_key.end()) {
        // Warmup
        forward_core(g_x_buf, g_enc_buf, g_temb_buf, g_ssts_buf, graph_rope_ptr,
                     seq, ca_l, temb_row_stride, sl, el, g_graph_stream);
        CK(cudaStreamSynchronize(g_graph_stream));
        // Capture
        CK(cudaStreamBeginCapture(g_graph_stream, cudaStreamCaptureModeRelaxed));
        forward_core(g_x_buf, g_enc_buf, g_temb_buf, g_ssts_buf, graph_rope_ptr,
                     seq, ca_l, temb_row_stride, sl, el, g_graph_stream);
        cudaGraph_t graph;
        CK(cudaStreamEndCapture(g_graph_stream, &graph));
        cudaGraphExec_t graph_exec;
        CK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        CK(cudaGraphDestroy(graph));
        g_graphs_by_key[graph_key] = graph_exec;
        printf("  Built graph for seq=%d ctx=%d ca=%d temb=%d rope=%d layers=[%d,%d)\n",
               seq, actual_ctx, ca_l, temb_rows, has_rope ? 1 : 0, sl, el);
        CK(cudaGraphLaunch(graph_exec, g_graph_stream));
    } else {
        CK(cudaGraphLaunch(git->second, g_graph_stream));
    }

    CK(cudaMemcpyAsync(x_t.data_ptr(), g_x_buf, x_bytes, cudaMemcpyDeviceToDevice, g_graph_stream));
    CK(cudaStreamSynchronize(g_graph_stream));
    return x_t;
}

// Debug: expose internal buffers as torch tensors (view, no copy)
torch::Tensor get_buf(std::string name, int64_t seq) {
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto opts_fp8 = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA);
    void* p = nullptr;
    bool is_fp8 = false;
    int n_rows = seq;
    int n_cols = g_D;
    if      (name == "b_norm")         { p = b_norm; }
    else if (name == "b_norm_fp8")     { p = b_norm_fp8; is_fp8 = true; }
    else if (name == "b_q")            { p = b_q; }
    else if (name == "b_k")            { p = b_k; }
    else if (name == "b_v")            { p = b_v; }
    else if (name == "b_attn")         { p = b_attn; }
    else if (name == "b_sa_out")       { p = b_sa_out; }
    else if (name == "b_ca_norm")      { p = b_ca_norm; }
    else if (name == "b_ca_norm_fp8")  { p = b_ca_norm_fp8; is_fp8 = true; }
    else if (name == "b_ca_q")         { p = b_ca_q; }
    else if (name == "b_ca_k")         { p = b_ca_k; n_rows = g_ctx; }
    else if (name == "b_ca_v")         { p = b_ca_v; n_rows = g_ctx; }
    else if (name == "b_ca_out")       { p = b_ca_out; }
    else if (name == "b_ffn_norm")     { p = b_ffn_norm; }
    else if (name == "b_ffn_mid")      { p = b_ffn_mid; n_cols = g_FFN; }
    else if (name == "b_ffn_out")      { p = b_ffn_out; }
    else TORCH_CHECK(false, "unknown buffer: ", name);
    auto& opts = is_fp8 ? opts_fp8 : opts_bf16;
    return torch::from_blob(p, {n_rows, n_cols}, opts).clone();
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
quantize_fp4_rowwise_bf16(torch::Tensor in_t) {
    TORCH_CHECK(in_t.is_cuda(), "input must be CUDA");
    TORCH_CHECK(in_t.scalar_type() == torch::kBFloat16, "input must be bf16");
    TORCH_CHECK(in_t.dim() == 2, "input must be rank-2 [rows, cols]");
    auto in_c = in_t.contiguous();
    int rows = (int)in_c.size(0);
    int cols = (int)in_c.size(1);
    TORCH_CHECK(cols % 16 == 0, "FP4 quantization requires cols divisible by 16, got ", cols);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto packed = torch::zeros({rows, cols / 2}, opts_u8);
    auto scales = torch::zeros({fp4_scale_rows(rows), fp4_scale_cols(cols)}, opts_u8);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int FP4_WARPS_PER_BLOCK = 4;
    constexpr int FP4_THREADS = FP4_WARPS_PER_BLOCK * 32;
    int grid_x = (cols + (FP4_WARPS_PER_BLOCK * 32 - 1)) / (FP4_WARPS_PER_BLOCK * 32);
    k_quantize_rowwise_bf16_to_fp4<<<dim3(grid_x, rows), FP4_THREADS, 0, stream>>>(
        (__nv_bfloat16*)in_c.data_ptr(),
        (uint8_t*)packed.data_ptr(),
        (uint8_t*)scales.data_ptr(),
        rows, cols, (int)scales.size(1));
    auto swizzled = swizzle_fp4_scales_rowwise(scales);
    return {packed, scales, swizzled};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
quantize_fp4_colmajor_bf16(torch::Tensor in_t) {
    TORCH_CHECK(in_t.is_cuda(), "input must be CUDA");
    TORCH_CHECK(in_t.scalar_type() == torch::kBFloat16, "input must be bf16");
    TORCH_CHECK(in_t.dim() == 2, "input must be rank-2 [rows, cols]");
    auto in_c = in_t.contiguous();
    int rows = (int)in_c.size(0);
    int cols = (int)in_c.size(1);
    TORCH_CHECK(rows % 2 == 0, "ColumnMajor FP4 packing requires even row count, got ", rows);
    TORCH_CHECK(cols % 16 == 0, "FP4 quantization requires cols divisible by 16, got ", cols);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto packed = torch::zeros({cols, rows / 2}, opts_u8);
    auto scales = torch::zeros({fp4_scale_rows(rows), fp4_scale_cols(cols)}, opts_u8);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    k_quantize_colmajor_bf16_to_fp4<<<dim3(cols / 16, rows / 2), 32, 0, stream>>>(
        (__nv_bfloat16*)in_c.data_ptr(),
        (uint8_t*)packed.data_ptr(),
        (uint8_t*)scales.data_ptr(),
        rows, cols, (int)scales.size(1));
    auto swizzled = swizzle_fp4_scales_rowwise(scales);
    return {packed, scales, swizzled};
}

static torch::Tensor run_fp4_gemm_rowwise_cublas(torch::Tensor w_t, torch::Tensor act_t) {
    TORCH_CHECK(w_t.is_cuda() && act_t.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(w_t.scalar_type() == torch::kBFloat16 && act_t.scalar_type() == torch::kBFloat16,
                "inputs must be bf16");
    TORCH_CHECK(w_t.dim() == 2 && act_t.dim() == 2, "inputs must be rank-2");
    int N = (int)w_t.size(0);
    int K = (int)w_t.size(1);
    int M = (int)act_t.size(0);
    TORCH_CHECK((int)act_t.size(1) == K, "K mismatch");
    TORCH_CHECK(K % 32 == 0, "FP4 GEMM requires K divisible by 32, got ", K);

    ensure_gemm_runtime();
    auto [w_packed, w_scales, w_swizzled] = quantize_fp4_colmajor_bf16(w_t);
    auto [a_packed, a_scales, a_swizzled] = quantize_fp4_rowwise_bf16(act_t);
    (void)w_scales;
    (void)a_scales;

    auto out = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    FP4Gemm gemm;
    gemm.setup(g_ltH, M, N, K,
               (uint8_t*)w_swizzled.data_ptr(),
               (uint8_t*)a_swizzled.data_ptr(),
               g_gemm_ws, g_gemm_ws_bytes);
    gemm.run(g_ltH,
             w_packed.data_ptr(),
             a_packed.data_ptr(),
             out.data_ptr(),
             g_gemm_ws, g_gemm_ws_bytes,
             g_scaleA, g_scaleZero, stream);
    return out;
}

static torch::Tensor run_fp4_gemm_rowwise_packed_cublas(
    torch::Tensor w_packed,
    torch::Tensor w_swizzled,
    torch::Tensor act_packed,
    torch::Tensor act_swizzled,
    int64_t M,
    int64_t N,
    int64_t K) {
    TORCH_CHECK(w_packed.is_cuda() && w_swizzled.is_cuda() && act_packed.is_cuda() && act_swizzled.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(w_packed.scalar_type() == torch::kUInt8 &&
                w_swizzled.scalar_type() == torch::kUInt8 &&
                act_packed.scalar_type() == torch::kUInt8 &&
                act_swizzled.scalar_type() == torch::kUInt8,
                "packed/scales must be uint8 tensors");
    TORCH_CHECK(K % 32 == 0, "FP4 GEMM requires K divisible by 32, got ", K);
    ensure_gemm_runtime();
    auto out = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    FP4Gemm gemm;
    gemm.setup(g_ltH, (int)M, (int)N, (int)K,
               (uint8_t*)w_swizzled.data_ptr(),
               (uint8_t*)act_swizzled.data_ptr(),
               g_gemm_ws, g_gemm_ws_bytes);
    gemm.run(g_ltH,
             w_packed.data_ptr(),
             act_packed.data_ptr(),
             out.data_ptr(),
             g_gemm_ws, g_gemm_ws_bytes,
             g_scaleA, g_scaleZero, stream);
    return out;
}

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
static torch::Tensor run_fp4_gemm_rowwise_cutlass(torch::Tensor w_t, torch::Tensor act_t) {
    TORCH_CHECK(w_t.is_cuda() && act_t.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(w_t.scalar_type() == torch::kBFloat16 && act_t.scalar_type() == torch::kBFloat16,
                "inputs must be bf16");
    TORCH_CHECK(w_t.dim() == 2 && act_t.dim() == 2, "inputs must be rank-2");
    int N = (int)w_t.size(0);
    int K = (int)w_t.size(1);
    int M = (int)act_t.size(0);
    TORCH_CHECK((int)act_t.size(1) == K, "K mismatch");
    TORCH_CHECK(K % 32 == 0, "FP4 GEMM requires K divisible by 32, got ", K);
    auto out = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CutlassFp4RowwiseGemm gemm;
    gemm.setup(M, N, K);
    auto [w_packed, _w_scales, w_swizzled] = quantize_fp4_rowwise_bf16(w_t);
    auto [a_packed, _a_scales, a_swizzled] = quantize_fp4_rowwise_bf16(act_t);

    gemm.run(
        w_packed.data_ptr(),
        w_swizzled.data_ptr(),
        a_packed.data_ptr(),
        a_swizzled.data_ptr(),
        out.data_ptr(),
        stream);
    return out;
}

static torch::Tensor run_fp4_gemm_rowwise_packed_cutlass(
    torch::Tensor w_packed,
    torch::Tensor w_swizzled,
    torch::Tensor act_packed,
    torch::Tensor act_swizzled,
    int64_t M,
    int64_t N,
    int64_t K) {
    TORCH_CHECK(w_packed.is_cuda() && w_swizzled.is_cuda() && act_packed.is_cuda() && act_swizzled.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(w_packed.scalar_type() == torch::kUInt8 &&
                w_swizzled.scalar_type() == torch::kUInt8 &&
                act_packed.scalar_type() == torch::kUInt8 &&
                act_swizzled.scalar_type() == torch::kUInt8,
                "packed/scales must be uint8 tensors");
    TORCH_CHECK(K % 32 == 0, "FP4 GEMM requires K divisible by 32, got ", K);
    auto out = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CutlassFp4RowwiseGemm gemm;
    gemm.setup((int)M, (int)N, (int)K);
    gemm.run(
        w_packed.data_ptr(),
        w_swizzled.data_ptr(),
        act_packed.data_ptr(),
        act_swizzled.data_ptr(),
        out.data_ptr(),
        stream);
    return out;
}
#endif

static torch::Tensor run_fp4_gemm_rowwise(torch::Tensor w_t, torch::Tensor act_t) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    return run_fp4_gemm_rowwise_cutlass(w_t, act_t);
#else
    return run_fp4_gemm_rowwise_cublas(w_t, act_t);
#endif
}

static torch::Tensor run_fp4_gemm_rowwise_packed(
    torch::Tensor w_packed,
    torch::Tensor w_swizzled,
    torch::Tensor act_packed,
    torch::Tensor act_swizzled,
    int64_t M,
    int64_t N,
    int64_t K) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    return run_fp4_gemm_rowwise_packed_cutlass(
        w_packed, w_swizzled, act_packed, act_swizzled, M, N, K);
#else
    return run_fp4_gemm_rowwise_packed_cublas(
        w_packed, w_swizzled, act_packed, act_swizzled, M, N, K);
#endif
}

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
template <class LayoutSF>
static torch::Tensor debug_scale_layout_map(LayoutSF layout_sf, int rows, int cols) {
    TORCH_CHECK(cols % 16 == 0, "cols must be divisible by 16");
    auto map = torch::full(
        {(long)cute::size(cute::filter_zeros(layout_sf))},
        -1,
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto* ptr = map.data_ptr<int32_t>();
    int blocks = cols / 16;
    for (int row = 0; row < rows; ++row) {
        for (int block16 = 0; block16 < blocks; ++block16) {
            int32_t logical_id = row * blocks + block16;
            size_t idx0 = (size_t)layout_sf(cute::make_coord(row, block16 * 16, 0));
            for (int kk = 1; kk < 16; ++kk) {
                size_t idxk = (size_t)layout_sf(cute::make_coord(row, block16 * 16 + kk, 0));
                TORCH_CHECK(idxk == idx0,
                            "scale layout maps one 16-wide block to multiple indices at row=", row,
                            " block16=", block16, " kk=", kk, " idx0=", idx0, " idxk=", idxk);
            }
            TORCH_CHECK(idx0 < (size_t)map.numel(), "scale layout index out of range");
            TORCH_CHECK(ptr[idx0] == -1 || ptr[idx0] == logical_id,
                        "scale layout alias mismatch at raw index ", idx0,
                        " existing=", ptr[idx0], " new=", logical_id);
            ptr[idx0] = logical_id;
        }
    }
    return map;
}

static std::tuple<torch::Tensor, torch::Tensor> debug_ffn1_ffn2_scale_layout_maps(
    int64_t seq, int64_t d, int64_t ffn) {
    TORCH_CHECK(seq > 0 && d > 0 && ffn > 0, "seq/d/ffn must be positive");
    TORCH_CHECK(ffn % 16 == 0, "ffn must be divisible by 16");
    CutlassFusedFfn1Fp4 producer;
    producer.setup((int)seq, (int)ffn, (int)d);
    CutlassFp4RowwiseGemm consumer;
    consumer.setup((int)seq, (int)d, (int)ffn);
    auto producer_map = debug_scale_layout_map(producer.layout_SFD, (int)seq, (int)ffn);
    auto consumer_map = debug_scale_layout_map(consumer.layout_SFA, (int)seq, (int)ffn);
    return {producer_map, consumer_map};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
debug_compare_ffn1_fp4_producers_impl(
    torch::Tensor w_t,
    torch::Tensor act_t,
    torch::Tensor bias_t,
    float norm_constant) {
    TORCH_CHECK(w_t.is_cuda() && act_t.is_cuda() && bias_t.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(w_t.scalar_type() == torch::kBFloat16 &&
                act_t.scalar_type() == torch::kBFloat16 &&
                bias_t.scalar_type() == torch::kBFloat16,
                "inputs must be bf16");
    TORCH_CHECK(w_t.dim() == 2 && act_t.dim() == 2 && bias_t.dim() == 1,
                "expected w[N,K], act[M,K], bias[N]");
    auto w_bf16 = w_t.contiguous();
    auto act_bf16 = act_t.contiguous();
    auto bias_bf16 = bias_t.contiguous();
    int N = (int)w_bf16.size(0);
    int K = (int)w_bf16.size(1);
    int M = (int)act_bf16.size(0);
    TORCH_CHECK((int)act_bf16.size(1) == K, "K mismatch");
    TORCH_CHECK((int)bias_bf16.numel() == N, "bias size mismatch");
    TORCH_CHECK(N % 16 == 0, "FFN output dim must be divisible by 16, got ", N);

    ensure_gemm_runtime();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto w_fp8 = w_bf16.to(torch::kFloat8_e4m3fn).contiguous();
    auto act_fp8 = act_bf16.to(torch::kFloat8_e4m3fn).contiguous();

    auto ref_mid = torch::zeros({M, N}, opts_bf16);
    FP8Gemm ref_gemm;
    ref_gemm.setup(g_ltH, M, N, K, g_scaleA, g_scaleB, g_gemm_ws, g_gemm_ws_bytes, 0);
    ref_gemm.run(g_ltH,
                 w_fp8.data_ptr(),
                 act_fp8.data_ptr(),
                 ref_mid.data_ptr(),
                 g_gemm_ws, g_gemm_ws_bytes,
                 1.f, 0.f, stream,
                 bias_bf16.data_ptr());

    int rows_padded = fp4_scale_rows(M);
    int scale_cols = fp4_scale_cols(N);
    auto ref_packed = torch::zeros({rows_padded, N / 2}, opts_u8);
    auto ref_scales = torch::empty({rows_padded, scale_cols}, opts_u8);
    CK(cudaMemsetAsync(ref_scales.data_ptr(), fp4_scale_one_byte(), ref_scales.nbytes(), stream));
    k_silu_to_fp4_rowwise_swizzled<<<dim3(N / 16, M), 32, 0, stream>>>(
        (__nv_bfloat16*)ref_mid.data_ptr(),
        (uint8_t*)ref_packed.data_ptr(),
        (uint8_t*)ref_scales.data_ptr(),
        M, N, scale_cols);

    auto fused_src = torch::zeros({M, N}, opts_bf16);
    auto fused_packed = torch::zeros({rows_padded, N / 2}, opts_u8);
    auto fused_scales = torch::empty({rows_padded, scale_cols}, opts_u8);
    CK(cudaMemsetAsync(fused_scales.data_ptr(), fp4_scale_one_byte(), fused_scales.nbytes(), stream));
    auto normconst = torch::full({1}, norm_constant, opts_f32);

    CutlassFusedFfn1Fp4 fused;
    fused.setup(M, N, K);
    fused.run(
        w_fp8.data_ptr(),
        act_fp8.data_ptr(),
        (__nv_bfloat16*)fused_src.data_ptr(),
        fused_packed.data_ptr(),
        fused_scales.data_ptr(),
        (__nv_bfloat16*)bias_bf16.data_ptr(),
        normconst.data_ptr<float>(),
        stream);

    CK(cudaStreamSynchronize(stream));
    return {ref_mid, ref_packed, ref_scales, fused_packed, fused_scales};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
debug_compare_ffn1_fp4_producers(torch::Tensor w_t, torch::Tensor act_t, torch::Tensor bias_t) {
    return debug_compare_ffn1_fp4_producers_impl(w_t, act_t, bias_t, 6.0f);
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
debug_compare_ffn1_fp4_producers_norm(
    torch::Tensor w_t,
    torch::Tensor act_t,
    torch::Tensor bias_t,
    double norm_constant) {
    return debug_compare_ffn1_fp4_producers_impl(w_t, act_t, bias_t, (float)norm_constant);
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
debug_compare_ffn1_fp4_identity_producer(
    torch::Tensor w_t,
    torch::Tensor act_t,
    torch::Tensor bias_t,
    double norm_constant) {
    TORCH_CHECK(w_t.is_cuda() && act_t.is_cuda() && bias_t.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(w_t.scalar_type() == torch::kBFloat16 &&
                act_t.scalar_type() == torch::kBFloat16 &&
                bias_t.scalar_type() == torch::kBFloat16,
                "inputs must be bf16");
    TORCH_CHECK(w_t.dim() == 2 && act_t.dim() == 2 && bias_t.dim() == 1,
                "expected w[N,K], act[M,K], bias[N]");
    auto w_bf16 = w_t.contiguous();
    auto act_bf16 = act_t.contiguous();
    auto bias_bf16 = bias_t.contiguous();
    int N = (int)w_bf16.size(0);
    int K = (int)w_bf16.size(1);
    int M = (int)act_bf16.size(0);
    TORCH_CHECK((int)act_bf16.size(1) == K, "K mismatch");
    TORCH_CHECK((int)bias_bf16.numel() == N, "bias size mismatch");
    TORCH_CHECK(N % 16 == 0, "FFN output dim must be divisible by 16, got ", N);

    ensure_gemm_runtime();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto w_fp8 = w_bf16.to(torch::kFloat8_e4m3fn).contiguous();
    auto act_fp8 = act_bf16.to(torch::kFloat8_e4m3fn).contiguous();

    auto ref_mid = torch::zeros({M, N}, opts_bf16);
    FP8Gemm ref_gemm;
    ref_gemm.setup(g_ltH, M, N, K, g_scaleA, g_scaleB, g_gemm_ws, g_gemm_ws_bytes, 0);
    ref_gemm.run(g_ltH,
                 w_fp8.data_ptr(),
                 act_fp8.data_ptr(),
                 ref_mid.data_ptr(),
                 g_gemm_ws, g_gemm_ws_bytes,
                 1.f, 0.f, stream,
                 bias_bf16.data_ptr());

    int rows_padded = fp4_scale_rows(M);
    int scale_cols = fp4_scale_cols(N);
    auto fused_src = torch::zeros({M, N}, opts_bf16);
    auto fused_packed = torch::zeros({rows_padded, N / 2}, opts_u8);
    auto fused_scales = torch::empty({rows_padded, scale_cols}, opts_u8);
    CK(cudaMemsetAsync(fused_scales.data_ptr(), fp4_scale_one_byte(), fused_scales.nbytes(), stream));
    auto normconst = torch::full({1}, (float)norm_constant, opts_f32);

    CutlassFusedFfn1Fp4Identity fused;
    fused.setup(M, N, K);
    fused.run(
        w_fp8.data_ptr(),
        act_fp8.data_ptr(),
        (__nv_bfloat16*)fused_src.data_ptr(),
        fused_packed.data_ptr(),
        fused_scales.data_ptr(),
        (__nv_bfloat16*)bias_bf16.data_ptr(),
        normconst.data_ptr<float>(),
        stream);

    CK(cudaStreamSynchronize(stream));
    return {ref_mid, fused_packed, fused_scales};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
debug_compare_ffn1_fp4_blockscale_only(
    torch::Tensor w_t,
    torch::Tensor act_t,
    double norm_constant) {
    TORCH_CHECK(w_t.is_cuda() && act_t.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(w_t.scalar_type() == torch::kBFloat16 &&
                act_t.scalar_type() == torch::kBFloat16,
                "inputs must be bf16");
    TORCH_CHECK(w_t.dim() == 2 && act_t.dim() == 2,
                "expected w[N,K], act[M,K]");
    auto w_bf16 = w_t.contiguous();
    auto act_bf16 = act_t.contiguous();
    int N = (int)w_bf16.size(0);
    int K = (int)w_bf16.size(1);
    int M = (int)act_bf16.size(0);
    TORCH_CHECK((int)act_bf16.size(1) == K, "K mismatch");
    TORCH_CHECK(N % 16 == 0, "FFN output dim must be divisible by 16, got ", N);

    ensure_gemm_runtime();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto w_fp8 = w_bf16.to(torch::kFloat8_e4m3fn).contiguous();
    auto act_fp8 = act_bf16.to(torch::kFloat8_e4m3fn).contiguous();

    auto ref_mid = torch::zeros({M, N}, opts_bf16);
    FP8Gemm ref_gemm;
    ref_gemm.setup(g_ltH, M, N, K, g_scaleA, g_scaleB, g_gemm_ws, g_gemm_ws_bytes, 0);
    ref_gemm.run(g_ltH,
                 w_fp8.data_ptr(),
                 act_fp8.data_ptr(),
                 ref_mid.data_ptr(),
                 g_gemm_ws, g_gemm_ws_bytes,
                 1.f, 0.f, stream,
                 nullptr);

    int rows_padded = fp4_scale_rows(M);
    int scale_cols = fp4_scale_cols(N);
    auto fused_src = torch::zeros({M, N}, opts_bf16);
    auto fused_packed = torch::zeros({rows_padded, N / 2}, opts_u8);
    auto fused_scales = torch::empty({rows_padded, scale_cols}, opts_u8);
    CK(cudaMemsetAsync(fused_scales.data_ptr(), fp4_scale_one_byte(), fused_scales.nbytes(), stream));
    auto normconst = torch::full({1}, (float)norm_constant, opts_f32);

    CutlassFfn1Fp4BlockScaleOnly fused;
    fused.setup(M, N, K);
    fused.run(
        w_fp8.data_ptr(),
        act_fp8.data_ptr(),
        (__nv_bfloat16*)fused_src.data_ptr(),
        fused_packed.data_ptr(),
        fused_scales.data_ptr(),
        normconst.data_ptr<float>(),
        stream);

    CK(cudaStreamSynchronize(stream));
    return {ref_mid, fused_packed, fused_scales};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
debug_compare_ffn1_fp4_blockscale_only_e8m0(
    torch::Tensor w_t,
    torch::Tensor act_t,
    double norm_constant) {
    TORCH_CHECK(w_t.is_cuda() && act_t.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(w_t.scalar_type() == torch::kBFloat16 &&
                act_t.scalar_type() == torch::kBFloat16,
                "inputs must be bf16");
    TORCH_CHECK(w_t.dim() == 2 && act_t.dim() == 2,
                "expected w[N,K], act[M,K]");
    auto w_bf16 = w_t.contiguous();
    auto act_bf16 = act_t.contiguous();
    int N = (int)w_bf16.size(0);
    int K = (int)w_bf16.size(1);
    int M = (int)act_bf16.size(0);
    TORCH_CHECK((int)act_bf16.size(1) == K, "K mismatch");
    TORCH_CHECK(N % 16 == 0, "FFN output dim must be divisible by 16, got ", N);

    ensure_gemm_runtime();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto w_fp8 = w_bf16.to(torch::kFloat8_e4m3fn).contiguous();
    auto act_fp8 = act_bf16.to(torch::kFloat8_e4m3fn).contiguous();

    auto ref_mid = torch::zeros({M, N}, opts_bf16);
    FP8Gemm ref_gemm;
    ref_gemm.setup(g_ltH, M, N, K, g_scaleA, g_scaleB, g_gemm_ws, g_gemm_ws_bytes, 0);
    ref_gemm.run(g_ltH,
                 w_fp8.data_ptr(),
                 act_fp8.data_ptr(),
                 ref_mid.data_ptr(),
                 g_gemm_ws, g_gemm_ws_bytes,
                 1.f, 0.f, stream,
                 nullptr);

    int rows_padded = fp4_scale_rows(M);
    int scale_cols = fp4_scale_cols(N);
    auto fused_src = torch::zeros({M, N}, opts_bf16);
    auto fused_packed = torch::zeros({rows_padded, N / 2}, opts_u8);
    auto fused_scales = torch::zeros({rows_padded, scale_cols}, opts_u8);
    auto normconst = torch::full({1}, (float)norm_constant, opts_f32);

    CutlassFfn1Fp4BlockScaleOnlyE8M0 fused;
    fused.setup(M, N, K);
    fused.run(
        w_fp8.data_ptr(),
        act_fp8.data_ptr(),
        (__nv_bfloat16*)fused_src.data_ptr(),
        fused_packed.data_ptr(),
        fused_scales.data_ptr(),
        normconst.data_ptr<float>(),
        stream);

    CK(cudaStreamSynchronize(stream));
    return {ref_mid, fused_packed, fused_scales};
}
#endif

// ============================================================
// PYBIND11 module
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &engine_init, "Initialize Kairos engine",
          py::arg("D"), py::arg("NH"), py::arg("HD"), py::arg("NL"),
          py::arg("FFN"), py::arg("CA_K_DIM"), py::arg("max_seq"), py::arg("ctx"),
          py::arg("gdn_key_dim") = 0, py::arg("gdn_value_dim") = 0);
    m.def("prepare_seq", &engine_prepare_seq, "Pre-build GEMMs for a specific seq_len");
    m.def("prepare_ca", &engine_prepare_ca, "Pre-build CA GEMMs for a specific ca_len");
    m.def("load", &engine_load, "Load one layer's weights");
    m.def("load_gdn", &engine_load_gdn, "Load GDN weights for one linear attention layer");
    m.def("reset_gdn_state", &engine_reset_gdn_state, "Reset GDN recurrent state (call between inferences)");
    m.def("build_ca_kv_cache", &engine_build_ca_kv_cache,
          "Build CA K/V cache inside the engine from encoder states");
    m.def("set_ca_kv_cache", &engine_set_ca_kv_cache,
          "Write pre-computed CA K/V for one layer into the engine cache");
    m.def("set_ca_kv_cache_valid", &engine_set_ca_kv_cache_valid,
          "Mark CA K/V cache as valid with actual context length",
          py::arg("valid"), py::arg("actual_ctx"));
    m.def("forward", &engine_forward,
          "Forward pass through a range of quadratic layers (skips GatedDeltaNet layers).",
          py::arg("x_t"), py::arg("enc_t"), py::arg("temb_t"), py::arg("ssts_t"),
          py::arg("rope_t"), py::arg("ca_len"),
          py::arg("start_layer") = 0, py::arg("end_layer") = -1);
    m.def("invalidate_ca_kv_cache", &engine_invalidate_ca_kv_cache,
          "Invalidate CA K/V cache (call when text encoder output changes)");
    m.def("nccl_get_unique_id", &engine_nccl_get_unique_id,
          "Generate NCCL unique id (call on rank 0)");
    m.def("nccl_comm_init", &engine_nccl_comm_init,
          "Initialize NCCL comm from broadcast id");
    m.def("set_profile_mode", &set_profile_mode,
          "0=all,1=gemm,2=ewise,3=sdpa,4=no_sdpa,5=no_gemm,6=no_ewise");
    m.def("set_torch_sdpa", &set_torch_sdpa,
          "Use PyTorch SDPA instead of cuDNN SDPA (for correctness debugging)");
    m.def("set_sdpa_backend", &set_sdpa_backend,
          "Set engine SA backend: cudnn, torch, sage3_py, or sage3_cpp");
    m.def("get_sdpa_backend", &get_sdpa_backend,
          "Get current engine SA backend");
    m.def("set_skip_ca", &set_skip_ca,
          "Skip all CA operations in engine (let Python handle cross-attention)");
    m.def("set_use_graph", &set_use_graph, "Enable/disable CUDA graph capture");
    m.def("get_buf", &get_buf, "Get a clone of an internal buffer by name");
    m.def("set_gdn_cublas_chunk", &engine_set_gdn_cublas_chunk,
          "Toggle cuBLASLt chunk GDN (true) vs fused recurrent (false)",
          py::arg("enable") = true);
    m.def("set_force_bf16_gemms", &set_force_bf16_gemms,
          "Force all GEMMs to BF16 (no FP8) for quality debugging");
    m.def("run_chunk_fwd_o", [](torch::Tensor q_t, torch::Tensor k_t,
                                  torch::Tensor w_t, torch::Tensor u_t,
                                  torch::Tensor h_snap_t, torch::Tensor gcum_t,
                                  int64_t T, int64_t NH, int64_t K, int64_t V,
                                  double scale) {
        // Run k_gdn_chunk_fwd_o with given inputs
        int NV_blocks = (int)V / GDN_BV;
        auto o_out = torch::zeros({T, NH, V}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        int NC_local = ((int)T + GDN_CHUNK_BT - 1) / GDN_CHUNK_BT;
        size_t fwd_o_smem = GDN_CHUNK_BT * GDN_BV * sizeof(float) + GDN_CHUNK_BT * sizeof(float);
        k_gdn_chunk_fwd_o<<<dim3(NV_blocks, NC_local, (int)NH), 128, fwd_o_smem, stream>>>(
            (__nv_bfloat16*)q_t.data_ptr(), (__nv_bfloat16*)k_t.data_ptr(),
            (float*)w_t.data_ptr(), (float*)u_t.data_ptr(),
            (__nv_bfloat16*)h_snap_t.data_ptr(), (float*)gcum_t.data_ptr(),
            (__nv_bfloat16*)o_out.data_ptr(),
            (int)T, (int)NH, (int)K, (int)V, (float)scale);
        CK(cudaStreamSynchronize(stream));
        return o_out;
    }, "Run chunk forward output kernel");
    m.def("run_chunk_h_simple", [](torch::Tensor k_t,
                                    torch::Tensor w_t, torch::Tensor u_t,
                                    torch::Tensor gcum_t,
                                    int64_t T, int64_t NC, int64_t NH, int64_t K, int64_t V) {
        auto h_out = torch::zeros({(NC+1) * NH * K * V}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto state = torch::zeros({NH * K * V}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        int NV_blocks = (int)V / GDN_BV;
        k_gdn_chunk_h_simple<<<dim3(NV_blocks, (int)NH), 32, 0, stream>>>(
            (__nv_bfloat16*)k_t.data_ptr(),
            (float*)w_t.data_ptr(), (float*)u_t.data_ptr(), (float*)gcum_t.data_ptr(),
            (__nv_bfloat16*)h_out.data_ptr(), (float*)state.data_ptr(), nullptr, nullptr,
            (int)T, (int)NC, (int)NH, (int)K, (int)V);
        CK(cudaStreamSynchronize(stream));
        return std::make_tuple(h_out.reshape({(long)(NC+1), (long)NH, (long)K, (long)V}), state.reshape({(long)NH, (long)K, (long)V}));
    }, "Run simple chunk_h kernel");
    m.def("run_chunk_prepare", [](torch::Tensor k_t, torch::Tensor v_t,
                                   torch::Tensor g_t, torch::Tensor beta_t,
                                   int64_t T, int64_t NH, int64_t K, int64_t V) {
        // Run k_gdn_chunk_prepare and return w, u, gcum as tensors
        const int BT = GDN_CHUNK_BT;
        int NC = ((int)T + BT - 1) / BT;
        size_t w_elems = (size_t)NC * NH * BT * K;
        size_t u_elems = (size_t)NC * NH * BT * V;
        size_t g_elems = (size_t)NC * NH * BT;
        auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto w_out = torch::zeros({(long)w_elems}, opts_f32);
        auto u_out = torch::zeros({(long)u_elems}, opts_f32);
        auto g_out = torch::zeros({(long)g_elems}, opts_f32);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        size_t smem = gdn_chunk_prepare_smem((int)K, (int)V);
        if (smem > 48 * 1024)
            cudaFuncSetAttribute(k_gdn_chunk_prepare,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        k_gdn_chunk_prepare<<<dim3(NC, (int)NH), 64, smem, stream>>>(
            (__nv_bfloat16*)k_t.data_ptr(), (__nv_bfloat16*)v_t.data_ptr(),
            (float*)g_t.data_ptr(), (float*)beta_t.data_ptr(),
            (float*)w_out.data_ptr(), (float*)u_out.data_ptr(), (float*)g_out.data_ptr(),
            (int)T, (int)NH, (int)K, (int)V);
        CK(cudaStreamSynchronize(stream));
        return std::make_tuple(
            w_out.reshape({NC, (long)NH, BT, (long)K}),
            u_out.reshape({NC, (long)NH, BT, (long)V}),
            g_out.reshape({NC, (long)NH, BT}));
    }, "Run chunk prepare kernel and return w, u, gcum tensors");
    m.def("run_sage3_cpp", [](torch::Tensor q_t, torch::Tensor k_t, torch::Tensor v_t) {
        TORCH_CHECK(g_ready, "engine not initialized");
        TORCH_CHECK(q_t.is_cuda() && k_t.is_cuda() && v_t.is_cuda(), "inputs must be CUDA");
        TORCH_CHECK(q_t.scalar_type() == torch::kBFloat16 &&
                    k_t.scalar_type() == torch::kBFloat16 &&
                    v_t.scalar_type() == torch::kBFloat16, "inputs must be bf16");
        TORCH_CHECK(q_t.dim() == 3 && k_t.dim() == 3 && v_t.dim() == 3, "inputs must be [T,NH,HD]");
        int T = (int)q_t.size(0);
        int NH = (int)q_t.size(1);
        int HD = (int)q_t.size(2);
        TORCH_CHECK((int)k_t.size(0) == T && (int)k_t.size(1) == NH && (int)k_t.size(2) == HD, "k shape mismatch");
        TORCH_CHECK((int)v_t.size(0) == T && (int)v_t.size(1) == NH && (int)v_t.size(2) == HD, "v shape mismatch");
        auto o_out = torch::zeros({T, NH, HD}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        sageattn3_cpp_bf16(
            q_t.data_ptr(), k_t.data_ptr(), v_t.data_ptr(),
            o_out.data_ptr(), nullptr,
            T, T, NH, HD, stream);
        CK(cudaStreamSynchronize(stream));
        return o_out;
    }, "Run native SageAttention3 path directly");
    m.def("run_fp4_gemm", [](torch::Tensor w_t, torch::Tensor act_t) {
        auto out = run_fp4_gemm_rowwise(w_t, act_t);
        CK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
        return out;
    }, "Run standalone SM120 rowwise FP4 GEMM [N,K] x [M,K] -> [M,N]");
    m.def("quantize_fp4_rowwise", [](torch::Tensor in_t) {
        auto [packed, scales, swizzled] = quantize_fp4_rowwise_bf16(in_t);
        CK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
        return std::make_tuple(packed, scales, swizzled);
    }, "Quantize a bf16 [rows,cols] matrix into rowwise FP4 packed data and scales");
    m.def("debug_quantize_fp4_colmajor_custom", [](torch::Tensor in_t) {
        auto [packed, scales, swizzled] = quantize_fp4_colmajor_bf16(in_t);
        CK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
        return std::make_tuple(packed, scales, swizzled);
    }, "Debug helper: current custom ColumnMajor FP4 packer");
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    m.def("debug_quantize_fp4_rowmajor_cutlass_host", [](torch::Tensor in_t) {
        TORCH_CHECK(in_t.dim() == 2 && in_t.scalar_type() == torch::kBFloat16, "input must be bf16 rank-2");
        int M = (int)in_t.size(0), K = (int)in_t.size(1);
        CutlassFp4RowwiseGemm gemm;
        gemm.setup(M, 128, K);
        auto [packed, scales] = pack_cutlass_operand_host_tensors<CutlassFp4RowwiseGemm::ElementA>(
            in_t, gemm.layout_A, gemm.layout_SFA, M, K);
        return std::make_tuple(packed, scales);
    }, "Debug helper: CUTLASS-native RowMajor FP4 host pack");
    m.def("debug_quantize_fp4_colmajor_cutlass_host", [](torch::Tensor in_t) {
        TORCH_CHECK(in_t.dim() == 2 && in_t.scalar_type() == torch::kBFloat16, "input must be bf16 rank-2");
        int N = (int)in_t.size(0), K = (int)in_t.size(1);
        CutlassFp4RowwiseGemm gemm;
        gemm.setup(128, N, K);
        auto [packed, scales] = pack_cutlass_operand_host_tensors<CutlassFp4RowwiseGemm::ElementB>(
            in_t, gemm.layout_B, gemm.layout_SFB, N, K);
        return std::make_tuple(packed, scales);
    }, "Debug helper: CUTLASS-native ColumnMajor FP4 host pack");
    m.def("debug_ffn1_ffn2_scale_layout_maps", [](int64_t seq, int64_t d, int64_t ffn) {
        return debug_ffn1_ffn2_scale_layout_maps(seq, d, ffn);
    }, "Debug helper: compare raw scale-slot maps for FFN1 fused SFD vs FFN2 activation SFA");
    m.def("debug_compare_ffn1_fp4_producers",
          [](torch::Tensor w_t, torch::Tensor act_t, torch::Tensor bias_t) {
              return debug_compare_ffn1_fp4_producers(w_t, act_t, bias_t);
          },
          "Debug helper: compare fused FFN1 FP4 producer vs engine-style FP8 GEMM + SiLU->FP4 reference");
    m.def("debug_compare_ffn1_fp4_producers_norm",
          [](torch::Tensor w_t, torch::Tensor act_t, torch::Tensor bias_t, double norm_constant) {
              return debug_compare_ffn1_fp4_producers_norm(w_t, act_t, bias_t, norm_constant);
          },
          "Debug helper: compare fused FFN1 FP4 producer vs reference with custom norm_constant");
    m.def("debug_compare_ffn1_fp4_identity_producer",
          [](torch::Tensor w_t, torch::Tensor act_t, torch::Tensor bias_t, double norm_constant) {
              return debug_compare_ffn1_fp4_identity_producer(w_t, act_t, bias_t, norm_constant);
          },
          "Debug helper: compare identity-activation fused FFN1 FP4 producer vs raw GEMM+bias reference");
    m.def("debug_compare_ffn1_fp4_blockscale_only",
          [](torch::Tensor w_t, torch::Tensor act_t, double norm_constant) {
              return debug_compare_ffn1_fp4_blockscale_only(w_t, act_t, norm_constant);
          },
          "Debug helper: compare blockscale-only FP4 producer vs raw GEMM reference");
    m.def("debug_compare_ffn1_fp4_blockscale_only_e8m0",
          [](torch::Tensor w_t, torch::Tensor act_t, double norm_constant) {
              return debug_compare_ffn1_fp4_blockscale_only_e8m0(w_t, act_t, norm_constant);
          },
          "Debug helper: compare blockscale-only FP4 producer with UE8M0 scales vs raw GEMM reference");
#endif
    m.def("run_fp4_gemm_packed",
          [](torch::Tensor w_packed, torch::Tensor w_swizzled,
             torch::Tensor act_packed, torch::Tensor act_swizzled,
             int64_t M, int64_t N, int64_t K) {
              auto out = run_fp4_gemm_rowwise_packed(
                  w_packed, w_swizzled, act_packed, act_swizzled, M, N, K);
              CK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
              return out;
          },
          "Run standalone SM120 rowwise FP4 GEMM on pre-quantized packed operands");
    m.def("run_gdn_recurrent", [](torch::Tensor q_t, torch::Tensor k_t, torch::Tensor v_t,
                                   torch::Tensor g_t, torch::Tensor beta_t, double scale) {
        TORCH_CHECK(g_ready, "engine not initialized");
        TORCH_CHECK(q_t.is_cuda() && k_t.is_cuda() && v_t.is_cuda() &&
                    g_t.is_cuda() && beta_t.is_cuda(), "inputs must be CUDA");
        TORCH_CHECK(q_t.scalar_type() == torch::kBFloat16 &&
                    k_t.scalar_type() == torch::kBFloat16 &&
                    v_t.scalar_type() == torch::kBFloat16, "q/k/v must be bf16");
        TORCH_CHECK(g_t.scalar_type() == torch::kFloat32 &&
                    beta_t.scalar_type() == torch::kFloat32, "g and beta must be f32");
        TORCH_CHECK(q_t.dim() == 3 && k_t.dim() == 3 && v_t.dim() == 3, "q/k/v must be [T,NH,K/V]");
        TORCH_CHECK(g_t.dim() == 2 && beta_t.dim() == 2, "g/beta must be [T,NH]");
        int T = (int)q_t.size(0);
        int NH = (int)q_t.size(1);
        int K = (int)q_t.size(2);
        int V = (int)v_t.size(2);
        TORCH_CHECK((int)k_t.size(0) == T && (int)k_t.size(1) == NH && (int)k_t.size(2) == K, "k shape mismatch");
        TORCH_CHECK((int)v_t.size(0) == T && (int)v_t.size(1) == NH, "v shape mismatch");
        TORCH_CHECK((int)g_t.size(0) == T && (int)g_t.size(1) == NH, "g shape mismatch");
        TORCH_CHECK((int)beta_t.size(0) == T && (int)beta_t.size(1) == NH, "beta shape mismatch");
        TORCH_CHECK(K % GDN_BK == 0 && V % GDN_BV == 0, "K/V must align with GDN tiles");
        auto o_out = torch::zeros({T, NH, V}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto state = torch::zeros({NH, K, V}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        int NV_blocks = V / GDN_BV;
        int total_tiles = NH * NV_blocks;
        int zero = 0;
        CK(cudaMemcpyToSymbolAsync(g_gdn_work_counter, &zero, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
        int n_persistent = std::min(total_tiles, g_nsm * 8);
        k_gdn_recurrent<<<n_persistent, 32, 0, stream>>>(
            (__nv_bfloat16*)q_t.data_ptr(), (__nv_bfloat16*)k_t.data_ptr(), (__nv_bfloat16*)v_t.data_ptr(),
            (float*)g_t.data_ptr(), (float*)beta_t.data_ptr(),
            (__nv_bfloat16*)o_out.data_ptr(), (float*)state.data_ptr(),
            T, NH, K, V, (float)scale, total_tiles);
        CK(cudaStreamSynchronize(stream));
        return std::make_tuple(o_out, state);
    }, "Run fused GDN recurrent kernel directly");
}
