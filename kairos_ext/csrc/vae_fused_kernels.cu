// VAE fused kernels for Kairos: RMSNorm+SiLU fusion
// Supports bf16 and fp16 I/O, fp32 internal math.
//
// Two kernel variants:
//   k_rms_silu_cf:  channels-FIRST [B,C,T,H,W] — 2-pass, coalesced spatial reads
//   k_rms_silu_warp: [N,C] contiguous — 1-pass, warp-per-position
//
// The channels-first kernel is preferred since VAE uses channels-first layout,
// avoiding permute overhead.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

// ---- type-generic load/store ----
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
template<typename T> __device__ __forceinline__ T from_float(float x);
template<> __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// ============================================================
// CHANNELS-FIRST KERNELS (native for VAE — no permute needed)
// ============================================================
// Each thread handles one spatial position.
// Reads are coalesced: adjacent threads read adjacent spatial positions
// within the same channel plane (contiguous in channels-first layout).
// 2-pass: pass1 computes sumsq, pass2 normalizes+silu+writes.
// Pass2 benefits from L2 cache (data just read in pass1).

template<typename T>
__global__ void k_rms_silu_cf(
    const T* __restrict__ x,   // [B, C, T, H, W] contiguous
    T* __restrict__ y,
    const float* __restrict__ gamma, // [C]
    int N_spatial,      // B * T * H * W
    int C,
    int spatial_size,   // T * H * W (stride between channels)
    float scale)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= N_spatial) return;

    // Compute batch index and offset within batch
    int batch = pos / spatial_size;
    int spatial = pos % spatial_size;
    size_t base = (size_t)batch * C * spatial_size + spatial;

    // Pass 1: compute sum of squares over channels
    float sumsq = 0.f;
    for (int c = 0; c < C; c++) {
        float v = to_float(x[base + (size_t)c * spatial_size]);
        sumsq += v * v;
    }
    float inv_rms = rsqrtf(sumsq + 1e-12f) * scale;

    // Pass 2: normalize + silu + write (data likely in L2 from pass 1)
    for (int c = 0; c < C; c++) {
        float v = to_float(x[base + (size_t)c * spatial_size]);
        float norm = v * inv_rms * gamma[c];
        float sig = 1.f / (1.f + __expf(-norm));
        y[base + (size_t)c * spatial_size] = from_float<T>(norm * sig);
    }
}

template<typename T>
__global__ void k_add_rms_silu_cf(
    const T* __restrict__ xa,
    const T* __restrict__ xb,
    T* __restrict__ y,
    const float* __restrict__ gamma,
    int N_spatial, int C, int spatial_size, float scale)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= N_spatial) return;

    int batch = pos / spatial_size;
    int spatial = pos % spatial_size;
    size_t base = (size_t)batch * C * spatial_size + spatial;

    // Pass 1: add + sumsq
    float sumsq = 0.f;
    for (int c = 0; c < C; c++) {
        size_t off = base + (size_t)c * spatial_size;
        float v = to_float(xa[off]) + to_float(xb[off]);
        sumsq += v * v;
    }
    float inv_rms = rsqrtf(sumsq + 1e-12f) * scale;

    // Pass 2: add again + normalize + silu + write
    for (int c = 0; c < C; c++) {
        size_t off = base + (size_t)c * spatial_size;
        float v = to_float(xa[off]) + to_float(xb[off]);
        float norm = v * inv_rms * gamma[c];
        float sig = 1.f / (1.f + __expf(-norm));
        y[off] = from_float<T>(norm * sig);
    }
}

// ============================================================
// DISPATCH
// ============================================================

// rms_silu on channels-first [B, C, T, H, W] contiguous
torch::Tensor rms_silu_fused_cf(torch::Tensor x, torch::Tensor gamma_param) {
    TORCH_CHECK(x.is_cuda() && gamma_param.is_cuda(), "cuda only");
    TORCH_CHECK(x.dim() == 5, "x must be [B, C, T, H, W]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto dtype = x.scalar_type();
    int64_t B = x.size(0), C = x.size(1), T_ = x.size(2), H = x.size(3), W = x.size(4);
    int spatial_size = (int)(T_ * H * W);
    int N_spatial = (int)(B * spatial_size);

    // Flatten gamma to [C] fp32
    auto gamma = gamma_param.contiguous().view({-1});
    TORCH_CHECK(gamma.numel() == C, "gamma size mismatch");
    if (gamma.scalar_type() != torch::kFloat)
        gamma = gamma.to(torch::kFloat);

    float scale = sqrtf((float)C);
    auto y = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = 256;
    int blocks = (N_spatial + threads - 1) / threads;
    const float* gp = gamma.data_ptr<float>();

    if (dtype == torch::kBFloat16) {
        auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
        auto yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
        k_rms_silu_cf<<<blocks, threads, 0, stream>>>(
            xp, yp, gp, N_spatial, (int)C, spatial_size, scale);
    } else if (dtype == torch::kHalf) {
        auto xp = reinterpret_cast<const __half*>(x.data_ptr());
        auto yp = reinterpret_cast<__half*>(y.data_ptr());
        k_rms_silu_cf<<<blocks, threads, 0, stream>>>(
            xp, yp, gp, N_spatial, (int)C, spatial_size, scale);
    } else {
        TORCH_CHECK(false, "need bf16 or fp16, got ", dtype);
    }
    return y;
}

// add_rms_silu on channels-first
torch::Tensor add_rms_silu_fused_cf(
    torch::Tensor xa, torch::Tensor xb, torch::Tensor gamma_param)
{
    TORCH_CHECK(xa.is_cuda() && xb.is_cuda() && gamma_param.is_cuda(), "cuda only");
    TORCH_CHECK(xa.dim() == 5 && xb.dim() == 5, "5D tensors");
    TORCH_CHECK(xa.scalar_type() == xb.scalar_type(), "dtype mismatch");
    TORCH_CHECK(xa.sizes() == xb.sizes(), "shape mismatch");
    TORCH_CHECK(xa.is_contiguous() && xb.is_contiguous(), "contiguous");

    auto dtype = xa.scalar_type();
    int64_t B = xa.size(0), C = xa.size(1), T_ = xa.size(2), H = xa.size(3), W = xa.size(4);
    int spatial_size = (int)(T_ * H * W);
    int N_spatial = (int)(B * spatial_size);

    auto gamma = gamma_param.contiguous().view({-1});
    TORCH_CHECK(gamma.numel() == C, "gamma size mismatch");
    if (gamma.scalar_type() != torch::kFloat)
        gamma = gamma.to(torch::kFloat);

    float scale = sqrtf((float)C);
    auto y = torch::empty_like(xa);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = 256;
    int blocks = (N_spatial + threads - 1) / threads;
    const float* gp = gamma.data_ptr<float>();

    if (dtype == torch::kBFloat16) {
        auto xap = reinterpret_cast<const __nv_bfloat16*>(xa.data_ptr());
        auto xbp = reinterpret_cast<const __nv_bfloat16*>(xb.data_ptr());
        auto yp  = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
        k_add_rms_silu_cf<<<blocks, threads, 0, stream>>>(
            xap, xbp, yp, gp, N_spatial, (int)C, spatial_size, scale);
    } else if (dtype == torch::kHalf) {
        auto xap = reinterpret_cast<const __half*>(xa.data_ptr());
        auto xbp = reinterpret_cast<const __half*>(xb.data_ptr());
        auto yp  = reinterpret_cast<__half*>(y.data_ptr());
        k_add_rms_silu_cf<<<blocks, threads, 0, stream>>>(
            xap, xbp, yp, gp, N_spatial, (int)C, spatial_size, scale);
    } else {
        TORCH_CHECK(false, "need bf16 or fp16");
    }
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_silu_fused_cf", &rms_silu_fused_cf,
          "Fused RMSNorm + SiLU on channels-first [B,C,T,H,W] (bf16/fp16)");
    m.def("add_rms_silu_fused_cf", &add_rms_silu_fused_cf,
          "Fused (a+b) -> RMSNorm -> SiLU on channels-first [B,C,T,H,W] (bf16/fp16)");
}
