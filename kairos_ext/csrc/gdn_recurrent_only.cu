#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>

namespace {

#ifndef GDN_BK
#define GDN_BK 8
#endif
#ifndef GDN_BV
#define GDN_BV 8
#endif
#ifndef GDN_PERSISTENT_BLOCKS_PER_SM
#define GDN_PERSISTENT_BLOCKS_PER_SM 8
#endif
#ifndef GDN_WARPS_PER_CTA
#define GDN_WARPS_PER_CTA 1
#endif
#ifndef GDN_PIPELINE_QK
#define GDN_PIPELINE_QK 0
#endif

inline void ck(cudaError_t err, const char* file, int line) {
    TORCH_CHECK(err == cudaSuccess, "CUDA error ", cudaGetErrorString(err), " at ", file, ":", line);
}
#define CK(x) ck((x), __FILE__, __LINE__)

__device__ __forceinline__ float warp_reduce(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

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

static __device__ int g_gdn_work_counter;

__global__ void __launch_bounds__(32 * GDN_WARPS_PER_CTA) k_gdn_recurrent_only(
    const __nv_bfloat16* __restrict__ q,   // [T, NH, K]
    const __nv_bfloat16* __restrict__ k,   // [T, NH, K]
    const __nv_bfloat16* __restrict__ v,   // [T, NH, V]
    const float* __restrict__ g,           // [T, NH]
    const float* __restrict__ beta,        // [T, NH]
    __nv_bfloat16* __restrict__ o,         // [T, NH, V]
    float* __restrict__ state,             // [NH, K, V]
    int T, int NH, int K, int V, int total_work) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int k_off = lane * GDN_BK;

#if GDN_WARPS_PER_CTA == 1
#if GDN_PIPELINE_QK
    __shared__ __align__(16) __nv_bfloat16 s_q_pipe[2][256];
    __shared__ __align__(16) __nv_bfloat16 s_k_pipe[2][256];
#endif
    while (true) {
        int tile_id;
        if (lane == 0) tile_id = atomicAdd(&g_gdn_work_counter, 1);
        tile_id = __shfl_sync(0xffffffff, tile_id, 0);
        if (tile_id >= total_work) break;

        int nv_tiles = V / GDN_BV;
        int h = tile_id / nv_tiles;
        int bv = tile_id % nv_tiles;
        int v_off = bv * GDN_BV;

        float hr[GDN_BK][GDN_BV];
        float* st = state + (size_t)h * K * V;
        #pragma unroll
        for (int ki = 0; ki < GDN_BK; ki++) {
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++) {
                hr[ki][vi] = st[(size_t)(k_off + ki) * V + v_off + vi];
            }
        }

#if GDN_PIPELINE_QK
        size_t qk_base0 = (size_t)h * K;
        __pipeline_memcpy_async(reinterpret_cast<void*>(s_q_pipe[0] + k_off),
                                reinterpret_cast<const void*>(q + qk_base0 + k_off),
                                sizeof(int4));
        __pipeline_memcpy_async(reinterpret_cast<void*>(s_k_pipe[0] + k_off),
                                reinterpret_cast<const void*>(k + qk_base0 + k_off),
                                sizeof(int4));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncwarp();
#endif

        for (int t = 0; t < T; t++) {
            size_t v_base = ((size_t)t * NH + h) * V;

            float my_k[GDN_BK];
            float my_q[GDN_BK];
#if GDN_PIPELINE_QK
            int curr_buf = t & 1;
            int next_buf = curr_buf ^ 1;
            if (t + 1 < T) {
                size_t qk_next_base = ((size_t)(t + 1) * NH + h) * K;
                __pipeline_memcpy_async(reinterpret_cast<void*>(s_q_pipe[next_buf] + k_off),
                                        reinterpret_cast<const void*>(q + qk_next_base + k_off),
                                        sizeof(int4));
                __pipeline_memcpy_async(reinterpret_cast<void*>(s_k_pipe[next_buf] + k_off),
                                        reinterpret_cast<const void*>(k + qk_next_base + k_off),
                                        sizeof(int4));
                __pipeline_commit();
            }
            load_bf16x8_to_f32(s_k_pipe[curr_buf] + k_off, my_k);
            load_bf16x8_to_f32(s_q_pipe[curr_buf] + k_off, my_q);
#else
            size_t qk_base = ((size_t)t * NH + h) * K;
            load_bf16x8_to_f32(k + qk_base + k_off, my_k);
            load_bf16x8_to_f32(q + qk_base + k_off, my_q);
#endif
            float gt = (lane == 0) ? g[(size_t)t * NH + h] : 0.0f;
            float bt = (lane == 0) ? beta[(size_t)t * NH + h] : 0.0f;
            gt = __shfl_sync(0xffffffff, gt, 0);
            bt = __shfl_sync(0xffffffff, bt, 0);
            float qk_dot = 0.0f;
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++) {
                qk_dot += my_q[ki] * my_k[ki];
            }
            qk_dot = warp_reduce(qk_dot);
            qk_dot = __shfl_sync(0xffffffff, qk_dot, 0);

            float decay = __expf(gt);
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++) {
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++) {
                    hr[ki][vi] *= decay;
                }
            }

            float hk[GDN_BV];
            float hq[GDN_BV];
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++) {
                float acc_k = 0.0f;
                float acc_q = 0.0f;
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++) {
                    float hval = hr[ki][vi];
                    acc_k += hval * my_k[ki];
                    acc_q += hval * my_q[ki];
                }
                hk[vi] = acc_k;
                hq[vi] = acc_q;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++) {
                    hk[vi] += __shfl_down_sync(0xffffffff, hk[vi], off);
                    hq[vi] += __shfl_down_sync(0xffffffff, hq[vi], off);
                }
            }
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++) {
                hk[vi] = __shfl_sync(0xffffffff, hk[vi], 0);
                hq[vi] = __shfl_sync(0xffffffff, hq[vi], 0);
            }

            float v_vals[GDN_BV];
            load_bf16x8_to_f32(v + v_base + v_off, v_vals);
            float ov[GDN_BV];
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++) {
                float vn = bt * (v_vals[vi] - hk[vi]);
                ov[vi] = hq[vi] + qk_dot * vn;
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++) {
                    hr[ki][vi] += my_k[ki] * vn;
                }
            }
            if (lane == 0) {
                size_t o_base = ((size_t)t * NH + h) * V;
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++) {
                    o[o_base + v_off + vi] = __float2bfloat16(ov[vi]);
                }
            }
#if GDN_PIPELINE_QK
            if (t + 1 < T) {
                __pipeline_wait_prior(0);
                __syncwarp();
            }
#endif
        }

        #pragma unroll
        for (int ki = 0; ki < GDN_BK; ki++) {
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++) {
                st[(size_t)(k_off + ki) * V + v_off + vi] = hr[ki][vi];
            }
        }
    }
#else
    __shared__ int s_group_id;
    __shared__ __align__(16) __nv_bfloat16 s_q[256];
    __shared__ __align__(16) __nv_bfloat16 s_k[256];
    __shared__ float s_g;
    __shared__ float s_beta;

    int nv_tiles = V / GDN_BV;
    int groups_per_head = (nv_tiles + GDN_WARPS_PER_CTA - 1) / GDN_WARPS_PER_CTA;

    while (true) {
        if (threadIdx.x == 0) s_group_id = atomicAdd(&g_gdn_work_counter, 1);
        __syncthreads();
        int group_id = s_group_id;
        if (group_id >= total_work) break;

        int h = group_id / groups_per_head;
        int bv_group = group_id % groups_per_head;
        int bv = bv_group * GDN_WARPS_PER_CTA + warp;
        bool active = bv < nv_tiles;
        int v_off = bv * GDN_BV;

        float hr[GDN_BK][GDN_BV];
        float* st = nullptr;
        if (active) {
            st = state + (size_t)h * K * V;
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++) {
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++) {
                    hr[ki][vi] = st[(size_t)(k_off + ki) * V + v_off + vi];
                }
            }
        }

        for (int t = 0; t < T; t++) {
            size_t qk_base = ((size_t)t * NH + h) * K;
            size_t v_base = ((size_t)t * NH + h) * V;

            if (warp == 0) {
                *reinterpret_cast<int4*>(s_q + k_off) =
                    *reinterpret_cast<const int4*>(q + qk_base + k_off);
                *reinterpret_cast<int4*>(s_k + k_off) =
                    *reinterpret_cast<const int4*>(k + qk_base + k_off);
                if (lane == 0) {
                    s_g = g[(size_t)t * NH + h];
                    s_beta = beta[(size_t)t * NH + h];
                }
            }
            __syncthreads();

            if (active) {
                float my_k[GDN_BK];
                float my_q[GDN_BK];
                load_bf16x8_to_f32(s_k + k_off, my_k);
                load_bf16x8_to_f32(s_q + k_off, my_q);
                float gt = s_g;
                float bt = s_beta;

                float qk_dot = 0.0f;
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++) {
                    qk_dot += my_q[ki] * my_k[ki];
                }
                qk_dot = warp_reduce(qk_dot);
                qk_dot = __shfl_sync(0xffffffff, qk_dot, 0);

                float decay = __expf(gt);
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++) {
                    #pragma unroll
                    for (int vi = 0; vi < GDN_BV; vi++) {
                        hr[ki][vi] *= decay;
                    }
                }

                float hk[GDN_BV];
                float hq[GDN_BV];
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++) {
                    float acc_k = 0.0f;
                    float acc_q = 0.0f;
                    #pragma unroll
                    for (int ki = 0; ki < GDN_BK; ki++) {
                        float hval = hr[ki][vi];
                        acc_k += hval * my_k[ki];
                        acc_q += hval * my_q[ki];
                    }
                    hk[vi] = acc_k;
                    hq[vi] = acc_q;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    #pragma unroll
                    for (int vi = 0; vi < GDN_BV; vi++) {
                        hk[vi] += __shfl_down_sync(0xffffffff, hk[vi], off);
                        hq[vi] += __shfl_down_sync(0xffffffff, hq[vi], off);
                    }
                }
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++) {
                    hk[vi] = __shfl_sync(0xffffffff, hk[vi], 0);
                    hq[vi] = __shfl_sync(0xffffffff, hq[vi], 0);
                }

                float v_vals[GDN_BV];
                load_bf16x8_to_f32(v + v_base + v_off, v_vals);
                float ov[GDN_BV];
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++) {
                    float vn = bt * (v_vals[vi] - hk[vi]);
                    ov[vi] = hq[vi] + qk_dot * vn;
                    #pragma unroll
                    for (int ki = 0; ki < GDN_BK; ki++) {
                        hr[ki][vi] += my_k[ki] * vn;
                    }
                }
                if (lane == 0) {
                    size_t o_base = ((size_t)t * NH + h) * V;
                    #pragma unroll
                    for (int vi = 0; vi < GDN_BV; vi++) {
                        o[o_base + v_off + vi] = __float2bfloat16(ov[vi]);
                    }
                }
            }
            __syncthreads();
        }

        if (active) {
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++) {
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++) {
                    st[(size_t)(k_off + ki) * V + v_off + vi] = hr[ki][vi];
                }
            }
        }
        __syncthreads();
    }
#endif
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_gdn_recurrent", [](torch::Tensor q_t,
                                  torch::Tensor k_t,
                                  torch::Tensor v_t,
                                  torch::Tensor g_t,
                                  torch::Tensor beta_t) {
        TORCH_CHECK(q_t.is_cuda() && k_t.is_cuda() && v_t.is_cuda() &&
                    g_t.is_cuda() && beta_t.is_cuda(), "inputs must be CUDA");
        TORCH_CHECK(q_t.scalar_type() == torch::kBFloat16 &&
                    k_t.scalar_type() == torch::kBFloat16 &&
                    v_t.scalar_type() == torch::kBFloat16, "q/k/v must be bf16");
        TORCH_CHECK(g_t.scalar_type() == torch::kFloat32 &&
                    beta_t.scalar_type() == torch::kFloat32, "g/beta must be f32");
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
        TORCH_CHECK(K == 256, "standalone kernel currently assumes K == 256");
        TORCH_CHECK(V % GDN_BV == 0, "V must be aligned to GDN_BV");

        auto o_out = torch::zeros({T, NH, V}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto state = torch::zeros({NH, K, V}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        cudaDeviceProp prop{};
        CK(cudaGetDeviceProperties(&prop, at::cuda::current_device()));
        int nv_tiles = V / GDN_BV;
        constexpr int warps_per_cta = GDN_WARPS_PER_CTA;
        int total_work = NH * ((nv_tiles + warps_per_cta - 1) / warps_per_cta);
        int zero = 0;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        CK(cudaMemcpyToSymbolAsync(g_gdn_work_counter, &zero, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
        int n_persistent = std::min(total_work, prop.multiProcessorCount * GDN_PERSISTENT_BLOCKS_PER_SM);
        k_gdn_recurrent_only<<<n_persistent, 32 * warps_per_cta, 0, stream>>>(
            (__nv_bfloat16*)q_t.data_ptr(), (__nv_bfloat16*)k_t.data_ptr(), (__nv_bfloat16*)v_t.data_ptr(),
            (float*)g_t.data_ptr(), (float*)beta_t.data_ptr(),
            (__nv_bfloat16*)o_out.data_ptr(), (float*)state.data_ptr(),
            T, NH, K, V, total_work);
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(stream));
        return std::make_tuple(o_out, state);
    }, "Run standalone GDN recurrent kernel");
}
