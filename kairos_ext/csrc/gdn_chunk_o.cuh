#pragma once
/**
 * gdn_chunk_o.cuh -- Output kernel for chunk-based GatedDeltaNet.
 *
 * Matching Triton's chunk_fwd_kernel_o structure:
 *   Grid: dim3(V/BV, NC, NH) — PARALLEL across chunks (not sequential!)
 *   Block: 128 threads (4 warps)
 *
 * Each block processes one (chunk, head, V-tile) combination:
 *   1. Loop over K-tiles: accumulate b_o += Q@H, b_A += Q@K^T
 *   2. Apply gating exp(g) and causal mask to b_A
 *   3. Pre-compute v_new = u - w@h from prepare output
 *   4. Output = scale * (b_o + b_A @ v_new)
 *
 * Uses warp-level computation (not MMA yet — ready for MMA upgrade).
 * The key speedup vs v2: parallel across chunks, not sequential.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef GDN_BV
#define GDN_BV 8
#endif
#ifndef GDN_BK
#define GDN_BK 8
#endif
#ifndef GDN_CHUNK_BT
#define GDN_CHUNK_BT 64
#endif

// Each warp handles a subset of tokens for the output
// 4 warps × 16 tokens per warp = 64 tokens = BT
#define FWD_O_TOKENS_PER_WARP 16

__global__ void __launch_bounds__(256)
k_gdn_chunk_fwd_o(
    const __nv_bfloat16* __restrict__ q_in,      // [T, NH, K]
    const __nv_bfloat16* __restrict__ k_in,      // [T, NH, K]
    const float*         __restrict__ w_in,       // [NC, NH, BT, K]
    const float*         __restrict__ u_in,       // [NC, NH, BT, V]
    const __nv_bfloat16* __restrict__ h_out,      // [NC+1, NH, K, V] bf16
    const float*         __restrict__ gcum_in,    // [NC, NH, BT]
    __nv_bfloat16*       __restrict__ o_out,      // [T, NH, V]
    int T, int NH, int K, int V, float scale)
{
    const int BT = GDN_CHUNK_BT;  // 64
    const int bv   = blockIdx.x;   // V-tile index
    const int c    = blockIdx.y;   // chunk index (PARALLEL!)
    const int head = blockIdx.z;   // head index
    const int tid  = threadIdx.x;  // 0..127
    const int warp_id = tid >> 5;  // 0..3
    const int lane = tid & 31;
    const int v_off = bv * GDN_BV;
    const int k_off = lane * GDN_BK;

    const int t_base = c * BT;
    const int chunk_len = min(BT, T - t_base);
    if (chunk_len <= 0) return;

    size_t chunk_w_base = ((size_t)c * NH + head) * BT * K;
    size_t chunk_u_base = ((size_t)c * NH + head) * BT * V;
    size_t chunk_g_base = ((size_t)c * NH + head) * BT;

    // Shared memory: v_new [BT, GDN_BV] + gcum [BT]
    __shared__ float s_vn[GDN_CHUNK_BT * GDN_BV];
    __shared__ float s_g[GDN_CHUNK_BT];

    // Load state snapshot h[c] into registers
    float h_tile[GDN_BK][GDN_BV];
    size_t h_base = ((size_t)c * NH + head) * (size_t)K * V;
    #pragma unroll
    for (int ki = 0; ki < GDN_BK; ki++)
        #pragma unroll
        for (int vi = 0; vi < GDN_BV; vi++)
            h_tile[ki][vi] = __bfloat162float(h_out[h_base + (size_t)(k_off + ki) * V + v_off + vi]);

    // Load gcum into SMEM (cooperative: 128 threads load 64 values)
    if (tid < BT)
        s_g[tid] = gcum_in[chunk_g_base + tid];
    __syncthreads();

    // ================================================================
    // PRE-COMPUTE v_new[j] = u[j] - w[j]@h for ALL j
    // Distribute across 4 warps: each warp handles 16 tokens
    // ================================================================
    {
        int n_warps = blockDim.x >> 5;
        for (int j = warp_id; j < chunk_len; j += n_warps) {
            float my_w[GDN_BK];
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++)
                my_w[ki] = w_in[chunk_w_base + j * K + k_off + ki];

            float wh[GDN_BV];
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++) {
                float acc = 0.0f;
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++)
                    acc += my_w[ki] * h_tile[ki][vi];
                wh[vi] = acc;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    wh[vi] += __shfl_xor_sync(0xffffffff, wh[vi], off);

            if (lane == 0) {
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    s_vn[j * GDN_BV + vi] = u_in[chunk_u_base + j * V + v_off + vi] - wh[vi];
            }
        }
    }
    __syncthreads();

    // ================================================================
    // OUTPUT: Each warp processes interleaved tokens for load balance
    // 8 warps, each handles every 8th token
    // ================================================================
    {
        int n_warps = blockDim.x >> 5;  // 8 warps with 256 threads
        for (int i = warp_id; i < chunk_len; i += n_warps) {
            int t_i = t_base + i;
            float gi = s_g[i];

            // Load Q
            float my_q[GDN_BK];
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++)
                my_q[ki] = __bfloat162float(
                    q_in[((size_t)t_i * NH + head) * K + k_off + ki]);

            // Term 1: Q @ h * exp(g)
            float decay_i = expf(gi);
            float ov[GDN_BV];
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++) {
                float acc = 0.0f;
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++)
                    acc += my_q[ki] * h_tile[ki][vi];
                ov[vi] = acc * decay_i;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    ov[vi] += __shfl_xor_sync(0xffffffff, ov[vi], off);

            // Term 2: causal attention with pre-computed v_new
            float attn[GDN_BV];
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                attn[vi] = 0.0f;

            for (int j = 0; j <= i; j++) {
                float gj = s_g[j];

                float my_k[GDN_BK];
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++)
                    my_k[ki] = __bfloat162float(
                        k_in[((size_t)(t_base + j) * NH + head) * K + k_off + ki]);

                float score = 0.0f;
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++)
                    score += my_q[ki] * my_k[ki];
                for (int off = 16; off > 0; off >>= 1)
                    score += __shfl_xor_sync(0xffffffff, score, off);

                float gate = expf(gi - gj) * score;
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    attn[vi] += gate * s_vn[j * GDN_BV + vi];
            }

            // Write output
            if (lane == 0) {
                size_t o_base = ((size_t)t_i * NH + head) * V + v_off;
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    o_out[o_base + vi] = __float2bfloat16(ov[vi] + attn[vi]);
            }
        }
    }
}
