#pragma once
/**
 * gdn_chunk_h_simple.cuh -- Simple warp-register chunk_h kernel.
 *
 * Correct port of Triton's chunk_gated_delta_rule_fwd_kernel_h order:
 *   1. Store h snapshot (h_old, before any updates)
 *   2. v_new = u - w @ h_old (using state BEFORE decay)
 *   3. Gate v_new by exp(g_last - g[i])
 *   4. Decay h: h *= exp(g_last)
 *   5. h += K^T @ gated_v_new
 *
 * Grid: dim3(V/BV, NH)  Block: 32 threads (one warp)
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

__global__ void __launch_bounds__(32)
k_gdn_chunk_h_simple(
    const __nv_bfloat16* __restrict__ k_in,      // [T, NH, K] (L2-normalized externally)
    const float*         __restrict__ w_in,      // [NC, NH, BT, K]
    const float*         __restrict__ u_in,      // [NC, NH, BT, V]
    const float*         __restrict__ gcum_in,   // [NC, NH, BT]
    __nv_bfloat16*       __restrict__ h_out,     // [NC+1, NH, K, V] snapshots, or nullptr
    float*               __restrict__ ht,        // [NH, K, V] final state, or nullptr
    const float*         __restrict__ h0,        // [NH, K, V] initial state, or nullptr
    __nv_bfloat16*       __restrict__ v_new_out, // [NC, NH, BT, V] delta values, or nullptr
    int T, int NC, int NH, int K, int V)
{
    const int BT = GDN_CHUNK_BT;
    const int bv   = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    const int v_off = bv * GDN_BV;
    const int k_off = lane * GDN_BK;

    // State tile h[GDN_BK][GDN_BV] in fp32 registers
    float hr[GDN_BK][GDN_BV];

    // Initialize from h0 or zero
    if (h0 != nullptr) {
        const float* src = h0 + (size_t)head * K * V;
        #pragma unroll
        for (int ki = 0; ki < GDN_BK; ki++)
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                hr[ki][vi] = src[(size_t)(k_off + ki) * V + v_off + vi];
    } else {
        #pragma unroll
        for (int ki = 0; ki < GDN_BK; ki++)
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                hr[ki][vi] = 0.0f;
    }

    for (int c = 0; c < NC; c++) {
        size_t chunk_w_base = ((size_t)c * NH + head) * BT * K;
        size_t chunk_u_base = ((size_t)c * NH + head) * BT * V;
        size_t chunk_g_base = ((size_t)c * NH + head) * BT;
        int t_base = c * BT;
        int chunk_len = min(BT, T - t_base);

        // ============================================================
        // STEP 1: Store h snapshot BEFORE any updates
        // ============================================================
        if (h_out != nullptr) {
            __nv_bfloat16* dst = h_out + ((size_t)c * NH + head) * K * V;
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++)
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    dst[(size_t)(k_off + ki) * V + v_off + vi] =
                        __float2bfloat16(hr[ki][vi]);
        }

        // ============================================================
        // STEP 2: Compute v_new = u - w @ h_old for ALL tokens
        // STEP 3: Gate v_new by exp(g_last - g[i])
        // Then store gated v_new temporarily.
        //
        // We process token-by-token but use h_old (UNDECAYED state)
        // for the w@h computation. h_old = hr at this point.
        // ============================================================
        float g_last = gcum_in[chunk_g_base + chunk_len - 1];

        // We need to store gated_v_new for the K^T @ v_new update.
        // Can't store BT*GDN_BV floats in registers. Process in two passes:
        // Pass 1: compute v_new, gate it, store to v_new_out or temp buffer.
        // Pass 2: load back and do K^T @ v_new.
        //
        // OR: fuse into one pass by updating h after EACH token.
        // But then w @ h would use partially-updated h (wrong!).
        //
        // The correct approach: save h_old, decay h, then process tokens
        // using h_old for w@h and updating hr incrementally.

        // Save h_old
        float h_old[GDN_BK][GDN_BV];
        #pragma unroll
        for (int ki = 0; ki < GDN_BK; ki++)
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                h_old[ki][vi] = hr[ki][vi];

        // Decay hr by exp(g_last) — this is step 4 moved up
        float decay = expf(g_last);
        #pragma unroll
        for (int ki = 0; ki < GDN_BK; ki++)
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                hr[ki][vi] *= decay;

        // ============================================================
        // STEPS 2+3+5 fused: For each token, compute v_new from h_old,
        // gate it, and accumulate K^T @ gated_v_new into hr.
        // ============================================================
        for (int i = 0; i < chunk_len; i++) {
            int t_i = t_base + i;
            float gi = gcum_in[chunk_g_base + i];

            // v_new = u[i] - w[i] @ h_old (warp reduce over K)
            float my_w[GDN_BK];
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++)
                my_w[ki] = w_in[chunk_w_base + i * K + k_off + ki];

            float wh[GDN_BV];
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++) {
                float acc = 0.0f;
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++)
                    acc += my_w[ki] * h_old[ki][vi];
                wh[vi] = acc;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    wh[vi] += __shfl_xor_sync(0xffffffff, wh[vi], off);

            float vn[GDN_BV];
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                vn[vi] = u_in[chunk_u_base + i * V + v_off + vi] - wh[vi];

            // Store v_new (before gating) for output kernel
            if (v_new_out != nullptr && lane == 0) {
                size_t vn_base = ((size_t)c * NH + head) * BT * V;
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    v_new_out[vn_base + i * V + v_off + vi] =
                        __float2bfloat16(vn[vi]);
            }

            // Gate: v_new *= exp(g_last - g[i])
            float gate = expf(g_last - gi);
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                vn[vi] *= gate;

            // K^T @ gated_v_new: load K[t_i] and accumulate into hr
            // K is already L2-normalized externally (by the engine before calling chunk)
            float my_k[GDN_BK];
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++)
                my_k[ki] = __bfloat162float(
                    k_in[((size_t)t_i * NH + head) * K + k_off + ki]);

            // h += k[:, None] * v_new[None, :]
            // Each lane contributes its k elements, v_new is same across lanes
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++)
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    hr[ki][vi] += my_k[ki] * vn[vi];
        } // end token loop
    } // end chunk loop

    // ============================================================
    // Store final h snapshot and final state
    // ============================================================
    if (h_out != nullptr) {
        __nv_bfloat16* dst = h_out + ((size_t)NC * NH + head) * K * V;
        #pragma unroll
        for (int ki = 0; ki < GDN_BK; ki++)
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                dst[(size_t)(k_off + ki) * V + v_off + vi] =
                    __float2bfloat16(hr[ki][vi]);
    }
    if (ht != nullptr) {
        float* dst = ht + (size_t)head * K * V;
        #pragma unroll
        for (int ki = 0; ki < GDN_BK; ki++)
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                dst[(size_t)(k_off + ki) * V + v_off + vi] = hr[ki][vi];
    }
}
