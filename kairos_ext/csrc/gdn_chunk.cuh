#pragma once
/**
 * gdn_chunk.cuh -- 3-kernel chunk GatedDeltaNet forward for RTX 5090.
 *
 * Fits within 48 KB default shared memory per block (SM120 GeForce limit).
 *
 * Kernel 1: k_gdn_chunk_prepare  -- parallel across (NC, NH), 64 threads
 *   Computes: g_cum (prefix sum of gates), K L2 norms, gated lower-triangular
 *   matrix A, forward substitution A_inv = (I-A)^{-1}, then:
 *     w[i,k] = sum_j A_inv[i,j] * beta[j] * krnorm[j] * exp(gcum[j]) * K_normed[j,k]
 *     u[i,v] = sum_j A_inv[i,j] * beta[j] * V[j,v]
 *   SMEM: ~25 KB (tiled K @ K^T with BK_TILE=64)
 *
 * Kernel 2: k_gdn_chunk_state_output -- sequential over chunks, parallel (V/16, NH)
 *   Warp-only (32 threads), state h[8][16] in registers.
 *   Per chunk: decay h by exp(g_last), compute v_new = u - w@h,
 *   compute output = scale * Q_norm @ h, gate v_new, update h.
 *   No shared memory needed.
 *
 * Designed for Kairos 4B: K=256, V=512, NH=20.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// --------------- compile-time knobs --------------------------------
#define GDN_CHUNK_BT   64   // tokens per chunk
#define GDN_CHUNK_BV   64   // V-tile width (unused by new kernels, kept for compat)
#define GDN_BK_STATE   64   // K-tile width for prepare kernel

// Prepare kernel tiling
#define PREP_BK_TILE   64   // K columns loaded at a time into SMEM
#define PREP_BV_TILE   64   // V columns processed at a time in u computation

// State kernel parameters (must match k_gdn_recurrent)
#ifndef GDN_BV
#define GDN_BV 16
#endif
#ifndef GDN_BK
#define GDN_BK 8
#endif


// ===================================================================
// Prefix sum helper (single-thread, small array)
// ===================================================================
__device__ __forceinline__
void _chunk_prefix_sum(float* buf, int n) {
    for (int i = 1; i < n; i++)
        buf[i] += buf[i - 1];
}


// ===================================================================
// Kernel 1: k_gdn_chunk_prepare
// ===================================================================
// Grid:  dim3(NC, NH)
// Block: 64 threads (one thread per token in chunk)
//
// Shared memory layout (~25 KB):
//   s_k     [BT, BK_TILE]  bf16 =  8192 B  (tiled, 4 passes for K=256)
//   s_A     [BT, BT]       fp32 = 16384 B
//   s_gcum  [BT]           fp32 =   256 B
//   s_beta  [BT]           fp32 =   256 B
//   s_krnorm[BT]           fp32 =   256 B
//   Total: ~25344 B = 24.75 KB
// ===================================================================
__global__ void __launch_bounds__(64)
k_gdn_chunk_prepare(
    const __nv_bfloat16* __restrict__ k_in,      // [T, NH, K]
    const __nv_bfloat16* __restrict__ v_in,      // [T, NH, V]
    const float*         __restrict__ g_in,      // [T, NH]
    const float*         __restrict__ beta_in,   // [T, NH]
    float*               __restrict__ w_out,     // [NC, NH, BT, K]
    float*               __restrict__ u_out,     // [NC, NH, BT, V]
    float*               __restrict__ gcum_out,  // [NC, NH, BT]
    int T, int NH, int K, int V)
{
    const int BT = GDN_CHUNK_BT;     // 64
    const int BK_TILE = PREP_BK_TILE; // 64
    const int BV_TILE = PREP_BV_TILE; // 64

    const int chunk = blockIdx.x;
    const int head  = blockIdx.y;
    const int tid   = threadIdx.x;   // 0..63, one per token in chunk
    const int t_base = chunk * BT;
    const int chunk_len = min(BT, T - t_base);

    // NK = number of K-tiles, NV = number of V-tiles
    const int NK = (K + BK_TILE - 1) / BK_TILE;
    const int NV = (V + BV_TILE - 1) / BV_TILE;

    // ---- Shared memory ----
    extern __shared__ char _smem[];
    char* sp = _smem;
    __nv_bfloat16* s_k = (__nv_bfloat16*)sp;  sp += BT * BK_TILE * sizeof(__nv_bfloat16);
    float* s_A      = (float*)sp;  sp += BT * BT * sizeof(float);
    float* s_gcum   = (float*)sp;  sp += BT * sizeof(float);
    float* s_beta   = (float*)sp;  sp += BT * sizeof(float);
    float* s_krnorm = (float*)sp;  sp += BT * sizeof(float);

    // ---- Step 1: Load g, beta. Compute g_cum and K L2 norms ----
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

    // K L2 norm: each thread tid computes norm of K[tid, :]
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

    // Store gcum to global (will be shifted AFTER step 4 computes gcum_shift)
    // Placeholder — actual store happens after gcum_shift is known.

    // ---- Step 2: Compute A[i,j] = beta[i]*exp(gcum[i]-gcum[j])*krnorm[i]*krnorm[j]*dot(K_n[i],K_n[j]) for j<i ----
    // Initialize A to zero
    for (int j = 0; j < BT; j++) s_A[tid * BT + j] = 0.0f;
    __syncthreads();

    // Tile K in BK_TILE=64 chunks to compute dot products
    for (int kt = 0; kt < NK; kt++) {
        int k_off = kt * BK_TILE;
        int k_end = min(k_off + BK_TILE, K);
        int k_width = k_end - k_off;

        // Load s_k[BT, BK_TILE] from global -- each thread loads its own row
        if (tid < chunk_len) {
            int tg = t_base + tid;
            for (int kk = 0; kk < k_width; kk++) {
                s_k[tid * BK_TILE + kk] = k_in[((size_t)tg * NH + head) * K + k_off + kk];
            }
            for (int kk = k_width; kk < BK_TILE; kk++) {
                s_k[tid * BK_TILE + kk] = __float2bfloat16(0.0f);
            }
        } else {
            for (int kk = 0; kk < BK_TILE; kk++)
                s_k[tid * BK_TILE + kk] = __float2bfloat16(0.0f);
        }
        __syncthreads();

        // Each thread tid computes partial dot products for row tid against all j < tid
        if (tid < chunk_len) {
            for (int j = 0; j < tid; j++) {
                float dot = 0.0f;
                for (int kk = 0; kk < k_width; kk++) {
                    dot += __bfloat162float(s_k[tid * BK_TILE + kk])
                         * __bfloat162float(s_k[j   * BK_TILE + kk]);
                }
                s_A[tid * BT + j] += dot;  // accumulate across K-tiles
            }
        }
        __syncthreads();
    }

    // Apply gating, beta, norm to A (lower triangular)
    if (tid < chunk_len) {
        float my_gcum = s_gcum[tid];
        float my_beta_val = s_beta[tid];
        // No krnorm: K is already L2-normalized externally (matching Triton)
        for (int j = 0; j < tid; j++) {
            s_A[tid * BT + j] *= my_beta_val * expf(my_gcum - s_gcum[j]);
        }
    }
    // Zero out non-lower-triangular and diagonal
    for (int j = tid; j < BT; j++) {
        s_A[tid * BT + j] = 0.0f;
    }
    __syncthreads();

    // ---- Step 3: Forward substitution (I - A)^{-1} ----
    // Matching Triton's solve_tril: negate A, then solve, then add I.
    // Result: (I - A)^{-1} = I + A + A^2 + ... (Neumann series)
    if (tid == 0) {
        // Negate lower triangle (matching Triton: b_A = -b_A)
        for (int i = 0; i < chunk_len; i++)
            for (int j = 0; j < i; j++)
                s_A[i * BT + j] = -s_A[i * BT + j];

        // Row-wise forward substitution on -A
        // For each row i >= 2: a[i,:] = -A[i,:] + sum(-A[i,:] * b_A[:,:])
        // This matches Triton's: b_a = -load(A[i]) + sum(b_a * b_A)
        for (int i = 2; i < chunk_len; i++) {
            for (int j = 0; j < i; j++) {
                float a_ij = s_A[i * BT + j];  // already negated
                float acc = 0.0f;
                for (int kk = 0; kk < i; kk++)
                    acc += s_A[i * BT + kk] * s_A[kk * BT + j];
                s_A[i * BT + j] = a_ij + acc;
            }
        }

        // Add identity
        for (int i = 0; i < chunk_len; i++)
            s_A[i * BT + i] = 1.0f;
    }
    __syncthreads();
    // Now s_A = (I - A)^{-1}

    // ---- Step 4: Compute w[i,k] = sum_j A_inv[i,j] * beta[j] * exp(gcum[j]) * K_normed[j,k] ----
    // K_normed[j,k] = K[j,k] * krnorm[j] is stored in s_k after loading.
    //
    {
        size_t w_base = ((size_t)chunk * NH + head) * BT * K;
        for (int kt = 0; kt < NK; kt++) {
            int k_off = kt * BK_TILE;
            int k_end = min(k_off + BK_TILE, K);
            int k_width = k_end - k_off;

            // Load s_k[BT, BK_TILE] with raw K (already L2-normalized externally)
            if (tid < chunk_len) {
                int tg = t_base + tid;
                for (int kk = 0; kk < k_width; kk++) {
                    s_k[tid * BK_TILE + kk] = k_in[((size_t)tg * NH + head) * K + k_off + kk];
                }
                for (int kk = k_width; kk < BK_TILE; kk++)
                    s_k[tid * BK_TILE + kk] = __float2bfloat16(0.0f);
            } else {
                for (int kk = 0; kk < BK_TILE; kk++)
                    s_k[tid * BK_TILE + kk] = __float2bfloat16(0.0f);
            }
            __syncthreads();

            // Each thread tid computes w_shifted[tid, k_off:k_off+k_width]
            if (tid < chunk_len) {
                for (int kk = 0; kk < k_width; kk++) {
                    float acc = 0.0f;
                    for (int j = 0; j <= tid; j++) {
                        float c = s_beta[j] * expf(s_gcum[j]);
                        acc += s_A[tid * BT + j] * c * __bfloat162float(s_k[j * BK_TILE + kk]);
                    }
                    w_out[w_base + tid * K + k_off + kk] = acc;
                }
            } else {
                for (int kk = 0; kk < k_width; kk++)
                    w_out[w_base + tid * K + k_off + kk] = 0.0f;
            }
            __syncthreads();
        }
    }

    // Store ORIGINAL gcum to global (unshifted).
    // The shift only affects the w computation to avoid exp() underflow.
    // The state kernel uses original gcum for decay/gating.
    {
        size_t gcum_base = ((size_t)chunk * NH + head) * BT;
        gcum_out[gcum_base + tid] = s_gcum[tid];
    }
    __syncthreads();

    // ---- Step 5: Compute u[i,v] = sum_j A_inv[i,j] * beta[j] * V[j,v] ----
    // Tile V in BV_TILE=64 chunks to limit register/SMEM usage.
    // Reuse s_k buffer for V tiles (same size: BT * BV_TILE * sizeof(bf16) = 8KB)
    {
        __nv_bfloat16* s_v_tile = s_k;  // reuse the s_k SMEM buffer
        size_t u_base = ((size_t)chunk * NH + head) * BT * V;

        for (int vt = 0; vt < NV; vt++) {
            int v_off = vt * BV_TILE;
            int v_end = min(v_off + BV_TILE, V);
            int v_width = v_end - v_off;

            // Load V tile: s_v_tile[BT, BV_TILE]
            if (tid < chunk_len) {
                int tg = t_base + tid;
                for (int vv = 0; vv < v_width; vv++) {
                    s_v_tile[tid * BV_TILE + vv] =
                        v_in[((size_t)tg * NH + head) * V + v_off + vv];
                }
                for (int vv = v_width; vv < BV_TILE; vv++)
                    s_v_tile[tid * BV_TILE + vv] = __float2bfloat16(0.0f);
            } else {
                for (int vv = 0; vv < BV_TILE; vv++)
                    s_v_tile[tid * BV_TILE + vv] = __float2bfloat16(0.0f);
            }
            __syncthreads();

            // Each thread tid computes u[tid, v_off:v_off+v_width]
            if (tid < chunk_len) {
                for (int vv = 0; vv < v_width; vv++) {
                    float acc = 0.0f;
                    for (int j = 0; j <= tid; j++) {
                        acc += s_A[tid * BT + j] * s_beta[j]
                             * __bfloat162float(s_v_tile[j * BV_TILE + vv]);
                    }
                    u_out[u_base + tid * V + v_off + vv] = acc;
                }
            } else {
                for (int vv = 0; vv < v_width; vv++)
                    u_out[u_base + tid * V + v_off + vv] = 0.0f;
            }
            __syncthreads();
        }
    }
}


// ===================================================================
// Kernel 2: k_gdn_chunk_state_output
// ===================================================================
// Grid:  dim3(V/GDN_BV, NH)   -- same as k_gdn_recurrent
// Block: 32 threads (1 warp)
//
// State h[BK][BV] = h[8][16] in registers per thread (128 floats).
// Sequential over NC chunks.
//
// Per chunk:
//   1. Decay state: h *= exp(g_cum_last)
//   2. For each token i in chunk:
//      a. v_new[v] = u[i,v] - sum_k(w[i,k]*h[k,v])   (w@h, warp reduce)
//      b. output[v] = scale * sum_k(Q_normed[i,k]*h[k,v])  (Q@h, warp reduce)
//      c. Gate v_new: vg[v] = v_new[v] * exp(g_last - g_cum[i])
//      d. State update: h[k,v] += K_normed[i,k] * vg[v]
//   3. After all tokens, h already updated for next chunk.
//
// No shared memory needed (pure warp-only register kernel).
// ===================================================================
__global__ void __launch_bounds__(32)
k_gdn_chunk_state_output(
    const __nv_bfloat16* __restrict__ q_in,     // [T, NH, K]
    const __nv_bfloat16* __restrict__ k_in,     // [T, NH, K]
    const float*         __restrict__ w_in,     // [NC, NH, BT, K]
    const float*         __restrict__ u_in,     // [NC, NH, BT, V]
    const float*         __restrict__ gcum_in,  // [NC, NH, BT]
    __nv_bfloat16*       __restrict__ o_out,    // [T, NH, V]
    float*               __restrict__ state,    // [NH, K, V]
    __nv_bfloat16*       __restrict__ h_snap,   // [NC+1, NH, K, V] or nullptr
    int T, int NH, int K, int V, float scale)
{
    const int BT = GDN_CHUNK_BT;
    const int bv   = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;  // 0..31
    const int v_off = bv * GDN_BV;
    const int k_off = lane * GDN_BK;
    const int NC = (T + BT - 1) / BT;

    // Load state tile into registers
    float hr[GDN_BK][GDN_BV];
    float* st = state + (size_t)head * K * V;
    #pragma unroll
    for (int ki = 0; ki < GDN_BK; ki++)
        #pragma unroll
        for (int vi = 0; vi < GDN_BV; vi++)
            hr[ki][vi] = st[(size_t)(k_off + ki) * V + v_off + vi];

    // Process chunks using the ORIGINAL chunk algorithm formula.
    // State propagation: h = h * exp(g_last) + K^T @ (gated v_new)
    // Output: written but may be overwritten by fwd_o if h_snap is used.
    for (int c = 0; c < NC; c++) {
        const int t_base = c * BT;
        const int chunk_len = min(BT, T - t_base);

        // Store state snapshot h[c] BEFORE any decay/updates
        if (h_snap != nullptr) {
            size_t snap_base = ((size_t)c * NH + head) * (size_t)K * V;
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++)
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    h_snap[snap_base + (size_t)(k_off + ki) * V + v_off + vi] =
                        __float2bfloat16(hr[ki][vi]);
        }

        size_t chunk_w_base = ((size_t)c * NH + head) * BT * K;
        size_t chunk_u_base = ((size_t)c * NH + head) * BT * V;
        size_t chunk_g_base = ((size_t)c * NH + head) * BT;

        float g_last = gcum_in[chunk_g_base + chunk_len - 1];

        // Decay state by FULL chunk gate (correct for state propagation)
        float decay = expf(g_last);
        #pragma unroll
        for (int ki = 0; ki < GDN_BK; ki++)
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                hr[ki][vi] *= decay;

        // Process each token in chunk
        for (int i = 0; i < chunk_len; i++) {
            int t_i = t_base + i;
            float gi = gcum_in[chunk_g_base + i];

            // ---- Load w_shifted[i, :] ----
            float my_w[GDN_BK];
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++)
                my_w[ki] = w_in[chunk_w_base + i * K + k_off + ki];

            // ---- v_new = u - w_actual @ h ----
            // w_shifted @ h gives (w_actual / exp(gcum_shift)) @ h
            // multiply by exp(gcum_shift) to get w_actual @ h
            float wh[GDN_BV];
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++) {
                float acc = 0.0f;
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++)
                    acc += my_w[ki] * hr[ki][vi];
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

            // ---- Output (placeholder — overwritten by fwd_o if h_snap used) ----
            if (h_snap == nullptr) {
                float my_q[GDN_BK];
                float q_sq = 0.0f;
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++) {
                    my_q[ki] = __bfloat162float(
                        q_in[((size_t)t_i * NH + head) * K + k_off + ki]);
                    q_sq += my_q[ki] * my_q[ki];
                }
                for (int off = 16; off > 0; off >>= 1)
                    q_sq += __shfl_xor_sync(0xffffffff, q_sq, off);
                float q_sc = rsqrtf(q_sq + 1e-6f) * scale;
                #pragma unroll
                for (int ki = 0; ki < GDN_BK; ki++) my_q[ki] *= q_sc;

                float ov[GDN_BV];
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (int ki = 0; ki < GDN_BK; ki++)
                        acc += my_q[ki] * hr[ki][vi];
                    ov[vi] = acc;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    #pragma unroll
                    for (int vi = 0; vi < GDN_BV; vi++)
                        ov[vi] += __shfl_xor_sync(0xffffffff, ov[vi], off);

                if (lane == 0) {
                    size_t o_base = ((size_t)t_i * NH + head) * V + v_off;
                    #pragma unroll
                    for (int vi = 0; vi < GDN_BV; vi++)
                        o_out[o_base + vi] = __float2bfloat16(ov[vi]);
                }
            }

            // ---- Gate v_new and update state ----
            float gate = expf(g_last - gi);
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                vn[vi] *= gate;

            // Load K_normed
            float my_k[GDN_BK];
            float k_sq = 0.0f;
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++) {
                my_k[ki] = __bfloat162float(
                    k_in[((size_t)t_i * NH + head) * K + k_off + ki]);
                k_sq += my_k[ki] * my_k[ki];
            }
            for (int off = 16; off > 0; off >>= 1)
                k_sq += __shfl_xor_sync(0xffffffff, k_sq, off);
            float k_sc = rsqrtf(k_sq + 1e-6f);
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++) my_k[ki] *= k_sc;

            // State update: h += k_norm * gated_vn
            #pragma unroll
            for (int ki = 0; ki < GDN_BK; ki++)
                #pragma unroll
                for (int vi = 0; vi < GDN_BV; vi++)
                    hr[ki][vi] += my_k[ki] * vn[vi];
        } // end token loop
    } // end chunk loop

    // Store final snapshot h[NC]
    if (h_snap != nullptr) {
        size_t snap_base = ((size_t)NC * NH + head) * (size_t)K * V;
        #pragma unroll
        for (int ki = 0; ki < GDN_BK; ki++)
            #pragma unroll
            for (int vi = 0; vi < GDN_BV; vi++)
                h_snap[snap_base + (size_t)(k_off + ki) * V + v_off + vi] =
                    __float2bfloat16(hr[ki][vi]);
    }

    // Store final state
    #pragma unroll
    for (int ki = 0; ki < GDN_BK; ki++)
        #pragma unroll
        for (int vi = 0; vi < GDN_BV; vi++)
            st[(size_t)(k_off + ki) * V + v_off + vi] = hr[ki][vi];
}


// ===================================================================
//  SMEM size for prepare kernel
// ===================================================================
inline size_t gdn_chunk_prepare_smem(int /*K*/, int /*V*/) {
    const int BT = GDN_CHUNK_BT;
    const int BK_TILE = PREP_BK_TILE;
    return BT * BK_TILE * sizeof(__nv_bfloat16)   // s_k (tiled)
         + BT * BT * sizeof(float)                // s_A
         + BT * 3  * sizeof(float);               // s_gcum, s_beta, s_krnorm
}


// ===================================================================
//  Launch helper (drop-in replacement for previous gdn_chunk_forward)
// ===================================================================
inline void gdn_chunk_forward(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float* g, const float* beta,
    __nv_bfloat16* o, float* state,
    float* w_buf, float* u_buf, float* gcum_buf,
    int T, int NH, int K, int V, float scale,
    cudaStream_t stream)
{
    const int BT = GDN_CHUNK_BT;
    int NC = (T + BT - 1) / BT;

    // ---- Kernel 1: Prepare (parallel across chunks and heads) ----
    {
        size_t smem = gdn_chunk_prepare_smem(K, V);
        dim3 grid(NC, NH);
        dim3 block(64);

        // smem should be ~25KB, well under 48KB default
        if (smem > 48 * 1024) {
            cudaFuncSetAttribute(k_gdn_chunk_prepare,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        }

        k_gdn_chunk_prepare<<<grid, block, smem, stream>>>(
            k, v, g, beta,
            w_buf, u_buf, gcum_buf,
            T, NH, K, V);
    }

    // ---- Kernel 2: State propagation + output (sequential over chunks) ----
    {
        int NV_blocks = V / GDN_BV;  // 512/16 = 32
        dim3 grid(NV_blocks, NH);
        dim3 block(32);

        k_gdn_chunk_state_output<<<grid, block, 0, stream>>>(
            q, k,
            w_buf, u_buf, gcum_buf,
            o, state, nullptr,  // no h_snap for legacy 2-kernel path
            T, NH, K, V, scale);
    }
}


// ===================================================================
//  Scratch buffer size (for engine_init allocation)
// ===================================================================
inline size_t gdn_chunk_scratch_bytes(int T, int NH, int K, int V) {
    const int BT = GDN_CHUNK_BT;
    int NC = (T + BT - 1) / BT;
    size_t w_bytes    = (size_t)NC * NH * BT * K * sizeof(float);
    size_t u_bytes    = (size_t)NC * NH * BT * V * sizeof(float);
    size_t gcum_bytes = (size_t)NC * NH * BT * sizeof(float);
    return w_bytes + u_bytes + gcum_bytes;
}
