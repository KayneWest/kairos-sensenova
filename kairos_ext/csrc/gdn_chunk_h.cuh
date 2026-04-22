#pragma once
/**
 * gdn_chunk_h.cuh -- Register-resident state propagation kernel for
 *                    GatedDeltaNet chunk-wise forward pass.
 *
 * Direct CUDA port of Triton's chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
 * keeping the GDN state h[K=256, BV=32] entirely in fp32 registers across all
 * chunk iterations. State is only briefly staged to SMEM when needed as a
 * B-operand for the w@h MMA computation.
 *
 * This kernel computes the inter-chunk recurrence:
 *   For each chunk t:
 *     1. Store h[t] snapshot (for the output kernel)
 *     2. v_new = u[t] - w[t] @ h        (uses tensor core MMA)
 *     3. Gate v_new by exp(g_last - g[i])
 *     4. Decay h by exp(g_last)
 *     5. h += K^T @ v_new_gated          (uses tensor core MMA)
 *
 * Design parameters (Kairos 4B: K=256, V=512, NH=20):
 *   BT = 64    tokens per chunk
 *   BV = 32    V-tile width  => V/BV = 16 blocks per head
 *   K  = 256   fixed, decomposed into 4 K-tiles of 64
 *   Block: 128 threads = 4 warps
 *
 * Register state layout:
 *   h_state[4][4][4] fp32 -- [K_tile][N_tile][4_accum_regs]
 *   64 registers per thread for state, ~110 total. Within SM120's 255 limit.
 *
 * Optimizations (v2):
 *   - Padded SMEM strides to eliminate bank conflicts
 *   - cp.async for non-blocking global->SMEM copies (bypasses registers)
 *   - Double-buffered s_a for software-pipelined w@h and K^T@v loops
 *
 * Shared memory layout (~29 KB, fits 48 KB default on SM120):
 *   s_hstage [64][S_H_STRIDE=40] bf16 = 5120 B   state staging for MMA B-operand
 *   s_a0     [64][S_A_STRIDE=72] bf16 = 9216 B   A-operand buffer 0
 *   s_a1     [64][S_A_STRIDE=72] bf16 = 9216 B   A-operand buffer 1 (double buffer)
 *   s_v      [64][S_V_STRIDE=40] bf16 = 5120 B   u / v_new tile
 *   s_g      [64]                f32  =  256 B   gate values per chunk
 *   Total: 28928 B = ~28.3 KB
 *
 * MMA: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
 *   Each tl.dot(A[64,64], B[64,32]) decomposes into:
 *     4 M-warps x 4 N-tiles x 4 K-steps = 64 MMA instructions
 *
 * Grid: dim3(V/BV, NH) = dim3(16, 20) = 320 blocks
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ===================================================================
// Compile-time constants
// ===================================================================
#define GDN_H_BT   64   // tokens per chunk
#define GDN_H_BV   32   // V-tile width
#define GDN_H_BK   64   // K-tile width (K=256 = 4 tiles)
#define GDN_H_NK   4    // number of K-tiles
#define GDN_H_K    256  // total K dimension

// MMA tile dimensions
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

// Number of N-tiles: BV / MMA_N = 32 / 8 = 4
#define GDN_H_NT (GDN_H_BV / MMA_N)

// Padded SMEM strides to reduce bank conflicts.
#define S_A_STRIDE  72   // padded from 64
#define S_H_STRIDE  40   // padded from 32
#define S_V_STRIDE  40   // padded from 32

// Identity swizzle (padding handles bank conflicts instead)
__device__ __forceinline__
int swizzle_col(int row, int col) {
    (void)row;
    return col;
}

// ===================================================================
// PTX MMA wrapper
// ===================================================================
__device__ __forceinline__
void mma_m16n8k16_bf16(float d[4], unsigned a[4], unsigned b[2], float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// ===================================================================
// Helper: pack two bf16 values into one unsigned (low, high)
// ===================================================================
__device__ __forceinline__
unsigned pack_bf16_pair(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    unsigned short lo_bits = *reinterpret_cast<unsigned short*>(&lo);
    unsigned short hi_bits = *reinterpret_cast<unsigned short*>(&hi);
    return (unsigned)lo_bits | ((unsigned)hi_bits << 16);
}

__device__ __forceinline__
unsigned pack_bf16_from_float(float lo, float hi) {
    return pack_bf16_pair(__float2bfloat16(lo), __float2bfloat16(hi));
}

// ===================================================================
// Async copy helpers (cp.async, SM80+)
// ===================================================================

/// Copy 16 bytes from global to shared memory without touching registers.
__device__ __forceinline__
void async_copy_16B(void* smem_dst, const void* global_src) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        : : "r"(smem_addr), "l"(global_src));
}

/// Commit the current group of cp.async operations.
__device__ __forceinline__
void async_copy_commit() {
    asm volatile("cp.async.commit_group;\n");
}

/// Wait for all outstanding cp.async groups to complete.
__device__ __forceinline__
void async_copy_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

/// Wait for all but N outstanding cp.async groups to complete.
template <int N>
__device__ __forceinline__
void async_copy_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

// ===================================================================
// Fragment loaders from shared memory
// ===================================================================

/**
 * Load A-operand fragment from row-major bf16 shared memory.
 * A is [M, K] stored row-major with stride stride_cols elements.
 * Loads a 16x16 sub-tile at (m_off, k_off).
 *
 * MMA A-fragment layout (per thread in warp):
 *   row0 = (lane/4) % 8,  row1 = row0 + 8
 *   col  = (lane%4) * 2
 *   frag[0] = {A[row0, col], A[row0, col+1]}
 *   frag[1] = {A[row0, col+8], A[row0, col+9]}
 *   frag[2] = {A[row1, col], A[row1, col+1]}
 *   frag[3] = {A[row1, col+8], A[row1, col+9]}
 */
__device__ __forceinline__
void load_a_frag(unsigned frag[4],
                 const __nv_bfloat16* __restrict__ smem,
                 int m_off, int k_off, int stride_cols) {
    int lane = threadIdx.x & 31;
    int sub_row = lane & 7;
    int sub_id  = lane >> 3;
    int row = (sub_id >> 1) * 8 + sub_row;
    int col = (sub_id & 1) * 8;
    // Apply XOR swizzle to column
    int s_col = swizzle_col(m_off + row, k_off + col);
    unsigned addr = __cvta_generic_to_shared(
        &smem[(m_off + row) * stride_cols + s_col]);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
        : "r"(addr));
}

/**
 * Load B-operand fragment from row-major bf16 shared memory.
 * B is [K, N] stored row-major with stride stride_cols elements.
 * Loads a 16x8 sub-tile at (k_off, n_off).
 *
 * MMA B-fragment layout for .row.col (B consumed col-major):
 *   k0 = (lane%4) * 2,  n = lane/4    [n ranges 0..7]
 *   frag[0] = {B[k0, n], B[k0+1, n]}
 *   frag[1] = {B[k0+8, n], B[k0+8+1, n]}
 */
__device__ __forceinline__
void load_b_frag(unsigned frag[2],
                 const __nv_bfloat16* __restrict__ smem,
                 int k_off, int n_off, int stride_cols) {
    int lane = threadIdx.x & 31;
    int sub_row = lane & 7;
    int sub_id  = lane >> 3;
    int row = (sub_id & 1) * 8 + sub_row;
    // Apply XOR swizzle to column
    int s_col = swizzle_col(k_off + row, n_off);
    unsigned addr = __cvta_generic_to_shared(
        &smem[(k_off + row) * stride_cols + s_col]);
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(frag[0]), "=r"(frag[1])
        : "r"(addr));
}

// ===================================================================
// Cooperative global->SMEM loaders (all 128 threads participate)
// ===================================================================

/**
 * Load a bf16 matrix from global to SWIZZLED shared memory using cp.async.
 * Applies XOR swizzle: smem[row * stride + swizzle_col(row, col)]
 * Note: cp.async can't apply swizzle directly, so we use regular loads
 * with swizzled writes for correctness. TODO: restructure for cp.async.
 */
__device__ __forceinline__
void cooperative_load_bf16_swizzled(
    __nv_bfloat16* __restrict__ dst, int dst_stride,
    const __nv_bfloat16* __restrict__ src, int src_stride,
    int rows, int cols)
{
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int total = rows * cols;
    for (int i = tid; i < total; i += nthreads) {
        int row = i / cols;
        int col = i % cols;
        int s_col = swizzle_col(row, col);
        dst[row * dst_stride + s_col] = src[row * src_stride + col];
    }
}

/**
 * Load float32 matrix from global to SWIZZLED shared bf16 memory.
 */
__device__ __forceinline__
void cooperative_load_f32_to_bf16_swizzled(
    __nv_bfloat16* __restrict__ dst, int dst_stride,
    const float* __restrict__ src, int src_stride,
    int rows, int cols)
{
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int total = rows * cols;
    for (int i = tid; i < total; i += nthreads) {
        int row = i / cols;
        int col = i % cols;
        int s_col = swizzle_col(row, col);
        dst[row * dst_stride + s_col] = __float2bfloat16(src[row * src_stride + col]);
    }
}

/**
 * Load TRANSPOSED float32 matrix into SWIZZLED shared bf16 memory.
 * src is [src_rows, src_stride] float, dst is [dst_rows, dst_stride] bf16.
 * dst[k, t] = bf16(src[t, k_base + k]) with swizzle on dst columns.
 */
__device__ __forceinline__
void cooperative_load_f32_to_bf16_transposed_swizzled(
    __nv_bfloat16* __restrict__ dst, int dst_stride,
    const float* __restrict__ src, int src_stride,
    int dst_rows, int dst_cols, int src_col_offset)
{
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int total = dst_rows * dst_cols;
    for (int i = tid; i < total; i += nthreads) {
        int k_row = i / dst_cols;
        int t_col = i % dst_cols;
        int s_col = swizzle_col(k_row, t_col);
        dst[k_row * dst_stride + s_col] =
            __float2bfloat16(src[t_col * src_stride + src_col_offset + k_row]);
    }
}

/**
 * Store a bf16 matrix from shared memory to global memory (vectorized).
 */
__device__ __forceinline__
void cooperative_store_bf16(const __nv_bfloat16* __restrict__ src,
                            __nv_bfloat16* __restrict__ dst,
                            int total_elems) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int total_vec4 = total_elems / 4;
    const uint2* src4 = reinterpret_cast<const uint2*>(src);
    uint2* dst4 = reinterpret_cast<uint2*>(dst);
    for (int i = tid; i < total_vec4; i += nthreads) {
        dst4[i] = src4[i];
    }
    int done = total_vec4 * 4;
    for (int i = done + tid; i < total_elems; i += nthreads) {
        dst[i] = src[i];
    }
}

// ===================================================================
// Kernel: k_gdn_chunk_h
// ===================================================================
// Grid:  dim3(V / GDN_H_BV, NH)  = dim3(16, 20) for Kairos 4B
// Block: 128 threads (4 warps)
//
// State h[K=256, BV=32] lives entirely in fp32 REGISTERS.
// Per thread: h_state[4 K-tiles][4 N-tiles][4 accum regs] = 64 floats.
//
// Accumulator mapping (for mma m16n8k16):
//   lane = threadIdx.x & 31, groupID = lane/4, tid_in_group = lane%4
//   For h_state[kt][nt]:
//     [0] = h[kt*64 + warp*16 + groupID,     nt*8 + tid_in_group*2]
//     [1] = h[kt*64 + warp*16 + groupID,     nt*8 + tid_in_group*2 + 1]
//     [2] = h[kt*64 + warp*16 + groupID + 8, nt*8 + tid_in_group*2]
//     [3] = h[kt*64 + warp*16 + groupID + 8, nt*8 + tid_in_group*2 + 1]
//
// Shared memory layout (padded + double-buffered):
//   s_hstage [64][S_H_STRIDE=40] bf16 = 5120 B   (state staging)
//   s_a0     [64][S_A_STRIDE=72] bf16 = 9216 B   (A-operand buffer 0)
//   s_a1     [64][S_A_STRIDE=72] bf16 = 9216 B   (A-operand buffer 1)
//   s_v      [64][S_V_STRIDE=40] bf16 = 5120 B   (u / v_new tile)
//   s_g      [64]                f32  =  256 B   (gate values)
//   Total: 28928 B = ~28.3 KB
// ===================================================================
__global__ void __launch_bounds__(128, 2)
k_gdn_chunk_h(
    const float*         __restrict__ w,       // [NC, NH, BT, K]
    const float*         __restrict__ u,       // [NC, NH, BT, V]
    const float*         __restrict__ gcum,    // [NC, NH, BT]
    __nv_bfloat16*       __restrict__ h_out,   // [NC+1, NH, K, V]
    float*               __restrict__ ht,      // [NH, K, V] or nullptr
    const float*         __restrict__ h0,      // [NH, K, V] initial state, or nullptr
    int NC, int NH, int K, int V)
{
    // Block identity
    const int i_v  = blockIdx.x;       // V-tile index
    const int i_nh = blockIdx.y;       // head index
    const int tid  = threadIdx.x;      // 0..127
    const int warp_id = tid >> 5;      // 0..3
    const int lane = tid & 31;
    const int groupID = lane >> 2;             // 0..7
    const int tid_in_group = lane & 3;         // 0..3
    const int v_offset = i_v * GDN_H_BV;

    // ---- Shared memory layout (padded + double-buffered) ----
    extern __shared__ char _smem[];
    char* sp = _smem;

    // s_hstage: staging buffer for one K-tile of h [64, S_H_STRIDE] bf16
    __nv_bfloat16* s_hstage = reinterpret_cast<__nv_bfloat16*>(sp);
    sp += GDN_H_BK * S_H_STRIDE * sizeof(__nv_bfloat16);               // 5120 B

    // s_a: double-buffered A-operand tiles [64, S_A_STRIDE] bf16 x 2
    __nv_bfloat16* s_a_buf[2];
    s_a_buf[0] = reinterpret_cast<__nv_bfloat16*>(sp);
    sp += GDN_H_BT * S_A_STRIDE * sizeof(__nv_bfloat16);               // 9216 B
    s_a_buf[1] = reinterpret_cast<__nv_bfloat16*>(sp);
    sp += GDN_H_BT * S_A_STRIDE * sizeof(__nv_bfloat16);               // 9216 B

    // s_v: v_new tile [BT=64, S_V_STRIDE] bf16
    __nv_bfloat16* s_v = reinterpret_cast<__nv_bfloat16*>(sp);
    sp += GDN_H_BT * S_V_STRIDE * sizeof(__nv_bfloat16);               // 5120 B

    // s_g: gate values [BT=64] float32
    float* s_g = reinterpret_cast<float*>(sp);
    // sp += GDN_H_BT * sizeof(float);                                  //  256 B

    // ---- Register-resident state: h[K=256, BV=32] ----
    // h_state[kt][nt][4] where kt=0..3, nt=0..3, 4 accum regs per MMA tile
    // Each warp owns 16 rows of each K-tile (warp_id * 16 .. warp_id * 16 + 15)
    float h_state[GDN_H_NK][GDN_H_NT][4];

    // ---- Initialize state from h0 or zero ----
    if (h0 != nullptr) {
        const float* h0_ptr = h0 + (size_t)i_nh * K * V;
        #pragma unroll
        for (int kt = 0; kt < GDN_H_NK; kt++) {
            #pragma unroll
            for (int nt = 0; nt < GDN_H_NT; nt++) {
                int row0 = kt * GDN_H_BK + warp_id * MMA_M + groupID;
                int row1 = row0 + 8;
                int col0 = v_offset + nt * MMA_N + tid_in_group * 2;
                h_state[kt][nt][0] = h0_ptr[row0 * V + col0];
                h_state[kt][nt][1] = h0_ptr[row0 * V + col0 + 1];
                h_state[kt][nt][2] = h0_ptr[row1 * V + col0];
                h_state[kt][nt][3] = h0_ptr[row1 * V + col0 + 1];
            }
        }
    } else {
        #pragma unroll
        for (int kt = 0; kt < GDN_H_NK; kt++)
            #pragma unroll
            for (int nt = 0; nt < GDN_H_NT; nt++)
                #pragma unroll
                for (int r = 0; r < 4; r++)
                    h_state[kt][nt][r] = 0.0f;
    }

    // ---- Main loop over chunks ----
    for (int i_t = 0; i_t < NC; i_t++) {

        // ============================================================
        // STEP 1: Store state snapshot h[i_t] to global h_out
        // ============================================================
        if (h_out != nullptr) {
            __nv_bfloat16* h_dst = h_out
                + ((size_t)i_t * NH + i_nh) * K * V
                + v_offset;

            #pragma unroll
            for (int kt = 0; kt < GDN_H_NK; kt++) {
                #pragma unroll
                for (int nt = 0; nt < GDN_H_NT; nt++) {
                    int row0 = kt * GDN_H_BK + warp_id * MMA_M + groupID;
                    int row1 = row0 + 8;
                    int col0 = nt * MMA_N + tid_in_group * 2;

                    h_dst[row0 * V + col0]     = __float2bfloat16(h_state[kt][nt][0]);
                    h_dst[row0 * V + col0 + 1] = __float2bfloat16(h_state[kt][nt][1]);
                    h_dst[row1 * V + col0]     = __float2bfloat16(h_state[kt][nt][2]);
                    h_dst[row1 * V + col0 + 1] = __float2bfloat16(h_state[kt][nt][3]);
                }
            }
        }

        // ============================================================
        // STEP 2: Compute v_new = u - w @ h   (MMA, software-pipelined)
        // ============================================================
        // w @ h: w[BT=64, K=256] @ h[K=256, BV=32] -> [BT, BV]
        // Decomposed into 4 K-tiles, each requiring staging h to SMEM.
        // Double-buffered s_a with cp.async for pipelining.

        // Per-thread MMA accumulators for v_new result
        float v_acc[GDN_H_NT][4];
        #pragma unroll
        for (int nt = 0; nt < GDN_H_NT; nt++)
            #pragma unroll
            for (int r = 0; r < 4; r++)
                v_acc[nt][r] = 0.0f;

        {
            // Base pointer for w[i_t, i_nh, :, :]
            const float* w_base = w
                + ((size_t)i_t * NH + i_nh) * GDN_H_BT * GDN_H_K;

            // --- Prologue: load first w-tile (kt=0) into s_a_buf[0] ---
            cooperative_load_f32_to_bf16_swizzled(
                s_a_buf[0], S_A_STRIDE,
                w_base + 0 * GDN_H_BK, GDN_H_K,
                GDN_H_BT, GDN_H_BK);

            int cur_buf = 0;

            #pragma unroll
            for (int kt = 0; kt < GDN_H_NK; kt++) {
                // --- Stage h_state[kt] to s_hstage[64, S_H_STRIDE] bf16 ---
                #pragma unroll
                for (int nt = 0; nt < GDN_H_NT; nt++) {
                    int row0 = warp_id * MMA_M + groupID;
                    int row1 = row0 + 8;
                    int col0 = nt * MMA_N + tid_in_group * 2;

                    s_hstage[row0 * S_H_STRIDE + swizzle_col(row0, col0)]     = __float2bfloat16(h_state[kt][nt][0]);
                    s_hstage[row0 * S_H_STRIDE + swizzle_col(row0, col0 + 1)] = __float2bfloat16(h_state[kt][nt][1]);
                    s_hstage[row1 * S_H_STRIDE + swizzle_col(row1, col0)]     = __float2bfloat16(h_state[kt][nt][2]);
                    s_hstage[row1 * S_H_STRIDE + swizzle_col(row1, col0 + 1)] = __float2bfloat16(h_state[kt][nt][3]);
                }
                __syncthreads();  // ensure s_hstage + s_a_buf[cur_buf] ready

                // --- Start loading NEXT w-tile into alternate buffer (overlaps with MMA) ---
                int next_buf = 1 - cur_buf;
                if (kt + 1 < GDN_H_NK) {
                    cooperative_load_f32_to_bf16_swizzled(
                        s_a_buf[next_buf], S_A_STRIDE,
                        w_base + (kt + 1) * GDN_H_BK, GDN_H_K,
                        GDN_H_BT, GDN_H_BK);
                }

                // --- MMA: v_acc += s_a_buf[cur_buf][64, 64] @ s_hstage[64, 32] ---
                {
                    int m_off = warp_id * MMA_M;

                    #pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        int k_off = ks * MMA_K;

                        unsigned a_frag[4];
                        load_a_frag(a_frag, s_a_buf[cur_buf], m_off, k_off, S_A_STRIDE);

                        #pragma unroll
                        for (int nt = 0; nt < GDN_H_NT; nt++) {
                            int n_off = nt * MMA_N;

                            unsigned b_frag[2];
                            load_b_frag(b_frag, s_hstage, k_off, n_off, S_H_STRIDE);

                            float d[4];
                            mma_m16n8k16_bf16(d, a_frag, b_frag, v_acc[nt]);
                            #pragma unroll
                            for (int r = 0; r < 4; r++)
                                v_acc[nt][r] = d[r];
                        }
                    }
                }
                __syncthreads();  // wait for MMA reads + next tile load writes
                cur_buf = next_buf;
            } // end K-tile loop for w@h
        }

        // --- Load u[i_t] -> s_v[64, S_V_STRIDE] as bf16 ---
        {
            const float* u_src = u
                + ((size_t)i_t * NH + i_nh) * GDN_H_BT * V
                + v_offset;
            cooperative_load_f32_to_bf16_swizzled(
                s_v, S_V_STRIDE,
                u_src, V,
                GDN_H_BT, GDN_H_BV);
        }
        __syncthreads();

        // --- Compute v_new = u - v_acc, store to s_v as bf16 ---
        {
            int m_off = warp_id * MMA_M;

            #pragma unroll
            for (int nt = 0; nt < GDN_H_NT; nt++) {
                int n_off = nt * MMA_N;

                int r0 = m_off + groupID;
                int r1 = r0 + 8;
                int c0 = n_off + tid_in_group * 2;

                int idx00 = r0 * S_V_STRIDE + c0;
                int idx01 = idx00 + 1;
                int idx10 = r1 * S_V_STRIDE + c0;
                int idx11 = idx10 + 1;

                float u00 = __bfloat162float(s_v[idx00]);
                float u01 = __bfloat162float(s_v[idx01]);
                float u10 = __bfloat162float(s_v[idx10]);
                float u11 = __bfloat162float(s_v[idx11]);

                s_v[idx00] = __float2bfloat16(u00 - v_acc[nt][0]);
                s_v[idx01] = __float2bfloat16(u01 - v_acc[nt][1]);
                s_v[idx10] = __float2bfloat16(u10 - v_acc[nt][2]);
                s_v[idx11] = __float2bfloat16(u11 - v_acc[nt][3]);
            }
        }
        __syncthreads();

        // ============================================================
        // STEP 3: Load gates, apply gating to v_new in s_v
        // ============================================================
        {
            const float* g_src = gcum
                + ((size_t)i_t * NH + i_nh) * GDN_H_BT;
            if (tid < GDN_H_BT) {
                s_g[tid] = g_src[tid];
            }
        }
        __syncthreads();

        float g_last = s_g[GDN_H_BT - 1];
        float decay = expf(g_last);

        // Apply per-token gating: v_new[t, :] *= exp(g_last - g_cum[t])
        {
            int total = GDN_H_BT * GDN_H_BV;  // 2048
            for (int i = tid; i < total; i += 128) {
                int row = i / GDN_H_BV;
                int col = i % GDN_H_BV;
                float gate = expf(g_last - s_g[row]);
                int sc = swizzle_col(row, col);
                float val = __bfloat162float(s_v[row * S_V_STRIDE + sc]) * gate;
                s_v[row * S_V_STRIDE + sc] = __float2bfloat16(val);
            }
        }
        __syncthreads();

        // ============================================================
        // STEP 4: Decay state + h += K^T @ v_new_gated (software-pipelined)
        // ============================================================

        // Decay all state registers
        #pragma unroll
        for (int kt = 0; kt < GDN_H_NK; kt++)
            #pragma unroll
            for (int nt = 0; nt < GDN_H_NT; nt++)
                #pragma unroll
                for (int r = 0; r < 4; r++)
                    h_state[kt][nt][r] *= decay;

        // Accumulate K^T @ v_new for each K-tile
        // K^T is w^T: we load w[i_t, i_nh, :, kt*64:(kt+1)*64]^T as [64, BT]
        // Double-buffered s_a with pipelined loads.
        {
            const float* w_base = w
                + ((size_t)i_t * NH + i_nh) * GDN_H_BT * GDN_H_K;

            // --- Prologue: load first transposed w-tile (kt=0) into s_a_buf[0] ---
            cooperative_load_f32_to_bf16_transposed_swizzled(
                s_a_buf[0], S_A_STRIDE,
                w_base, GDN_H_K,
                GDN_H_BK, GDN_H_BT, 0 * GDN_H_BK);

            int cur_buf = 0;

            #pragma unroll
            for (int kt = 0; kt < GDN_H_NK; kt++) {
                __syncthreads();  // ensure s_a_buf[cur_buf] ready

                // --- Start loading NEXT transposed tile into alternate buffer ---
                int next_buf = 1 - cur_buf;
                if (kt + 1 < GDN_H_NK) {
                    cooperative_load_f32_to_bf16_transposed_swizzled(
                        s_a_buf[next_buf], S_A_STRIDE,
                        w_base, GDN_H_K,
                        GDN_H_BK, GDN_H_BT, (kt + 1) * GDN_H_BK);
                }

                // MMA: h_state[kt] += s_a_buf[cur_buf][64, BT=64] @ s_v[BT=64, BV=32]
                {
                    int m_off = warp_id * MMA_M;

                    #pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        int k_off = ks * MMA_K;

                        unsigned a_frag[4];
                        load_a_frag(a_frag, s_a_buf[cur_buf], m_off, k_off, S_A_STRIDE);

                        #pragma unroll
                        for (int nt = 0; nt < GDN_H_NT; nt++) {
                            int n_off = nt * MMA_N;

                            unsigned b_frag[2];
                            load_b_frag(b_frag, s_v, k_off, n_off, S_V_STRIDE);

                            float d[4];
                            mma_m16n8k16_bf16(d, a_frag, b_frag, h_state[kt][nt]);
                            #pragma unroll
                            for (int r = 0; r < 4; r++)
                                h_state[kt][nt][r] = d[r];
                        }
                    }
                }
                cur_buf = next_buf;
            } // end K-tile loop for state update
            __syncthreads();  // final sync after last iteration
        }

    } // end chunk loop

    // ============================================================
    // Store final state snapshot h[NC] to h_out and optionally ht
    // ============================================================
    if (h_out != nullptr) {
        __nv_bfloat16* h_dst = h_out
            + ((size_t)NC * NH + i_nh) * K * V
            + v_offset;

        #pragma unroll
        for (int kt = 0; kt < GDN_H_NK; kt++) {
            #pragma unroll
            for (int nt = 0; nt < GDN_H_NT; nt++) {
                int row0 = kt * GDN_H_BK + warp_id * MMA_M + groupID;
                int row1 = row0 + 8;
                int col0 = nt * MMA_N + tid_in_group * 2;

                h_dst[row0 * V + col0]     = __float2bfloat16(h_state[kt][nt][0]);
                h_dst[row0 * V + col0 + 1] = __float2bfloat16(h_state[kt][nt][1]);
                h_dst[row1 * V + col0]     = __float2bfloat16(h_state[kt][nt][2]);
                h_dst[row1 * V + col0 + 1] = __float2bfloat16(h_state[kt][nt][3]);
            }
        }
    }

    if (ht != nullptr) {
        float* ht_dst = ht + (size_t)i_nh * K * V + v_offset;

        #pragma unroll
        for (int kt = 0; kt < GDN_H_NK; kt++) {
            #pragma unroll
            for (int nt = 0; nt < GDN_H_NT; nt++) {
                int row0 = kt * GDN_H_BK + warp_id * MMA_M + groupID;
                int row1 = row0 + 8;
                int col0 = nt * MMA_N + tid_in_group * 2;

                ht_dst[row0 * V + col0]     = h_state[kt][nt][0];
                ht_dst[row0 * V + col0 + 1] = h_state[kt][nt][1];
                ht_dst[row1 * V + col0]     = h_state[kt][nt][2];
                ht_dst[row1 * V + col0 + 1] = h_state[kt][nt][3];
            }
        }
    }
}


// ===================================================================
// SMEM size for k_gdn_chunk_h
// ===================================================================
inline size_t gdn_chunk_h_smem_bytes() {
    return GDN_H_BK * S_H_STRIDE * sizeof(__nv_bfloat16)       // s_hstage: 4096
         + GDN_H_BT * S_A_STRIDE * sizeof(__nv_bfloat16) * 2   // s_a x2:  16384
         + GDN_H_BT * S_V_STRIDE * sizeof(__nv_bfloat16)       // s_v:      4096
         + GDN_H_BT * sizeof(float);                            // s_g:       256
    // Total: ~29 KB with padding
}


// ===================================================================
// Launch helper
// ===================================================================
inline void gdn_chunk_h_launch(
    const float* w,             // [NC, NH, BT, K]
    const float* u,             // [NC, NH, BT, V]
    const float* gcum,          // [NC, NH, BT]
    __nv_bfloat16* h_out,       // [NC+1, NH, K, V]
    float* ht,                  // [NH, K, V] or nullptr
    const float* h0,            // [NH, K, V] or nullptr
    int NC, int NH, int K, int V,
    cudaStream_t stream)
{
    dim3 grid(V / GDN_H_BV, NH);   // (16, 20) for Kairos 4B
    dim3 block(128);                // 4 warps

    size_t smem = gdn_chunk_h_smem_bytes();

    // ~28.3KB, well under 48KB default.
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(k_gdn_chunk_h,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }

    k_gdn_chunk_h<<<grid, block, smem, stream>>>(
        w, u, gcum, h_out, ht, h0, NC, NH, K, V);
}
