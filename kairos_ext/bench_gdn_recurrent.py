import argparse
import math
import os
import time

import torch

def init_ext(seq_len: int, nh: int, kdim: int, vdim: int):
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(__file__), "csrc", "gdn_recurrent_only.cu")
    cudnn_fe = os.environ.get("CUDNN_FRONTEND", "/tmp/cudnn-frontend/include")
    cudnn_inc = os.environ.get("CUDNN_INC", "")
    nccl_home = os.environ.get("NCCL_HOME", "/usr")
    include_paths = [cudnn_fe, f"{nccl_home}/include"]
    if cudnn_inc:
        include_paths.append(cudnn_inc)
    gdn_bv = int(os.environ.get("GDN_ONLY_BV", "8"))
    gdn_pb = int(os.environ.get("GDN_ONLY_PERSISTENT_BLOCKS", "8"))
    gdn_warp_cta = int(os.environ.get("GDN_ONLY_WARPS_PER_CTA", "1"))
    gdn_pipe_qk = int(os.environ.get("GDN_ONLY_PIPELINE_QK", "0"))
    prev_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    try:
        ext = load(
            name=f"gdn_recurrent_only_bv{gdn_bv}_pb{gdn_pb}_wcta{gdn_warp_cta}_pipe{gdn_pipe_qk}",
            sources=[src],
            extra_include_paths=include_paths,
            extra_cuda_cflags=[
                "-O3",
                "-w",
                "-std=c++17",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                f"-DGDN_BV={gdn_bv}",
                f"-DGDN_PERSISTENT_BLOCKS_PER_SM={gdn_pb}",
                f"-DGDN_WARPS_PER_CTA={gdn_warp_cta}",
                f"-DGDN_PIPELINE_QK={gdn_pipe_qk}",
            ],
            verbose=False,
        )
    finally:
        if prev_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev_arch_list
    return ext


def make_inputs(seq_len: int, nh: int, kdim: int, vdim: int, device: torch.device, seed: int):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = torch.randn(seq_len, nh, kdim, device=device, dtype=torch.float32, generator=gen)
    k = torch.randn(seq_len, nh, kdim, device=device, dtype=torch.float32, generator=gen)
    v = torch.randn(seq_len, nh, vdim, device=device, dtype=torch.float32, generator=gen)
    # Match engine preprocessing for the recurrent path: L2 normalize q/k and scale q by 1/sqrt(K).
    q = torch.nn.functional.normalize(q, dim=-1) * (1.0 / math.sqrt(kdim))
    k = torch.nn.functional.normalize(k, dim=-1)
    # Gate values in the same range as engine_compute_gates outputs.
    g = torch.empty(seq_len, nh, device=device, dtype=torch.float32).uniform_(-6.0, 0.0, generator=gen)
    beta = torch.empty(seq_len, nh, device=device, dtype=torch.float32).uniform_(0.0, 1.0, generator=gen)
    return q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), g, beta


@torch.no_grad()
def run_reference(q, k, v, g, beta):
    # q/k are expected to already be normalized/scaled exactly as the engine sees them.
    t, nh, kdim = q.shape
    vdim = v.shape[-1]
    qf = q.float()
    kf = k.float()
    vf = v.float()
    gf = torch.exp(g.float())
    bf = beta.float()
    state = torch.zeros(nh, kdim, vdim, device=q.device, dtype=torch.float32)
    out = torch.zeros(t, nh, vdim, device=q.device, dtype=torch.float32)
    for ti in range(t):
        for hi in range(nh):
            st = state[hi]
            st.mul_(gf[ti, hi])
            old_v = torch.matmul(kf[ti, hi], st)  # [V]
            vn = bf[ti, hi] * (vf[ti, hi] - old_v)
            st.add_(torch.outer(kf[ti, hi], vn))
            out[ti, hi] = torch.matmul(qf[ti, hi], st)
    return out.to(torch.bfloat16), state


def benchmark(ext, q, k, v, g, beta, iters: int, warmup: int):
    for _ in range(warmup):
        ext.run_gdn_recurrent(q, k, v, g, beta)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out, state = ext.run_gdn_recurrent(q, k, v, g, beta)
    end.record()
    end.synchronize()
    return out, state, start.elapsed_time(end) / iters


def correctness(ext, q, k, v, g, beta):
    out_ref, state_ref = run_reference(q, k, v, g, beta)
    out, state = ext.run_gdn_recurrent(q, k, v, g, beta)
    out_cos = torch.nn.functional.cosine_similarity(
        out_ref.flatten().float().unsqueeze(0), out.flatten().float().unsqueeze(0)
    ).item()
    state_cos = torch.nn.functional.cosine_similarity(
        state_ref.flatten().float().unsqueeze(0), state.flatten().float().unsqueeze(0)
    ).item()
    print(f"out_cos={out_cos:.6f}", flush=True)
    print(f"state_cos={state_cos:.6f}", flush=True)
    print(f"out_max_abs={(out_ref.float() - out.float()).abs().max().item():.6f}", flush=True)
    print(f"state_max_abs={(state_ref - state).abs().max().item():.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=13200)
    parser.add_argument("--nh", type=int, default=20)
    parser.add_argument("--kdim", type=int, default=256)
    parser.add_argument("--vdim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--correctness", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"shape: T={args.seq_len} NH={args.nh} K={args.kdim} V={args.vdim}", flush=True)
    t0 = time.time()
    ext = init_ext(args.seq_len, args.nh, args.kdim, args.vdim)
    print(f"ext_ready_s={time.time()-t0:.3f}", flush=True)
    q, k, v, g, beta = make_inputs(args.seq_len, args.nh, args.kdim, args.vdim, device, args.seed)
    if args.correctness:
        correctness(ext, q, k, v, g, beta)
        return
    out, state, avg_ms = benchmark(ext, q, k, v, g, beta, args.iters, args.warmup)
    print(f"avg_ms={avg_ms:.4f}", flush=True)
    print(f"out_norm={out.float().norm().item():.4f}", flush=True)
    print(f"state_norm={state.norm().item():.4f}", flush=True)


if __name__ == "__main__":
    main()
