#!/usr/bin/env python3
"""
Profile Kairos 4B stock model to identify optimization targets.

Usage:
    # Quick nsys profile (timeline):
    nsys profile -o kairos_stock python kairos_ext/profile_kairos.py --mode stock

    # Quick stock timing (no profiler overhead):
    python kairos_ext/profile_kairos.py --mode stock --attn-backend sdpa

    # Profile engine:
    python kairos_ext/profile_kairos.py --mode engine

    # NCU kernel-level profile (slow, picks specific kernels):
    ncu --set full -o kairos_ncu python kairos_ext/profile_kairos.py --mode stock --iters 1
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "kairos", "third_party"))
import kairos_ext._apex_shim  # noqa: F401 — install FusedRMSNorm shim before kairos imports

import torch
import torch.nn as nn

# Force SDPA fallback mode for flash_attention when requested.
os.environ["KAIROS_COMPAT_ATTN"] = "1"


def build_dit(device="cuda", attn_backend="sdpa"):
    """Build KairosDiT with standard 4B config."""
    # Avoid heavy transitive imports (modelscope, mmengine pipelines)
    import importlib, types
    # Stub out kairos.apis.builder so kairos_dit.py can import DITS
    kairos_apis = types.ModuleType("kairos.apis")
    kairos_apis_builder = types.ModuleType("kairos.apis.builder")
    class _Registry:
        def register_module(self, **kw):
            def dec(cls): return cls
            return dec
    kairos_apis_builder.DITS = _Registry()
    sys.modules.setdefault("kairos.apis", kairos_apis)
    sys.modules.setdefault("kairos.apis.builder", kairos_apis_builder)
    # Stub parallel_state if not initialized
    if "kairos.modules.dits.kairos_dit" not in sys.modules:
        ps_mod = types.ModuleType("kairos.third_party.fla.ops.utils.parallel_state")
        ps_mod.get_context_parallel_rank = lambda: 0
        ps_mod.get_context_parallel_world_size = lambda: 1
        ps_mod.get_context_parallel_group = lambda: None
        sys.modules.setdefault("kairos.third_party.fla.ops.utils.parallel_state", ps_mod)

    # Disable torch.compile (causes issues with monkeypatching)
    import torch._dynamo as _dynamo
    _dynamo.config.suppress_errors = True
    _dynamo.reset()

    os.environ["KAIROS_ATTN_BACKEND"] = attn_backend

    from kairos.modules.dits.kairos_dit import KairosDiT
    import kairos.modules.dits.kairos_dit as _kd
    if attn_backend == "sdpa":
        # Force SDPA fallback (no flash_attn / SageAttention dependency).
        _kd.FLASH_ATTN_2_AVAILABLE = False
        _kd.FLASH_ATTN_3_AVAILABLE = False
        _kd.SAGE_ATTN_AVAILABLE = False
        if hasattr(_kd, "SAGE3_ATTN_AVAILABLE"):
            _kd.SAGE3_ATTN_AVAILABLE = False
        # Patch flash_attention to always use compatibility mode.
        _orig_fa = _kd.flash_attention
        def _compat_fa(q, k, v, num_heads, compatibility_mode=False, attn_mask=None, window_size=(-1,-1), return_attn_probs=False):
            return _orig_fa(q, k, v, num_heads, compatibility_mode=True, attn_mask=attn_mask, window_size=window_size, return_attn_probs=return_attn_probs)
        _kd.flash_attention = _compat_fa

    # Unwrap torch.compile from DiTBlock.forward
    if hasattr(_kd.DiTBlock.forward, '__wrapped__'):
        _kd.DiTBlock.forward = _kd.DiTBlock.forward.__wrapped__
    # Also disable compile decorator for new instances
    _kd.DiTBlock.forward = torch.compiler.disable(_kd.DiTBlock.forward)

    dit = KairosDiT(
        has_image_input=False,
        patch_size=[1, 2, 2],
        in_dim=16,
        dim=2560,
        ffn_dim=10240,
        freq_dim=256,
        text_dim=3584,
        out_dim=16,
        num_heads=20,
        num_layers=32,
        eps=1e-6,
        seperated_timestep=True,
        require_clip_embedding=False,
        require_vae_embedding=False,
        fuse_vae_embedding_in_latents=True,
        dilated_lengths=[1, 1, 4, 1],
        use_first_frame_cond=False,
        use_seq_parallel=False,
        use_tp_in_getaeddeltanet=False,
        use_tp_in_self_attn=False,
        attend_k0=False,
    )
    return dit


def make_block_inputs(dit, seq_len, ctx_len, device="cuda"):
    """Create inputs for block-level forward."""
    dim = 2560
    nh = 20
    hd = dim // nh
    x = torch.randn(1, seq_len, dim, device=device, dtype=torch.bfloat16)
    context = torch.randn(1, ctx_len, dim, device=device, dtype=torch.bfloat16)
    t_mod = torch.randn(1, 6, dim, device=device, dtype=torch.float32) * 0.1
    freqs = dit.freqs
    # Factor seq_len into grid
    import math
    F = 1
    rem = seq_len
    for f_try in [21, 13, 7, 5, 3, 1]:
        if rem % f_try == 0:
            F = f_try
            rem //= f_try
            break
    H = int(math.isqrt(rem))
    while rem % H != 0:
        H -= 1
    W = rem // H
    grid_size = (F, H, W)
    freq_rope = torch.cat([
        freqs[0][:F].view(F, 1, 1, -1).expand(F, H, W, -1),
        freqs[1][:H].view(1, H, 1, -1).expand(F, H, W, -1),
        freqs[2][:W].view(1, 1, W, -1).expand(F, H, W, -1),
    ], dim=-1).reshape(F * H * W, 1, -1).to(device)
    return x, context, t_mod, freq_rope, grid_size


@torch.no_grad()
def profile_stock(dit, x, context, t_mod, freqs, grid_size, warmup=3, iters=5):
    """Profile stock model forward through blocks."""
    # Warmup
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(warmup):
            x_in = x.clone()
            for block in dit.blocks:
                x_in = block(x_in, context, t_mod, freqs, grid_size)
    torch.cuda.synchronize()

    # Per-layer timing
    print("\n--- Per-layer timing (stock) ---")
    layer_times = []
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for li, block in enumerate(dit.blocks):
            x_in = x.clone()
            for j in range(li):
                x_in = dit.blocks[j](x_in, context, t_mod, freqs, grid_size)
            torch.cuda.synchronize()
            times = []
            for _ in range(iters):
                x_iter = x_in.clone()
                torch.cuda.synchronize()
                t0 = time.time()
                x_iter = block(x_iter, context, t_mod, freqs, grid_size)
                torch.cuda.synchronize()
                times.append((time.time() - t0) * 1000)
            avg = sum(times) / len(times)
            kind = "GDN" if block.use_linear_attn else "QAT"
            layer_times.append((li, kind, avg))
            print(f"  Layer {li:2d} [{kind}]: {avg:7.2f} ms")

    # Full forward timing
    torch.cuda.synchronize()
    full_times = []
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(iters):
            x_in = x.clone()
            torch.cuda.synchronize()
            t0 = time.time()
            for block in dit.blocks:
                x_in = block(x_in, context, t_mod, freqs, grid_size)
            torch.cuda.synchronize()
            full_times.append((time.time() - t0) * 1000)

    avg_full = sum(full_times) / len(full_times)
    qat_total = sum(t for _, k, t in layer_times if k == "QAT")
    gdn_total = sum(t for _, k, t in layer_times if k == "GDN")
    print(f"\n--- Summary (stock) ---")
    print(f"  Full forward:  {avg_full:.1f} ms")
    print(f"  QAT layers:    {qat_total:.1f} ms ({qat_total/avg_full*100:.0f}%)")
    print(f"  GDN layers:    {gdn_total:.1f} ms ({gdn_total/avg_full*100:.0f}%)")
    print(f"  QAT avg/layer: {qat_total/24:.2f} ms")
    print(f"  GDN avg/layer: {gdn_total/8:.2f} ms")
    return avg_full


@torch.no_grad()
def profile_engine(dit, x, context, t_mod, freqs, grid_size, seq_len, ctx_len,
                   warmup=3, iters=5, engine_sdpa_backend="cudnn"):
    """Profile engine-patched model."""
    os.environ["KAIROS_ENGINE_SDPA_BACKEND"] = engine_sdpa_backend
    from kairos_ext.kairos_engine_patch import patch_engine
    nl = patch_engine(dit, max_seq=seq_len, ctx_len=ctx_len,
                      seq_list=[seq_len], verbose=True)

    engine_runner = getattr(dit, "_kairos_engine_run_blocks", None)

    def run_engine_once(x_in):
        if engine_runner is not None:
            return engine_runner(x_in, context, t_mod, freqs, grid_size)
        for block in dit.blocks:
            x_in = block(x_in, context, t_mod, freqs, grid_size)
        return x_in

    # Warmup
    for _ in range(warmup):
        x_in = x.clone()
        x_in = run_engine_once(x_in)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(iters):
        x_in = x.clone()
        torch.cuda.synchronize()
        t0 = time.time()
        x_in = run_engine_once(x_in)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)

    avg = sum(times) / len(times)
    mn = min(times)
    print(f"\n--- Engine timing ---")
    print(f"  avg={avg:.1f} ms, min={mn:.1f} ms ({iters} iters)")
    return avg


def _prepare_engine_inputs(x, context, t_mod, freqs, hd):
    xi = x.squeeze(0).contiguous() if x.dim() == 3 else x.contiguous()
    enci = context.squeeze(0).contiguous() if context.dim() == 3 else context.contiguous()

    ei = t_mod.float().contiguous()
    while ei.dim() > 3:
        ei = ei.squeeze(2) if ei.shape[2] == 1 else ei.squeeze(0)
    if ei.dim() == 2:
        ei = ei.unsqueeze(0)

    from kairos_ext.kairos_engine_patch import _kairos_rope_to_helios
    if freqs is not None and freqs.numel() > 0:
        if freqs.is_complex():
            rope = _kairos_rope_to_helios(freqs.detach(), hd).to(xi.device)
        else:
            rope = freqs.float().contiguous()
            if rope.dim() == 3:
                rope = rope.squeeze(0)
    else:
        rope = torch.empty(0, device=xi.device, dtype=torch.float32)
    return xi, enci, ei, rope


@torch.no_grad()
def profile_engine_breakdown(dit, x, context, t_mod, freqs, grid_size, seq_len, ctx_len,
                             warmup=1, iters=2, engine_sdpa_backend="cudnn"):
    del grid_size
    os.environ["KAIROS_ENGINE_SDPA_BACKEND"] = engine_sdpa_backend
    from kairos_ext.kairos_engine_patch import patch_engine, get_ext

    pre_blocks = list(dit.blocks)
    first_nh = getattr(pre_blocks[0].self_attn, "num_heads", 20)
    hd = x.shape[-1] // first_nh
    ssts_stacked = torch.stack([b.modulation.data.float().squeeze(0) for b in pre_blocks]).contiguous()
    gdn_layers = [i for i, b in enumerate(pre_blocks) if getattr(b, "use_linear_attn", False)]
    quad_layers = [i for i, b in enumerate(pre_blocks) if not getattr(b, "use_linear_attn", False)]

    nl = patch_engine(dit, max_seq=seq_len, ctx_len=ctx_len,
                      seq_list=[seq_len], verbose=True)
    ext = get_ext()

    xi, enci, ei, rope = _prepare_engine_inputs(x, context, t_mod, freqs, hd)
    ca_len = xi.shape[0]

    ext.reset_gdn_state()
    ext.build_ca_kv_cache(enci)
    ext.prepare_ca(ca_len)

    def time_range(start_layer, end_layer, profile_mode=0, warmup_local=warmup, iters_local=iters):
        ext.set_profile_mode(profile_mode)
        for _ in range(warmup_local):
            ext.reset_gdn_state()
            _ = ext.forward(xi.clone(), enci, ei, ssts_stacked, rope, ca_len, start_layer, end_layer)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters_local)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters_local)]
        times = []
        for i in range(iters_local):
            ext.reset_gdn_state()
            starts[i].record()
            _ = ext.forward(xi.clone(), enci, ei, ssts_stacked, rope, ca_len, start_layer, end_layer)
            ends[i].record()
        torch.cuda.synchronize()
        for i in range(iters_local):
            times.append(starts[i].elapsed_time(ends[i]))
        return sum(times) / len(times), min(times)

    print("\n--- Engine Layer Breakdown ---")
    full_avg, full_min = time_range(0, nl)
    print(f"  Full engine: avg={full_avg:.1f} ms, min={full_min:.1f} ms")

    layer_rows = []
    for li in range(nl):
        avg_ms, min_ms = time_range(li, li + 1, warmup_local=0, iters_local=1)
        kind = "GDN" if li in gdn_layers else "QAT"
        layer_rows.append((avg_ms, min_ms, li, kind))

    layer_rows.sort(reverse=True)
    print("\n  Slowest individual layers:")
    for avg_ms, min_ms, li, kind in layer_rows[:8]:
        print(f"    Layer {li:2d} [{kind}]: {avg_ms:7.2f} ms")

    quad_times = [avg for avg, _, _, kind in layer_rows if kind == "QAT"]
    gdn_times = [avg for avg, _, _, kind in layer_rows if kind == "GDN"]
    print("\n  Layer-type averages:")
    if quad_times:
        print(f"    QAT avg/layer: {sum(quad_times)/len(quad_times):.2f} ms")
    if gdn_times:
        print(f"    GDN avg/layer: {sum(gdn_times)/len(gdn_times):.2f} ms")

    mode_names = {
        0: "all",
        1: "gemm_only",
        2: "ewise_only",
        3: "sdpa_only",
        4: "no_sdpa",
        5: "no_gemm",
        6: "no_ewise",
    }
    rep_quad = next((li for _, _, li, kind in layer_rows if kind == "QAT"), None)
    rep_gdn = next((li for _, _, li, kind in layer_rows if kind == "GDN"), None)

    for title, li in (("Representative QAT", rep_quad), ("Representative GDN", rep_gdn)):
        if li is None:
            continue
        print(f"\n  {title} layer {li}:")
        for mode in (0, 1, 2, 3, 4, 5, 6):
            avg_ms, _ = time_range(li, li + 1, profile_mode=mode, warmup_local=0, iters_local=1)
            print(f"    {mode_names[mode]:>10}: {avg_ms:7.2f} ms")
        ext.set_profile_mode(0)

    return {
        "full_avg_ms": full_avg,
        "full_min_ms": full_min,
        "layer_rows": layer_rows,
        "quad_avg_ms": (sum(quad_times) / len(quad_times)) if quad_times else None,
        "gdn_avg_ms": (sum(gdn_times) / len(gdn_times)) if gdn_times else None,
        "rep_quad": rep_quad,
        "rep_gdn": rep_gdn,
    }


@torch.no_grad()
def profile_with_torch_profiler(dit, x, context, t_mod, freqs, grid_size, tag="stock"):
    """Use torch.profiler for detailed kernel-level breakdown."""
    from torch.profiler import profile, ProfilerActivity, schedule

    # Warmup
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(2):
            x_in = x.clone()
            for block in dit.blocks:
                x_in = block(x_in, context, t_mod, freqs, grid_size)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            x_in = x.clone()
            for block in dit.blocks:
                x_in = block(x_in, context, t_mod, freqs, grid_size)
            torch.cuda.synchronize()

    # Print CUDA kernel summary
    print(f"\n--- Torch Profiler: Top CUDA kernels ({tag}) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Save trace
    trace_path = f"/tmp/kairos_{tag}_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"  Trace saved to {trace_path}")


@torch.no_grad()
def correctness_check(args):
    """Run stock and engine on identical inputs, compare outputs per-layer and overall."""
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("=" * 60)
    print("Correctness Check: Stock vs Engine")
    print("=" * 60)

    seq_len, ctx_len = args.seq_len, args.ctx_len

    # Build stock model
    print("\nBuilding stock model...")
    dit_stock = build_dit(device=device)
    dit_stock = dit_stock.to(device).to(torch.bfloat16).eval()

    x, context, t_mod, freqs, grid_size = make_block_inputs(
        dit_stock, seq_len, ctx_len, device
    )
    print(f"  Grid: {grid_size}, seq={seq_len}, ctx={ctx_len}")

    # Run stock forward
    print("\nRunning stock forward...")
    x_stock = x.clone()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        # Collect per-layer outputs
        stock_layer_outs = []
        x_s = x_stock.clone()
        for li, block in enumerate(dit_stock.blocks):
            x_s = block(x_s, context, t_mod, freqs, grid_size)
            stock_layer_outs.append(x_s.clone())
    torch.cuda.synchronize()
    y_stock = stock_layer_outs[-1]
    print(f"  Stock output: shape={y_stock.shape}, norm={y_stock.float().norm().item():.2f}")

    # Save stock state dict to CPU, free stock model
    print("\nSaving stock weights to CPU, freeing stock model...")
    stock_sd = {k: v.cpu() for k, v in dit_stock.state_dict().items()}
    del dit_stock, stock_layer_outs
    torch.cuda.empty_cache()
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # Build engine model with same weights
    print("Building engine model...")
    os.environ["KAIROS_ENGINE_SDPA_BACKEND"] = args.engine_sdpa_backend
    dit_engine = build_dit(device=device)
    dit_engine = dit_engine.to(device).to(torch.bfloat16).eval()
    dit_engine.load_state_dict(stock_sd)
    del stock_sd

    from kairos_ext.kairos_engine_patch import patch_engine
    nl = patch_engine(dit_engine, max_seq=seq_len, ctx_len=ctx_len,
                      seq_list=[seq_len], verbose=True)

    # Run engine forward
    print("\nRunning engine forward...")
    x_e = x.clone()
    for block in dit_engine.blocks:
        x_e = block(x_e, context, t_mod, freqs, grid_size)
    torch.cuda.synchronize()
    y_engine = x_e
    print(f"  Engine output: shape={y_engine.shape}, norm={y_engine.float().norm().item():.2f}")

    # Overall comparison
    y_s = y_stock.flatten().float()
    y_e = y_engine.flatten().float()
    cos_sim = torch.nn.functional.cosine_similarity(y_s.unsqueeze(0), y_e.unsqueeze(0)).item()
    abs_diff = (y_s - y_e).abs()
    rel_diff = abs_diff / (y_s.abs().clamp(min=1e-6))

    print(f"\n{'='*60}")
    print(f"OVERALL CORRECTNESS (layer {nl-1} output)")
    print(f"{'='*60}")
    print(f"  Cosine similarity:  {cos_sim:.6f}")
    print(f"  Max abs diff:       {abs_diff.max().item():.6f}")
    print(f"  Mean abs diff:      {abs_diff.mean().item():.6f}")
    print(f"  Max rel diff:       {rel_diff.max().item():.6f}")
    print(f"  Mean rel diff:      {rel_diff.mean().item():.6f}")

    if cos_sim > 0.999:
        print(f"  Result: EXCELLENT (cos_sim > 0.999)")
    elif cos_sim > 0.99:
        print(f"  Result: GOOD (cos_sim > 0.99, typical for FP8 quantization)")
    elif cos_sim > 0.95:
        print(f"  Result: ACCEPTABLE (cos_sim > 0.95, significant FP8 drift)")
    elif cos_sim > 0.8:
        print(f"  Result: POOR (cos_sim > 0.8, possible bug)")
    else:
        print(f"  Result: FAIL (cos_sim < 0.8, likely correctness bug)")

    print(f"\n  Stock norm:  {y_stock.float().norm().item():.2f}")
    print(f"  Engine norm: {y_engine.float().norm().item():.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stock", "engine", "engine_breakdown", "both", "torch_prof", "correctness"], default="stock")
    parser.add_argument("--attn-backend", choices=["sdpa", "flashattention", "sageattention", "sageattention3", "auto"], default="sdpa")
    parser.add_argument("--engine-sdpa-backend", choices=["cudnn", "torch", "sage3_py", "sage3_cpp"], default="cudnn")
    parser.add_argument("--seq-len", type=int, default=7800)
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"GPU: {torch.cuda.get_device_name(0)}, SM {torch.cuda.get_device_capability(0)}")
    print(f"seq_len={args.seq_len}, ctx_len={args.ctx_len}")
    print(f"attn_backend={args.attn_backend}")
    print(f"engine_sdpa_backend={args.engine_sdpa_backend}")

    print("\nBuilding KairosDiT...")
    dit = build_dit(device=device, attn_backend=args.attn_backend)
    dit = dit.to(device).to(torch.bfloat16).eval()
    print("  Using synthetic (random) weights")

    x, context, t_mod, freqs, grid_size = make_block_inputs(
        dit, args.seq_len, args.ctx_len, device
    )
    print(f"  Grid: {grid_size}")

    if args.mode == "correctness":
        correctness_check(args)
        return

    if args.mode == "stock" or args.mode == "both":
        stock_ms = profile_stock(dit, x, context, t_mod, freqs, grid_size,
                                 warmup=args.warmup, iters=args.iters)

    if args.mode == "torch_prof":
        profile_with_torch_profiler(dit, x, context, t_mod, freqs, grid_size, "stock")

    if args.mode == "engine" or args.mode == "both":
        if args.mode == "both":
            # Rebuild model for engine (patch_engine modifies in place)
            dit = build_dit(device=device, attn_backend=args.attn_backend).to(device).to(torch.bfloat16).eval()
        engine_ms = profile_engine(dit, x, context, t_mod, freqs, grid_size,
                                   args.seq_len, args.ctx_len,
                                   warmup=args.warmup, iters=args.iters,
                                   engine_sdpa_backend=args.engine_sdpa_backend)
        if args.mode == "both":
            print(f"\n--- Speedup: {stock_ms/engine_ms:.2f}x ---")
    elif args.mode == "engine_breakdown":
        profile_engine_breakdown(dit, x, context, t_mod, freqs, grid_size,
                                 args.seq_len, args.ctx_len,
                                 warmup=max(1, min(args.warmup, 2)),
                                 iters=max(1, min(args.iters, 2)),
                                 engine_sdpa_backend=args.engine_sdpa_backend)


if __name__ == "__main__":
    main()
