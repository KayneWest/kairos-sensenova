#!/usr/bin/env python3
"""
kairos_ext/benchmark_kairos.py — Benchmark the Kairos engine on 5090.

Loads the Kairos 4B DiT, patches it with the CUDA engine, and measures:
  1. Correctness: cosine similarity between stock and engine outputs
  2. Speed: ms/forward for stock vs engine

Usage:
    # Single GPU, synthetic weights (fast smoke test):
    python kairos_ext/benchmark_kairos.py --synthetic

    # Single GPU, real weights:
    python kairos_ext/benchmark_kairos.py --model-path models/Kairos-model/kairos-sensenova-robot-4B-480P-distilled/kairos-robot-4B-480P-distilled.safetensors

    # With timing:
    ENGINE_TIME=1 python kairos_ext/benchmark_kairos.py --synthetic
"""
import argparse
import os
import sys
import time

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kairos_ext._apex_shim  # noqa: install FusedRMSNorm shim

import torch
import torch.nn as nn


def build_dit(dim=2560, num_heads=20, ffn_dim=10240, num_layers=32,
              freq_dim=256, text_dim=3584, device="cuda"):
    """Build a KairosDiT with the standard 4B config."""
    from kairos.modules.dits.kairos_dit import KairosDiT

    dit = KairosDiT(
        has_image_input=False,
        patch_size=[1, 2, 2],
        in_dim=16,
        dim=dim,
        ffn_dim=ffn_dim,
        freq_dim=freq_dim,
        text_dim=text_dim,
        out_dim=16,
        num_heads=num_heads,
        num_layers=num_layers,
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


def load_weights(dit, model_path, device="cuda"):
    """Load safetensors weights into the DiT."""
    from safetensors.torch import load_file
    print(f"Loading weights from {model_path}...")
    state_dict = load_file(model_path, device=str(device))
    # The safetensors may have a prefix — try loading as-is first
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    print(f"  Loaded {len(state_dict)} tensors")


def make_inputs(dit, seq_len=7800, ctx_len=512, device="cuda"):
    """Create synthetic inputs matching Kairos's forward signature."""
    dim = dit.dim
    num_heads = dit.blocks[0].num_heads if hasattr(dit.blocks[0], 'num_heads') else 20

    # x: [B, C, F, H, W] latent — we'll use pre-patchified [B, seq, D] for block-level test
    # For block-level testing, we prepare the intermediate tensors directly
    x = torch.randn(1, seq_len, dim, device=device, dtype=torch.bfloat16)
    context = torch.randn(1, ctx_len, dim, device=device, dtype=torch.bfloat16)

    # t_mod: [B, 6, D] (output of time_projection)
    t_mod = torch.randn(1, 6, dim, device=device, dtype=torch.float32) * 0.1

    # freqs: complex rope [seq, 1, hd/2]
    hd = dim // num_heads
    freqs = dit.freqs  # tuple of 3 complex tensors
    # For testing, create a dummy grid
    # Typical 480p: F=21 frames, H=30, W=52 -> seq=21*30*52 = 32760 or similar
    # For our test seq_len, pick reasonable grid dims
    # seq_len = F * H * W, pick something that works
    import math
    # Try to factor seq_len
    F = 1
    rem = seq_len
    # Simple factoring for test
    for f_try in [21, 13, 7, 5, 3, 1]:
        if rem % f_try == 0:
            F = f_try
            rem = rem // f_try
            break
    H = int(math.isqrt(rem))
    while rem % H != 0:
        H -= 1
    W = rem // H
    grid_size = (F, H, W)
    print(f"  Grid: F={F} H={H} W={W} -> seq={F*H*W}")
    assert F * H * W == seq_len

    # Build RoPE for this grid
    freq_rope = torch.cat([
        freqs[0][:F].view(F, 1, 1, -1).expand(F, H, W, -1),
        freqs[1][:H].view(1, H, 1, -1).expand(F, H, W, -1),
        freqs[2][:W].view(1, 1, W, -1).expand(F, H, W, -1),
    ], dim=-1).reshape(F * H * W, 1, -1).to(device)

    return x, context, t_mod, freq_rope, grid_size


def run_stock(blocks, x, context, t_mod, freqs, grid_size, context_mask=None):
    """Run the stock (unpatched) model block loop."""
    for block in blocks:
        x = block(x, context, t_mod, freqs, grid_size, context_mask=context_mask)
    return x


@torch.no_grad()
def benchmark(args):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("=" * 60)
    print("Kairos Engine Benchmark")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute cap: {torch.cuda.get_device_capability(0)}")

    # Build model
    print("\nBuilding KairosDiT...")
    dit = build_dit(device=device)

    if args.model_path and os.path.exists(args.model_path):
        load_weights(dit, args.model_path, device)
    elif args.synthetic:
        print("  Using random (synthetic) weights")
    else:
        print("  WARNING: No weights loaded and --synthetic not set.")
        print("  Correctness comparison will use random weights.")

    dit = dit.to(device).to(torch.bfloat16).eval()

    # Create inputs
    seq_len = args.seq_len
    ctx_len = args.ctx_len
    print(f"\nCreating inputs: seq_len={seq_len} ctx_len={ctx_len}")
    x, context, t_mod, freqs, grid_size = make_inputs(
        dit, seq_len=seq_len, ctx_len=ctx_len, device=device
    )

    # ---- Stock forward (before patching) ----
    if args.correctness:
        print("\nRunning stock forward...")
        torch.cuda.synchronize()
        x_stock_in = x.clone()
        t0 = time.time()
        y_stock = run_stock(dit.blocks, x_stock_in, context, t_mod, freqs, grid_size)
        torch.cuda.synchronize()
        stock_ms = (time.time() - t0) * 1000
        print(f"  Stock: {stock_ms:.1f} ms")
        y_stock_flat = y_stock.flatten().float()

    # ---- Patch with engine ----
    print("\nPatching with engine...")
    from kairos_ext.kairos_engine_patch import patch_engine
    nl = patch_engine(dit, max_seq=seq_len, ctx_len=ctx_len,
                      seq_list=[seq_len], verbose=True)

    # ---- Engine forward ----
    print("\nRunning engine forward...")
    x_engine_in = x.clone()
    # Warmup
    for _ in range(args.warmup):
        _ = run_stock(dit.blocks, x_engine_in.clone(), context, t_mod, freqs, grid_size)
    torch.cuda.synchronize()

    # Timed run
    times = []
    for _ in range(args.iters):
        x_in = x.clone()
        torch.cuda.synchronize()
        t0 = time.time()
        y_engine = run_stock(dit.blocks, x_in, context, t_mod, freqs, grid_size)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    print(f"  Engine: avg={avg_ms:.1f} ms, min={min_ms:.1f} ms ({args.iters} iters)")

    # ---- Correctness check ----
    if args.correctness:
        y_engine_flat = y_engine.flatten().float()
        cos_sim = torch.nn.functional.cosine_similarity(
            y_stock_flat.unsqueeze(0), y_engine_flat.unsqueeze(0)
        ).item()
        abs_diff = (y_stock_flat - y_engine_flat).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        print(f"\n--- Correctness ---")
        print(f"  cos_sim:   {cos_sim:.6f}")
        print(f"  max_diff:  {max_diff:.6f}")
        print(f"  mean_diff: {mean_diff:.6f}")
        if cos_sim > 0.99:
            print(f"  PASS (cos_sim > 0.99)")
        elif cos_sim > 0.95:
            print(f"  WARN (cos_sim > 0.95, acceptable for FP8)")
        else:
            print(f"  FAIL (cos_sim < 0.95)")
    else:
        print("\n  Skipping correctness check (use --correctness to enable)")

    if args.correctness:
        speedup = stock_ms / avg_ms
        print(f"\n--- Summary ---")
        print(f"  Stock:   {stock_ms:.1f} ms")
        print(f"  Engine:  {avg_ms:.1f} ms (min {min_ms:.1f})")
        print(f"  Speedup: {speedup:.2f}x")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Kairos CUDA engine")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to safetensors weights")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use random weights (smoke test)")
    parser.add_argument("--seq-len", type=int, default=7800,
                        help="Sequence length (F*H*W tokens)")
    parser.add_argument("--ctx-len", type=int, default=512,
                        help="Context (text encoder) length")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10,
                        help="Timing iterations")
    parser.add_argument("--correctness", action="store_true",
                        help="Compare engine vs stock output (slower)")
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
