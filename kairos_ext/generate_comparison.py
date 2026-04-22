#!/usr/bin/env python3
"""
Generate side-by-side video comparison: Stock (PyTorch+Triton) vs Engine (CUDA).

Usage:
    TORCHDYNAMO_DISABLE=1 \
    CUDNN_INC=... CUDNN_LIB=... NCCL_HOME=... \
    PYTHONPATH=kairos/third_party:$PYTHONPATH \
    python kairos_ext/generate_comparison.py \
        --config kairos/configs/kairos_4b_config_DMD.py \
        --input examples/example_t2v_480P.json \
        --output output/comparison
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kairos_ext._apex_shim  # noqa

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Fix flash_attn: if it's a stub without flash_attn_func, patch for SDPA fallback
def _patch_flash_attn():
    """If flash_attn is missing or a stub, patch kairos_dit to use SDPA."""
    try:
        from flash_attn import flash_attn_func  # noqa
        return  # Real flash_attn available
    except (ImportError, AttributeError):
        pass

    # Monkey-patch: make flash_attn.flash_attn_func use SDPA
    from einops import rearrange
    import torch.nn.functional as F

    def _sdpa_flash_attn_func(q, k, v, window_size=(-1, -1), return_attn_probs=False):
        # q,k,v: [B, S, N, D] -> rearrange to [B, N, S, D] for SDPA
        q2 = rearrange(q, "b s n d -> b n s d")
        k2 = rearrange(k, "b s n d -> b n s d")
        v2 = rearrange(v, "b s n d -> b n s d")
        x = F.scaled_dot_product_attention(q2, k2, v2)
        x = rearrange(x, "b n s d -> b s n d")
        if return_attn_probs:
            return x, None
        return x

    import flash_attn as _fa
    _fa.flash_attn_func = _sdpa_flash_attn_func
    print("  Patched flash_attn with SDPA fallback")

_patch_flash_attn()

# Fix transformers compat: Qwen2RMSNorm renamed in newer versions
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm
except ImportError:
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRMSNorm as _rms
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qmod
        _qmod.Qwen2RMSNorm = _rms
    except ImportError:
        pass

import torch
import torch.distributed as dist
import numpy as np


def build_pipeline(cfg_path):
    """Build the Kairos pipeline from config (single-GPU, no dist)."""
    from mmengine import Config
    from kairos.modules.utils import parallel_state

    parallel_state.reset_cfg()
    cfg = Config.fromfile(cfg_path)

    # Force single-GPU, no parallel
    cfg.pipeline.pipeline_args.dit_config["use_seq_parallel"] = False
    cfg.pipeline.pipeline_args.dit_config["use_tp_in_getaeddeltanet"] = False
    cfg.pipeline.pipeline_args.dit_config["use_tp_in_self_attn"] = False
    cfg.pipeline.parallel_mode = "none"
    cfg.pipeline.use_cfg_parallel = False

    # Fix weight paths if default doesn't exist
    dit_path = cfg.pipeline.get("pretrained_dit", "")
    if dit_path and not os.path.exists(dit_path):
        alt = "models/Kairos-model/kairos-sensenova-robot-4B-480P-distilled/kairos-robot-4B-480P-distilled.safetensors"
        if os.path.exists(alt):
            cfg.pipeline.pretrained_dit = alt
            print(f"  Using alt weight path: {alt}")

    from kairos.apis.builder import build_model_pipeline
    pipeline = build_model_pipeline(cfg.pipeline)
    return pipeline, cfg


def find_dit(api_or_pipeline):
    """Find the DiT model inside the API/pipeline hierarchy."""
    # api.pipe is the pipeline, pipeline.dit or pipeline.model is the DiT
    obj = api_or_pipeline
    if hasattr(obj, 'pipe'):
        obj = obj.pipe
    # Search for DiT by attribute name
    for attr in ['dit', 'model', 'transformer', 'backbone']:
        if hasattr(obj, attr):
            candidate = getattr(obj, attr)
            if hasattr(candidate, 'blocks') and hasattr(candidate, 'head'):
                return candidate
    # Fallback: search all submodules
    for name, module in obj.named_modules():
        if hasattr(module, 'blocks') and hasattr(module, 'head'):
            return module
    raise RuntimeError("Could not find DiT model in pipeline")


def patch_dit_engine(api_or_pipeline, max_seq=32760, ctx_len=512):
    """Patch the DiT inside the pipeline with our CUDA engine."""
    from kairos_ext.kairos_engine_patch import patch_engine
    dit = find_dit(api_or_pipeline)
    print(f"  Found DiT: {type(dit).__name__} with {len(dit.blocks)} blocks")
    nl = patch_engine(dit, max_seq=max_seq, ctx_len=ctx_len, verbose=True)
    return nl


def patch_vae_engine(api_or_pipeline):
    """Patch the VAE decoder with fused RMSNorm+SiLU CUDA kernels."""
    from kairos_ext.vae_patch import patch_vae
    obj = api_or_pipeline
    if hasattr(obj, 'pipe'):
        obj = obj.pipe
    # Find VAE: pipeline.vae or pipeline.vae_decode
    vae = None
    for attr in ['vae', 'vae_decode', 'video_vae']:
        if hasattr(obj, attr):
            vae = getattr(obj, attr)
            break
    if vae is None:
        print("  WARNING: Could not find VAE in pipeline, skipping VAE patch")
        return 0
    print(f"  Found VAE: {type(vae).__name__}")
    return patch_vae(vae, verbose=True)


def generate_video(pipeline, input_args, seed=0):
    """Run inference and return (video_tensor, elapsed_seconds, num_frames)."""
    prompt = input_args["prompt"]
    negative_prompt = input_args.get("negative_prompt", "")
    height = input_args.get("height", 480)
    width = input_args.get("width", 640)
    num_frames = input_args.get("num_frames", 81)
    cfg_scale = input_args.get("cfg_scale", 5)
    tiled = input_args.get("tiled", True)

    print(f"  Generating {num_frames} frames at {height}x{width}...")
    torch.cuda.synchronize()
    t0 = time.time()

    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        cfg_scale=cfg_scale,
        seed=seed,
        tiled=tiled,
    )

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    # Extract video — pipeline may return dict, tuple, or tensor
    if isinstance(result, dict):
        video = result.get("video", result.get("frames", result.get("output")))
    elif isinstance(result, (list, tuple)):
        video = result[0] if len(result) == 1 else result
    else:
        video = result

    return video, elapsed, num_frames


def save_video_with_fps(video, fps_text, output_path, video_fps=16):
    """Save video with FPS overlay text burned in."""
    try:
        import cv2
    except ImportError:
        print("  WARNING: cv2 not available, saving without FPS overlay")
        from kairos.modules.utils import save_video
        save_video(video, output_path, fps=video_fps)
        return

    # Convert video to numpy if tensor
    if torch.is_tensor(video):
        # Expected: [C, T, H, W] or [T, H, W, C] or [T, C, H, W]
        v = video.cpu().float()
        if v.dim() == 4:
            if v.shape[0] == 3:  # [C, T, H, W]
                v = v.permute(1, 2, 3, 0)  # → [T, H, W, C]
            elif v.shape[1] == 3:  # [T, C, H, W]
                v = v.permute(0, 2, 3, 1)  # → [T, H, W, C]
        v = (v.clamp(0, 1) * 255).byte().numpy()
    else:
        v = np.array(video)
        if v.max() <= 1.0:
            v = (v * 255).astype(np.uint8)

    T, H, W, C = v.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, video_fps, (W, H))

    for t in range(T):
        frame = v[t]
        if C == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Burn FPS text
        cv2.putText(frame, fps_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        writer.write(frame)
    writer.release()
    print(f"  Saved: {output_path} ({T} frames)")


def stitch_side_by_side(left_path, right_path, output_path, left_label, right_label):
    """Stitch two videos side-by-side."""
    try:
        import cv2
    except ImportError:
        print("  WARNING: cv2 not available, skipping stitch")
        return

    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)

    fps = cap_l.get(cv2.CAP_PROP_FPS)
    w_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))

    h_out = max(h_l, h_r)
    w_out = w_l + w_r + 4  # 4px gap

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w_out, h_out))

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        # Resize if heights differ
        if frame_l.shape[0] != h_out:
            frame_l = cv2.resize(frame_l, (w_l, h_out))
        if frame_r.shape[0] != h_out:
            frame_r = cv2.resize(frame_r, (w_r, h_out))

        # Add label bars
        cv2.putText(frame_l, left_label, (10, h_out - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_r, right_label, (10, h_out - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Gap
        gap = np.zeros((h_out, 4, 3), dtype=np.uint8)
        combined = np.hstack([frame_l, gap, frame_r])
        writer.write(combined)

    cap_l.release()
    cap_r.release()
    writer.release()
    print(f"  Side-by-side: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="kairos/configs/kairos_4b_config_DMD.py")
    parser.add_argument("--input", default="examples/example_t2v_480P.json")
    parser.add_argument("--output", default="output/comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-fps", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {args.config}")
    print(f"Input: {args.input}")

    with open(args.input) as f:
        input_args = json.load(f)

    stock_compiled = os.environ.get("TORCHDYNAMO_DISABLE", "") not in {"1", "true", "TRUE"}
    engine_sdpa_backend = os.environ.get("KAIROS_ENGINE_SDPA_BACKEND", "").strip().lower()
    left_label = "STOCK (PyTorch+Triton+compile)" if stock_compiled else "STOCK (PyTorch+Triton)"
    if engine_sdpa_backend == "sage3_cpp":
        right_label = "ENGINE (FP8 + Sage3 C++)"
    elif engine_sdpa_backend == "sage3_py":
        right_label = "ENGINE (FP8 + Sage3 Python)"
    elif engine_sdpa_backend:
        right_label = f"ENGINE ({engine_sdpa_backend})"
    else:
        right_label = "ENGINE (CUDA FP8+MMA)"

    # ========== STOCK ==========
    print("\n" + "=" * 60)
    print("STOCK MODEL (PyTorch + Triton)")
    print("=" * 60)

    pipeline, cfg = build_pipeline(args.config)
    video_stock, elapsed_stock, nf = generate_video(pipeline, input_args, seed=args.seed)
    fps_stock = nf / elapsed_stock

    stock_path = os.path.join(args.output, "stock.mp4")
    fps_text_stock = f"Stock: {fps_stock:.2f} FPS ({elapsed_stock:.1f}s)"
    print(f"  {fps_text_stock}")
    save_video_with_fps(video_stock, fps_text_stock, stock_path, args.video_fps)

    # Free stock pipeline + video tensor to reclaim VRAM for engine
    del pipeline, video_stock
    torch.cuda.empty_cache()
    import gc; gc.collect()
    torch.cuda.empty_cache()
    print(f"  VRAM after cleanup: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ========== ENGINE ==========
    print("\n" + "=" * 60)
    print("ENGINE MODEL (CUDA FP8 + MMA)")
    print("=" * 60)

    pipeline_eng, _ = build_pipeline(args.config)

    # Estimate max_seq from video dims
    # VAE: 8x spatial compression, ~4x temporal compression ((T-1)//4 + 1)
    # Patchify: [1, 2, 2] -> additional 2x spatial each dim
    h, w = input_args.get("height", 480), input_args.get("width", 640)
    nf_gen = input_args.get("num_frames", 81)
    T_lat = (nf_gen - 1) // 4 + 1 if nf_gen > 1 else 1  # VAE temporal compression
    H_lat = h // (8 * 2)   # VAE 8x + patch 2x
    W_lat = w // (8 * 2)
    max_seq = T_lat * H_lat * W_lat
    print(f"  Estimated max_seq: {T_lat}×{H_lat}×{W_lat} = {max_seq}")

    patch_dit_engine(pipeline_eng, max_seq=max_seq, ctx_len=512)
    # patch_vae_engine(pipeline_eng)  # disabled — investigating output quality

    video_engine, elapsed_engine, nf = generate_video(pipeline_eng, input_args, seed=args.seed)
    fps_engine = nf / elapsed_engine

    engine_path = os.path.join(args.output, "engine.mp4")
    fps_text_engine = f"Engine: {fps_engine:.2f} FPS ({elapsed_engine:.1f}s)"
    print(f"  {fps_text_engine}")
    save_video_with_fps(video_engine, fps_text_engine, engine_path, args.video_fps)

    # ========== STITCH ==========
    print("\n" + "=" * 60)
    print("STITCHING SIDE-BY-SIDE")
    print("=" * 60)

    comparison_path = os.path.join(args.output, "comparison.mp4")
    stitch_side_by_side(stock_path, engine_path, comparison_path, left_label, right_label)

    # ========== SUMMARY ==========
    speedup = fps_engine / fps_stock if fps_stock > 0 else 0
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Stock:   {fps_stock:.2f} FPS ({elapsed_stock:.1f}s for {nf} frames)")
    print(f"  Engine:  {fps_engine:.2f} FPS ({elapsed_engine:.1f}s for {nf} frames)")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"\n  Videos saved to: {args.output}/")
    print(f"    stock.mp4      - Stock model output")
    print(f"    engine.mp4     - Engine model output")
    print(f"    comparison.mp4 - Side-by-side comparison")


if __name__ == "__main__":
    main()
