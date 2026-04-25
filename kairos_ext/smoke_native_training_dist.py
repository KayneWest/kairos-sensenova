#!/usr/bin/env python3
import os
import sys

import torch
import torch.distributed as dist


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "kairos", "third_party"))

import kairos_ext._apex_shim  # noqa: F401
from kairos.modules.dits.kairos_dit import KairosDiT
from kairos_ext.kairos_engine_patch import patch_engine


DEBUG = os.environ.get("KAIROS_DIST_SMOKE_DEBUG", "0") == "1"


def log(rank, msg):
    if DEBUG:
        print(f"[rank {rank}] {msg}", flush=True)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    result = {
        "rank": rank,
        "ok": False,
        "error": None,
        "fallback": None,
        "x_grad_finite": None,
        "context_grad_finite": None,
    }

    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        log(rank, "building model")

        dit = KairosDiT(
            has_image_input=False,
            patch_size=[1, 2, 2],
            in_dim=16,
            dim=512,
            ffn_dim=2048,
            freq_dim=128,
            text_dim=1024,
            out_dim=16,
            num_heads=4,
            num_layers=4,
            eps=1e-6,
            seperated_timestep=True,
            require_clip_embedding=False,
            require_vae_embedding=False,
            fuse_vae_embedding_in_latents=True,
            dilated_lengths=[1, 1, 4, 1],
            use_first_frame_cond=False,
            use_seq_parallel=True,
            use_tp_in_getaeddeltanet=False,
            use_tp_in_self_attn=False,
            attend_k0=False,
        ).to(device=device, dtype=torch.bfloat16)

        log(rank, "patching engine")
        patch_engine(
            dit,
            max_seq=16,
            ctx_len=8,
            enable_training_bridge=True,
            verbose=(rank == 0 and DEBUG),
        )
        dit.train()
        dit._kairos_engine_reset_native_training_fallback_report()
        log(rank, "engine ready")

        x = torch.randn(1, 16, 1, 4, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
        timestep = torch.randint(0, 1000, (1,), device=device).float()
        context = torch.randn(1, 8, 1024, device=device, dtype=torch.bfloat16, requires_grad=True)

        log(rank, "forward start")
        out = dit(x, timestep, context)
        log(rank, "forward done")
        loss = out.float().square().mean()
        log(rank, "backward start")
        loss.backward()
        log(rank, "backward done")

        result["ok"] = bool(torch.isfinite(out).all().item() and torch.isfinite(loss).item())
        result["fallback"] = dit._kairos_engine_native_training_fallback_report()
        result["x_grad_finite"] = bool(torch.isfinite(x.grad).all().item())
        result["context_grad_finite"] = bool(torch.isfinite(context.grad).all().item())
    except Exception as e:
        log(rank, f"error {type(e).__name__}: {e}")
        result["error"] = f"{type(e).__name__}: {e}"

    log(rank, "all_gather start")
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, result)
    log(rank, "all_gather done")

    if rank == 0:
        for item in gathered:
            print(item)
        all_ok = all(
            item["ok"]
            and item["x_grad_finite"]
            and item["context_grad_finite"]
            and item["fallback"] == {"counts": {}, "last": None}
            for item in gathered
        )
        print("distributed_smoke_ok", all_ok)
        if not all_ok:
            raise SystemExit(1)

    log(rank, "barrier start")
    dist.barrier()
    log(rank, "barrier done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
