"""
kairos_ext/kairos_engine_patch.py — Full CUDA engine for Kairos 4B.

Patches the KairosDiT transformer to run ALL 32 layers through the
CUDA engine — both quadratic-attention layers (24) and GatedDeltaNet
layers (8). The GDN layers use fused recurrent CUDA kernels.

Usage:
    from kairos_ext.kairos_engine_patch import patch_engine
    patch_engine(transformer, max_seq=7800, ctx_len=512)
"""
import os
import time
import types
import glob
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange

_ext = None
_native_train_ops = None


def _prefer_current_py_paths(paths):
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"

    def _key(path):
        return (py_tag not in path, path)

    return sorted(paths, key=_key)


def _detect_nvidia_pkg_paths(pkg_name, header_name, lib_glob):
    prefix = os.environ.get("CONDA_PREFIX", "")
    inc_dir = ""
    lib_dir = ""
    if not prefix:
        return inc_dir, lib_dir

    include_matches = []
    lib_matches = []
    for patt in (
        os.path.join(prefix, "lib", "python*", "site-packages", "nvidia", pkg_name, "include"),
        os.path.join(prefix, "**", "nvidia", pkg_name, "include"),
    ):
        include_matches.extend(glob.glob(patt, recursive=True))
    for patt in (
        os.path.join(prefix, "lib", "python*", "site-packages", "nvidia", pkg_name, "lib"),
        os.path.join(prefix, "**", "nvidia", pkg_name, "lib"),
    ):
        lib_matches.extend(glob.glob(patt, recursive=True))

    for match in _prefer_current_py_paths(set(include_matches)):
        if os.path.isfile(os.path.join(match, header_name)):
            inc_dir = match
            break
    for match in _prefer_current_py_paths(set(lib_matches)):
        if glob.glob(os.path.join(match, lib_glob)):
            lib_dir = match
            break
    return inc_dir, lib_dir


def _detect_cudnn_paths():
    cudnn_inc = os.environ.get("CUDNN_INC", "")
    cudnn_lib = os.environ.get("CUDNN_LIB", "")
    if not cudnn_inc or not cudnn_lib:
        auto_inc, auto_lib = _detect_nvidia_pkg_paths("cudnn", "cudnn.h", "libcudnn.so*")
        cudnn_inc = cudnn_inc or auto_inc
        cudnn_lib = cudnn_lib or auto_lib
    return cudnn_inc, cudnn_lib


def _detect_nccl_root():
    nccl_home = os.environ.get("NCCL_HOME", "/usr")
    if nccl_home != "/usr":
        return nccl_home
    nccl_inc, nccl_lib = _detect_nvidia_pkg_paths("nccl", "nccl.h", "libnccl.so*")
    if nccl_inc and nccl_lib:
        return os.path.dirname(nccl_inc.rstrip("/"))
    return nccl_home


def get_ext():
    """JIT-compile the Kairos engine (caches the extension)."""
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load

    cu = os.environ.get(
        "KAIROS_ENGINE_CU",
        os.path.join(os.path.dirname(__file__), "csrc", "kairos_engine.cu"),
    )
    sage_root = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "kairos", "third_party", "SageAttention", "sageattention3_blackwell",
    )
    sage_sources = [
        os.path.join(sage_root, "sageattn3", "blackwell", "api.cu"),
        os.path.join(sage_root, "sageattn3", "quantization", "fp4_quantization_4d.cu"),
    ]
    cudnn_fe = os.environ.get("CUDNN_FRONTEND", "/tmp/cudnn-frontend/include")
    cudnn_inc, cudnn_lib = _detect_cudnn_paths()
    if not cudnn_lib and cudnn_inc:
        cudnn_root = os.path.dirname(cudnn_inc.rstrip("/"))
        guessed_cudnn_lib = os.path.join(cudnn_root, "lib")
        if os.path.isdir(guessed_cudnn_lib):
            cudnn_lib = guessed_cudnn_lib
    nccl_inc = _detect_nccl_root()
    # CUTLASS headers for CuTe MMA atoms (tensor core chunk GDN kernel)
    cutlass_inc = os.environ.get("CUTLASS_INC",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "cutlass", "include"))
    include_paths = [
        cudnn_fe,
        f"{nccl_inc}/include",
        cutlass_inc,
        os.path.join(sage_root, "sageattn3"),
        os.path.join(sage_root, "csrc", "cutlass", "include"),
        os.path.join(sage_root, "csrc", "cutlass", "tools", "util", "include"),
    ]
    if cudnn_inc:
        include_paths.append(cudnn_inc)

    def _pick_lib(name, libdir, short):
        if not libdir:
            return f"-l{short}"
        import glob as _g
        for patt in (f"lib{short}.so", f"lib{short}.so.*"):
            matches = sorted(_g.glob(os.path.join(libdir, patt)))
            if matches:
                return f"-l:{os.path.basename(matches[-1])}"
        return f"-l{short}"

    cudnn_flag = _pick_lib("cudnn", cudnn_lib, "cudnn")
    nccl_flag = _pick_lib("nccl", f"{nccl_inc}/lib", "nccl")
    ldflags = [
        "-lcublasLt", "-lcublas", cudnn_flag, "-lnvrtc", "-lcuda", nccl_flag,
        f"-L{nccl_inc}/lib",
    ]
    if cudnn_lib:
        ldflags.insert(0, f"-L{cudnn_lib}")
        os.environ["LD_LIBRARY_PATH"] = (
            cudnn_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )
    os.environ["LD_LIBRARY_PATH"] = (
        f"{nccl_inc}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    print(f"  JIT compiling kairos_engine from {cu}...")
    print(f"    include_paths={include_paths}")
    print(f"    ldflags={ldflags}")
    # Embedded SageAttention3 uses Blackwell "a"-feature instructions
    # (e.g. block-scaled MMA and FP4 conversions), so force the JIT build
    # to target the "a" architecture variant instead of torch's default
    # plain sm_120 / sm_121 arch selection.
    prev_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    blackwell_arches = []
    for dev_idx in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(dev_idx)
        if (major, minor) in {(10, 0), (10, 3), (11, 0), (12, 0), (12, 1)}:
            blackwell_arches.append(f"{major}.{minor}a")
        else:
            blackwell_arches.append(f"{major}.{minor}")
    if blackwell_arches:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted(set(blackwell_arches)))
        print(f"    TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")
    t0 = time.time()
    try:
        _ext = load(
            name="kairos_engine",
            sources=[cu] + sage_sources,
            extra_include_paths=include_paths,
            extra_cuda_cflags=[
                "-O3",
                "-w",
                "-std=c++17",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                "-DNDEBUG",
                "-DQBLKSIZE=128",
                "-DKBLKSIZE=128",
                "-DCTA256",
                "-DDQINRMEM",
                "-DEXECMODE=0",
                "-DKAIROS_EMBEDDED_SAGEATTN3=1",
            ],
            extra_ldflags=ldflags,
            verbose=False,
        )
    finally:
        if prev_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev_arch_list
    print(f"  Compiled in {time.time()-t0:.1f}s")
    return _ext


def _get_native_train_ops():
    global _native_train_ops
    if _native_train_ops is None:
        from kairos_ext.engine_autograd import (
            gate_residual_ssts,
            ln_adaln_ssts,
            linear_bf16,
            rmsnorm,
            layernorm_affine,
            attention_core,
            rope_apply_core,
            gdn_l2norm_scale,
            gdn_causal_conv_silu,
            gdn_compute_gates,
            gdn_recurrent,
            gdn_rmsnorm_silu_gate,
        )
        _native_train_ops = (
            gate_residual_ssts,
            ln_adaln_ssts,
            linear_bf16,
            rmsnorm,
            layernorm_affine,
            attention_core,
            rope_apply_core,
            gdn_l2norm_scale,
            gdn_causal_conv_silu,
            gdn_compute_gates,
            gdn_recurrent,
            gdn_rmsnorm_silu_gate,
        )
    return _native_train_ops


# --- Sharding helpers (for TP) ------------------------------------------------
def _split_col(w: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """Column-parallel: split output (N) dim. w=[N, K] -> [N/W, K]."""
    N = w.shape[0]
    assert N % world == 0, f"N={N} not divisible by world={world}"
    chunk = N // world
    return w[rank * chunk: (rank + 1) * chunk].contiguous()


def _split_row(w: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """Row-parallel: split input (K) dim. w=[N, K] -> [N, K/W]."""
    K = w.shape[1]
    assert K % world == 0, f"K={K} not divisible by world={world}"
    chunk = K // world
    return w[:, rank * chunk: (rank + 1) * chunk].contiguous()


def _split_1d(x: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """Split 1D tensor: x=[D] -> [D/W]."""
    D = x.shape[0]
    assert D % world == 0, f"D={D} not divisible by world={world}"
    chunk = D // world
    return x[rank * chunk: (rank + 1) * chunk].contiguous()


def setup_nccl_from_dist(rank: int, world: int, device: torch.device):
    """Share NCCL unique id from rank 0 via a shared file."""
    import time as _t
    ext = get_ext()
    uid_path = os.environ.get("KAIROS_ENGINE_NCCL_UID_PATH",
                              "/tmp/_kairos_nccl_uid.bin")
    ready_path = uid_path + ".ready"
    if rank == 0:
        for p in (uid_path, ready_path):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        uid = ext.nccl_get_unique_id()
        with open(uid_path, "wb") as f:
            f.write(bytes(uid.tolist()))
        with open(ready_path, "w") as f:
            f.write("ok")
        uid_cpu = uid
    else:
        for _ in range(600):
            if os.path.exists(ready_path):
                break
            _t.sleep(0.1)
        else:
            raise RuntimeError(f"rank={rank} timed out waiting for NCCL uid file {ready_path}")
        with open(uid_path, "rb") as f:
            data = f.read()
        uid_cpu = torch.tensor(list(data), dtype=torch.uint8)
    ext.nccl_comm_init(uid_cpu, rank, world)


# --- Block attribute extraction ------------------------------------------------

def _is_gdn_layer(idx: int) -> bool:
    """Return True if layer `idx` uses GatedDeltaNet (every 4th starting at 3)."""
    return (idx % 4 == 3)


def _get_block_attrs(block):
    """Extract weight/bias/norm tensors from a Kairos DiTBlock (quadratic layers only).

    For GatedDeltaNet layers, returns None. Use _get_gdn_block_attrs for those.
    """
    if block.use_linear_attn:
        return None

    sa = block.self_attn
    ca = block.cross_attn
    attrs = {}
    # SA projections
    attrs["sq_w"] = sa.q.weight.data
    attrs["sk_w"] = sa.k.weight.data
    attrs["sv_w"] = sa.v.weight.data
    attrs["so_w"] = sa.o.weight.data
    attrs["sq_b"] = sa.q.bias
    attrs["sk_b"] = sa.k.bias
    attrs["sv_b"] = sa.v.bias
    attrs["so_b"] = sa.o.bias
    # SA RMSNorm
    attrs["rms_q"] = sa.norm_q.weight.data
    attrs["rms_k"] = sa.norm_k.weight.data
    # CA projections
    attrs["cq_w"] = ca.q.weight.data
    attrs["ck_w"] = ca.k.weight.data
    attrs["cv_w"] = ca.v.weight.data
    attrs["co_w"] = ca.o.weight.data
    attrs["cq_b"] = ca.q.bias
    attrs["ck_b"] = ca.k.bias
    attrs["cv_b"] = ca.v.bias
    attrs["co_b"] = ca.o.bias
    # CA RMSNorm
    attrs["ca_rms_q"] = ca.norm_q.weight.data
    attrs["ca_rms_k"] = ca.norm_k.weight.data
    # cross_attn_norm (affine LayerNorm with weight+bias)
    n2 = block.cross_attn_norm
    attrs["n2w"] = n2.weight.data if hasattr(n2, "weight") and n2.weight is not None else None
    attrs["n2b"] = n2.bias.data if hasattr(n2, "bias") and n2.bias is not None else None
    # FFN
    attrs["f1_w"] = block.ffn[0].weight.data
    attrs["f2_w"] = block.ffn[2].weight.data
    attrs["f1_b"] = block.ffn[0].bias
    attrs["f2_b"] = block.ffn[2].bias
    # Modulation [1, 6, D]
    attrs["sst"] = block.modulation.data
    # Num heads
    attrs["nh"] = sa.num_heads
    return attrs


def _get_gdn_block_attrs(block):
    """Extract GDN-specific + CA/FFN weights from a GatedDeltaNet DiTBlock."""
    assert block.use_linear_attn, "Not a GDN block"
    gdn = block.gated_delta
    ca = block.cross_attn
    attrs = {}
    # GDN projections
    attrs["gdn_q_w"] = gdn.q_proj.weight.data
    attrs["gdn_k_w"] = gdn.k_proj.weight.data
    attrs["gdn_v_w"] = gdn.v_proj.weight.data
    attrs["gdn_a_w"] = gdn.a_proj.weight.data
    attrs["gdn_b_w"] = gdn.b_proj.weight.data
    attrs["gdn_g_w"] = gdn.g_proj.weight.data
    attrs["gdn_o_w"] = gdn.o_proj.weight.data
    attrs["gdn_q_b"] = getattr(gdn.q_proj, "bias", None)
    attrs["gdn_k_b"] = getattr(gdn.k_proj, "bias", None)
    attrs["gdn_v_b"] = getattr(gdn.v_proj, "bias", None)
    attrs["gdn_a_b"] = getattr(gdn.a_proj, "bias", None)
    attrs["gdn_b_b"] = getattr(gdn.b_proj, "bias", None)
    attrs["gdn_g_b"] = getattr(gdn.g_proj, "bias", None)
    attrs["gdn_o_b"] = getattr(gdn.o_proj, "bias", None)
    # Short conv weights [C, 1, K]
    attrs["conv_q_w"] = gdn.q_conv1d.weight.data
    attrs["conv_k_w"] = gdn.k_conv1d.weight.data
    attrs["conv_v_w"] = gdn.v_conv1d.weight.data
    # Gate parameters
    attrs["A_log"] = gdn.A_log.data
    attrs["dt_bias"] = gdn.dt_bias.data
    # Output norm
    attrs["o_norm_w"] = gdn.o_norm.weight.data
    # Key/value dims
    attrs["key_dim"] = gdn.q_proj.weight.shape[0]
    attrs["value_dim"] = gdn.v_proj.weight.shape[0]
    # CA projections (same structure as quadratic)
    attrs["cq_w"] = ca.q.weight.data
    attrs["ck_w"] = ca.k.weight.data
    attrs["cv_w"] = ca.v.weight.data
    attrs["co_w"] = ca.o.weight.data
    attrs["cq_b"] = ca.q.bias
    attrs["ck_b"] = ca.k.bias
    attrs["cv_b"] = ca.v.bias
    attrs["co_b"] = ca.o.bias
    attrs["ca_rms_q"] = ca.norm_q.weight.data
    attrs["ca_rms_k"] = ca.norm_k.weight.data
    # cross_attn_norm
    n2 = block.cross_attn_norm
    attrs["n2w"] = n2.weight.data if hasattr(n2, "weight") and n2.weight is not None else None
    attrs["n2b"] = n2.bias.data if hasattr(n2, "bias") and n2.bias is not None else None
    # FFN
    attrs["f1_w"] = block.ffn[0].weight.data
    attrs["f2_w"] = block.ffn[2].weight.data
    attrs["f1_b"] = block.ffn[0].bias
    attrs["f2_b"] = block.ffn[2].bias
    # Modulation
    attrs["sst"] = block.modulation.data
    attrs["nh"] = ca.num_heads
    return attrs


def _find_blocks(transformer):
    """Find the ModuleList of transformer blocks."""
    for name, mod in transformer.named_children():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            b0 = mod[0]
            if hasattr(b0, "modulation"):
                return list(mod), name
    return None, None


def _native_linear_rows(linear_bf16, x, linear):
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    y2 = linear_bf16(x2, linear.weight, linear.bias)
    return y2.reshape(*orig_shape[:-1], y2.shape[-1])


def _native_mlp_rows(layernorm_affine, linear_bf16, x, mlp):
    if mlp.has_pos_emb:
        x = x + mlp.emb_pos.to(dtype=x.dtype, device=x.device)
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1]).contiguous()
    x = layernorm_affine(x, mlp.proj[0].weight, mlp.proj[0].bias, mlp.proj[0].eps)
    x = _native_linear_rows(linear_bf16, x, mlp.proj[1])
    x = torch.nn.functional.silu(x)
    x = _native_linear_rows(linear_bf16, x, mlp.proj[3])
    x = layernorm_affine(x, mlp.proj[4].weight, mlp.proj[4].bias, mlp.proj[4].eps)
    return x.reshape(*orig_shape[:-1], x.shape[-1])


def _build_2d_sincos_pos_embed(embed_dim: int, h: int, w: int, device=None, dtype=None):
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4."
    device = device or "cpu"
    dtype = dtype or torch.float32
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    dim_each = embed_dim // 2
    omega = torch.arange(dim_each // 2, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / (dim_each // 2)))
    out_y = yy[:, None] * omega[None, :]
    out_x = xx[:, None] * omega[None, :]
    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)
    return torch.cat([pos_y, pos_x], dim=1).unsqueeze(0)


def _native_pos_embed_2d(x, h, w):
    pos = _build_2d_sincos_pos_embed(x.shape[-1], h, w, device=x.device, dtype=torch.float32)
    return x + pos.to(x.dtype)


def _native_attention_lse(qh, kh, attn_mask=None):
    qf = rearrange(qh.float(), "b s n d -> b n s d")
    kf = rearrange(kh.float(), "b s n d -> b n s d")
    scale = qf.shape[-1] ** -0.5
    scores = torch.einsum("bnsd,bntd->bnst", qf, kf) * scale
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask.to(torch.bool), -float("inf"))
    return torch.logsumexp(scores, dim=-1).transpose(1, 2).contiguous()


def _native_quadratic_slice_reason(block, x, t_mod):
    world = getattr(block, "world", 1)
    if block.use_linear_attn:
        return "linear_attn_block"
    if x.dtype != torch.bfloat16:
        return f"x_dtype={x.dtype}"
    if x.dim() != 3:
        return f"x_dim={x.dim()}"
    if world > 1 and getattr(block, "use_tp_in_self_attn", False):
        return "use_tp_in_self_attn"
    if t_mod.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"t_mod_dtype={t_mod.dtype}"
    return None


def _native_gdn_slice_reason(block, x, t_mod):
    world = getattr(block, "world", 1)
    if not block.use_linear_attn:
        return "quadratic_block"
    if x.dtype != torch.bfloat16:
        return f"x_dtype={x.dtype}"
    if x.dim() != 3:
        return f"x_dim={x.dim()}"
    if world > 1:
        if getattr(block, "use_tp_in_getaeddeltanet", False):
            return "use_tp_in_getaeddeltanet"
        if getattr(block, "use_seq_parallel", False):
            if not (dist.is_available() and dist.is_initialized()):
                return "dist_uninitialized"
            if getattr(block, "context_group", None) is None:
                return "context_group_none"
        else:
            return f"world={world}"
    if t_mod.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"t_mod_dtype={t_mod.dtype}"
    return None


def _native_text_embedding(self, context, linear_bf16):
    context = _native_linear_rows(linear_bf16, context, self.text_embedding[0])
    context = torch.nn.functional.silu(context)
    context = _native_linear_rows(linear_bf16, context, self.text_embedding[2])
    return context


def _native_time_embedding(self, t, linear_bf16):
    t = _native_linear_rows(linear_bf16, t, self.time_embedding[0])
    t = torch.nn.functional.silu(t)
    t = _native_linear_rows(linear_bf16, t, self.time_embedding[2])
    return t


def _native_time_projection(self, t, linear_bf16):
    t = torch.nn.functional.silu(t)
    return _native_linear_rows(linear_bf16, t, self.time_projection[1])


def _native_head_forward(self, x, t_mod, linear_bf16):
    head = self.head
    x_norm = torch.nn.functional.layer_norm(
        x.float(),
        (head.dim,),
        weight=None,
        bias=None,
        eps=head.norm.eps,
    ).to(x.dtype)
    if len(t_mod.shape) == 3:
        shift, scale = (
            head.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)
        ).chunk(2, dim=2)
        x_mod = x_norm * (1 + scale.squeeze(2)) + shift.squeeze(2)
    else:
        shift, scale = (head.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x_mod = x_norm * (1 + scale) + shift
    out = _native_linear_rows(linear_bf16, x_mod, head.head)
    return out


def _native_conv_module(x, conv):
    if isinstance(conv, nn.Conv2d):
        return F.conv2d(
            x,
            conv.weight,
            conv.bias,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
        )
    if isinstance(conv, nn.Conv3d):
        return F.conv3d(
            x,
            conv.weight,
            conv.bias,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
        )
    raise TypeError(f"Unsupported conv module type: {type(conv)!r}")


def _native_patchify(self, x, control_camera_latents_input=None):
    x = _native_conv_module(x, self.patch_embedding)
    if self.control_adapter is not None and control_camera_latents_input is not None:
        y_camera = self.control_adapter(control_camera_latents_input)
        x = [u + v for u, v in zip(x, y_camera)]
        x = x[0].unsqueeze(0)
    grid_size = x.shape[2:]
    return x, grid_size


def _native_image_downsample(self, first_frame_latents):
    x = first_frame_latents.squeeze(2)
    for layer in self.image_downsample:
        x = _native_conv_module(x, layer)
    return x


def _native_get_owner_chunk_info(chunk_id, chunk_size, local_seq_len):
    global_start = chunk_id * chunk_size
    owner_rank = global_start // local_seq_len
    owner_local_start = global_start % local_seq_len
    owner_local_end = owner_local_start + chunk_size
    return owner_rank, owner_local_start, owner_local_end


class _OwnerBroadcastFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_chunk, owner_rank, group):
        ctx.owner_rank = owner_rank
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.input_requires_grad = x_chunk.requires_grad

        if ctx.rank == owner_rank:
            out = x_chunk.clone()
        else:
            out = torch.empty_like(x_chunk)
        dist.broadcast(out, src=owner_rank, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_owner = grad_output.contiguous()
        dist.all_reduce(grad_owner, op=dist.ReduceOp.SUM, group=ctx.group)
        if ctx.input_requires_grad:
            return grad_owner, None, None
        return None, None, None


def _native_broadcast_from_owner(x_chunk, owner_rank, chunk_shape, dtype, device, group):
    rank = dist.get_rank(group)
    if rank != owner_rank:
        x_chunk = torch.empty(chunk_shape, device=device, dtype=dtype, requires_grad=True)
    return _OwnerBroadcastFn.apply(x_chunk, owner_rank, group)


def _native_cross_attn_forward(block, x, context, context_mask, linear_bf16, rmsnorm, layernorm_affine, attention_core):
    ca = block.cross_attn
    x_rows = x.squeeze(0).contiguous()
    norm = block.cross_attn_norm
    ca_in = layernorm_affine(x_rows, norm.weight, norm.bias, norm.eps)
    q = rmsnorm(
        linear_bf16(ca_in, ca.q.weight, ca.q.bias),
        ca.norm_q.weight,
        ca.norm_q.eps,
    )
    qh = rearrange(q.unsqueeze(0), "b s (n d) -> b s n d", n=ca.num_heads)

    ctx_full = context
    img = None
    ctx = ctx_full
    if ca.has_image_input:
        img = ctx_full[:, :257, :]
        ctx = ctx_full[:, 257:, :]

    ctx_rows = ctx.squeeze(0).contiguous()
    k = rmsnorm(
        linear_bf16(ctx_rows, ca.k.weight, ca.k.bias),
        ca.norm_k.weight,
        ca.norm_k.eps,
    )
    v = linear_bf16(ctx_rows, ca.v.weight, ca.v.bias)
    kh = rearrange(k.unsqueeze(0), "b s (n d) -> b s n d", n=ca.num_heads)
    vh = rearrange(v.unsqueeze(0), "b s (n d) -> b s n d", n=ca.num_heads)

    attn_mask_local = context_mask
    if attn_mask_local is not None:
        B, Lq, S = x.shape[0], x.shape[1], ctx.shape[1]
        attn_mask_local = attn_mask_local.view(B, 1, 1, S).expand(B, 1, Lq, S)
    attn_out = attention_core(
        qh,
        kh,
        vh,
        num_heads=ca.num_heads,
        attn_mask=attn_mask_local,
    )
    if img is not None:
        img_rows = img.squeeze(0).contiguous()
        k_img = rmsnorm(
            linear_bf16(img_rows, ca.k_img.weight, ca.k_img.bias),
            ca.norm_k_img.weight,
            ca.norm_k_img.eps,
        )
        v_img = linear_bf16(img_rows, ca.v_img.weight, ca.v_img.bias)
        kh_img = rearrange(k_img.unsqueeze(0), "b s (n d) -> b s n d", n=ca.num_heads)
        vh_img = rearrange(v_img.unsqueeze(0), "b s (n d) -> b s n d", n=ca.num_heads)
        img_attn_out = attention_core(
            qh,
            kh_img,
            vh_img,
            num_heads=ca.num_heads,
        )
        attn_out = attn_out + img_attn_out
    attn_out = rearrange(attn_out, "b s n d -> b s (n d)", n=ca.num_heads)
    return linear_bf16(attn_out.squeeze(0).contiguous(), ca.o.weight, ca.o.bias).unsqueeze(0)


def _native_self_attn_forward(sa, input_x_local, freqs, L, linear_bf16, rmsnorm, attention_core, rope_apply_core):
    q = linear_bf16(input_x_local, sa.q.weight, sa.q.bias).unsqueeze(0)
    k = linear_bf16(input_x_local, sa.k.weight, sa.k.bias).unsqueeze(0)
    v = linear_bf16(input_x_local, sa.v.weight, sa.v.bias).unsqueeze(0)
    q = rmsnorm(q.squeeze(0), sa.norm_q.weight, sa.norm_q.eps).unsqueeze(0)
    k = rmsnorm(k.squeeze(0), sa.norm_k.weight, sa.norm_k.eps).unsqueeze(0)
    q = rope_apply_core(q, freqs, sa.num_heads)
    k = rope_apply_core(k, freqs, sa.num_heads)
    if sa.attend_k0:
        fk = k[:, :1]
        fv = v[:, :1]
    dilated_length = sa.dilated_length
    use_dilated = dilated_length > 1 and q.shape[1] // (dilated_length * L) > 1
    pad_len = 0
    if use_dilated:
        assert q.shape[1] % L == 0, "L should equal to the num of tokens per frame"
        pad_len = dilated_length * L - q.shape[1] % (dilated_length * L)
        if pad_len != 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
        q = rearrange(q, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)
        k = rearrange(k, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)
        v = rearrange(v, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)
        if sa.attend_k0:
            fk = fk.unsqueeze(1).expand(-1, dilated_length, -1, -1).flatten(0, 1)
            fv = fv.unsqueeze(1).expand(-1, dilated_length, -1, -1).flatten(0, 1)
    qh = rearrange(q, "b s (n d) -> b s n d", n=sa.num_heads)
    kh = rearrange(k, "b s (n d) -> b s n d", n=sa.num_heads)
    vh = rearrange(v, "b s (n d) -> b s n d", n=sa.num_heads)
    attn_core = attention_core(
        qh,
        kh,
        vh,
        num_heads=sa.num_heads,
        window_size=(L * sa.window_size, L * sa.window_size),
    )
    if sa.attend_k0:
        fk_h = rearrange(fk, "b s (n d) -> b s n d", n=sa.num_heads)
        fv_h = rearrange(fv, "b s (n d) -> b s n d", n=sa.num_heads)
        softmax_scale = qh.shape[-1] ** -0.5
        logits0 = (qh * fk_h).sum(dim=-1) * softmax_scale
        lse = _native_attention_lse(qh, kh)
        lse_total = torch.logaddexp(lse, logits0.float())
        w_swa = torch.exp(lse - lse_total).to(attn_core.dtype).unsqueeze(-1)
        w0 = torch.exp(logits0.float() - lse_total).to(attn_core.dtype).unsqueeze(-1)
        attn_core = attn_core * w_swa + fv_h.expand_as(qh) * w0
    attn_core = rearrange(attn_core, "b s n d -> b s (n d)", n=sa.num_heads)
    if use_dilated:
        attn_core = rearrange(attn_core, "(b d) (n l) c -> b (n d l) c", l=L, d=dilated_length)
        if pad_len != 0:
            attn_core = attn_core[:, :-pad_len]
    return linear_bf16(attn_core.squeeze(0).contiguous(), sa.o.weight, sa.o.bias).unsqueeze(0)


def _can_use_native_quadratic_slices(block, x, t_mod):
    return _native_quadratic_slice_reason(block, x, t_mod) is None


def _can_use_native_gdn_slices(block, x, t_mod):
    return _native_gdn_slice_reason(block, x, t_mod) is None


def _run_quadratic_block_native_slices(block, x, context, t_mod, freqs, grid_size, context_mask=None):
    gate_residual_ssts, ln_adaln_ssts, linear_bf16, rmsnorm, layernorm_affine, attention_core, rope_apply_core, *_ = _get_native_train_ops()

    (f, h, w) = grid_size
    B, _, D = x.shape
    if B != 1:
        outs = []
        for b in range(B):
            t_mod_b = t_mod[b:b + 1] if t_mod.shape[0] == B else t_mod
            context_b = context[b:b + 1] if context.shape[0] == B else context
            context_mask_b = context_mask[b:b + 1] if (context_mask is not None and context_mask.shape[0] == B) else context_mask
            outs.append(
                _run_quadratic_block_native_slices(
                    block,
                    x[b:b + 1],
                    context_b,
                    t_mod_b,
                    freqs,
                    grid_size,
                    context_mask=context_mask_b,
                )
            )
        return torch.cat(outs, dim=0)
    L = h * w
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1

    mod_dtype = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
    t_mod_chunks = t_mod.chunk(6, dim=chunk_dim)
    mod_chunks = mod_dtype.chunk(6, dim=1)

    def get_mod_parts(idx, start=None, end=None):
        t_c = t_mod_chunks[idx]
        m_c = mod_chunks[idx]
        if has_seq:
            t_c = t_c.squeeze(2)
            m_c = m_c.squeeze(1)
            if start is not None:
                t_c = t_c[:, start:end, :]
        return t_c, m_c

    def as_rows(t_chunk):
        rows = t_chunk.squeeze(0)
        if rows.dim() == 1:
            rows = rows.unsqueeze(0)
        return rows.float().contiguous()

    def as_ssts(mod_chunk):
        flat = mod_chunk.reshape(-1, mod_chunk.shape[-1])
        return flat[0].float().contiguous()

    x_rows = x.squeeze(0).contiguous()

    t_scale_msa, m_scale_msa = get_mod_parts(0)
    t_shift_msa, m_shift_msa = get_mod_parts(1)
    gate_t_msa, gate_m_msa = get_mod_parts(2)

    input_x_local = ln_adaln_ssts(
        x_rows,
        as_ssts(m_scale_msa),
        as_ssts(m_shift_msa),
        as_rows(t_scale_msa),
        as_rows(t_shift_msa),
    )

    sa = block.self_attn
    attn_out = _native_self_attn_forward(sa, input_x_local, freqs, L, linear_bf16, rmsnorm, attention_core, rope_apply_core)
    x = gate_residual_ssts(
        x_rows,
        attn_out.squeeze(0).contiguous(),
        as_ssts(gate_m_msa),
        as_rows(gate_t_msa),
    ).unsqueeze(0)

    attn_out = _native_cross_attn_forward(
        block,
        x,
        context,
        context_mask,
        linear_bf16,
        rmsnorm,
        layernorm_affine,
        attention_core,
    )
    x = x + attn_out

    chunk_size = 2310
    out = torch.empty_like(x)
    for start in range(0, x.shape[1], chunk_size):
        end = min(start + chunk_size, x.shape[1])
        x_chunk = x[:, start:end, :]
        x_chunk_rows = x_chunk.squeeze(0).contiguous()

        t_scale_ffn, m_scale_ffn = get_mod_parts(3, start, end)
        t_shift_ffn, m_shift_ffn = get_mod_parts(4, start, end)
        t_gate_ffn, m_gate_ffn = get_mod_parts(5, start, end)

        inp_chunk = ln_adaln_ssts(
            x_chunk_rows,
            as_ssts(m_scale_ffn),
            as_ssts(m_shift_ffn),
            as_rows(t_scale_ffn),
            as_rows(t_shift_ffn),
        )
        ffn_mid = linear_bf16(inp_chunk, block.ffn[0].weight, block.ffn[0].bias)
        ffn_mid = torch.nn.functional.silu(ffn_mid)
        ffn_out = linear_bf16(ffn_mid, block.ffn[2].weight, block.ffn[2].bias).unsqueeze(0)
        out[:, start:end, :] = gate_residual_ssts(
            x_chunk_rows,
            ffn_out.squeeze(0).contiguous(),
            as_ssts(m_gate_ffn),
            as_rows(t_gate_ffn),
        ).unsqueeze(0)

    return out


def _run_gdn_block_native_slices(block, x, context, t_mod, freqs, grid_size, context_mask=None):
    (
        gate_residual_ssts,
        ln_adaln_ssts,
        linear_bf16,
        rmsnorm,
        layernorm_affine,
        attention_core,
        _rope_apply_core,
        gdn_l2norm_scale,
        gdn_causal_conv_silu,
        gdn_compute_gates,
        gdn_recurrent,
        gdn_rmsnorm_silu_gate,
    ) = _get_native_train_ops()

    (f, h, w) = grid_size
    B, _, D = x.shape
    if B != 1:
        outs = []
        for b in range(B):
            t_mod_b = t_mod[b:b + 1] if t_mod.shape[0] == B else t_mod
            context_b = context[b:b + 1] if context.shape[0] == B else context
            context_mask_b = context_mask[b:b + 1] if (context_mask is not None and context_mask.shape[0] == B) else context_mask
            outs.append(
                _run_gdn_block_native_slices(
                    block,
                    x[b:b + 1],
                    context_b,
                    t_mod_b,
                    freqs,
                    grid_size,
                    context_mask=context_mask_b,
                )
            )
        return torch.cat(outs, dim=0)
    L = h * w
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1

    mod_dtype = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
    t_mod_chunks = t_mod.chunk(6, dim=chunk_dim)
    mod_chunks = mod_dtype.chunk(6, dim=1)

    def get_mod_parts(idx, start=None, end=None):
        t_c = t_mod_chunks[idx]
        m_c = mod_chunks[idx]
        if has_seq:
            t_c = t_c.squeeze(2)
            m_c = m_c.squeeze(1)
            if start is not None:
                t_c = t_c[:, start:end, :]
        return t_c, m_c

    def as_rows(t_chunk):
        rows = t_chunk.squeeze(0)
        if rows.dim() == 1:
            rows = rows.unsqueeze(0)
        return rows.float().contiguous()

    def as_ssts(mod_chunk):
        flat = mod_chunk.reshape(-1, mod_chunk.shape[-1])
        return flat[0].float().contiguous()

    x_rows = x.squeeze(0).contiguous()

    t_scale_msa, m_scale_msa = get_mod_parts(0)
    t_shift_msa, m_shift_msa = get_mod_parts(1)
    gate_t_msa, gate_m_msa = get_mod_parts(2)

    input_x_local = ln_adaln_ssts(
        x_rows,
        as_ssts(m_scale_msa),
        as_ssts(m_shift_msa),
        as_rows(t_scale_msa),
        as_rows(t_shift_msa),
    )
    hidden_states = input_x_local.unsqueeze(0)
    hidden_rows = input_x_local

    gdn = block.gated_delta
    seq_len = hidden_states.shape[1]

    q_proj = linear_bf16(hidden_rows, gdn.q_proj.weight, gdn.q_proj.bias).contiguous()
    k_proj = linear_bf16(hidden_rows, gdn.k_proj.weight, gdn.k_proj.bias).contiguous()
    v_proj = linear_bf16(hidden_rows, gdn.v_proj.weight, gdn.v_proj.bias).contiguous()
    q_conv = gdn_causal_conv_silu(q_proj, gdn.q_conv1d.weight.float().squeeze(1).contiguous())
    k_conv = gdn_causal_conv_silu(k_proj, gdn.k_conv1d.weight.float().squeeze(1).contiguous())
    v_conv = gdn_causal_conv_silu(v_proj, gdn.v_conv1d.weight.float().squeeze(1).contiguous())

    q_heads = q_conv.view(seq_len, gdn.num_heads, gdn.head_k_dim).contiguous()
    k_heads = k_conv.view(seq_len, gdn.num_heads, gdn.head_k_dim).contiguous()
    v_heads = v_conv.view(seq_len, gdn.num_v_heads, gdn.head_v_dim).contiguous()

    if gdn.num_v_heads > gdn.num_heads:
        repeat_factor = gdn.num_v_heads // gdn.num_heads
        q_heads = q_heads.repeat_interleave(repeat_factor, dim=1)
        k_heads = k_heads.repeat_interleave(repeat_factor, dim=1)

    beta_in = linear_bf16(hidden_rows, gdn.b_proj.weight, gdn.b_proj.bias).contiguous()
    a_in = linear_bf16(hidden_rows, gdn.a_proj.weight, gdn.a_proj.bias).contiguous()
    g_vals, beta_vals = gdn_compute_gates(
        a_in,
        beta_in,
        gdn.A_log.float().contiguous(),
        gdn.dt_bias.float().contiguous(),
    )
    if gdn.allow_neg_eigval:
        beta_vals = beta_vals * 2.0

    q_norm, k_norm = gdn_l2norm_scale(q_heads, k_heads, gdn.head_k_dim ** -0.5)
    world = getattr(block, "world", 1)
    if world > 1 and getattr(block, "use_seq_parallel", False):
        group = getattr(block, "context_group", None)
        rank = getattr(block, "context_group_rank", 0)
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Distributed must be initialized for multi-rank native GDN path.")
        if group is None:
            raise RuntimeError("context_group must be set for multi-rank native GDN path.")
        local_seq_len = q_norm.shape[0]
        chunk_size = local_seq_len
        total_seq_len = local_seq_len * world
        assert local_seq_len % chunk_size == 0
        assert total_seq_len % chunk_size == 0
        num_chunks = total_seq_len // chunk_size
        recurrent_out = None
        state = None
        dummy_dep = None
        for chunk_id in range(num_chunks):
            owner_rank, owner_local_start, owner_local_end = _native_get_owner_chunk_info(
                chunk_id=chunk_id,
                chunk_size=chunk_size,
                local_seq_len=local_seq_len,
            )

            def _owner_chunk(tensor):
                local = tensor[owner_local_start:owner_local_end].contiguous() if rank == owner_rank else None
                return _native_broadcast_from_owner(
                    local,
                    owner_rank=owner_rank,
                    chunk_shape=(chunk_size, *tensor.shape[1:]),
                    dtype=tensor.dtype,
                    device=tensor.device,
                    group=group,
                )

            q_chunk = _owner_chunk(q_norm)
            k_chunk = _owner_chunk(k_norm)
            v_chunk = _owner_chunk(v_heads)
            g_chunk = _owner_chunk(g_vals)
            beta_chunk = _owner_chunk(beta_vals)
            out_chunk, state = gdn_recurrent(
                q_chunk,
                k_chunk,
                v_chunk,
                g_chunk,
                beta_chunk,
                state0=state,
                scale=gdn.head_k_dim ** -0.5,
            )
            chunk_dep = out_chunk.reshape(-1)[:1].float().sum() * 0.0
            dummy_dep = chunk_dep if dummy_dep is None else (dummy_dep + chunk_dep)
            if rank == owner_rank:
                recurrent_out = out_chunk
        if recurrent_out is None:
            raise RuntimeError("No local recurrent output produced for native multi-rank GDN path.")
        if dummy_dep is not None:
            recurrent_out = recurrent_out + dummy_dep.to(recurrent_out.dtype)
    else:
        recurrent_out, _ = gdn_recurrent(
            q_norm,
            k_norm,
            v_heads,
            g_vals,
            beta_vals,
            scale=gdn.head_k_dim ** -0.5,
        )

    if gdn.use_gate:
        gate_vals = linear_bf16(hidden_rows, gdn.g_proj.weight, gdn.g_proj.bias).view(seq_len, gdn.num_v_heads, gdn.head_v_dim).contiguous()
        attn_out = gdn_rmsnorm_silu_gate(
            recurrent_out.contiguous(),
            gate_vals,
            gdn.o_norm.weight,
            eps=gdn.o_norm.eps,
        ).unsqueeze(0)
    else:
        attn_out = rmsnorm(
            recurrent_out.reshape(-1, gdn.head_v_dim).contiguous(),
            gdn.o_norm.weight,
            gdn.o_norm.eps,
        ).reshape(seq_len, gdn.num_v_heads, gdn.head_v_dim).unsqueeze(0)
    attn_out = linear_bf16(attn_out.reshape(seq_len, -1).contiguous(), gdn.o_proj.weight, gdn.o_proj.bias).unsqueeze(0)

    x = gate_residual_ssts(
        x_rows,
        attn_out.squeeze(0).contiguous(),
        as_ssts(gate_m_msa),
        as_rows(gate_t_msa),
    ).unsqueeze(0)

    attn_out = _native_cross_attn_forward(
        block,
        x,
        context,
        context_mask,
        linear_bf16,
        rmsnorm,
        layernorm_affine,
        attention_core,
    )
    x = x + attn_out

    chunk_size = 2310
    out = torch.empty_like(x)
    for start in range(0, x.shape[1], chunk_size):
        end = min(start + chunk_size, x.shape[1])
        x_chunk = x[:, start:end, :]
        x_chunk_rows = x_chunk.squeeze(0).contiguous()

        t_scale_ffn, m_scale_ffn = get_mod_parts(3, start, end)
        t_shift_ffn, m_shift_ffn = get_mod_parts(4, start, end)
        t_gate_ffn, m_gate_ffn = get_mod_parts(5, start, end)

        inp_chunk = ln_adaln_ssts(
            x_chunk_rows,
            as_ssts(m_scale_ffn),
            as_ssts(m_shift_ffn),
            as_rows(t_scale_ffn),
            as_rows(t_shift_ffn),
        )
        ffn_mid = linear_bf16(inp_chunk, block.ffn[0].weight, block.ffn[0].bias)
        ffn_mid = torch.nn.functional.silu(ffn_mid)
        ffn_out = linear_bf16(ffn_mid, block.ffn[2].weight, block.ffn[2].bias).unsqueeze(0)
        out[:, start:end, :] = gate_residual_ssts(
            x_chunk_rows,
            ffn_out.squeeze(0).contiguous(),
            as_ssts(m_gate_ffn),
            as_rows(t_gate_ffn),
        ).unsqueeze(0)

    return out


def _kairos_rope_to_helios(freqs_complex, hd):
    """Convert Kairos 3D complex RoPE to Helios [seq, 2*hd] float layout.

    Kairos RoPE is a complex tensor [seq, 1, c] where c = hd/2.
    The engine expects [seq, 2*hd] float where:
      offset 2*k       = cos[k]  (real part)
      offset hd + 2*k+1 = sin[k]  (imaginary part)
    """
    # freqs_complex: [seq, 1, c] complex64
    fc = freqs_complex.squeeze(1)  # [seq, c]
    seq = fc.shape[0]
    c = fc.shape[1]  # = hd/2
    r = torch.zeros(seq, 2 * hd, dtype=torch.float32, device=fc.device)
    for k in range(c):
        r[:, 2 * k] = fc[:, k].real.float()
        r[:, hd + 2 * k + 1] = fc[:, k].imag.float()
    return r.contiguous()


# --- Main patch entry point ---------------------------------------------------
def patch_engine(transformer, max_seq, ctx_len, seq_list=None, verbose=True,
                 tp_rank=0, tp_world=1, enable_training_bridge=None):
    """
    Install the Kairos engine into `transformer`, replacing its `.blocks`
    ModuleList with Runner stubs.

    ALL layers (quadratic + GatedDeltaNet) run through the CUDA engine.

    If tp_world > 1, the caller must have already invoked
    `setup_nccl_from_dist(tp_rank, tp_world, device)` before calling this.
    """
    ext = get_ext()
    t0 = time.time()

    blocks, block_name = _find_blocks(transformer)
    assert blocks, "No Kairos blocks found on transformer"
    orig_blocks = tuple(blocks)

    # Find a quadratic block to determine dimensions
    first_quad = None
    for b in blocks:
        if not b.use_linear_attn:
            first_quad = b
            break
    assert first_quad is not None, "No quadratic attention blocks found"

    a0 = _get_block_attrs(first_quad)
    dim = a0["sst"].shape[-1]
    nh = a0["nh"]
    hd = dim // nh
    ffn = a0["f1_w"].shape[0]
    ca_k_dim = a0["ck_w"].shape[1]
    nl = len(blocks)

    # Detect GDN dimensions from first GDN block
    gdn_key_dim = 0
    gdn_value_dim = 0
    first_gdn = None
    for b in blocks:
        if b.use_linear_attn:
            first_gdn = b
            ga = _get_gdn_block_attrs(b)
            gdn_key_dim = ga["key_dim"]
            gdn_value_dim = ga["value_dim"]
            break

    if verbose:
        print(f"  Kairos Model: dim={dim} nh={nh} hd={hd} ffn={ffn} "
              f"ca_k={ca_k_dim} layers={nl}")
        print(f"  TP: rank={tp_rank} world={tp_world}")
        n_gdn = sum(1 for i in range(nl) if _is_gdn_layer(i))
        n_quad = nl - n_gdn
        print(f"  Hybrid: {n_quad} quadratic + {n_gdn} GatedDeltaNet (all in engine)")
        if gdn_key_dim > 0:
            print(f"  GDN: key_dim={gdn_key_dim} value_dim={gdn_value_dim}")
        print(f"  Init: max_seq={max_seq} ctx={ctx_len}")

    ext.init(dim, nh, hd, nl, ffn, ca_k_dim, max_seq, ctx_len,
             gdn_key_dim, gdn_value_dim)
    sdpa_backend = os.environ.get("KAIROS_ENGINE_SDPA_BACKEND", "").strip().lower()
    if sdpa_backend:
        ext.set_sdpa_backend(sdpa_backend)
        if verbose:
            print(f"  SA backend: {sdpa_backend}")
    use_graph = os.environ.get("KAIROS_ENGINE_USE_GRAPH", "").strip().lower() in {"1", "true", "yes", "on"}
    ext.set_use_graph(1 if use_graph else 0)
    if verbose and use_graph:
        print("  CUDA graphs: enabled")
    # Chunk GDN is correct but slower (0.63x) — needs MMA fwd_o for speed.
    # Default to recurrent path (1.25x) until MMA fwd_o is implemented.
    # ext.set_gdn_cublas_chunk(True)  # enable for chunk path

    if seq_list:
        for s in sorted(set(seq_list)):
            ext.prepare_seq(s)
            if verbose:
                print(f"  Prepared seq={s}")

    dev = a0["sst"].device
    bf16 = torch.bfloat16
    f32 = torch.float32
    zb = lambda n: torch.zeros(n, device=dev, dtype=bf16)

    def full_bias(bias_or_none, n):
        if bias_or_none is None:
            return zb(n)
        v = bias_or_none
        return v.data.contiguous() if hasattr(v, "data") else v.contiguous()

    def sc(w):
        return _split_col(w, tp_rank, tp_world)

    def sr(w):
        return _split_row(w, tp_rank, tp_world)

    def sb(b):
        return _split_1d(b, tp_rank, tp_world)

    inv_w = 1.0 / float(tp_world)

    def _load_ca_ffn_weights(i, a):
        """Load CA + FFN weights for a layer (shared by quadratic and GDN)."""
        cq_w = sc(a["cq_w"].contiguous())
        ck_w = sc(a["ck_w"].contiguous())
        cv_w = sc(a["cv_w"].contiguous())
        co_w = sr(a["co_w"].contiguous())
        f1_w = sc(a["f1_w"].contiguous())
        f2_w = sr(a["f2_w"].contiguous())
        cq_b = sb(full_bias(a["cq_b"], dim))
        ck_b = sb(full_bias(a["ck_b"], dim))
        cv_b = sb(full_bias(a["cv_b"], dim))
        co_b = (full_bias(a["co_b"], dim).float() * inv_w).to(bf16).contiguous()
        f1_b = sb(full_bias(a["f1_b"], ffn))
        f2_b = (full_bias(a["f2_b"], dim).float() * inv_w).to(bf16).contiguous()
        ca_rms_q = sb(a["ca_rms_q"].float().contiguous())
        ca_rms_k = sb(a["ca_rms_k"].float().contiguous())
        n2w = (a["n2w"].float().contiguous() if a["n2w"] is not None
               else torch.empty(0, device=dev, dtype=f32))
        n2b = (a["n2b"].float().contiguous() if a["n2b"] is not None
               else torch.empty(0, device=dev, dtype=f32))
        sst = a["sst"].contiguous().float()
        return (cq_w, ck_w, cv_w, co_w, f1_w, f2_w,
                cq_b, ck_b, cv_b, co_b, f1_b, f2_b,
                ca_rms_q, ca_rms_k, n2w, n2b, sst)

    ssts_stacked = None

    def _sync_engine_weights(force=False):
        nonlocal ssts_stacked
        current_versions = tuple(int(p._version) for b in orig_blocks for p in b.parameters())
        if (not force) and current_versions == getattr(transformer, "_kairos_engine_block_versions", None):
            return False

        for i, b in enumerate(orig_blocks):
            if _is_gdn_layer(i):
                # GDN layer: load CA/FFN via ext.load() with dummy SA weights,
                # then load GDN-specific weights via ext.load_gdn()
                ga = _get_gdn_block_attrs(b)
                (cq_w, ck_w, cv_w, co_w, f1_w, f2_w,
                 cq_b, ck_b, cv_b, co_b, f1_b, f2_b,
                 ca_rms_q, ca_rms_k, n2w, n2b, sst) = _load_ca_ffn_weights(i, ga)
                z_w = torch.zeros(dim // tp_world, dim, device=dev, dtype=bf16)
                z_b = zb(dim // tp_world)
                z_rms = torch.ones(dim // tp_world, device=dev, dtype=f32)
                z_so_b = zb(dim)
                ext.load(i,
                         z_w, z_w, z_w, z_w[:, :dim // tp_world],
                         cq_w, ck_w, cv_w, co_w,
                         f1_w, f2_w,
                         z_b, z_b, z_b, z_so_b,
                         cq_b, ck_b, cv_b, co_b,
                         f1_b, f2_b,
                         z_rms, z_rms, ca_rms_q, ca_rms_k,
                         n2w, n2b, sst)
                gdn_q_w = sc(ga["gdn_q_w"].contiguous())
                gdn_k_w = sc(ga["gdn_k_w"].contiguous())
                gdn_v_w = sc(ga["gdn_v_w"].contiguous())
                gdn_g_w = sc(ga["gdn_g_w"].contiguous())
                gdn_o_w = sr(ga["gdn_o_w"].contiguous())
                gdn_a_w = ga["gdn_a_w"].contiguous()
                gdn_b_w = ga["gdn_b_w"].contiguous()
                ext.load_gdn(i,
                             gdn_q_w, gdn_k_w, gdn_v_w,
                             gdn_a_w, gdn_b_w, gdn_g_w, gdn_o_w,
                             full_bias(ga["gdn_q_b"], gdn_key_dim // tp_world),
                             full_bias(ga["gdn_k_b"], gdn_key_dim // tp_world),
                             full_bias(ga["gdn_v_b"], gdn_value_dim // tp_world),
                             full_bias(ga["gdn_a_b"], nh),
                             full_bias(ga["gdn_b_b"], nh),
                             full_bias(ga["gdn_g_b"], gdn_value_dim // tp_world),
                             (full_bias(ga["gdn_o_b"], dim).float() * inv_w).to(bf16).contiguous(),
                             ga["conv_q_w"], ga["conv_k_w"], ga["conv_v_w"],
                             ga["A_log"], ga["dt_bias"], ga["o_norm_w"])
            else:
                a = _get_block_attrs(b)
                sq_w = sc(a["sq_w"].contiguous())
                sk_w = sc(a["sk_w"].contiguous())
                sv_w = sc(a["sv_w"].contiguous())
                so_w = sr(a["so_w"].contiguous())
                sq_b = sb(full_bias(a["sq_b"], dim))
                sk_b = sb(full_bias(a["sk_b"], dim))
                sv_b = sb(full_bias(a["sv_b"], dim))
                so_b = (full_bias(a["so_b"], dim).float() * inv_w).to(bf16).contiguous()
                rms_q = sb(a["rms_q"].float().contiguous())
                rms_k = sb(a["rms_k"].float().contiguous())
                (cq_w, ck_w, cv_w, co_w, f1_w, f2_w,
                 cq_b, ck_b, cv_b, co_b, f1_b, f2_b,
                 ca_rms_q, ca_rms_k, n2w, n2b, sst) = _load_ca_ffn_weights(i, a)
                ext.load(i,
                         sq_w, sk_w, sv_w, so_w,
                         cq_w, ck_w, cv_w, co_w,
                         f1_w, f2_w,
                         sq_b, sk_b, sv_b, so_b,
                         cq_b, ck_b, cv_b, co_b,
                         f1_b, f2_b,
                         rms_q, rms_k, ca_rms_q, ca_rms_k,
                         n2w, n2b, sst)

        ssts_list = []
        for b in orig_blocks:
            sst_data = b.modulation.data.float().squeeze(0)
            ssts_list.append(sst_data)
        ssts_stacked = torch.stack(ssts_list).contiguous()
        transformer._kairos_engine_block_versions = current_versions
        return True

    _sync_engine_weights(force=True)

    _prepared_ca_lens = set()
    _rope_cache: dict = {}
    _context_sample_indices: dict = {}
    _runner_hd = hd
    _engine_ctx_len = ctx_len
    from kairos.modules.dits.kairos_dit import sinusoidal_embedding_1d

    def _get_context_sample_idx(numel: int, device: torch.device) -> torch.Tensor:
        key = (numel, device.type, device.index)
        idx = _context_sample_indices.get(key)
        if idx is not None:
            return idx
        sample_count = min(256, numel)
        if sample_count == numel:
            idx = torch.arange(numel, device=device, dtype=torch.long)
        else:
            idx = torch.linspace(0, numel - 1, steps=sample_count, device=device)
            idx = idx.round().to(torch.long)
        _context_sample_indices[key] = idx
        return idx

    def _make_context_content_key(enci: torch.Tensor):
        flat = enci.reshape(-1)
        if flat.numel() == 0:
            sample_bytes = b""
        else:
            sample_idx = _get_context_sample_idx(flat.numel(), flat.device)
            sample = flat.index_select(0, sample_idx).detach().to(dtype=torch.float32)
            sample_bytes = sample.cpu().numpy().tobytes()
        return (
            tuple(enci.shape),
            str(enci.dtype),
            enci.device.type,
            enci.device.index,
            sample_bytes,
        )

    def _run_engine_blocks(x, context, t_mod, freqs, grid_size, context_mask=None):
        del grid_size, context_mask
        first._engine_forward_calls += 1
        if x.dim() == 3 and x.shape[0] != 1:
            raise RuntimeError(f"Kairos engine patch currently supports batch size 1, got x shape {tuple(x.shape)}")
        if context.dim() == 3 and context.shape[0] != 1:
            raise RuntimeError(
                f"Kairos engine patch currently supports batch size 1 context, got context shape {tuple(context.shape)}"
            )

        xi = x.squeeze(0).contiguous() if x.dim() == 3 else x.contiguous()
        enci = context.squeeze(0).contiguous() if context.dim() == 3 else context.contiguous()
        seq = xi.shape[0]
        ctx_ptr_key = (
            int(enci.data_ptr()),
            tuple(enci.shape),
            str(enci.dtype),
            enci.device.type,
            enci.device.index,
        )

        # Prepare temb: [B, 6, D] or [B, seq, 6, D] → [1, 6, D] or [seq, 6, D]
        ei = t_mod.float().contiguous()
        while ei.dim() > 3:
            ei = ei.squeeze(2) if ei.shape[2] == 1 else ei.squeeze(0)
        if ei.dim() == 2:
            ei = ei.unsqueeze(0)

        # RoPE: Kairos complex → Helios float
        if freqs is not None and freqs.numel() > 0:
            if freqs.is_complex():
                ck = (seq, int(freqs.shape[-1]), _runner_hd)
                cached = _rope_cache.get(ck)
                if cached is None:
                    cached = _kairos_rope_to_helios(
                        freqs.detach(), _runner_hd
                    ).to(xi.device)
                    _rope_cache[ck] = cached
                rope = cached
            else:
                rope = freqs.float().contiguous()
                if rope.dim() == 3:
                    rope = rope.squeeze(0)
        else:
            rope = torch.empty(0, device=xi.device, dtype=torch.float32)

        ca_len = seq

        # Reset GDN recurrent state before each forward pass
        ext.reset_gdn_state()

        # CA K/V is static across denoise steps for a fixed prompt/context.
        # The pointer may change across denoise steps even when the contents do not,
        # so use a cheap content fingerprint and keep a pointer fast-path.
        if ctx_ptr_key == first._ca_cache_ptr_key:
            rebuilt_ca_cache = False
        else:
            ctx_content_key = _make_context_content_key(enci)
            rebuilt_ca_cache = (ctx_content_key != first._ca_cache_key)
            first._ca_cache_ptr_key = ctx_ptr_key
        if rebuilt_ca_cache:
            ext.build_ca_kv_cache(enci)
            first._ca_cache_key = ctx_content_key
            first._ca_cache_ctx_rows = int(enci.shape[0])
            first._ca_cache_rebuilds += 1

        # Warm CA SDPA/GEMM descriptors only when this length is first seen,
        # or immediately after the CA cache rebuild changed actual_ctx.
        if rebuilt_ca_cache or ca_len not in _prepared_ca_lens:
            ext.prepare_ca(ca_len)
            _prepared_ca_lens.add(ca_len)
            first._ca_prepare_rebuilds += 1

        xi = ext.forward(xi, enci, ei, ssts_stacked, rope, ca_len, 0, nl)
        return xi.unsqueeze(0) if x.dim() == 3 else xi

    def _prepare_forward_inputs(
            self,
            x,
            timestep,
            context,
            context_mask=None,
            clip_feature=None,
            y=None,
            first_frame_latents=None,
        ):
            native_training_slices = os.environ.get(
                "KAIROS_ENGINE_NATIVE_TRAINING_SLICES", "1"
            ).strip().lower() in {"1", "true", "yes", "on"}
            native_ops = _get_native_train_ops() if (native_training_slices and self.training and x.dtype == torch.bfloat16) else None
            linear_bf16 = native_ops[2] if native_ops is not None else None
            layernorm_affine = native_ops[4] if native_ops is not None else None

            t_in = sinusoidal_embedding_1d(self.freq_dim, timestep)
            if linear_bf16 is not None:
                t = _native_time_embedding(self, t_in, linear_bf16)
                t_mod = _native_time_projection(self, t, linear_bf16).unflatten(1, (6, self.dim))
                context = _native_text_embedding(self, context, linear_bf16)
            else:
                t = self.time_embedding(t_in)
                t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
                context = self.text_embedding(context)

            if first_frame_latents is not None and self.use_first_frame_cond:
                first_frame_latents = first_frame_latents.to(context.device)
                if linear_bf16 is not None:
                    img_context = _native_image_downsample(self, first_frame_latents)
                else:
                    img_context = self.image_downsample(first_frame_latents.squeeze(2))
                _, _, fh, fw = img_context.shape
                img_context = img_context.flatten(2).transpose(-2, -1)
                if linear_bf16 is not None and layernorm_affine is not None:
                    img_context = _native_mlp_rows(layernorm_affine, linear_bf16, img_context, self.image_embedding)
                    img_context = _native_pos_embed_2d(img_context, fh, fw)
                else:
                    img_context = self.image_embedding(img_context)
                    img_context = self.image_pos_embed(img_context, h=fh, w=fw)
                context = torch.cat([img_context, context], dim=1)
                if context_mask is not None:
                    context_mask = torch.cat([
                        torch.ones(
                            context.shape[0], img_context.shape[1],
                            dtype=context_mask.dtype, device=context_mask.device
                        ),
                        context_mask
                    ], dim=1)

            if self.has_image_input:
                x = torch.cat([x, y], dim=1)
                if linear_bf16 is not None and layernorm_affine is not None:
                    clip_embdding = _native_mlp_rows(layernorm_affine, linear_bf16, clip_feature, self.img_emb)
                else:
                    clip_embdding = self.img_emb(clip_feature)
                context = torch.cat([clip_embdding, context], dim=1)

            if linear_bf16 is not None:
                x, (f, h, w) = _native_patchify(self, x)
            else:
                x, (f, h, w) = self.patchify(x)
            grid_size = (f, h, w)
            use_unified_sequence_parallel = getattr(self, "use_seq_parallel", False)
            cp_world_size = 0
            if use_unified_sequence_parallel:
                from kairos.modules.utils.parallel_utils import parallel_state
                cp_world_size = parallel_state.get_context_parallel_world_size()
            if use_unified_sequence_parallel and cp_world_size > 1:
                from kairos.modules.utils.parallel_utils import parallel_state
                from kairos.modules.utils.tp_utils import _distribute_input_sp
                cp_rank = parallel_state.get_context_parallel_rank()
                x_full = rearrange(x, 'b c f h w -> b (f h w) c')
                x = _distribute_input_sp(x_full, parallel_state.get_context_parallel_group())
                del x_full

                total_tokens = f * h * w
                assert total_tokens % cp_world_size == 0
                local_tokens = total_tokens // cp_world_size
                start = cp_rank * local_tokens
                end = start + local_tokens
                global_token_idx = torch.arange(start, end, device=x.device)
                frame_idx = global_token_idx // (h * w)
                rem = global_token_idx % (h * w)
                h_idx = rem // w
                w_idx = rem % w
                freq_f_table = self.freqs[0] if self.freqs[0].device == x.device else self.freqs[0].to(x.device)
                freq_h_table = self.freqs[1] if self.freqs[1].device == x.device else self.freqs[1].to(x.device)
                freq_w_table = self.freqs[2] if self.freqs[2].device == x.device else self.freqs[2].to(x.device)
                freqs = torch.cat([
                    freq_f_table.index_select(0, frame_idx),
                    freq_h_table.index_select(0, h_idx),
                    freq_w_table.index_select(0, w_idx),
                ], dim=-1).unsqueeze(1)
            else:
                if x.dim() == 5:
                    x = x.flatten(2).transpose(-2, -1).contiguous()
                freqs = torch.cat([
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
                ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
            return x, context, t, t_mod, freqs, grid_size, context_mask, (f, h, w)

    def _run_orig_blocks(
        self,
        x,
        context,
        t_mod,
        freqs,
        grid_size,
        context_mask=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        native_training_slices = os.environ.get(
            "KAIROS_ENGINE_NATIVE_TRAINING_SLICES", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                block_x, block_context, block_t_mod, block_freqs, block_grid_size = inputs
                block_context_mask = kwargs.get("context_mask")
                if native_training_slices and self.training:
                    if module.use_linear_attn:
                        reason = _native_gdn_slice_reason(module, block_x, block_t_mod)
                        if reason is None:
                            return _run_gdn_block_native_slices(
                                module,
                                block_x,
                                block_context,
                                block_t_mod,
                                block_freqs,
                                block_grid_size,
                                context_mask=block_context_mask,
                            )
                        self._kairos_engine_native_fallback_counts[reason] = (
                            self._kairos_engine_native_fallback_counts.get(reason, 0) + 1
                        )
                        self._kairos_engine_native_last_fallback = {
                            "kind": "gdn",
                            "reason": reason,
                            "shape": tuple(block_x.shape),
                            "dtype": str(block_x.dtype),
                        }
                    else:
                        reason = _native_quadratic_slice_reason(module, block_x, block_t_mod)
                        if reason is None:
                            return _run_quadratic_block_native_slices(
                                module,
                                block_x,
                                block_context,
                                block_t_mod,
                                block_freqs,
                                block_grid_size,
                                context_mask=block_context_mask,
                            )
                        self._kairos_engine_native_fallback_counts[reason] = (
                            self._kairos_engine_native_fallback_counts.get(reason, 0) + 1
                        )
                        self._kairos_engine_native_last_fallback = {
                            "kind": "quadratic",
                            "reason": reason,
                            "shape": tuple(block_x.shape),
                            "dtype": str(block_x.dtype),
                        }
                return module(*inputs, **kwargs)
            return custom_forward

        for block in orig_blocks:
            block.train(self.training)
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, grid_size, context_mask=context_mask,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, grid_size, context_mask=context_mask,
                        use_reentrant=False,
                    )
            else:
                if (
                    native_training_slices
                    and self.training
                    and _can_use_native_quadratic_slices(block, x, t_mod)
                ):
                    x = _run_quadratic_block_native_slices(
                        block,
                        x,
                        context,
                        t_mod,
                        freqs,
                        grid_size,
                        context_mask=context_mask,
                    )
                elif (
                    native_training_slices
                    and self.training
                    and _can_use_native_gdn_slices(block, x, t_mod)
                ):
                    x = _run_gdn_block_native_slices(
                        block,
                        x,
                        context,
                        t_mod,
                        freqs,
                        grid_size,
                        context_mask=context_mask,
                    )
                else:
                    x = block(x, context, t_mod, freqs, grid_size, context_mask=context_mask)
        return x

    def _orig_model_forward(
            self,
            x,
            timestep,
            context,
            context_mask=None,
            clip_feature=None,
            y=None,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
            first_frame_latents=None,
        ):
            train_state = self._kairos_engine_make_train_state(
                x,
                timestep,
                context,
                context_mask=context_mask,
                clip_feature=clip_feature,
                y=y,
                first_frame_latents=first_frame_latents,
            )
            prepared_state = self._kairos_engine_train_prepare_state(train_state)
            return self._kairos_engine_execute_prepared(
                prepared_state,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

    def _engine_inference_forward(
        self,
        x,
        timestep,
        context,
        context_mask=None,
        clip_feature=None,
        y=None,
        first_frame_latents=None,
    ):
        _sync_engine_weights(force=False)
        x, context, t, t_mod, freqs, grid_size, context_mask, f_hw = _prepare_forward_inputs(
            self,
            x,
            timestep,
            context,
            context_mask=context_mask,
            clip_feature=clip_feature,
            y=y,
            first_frame_latents=first_frame_latents,
        )
        x = _run_engine_blocks(
            x, context, t_mod, freqs, grid_size, context_mask=context_mask
        )
        x = self.head(x, t)
        x = self.unpatchify(x, f_hw)
        return x

    class Runner(nn.Module):
        """Block-level stub. Block 0 runs all layers through the engine.
        Blocks 1..N-2 are no-ops. Block N-1 returns the cached result.
        """

        def __init__(self, idx):
            super().__init__()
            self.idx = idx

        def forward(self, x, context, t_mod, freqs, grid_size, context_mask=None):
            if self.idx == 0:
                self._result = _run_engine_blocks(
                    x, context, t_mod, freqs, grid_size, context_mask=context_mask
                )

            if self.idx == nl - 1:
                return self._first_runner._result
            return x

    first = Runner(0)
    runners = [first]
    for i in range(1, nl):
        r = Runner(i)
        object.__setattr__(r, "_first_runner", first)
        runners.append(r)
    object.__setattr__(first, "_first_runner", first)
    first._ca_cache_key = None
    first._ca_cache_ptr_key = None
    first._ca_cache_ctx_rows = None
    first._ca_cache_rebuilds = 0
    first._ca_prepare_rebuilds = 0
    first._engine_forward_calls = 0
    setattr(transformer, block_name, nn.ModuleList(runners))
    transformer._kairos_engine_run_blocks = _run_engine_blocks
    transformer._kairos_engine_orig_blocks = nn.ModuleList(orig_blocks)
    transformer._kairos_engine_native_fallback_counts = {}
    transformer._kairos_engine_native_last_fallback = None

    if not hasattr(transformer, "_kairos_engine_orig_forward"):
        transformer._kairos_engine_orig_forward = transformer.forward

    def _native_training_fallback_report(self):
        return {
            "counts": dict(self._kairos_engine_native_fallback_counts),
            "last": self._kairos_engine_native_last_fallback,
        }

    def _reset_native_training_fallback_report(self):
        self._kairos_engine_native_fallback_counts = {}
        self._kairos_engine_native_last_fallback = None

    if enable_training_bridge is None:
        enable_training_bridge = os.environ.get(
            "KAIROS_ENGINE_ENABLE_BACKWARD_BRIDGE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
    backward_checkpoint = os.environ.get(
        "KAIROS_ENGINE_BACKWARD_CHECKPOINT", "1"
    ).strip().lower() in {"1", "true", "yes", "on"}
    backward_checkpoint_offload = os.environ.get(
        "KAIROS_ENGINE_BACKWARD_CHECKPOINT_OFFLOAD", "1"
    ).strip().lower() in {"1", "true", "yes", "on"}
    backward_release_engine_weights = os.environ.get(
        "KAIROS_ENGINE_BACKWARD_RELEASE_WEIGHTS", "1"
    ).strip().lower() in {"1", "true", "yes", "on"}

    def _empty_like_ref(ref, dtype=None):
        return torch.empty(0, device=ref.device, dtype=(dtype or ref.dtype))

    def _pack_bridge_inputs(x, context_mask=None, clip_feature=None, y=None, first_frame_latents=None):
        context_mask_t = context_mask if context_mask is not None else _empty_like_ref(x, dtype=torch.uint8)
        clip_feature_t = clip_feature if clip_feature is not None else _empty_like_ref(x)
        y_t = y if y is not None else _empty_like_ref(x)
        first_frame_t = first_frame_latents if first_frame_latents is not None else _empty_like_ref(x)
        return context_mask_t, clip_feature_t, y_t, first_frame_t

    _TRAIN_STATE_KEYS = (
        "x",
        "timestep",
        "context",
        "context_mask",
        "clip_feature",
        "y",
        "first_frame_latents",
    )
    _TRAIN_OPTIONAL_KEYS = (
        "context_mask",
        "clip_feature",
        "y",
        "first_frame_latents",
    )
    _TRAIN_GRAD_KEYS = (
        "x",
        "context",
        "clip_feature",
        "y",
        "first_frame_latents",
    )

    def _make_engine_train_state(
        self,
        x,
        timestep,
        context,
        context_mask=None,
        clip_feature=None,
        y=None,
        first_frame_latents=None,
    ):
        context_mask_t, clip_feature_t, y_t, first_frame_t = _pack_bridge_inputs(
            x,
            context_mask=context_mask,
            clip_feature=clip_feature,
            y=y,
            first_frame_latents=first_frame_latents,
        )
        return {
            "x": x,
            "timestep": timestep,
            "context": context,
            "context_mask": context_mask_t,
            "clip_feature": clip_feature_t,
            "y": y_t,
            "first_frame_latents": first_frame_t,
        }

    def _engine_train_state_tensors(train_state, detach=False):
        tensors = []
        for key in _TRAIN_STATE_KEYS:
            tensor = train_state[key]
            tensors.append(tensor.detach() if detach else tensor)
        return tuple(tensors)

    def _engine_train_state_meta(train_state):
        meta = {
            "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
            "cuda_autocast_dtype": torch.get_autocast_dtype("cuda"),
            "has": {},
            "requires_grad": {},
        }
        for key in _TRAIN_OPTIONAL_KEYS:
            meta["has"][key] = train_state[key].numel() != 0
        for key in _TRAIN_GRAD_KEYS:
            meta["requires_grad"][key] = train_state[key].requires_grad
        return meta

    def _restore_engine_train_state(state_tensors, state_meta):
        train_state = {}
        for key, tensor in zip(_TRAIN_STATE_KEYS, state_tensors):
            restored = tensor.detach()
            if key in _TRAIN_GRAD_KEYS:
                restored = restored.requires_grad_(state_meta["requires_grad"][key])
            train_state[key] = restored
        for key in _TRAIN_OPTIONAL_KEYS:
            if not state_meta["has"][key]:
                train_state[key] = None
        return train_state

    def _engine_inference_forward_state(self, train_state):
        return _engine_inference_forward(
            self,
            train_state["x"],
            train_state["timestep"],
            train_state["context"],
            context_mask=train_state["context_mask"],
            clip_feature=train_state["clip_feature"],
            y=train_state["y"],
            first_frame_latents=train_state["first_frame_latents"],
        )

    def _engine_train_prepare_state(self, train_state):
        context_mask = train_state["context_mask"]
        clip_feature = train_state["clip_feature"]
        y = train_state["y"]
        first_frame_latents = train_state["first_frame_latents"]
        if context_mask is not None and context_mask.numel() == 0:
            context_mask = None
        if clip_feature is not None and clip_feature.numel() == 0:
            clip_feature = None
        if y is not None and y.numel() == 0:
            y = None
        if first_frame_latents is not None and first_frame_latents.numel() == 0:
            first_frame_latents = None
        x, context, t, t_mod, freqs, grid_size, context_mask, f_hw = _prepare_forward_inputs(
            self,
            train_state["x"],
            train_state["timestep"],
            train_state["context"],
            context_mask=context_mask,
            clip_feature=clip_feature,
            y=y,
            first_frame_latents=first_frame_latents,
        )
        return {
            "x": x,
            "context": context,
            "t": t,
            "t_mod": t_mod,
            "freqs": freqs,
            "grid_size": grid_size,
            "context_mask": context_mask,
            "f_hw": f_hw,
        }

    def _engine_execute_prepared(
        self,
        prepared_state,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        x = _run_orig_blocks(
            self,
            prepared_state["x"],
            prepared_state["context"],
            prepared_state["t_mod"],
            prepared_state["freqs"],
            prepared_state["grid_size"],
            context_mask=prepared_state["context_mask"],
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        native_training_slices = os.environ.get(
            "KAIROS_ENGINE_NATIVE_TRAINING_SLICES", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if native_training_slices and self.training and x.dtype == torch.bfloat16:
            linear_bf16 = _get_native_train_ops()[2]
            x = _native_head_forward(self, x, prepared_state["t"], linear_bf16)
        else:
            x = self.head(x, prepared_state["t"])
        x = self.unpatchify(x, prepared_state["f_hw"])
        return x

    def _engine_train_execute_prepared(self, prepared_state):
        return self._kairos_engine_execute_prepared(
            prepared_state,
            use_gradient_checkpointing=backward_checkpoint,
            use_gradient_checkpointing_offload=(
                backward_checkpoint and backward_checkpoint_offload
            ),
        )

    def _engine_train_recompute(
        self,
        x,
        timestep,
        context,
        context_mask=None,
        clip_feature=None,
        y=None,
        first_frame_latents=None,
    ):
        train_state = self._kairos_engine_make_train_state(
            x,
            timestep,
            context,
            context_mask=context_mask,
            clip_feature=clip_feature,
            y=y,
            first_frame_latents=first_frame_latents,
        )
        return self._kairos_engine_train_recompute_state(train_state)

    def _engine_train_recompute_state(self, train_state):
        prepared_state = self._kairos_engine_train_prepare_state(train_state)
        return self._kairos_engine_train_execute_prepared(prepared_state)

    class _EngineBackwardBridge(torch.autograd.Function):

        @staticmethod
        def forward(
            ctx,
            x,
            timestep,
            context,
            context_mask_t,
            clip_feature_t,
            y_t,
            first_frame_latents_t,
            *params,
        ):
            ctx.params = params
            train_state = {
                "x": x,
                "timestep": timestep,
                "context": context,
                "context_mask": context_mask_t,
                "clip_feature": clip_feature_t,
                "y": y_t,
                "first_frame_latents": first_frame_latents_t,
            }
            ctx.train_state_meta = _engine_train_state_meta(train_state)
            ctx.save_for_backward(*_engine_train_state_tensors(train_state, detach=True))
            with torch.no_grad():
                prepared_state = transformer._kairos_engine_train_prepare_state(train_state)
                return transformer._kairos_engine_train_execute_prepared(prepared_state)

        @staticmethod
        def backward(ctx, grad_output):
            train_state = _restore_engine_train_state(ctx.saved_tensors, ctx.train_state_meta)
            grad_slots = [
                key
                for key in _TRAIN_GRAD_KEYS
                if train_state[key] is not None and train_state[key].requires_grad
            ]

            with torch.enable_grad(), torch.amp.autocast(
                "cuda",
                dtype=ctx.train_state_meta["cuda_autocast_dtype"],
                enabled=ctx.train_state_meta["cuda_autocast_enabled"],
            ):
                if backward_release_engine_weights:
                    ext.release_loaded_weights()
                    transformer._kairos_engine_block_versions = None
                if backward_checkpoint_offload:
                    torch.cuda.empty_cache()
                out = transformer._kairos_engine_train_recompute_state(train_state)
                torch.autograd.backward(out, grad_output)

            grad_x = None
            grad_context = None
            grad_clip_feature = None
            grad_y = None
            grad_first_frame = None
            param_grads = [None] * len(ctx.params)
            grad_by_slot = {
                "x": train_state["x"].grad if train_state["x"].requires_grad else None,
                "context": train_state["context"].grad if train_state["context"].requires_grad else None,
                "clip_feature": (
                    train_state["clip_feature"].grad
                    if train_state["clip_feature"] is not None and train_state["clip_feature"].requires_grad
                    else None
                ),
                "y": (
                    train_state["y"].grad
                    if train_state["y"] is not None and train_state["y"].requires_grad
                    else None
                ),
                "first_frame_latents": (
                    train_state["first_frame_latents"].grad
                    if train_state["first_frame_latents"] is not None and train_state["first_frame_latents"].requires_grad
                    else None
                ),
            }
            for slot in grad_slots:
                grad = grad_by_slot[slot]
                if slot == "x":
                    grad_x = grad
                elif slot == "context":
                    grad_context = grad
                elif slot == "clip_feature":
                    grad_clip_feature = grad
                elif slot == "y":
                    grad_y = grad
                elif slot == "first_frame_latents":
                    grad_first_frame = grad

            return (
                grad_x,
                None,
                grad_context,
                None,
                grad_clip_feature,
                grad_y,
                grad_first_frame,
                *param_grads,
            )

    def _engine_train_forward(
        self,
        x,
        timestep,
        context,
        context_mask=None,
        clip_feature=None,
        y=None,
        first_frame_latents=None,
    ):
        params = tuple(p for p in self.parameters() if p.requires_grad)
        train_state = self._kairos_engine_make_train_state(
            x,
            timestep,
            context,
            context_mask=context_mask,
            clip_feature=clip_feature,
            y=y,
            first_frame_latents=first_frame_latents,
        )
        return _EngineBackwardBridge.apply(
            *_engine_train_state_tensors(train_state, detach=False),
            *params,
        )

    def _engine_forward(
        self,
        x,
        timestep,
        context,
        context_mask=None,
        clip_feature=None,
        y=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        first_frame_latents=None,
        **kwargs,
    ):
        # KairosDiT.forward accepts **kwargs but ignores them; keep the engine
        # state pipeline active instead of bouncing through the old wrapper.
        del kwargs

        if use_gradient_checkpointing or use_gradient_checkpointing_offload:
            train_state = self._kairos_engine_make_train_state(
                x,
                timestep,
                context,
                context_mask=context_mask,
                clip_feature=clip_feature,
                y=y,
                first_frame_latents=first_frame_latents,
            )
            prepared_state = self._kairos_engine_train_prepare_state(train_state)
            return self._kairos_engine_execute_prepared(
                prepared_state,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

        if self.training:
            if not enable_training_bridge:
                train_state = self._kairos_engine_make_train_state(
                    x,
                    timestep,
                    context,
                    context_mask=context_mask,
                    clip_feature=clip_feature,
                    y=y,
                    first_frame_latents=first_frame_latents,
                )
                prepared_state = self._kairos_engine_train_prepare_state(train_state)
                return self._kairos_engine_execute_prepared(
                    prepared_state,
                    use_gradient_checkpointing=False,
                    use_gradient_checkpointing_offload=False,
                )

            return self._kairos_engine_train_forward(
                x,
                timestep,
                context,
                context_mask=context_mask,
                clip_feature=clip_feature,
                y=y,
                first_frame_latents=first_frame_latents,
            )

        return _engine_inference_forward(
            self,
            x,
            timestep,
            context,
            context_mask=context_mask,
            clip_feature=clip_feature,
            y=y,
            first_frame_latents=first_frame_latents,
        )

    transformer._kairos_engine_inference_forward = types.MethodType(_engine_inference_forward, transformer)
    transformer._kairos_engine_inference_forward_state = types.MethodType(_engine_inference_forward_state, transformer)
    transformer._kairos_engine_make_train_state = types.MethodType(_make_engine_train_state, transformer)
    transformer._kairos_engine_train_prepare_state = types.MethodType(_engine_train_prepare_state, transformer)
    transformer._kairos_engine_execute_prepared = types.MethodType(_engine_execute_prepared, transformer)
    transformer._kairos_engine_train_execute_prepared = types.MethodType(_engine_train_execute_prepared, transformer)
    transformer._kairos_engine_train_recompute = types.MethodType(_engine_train_recompute, transformer)
    transformer._kairos_engine_train_recompute_state = types.MethodType(_engine_train_recompute_state, transformer)
    transformer._kairos_engine_train_forward = types.MethodType(_engine_train_forward, transformer)
    transformer._kairos_engine_native_training_fallback_report = types.MethodType(_native_training_fallback_report, transformer)
    transformer._kairos_engine_reset_native_training_fallback_report = types.MethodType(_reset_native_training_fallback_report, transformer)
    transformer.forward = types.MethodType(_engine_forward, transformer)

    if verbose:
        if enable_training_bridge:
            print("  Training bridge: enabled (engine forward + recompute backward)")
        print(f"  Kairos engine ready in {time.time()-t0:.1f}s")
    return nl
