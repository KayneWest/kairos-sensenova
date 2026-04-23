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
import torch
import torch.nn as nn

_ext = None


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
    cudnn_inc = os.environ.get("CUDNN_INC", "")
    cudnn_lib = os.environ.get("CUDNN_LIB", "")
    if not cudnn_lib and cudnn_inc:
        cudnn_root = os.path.dirname(cudnn_inc.rstrip("/"))
        guessed_cudnn_lib = os.path.join(cudnn_root, "lib")
        if os.path.isdir(guessed_cudnn_lib):
            cudnn_lib = guessed_cudnn_lib
    nccl_inc = os.environ.get("NCCL_HOME", "/usr")
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
                 tp_rank=0, tp_world=1):
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

    # Load weights for ALL layers
    for i, b in enumerate(blocks):
        if _is_gdn_layer(i):
            # GDN layer: load CA/FFN via ext.load() with dummy SA weights,
            # then load GDN-specific weights via ext.load_gdn()
            ga = _get_gdn_block_attrs(b)
            (cq_w, ck_w, cv_w, co_w, f1_w, f2_w,
             cq_b, ck_b, cv_b, co_b, f1_b, f2_b,
             ca_rms_q, ca_rms_k, n2w, n2b, sst) = _load_ca_ffn_weights(i, ga)
            # Load with dummy SA weights (engine uses GDN path for these layers)
            z_w = torch.zeros(dim // tp_world, dim, device=dev, dtype=bf16)
            z_b = zb(dim // tp_world)
            z_rms = torch.ones(dim // tp_world, device=dev, dtype=f32)
            z_so_b = zb(dim)
            ext.load(i,
                     z_w, z_w, z_w, z_w[:, :dim // tp_world],  # dummy SA
                     cq_w, ck_w, cv_w, co_w,
                     f1_w, f2_w,
                     z_b, z_b, z_b, z_so_b,  # dummy SA biases
                     cq_b, ck_b, cv_b, co_b,
                     f1_b, f2_b,
                     z_rms, z_rms, ca_rms_q, ca_rms_k,
                     n2w, n2b, sst)
            # Load GDN-specific weights
            gdn_q_w = sc(ga["gdn_q_w"].contiguous())
            gdn_k_w = sc(ga["gdn_k_w"].contiguous())
            gdn_v_w = sc(ga["gdn_v_w"].contiguous())
            gdn_g_w = sc(ga["gdn_g_w"].contiguous())
            gdn_o_w = sr(ga["gdn_o_w"].contiguous())
            # a/b projections: replicated (small, NH outputs)
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
            # Quadratic layer
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

    # Stack modulation tables for the engine (all layers)
    ssts_list = []
    for i, b in enumerate(blocks):
        sst_data = b.modulation.data.float().squeeze(0)
        ssts_list.append(sst_data)
    ssts_stacked = torch.stack(ssts_list).contiguous()

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
            sample = flat.index_select(0, sample_idx).to(dtype=torch.float32)
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
                return self._first._result
            return x

    first = Runner(0)
    runners = [first]
    for i in range(1, nl):
        r = Runner(i)
        r._first = first
        runners.append(r)
    first._first = first
    first._ca_cache_key = None
    first._ca_cache_ptr_key = None
    first._ca_cache_ctx_rows = None
    first._ca_cache_rebuilds = 0
    first._ca_prepare_rebuilds = 0
    first._engine_forward_calls = 0
    setattr(transformer, block_name, nn.ModuleList(runners))
    transformer._kairos_engine_run_blocks = _run_engine_blocks

    if not hasattr(transformer, "_kairos_engine_orig_forward"):
        transformer._kairos_engine_orig_forward = transformer.forward

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
        if self.training or use_gradient_checkpointing or use_gradient_checkpointing_offload:
            return self._kairos_engine_orig_forward(
                x,
                timestep,
                context,
                context_mask=context_mask,
                clip_feature=clip_feature,
                y=y,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                first_frame_latents=first_frame_latents,
                **kwargs,
            )

        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep)
        )
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if first_frame_latents is not None and self.use_first_frame_cond:
            first_frame_latents = first_frame_latents.to(context.device)
            img_context = self.image_downsample(first_frame_latents.squeeze(2))
            _, _, fh, fw = img_context.shape
            img_context = img_context.flatten(2).transpose(-2, -1)
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
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)
        grid_size = (f, h, w)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        x = _run_engine_blocks(
            x, context, t_mod, freqs, grid_size, context_mask=context_mask
        )
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    transformer.forward = types.MethodType(_engine_forward, transformer)

    if verbose:
        print(f"  Kairos engine ready in {time.time()-t0:.1f}s")
    return nl
