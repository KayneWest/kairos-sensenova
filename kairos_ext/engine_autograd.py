import os

import torch
import torch.nn.functional as F
from einops import rearrange

from kairos_ext.kairos_engine_patch import get_ext

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except Exception:
    flash_attn_interface = None
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = hasattr(flash_attn, "flash_attn_func")
except Exception:
    flash_attn = None
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except Exception:
    sageattn = None
    SAGE_ATTN_AVAILABLE = False


def _flash_attention_impl(q, k, v, num_heads, compatibility_mode=False, attn_mask=None, window_size=(-1, -1)):
    if compatibility_mode or attn_mask is not None:
        q = rearrange(q, "b s n d -> b n s d")
        k = rearrange(k, "b s n d -> b n s d")
        v = rearrange(v, "b s n d -> b n s d")
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "b n s d -> b s n d")
    elif FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_func(q, k, v, window_size=window_size)
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_func(q, k, v, window_size=window_size)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s n d -> b n s d")
        k = rearrange(k, "b s n d -> b n s d")
        v = rearrange(v, "b s n d -> b n s d")
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s n d")
    else:
        q = rearrange(q, "b s n d -> b n s d")
        k = rearrange(k, "b s n d -> b n s d")
        v = rearrange(v, "b s n d -> b n s d")
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "b n s d -> b s n d")
    return x


def _rope_apply_impl(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float16).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def _attention_valid_mask(attn_mask, window_size, q0, q1, k0, k1, device):
    valid = None
    if attn_mask is not None and attn_mask.numel() != 0:
        valid = attn_mask[..., q0:q1, k0:k1].to(torch.bool)
    wl, wr = window_size
    if wl != -1 or wr != -1:
        q_idx = torch.arange(q0, q1, device=device)[:, None]
        k_idx = torch.arange(k0, k1, device=device)[None, :]
        win_valid = torch.ones((q1 - q0, k1 - k0), device=device, dtype=torch.bool)
        if wl != -1:
            win_valid &= k_idx >= (q_idx - wl)
        if wr != -1:
            win_valid &= k_idx <= (q_idx + wr)
        win_valid = win_valid.view(1, 1, q1 - q0, k1 - k0)
        valid = win_valid if valid is None else (valid & win_valid)
    return valid


def _attention_effective_window(attn_mask, window_size, compatibility_mode):
    # Match the current training forward exactly. The forward path does not
    # materialize `window_size` into an explicit mask, and on this stack the
    # selected backend behaves like global attention unless the caller passes
    # an explicit attn_mask. Use the same effective semantics in backward.
    _ = window_size
    _ = compatibility_mode
    return (-1, -1)


def _attention_backward_blockwise(q, k, v, out, grad_out, attn_mask, window_size):
    qf = q.float().permute(0, 2, 1, 3).contiguous()
    kf = k.float().permute(0, 2, 1, 3).contiguous()
    vf = v.float().permute(0, 2, 1, 3).contiguous()
    of = out.float().permute(0, 2, 1, 3).contiguous()
    gof = grad_out.float().permute(0, 2, 1, 3).contiguous()

    B, H, SQ, D = qf.shape
    SK = kf.shape[2]
    scale = D ** -0.5
    q_block = int(os.environ.get("KAIROS_ATTN_BWD_Q_BLOCK", "128"))
    k_block = int(os.environ.get("KAIROS_ATTN_BWD_K_BLOCK", "256"))

    dq = torch.zeros_like(qf)
    dk = torch.zeros_like(kf)
    dv = torch.zeros_like(vf)

    for q0 in range(0, SQ, q_block):
        q1 = min(q0 + q_block, SQ)
        Q = qf[:, :, q0:q1, :]
        dO = gof[:, :, q0:q1, :]
        O = of[:, :, q0:q1, :]
        D_term = (dO * O).sum(dim=-1)

        m = torch.full((B, H, q1 - q0), -float("inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((B, H, q1 - q0), device=q.device, dtype=torch.float32)

        for k0 in range(0, SK, k_block):
            k1 = min(k0 + k_block, SK)
            K = kf[:, :, k0:k1, :]
            scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) * scale
            valid = _attention_valid_mask(attn_mask, window_size, q0, q1, k0, k1, q.device)
            if valid is not None:
                scores = scores.masked_fill(~valid, -float("inf"))
            block_max = scores.max(dim=-1).values
            m_new = torch.maximum(m, block_max)
            m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
            m_new_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            alpha = torch.where(torch.isfinite(m), torch.exp(m_safe - m_new_safe), torch.zeros_like(m))
            beta = torch.exp(scores - m_new_safe.unsqueeze(-1))
            if valid is not None:
                beta = torch.where(valid, beta, torch.zeros_like(beta))
            l = l * alpha + beta.sum(dim=-1)
            m = m_new

        lse = m + torch.log(torch.clamp_min(l, 1e-20))
        lse = torch.where(l > 0, lse, torch.full_like(lse, -float("inf")))
        dQ = torch.zeros_like(Q)

        for k0 in range(0, SK, k_block):
            k1 = min(k0 + k_block, SK)
            K = kf[:, :, k0:k1, :]
            V = vf[:, :, k0:k1, :]
            scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) * scale
            valid = _attention_valid_mask(attn_mask, window_size, q0, q1, k0, k1, q.device)
            if valid is not None:
                scores = scores.masked_fill(~valid, -float("inf"))
            lse_safe = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
            P = torch.exp(scores - lse_safe.unsqueeze(-1))
            if valid is not None:
                P = torch.where(valid, P, torch.zeros_like(P))
            dV_block = torch.einsum("bhqk,bhqd->bhkd", P, dO)
            dP = torch.einsum("bhqd,bhkd->bhqk", dO, V)
            dS = P * (dP - D_term.unsqueeze(-1))
            dQ = dQ + torch.einsum("bhqk,bhkd->bhqd", dS, K) * scale
            dk[:, :, k0:k1, :] += torch.einsum("bhqk,bhqd->bhkd", dS, Q) * scale
            dv[:, :, k0:k1, :] += dV_block

        dq[:, :, q0:q1, :] = dQ

    return (
        dq.permute(0, 2, 1, 3).to(q.dtype),
        dk.permute(0, 2, 1, 3).to(k.dtype),
        dv.permute(0, 2, 1, 3).to(v.dtype),
    )


class GateResidualSstsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, ssts_gate, temb_gate):
        ext = get_ext()
        (out,) = ext.gate_residual_ssts_train(
            x.contiguous(),
            y.contiguous(),
            ssts_gate.contiguous(),
            temb_gate.contiguous(),
        )
        ctx.save_for_backward(y, ssts_gate, temb_gate)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext = get_ext()
        y, ssts_gate, temb_gate = ctx.saved_tensors
        dx, dy, dssts, dtemb = ext.gate_residual_ssts_backward(
            grad_out.contiguous(),
            y.contiguous(),
            ssts_gate.contiguous(),
            temb_gate.contiguous(),
        )
        return dx, dy, dssts, dtemb


def gate_residual_ssts(x, y, ssts_gate, temb_gate):
    return GateResidualSstsFunction.apply(x, y, ssts_gate, temb_gate)


class LnAdaLnSstsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ssts_scale, ssts_shift, temb_scale, temb_shift):
        ext = get_ext()
        (out,) = ext.ln_adaln_ssts_train(
            x.contiguous(),
            ssts_scale.contiguous(),
            ssts_shift.contiguous(),
            temb_scale.contiguous(),
            temb_shift.contiguous(),
        )
        ctx.save_for_backward(x, ssts_scale, ssts_shift, temb_scale, temb_shift)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext = get_ext()
        x, ssts_scale, ssts_shift, temb_scale, temb_shift = ctx.saved_tensors
        dx, dssts_scale, dssts_shift, dtemb_scale, dtemb_shift = ext.ln_adaln_ssts_backward(
            grad_out.contiguous(),
            x.contiguous(),
            ssts_scale.contiguous(),
            ssts_shift.contiguous(),
            temb_scale.contiguous(),
            temb_shift.contiguous(),
        )
        return dx, dssts_scale, dssts_shift, dtemb_scale, dtemb_shift


def ln_adaln_ssts(x, ssts_scale, ssts_shift, temb_scale, temb_shift):
    return LnAdaLnSstsFunction.apply(x, ssts_scale, ssts_shift, temb_scale, temb_shift)


class LinearBf16Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias):
        ext = get_ext()
        bias_saved = bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        (out,) = ext.linear_bf16_train(
            x.contiguous(),
            w.contiguous(),
            bias_saved.contiguous(),
        )
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, w, bias_saved)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext = get_ext()
        x, w, bias_saved = ctx.saved_tensors
        dx, dw, db = ext.linear_bf16_backward(
            grad_out.contiguous(),
            x.contiguous(),
            w.contiguous(),
            bias_saved.contiguous(),
        )
        return dx, dw, (db if ctx.has_bias else None)


def linear_bf16(x, w, bias=None):
    return LinearBf16Function.apply(x, w, bias)


class RmsNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        ext = get_ext()
        (out,) = ext.rmsnorm_train(
            x.contiguous(),
            w.contiguous(),
            float(eps),
        )
        ctx.eps = float(eps)
        ctx.save_for_backward(x, w)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext = get_ext()
        x, w = ctx.saved_tensors
        dx, dw = ext.rmsnorm_backward(
            grad_out.contiguous(),
            x.contiguous(),
            w.contiguous(),
            ctx.eps,
        )
        return dx, dw, None


def rmsnorm(x, w, eps=1e-6):
    return RmsNormFunction.apply(x, w, eps)


class LayerNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, eps):
        ext = get_ext()
        w_saved = w if w is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        b_saved = b if b is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        (out,) = ext.layernorm_affine_train(
            x.contiguous(),
            w_saved.contiguous(),
            b_saved.contiguous(),
            float(eps),
        )
        ctx.has_w = w is not None
        ctx.has_b = b is not None
        ctx.eps = float(eps)
        ctx.save_for_backward(x, w_saved, b_saved)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext = get_ext()
        x, w_saved, b_saved = ctx.saved_tensors
        dx, dw, db = ext.layernorm_affine_backward(
            grad_out.contiguous(),
            x.contiguous(),
            w_saved.contiguous(),
            b_saved.contiguous(),
            ctx.eps,
        )
        return dx, (dw if ctx.has_w else None), (db if ctx.has_b else None), None


def layernorm_affine(x, w=None, b=None, eps=1e-6):
    return LayerNormAffineFunction.apply(x, w, b, eps)


class AttentionCoreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, num_heads, attn_mask, window_left, window_right, compatibility_mode):
        ctx.num_heads = int(num_heads)
        ctx.window_size = (int(window_left), int(window_right))
        ctx.compatibility_mode = bool(compatibility_mode)
        ctx.has_attn_mask = attn_mask is not None and attn_mask.numel() != 0
        attn_mask_saved = attn_mask if ctx.has_attn_mask else torch.empty(0, device=q.device, dtype=q.dtype)
        out = _flash_attention_impl(
            q,
            k,
            v,
            num_heads=ctx.num_heads,
            compatibility_mode=ctx.compatibility_mode,
            attn_mask=(attn_mask_saved if ctx.has_attn_mask else None),
            window_size=ctx.window_size,
        )
        ctx.save_for_backward(q, k, v, out, attn_mask_saved)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, attn_mask_saved = ctx.saved_tensors
        effective_window = _attention_effective_window(
            attn_mask_saved if ctx.has_attn_mask else None,
            ctx.window_size,
            ctx.compatibility_mode,
        )
        dq, dk, dv = _attention_backward_blockwise(
            q,
            k,
            v,
            out,
            grad_out,
            attn_mask_saved if ctx.has_attn_mask else None,
            effective_window,
        )
        return dq, dk, dv, None, None, None, None, None


def attention_core(q, k, v, num_heads, attn_mask=None, window_size=(-1, -1), compatibility_mode=False):
    attn_mask_t = attn_mask if attn_mask is not None else torch.empty(0, device=q.device, dtype=q.dtype)
    return AttentionCoreFunction.apply(
        q,
        k,
        v,
        int(num_heads),
        attn_mask_t,
        int(window_size[0]),
        int(window_size[1]),
        bool(compatibility_mode),
    )


class RopeApplyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freqs, num_heads):
        ctx.num_heads = int(num_heads)
        ctx.save_for_backward(x, freqs)
        return _rope_apply_impl(x, freqs, ctx.num_heads)

    @staticmethod
    def backward(ctx, grad_out):
        x, freqs = ctx.saved_tensors
        with torch.enable_grad():
            x_r = x.detach().requires_grad_(True)
            out = _rope_apply_impl(x_r, freqs, ctx.num_heads)
            torch.autograd.backward(out, grad_out)
        return x_r.grad, None, None


def rope_apply_core(x, freqs, num_heads):
    return RopeApplyFunction.apply(x, freqs, int(num_heads))


class GdnL2NormScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, scale):
        ext = get_ext()
        q_out, k_out = ext.gdn_l2norm_scale_train(
            q.contiguous(),
            k.contiguous(),
            float(scale),
        )
        ctx.scale = float(scale)
        ctx.save_for_backward(q, k)
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        ext = get_ext()
        q, k = ctx.saved_tensors
        dq, dk = ext.gdn_l2norm_scale_backward(
            grad_q.contiguous(),
            grad_k.contiguous(),
            q.contiguous(),
            k.contiguous(),
            ctx.scale,
        )
        return dq, dk, None


def gdn_l2norm_scale(q, k, scale):
    return GdnL2NormScaleFunction.apply(q, k, scale)


class GdnCausalConvSiluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ext = get_ext()
        (out,) = ext.gdn_causal_conv_silu_train(
            x.contiguous(),
            w.contiguous(),
        )
        ctx.save_for_backward(x, w)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext = get_ext()
        x, w = ctx.saved_tensors
        dx, dw = ext.gdn_causal_conv_silu_backward(
            grad_out.contiguous(),
            x.contiguous(),
            w.contiguous(),
        )
        return dx, dw


def gdn_causal_conv_silu(x, w):
    return GdnCausalConvSiluFunction.apply(x, w)


class GdnComputeGatesFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_proj, b_proj, A_log, dt_bias):
        ext = get_ext()
        g_out, beta_out = ext.gdn_compute_gates_train(
            a_proj.contiguous(),
            b_proj.contiguous(),
            A_log.contiguous(),
            dt_bias.contiguous(),
        )
        ctx.save_for_backward(a_proj, b_proj, A_log, dt_bias)
        return g_out, beta_out

    @staticmethod
    def backward(ctx, grad_g, grad_beta):
        ext = get_ext()
        a_proj, b_proj, A_log, dt_bias = ctx.saved_tensors
        da, db, dA, ddt = ext.gdn_compute_gates_backward(
            grad_g.contiguous(),
            grad_beta.contiguous(),
            a_proj.contiguous(),
            b_proj.contiguous(),
            A_log.contiguous(),
            dt_bias.contiguous(),
        )
        return da, db, dA, ddt


def gdn_compute_gates(a_proj, b_proj, A_log, dt_bias):
    return GdnComputeGatesFunction.apply(a_proj, b_proj, A_log, dt_bias)


class GdnRecurrentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, state0, scale):
        ext = get_ext()
        o_out, stateT = ext.gdn_recurrent_train(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            state0.contiguous(),
            float(scale),
        )
        ctx.scale = float(scale)
        ctx.save_for_backward(q, k, v, g, beta, stateT)
        return o_out, stateT

    @staticmethod
    def backward(ctx, grad_o, grad_stateT):
        ext = get_ext()
        q, k, v, g, beta, stateT = ctx.saved_tensors
        if grad_stateT is None:
            grad_stateT = torch.zeros_like(stateT)
        dq, dk, dv, dg, dbeta, dstate0 = ext.gdn_recurrent_backward(
            grad_o.contiguous(),
            grad_stateT.contiguous(),
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            stateT.contiguous(),
            ctx.scale,
        )
        return dq, dk, dv, dg, dbeta, dstate0, None


def gdn_recurrent(q, k, v, g, beta, state0=None, scale=1.0):
    if state0 is None:
        state0 = torch.zeros(
            (q.shape[1], q.shape[2], v.shape[2]),
            device=q.device,
            dtype=torch.float32,
        )
    return GdnRecurrentFunction.apply(q, k, v, g, beta, state0, scale)


class GdnRmsNormSiluGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rec_out, gate, weight, eps):
        ext = get_ext()
        (out,) = ext.gdn_rmsnorm_silu_gate_train(
            rec_out.contiguous(),
            gate.contiguous(),
            weight.contiguous(),
            float(eps),
        )
        ctx.eps = float(eps)
        ctx.save_for_backward(rec_out, gate, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext = get_ext()
        rec_out, gate, weight = ctx.saved_tensors
        drec, dgate, dweight = ext.gdn_rmsnorm_silu_gate_backward(
            grad_out.contiguous(),
            rec_out.contiguous(),
            gate.contiguous(),
            weight.contiguous(),
            ctx.eps,
        )
        return drec, dgate, dweight, None


def gdn_rmsnorm_silu_gate(rec_out, gate, weight, eps=1e-6):
    return GdnRmsNormSiluGateFunction.apply(rec_out, gate, weight, eps)
