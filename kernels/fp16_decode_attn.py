"""fp16 flash-decoding attention -- the *control* for the fused kernel.

Why this file exists
--------------------
The fused kernel beats PyTorch's fp16 SDPA by a large factor, but that number
answers the wrong question. Two different things are being changed at once:

1. the KV cache is 4-bit and never round-trips through DRAM as fp16, and
2. the work is split across the history (flash-decoding), so a batch-1 decode
   step fills the GPU instead of leaving most of it idle.

PyTorch's SDPA does not do (2) for ``q_len == 1`` -- measured at 8-33 GB/s on a
card with far more bandwidth than that -- so a fused-vs-SDPA comparison silently
credits the quantization for a parallelization win.

This kernel is the same shape as ``fused_decode_attn`` -- same split over the
history, same online softmax, same GQA amortization, same combine kernel -- with
the dequantization removed and K/V read as plain fp16. So:

    fp16 SDPA        -> this kernel      = the flash-decoding effect alone
    this kernel      -> fused 4-bit      = the quantization/fusion effect alone

Both numbers are reported separately in the audit. Neither is the headline on
its own.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    triton = None
    tl = None
    _IMPORT_ERROR = exc

if triton is not None:

    _LOG2E_F = tl.constexpr(1.4426950408889634)

    @triton.jit
    def _fp16_decode_attn_split(
        Q, K, V,
        ACC, MAXS, SUMS, OUT,
        sm_scale,
        S,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ab, stride_ah, stride_as, stride_ad,
        stride_mb, stride_mh, stride_ms,
        stride_ob, stride_oh, stride_od,
        HKV: tl.constexpr,
        GROUP: tl.constexpr,
        D: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SPLIT_SIZE: tl.constexpr,
        SINGLE_SPLIT: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_s = tl.program_id(1)
        b = pid_bh // HKV
        hkv = pid_bh % HKV

        offs_h = tl.arange(0, BLOCK_H)
        h_mask = offs_h < GROUP
        qh = hkv * GROUP + offs_h
        offs_d = tl.arange(0, D)

        q = tl.load(
            Q + b * stride_qb + qh[:, None] * stride_qh + offs_d[None, :] * stride_qd,
            mask=h_mask[:, None],
            other=0.0,
        ).to(tl.float16)

        k_base = K + b * stride_kb + hkv * stride_kh
        v_base = V + b * stride_vb + hkv * stride_vh

        m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)

        lo = pid_s * SPLIT_SIZE
        hi = tl.minimum(lo + SPLIT_SIZE, S)
        qk_scale = sm_scale * _LOG2E_F

        for start_n in range(lo, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < hi

            k = tl.load(
                k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float16)

            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            m_i = m_new

            v = tl.load(
                v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float16)
            acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)

        if SINGLE_SPLIT:
            tl.store(
                OUT + b * stride_ob + qh[:, None] * stride_oh + offs_d[None, :] * stride_od,
                acc / l_i[:, None],
                mask=h_mask[:, None],
            )
        else:
            tl.store(
                ACC + b * stride_ab + qh[:, None] * stride_ah + pid_s * stride_as
                + offs_d[None, :] * stride_ad,
                acc,
                mask=h_mask[:, None],
            )
            tl.store(MAXS + b * stride_mb + qh * stride_mh + pid_s * stride_ms, m_i, mask=h_mask)
            tl.store(SUMS + b * stride_mb + qh * stride_mh + pid_s * stride_ms, l_i, mask=h_mask)


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def fp16_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    num_splits: int | None = None,
    out: torch.Tensor | None = None,
    _workspace: dict | None = None,
) -> torch.Tensor:
    """Decode attention on an unquantized fp16 cache, same shape as the fused kernel.

    q: (B, HQ, D); k, v: (B, HKV, S, D) fp16. Returns (B, HQ, D) float32.
    """
    if triton is None:
        raise RuntimeError(f"triton unavailable: {_IMPORT_ERROR!r}")

    # Reuse the fused kernel's split heuristic and combine kernel verbatim, so
    # the control differs from it only in how K/V are read.
    from .fused_decode_attn import _combine_splits, pick_num_splits

    B, HQ, D = q.shape
    _, HKV, S, _ = k.shape
    if HQ % HKV != 0:
        raise ValueError(f"HQ={HQ} must be a multiple of HKV={HKV}")

    group = HQ // HKV
    sm_scale = 1.0 / math.sqrt(D) if sm_scale is None else sm_scale
    block_n = min(block_n, max(16, _next_pow2(S)))

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    n_blocks = triton.cdiv(S, block_n)
    if num_splits is None:
        num_splits = pick_num_splits(B, HKV, n_blocks)
    num_splits = max(1, min(num_splits, n_blocks))
    blocks_per_split = triton.cdiv(n_blocks, num_splits)
    num_splits = triton.cdiv(n_blocks, blocks_per_split)
    split_size = blocks_per_split * block_n

    if out is None:
        out = torch.empty((B, HQ, D), device=q.device, dtype=torch.float32)

    single = num_splits == 1
    ws = _workspace if _workspace is not None else {}
    key = (B, HQ, num_splits, D)
    if single:
        acc = ws.get("dummy_acc")
        if acc is None:
            acc = torch.empty((1, 1, 1, 1), device=q.device, dtype=torch.float32)
            ws["dummy_acc"] = acc
        maxs = sums = acc.view(-1)[:1].view(1, 1, 1)
    else:
        if ws.get("key") == key:
            acc, maxs, sums = ws["acc"], ws["maxs"], ws["sums"]
        else:
            acc = torch.empty((B, HQ, num_splits, D), device=q.device, dtype=torch.float32)
            maxs = torch.empty((B, HQ, num_splits), device=q.device, dtype=torch.float32)
            sums = torch.empty((B, HQ, num_splits), device=q.device, dtype=torch.float32)
            ws.update(key=key, acc=acc, maxs=maxs, sums=sums)

    _fp16_decode_attn_split[(B * HKV, num_splits)](
        q, k, v,
        acc, maxs, sums, out,
        sm_scale,
        S,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        maxs.stride(0), maxs.stride(1), maxs.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        HKV=HKV,
        GROUP=group,
        D=D,
        BLOCK_H=max(16, _next_pow2(group)),
        BLOCK_N=block_n,
        SPLIT_SIZE=split_size,
        SINGLE_SPLIT=single,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if not single:
        _combine_splits[(B * HQ,)](
            acc, maxs, sums, out,
            num_splits,
            acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
            maxs.stride(0), maxs.stride(1), maxs.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            HQ=HQ,
            D=D,
            num_warps=4,
        )

    return out
