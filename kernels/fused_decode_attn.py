"""Fused dequantize + decode-attention Triton kernel.

The problem
-----------
A naive quantized KV cache does this on every decode step::

    K_fp16 = dequantize(K_packed)     # writes S x D fp16 to DRAM
    V_fp16 = dequantize(V_packed)     # writes S x D fp16 to DRAM
    out    = attention(q, K_fp16, V_fp16)   # reads them straight back

Decode attention is entirely memory-bound (one query row against the whole
history), so the round trip through DRAM for the reconstituted cache is close to
*all* of the cost. With 4-bit codes the packed cache is 4x smaller than fp16, so
the naive path moves roughly

    0.5 D (read packed) + 2 D (write fp16) + 2 D (read fp16)  =  4.5 D bytes

per cached element-row, where the fused kernel moves 0.5 D. The whole point of
this project is to collect that ~4x of avoidable traffic.

The kernel
----------
Flash-decoding shape: split the history across programs, each computing a
partial online-softmax result, then a tiny second kernel reduces the partials.

Two details are worth calling out:

*Split-P unpacking without a shuffle.* Codes are packed so that byte ``j`` holds
dims ``j, j+DP, j+2*DP, ...`` (see ``quantize.pack_codes``). Rather than loading
``DP`` bytes and de-interleaving, the kernel builds an index vector over the full
head dim and loads byte ``d % DP`` with shift ``(d // DP) * nbits``. Each byte is
loaded ``P`` times, but those extra loads all hit the same cache line, so DRAM
traffic is unchanged and the result is a dense ``(BLOCK_N, D)`` tile with no
reshape, transpose-through-shared-memory, or unrolled accumulator list.

*GQA amortization.* One program owns one KV head and *all* the query heads that
share it. The dequantized K/V tile is produced once and consumed by every query
head in the group via ``tl.dot``, so the expensive part (unpacking the cache) is
paid once per group rather than once per query head.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment without triton
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc

from quantize import QuantizedTensor

LOG2E = 1.4426950408889634


def triton_available() -> tuple[bool, str]:
    if triton is None:
        return False, f"triton import failed: {_TRITON_IMPORT_ERROR!r}"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    return True, f"triton {triton.__version__}"


if triton is not None:

    @triton.jit
    def _fused_decode_attn_split(
        Q,
        KP, KS, KZ,
        VP, VS, VZ,
        ACC, MAXS, SUMS, OUT,
        sm_scale,
        S,
        stride_qb, stride_qh, stride_qd,
        stride_kpb, stride_kph, stride_kpn, stride_kpd,
        stride_ksb, stride_ksh, stride_ksn, stride_ksg,
        stride_vpb, stride_vph, stride_vpn, stride_vpd,
        stride_vsb, stride_vsh, stride_vsn, stride_vsg,
        stride_ab, stride_ah, stride_as, stride_ad,
        stride_mb, stride_mh, stride_ms,
        stride_ob, stride_oh, stride_od,
        HKV: tl.constexpr,
        GROUP: tl.constexpr,
        D: tl.constexpr,
        DP: tl.constexpr,
        NBITS: tl.constexpr,
        GS: tl.constexpr,
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
        qh = hkv * GROUP + offs_h  # absolute query-head index

        offs_d = tl.arange(0, D)
        byte_idx = offs_d % DP                 # which packed byte holds dim d
        bit_shift = (offs_d // DP) * NBITS     # which bit-slice inside that byte
        grp_idx = offs_d // GS                 # which quant group dim d is in
        code_mask = (1 << NBITS) - 1

        # (BLOCK_H, D) query tile, fp16 for tl.dot. sm_scale is applied to the
        # fp32 dot output instead of to q, to keep the fp16 operand well scaled.
        q = tl.load(
            Q + b * stride_qb + qh[:, None] * stride_qh + offs_d[None, :] * stride_qd,
            mask=h_mask[:, None],
            other=0.0,
        ).to(tl.float16)

        kp_base = KP + b * stride_kpb + hkv * stride_kph
        ks_base = KS + b * stride_ksb + hkv * stride_ksh
        kz_base = KZ + b * stride_ksb + hkv * stride_ksh
        vp_base = VP + b * stride_vpb + hkv * stride_vph
        vs_base = VS + b * stride_vsb + hkv * stride_vsh
        vz_base = VZ + b * stride_vsb + hkv * stride_vsh

        m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)

        lo = pid_s * SPLIT_SIZE
        hi = tl.minimum(lo + SPLIT_SIZE, S)
        qk_scale = sm_scale * LOG2E

        for start_n in range(lo, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < hi

            # ---- dequantize K tile: (BLOCK_N, D) -------------------------
            kp = tl.load(
                kp_base + offs_n[:, None] * stride_kpn + byte_idx[None, :] * stride_kpd,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            kcode = (kp >> bit_shift[None, :]) & code_mask
            ksc = tl.load(
                ks_base + offs_n[:, None] * stride_ksn + grp_idx[None, :] * stride_ksg,
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            kze = tl.load(
                kz_base + offs_n[:, None] * stride_ksn + grp_idx[None, :] * stride_ksg,
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            k_deq = (kcode.to(tl.float32) * ksc + kze).to(tl.float16)

            # ---- scores + online softmax ---------------------------------
            qk = tl.dot(q, tl.trans(k_deq), out_dtype=tl.float32) * qk_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            m_i = m_new

            # ---- dequantize V tile and accumulate ------------------------
            vp = tl.load(
                vp_base + offs_n[:, None] * stride_vpn + byte_idx[None, :] * stride_vpd,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
            vcode = (vp >> bit_shift[None, :]) & code_mask
            vsc = tl.load(
                vs_base + offs_n[:, None] * stride_vsn + grp_idx[None, :] * stride_vsg,
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            vze = tl.load(
                vz_base + offs_n[:, None] * stride_vsn + grp_idx[None, :] * stride_vsg,
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            v_deq = (vcode.to(tl.float32) * vsc + vze).to(tl.float16)

            acc += tl.dot(p.to(tl.float16), v_deq, out_dtype=tl.float32)

        if SINGLE_SPLIT:
            # Nothing to reduce -- normalize here and skip the combine launch.
            out = acc / l_i[:, None]
            tl.store(
                OUT + b * stride_ob + qh[:, None] * stride_oh + offs_d[None, :] * stride_od,
                out,
                mask=h_mask[:, None],
            )
        else:
            tl.store(
                ACC
                + b * stride_ab
                + qh[:, None] * stride_ah
                + pid_s * stride_as
                + offs_d[None, :] * stride_ad,
                acc,
                mask=h_mask[:, None],
            )
            m_ptr = MAXS + b * stride_mb + qh * stride_mh + pid_s * stride_ms
            l_ptr = SUMS + b * stride_mb + qh * stride_mh + pid_s * stride_ms
            tl.store(m_ptr, m_i, mask=h_mask)
            tl.store(l_ptr, l_i, mask=h_mask)

    @triton.jit
    def _combine_splits(
        ACC, MAXS, SUMS, OUT,
        NSPLIT,
        stride_ab, stride_ah, stride_as, stride_ad,
        stride_mb, stride_mh, stride_ms,
        stride_ob, stride_oh, stride_od,
        HQ: tl.constexpr,
        D: tl.constexpr,
    ):
        """Log-sum-exp reduction over the flash-decoding splits.

        Grid is (B*HQ,) and the split loop is sequential: NSPLIT is small (tens)
        and each iteration is a single D-wide vector, so a parallel tree would
        cost more in launch overhead than it saves.
        """
        pid = tl.program_id(0)
        b = pid // HQ
        h = pid % HQ

        offs_d = tl.arange(0, D)
        a_base = ACC + b * stride_ab + h * stride_ah
        m_base = MAXS + b * stride_mb + h * stride_mh
        l_base = SUMS + b * stride_mb + h * stride_mh

        acc = tl.zeros([D], dtype=tl.float32)
        m_g = float("-inf")
        l_g = 0.0

        for s in range(0, NSPLIT):
            m_s = tl.load(m_base + s * stride_ms)
            l_s = tl.load(l_base + s * stride_ms)
            m_new = tl.maximum(m_g, m_s)
            alpha = tl.math.exp2(m_g - m_new)
            beta = tl.math.exp2(m_s - m_new)
            a_s = tl.load(a_base + s * stride_as + offs_d * stride_ad)
            acc = acc * alpha + a_s * beta
            l_g = l_g * alpha + l_s * beta
            m_g = m_new

        tl.store(OUT + b * stride_ob + h * stride_oh + offs_d * stride_od, acc / l_g)


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def pick_num_splits(B: int, HKV: int, n_blocks: int, target_programs: int | None = None) -> int:
    """Choose how many history splits to use so the GPU is actually filled.

    At batch 1 there are only ``HKV`` programs' worth of natural parallelism
    (2 for Qwen2.5-1.5B), which would leave a 30-SM GPU almost entirely idle.
    Splitting the history is what makes decode attention occupy the device.
    """
    if target_programs is None:
        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            target_programs = props.multi_processor_count * 4
        except Exception:
            target_programs = 128
    want = max(1, target_programs // max(1, B * HKV))
    return max(1, min(n_blocks, want))


def fused_decode_attention(
    q: torch.Tensor,
    kq: QuantizedTensor,
    vq: QuantizedTensor,
    sm_scale: float | None = None,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    num_splits: int | None = None,
    out: torch.Tensor | None = None,
    _workspace: dict | None = None,
) -> torch.Tensor:
    """Single decode step of attention, computed directly on packed low-bit K/V.

    Parameters
    ----------
    q:
        ``(B, HQ, D)``. Cast to fp16 internally for the tensor-core path.
    kq, vq:
        Quantized caches with ``packed`` of shape ``(B, HKV, S, D//P)``.
    out:
        Optional preallocated ``(B, HQ, D)`` float32 output.
    _workspace:
        Optional dict reused across calls to hold the split partials. Passing
        one removes the per-call allocation, which matters when timing.

    Returns ``(B, HQ, D)`` float32.
    """
    ok, why = triton_available()
    if not ok:
        raise RuntimeError(f"fused_decode_attention requires Triton on CUDA: {why}")

    if kq.nbits != vq.nbits or kq.group_size != vq.group_size:
        raise ValueError("K and V must use the same nbits/group_size")

    B, HQ, D = q.shape
    Bk, HKV, S, DP = kq.packed.shape
    if (B, D) != (Bk, kq.head_dim):
        raise ValueError(f"q shape {tuple(q.shape)} incompatible with cache {tuple(kq.packed.shape)}")
    if vq.packed.shape != kq.packed.shape:
        raise ValueError("K and V caches must have the same shape")
    if HQ % HKV != 0:
        raise ValueError(f"HQ={HQ} must be a multiple of HKV={HKV}")
    if D & (D - 1):
        raise ValueError(f"head_dim must be a power of two, got {D}")

    group = HQ // HKV
    sm_scale = 1.0 / math.sqrt(D) if sm_scale is None else sm_scale
    block_n = min(block_n, max(16, _next_pow2(S)))

    q = q.contiguous()
    kp, ks, kz = kq.packed.contiguous(), kq.scale.contiguous(), kq.zero.contiguous()
    vp, vs, vz = vq.packed.contiguous(), vq.scale.contiguous(), vq.zero.contiguous()

    n_blocks = triton.cdiv(S, block_n)
    if num_splits is None:
        num_splits = pick_num_splits(B, HKV, n_blocks)
    num_splits = max(1, min(num_splits, n_blocks))
    blocks_per_split = triton.cdiv(n_blocks, num_splits)
    num_splits = triton.cdiv(n_blocks, blocks_per_split)  # tighten: no empty splits
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
        cached = ws.get("key")
        if cached == key:
            acc, maxs, sums = ws["acc"], ws["maxs"], ws["sums"]
        else:
            acc = torch.empty((B, HQ, num_splits, D), device=q.device, dtype=torch.float32)
            maxs = torch.empty((B, HQ, num_splits), device=q.device, dtype=torch.float32)
            sums = torch.empty((B, HQ, num_splits), device=q.device, dtype=torch.float32)
            ws.update(key=key, acc=acc, maxs=maxs, sums=sums)

    block_h = max(16, _next_pow2(group))

    _fused_decode_attn_split[(B * HKV, num_splits)](
        q,
        kp, ks, kz,
        vp, vs, vz,
        acc, maxs, sums, out,
        sm_scale,
        S,
        q.stride(0), q.stride(1), q.stride(2),
        kp.stride(0), kp.stride(1), kp.stride(2), kp.stride(3),
        ks.stride(0), ks.stride(1), ks.stride(2), ks.stride(3),
        vp.stride(0), vp.stride(1), vp.stride(2), vp.stride(3),
        vs.stride(0), vs.stride(1), vs.stride(2), vs.stride(3),
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        maxs.stride(0), maxs.stride(1), maxs.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        HKV=HKV,
        GROUP=group,
        D=D,
        DP=DP,
        NBITS=kq.nbits,
        GS=kq.group_size,
        BLOCK_H=block_h,
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
