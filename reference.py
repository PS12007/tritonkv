"""PyTorch reference / baseline decode-attention implementations.

Everything here is plain PyTorch. These serve two purposes:

1. ``reference_decode_attention`` is the fp32 **ground truth** the Triton kernel
   is checked against in ``test_correctness.py``.
2. The ``baseline_*`` functions are the things the Triton kernel is *benchmarked*
   against. They are written to be fair, not to be strawmen -- in particular
   ``baseline_dequant_sdpa_compiled`` lets ``torch.compile`` fuse the whole
   unpack+dequant chain into a single Inductor kernel, which is the strongest
   pure-PyTorch baseline we know how to write.

Shapes used throughout (single decode step, so the query length is 1):

    q            : (B, HQ, D)          float16/float32
    K, V caches  : (B, HKV, S, D)      float16 (or quantized equivalents)
    output       : (B, HQ, D)          float32

``HQ`` may be a multiple of ``HKV`` (grouped-query attention).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from quantize import QuantizedTensor, unpack_codes

__all__ = [
    "default_sm_scale",
    "expand_kv",
    "reference_decode_attention",
    "dequantize_to",
    "baseline_fp16_sdpa",
    "baseline_dequant_sdpa",
    "baseline_dequant_sdpa_compiled",
    "make_random_kv",
]


def default_sm_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)


def expand_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, HKV, S, D) -> (B, HKV*n_rep, S, D), repeating each KV head n_rep times."""
    if n_rep == 1:
        return x
    B, HKV, S, D = x.shape
    return x[:, :, None, :, :].expand(B, HKV, n_rep, S, D).reshape(B, HKV * n_rep, S, D)


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


def reference_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Naive fp32 single-step attention. This is the ground truth.

    Deliberately written the boring way -- explicit matmul, explicit softmax in
    float32 -- so there is no fused-kernel cleverness to be wrong about.

    q: (B, HQ, D); k, v: (B, HKV, S, D). Returns (B, HQ, D) float32.
    """
    B, HQ, D = q.shape
    _, HKV, S, _ = k.shape
    assert HQ % HKV == 0, f"HQ={HQ} must be a multiple of HKV={HKV}"
    n_rep = HQ // HKV
    sm_scale = default_sm_scale(D) if sm_scale is None else sm_scale

    qf = q.float()
    kf = expand_kv(k.float(), n_rep)  # (B, HQ, S, D)
    vf = expand_kv(v.float(), n_rep)

    # (B, HQ, S)
    scores = torch.einsum("bhd,bhsd->bhs", qf, kf) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,bhsd->bhd", probs, vf)
    return out


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def dequantize_to(qt: QuantizedTensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Dequantize straight into ``dtype`` (no float32 intermediate).

    Used by the baselines. Working in fp16 end-to-end roughly halves the traffic
    of the float32 path in ``quantize.dequantize_groupwise``, so this is the
    fair version of "dequantize the whole cache".
    """
    codes = unpack_codes(qt.packed, qt.nbits, qt.head_dim)
    cg = codes.reshape(*codes.shape[:-1], qt.n_groups, qt.group_size).to(dtype)
    scale = qt.scale.to(dtype).unsqueeze(-1)
    zero = qt.zero.to(dtype).unsqueeze(-1)
    out = cg * scale + zero
    return out.reshape(*codes.shape[:-1], qt.head_dim)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """SDPA for a single decode step. q: (B,HQ,D); k,v: (B,HKV,S,D) -> (B,HQ,D)."""
    B, HQ, D = q.shape
    HKV = k.shape[1]
    n_rep = HQ // HKV
    q4 = q.unsqueeze(2)  # (B, HQ, 1, D)
    if n_rep > 1:
        # enable_gqa avoids materializing the expanded K/V where supported.
        try:
            o = F.scaled_dot_product_attention(q4, k, v, scale=sm_scale, enable_gqa=True)
        except TypeError:
            o = F.scaled_dot_product_attention(
                q4, expand_kv(k, n_rep), expand_kv(v, n_rep), scale=sm_scale
            )
    else:
        o = F.scaled_dot_product_attention(q4, k, v, scale=sm_scale)
    return o.squeeze(2)


def baseline_fp16_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float | None = None
) -> torch.Tensor:
    """Unquantized fp16 SDPA. Speed upper bound / memory lower bound."""
    sm_scale = default_sm_scale(q.shape[-1]) if sm_scale is None else sm_scale
    return _sdpa(q, k, v, sm_scale)


def baseline_dequant_sdpa(
    q: torch.Tensor,
    kq: QuantizedTensor,
    vq: QuantizedTensor,
    sm_scale: float | None = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """THE baseline this project attacks: dequantize the whole cache, then attend.

    This is what a naive quantized-KV-cache implementation does on *every*
    decode step: it reconstitutes S x D full-precision K and V, throws them at
    SDPA, and then throws them away.
    """
    sm_scale = default_sm_scale(q.shape[-1]) if sm_scale is None else sm_scale
    k = dequantize_to(kq, dtype)
    v = dequantize_to(vq, dtype)
    return _sdpa(q.to(dtype), k, v, sm_scale)


_compiled_dequant = None


def _get_compiled_dequant():
    """Lazily build a torch.compile'd unpack+dequant.

    Inductor fuses the shift/mask/convert/scale chain into one kernel, which
    removes the intermediate uint8 and float tensors that the eager path writes
    out. This is a *much* stronger baseline than eager, and if the fused Triton
    kernel cannot beat it then the project's headline claim is not real.
    """
    global _compiled_dequant

    if _compiled_dequant is None:

        def _dq(packed, scale, zero, nbits: int, group_size: int, head_dim: int):
            P = 8 // nbits
            mask = (1 << nbits) - 1
            parts = []
            for p in range(P):
                parts.append((packed >> (p * nbits)) & mask)
            codes = torch.cat(parts, dim=-1)  # split-P -> natural order
            cg = codes.reshape(*codes.shape[:-1], head_dim // group_size, group_size).half()
            return (cg * scale.unsqueeze(-1) + zero.unsqueeze(-1)).reshape(
                *codes.shape[:-1], head_dim
            )

        try:
            _compiled_dequant = torch.compile(_dq, dynamic=False)
        except Exception:  # pragma: no cover - torch.compile unavailable
            _compiled_dequant = _dq

    return _compiled_dequant


def baseline_dequant_sdpa_compiled(
    q: torch.Tensor,
    kq: QuantizedTensor,
    vq: QuantizedTensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Same as :func:`baseline_dequant_sdpa` but with a compiled dequant."""
    sm_scale = default_sm_scale(q.shape[-1]) if sm_scale is None else sm_scale
    dq = _get_compiled_dequant()
    k = dq(kq.packed, kq.scale, kq.zero, kq.nbits, kq.group_size, kq.head_dim)
    v = dq(vq.packed, vq.scale, vq.zero, vq.nbits, vq.group_size, vq.head_dim)
    return _sdpa(q.half(), k, v, sm_scale)


# --------------------------------------------------------------------------
# Test data
# --------------------------------------------------------------------------


def make_random_kv(
    B: int,
    HQ: int,
    HKV: int,
    S: int,
    D: int,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
    outlier_frac: float = 0.01,
    outlier_scale: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random q/K/V that look roughly like real attention tensors.

    Pure Gaussian noise is an unrealistically *easy* input for a quantizer:
    real KV caches have heavy-tailed per-channel outliers, which is precisely
    what makes low-bit KV quantization hard. We inject a small fraction of
    large-magnitude entries so the correctness numbers are not flattered.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    def _rand(*shape):
        x = torch.randn(*shape, generator=g, dtype=torch.float32)
        if outlier_frac > 0:
            m = torch.rand(*shape, generator=g) < outlier_frac
            x = torch.where(m, x * outlier_scale, x)
        return x.to(device=device, dtype=dtype)

    q = _rand(B, HQ, D)
    k = _rand(B, HKV, S, D)
    v = _rand(B, HKV, S, D)
    return q, k, v
