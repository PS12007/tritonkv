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
    "sdpa_backend_report",
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


# --- Fair fp16 attention for a decode step ---------------------------------
#
# Torch's SDPA dispatcher does *not* pick the fastest kernel for this shape, and
# the difference is large enough to invent a speedup out of nothing. Measured
# here at B=1, HQ=12, HKV=2, S=8192, D=128 (CUDA-graph replay, median of 15):
#
#     enable_gqa=True, dispatcher's choice      986 us     8.5 GB/s
#     expanded KV,     dispatcher's choice      510 us    16.5 GB/s
#     expanded KV,     mem-efficient backend    511 us    16.4 GB/s
#     expanded KV,     cuDNN backend            154 us    54.4 GB/s
#     expanded KV,     math backend            1061 us     7.9 GB/s
#     hand-written fp16 bmm + softmax + bmm     152 us    55.1 GB/s
#
# FlashAttention is not a candidate at all on this machine: the Windows torch
# build reports "Torch was not compiled with flash attention".
#
# Quoting a speedup against the 986 us path would be quoting a 6.4x measurement
# artefact. So the baseline is *probed*: every candidate that runs for the given
# shape is timed once, the fastest is cached, and that is what the Triton kernel
# has to beat. Being generous to the baseline is the whole point.

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _SDPA_BACKENDS = [
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ]
except Exception:  # pragma: no cover - older torch
    SDPBackend = None
    sdpa_kernel = None
    _SDPA_BACKENDS = []

# shape key -> winning strategy *name*. Probed once per shape, then reused.
_SDPA_CHOICE: dict = {}


def _manual_fp16_attention(
    q4: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    """bmm -> softmax(fp32) -> bmm. Backend-independent, and on this GPU it ties
    with the best SDPA backend, so it keeps the baseline honest on machines
    where every fused backend happens to be unavailable."""
    scores = torch.matmul(q4, k.transpose(-1, -2)) * sm_scale  # (B, H, 1, S)
    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    return torch.matmul(probs, v)


def _run_strategy(name: str, q4, k, v, sm_scale: float, n_rep: int):
    """Execute one named fp16 attention strategy on the tensors given *now*.

    The probe caches a strategy *name*, never a closure: a cached closure would
    pin the probe's own tensors and silently return the first call's answer for
    every later call -- which also makes any rotating-cache benchmark measure a
    permanently hot cache.
    """
    if name == "sdpa_gqa_default":
        return F.scaled_dot_product_attention(q4, k, v, scale=sm_scale, enable_gqa=True)

    ke, ve = expand_kv(k, n_rep), expand_kv(v, n_rep)
    if name == "sdpa_expanded_default":
        return F.scaled_dot_product_attention(q4, ke, ve, scale=sm_scale)
    if name == "manual_fp16_bmm":
        return _manual_fp16_attention(q4, ke, ve, sm_scale)

    backend = dict(_SDPA_BACKENDS)[name[len("sdpa_expanded_"):]]
    with sdpa_kernel([backend]):
        return F.scaled_dot_product_attention(q4, ke, ve, scale=sm_scale)


def _candidate_names(q4, k, v, sm_scale: float, n_rep: int) -> list[str]:
    """Every strategy that actually runs for this shape."""
    names = []
    if n_rep > 1:
        names.append("sdpa_gqa_default")
    names.append("sdpa_expanded_default")
    names += [f"sdpa_expanded_{n}" for n, _ in _SDPA_BACKENDS]
    names.append("manual_fp16_bmm")

    ok = []
    for n in names:
        try:
            _run_strategy(n, q4, k, v, sm_scale, n_rep)
            ok.append(n)
        except Exception:
            continue  # backend genuinely unavailable for this shape
    return ok


def _time_thunk(fn, iters: int = 30) -> float:
    """Rank candidates on GPU time, via CUDA-graph replay where possible.

    Plain wall-clock ranking is unusable on Windows: WDDM submission costs
    50-300 us per call, which swamps a 0.5 MB read and had the probe picking a
    3.3 GB/s path over a 23.8 GB/s one at short context. Replaying a captured
    graph removes launch cost from the comparison, so the probe ranks kernels
    on the same basis the benchmark reports them.
    """
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
        torch.cuda.synchronize()
        for _ in range(2):
            g.replay()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(5):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e) / iters)
        del g
        return best
    except Exception:
        pass  # not capturable: fall back to wall clock

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def sdpa_backend_report() -> dict:
    """What was probed and what won, for the benchmark JSON and the README."""
    return {str(k): v for k, v in _SDPA_CHOICE.items()}


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """Fastest available fp16 decode attention. q: (B,HQ,D); k,v: (B,HKV,S,D)."""
    B, HQ, D = q.shape
    HKV = k.shape[1]
    n_rep = HQ // HKV
    q4 = q.unsqueeze(2)  # (B, HQ, 1, D)

    if not q.is_cuda or torch.cuda.is_current_stream_capturing():
        # Never probe mid-capture (it synchronizes). Fall back to the plain path.
        ke, ve = expand_kv(k, n_rep), expand_kv(v, n_rep)
        return F.scaled_dot_product_attention(q4, ke, ve, scale=sm_scale).squeeze(2)

    key = (tuple(q.shape), tuple(k.shape), str(q.dtype), q.device.index)
    name = _SDPA_CHOICE.get(key)
    if name is None:
        names = _candidate_names(q4, k, v, sm_scale, n_rep)
        timed = sorted(
            (_time_thunk(lambda n=n: _run_strategy(n, q4, k, v, sm_scale, n_rep)), n)
            for n in names
        )
        name = timed[0][1]
        _SDPA_CHOICE[key] = name

    out = _run_strategy(name, q4, k, v, sm_scale, n_rep)
    return out.squeeze(2) if out.dim() == 4 else out


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
