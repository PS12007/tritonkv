"""Numerical correctness tests.

Run with::

    python -m pytest test_correctness.py -v

The quantization tests run on CPU in about a second. The kernel tests need CUDA
and Triton and take a few seconds; they are skipped (not failed) otherwise.

The thresholds below are stated explicitly and asserted -- nothing here reports
"looks close". Two comparisons are made, and the distinction matters:

* **Kernel error** -- fused kernel vs. dequantize-then-attend *in float32* on the
  exact same dequantized numbers. Both see identical inputs, so any difference
  is the kernel's own arithmetic (fp16 tensor-core dot, online softmax
  rescaling, split reduction). This must be tiny.
* **End-to-end error** -- fused kernel vs. attention on the *unquantized* fp16
  cache. This is dominated by quantization, not by the kernel, and the test
  asserts exactly that by requiring the kernel's error to be no worse than the
  PyTorch dequant baseline's error by more than a small factor.
"""

from __future__ import annotations

import math

import pytest
import torch

from quantize import (
    SUPPORTED_BITS,
    dequantize_groupwise,
    pack_codes,
    quantize_groupwise,
    quantize_kv,
    unpack_codes,
)
from reference import (
    baseline_dequant_sdpa,
    default_sm_scale,
    make_random_kv,
    reference_decode_attention,
)
from kernels.fused_decode_attn import fused_decode_attention, triton_available

# ---------------------------------------------------------------------------
# Thresholds (single source of truth; the README quotes these)
# ---------------------------------------------------------------------------

# Fused kernel vs. float32 dequant-then-attend on identical dequantized values.
KERNEL_COSINE_MIN = 0.99999
KERNEL_REL_L2_MAX = 5e-3
# How much worse than the PyTorch fp16 dequant baseline the kernel is allowed to
# be, measured against the unquantized fp16 cache. 1.0 would mean "exactly as
# accurate"; 1.5 leaves room for the different softmax order.
KERNEL_VS_BASELINE_ERR_RATIO_MAX = 1.5

CUDA_OK, CUDA_WHY = triton_available()
requires_triton = pytest.mark.skipif(not CUDA_OK, reason=f"no Triton/CUDA: {CUDA_WHY}")


def _metrics(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    g = got.float().flatten()
    r = ref.float().flatten()
    err = g - r
    return {
        "cosine": torch.nn.functional.cosine_similarity(g[None], r[None]).item(),
        "max_abs_err": err.abs().max().item(),
        "rel_l2": (err.norm() / r.norm().clamp_min(1e-12)).item(),
    }


# ---------------------------------------------------------------------------
# Quantization reference (CPU, fast)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nbits", SUPPORTED_BITS)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_pack_unpack_roundtrip(nbits: int, head_dim: int):
    torch.manual_seed(0)
    qmax = (1 << nbits) - 1
    codes = torch.randint(0, qmax + 1, (3, 5, head_dim), dtype=torch.uint8)
    packed = pack_codes(codes, nbits)
    assert packed.shape == (3, 5, head_dim // (8 // nbits))
    back = unpack_codes(packed, nbits, head_dim)
    assert torch.equal(back, codes)


@pytest.mark.parametrize("nbits", SUPPORTED_BITS)
@pytest.mark.parametrize("group_size", [16, 32, 64])
def test_quant_error_within_half_step(nbits: int, group_size: int):
    """Reconstruction error must never exceed half a quantization step.

    This is the tightest property the scheme can be held to, and it catches
    off-by-one packing bugs that a loose cosine check would sail past.
    """
    torch.manual_seed(1)
    x = torch.randn(4, 7, 128)
    qt = quantize_groupwise(x, nbits, group_size)
    xh = dequantize_groupwise(qt)

    step = qt.scale.float().repeat_interleave(group_size, dim=-1)
    err = (xh - x).abs()
    # +tiny slack for the fp16 rounding of scale/zero themselves.
    assert torch.all(err <= step / 2 + 1e-2 * step + 1e-6), (
        f"max err {err.max().item()} vs half-step {(step / 2).max().item()}"
    )


@pytest.mark.parametrize("nbits", SUPPORTED_BITS)
def test_constant_group_is_exact(nbits: int):
    """A constant group has zero range; it must round-trip exactly, not NaN."""
    x = torch.full((2, 64), 3.5)
    qt = quantize_groupwise(x, nbits, 32)
    xh = dequantize_groupwise(qt)
    assert torch.allclose(xh, x, atol=1e-3)
    assert torch.isfinite(xh).all()


def test_higher_bits_are_more_accurate():
    torch.manual_seed(2)
    x = torch.randn(8, 256)
    e2 = (dequantize_groupwise(quantize_groupwise(x, 2, 32)) - x).norm().item()
    e4 = (dequantize_groupwise(quantize_groupwise(x, 4, 32)) - x).norm().item()
    assert e4 < e2, f"4-bit error {e4} should beat 2-bit error {e2}"


def test_effective_bits_accounts_for_metadata():
    x = torch.randn(4, 128)
    qt = quantize_groupwise(x, 4, 32)
    # 4 bits of code + 32 bits of fp16 scale/zero shared over 32 elements.
    assert math.isclose(qt.effective_bits_per_element(), 5.0)
    assert qt.nbytes() * 8 / x.numel() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

SHAPES = [
    # (B, HQ, HKV, D)  -- Qwen2.5-1.5B attention shape, plus GQA edge cases
    (1, 12, 2, 128),
    (1, 8, 8, 64),  # no GQA sharing (MHA)
    (2, 32, 8, 64),  # Llama-3.2-1B shape, batch 2
]
SEQ_LENS = [1, 17, 128, 512, 2048, 8192, 16384]


def _run_case(B, HQ, HKV, D, S, nbits, group_size, seed, block_n=64, num_splits=None):
    q, k, v = make_random_kv(B, HQ, HKV, S, D, device="cuda", seed=seed)
    kq, vq = quantize_kv(k, v, nbits, group_size)
    k_deq = dequantize_groupwise(kq, torch.float32)
    v_deq = dequantize_groupwise(vq, torch.float32)

    got = fused_decode_attention(q, kq, vq, block_n=block_n, num_splits=num_splits)
    ref_on_deq = reference_decode_attention(q, k_deq, v_deq)  # kernel-only error
    ref_true = reference_decode_attention(q, k, v)  # quantization included
    base = baseline_dequant_sdpa(q, kq, vq)
    return got, ref_on_deq, ref_true, base


@requires_triton
@pytest.mark.parametrize("S", SEQ_LENS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kernel_matches_dequant_reference_4bit(S: int, seed: int):
    B, HQ, HKV, D = SHAPES[0]
    got, ref_on_deq, _, _ = _run_case(B, HQ, HKV, D, S, 4, 32, seed)
    m = _metrics(got, ref_on_deq)
    assert m["cosine"] >= KERNEL_COSINE_MIN, f"S={S} seed={seed}: {m}"
    assert m["rel_l2"] <= KERNEL_REL_L2_MAX, f"S={S} seed={seed}: {m}"


@requires_triton
@pytest.mark.parametrize("S", [512, 2048, 8192, 16384])
@pytest.mark.parametrize("seed", [0, 1])
def test_kernel_matches_dequant_reference_2bit(S: int, seed: int):
    B, HQ, HKV, D = SHAPES[0]
    got, ref_on_deq, _, _ = _run_case(B, HQ, HKV, D, S, 2, 32, seed)
    m = _metrics(got, ref_on_deq)
    assert m["cosine"] >= KERNEL_COSINE_MIN, f"S={S} seed={seed}: {m}"
    assert m["rel_l2"] <= KERNEL_REL_L2_MAX, f"S={S} seed={seed}: {m}"


@requires_triton
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("nbits", SUPPORTED_BITS)
def test_kernel_shapes_and_gqa(shape, nbits: int):
    B, HQ, HKV, D = shape
    got, ref_on_deq, _, _ = _run_case(B, HQ, HKV, D, 1024, nbits, 32, seed=3)
    m = _metrics(got, ref_on_deq)
    assert m["cosine"] >= KERNEL_COSINE_MIN, f"{shape} {nbits}b: {m}"
    assert m["rel_l2"] <= KERNEL_REL_L2_MAX, f"{shape} {nbits}b: {m}"


@requires_triton
@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
def test_kernel_group_sizes(group_size: int):
    got, ref_on_deq, _, _ = _run_case(1, 12, 2, 128, 2048, 4, group_size, seed=4)
    m = _metrics(got, ref_on_deq)
    assert m["cosine"] >= KERNEL_COSINE_MIN, f"G={group_size}: {m}"


@requires_triton
def test_kernel_is_not_the_source_of_error():
    """The kernel must not be measurably less accurate than the naive baseline.

    Both are compared against the unquantized fp16 cache. If the fused kernel's
    error were meaningfully larger, the speedup would be bought with accuracy --
    which is the failure mode this project is trying not to have.
    """
    for S in (512, 2048, 8192, 16384):
        got, _, ref_true, base = _run_case(1, 12, 2, 128, S, 4, 32, seed=5)
        e_kernel = _metrics(got, ref_true)["rel_l2"]
        e_base = _metrics(base, ref_true)["rel_l2"]
        assert e_kernel <= KERNEL_VS_BASELINE_ERR_RATIO_MAX * e_base, (
            f"S={S}: kernel rel_l2 {e_kernel:.3e} vs baseline {e_base:.3e}"
        )


@requires_triton
@pytest.mark.parametrize("num_splits", [1, 2, 5, 17, 64])
def test_split_count_does_not_change_the_answer(num_splits: int):
    """Flash-decoding partitioning must be invisible in the output.

    An uneven split count (5, 17) is included on purpose: those exercise the
    ragged last split and the log-sum-exp rescaling that a power-of-two-only
    test would never touch.
    """
    ref = None
    got, ref_on_deq, _, _ = _run_case(1, 12, 2, 128, 4096, 4, 32, seed=6, num_splits=num_splits)
    m = _metrics(got, ref_on_deq)
    assert m["cosine"] >= KERNEL_COSINE_MIN, f"splits={num_splits}: {m}"
    assert m["rel_l2"] <= KERNEL_REL_L2_MAX, f"splits={num_splits}: {m}"


@requires_triton
@pytest.mark.parametrize("block_n", [16, 32, 64, 128])
def test_block_sizes(block_n: int):
    got, ref_on_deq, _, _ = _run_case(1, 12, 2, 128, 3000, 4, 32, seed=7, block_n=block_n)
    m = _metrics(got, ref_on_deq)
    assert m["cosine"] >= KERNEL_COSINE_MIN, f"BLOCK_N={block_n}: {m}"


@requires_triton
def test_output_is_finite_with_extreme_scores():
    """Large q magnitudes push the softmax to a near one-hot distribution.

    Online softmax with a bad rescaling produces NaN or inf here rather than a
    slightly-wrong number, so this is a cheap guard against a whole bug class.
    """
    B, HQ, HKV, D, S = 1, 12, 2, 128, 2048
    q, k, v = make_random_kv(B, HQ, HKV, S, D, device="cuda", seed=8)
    q = q * 50.0
    kq, vq = quantize_kv(k, v, 4, 32)
    got = fused_decode_attention(q, kq, vq)
    assert torch.isfinite(got).all()
    ref = reference_decode_attention(
        q, dequantize_groupwise(kq, torch.float32), dequantize_groupwise(vq, torch.float32)
    )
    assert _metrics(got, ref)["cosine"] >= KERNEL_COSINE_MIN


@requires_triton
def test_single_token_cache():
    """S=1: attention output must be exactly the single dequantized V row."""
    B, HQ, HKV, D = 1, 12, 2, 128
    q, k, v = make_random_kv(B, HQ, HKV, 1, D, device="cuda", seed=9)
    kq, vq = quantize_kv(k, v, 4, 32)
    got = fused_decode_attention(q, kq, vq)
    v_deq = dequantize_groupwise(vq, torch.float32)  # (1, HKV, 1, D)
    expect = v_deq[:, :, 0, :].repeat_interleave(HQ // HKV, dim=1)
    assert _metrics(got, expect)["rel_l2"] < 1e-3


@requires_triton
def test_per_element_worst_case_not_hidden_by_aggregate():
    """Aggregate cosine similarity can hide a handful of badly wrong elements.

    The audit calls this out explicitly, so it is asserted here too: check the
    single worst output element, not just the norm over all of them.
    """
    got, ref_on_deq, _, _ = _run_case(1, 12, 2, 128, 8192, 4, 32, seed=10)
    err = (got - ref_on_deq).abs()
    scale = ref_on_deq.abs().mean().clamp_min(1e-6)
    worst_relative = (err.max() / scale).item()
    assert worst_relative < 0.05, (
        f"worst element is {worst_relative:.3f}x the mean |output| off"
    )


@requires_triton
@pytest.mark.parametrize("S", [1, 17, 512, 2048, 8192])
@pytest.mark.parametrize("nbits", SUPPORTED_BITS)
@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
def test_metadata_broadcast_is_bitwise_identical(S: int, nbits: int, group_size: int):
    """The two metadata-load paths must agree to the last bit.

    ``meta_bcast=True`` loads the ``(BLOCK_N, n_groups)`` scale/zero tile that
    actually exists in memory and expands it in registers; ``False`` gathers a
    full ``(BLOCK_N, head_dim)`` tile with ``d // group_size``. They read the
    same fp16 values and feed them into the same arithmetic, so the outputs are
    not merely close, they are equal -- and asserting equality rather than a
    tolerance is what keeps the slow path from silently drifting away from the
    fast one now that only the fast one is on by default.
    """
    q, k, v = make_random_kv(1, 12, 2, S, 128, device="cuda", seed=11)
    kq, vq = quantize_kv(k, v, nbits, group_size)
    gathered = fused_decode_attention(q, kq, vq, meta_bcast=False)
    broadcast = fused_decode_attention(q, kq, vq, meta_bcast=True)
    assert torch.equal(gathered, broadcast), (
        f"S={S} nbits={nbits} gs={group_size}: max|diff| = "
        f"{(gathered - broadcast).abs().max().item():.3e}"
    )


@requires_triton
@pytest.mark.parametrize("S", [1, 17, 512, 2048, 8192])
@pytest.mark.parametrize("nbits", SUPPORTED_BITS)
@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
def test_fp16_dequant_is_bitwise_identical(S: int, nbits: int, group_size: int):
    """Doing the dequantization multiply-add at fp16 width must change nothing.

    The shipped path widens the per-group scale and zero to fp32, reconstructs
    ``code * scale + zero`` there, and narrows the result back to fp16 for
    ``tl.dot``. ``dequant_fp16=True`` does the same arithmetic without ever
    leaving fp16, which removes 80 conversion instructions per kernel.

    It is exactly equal rather than merely close, and that is not a coincidence:
    the code is a 2- or 4-bit integer and the scale and zero are fp16 values read
    straight from memory, so the product and sum are representable at fp16 and
    the wider intermediate had nothing to add. Asserting equality is what would
    catch that stopping being true -- at a larger bit width, or if the metadata
    ever became fp32 in memory.
    """
    q, k, v = make_random_kv(1, 12, 2, S, 128, device="cuda", seed=13)
    kq, vq = quantize_kv(k, v, nbits, group_size)
    wide = fused_decode_attention(q, kq, vq, dequant_fp16=False)
    narrow = fused_decode_attention(q, kq, vq, dequant_fp16=True)
    assert torch.equal(wide, narrow), (
        f"S={S} nbits={nbits} gs={group_size}: max|diff| = "
        f"{(wide - narrow).abs().max().item():.3e}"
    )
