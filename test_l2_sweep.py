"""CPU-only tests for l2_sweep.py's grid, zones and scoring.

    ./.venv/Scripts/python.exe -m pytest test_l2_sweep.py -q

The analysis scores pre-registered predictions, so the thing worth testing is
that it can *fail* them: a synthetic sweep built the way the L2 story says must
pass, and sweeps built the way the competing stories say must not.
"""
from __future__ import annotations

import math

import pytest

import l2_sweep as S

L2 = 33_554_432          # this repo's card
BPT = 1024               # fp16 bytes per token, HKV=2, D=128


def raw(median_ms: float, n: int = 30) -> list[float]:
    # Deterministic, tight, symmetric jitter: the median is the median.
    return [median_ms * (1 + 0.004 * ((i % 5) - 2)) for i in range(n)]


def point(ctx: int, ctrl_hot: float, fused_hot: float, ctrl_cold: float,
          fused_cold: float, quotable: bool = True) -> dict:
    fp16 = BPT * ctx
    q4 = int(fp16 / 3.2)

    def reg(ms):
        return {"raw_ms": raw(ms), "median_ms": ms, "quotable": quotable}

    return {
        "ctx": ctx, "fp16_bytes": fp16, "q4_bytes": q4,
        "x_fp16": fp16 / L2, "x_q4": q4 / L2,
        "correctness": {"control_cos_vs_ref": 0.9999999, "fused_cos_vs_dequant_ref": 0.9999998},
        "methods": {S.CONTROL: {"hot": reg(ctrl_hot), "cold": reg(ctrl_cold)},
                    S.FUSED: {"hot": reg(fused_hot), "cold": reg(fused_cold)}},
    }


def payload(points: list[dict]) -> dict:
    return {"env": {"gpu": "synthetic"}, "l2_bytes": L2,
            "model": {"num_kv_heads": 2, "head_dim": 128}, "points": points}


def l2_story(ctx: int) -> dict:
    """Timings from a toy model in which only L2 residency matters.

    Each kernel moves its bytes at an L2 rate while its cache fits and a DRAM
    rate once it does not. The fused kernel also pays a dequantization cost per
    element, independent of where the bytes come from -- which is what makes it
    lose when everything is in L2 and win once the fp16 bytes have to come from
    DRAM. Shaped to match the measured behaviour, not fitted to it.
    """
    fp16 = BPT * ctx
    q4 = fp16 / 3.2
    l2_bw, dram_bw = 1000.0, 300.0        # bytes per us, arbitrary
    dequant = 1.0 / 1000.0                 # us per fp16-equivalent byte

    def move(nbytes, resident):
        return nbytes / (l2_bw if resident else dram_bw)

    ctrl_hot = move(fp16, fp16 <= L2) / 1e3
    fused_hot = (fp16 * dequant + move(q4, q4 <= L2)) / 1e3
    ctrl_cold = move(fp16, False) / 1e3
    fused_cold = (fp16 * dequant + move(q4, False)) / 1e3
    return point(ctx, ctrl_hot, fused_hot, ctrl_cold, fused_cold)


# --- the memory-frugal reference -----------------------------------------------

@pytest.mark.parametrize("seq,chunk", [(333, 100), (256, 4096), (97, 1)])
def test_grouped_reference_matches_the_original_fp16(seq, chunk):
    from reference import make_random_kv, reference_decode_attention
    q, k, v = make_random_kv(1, 12, 2, seq, 128, device="cpu", seed=3)
    want = reference_decode_attention(q, k, v)
    got = S.reference_grouped(q, k, v, chunk=chunk)
    assert (got - want).abs().max().item() < 1e-5 * want.abs().max().item()


@pytest.mark.parametrize("nbits", [4, 2])
def test_grouped_reference_matches_the_original_quantized(nbits):
    import torch

    from quantize import dequantize_groupwise, quantize_kv
    from reference import make_random_kv, reference_decode_attention
    q, k, v = make_random_kv(1, 12, 2, 300, 128, device="cpu", seed=4)
    kq, vq = quantize_kv(k, v, nbits, 32)
    want = reference_decode_attention(q, dequantize_groupwise(kq, torch.float32),
                                      dequantize_groupwise(vq, torch.float32))
    got = S.reference_grouped(q, kq=kq, vq=vq, chunk=77)
    assert (got - want).abs().max().item() < 1e-5 * want.abs().max().item()


# --- grid --------------------------------------------------------------------

def test_grid_on_this_card_hits_the_designed_contexts():
    ctxs = [c for _, c in S.contexts_for(L2, BPT)]
    assert ctxs[:5] == [4096, 8192, 16384, 24576, 32768]
    assert ctxs[-1] == 393216
    assert len(ctxs) == len(S.X_GRID)


def test_grid_dedupes_on_a_small_l2_card():
    grid = S.contexts_for(1_000_000, BPT)          # 1 MB L2: low points collapse
    ctxs = [c for _, c in grid]
    assert len(ctxs) == len(set(ctxs))
    assert ctxs[0] == 256
    assert all(c % 128 == 0 for c in ctxs)


def test_grid_scales_with_l2():
    """The whole point: ctx at a given x is proportional to L2."""
    small = dict(S.contexts_for(6 * 2**20, BPT))
    big = dict(S.contexts_for(72 * 2**20, BPT))
    assert big[1.0] / small[1.0] == pytest.approx(12, rel=0.02)


# --- zones -------------------------------------------------------------------

@pytest.mark.parametrize("x16,expected", [
    (0.5, "A"), (0.75, "A"), (1.0, "-"), (1.25, "-"), (1.5, "B"), (2.0, "B"),
    (3.0, "-"), (4.0, "-"), (4.9, "C"), (12.0, "C"),
])
def test_zone_boundaries(x16, expected):
    assert S.zone(x16, x16 / 3.2) == expected


# --- crossing ----------------------------------------------------------------

def test_crossing_interpolates_in_log_x():
    x = S.crossing([0.5, 2.0], [0.5, 1.5])
    assert x == pytest.approx(1.0)                # halfway in log space


def test_crossing_below_grid_is_minus_inf_not_a_pass():
    assert S.crossing([0.125, 1.0], [1.2, 1.5]) == float("-inf")


def test_crossing_absent_is_none():
    assert S.crossing([0.125, 1.0, 8.0], [0.7, 0.8, 0.95]) is None


# --- scoring -----------------------------------------------------------------

def verdicts(rep):
    return {p["id"]: p["verdict"] for p in rep["predictions"]}


def test_the_l2_story_passes_everything():
    pts = [l2_story(c) for _, c in S.contexts_for(L2, BPT)]
    rep = S.analyze(payload(pts))
    v = verdicts(rep)
    assert v == {"P1": "HOLDS", "P2": "HOLDS", "P3": "HOLDS", "P4": "HOLDS", "P5": "HOLDS"}
    p2 = next(p for p in rep["predictions"] if p["id"] == "P2")
    assert 0.5 <= p2["x_star"] <= 2.0
    assert p2["ctx_star"] == pytest.approx(p2["x_star"] * L2 / BPT)


def test_a_context_length_story_fails_the_primary():
    """Quantization pays past a fixed context, L2 irrelevant: crossing far below 1x."""
    pts = []
    for _, c in S.contexts_for(L2, BPT):
        q = 0.8 if c < 6000 else 1.3           # flips at ~0.18x L2 on this card
        pts.append(point(c, 10.0 * q, 10.0, 10.0 * 1.3, 10.0))
    v = verdicts(S.analyze(payload(pts)))
    assert v["P1"] == "FAILS"
    assert v["P2"] == "FAILS"


def test_a_story_where_quantization_never_pays_hot_fails():
    pts = [point(c, 9.0, 10.0, 13.0, 10.0) for _, c in S.contexts_for(L2, BPT)]
    v = verdicts(S.analyze(payload(pts)))
    assert v["P1"] == "HOLDS"
    assert v["P2"] == "FAILS"                   # no crossing at all
    assert v["P3"] == "FAILS"


def test_zone_b_that_only_matches_cold_fails_p3():
    """hot > 1 is not enough; it has to beat the DRAM-resident ratio."""
    pts = []
    for _, c in S.contexts_for(L2, BPT):
        x = BPT * c / L2
        q = 0.8 if x <= 0.75 else 1.4
        pts.append(point(c, 10.0 * q, 10.0, 14.0, 10.0))   # hot == cold past 1x
    v = verdicts(S.analyze(payload(pts)))
    assert v["P3"] == "FAILS"
    assert v["P4"] == "HOLDS"


def test_skipped_points_are_ignored_and_empty_zones_untestable():
    pts = [l2_story(4096), {"ctx": 999999, "skipped": "budget"}]
    rep = S.analyze(payload(pts))
    v = verdicts(rep)
    assert rep["n_points"] == 1
    assert v["P3"] == "untestable" and v["P4"] == "untestable"


def test_render_names_every_prediction():
    rep = S.analyze(payload([l2_story(c) for _, c in S.contexts_for(L2, BPT)]))
    md = S.render(rep)
    for pid in ("P1", "P2", "P3", "P4", "P5"):
        assert f"**{pid} " in md
    assert "crossing at fp16 =" in md


def test_gate_failures_are_marked_not_hidden():
    pts = [l2_story(c) for _, c in S.contexts_for(L2, BPT)]
    pts[0]["methods"][S.CONTROL]["hot"]["quotable"] = False
    rep = S.analyze(payload(pts))
    assert rep["n_points_quotable"] == rep["n_points"] - 1
    assert "*" in S.render(rep).split("\n")[6]
