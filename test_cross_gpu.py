"""CPU-only tests for cross_gpu.py: every verdict must be able to fail.

    ./.venv/Scripts/python.exe -m pytest test_cross_gpu.py -q
"""
from __future__ import annotations

import pytest

import cross_gpu as X

MB = 1_000_000


def raw(ms: float, n: int = 40) -> list[float]:
    return [ms * (1 + 0.003 * ((i % 7) - 3)) for i in range(n)]


def bench(l2: int, quant_cold: dict, quant_hot: dict, footprint: float = 3.0,
          bad: set = frozenset(), contexts=(512, 2048, 8192, 16384)) -> dict:
    """A synthetic benchmark.json with only the rows the scorer reads."""
    rows = []
    for ctx in contexts:
        fused_ms = 0.010 * ctx / 2048
        for method, cold, hot, cache in (
            (X.CONTROL, fused_ms * quant_cold[ctx], fused_ms * quant_hot[ctx], 1024 * ctx),
            (X.FUSED, fused_ms, fused_ms, int(1024 * ctx / 3.2)),
        ):
            ok = (method, ctx) not in bad
            rows.append({"method": method, "ctx": ctx, "cold_raw_ms": raw(cold),
                         "graph_raw_ms": raw(hot), "quotable": ok, "clock_verified": ok,
                         "footprint_over_l2": footprint, "cache_bytes_1layer": cache})
    # dispersion_tier calibrates against quotable rows; give it some.
    rows.append({"method": "fp16_sdpa", "ctx": 512, "cold_raw_ms": raw(0.05),
                 "graph_raw_ms": raw(0.05), "quotable": True, "clock_verified": True})
    return {"env": {"gpu": "synthetic", "l2_cache_bytes": l2, "sm_count": 26,
                    "max_sm_clock_mhz": 3090.0},
            "bandwidth_probe": {"copy_gbps_median": 342.0},
            "contexts": list(contexts), "results": rows}


STORY_COLD = {512: 0.9, 2048: 1.2, 8192: 1.47, 16384: 1.44}
STORY_HOT = {512: 0.73, 2048: 0.81, 8192: 0.79, 16384: 0.73}


def test_the_l2_story_holds_on_this_card():
    c = X.score_card("here", bench(33_554_432, STORY_COLD, STORY_HOT))
    assert c["C1"]["verdict"] == "HOLDS"
    assert c["C2"]["verdict"] == "HOLDS"
    assert c["balance"] == pytest.approx(26 * 3.09 / 342)


def test_c1_fails_when_quantization_does_not_pay_from_dram():
    cold = {**STORY_COLD, 16384: 1.01}
    assert X.score_card("lowbal", bench(33_554_432, cold, STORY_HOT))["C1"]["verdict"] == "FAILS"


def test_c1_excludes_cells_that_were_not_really_cold():
    c = X.score_card("bigL2", bench(72 * MB, STORY_COLD, STORY_HOT, footprint=1.2))
    assert c["C1"]["verdict"] == "untestable"
    assert all("footprint" in x["status"] for x in c["C1"]["cells"])


def test_c1_excludes_unusable_rows_and_does_not_count_them():
    c = X.score_card("noisy", bench(33_554_432, STORY_COLD, STORY_HOT,
                                    bad={(X.FUSED, 16384)}))
    cells = {x["ctx"]: x["status"] for x in c["C1"]["cells"]}
    assert cells[16384] == "excluded: row not usable"
    assert c["C1"]["verdict"] == "untestable"


def test_c2_only_scores_contexts_that_fit_on_a_small_l2_card():
    # 3 MB L2: 0.75x is 2.25 MB, so 512 (0.52 MB) and 2048 (2.10 MB) only.
    c = X.score_card("small", bench(3 * MB, STORY_COLD, STORY_HOT))
    assert [x["ctx"] for x in c["C2"]["cells"]] == [512, 2048]
    assert c["C2"]["verdict"] == "HOLDS"


def test_c2_fails_if_quantization_wins_while_everything_fits():
    hot = {**STORY_HOT, 2048: 1.10}
    assert X.score_card("x", bench(33_554_432, STORY_COLD, hot))["C2"]["verdict"] == "FAILS"


def card(label, l2, x_star, m1=None, bal=None):
    return {"label": label, "l2_bytes": l2, "x_star": x_star,
            "ctx_star": x_star * l2 / 1024, "quant_cold_m1": m1, "balance": bal}


def test_c4_holds_when_the_crossing_scales_with_l2():
    a = X.score_across([card("small", 6 * MB, 1.1), card("big", 33 * MB, 1.05)])
    assert a["C4"]["verdict"] == "HOLDS"


def test_c4_fails_for_a_fixed_context_story():
    small = card("small", 6 * MB, 1.0)
    big = card("big", 33 * MB, 1.0)
    big["ctx_star"] = small["ctx_star"]           # same context on both cards
    big["x_star"] = big["ctx_star"] * 1024 / big["l2_bytes"]
    assert X.score_across([small, big])["C4"]["verdict"] == "FAILS"


def test_c4_is_only_partial_without_a_wide_pair():
    a = X.score_across([card("a", 24 * MB, 1.0), card("b", 33 * MB, 1.1)])
    assert a["C4"]["verdict"].startswith("partial")


def test_m1_is_descriptive_under_five_cards():
    cs = [card("a", 6 * MB, 1, 1.0, 0.16), card("b", 33 * MB, 1, 1.44, 0.235),
          card("c", 72 * MB, 1, 2.0, 0.35)]
    m1 = X.score_across(cs)["M1"]
    assert m1["n"] == 3 and m1["descriptive_only"]
    assert m1["rho_balance"] == pytest.approx(1.0)


def test_spearman_handles_ties_and_sign():
    assert X.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert X.spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(0.866, abs=1e-3)
    assert X.spearman([1], [1]) is None


def test_render_mentions_every_verdict():
    c = X.score_card("here", bench(33_554_432, STORY_COLD, STORY_HOT))
    md = X.render([c], X.score_across([c]))
    assert "C4" in md and "M1" in md and "HOLDS" in md
