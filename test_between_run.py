"""Tests for the between-run machinery, on synthetic runs with known answers.

The rest of this project's tests check the kernel. These check the thing that
decides what the kernel's numbers are allowed to say, which has now been the
source of seven corrections and zero kernel bugs -- so it is the part most worth
pinning.

Everything here is CPU-only and takes under a second; there is no GPU in this
file. Run it with the rest: `python -m pytest test_between_run.py -q`.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import audit_claims
import between_run
from audit_claims import Bench

CTX = 2048
NBITS = 4
METHODS = [
    "fp16_sdpa",
    "triton_fp16_control",
    "fused_triton_4b",
    "fused_gather_meta_4b",
    "fused_fold_zp_4b",
]


def _series(mean: float, n: int = 60, jitter: float = 0.002, seed: int = 0):
    """A tight, boring sample series. Jitter is small so the CI is narrow and
    any between-run movement has to come from the mean, not from the noise."""
    rng = random.Random(seed)
    return [mean * (1.0 + rng.uniform(-jitter, jitter)) for _ in range(n)]


def _payload(scale: float = 1.0, *, quotable: bool = True, seed: int = 0,
             samples: int = 50, group_size: int = 32) -> dict:
    """A benchmark.json-shaped payload where `fused_triton_4b` is `scale` times
    slower than it is at scale=1.0 and everything else is held fixed."""
    base = {
        "fp16_sdpa": 100.0,
        "triton_fp16_control": 10.0,
        "fused_triton_4b": 8.0,
        "fused_gather_meta_4b": 10.4,
        "fused_fold_zp_4b": 8.2,
    }
    results = []
    for i, m in enumerate(METHODS):
        mean = base[m] * (scale if m == "fused_triton_4b" else 1.0)
        clocks = {
            r: {"clocks": {"mem_mhz_mean": 10000.0 * (scale if m == "fused_triton_4b" else 1.0)}}
            for r in ("cold", "graph")
        }
        results.append({
            "method": m,
            "ctx": CTX,
            "quotable": quotable,
            "cold_raw_ms": _series(mean, seed=seed * 100 + i),
            "graph_raw_ms": _series(mean * 0.5, seed=seed * 100 + i + 50),
            "cold": {"median_ms": mean, "iqr_frac_of_median": 0.01},
            "graph": {"median_ms": mean * 0.5, "iqr_frac_of_median": 0.01},
            "pipelined": {"mean_ms": mean * 0.5},
            "clocks": clocks,
        })
    return {
        "results": results,
        "contexts": [CTX],
        "bit_widths": [NBITS],
        "model": "synthetic",
        "env": {},
        "correctness": [],
        "args": {"samples": samples, "passes": 1, "group_size": group_size,
                 "model": "synthetic", "batch": 1},
        "clock_monitoring": {"monitored": False},
        "wall_clock_seconds": 1.0,
    }


def _benches(*scales, quotable=None):
    quotable = quotable or [True] * len(scales)
    return [Bench(_payload(s, quotable=q, seed=i))
            for i, (s, q) in enumerate(zip(scales, quotable))]


# ---------------------------------------------------------------------------
# compare_ratios
# ---------------------------------------------------------------------------


def _find(records, name):
    return next(r for r in records if r["name"] == name)


def test_identical_runs_do_not_manufacture_spread():
    """Two runs with the same means: the run-to-run interval should be barely
    wider than one run's own, and no verdict may move."""
    recs = between_run.compare_ratios(_benches(1.0, 1.0), NBITS)
    assert recs, "no ratios were computed"
    for r in recs:
        assert r["verdict_stable"], r["name"]
        assert r["point_spread_frac"] < 0.02, (r["name"], r["point_spread_frac"])
        # Two independent draws of the same distribution widen the union a
        # little; a factor of two would mean the machinery is inventing spread.
        assert r["inflation"] < 2.0, (r["name"], r["inflation"])


def test_a_shifted_run_is_reported_as_a_shift():
    """One run where the fused kernel is 20% slower. The ratios that divide by
    it must move by about 20%, and the inflation must be large -- that is the
    whole point of the file."""
    recs = between_run.compare_ratios(_benches(1.0, 1.2), NBITS)
    q = _find(recs, "quant_cold")
    assert 0.18 < q["point_spread_frac"] < 0.22, q["point_spread_frac"]
    assert q["inflation"] > 5.0, q["inflation"]
    assert q["run_to_run_lo"] < min(x["ci_lo"] for x in q["runs"]) + 1e-9
    assert q["run_to_run_hi"] > max(x["ci_hi"] for x in q["runs"]) - 1e-9


def test_verdict_instability_is_detected():
    """A shift big enough to carry the ratio across the bar must show up as a
    verdict change, not merely as a wider interval."""
    # quant_cold = control / fused = 10 / (8*scale). At scale 1.0 that is 1.25x
    # (TRUE); at scale 1.35 it is 0.93x (FALSE).
    recs = between_run.compare_ratios(_benches(1.0, 1.35), NBITS)
    q = _find(recs, "quant_cold")
    assert not q["verdict_stable"], q["verdicts"]
    assert set(q["verdicts"]) == {"TRUE", "FALSE"}, q["verdicts"]


def test_quotability_is_tracked_per_run():
    recs = between_run.compare_ratios(
        _benches(1.0, 1.0, quotable=[True, False]), NBITS)
    assert all(not r["quotable_all_runs"] for r in recs)


def test_hot_and_cold_ratios_are_kept_apart():
    """The L2-conditional finding depends on the two regimes not being pooled."""
    recs = between_run.compare_ratios(_benches(1.0, 1.2), NBITS)
    names = {r["name"] for r in recs}
    assert {"quant_cold", "quant_hot"} <= names
    # The synthetic hot series is exactly half the cold one for every method,
    # so the ratio is identical -- if these ever differ, the regimes crossed.
    assert _find(recs, "quant_cold")["point_min"] == pytest.approx(
        _find(recs, "quant_hot")["point_min"], rel=0.05)


# ---------------------------------------------------------------------------
# compare_rows and the P-state check
# ---------------------------------------------------------------------------


def test_rows_report_time_and_clock_spread_separately():
    rows = between_run.compare_rows(_benches(1.0, 1.2))
    fused = [r for r in rows if r["method"] == "fused_triton_4b"
             and r["regime"] == "cold"][0]
    other = [r for r in rows if r["method"] == "fp16_sdpa"
             and r["regime"] == "cold"][0]
    assert fused["time_spread_frac"] == pytest.approx(0.2, rel=0.05)
    assert fused["mem_spread_frac"] == pytest.approx(0.2, rel=0.05)
    assert other["time_spread_frac"] == pytest.approx(0.0, abs=0.01)


def test_pearson_ignores_missing_clocks():
    assert between_run.pearson([1.0, 2.0, 3.0], [1.0, 2.0, None]) is None
    assert between_run.pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert between_run.pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# The pooling guard
# ---------------------------------------------------------------------------


def test_runs_with_different_configs_are_refused(tmp_path, monkeypatch, capsys):
    """Pooling a 50-sample run with a 10-sample one would put a number in the
    report that no run produced. It has to be an error, not a warning."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_payload(1.0, samples=50)), encoding="utf-8")
    b.write_text(json.dumps(_payload(1.0, samples=10)), encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "between_run.py", str(a), str(b),
        "--out-md", str(tmp_path / "o.md"), "--out-json", str(tmp_path / "o.json"),
    ])
    with pytest.raises(SystemExit) as e:
        between_run.main()
    assert "refusing to pool" in str(e.value)


def test_a_single_run_is_refused(tmp_path, monkeypatch):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(_payload(1.0)), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["between_run.py", str(a)])
    with pytest.raises(SystemExit):
        between_run.main()


def test_end_to_end_writes_both_reports(tmp_path, monkeypatch):
    paths = []
    for i, scale in enumerate((1.0, 1.2, 1.05)):
        p = tmp_path / f"run{i}.json"
        p.write_text(json.dumps(_payload(scale, seed=i)), encoding="utf-8")
        paths.append(str(p))
    md, js = tmp_path / "o.md", tmp_path / "o.json"
    monkeypatch.setattr("sys.argv", ["between_run.py", *paths,
                                     "--bits", "4",
                                     "--out-md", str(md), "--out-json", str(js)])
    between_run.main()
    assert md.exists() and js.exists()
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["meta"]["n_runs"] == 3
    assert payload["ratios"] and payload["rows"]
    text = md.read_text(encoding="utf-8")
    assert "run-to-run" in text.lower()
    assert "run3" in text  # every run gets a column


# ---------------------------------------------------------------------------
# The audit's use of it
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_between():
    """`audit_claims.BETWEEN` is module state; leave it as it was found."""
    saved = audit_claims.BETWEEN
    yield
    audit_claims.BETWEEN = saved


def test_audit_says_so_when_no_between_run_data_exists(clean_between):
    audit_claims.BETWEEN = {}
    (claim,) = audit_claims.audit_between_run()
    assert claim.verdict == "MISLEADING"
    assert "between_run" in claim.evidence


def test_audit_downgrades_a_verdict_that_moved(clean_between):
    audit_claims.BETWEEN = {
        ("quant_cold", CTX, NBITS): {
            "name": "quant_cold", "ctx": CTX, "nbits": NBITS,
            "runs": [{}, {}, {}],
            "point_min": 0.93, "point_max": 1.25,
            "run_to_run_lo": 0.92, "run_to_run_hi": 1.26,
            "inflation": 14.0, "verdicts": ["TRUE", "FALSE", "TRUE"],
            "verdict_stable": False, "quotable_all_runs": True,
        }
    }
    assert audit_claims.between_run_downgrade(
        "TRUE", "quant_cold", CTX, NBITS) == "TRUE BUT CONDITIONAL"
    assert audit_claims.between_run_downgrade(
        "FALSE", "quant_cold", CTX, NBITS) == "MISLEADING"
    note = audit_claims.between_run_note("quant_cold", CTX, NBITS)
    assert "0.930x-1.250x" in note and "14.0x" in note
    assert "changed between runs" in note


def test_note_keeps_three_decimals_where_two_would_hide_the_spread(clean_between):
    """A spread of 0.005x on a 0.73x ratio is real and must not print as
    "0.73x-0.73x"; a spread on a 33x ratio does not need the extra digit."""
    audit_claims.BETWEEN = {
        ("quant_hot", CTX, NBITS): {
            "runs": [{}, {}, {}], "point_min": 0.728, "point_max": 0.733,
            "run_to_run_lo": 0.727, "run_to_run_hi": 0.734, "inflation": 2.6,
            "verdicts": ["FALSE"] * 3, "verdict_stable": True,
            "quotable_all_runs": True,
        },
        ("speedup_vs_sdpa", CTX, NBITS): {
            "runs": [{}, {}, {}], "point_min": 33.324, "point_max": 33.852,
            "run_to_run_lo": 33.27, "run_to_run_hi": 33.95, "inflation": 5.8,
            "verdicts": ["TRUE"] * 3, "verdict_stable": True,
            "quotable_all_runs": True,
        },
    }
    assert "0.728x-0.733x" in audit_claims.between_run_note("quant_hot", CTX, NBITS)
    assert "33.32x-33.85x" in audit_claims.between_run_note(
        "speedup_vs_sdpa", CTX, NBITS)


def test_audit_leaves_a_stable_verdict_alone(clean_between):
    audit_claims.BETWEEN = {
        ("quant_cold", CTX, NBITS): {
            "name": "quant_cold", "ctx": CTX, "nbits": NBITS,
            "runs": [{}, {}],
            "point_min": 1.24, "point_max": 1.26,
            "run_to_run_lo": 1.23, "run_to_run_hi": 1.27,
            "inflation": 1.2, "verdicts": ["TRUE", "TRUE"],
            "verdict_stable": True, "quotable_all_runs": True,
        }
    }
    assert audit_claims.between_run_downgrade(
        "TRUE", "quant_cold", CTX, NBITS) == "TRUE"
    # An unknown ratio must pass through untouched rather than raise.
    assert audit_claims.between_run_downgrade("TRUE", "nope", 1, 4) == "TRUE"
    assert audit_claims.between_run_note("nope", 1, 4) == ""


def test_audit_flags_a_star_that_is_a_property_of_the_run(clean_between):
    audit_claims.BETWEEN = {
        ("quant_hot", CTX, NBITS): {
            "name": "quant_hot", "ctx": CTX, "nbits": NBITS,
            "runs": [{}, {}],
            "point_min": 1.1, "point_max": 1.12,
            "run_to_run_lo": 1.09, "run_to_run_hi": 1.13,
            "inflation": 1.3, "verdicts": ["TRUE", "TRUE"],
            "verdict_stable": True, "quotable_all_runs": False,
        }
    }
    note = audit_claims.between_run_note("quant_hot", CTX, NBITS)
    assert "not of the kernel" in note


def test_loader_tolerates_a_missing_file(tmp_path):
    assert audit_claims.load_between_run(tmp_path / "nope.json") == {}
