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
import statistics
from pathlib import Path

import numpy as np
import pytest

import analyze_dispersion
import audit_claims
import bandwidth_law
import clock_ramp
import between_run
import clock_excursions
import compare_protocols
import dispersion_tier
import thermal_check
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
             samples: int = 50, group_size: int = 32,
             methods: str | None = None, preload: float = 0.0) -> dict:
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
                 "model": "synthetic", "batch": 1,
                 **({"methods": methods} if methods else {}),
                 **({"preload": preload} if preload else {})},
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


def test_a_subset_run_is_not_pooled_with_a_full_one(tmp_path, monkeypatch):
    """`benchmark.py --methods` makes a run cheap enough to repeat often. Pooling
    one with a full run would compare rows measured after different amounts of
    preceding GPU work, which is exactly the variable under study."""
    full = _payload(1.0)
    subset = _payload(1.0)
    subset["args"]["methods"] = "attribution"
    a, b = tmp_path / "full.json", tmp_path / "subset.json"
    a.write_text(json.dumps(full), encoding="utf-8")
    b.write_text(json.dumps(subset), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["between_run.py", str(a), str(b)])
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


# ---------------------------------------------------------------------------
# clock_excursions: the P-state rate, and the statistic that measures it
# ---------------------------------------------------------------------------


def _clocked(mem_by_method: dict, *, quotable=True, ctx=CTX) -> dict:
    """A payload where each method's memory clock is dictated per regime."""
    d = _payload(1.0, quotable=quotable)
    for row in d["results"]:
        mhz = mem_by_method.get(row["method"])
        if mhz is None:
            continue
        for regime in ("cold", "graph"):
            row["clocks"][regime]["clocks"]["mem_mhz_mean"] = float(mhz)
    return d


def _groups(**spec):
    """name -> [(run label, Bench)] from {name: [clock, clock, ...]}."""
    return {
        name: [(f"{name}{i}", Bench(_clocked({"fused_triton_4b": mhz})))
               for i, mhz in enumerate(mhzs)]
        for name, mhzs in spec.items()
    }


def test_a_steady_clock_is_never_an_excursion():
    g = _groups(a=[11000, 11000, 11000])
    exc, cells = clock_excursions.find_excursions(g)
    assert exc == []
    assert all(c["n_states"] == 1 for c in cells.values())


def test_one_dropped_p_state_is_found_once():
    g = _groups(a=[11000, 11000, 10000])
    exc, _ = clock_excursions.find_excursions(g)
    fused = [e for e in exc if e["method"] == "fused_triton_4b"]
    assert len(fused) == len(("cold", "graph")), fused
    assert all(e["run"] == "a2" for e in fused)
    assert all(0.08 < e["drop_frac"] < 0.10 for e in fused)


def test_all_distinct_observations_do_not_all_become_excursions():
    """The bug this test exists for.

    An earlier version used the mode as the baseline. When every observation is
    distinct every count ties at one, and breaking the tie toward the highest
    clock reported four of six observations as excursions against a baseline one
    run reached once. The median has no tie to break.
    """
    g = _groups(a=[11500, 11000, 10500], b=[10300, 10000, 9800])
    exc, cells = clock_excursions.find_excursions(g)
    fused_cells = [c for k, c in cells.items() if k[0] == "fused_triton_4b"]
    assert all(c["n_states"] == 6 for c in fused_cells), "test setup: all distinct"
    per_regime = len([e for e in exc if e["method"] == "fused_triton_4b"]) / 2
    assert per_regime <= 2, f"{per_regime} of 6 flagged -- baseline is too high"


def test_an_evenly_split_cell_flags_at_most_half():
    """Half the runs in each of two P-states, 10% apart.

    The median baseline lands between them, so the low half is flagged and the
    high half is not. That is the intended reading -- a cell that sits 10% lower
    in half its runs is bimodal and the table should say so -- but the majority
    can never be flagged, which is what stops the statistic from calling a
    cell's normal state an anomaly.
    """
    g = _groups(a=[11000, 11000, 11000], b=[10000, 10000, 10000])
    exc, cells_ = clock_excursions.find_excursions(g)
    fused = [e for e in exc if e["method"] == "fused_triton_4b"]
    per_regime = len(fused) / 2
    assert per_regime == 3, per_regime
    assert {e["group"] for e in fused} == {"b"}
    assert all(c["n_states"] == 2 for k, c in cells_.items()
               if k[0] == "fused_triton_4b")


def test_gate_status_is_carried_through():
    """The point of the report is which excursions were *quoted*, so the row's
    gate status has to survive into the record."""
    g = {
        "a": [("a0", Bench(_clocked({"fused_triton_4b": 11000}))),
              ("a1", Bench(_clocked({"fused_triton_4b": 11000}))),
              ("a2", Bench(_clocked({"fused_triton_4b": 9500}, quotable=False)))],
    }
    exc, _ = clock_excursions.find_excursions(g)
    fused = [e for e in exc if e["method"] == "fused_triton_4b"]
    assert fused and all(e["row_quotable"] is False for e in fused)


def test_cells_are_intersected_across_groups():
    """A subset run times fewer methods. Cells only one group measured cannot be
    compared, and must not silently enter the denominator."""
    full = Bench(_payload(1.0))
    subset_payload = _payload(1.0)
    subset_payload["results"] = [r for r in subset_payload["results"]
                                 if r["method"] in ("fp16_sdpa", "fused_triton_4b")]
    g = {"full": [("f", full)], "subset": [("s", Bench(subset_payload))]}
    got = {m for m, _, _ in clock_excursions.cells(g)}
    assert got == {"fp16_sdpa", "fused_triton_4b"}


def test_render_reports_the_rate_and_the_gate():
    g = _groups(a=[11000, 11000, 10000])
    exc, cells_ = clock_excursions.find_excursions(g)
    md = clock_excursions.render(exc, cells_, g, clock_excursions.EXCURSION_FRAC)
    assert "Rate, by run group" in md
    assert "median MHz" in md
    assert "excursion" in md.lower()


def test_a_preloaded_run_is_not_pooled_with_a_plain_one(tmp_path, monkeypatch):
    """`--preload` deliberately changes the GPU's integrated load before timing.
    That is the variable under test, so two runs that differ in it are two
    experiments."""
    plain, warmed = _payload(1.0), _payload(1.0)
    warmed["args"]["preload"] = 300.0
    a, b = tmp_path / "plain.json", tmp_path / "warm.json"
    a.write_text(json.dumps(plain), encoding="utf-8")
    b.write_text(json.dumps(warmed), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["between_run.py", str(a), str(b)])
    with pytest.raises(SystemExit) as e:
        between_run.main()
    assert "refusing to pool" in str(e.value)


# ---------------------------------------------------------------------------
# compare_protocols: are two protocols measuring the same thing?
# ---------------------------------------------------------------------------


def _proto_group(name, *, scale=1.0, power=75.0, n=2):
    """A labelled group of runs where the fused kernel's speed and the reported
    power draw can be dialled independently."""
    entries = []
    for i in range(n):
        d = _payload(scale, seed=i)
        for row in d["results"]:
            for regime in ("cold", "graph"):
                row["clocks"][regime]["clocks"].update({
                    "sm_mhz_mean": 2700.0, "power_w_mean": power,
                    "temp_c_mean": 70.0, "n_samples": 14,
                })
        entries.append((f"{name}{i}", Bench(d)))
    return entries


def test_identical_protocols_are_not_flagged():
    groups = {"a": _proto_group("a"), "b": _proto_group("b")}
    recs = compare_protocols.ratio_ranges(groups, [CTX])
    assert recs
    assert not any(any(r["disjoint_from_base"].values()) for r in recs)


def test_a_shifted_protocol_is_flagged_as_disjoint():
    """The finding this script exists to produce: one protocol's range missing
    the reference protocol's entirely."""
    groups = {"full": _proto_group("full"), "subset": _proto_group("sub", scale=1.2)}
    recs = compare_protocols.ratio_ranges(groups, [CTX])
    q = next(r for r in recs if r["name"] == "quant_cold")
    assert q["disjoint_from_base"]["subset"] is True
    assert q["shift_from_base"]["subset"] < -0.1


def test_the_first_label_is_the_reference():
    groups = {"full": _proto_group("full"), "subset": _proto_group("sub", scale=1.2)}
    recs = compare_protocols.ratio_ranges(groups, [CTX])
    q = next(r for r in recs if r["name"] == "quant_cold")
    assert set(q["disjoint_from_base"]) == {"subset"}, "reference compares to others"


def test_power_difference_is_reported_per_row_not_averaged():
    """The correction this test is named for.

    Comparing a whole-run mean power hid the effect, because on a part pinned at
    its limit that average is flat by construction. The per-row figure is what
    carries the signal, so it has to survive into the record.
    """
    groups = {"full": _proto_group("full", power=74.2),
              "subset": _proto_group("sub", power=76.6)}
    telem = compare_protocols.telemetry_agreement(
        groups, [CTX], ["triton_fp16_control"])
    assert telem
    pw = telem[0]["telemetry"]["power W"]
    assert pw["per_group"] == {"full": 74.2, "subset": 76.6}
    assert pw["spread_frac"] == pytest.approx(76.6 / 74.2 - 1)


def test_render_names_the_reference_protocol():
    groups = {"full": _proto_group("full"), "subset": _proto_group("sub", scale=1.2)}
    recs = compare_protocols.ratio_ranges(groups, [CTX])
    telem = compare_protocols.telemetry_agreement(groups, [CTX], ["fused_triton_4b"])
    md = compare_protocols.render(recs, telem, groups)
    assert "Reference protocol: **`full`**" in md
    assert "not interchangeable" in md


def test_bandwidth_sensitivity_orders_rows_by_achieved_bandwidth():
    """The predictive form of the protocol finding.

    A row pulling little DRAM bandwidth cannot care what state the memory
    subsystem is in. Build two protocols where only the high-bandwidth method
    shifts, and the correlation must come out strongly positive.
    """
    def group(name, shift_fast):
        entries = []
        for i in range(2):
            d = _payload(1.0, seed=i)
            for row in d["results"]:
                # fp16_sdpa is slow over the same bytes -> low achieved GB/s;
                # fused_triton_4b is fast over them -> high.
                row["cache_bytes_1layer"] = 1_000_000
                if row["method"] == "fused_triton_4b":
                    row["cold"]["median_ms"] *= shift_fast
            entries.append((f"{name}{i}", Bench(d)))
        return entries

    groups = {"base": group("b", 1.0), "other": group("o", 0.90)}
    rows, corr, pair = compare_protocols.bandwidth_sensitivity(
        groups, [CTX], ["fp16_sdpa", "fused_triton_4b"])
    assert pair == ("base", "other")
    by = {r["method"]: r for r in rows}
    # fused is 12.5x faster over identical bytes, so 12.5x the achieved GB/s.
    assert by["fused_triton_4b"]["gb_s"] > by["fp16_sdpa"]["gb_s"]
    assert by["fused_triton_4b"]["shift"] == pytest.approx(-0.10, abs=0.01)
    assert by["fp16_sdpa"]["shift"] == pytest.approx(0.0, abs=0.001)


def test_bandwidth_sensitivity_needs_three_rows_for_a_correlation():
    groups = {"base": _proto_group("b"), "other": _proto_group("o")}
    rows, corr, _ = compare_protocols.bandwidth_sensitivity(
        groups, [CTX], ["fused_triton_4b"])
    assert len(rows) < 3 and corr is None


# ---------------------------------------------------------------------------
# compare_protocols: the 2x2, and which of the two factors actually moved
# ---------------------------------------------------------------------------
#
# `full` (12 methods, ~800 s) and `subset` (3 methods, ~205 s) differ in two
# ways at once, so neither "the run was longer" nor "the memory system was
# recently saturated" could be blamed. The fourth cell -- 12 methods *with* the
# preload -- is what separates them.

FEW, MANY = "attribution", "all"
NONE, PRE = 0.0, 300.0
LEVELS = ([FEW, MANY], [NONE, PRE])


def _cell_group(name, *, methods=None, preload=0.0, scale=1.0, n=3, nbytes=0):
    entries = []
    for i in range(n):
        d = _payload(scale, seed=i, methods=methods, preload=preload)
        if nbytes:
            for row in d["results"]:
                row["cache_bytes_1layer"] = nbytes
        entries.append((f"{name}{i}", Bench(d)))
    return entries


def _square(nbytes=0, **scales):
    """The four protocol cells, in the order the real command line supplies."""
    return {
        "full": _cell_group("f", scale=scales.get("full", 1.0), nbytes=nbytes),
        "subset": _cell_group("s", methods=FEW, nbytes=nbytes,
                              scale=scales.get("subset", 1.0)),
        "preloaded": _cell_group("p", methods=FEW, preload=PRE, nbytes=nbytes,
                                 scale=scales.get("preloaded", 1.0)),
        "fullpre": _cell_group("fp", preload=PRE, nbytes=nbytes,
                               scale=scales.get("fullpre", 1.0)),
    }


def _cells4(few_none, few_pre, many_none, many_pre, jitter=0.0):
    """Four cells, each given as the value it measures, with optional spread."""
    def v(x):
        return [x * (1 + jitter), x, x / (1 + jitter)] if jitter else [x, x, x]
    return {(FEW, NONE): v(few_none), (FEW, PRE): v(few_pre),
            (MANY, NONE): v(many_none), (MANY, PRE): v(many_pre)}


def test_the_design_is_read_from_the_runs_not_from_the_labels():
    """A label is a name this script was handed on the command line; the args
    are what the benchmark actually did."""
    cells, levels, note = compare_protocols.design_cells(_square())
    assert note is None
    assert cells == {(MANY, NONE): "full", (FEW, NONE): "subset",
                     (FEW, PRE): "preloaded", (MANY, PRE): "fullpre"}
    assert levels == ([FEW, MANY], [NONE, PRE]), "few methods first, then all"


def test_method_sets_are_ordered_by_size_not_alphabetically():
    """`all` sorts before `attribution` as a string, which would put the full
    method set in the 'few' slot and silently invert every simple effect."""
    _, levels, _ = compare_protocols.design_cells(_square())
    assert levels[0][0] == FEW and levels[0][1] == MANY


def test_two_labels_for_one_protocol_are_refused():
    groups = _square()
    groups["full"] = _cell_group("x", methods=FEW)   # named full, ran a subset
    cells, _, note = compare_protocols.design_cells(groups)
    assert cells is None and "same protocol" in note


def test_a_group_whose_runs_disagree_about_the_protocol_is_refused():
    groups = _square()
    groups["full"] = [("a", Bench(_payload())),
                      ("b", Bench(_payload(methods=FEW)))]
    cells, _, note = compare_protocols.design_cells(groups)
    assert cells is None and "do not share one protocol" in note and "full" in note


def test_three_protocols_are_not_a_2x2_and_the_missing_cell_is_named():
    """The state of the repo before the fourth protocol was run."""
    groups = _square()
    del groups["fullpre"]
    cells, _, note = compare_protocols.design_cells(groups)
    assert cells is None
    assert "incomplete" in note and "'all', 300.0" in note


def test_additive_factors_produce_no_interaction():
    """If the preload costs the same 2% at both method counts, the interaction
    is zero and both main effects come back exactly."""
    b = 1.25
    eff = compare_protocols.factorial_effects(
        _cells4(b, b * 0.98, b * 1.05, b * 0.98 * 1.05), LEVELS)
    assert eff["main_preload"] == pytest.approx(-0.02, abs=1e-9)
    assert eff["main_methods"] == pytest.approx(0.05, abs=1e-9)
    assert eff["interaction"] == pytest.approx(0.0, abs=1e-9)


def test_an_effect_confined_to_one_level_shows_up_as_an_interaction():
    """The outcome that would mean the two factors are not separable: the
    preload does something at three methods and nothing at twelve."""
    b = 1.25
    eff = compare_protocols.factorial_effects(_cells4(b, b * 0.95, b, b), LEVELS)
    assert eff["simple"]["preload_at_few"] == pytest.approx(-0.05, abs=1e-9)
    assert eff["simple"]["preload_at_many"] == pytest.approx(0.0, abs=1e-9)
    assert eff["interaction"] == pytest.approx(1 / 0.95 - 1, abs=1e-9)


def test_effects_are_multiplicative_not_additive():
    """Ratios and times live on a log scale: a factor that doubles a quantity
    and one that halves it have to cancel, which they only do in logs."""
    eff = compare_protocols.factorial_effects(_cells4(1.0, 2.0, 0.5, 1.0), LEVELS)
    assert eff["main_preload"] == pytest.approx(1.0)
    assert eff["main_methods"] == pytest.approx(-0.5)
    assert eff["interaction"] == pytest.approx(0.0, abs=1e-9)


def test_an_effect_smaller_than_a_cell_varies_by_is_not_resolved():
    """Three runs per cell support 'bigger than the cell's own range' and no
    finer claim, so the yardstick travels with every effect."""
    b = 1.25
    eff = compare_protocols.factorial_effects(
        _cells4(b, b * 0.999, b * 1.05, b * 0.999 * 1.05, jitter=0.01), LEVELS)
    assert eff["noise"] == pytest.approx(1.01 * 1.01 - 1, abs=1e-9)
    assert eff["resolved"]["main_preload"] is False
    assert eff["resolved"]["main_methods"] is True


def test_factorial_ratios_recover_a_planted_preload_effect():
    """End to end from run payloads. `quant_cold` is control/fused, so making
    the fused kernel 2% faster in both preloaded cells plants a +2% preload
    effect and nothing else."""
    groups = _square(preloaded=0.98, fullpre=0.98)
    cells, levels, note = compare_protocols.design_cells(groups)
    assert note is None
    recs = compare_protocols.factorial_ratios(groups, cells, levels, [CTX])
    q = next(r for r in recs if r["name"] == "quant_cold")
    assert q["main_preload"] == pytest.approx(0.0204, abs=0.005)
    assert q["main_methods"] == pytest.approx(0.0, abs=0.005)
    assert q["interaction"] == pytest.approx(0.0, abs=0.005)


def test_factorial_rows_carry_each_row_s_achieved_bandwidth():
    groups = _square(nbytes=1_000_000)
    cells, levels, _ = compare_protocols.design_cells(groups)
    rows = compare_protocols.factorial_rows(
        groups, cells, levels, [CTX], ["fp16_sdpa", "fused_triton_4b"])
    cold = {r["method"]: r for r in rows if r["regime"] == "cold"}
    assert cold["fused_triton_4b"]["gb_s"] > cold["fp16_sdpa"]["gb_s"]
    assert all(r["gb_s"] is None for r in rows if r["regime"] == "graph"), \
        "bandwidth is only meaningful on the DRAM-resident regime"


def test_each_protocol_gets_its_own_bandwidth_correlation():
    """The bug this guards. The compared group used to be whichever label came
    last on the command line, so adding a fourth protocol silently repointed an
    already-published correlation at a different pair of runs."""
    groups = _square(nbytes=1_000_000)
    _, _, pair_a = compare_protocols.bandwidth_sensitivity(
        groups, [CTX], ["fp16_sdpa", "fused_triton_4b"], other="subset")
    _, _, pair_b = compare_protocols.bandwidth_sensitivity(
        groups, [CTX], ["fp16_sdpa", "fused_triton_4b"], other="preloaded")
    assert pair_a == ("full", "subset")
    assert pair_b == ("full", "preloaded")


def test_corr_of_drops_rows_with_no_bandwidth_rather_than_scoring_them():
    r, n = compare_protocols.corr_of(
        [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (None, 4.0)])
    assert n == 3 and r == pytest.approx(1.0)


def test_render_declines_the_2x2_and_says_why():
    groups = _square()
    del groups["fullpre"]
    recs = compare_protocols.ratio_ranges(groups, [CTX])
    telem = compare_protocols.telemetry_agreement(groups, [CTX], ["fused_triton_4b"])
    cells, levels, note = compare_protocols.design_cells(groups)
    md = compare_protocols.render(recs, telem, groups, fac=[], fac_note=note,
                                  design=cells, levels=levels)
    assert "The 2x2" in md
    assert "Not computed" in md and "incomplete 2x2" in md


def test_render_reports_the_2x2_when_the_design_is_complete():
    groups = _square(preloaded=0.98, fullpre=0.98)
    recs = compare_protocols.ratio_ranges(groups, [CTX])
    telem = compare_protocols.telemetry_agreement(groups, [CTX], ["fused_triton_4b"])
    cells, levels, note = compare_protocols.design_cells(groups)
    fac = compare_protocols.factorial_ratios(groups, cells, levels, [CTX])
    md = compare_protocols.render(recs, telem, groups, fac=fac, fac_note=note,
                                  design=cells, levels=levels)
    assert "4 protocols, compared" in md
    assert "The 2x2" in md and "Simple effects" in md
    assert "Not computed" not in md


# ---------------------------------------------------------------------------
# dispersion_tier -- the second reporting tier
#
# The shapes below are not arbitrary: they reproduce the real run's structure.
# `_grid` passes the IQR gate while pinning its median only to ~1.7%, which is
# what `fused_gather_meta_4b@512` does in run 3 and what therefore sets the bar; `_wide` at n=400 fails the gate at ~5.8% IQR while pinning to ~0.6%,
# which is the case the tier exists for; `_drift` fails the gate *and* pins
# badly, which is the case that must keep failing.
# ---------------------------------------------------------------------------


def _wide(mean: float, n: int, spread: float = 0.06, seed: int = 0):
    """Uncorrelated noise. Wide per-sample spread, median pinned by sheer n."""
    rng = random.Random(seed)
    return [mean * (1.0 + rng.uniform(-spread, spread)) for _ in range(n)]


def _grid(mean: float, n: int = 24, spread: float = 0.045, seed: int = 0):
    """A shuffled uniform grid: the IQR is exactly `spread` by construction, so
    this row sits inside the gate for *every* seed. Drawing it randomly at this
    n put the IQR either side of 5% depending on the seed, which would have made
    the fixture decide the thing under test."""
    xs = [mean * (1.0 + spread * (2.0 * k / (n - 1) - 1.0)) for k in range(n)]
    random.Random(seed).shuffle(xs)
    return xs


def _drift(mean: float, n: int = 200, pct: float = 0.15, seed: int = 0):
    """A monotone ramp across the window: wide IQR *and* an unpinned median,
    because the block bootstrap keeps the ramp intact."""
    rng = random.Random(seed)
    return [mean * (1.0 + pct * i / (n - 1)) * (1.0 + rng.uniform(-0.01, 0.01))
            for i in range(n)]


def _tight(mean: float, n: int = 200, seed: int = 0):
    return _wide(mean, n, spread=0.005, seed=seed)


def _tier_payload(spec: dict) -> dict:
    """A results payload built from named sample series.

    `spec` maps method -> (cold_series, graph_series, clock_verified). The
    `quotable` flag is derived the way `benchmark.py` derives it -- clocks
    verified and both regimes inside the IQR gate -- so the fixture cannot
    disagree with the gate it is testing.
    """
    results = []
    for i, (method, (cold, graph, clock_ok)) in enumerate(spec.items()):
        # the gate's own IQR, not a re-implementation of it: a fixture that
        # disagreed with `describe` would be testing the disagreement
        tight = all(analyze_dispersion.iqr_frac(np.asarray(xs, dtype=float))
                    <= dispersion_tier.MAX_IQR_FRAC for xs in (cold, graph))
        results.append({
            "method": method,
            "ctx": CTX,
            "clock_verified": clock_ok,
            "quotable": bool(clock_ok and tight),
            "cold_raw_ms": cold,
            "graph_raw_ms": graph,
            "cold": {"median_ms": statistics.median(cold)},
            "graph": {"median_ms": statistics.median(graph)},
        })
    return {"results": results, "contexts": [CTX], "model": "synthetic",
            "env": {}, "correctness": [], "args": {}, "bit_widths": [NBITS]}


def _standard_spec() -> dict:
    """One tight row, one loose-but-accepted row that sets the bar, one row the
    tier should promote, and one it must not."""
    return {
        "fp16_sdpa": (_tight(100.0, seed=1), _tight(50.0, seed=2), True),
        # passes the gate at exactly 4.5% IQR but pins its median only to ~1.7%
        "triton_fp16_control": (_grid(10.0, seed=3), _tight(5.0, seed=4), True),
        # fails the gate at ~5.8% IQR, median pinned to ~0.6%
        "fused_triton_4b": (_wide(8.0, 400, seed=5), _tight(4.0, seed=6), True),
        # fails the gate and the median is not pinned either
        "fused_gather_meta_4b": (_drift(10.4, seed=7), _tight(5.2, seed=8), True),
    }


def _tier_report(spec: dict | None = None) -> dict:
    return dispersion_tier.build(_tier_payload(spec or _standard_spec()))


def _row(report: dict, method: str) -> dict:
    return dispersion_tier.by_row(report)[(method, CTX)]


def test_a_row_that_passes_the_gate_is_untouched():
    """Tier 1 is the gate's own verdict, restated. The tier must not re-derive
    it, because a star has to keep meaning what it meant before this existed."""
    report = _tier_report()
    r = _row(report, "fp16_sdpa")
    assert r["tier"] == dispersion_tier.TIER_QUOTABLE
    assert r["reason"] == "passes the gate"
    assert r["min_effect_frac"] is None  # a starred row carries no qualifier


def test_promotion_never_changes_the_quotable_count():
    """The property that makes this a report and not a widened gate."""
    payload = _tier_payload(_standard_spec())
    report = dispersion_tier.build(payload)
    assert report["counts"]["quotable"] == sum(
        1 for r in payload["results"] if r["quotable"])


def test_a_wide_row_with_a_pinned_median_is_promoted():
    report = _tier_report()
    r = _row(report, "fused_triton_4b")
    assert r["tier"] == dispersion_tier.TIER_PINNED
    assert r["worst_iqr_frac"] > dispersion_tier.MAX_IQR_FRAC
    assert r["worst_median_ci_halfwidth_frac"] < 0.01
    assert "fails the gate" in r["reason"]


def test_a_row_whose_median_is_not_pinned_stays_rejected():
    """The whole point of not widening `MAX_IQR_FRAC`: this row and the promoted
    one both fail the gate, and only one of them deserves to be readable."""
    report = _tier_report()
    r = _row(report, "fused_gather_meta_4b")
    assert r["tier"] == dispersion_tier.TIER_REJECTED
    assert "worse than the gate's own worst accepted" in r["reason"]


def test_the_bar_is_the_worst_number_the_gate_already_accepts():
    report = _tier_report()
    cal = report["calibration"]
    assert cal["worst"]["method"] == "triton_fp16_control"
    # ...and it is genuinely inside the gate, which is what makes it a fair bar
    assert cal["worst"]["iqr_frac"] <= dispersion_tier.MAX_IQR_FRAC
    assert cal["bar_frac"] == pytest.approx(
        cal["worst"]["median_ci_halfwidth_frac"])


def test_no_promoted_row_is_looser_than_the_bar():
    report = _tier_report()
    bar = report["calibration"]["bar_frac"]
    promoted = [r for r in report["rows"] if r["tier"] == dispersion_tier.TIER_PINNED]
    assert promoted
    assert all(r["worst_median_ci_halfwidth_frac"] <= bar for r in promoted)


def test_a_clock_rejected_row_is_never_promoted():
    """The gate is not a P-state filter, so a clock failure is not a dispersion
    question and this tier has nothing to say about it -- even for a row whose
    median is pinned perfectly."""
    spec = _standard_spec()
    spec["fused_triton_4b"] = (_tight(8.0, seed=9), _tight(4.0, seed=10), False)
    r = _row(_tier_report(spec), "fused_triton_4b")
    assert r["tier"] == dispersion_tier.TIER_REJECTED
    assert "clock-rejected" in r["reason"]


def test_a_row_is_judged_on_its_worse_regime():
    """One pinned regime does not carry a row whose other regime wanders."""
    spec = _standard_spec()
    spec["fused_triton_4b"] = (_tight(8.0, seed=11), _drift(4.0, seed=12), True)
    r = _row(_tier_report(spec), "fused_triton_4b")
    assert r["tier"] == dispersion_tier.TIER_REJECTED


def test_a_run_with_no_quotable_row_promotes_nothing():
    """With nothing accepted there is nothing to calibrate against, and the tier
    declines to exist rather than inventing a bar."""
    spec = {m: (c, g, False) for m, (c, g, _) in _standard_spec().items()}
    report = _tier_report(spec)
    assert report["calibration"]["bar_frac"] is None
    assert report["counts"]["pinned"] == 0


def test_usable_for_weighs_the_effect_against_the_row():
    """A promoted row is admissible per claim, not in general."""
    r = _row(_tier_report(), "fused_triton_4b")
    floor = r["min_effect_frac"]
    assert dispersion_tier.usable_for(r, floor * 1.01)
    assert not dispersion_tier.usable_for(r, floor * 0.99)
    # sign is irrelevant: a 20% slowdown is as large an effect as a 20% speedup
    assert dispersion_tier.usable_for(r, -floor * 1.01)


def test_a_quotable_row_needs_no_effect_size_and_a_rejected_one_is_never_usable():
    report = _tier_report()
    assert dispersion_tier.usable_for(_row(report, "fp16_sdpa"), 0.0001)
    assert not dispersion_tier.usable_for(_row(report, "fused_gather_meta_4b"), 10.0)


def test_the_floor_is_the_multiple_of_the_rows_own_uncertainty():
    r = _row(_tier_report(), "fused_triton_4b")
    assert r["min_effect_frac"] == pytest.approx(
        dispersion_tier.EFFECT_MULTIPLE * r["worst_median_ci_halfwidth_frac"])


def test_render_names_the_calibrator_and_lists_both_outcomes():
    md = dispersion_tier.render(_tier_report())
    assert "triton_fp16_control" in md          # the bar, named
    assert "Tier 2: gate-failed, median pinned" in md
    assert "fused_triton_4b" in md              # promoted
    assert "fused_gather_meta_4b" in md         # still rejected, and said so


def test_render_says_so_when_there_is_no_bar():
    spec = {m: (c, g, False) for m, (c, g, _) in _standard_spec().items()}
    md = dispersion_tier.render(_tier_report(spec))
    assert "no bar to calibrate" in md


def test_tier_end_to_end_writes_both_reports(tmp_path, monkeypatch):
    src = tmp_path / "benchmark.json"
    src.write_text(json.dumps(_tier_payload(_standard_spec())), encoding="utf-8")
    out, md = tmp_path / "tier.json", tmp_path / "tier.md"
    monkeypatch.setattr("sys.argv", ["dispersion_tier.py", "--input", str(src),
                                     "--out", str(out), "--md", str(md)])
    dispersion_tier.main()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["counts"]["pinned"] == 1
    assert report["counts"]["rejected"] == 1
    assert "Dispersion tiers" in md.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The tier, as the audit consumes it
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_tiers():
    """`audit_claims.TIERS` is module state; leave it as it was found."""
    saved = audit_claims.TIERS
    yield
    audit_claims.TIERS = saved


def _tiered(method: str, tier: int, floor: float | None = None) -> dict:
    return {"method": method, "ctx": CTX, "tier": tier,
            "tier_name": dispersion_tier.TIER_NAMES[tier],
            "min_effect_frac": floor,
            "worst_median_ci_halfwidth_frac": (floor / dispersion_tier.EFFECT_MULTIPLE
                                               if floor else 0.001)}


def _bench_with(quotable: bool) -> Bench:
    return Bench(_payload(quotable=quotable))


def test_a_quotable_row_is_unmarked(clean_tiers):
    audit_claims.TIERS = {}
    b = _bench_with(True)
    assert audit_claims.tier_mark(b, "fused_triton_4b", CTX, 0.3) == ""


def test_a_promoted_row_earns_the_tier_mark_for_a_large_effect(clean_tiers):
    b = _bench_with(False)
    audit_claims.TIERS = {
        ("fused_triton_4b", CTX): _tiered("fused_triton_4b",
                                          dispersion_tier.TIER_PINNED, 0.05)}
    assert audit_claims.tier_mark(b, "fused_triton_4b", CTX, 0.30) == audit_claims.TIER2_MARK


def test_a_promoted_row_is_starred_when_the_effect_is_too_small(clean_tiers):
    """The per-claim half of the tier: the same row is admissible for a 30%
    effect and not for a 2% one. A binary gate cannot express that."""
    b = _bench_with(False)
    audit_claims.TIERS = {
        ("fused_triton_4b", CTX): _tiered("fused_triton_4b",
                                          dispersion_tier.TIER_PINNED, 0.05)}
    assert audit_claims.tier_mark(b, "fused_triton_4b", CTX, 0.02) == "*"


def test_a_rejected_row_is_starred_however_large_the_effect(clean_tiers):
    b = _bench_with(False)
    audit_claims.TIERS = {
        ("fused_triton_4b", CTX): _tiered("fused_triton_4b",
                                          dispersion_tier.TIER_REJECTED)}
    assert audit_claims.tier_mark(b, "fused_triton_4b", CTX, 10.0) == "*"


def test_without_the_tier_file_the_audit_reads_exactly_as_before(clean_tiers):
    """The collapse property: absent the tier, every row is starred or not, and
    nothing in the audit can tell that `dispersion_tier.py` was ever written."""
    audit_claims.TIERS = {}
    assert audit_claims.tier_mark(_bench_with(False), "fused_triton_4b", CTX, 10.0) == "*"
    assert audit_claims.tier_mark(_bench_with(True), "fused_triton_4b", CTX, 0.0) == ""


def test_the_worse_marker_wins_over_the_better_one():
    """An unusable row has to dominate a merely-promoted one. String `max` gets
    this backwards, which is why `worst_mark` exists."""
    m = audit_claims.TIER2_MARK
    assert audit_claims.worst_mark("*", m) == "*"
    assert audit_claims.worst_mark(m, "*") == "*"
    assert audit_claims.worst_mark("", m) == m
    assert audit_claims.worst_mark("", "") == ""


def test_the_legend_explains_only_the_markers_that_appear():
    assert audit_claims.tier_legend(set()) == ""
    star_only = audit_claims.tier_legend({"*"})
    assert "did not pass" in star_only and audit_claims.TIER2_MARK not in star_only
    both = audit_claims.tier_legend({"*", audit_claims.TIER2_MARK})
    assert "did not pass" in both and audit_claims.TIER2_MARK in both


def test_usable_needs_the_effect_to_clear_the_rows_floor(clean_tiers):
    b = _bench_with(False)
    audit_claims.TIERS = {
        ("fused_triton_4b", CTX): _tiered("fused_triton_4b",
                                          dispersion_tier.TIER_PINNED, 0.05)}
    assert b.usable("fused_triton_4b", CTX, 0.06)
    assert not b.usable("fused_triton_4b", CTX, 0.04)
    # a row with no tier record at all is judged by the gate alone
    assert not b.usable("fp16_sdpa", CTX, 10.0)


def test_the_audit_says_so_when_no_tier_data_exists(clean_tiers):
    audit_claims.TIERS = {}
    (claim,) = audit_claims._tier_claim()
    assert claim.id == "method.dispersion_tier"
    assert claim.verdict == "MISLEADING"
    assert "dispersion_tier.json" in claim.evidence


def test_the_tier_claim_counts_all_three_verdicts(clean_tiers):
    audit_claims.TIERS = {
        ("a", CTX): _tiered("a", dispersion_tier.TIER_QUOTABLE),
        ("b", CTX): _tiered("b", dispersion_tier.TIER_PINNED, 0.05),
        ("c", CTX): _tiered("c", dispersion_tier.TIER_REJECTED),
    }
    (claim,) = audit_claims._tier_claim()
    assert claim.verdict == "TRUE BUT CONDITIONAL"
    assert "three verdicts, not two" in claim.evidence
    assert "cannot admit a measurement less certain" in claim.evidence


def test_load_tiers_tolerates_a_missing_file(tmp_path):
    assert audit_claims.load_tiers(tmp_path / "nope.json") == {}


def test_load_tiers_indexes_by_method_and_ctx(tmp_path):
    f = tmp_path / "tier.json"
    f.write_text(json.dumps(dispersion_tier.build(_tier_payload(_standard_spec()))),
                 encoding="utf-8")
    tiers = audit_claims.load_tiers(f)
    assert ("fused_triton_4b", CTX) in tiers
    assert tiers[("fused_triton_4b", CTX)]["tier"] == dispersion_tier.TIER_PINNED


# ---------------------------------------------------------------------------
# The six-run coverage counter behind the figure and the README table
# ---------------------------------------------------------------------------


def _coverage_payload(tiers: dict) -> dict:
    """A payload whose chain rows land in the tiers named by `tiers`.

    `tiers` maps method -> "quotable" | "pinned" | "rejected". The series are
    chosen so the tier falls out of the data rather than being asserted: tight
    passes, wide-but-many fails the IQR gate and pins, drifting fails both. The
    calibrator row is always present so there is a bar to judge against.
    """
    shape = {"quotable": lambda seed: (_tight(10.0, seed=seed), _tight(5.0, seed=seed + 1), True),
             "pinned": lambda seed: (_wide(10.0, 400, seed=seed), _tight(5.0, seed=seed + 1), True),
             "rejected": lambda seed: (_drift(10.0, seed=seed), _tight(5.0, seed=seed + 1), True)}
    spec = {"triton_fp16_control_bar": (_grid(10.0, seed=99), _tight(5.0, seed=98), True)}
    for i, (method, name) in enumerate(tiers.items()):
        spec[method] = shape[name](i * 7 + 1)
    return _tier_payload(spec)


def _write_runs(tmp_path, specs) -> list[str]:
    paths = []
    for i, spec in enumerate(specs):
        f = tmp_path / f"run{i}.json"
        f.write_text(json.dumps(_coverage_payload(spec)), encoding="utf-8")
        paths.append(f.name)
    return paths


def test_chain_coverage_counts_runs_not_rows(tmp_path, monkeypatch):
    """Two runs: the chain is complete in both once the pinned tier is admitted,
    and in only one on the gate alone."""
    import make_session_plots as msp

    monkeypatch.setattr(msp, "ROOT", tmp_path)
    monkeypatch.setattr(msp, "CHAIN", ("fp16_sdpa", "triton_fp16_control",
                                       "fused_triton_4b"))
    names = _write_runs(tmp_path, [
        {"fp16_sdpa": "quotable", "triton_fp16_control": "quotable",
         "fused_triton_4b": "quotable"},
        {"fp16_sdpa": "quotable", "triton_fp16_control": "quotable",
         "fused_triton_4b": "pinned"},
    ])
    gate, tier, n = msp._chain_coverage(names)
    assert n == 2
    assert gate[CTX] == 1
    assert tier[CTX] == 2


def test_chain_coverage_does_not_admit_a_rejected_row(tmp_path, monkeypatch):
    import make_session_plots as msp

    monkeypatch.setattr(msp, "ROOT", tmp_path)
    monkeypatch.setattr(msp, "CHAIN", ("fp16_sdpa", "triton_fp16_control",
                                       "fused_triton_4b"))
    names = _write_runs(tmp_path, [
        {"fp16_sdpa": "quotable", "triton_fp16_control": "quotable",
         "fused_triton_4b": "rejected"},
    ])
    gate, tier, n = msp._chain_coverage(names)
    assert n == 1
    assert gate[CTX] == 0 and tier[CTX] == 0


def test_chain_coverage_skips_runs_that_are_not_there(tmp_path, monkeypatch):
    """The figure is drawn from whatever runs exist; a missing file is not an
    error, but it must not count toward the denominator either."""
    import make_session_plots as msp

    monkeypatch.setattr(msp, "ROOT", tmp_path)
    monkeypatch.setattr(msp, "CHAIN", ("fp16_sdpa", "triton_fp16_control",
                                       "fused_triton_4b"))
    names = _write_runs(tmp_path, [
        {"fp16_sdpa": "quotable", "triton_fp16_control": "quotable",
         "fused_triton_4b": "quotable"},
    ])
    _, tier, n = msp._chain_coverage(names + ["nope.json"])
    assert n == 1 and tier[CTX] == 1
    assert msp._chain_coverage(["nope.json"]) is None


# ---------------------------------------------------------------------------
# bandwidth_law -- can the decomposition tell a law from a method label?
# ---------------------------------------------------------------------------


def _bw_rows(spec) -> list[dict]:
    """`spec` is (method, ctx, gb_s, shift_fraction) tuples."""
    return [{"method": m, "ctx": c, "gb_s": g, "shift": s} for m, c, g, s in spec]


def _bw_payload(rows, telemetry=None, protocols=("subset",)) -> dict:
    return {"bandwidth_sensitivity": {p: {"rows": rows} for p in protocols},
            "telemetry": telemetry or []}


def test_a_real_law_shows_up_within_each_method():
    """Bandwidth predicts inside a kernel as well as between kernels."""
    rows = _bw_rows([("A", 512, 10, 0.001), ("A", 2048, 40, 0.004),
                     ("A", 8192, 70, 0.007), ("A", 16384, 100, 0.010),
                     ("B", 512, 110, 0.011), ("B", 2048, 140, 0.014),
                     ("B", 8192, 170, 0.017), ("B", 16384, 200, 0.020)])
    rep = bandwidth_law.build(_bw_payload(rows))
    within = rep["per_protocol"]["subset"]["decomposition"]["within"]
    assert all(w["r"] > 0.99 for w in within)
    assert rep["n_positive"] == rep["n_looks"] == 2


def test_a_method_label_masquerading_as_a_law_is_caught():
    """The test this file exists for. Bandwidth separates the two methods
    perfectly and predicts *nothing* inside either, so the pooled correlation is
    high and the within-method correlation is not. A decomposition that could not
    return this answer would not be evidence when it returns the other one."""
    rows = _bw_rows([("A", 512, 10, 0.001), ("A", 2048, 20, 0.001),
                     ("A", 8192, 30, 0.001), ("A", 16384, 40, 0.001),
                     ("B", 512, 210, 0.020), ("B", 2048, 220, 0.020),
                     ("B", 8192, 230, 0.020), ("B", 16384, 240, 0.020)])
    rep = bandwidth_law.build(_bw_payload(rows))
    assert rep["per_protocol"]["subset"]["loo"]["r"] > 0.9   # pooled: looks like a law
    within = rep["per_protocol"]["subset"]["decomposition"]["within"]
    for w in within:                                        # within: says nothing
        assert np.isnan(w["r"]) or abs(w["r"]) < 0.5


def test_a_method_with_no_bandwidth_range_is_marked_untestable():
    """`fp16_sdpa` sits at 11-12 GB/s at every context, so it cannot test a
    bandwidth law and must not be counted as though it had."""
    rows = _bw_rows([("flat", 512, 11.0, 0.001), ("flat", 2048, 11.4, 0.002),
                     ("flat", 8192, 11.2, 0.001), ("flat", 16384, 11.5, 0.002),
                     ("wide", 512, 30, 0.003), ("wide", 2048, 90, 0.009),
                     ("wide", 8192, 150, 0.015), ("wide", 16384, 210, 0.021)])
    rep = bandwidth_law.build(_bw_payload(rows))
    within = {w["method"]: w for w in
              rep["per_protocol"]["subset"]["decomposition"]["within"]}
    assert not within["flat"]["testable"]
    assert within["wide"]["testable"]
    assert rep["n_looks"] == 1          # only the method with range is counted


def test_leave_one_out_finds_the_row_carrying_the_correlation():
    """Seven rows with no relationship plus one far-out point that manufactures
    one. Dropping that point has to be visible."""
    # the seven need a little scatter of their own: strip it out and dropping the
    # outlier leaves a constant y, where a correlation is undefined rather than
    # small, and the test would be asserting on a nan
    rows = _bw_rows([("A", i, 10 + i, 0.002 + 0.0002 * ((i * 3) % 5))
                     for i in range(7)]
                    + [("A", 99, 300, 0.060)])
    loo = bandwidth_law.leave_one_out(rows)
    assert loo["r"] > 0.9
    assert loo["worst"]["dropped"] == "A@99"
    assert loo["min_r"] < 0.5


def test_residuals_are_centred_and_name_the_planted_misfit():
    rows = _bw_rows([("A", 512, 10, 0.001), ("A", 2048, 50, 0.005),
                     ("A", 8192, 100, 0.010), ("A", 16384, 150, 0.015),
                     ("B", 512, 200, 0.020), ("B", 2048, 250, 0.025),
                     ("B", 8192, 300, 0.030),
                     ("odd", 4096, 60, 0.090)])   # far off the line
    res = bandwidth_law.residuals(rows)
    assert abs(sum(r["residual_pp"] for r in res)) < 1e-6
    worst = max(res, key=lambda r: abs(r["residual_pp"]))
    assert worst["method"] == "odd"


def test_between_method_monotonicity_is_reported_both_ways():
    rising = _bw_rows([("A", 1, 10, 0.001), ("A", 2, 12, 0.001),
                       ("B", 1, 200, 0.020), ("B", 2, 210, 0.020)])
    dec = bandwidth_law.decompose(rising)
    assert dec["between_monotone"]
    inverted = _bw_rows([("A", 1, 10, 0.030), ("A", 2, 12, 0.030),
                         ("B", 1, 200, 0.001), ("B", 2, 210, 0.001)])
    assert not bandwidth_law.decompose(inverted)["between_monotone"]


def test_the_sign_test_needs_every_look_to_agree():
    """One disagreeing look and the p-value is withheld rather than softened."""
    agree = _bw_rows([("A", 1, 10, 0.001), ("A", 2, 40, 0.004),
                      ("A", 3, 70, 0.007), ("A", 4, 100, 0.010),
                      ("B", 1, 110, 0.011), ("B", 2, 140, 0.014),
                      ("B", 3, 170, 0.017), ("B", 4, 200, 0.020)])
    rep = bandwidth_law.build(_bw_payload(agree, protocols=("p1", "p2", "p3")))
    assert rep["n_looks"] == 6 and rep["n_positive"] == 6
    assert rep["sign_test_p"] == pytest.approx(0.5 ** 6)

    mixed = list(agree)
    mixed[4] = {"method": "B", "ctx": 1, "gb_s": 110, "shift": 0.030}   # flips B
    rep2 = bandwidth_law.build(_bw_payload(mixed, protocols=("p1",)))
    assert rep2["n_positive"] < rep2["n_looks"]
    assert rep2["sign_test_p"] is None


def test_memory_clock_constancy_counts_only_real_p_state_moves():
    tel = [
        {"method": "steady", "ctx": 8192, "regime": "cold",
         "telemetry": {"mem MHz": {"per_group": {"full": 11001.0, "subset": 11001.0}}}},
        # inside the sampler's own wobble: the same P-state, not a move
        {"method": "wobble", "ctx": 8192, "regime": "cold",
         "telemetry": {"mem MHz": {"per_group": {"full": 11001.0, "subset": 11020.0}}}},
        {"method": "steps", "ctx": 16384, "regime": "cold",
         "telemetry": {"mem MHz": {"per_group": {"full": 11401.0, "subset": 11001.0}}}},
    ]
    mc = bandwidth_law.mem_clock_constancy(tel, ["full", "subset"])
    assert mc["n_rows"] == 3 and mc["n_varying"] == 1
    assert mc["worst"]["method"] == "steps"
    assert mc["worst"]["spread_mhz"] == pytest.approx(400.0)


def test_bandwidth_law_end_to_end(tmp_path, monkeypatch):
    rows = _bw_rows([("A", 512, 10, 0.001), ("A", 2048, 40, 0.004),
                     ("A", 8192, 70, 0.007), ("A", 16384, 100, 0.010),
                     ("B", 512, 110, 0.011), ("B", 2048, 140, 0.014),
                     ("B", 8192, 170, 0.017), ("B", 16384, 200, 0.020)])
    src = tmp_path / "cmp.json"
    src.write_text(json.dumps(_bw_payload(rows)), encoding="utf-8")
    out, md = tmp_path / "bw.json", tmp_path / "bw.md"
    monkeypatch.setattr("sys.argv", ["bandwidth_law.py", "--input", str(src),
                                     "--out", str(out), "--md", str(md)])
    bandwidth_law.main()
    text = md.read_text(encoding="utf-8")
    assert "Within a single kernel" in text and "Leave-one-out" in text
    assert json.loads(out.read_text(encoding="utf-8"))["n_positive"] == 2


def test_bandwidth_law_refuses_a_missing_input(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["bandwidth_law.py", "--input",
                                     str(tmp_path / "nope.json")])
    with pytest.raises(SystemExit):
        bandwidth_law.main()


# ---------------------------------------------------------------------------
# method.bandwidth_law, as the audit carries it
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_bwlaw():
    saved = audit_claims.BWLAW
    yield
    audit_claims.BWLAW = saved


def _bwlaw_report(shifts=(0.001, 0.004, 0.007, 0.010)):
    """A real bandwidth_law report, built by the real code, so the audit is
    tested against the shape that actually reaches it."""
    rows = ([{"method": "A", "ctx": c, "gb_s": g, "shift": s}
             for c, g, s in zip((512, 2048, 8192, 16384), (10, 40, 70, 100), shifts)]
            + [{"method": "B", "ctx": c, "gb_s": g, "shift": s}
               for c, g, s in zip((512, 2048, 8192, 16384), (110, 140, 170, 200),
                                  (0.011, 0.014, 0.017, 0.020))])
    return bandwidth_law.build({"bandwidth_sensitivity": {"subset": {"rows": rows}},
                                "telemetry": []})


def test_the_audit_says_so_when_the_decomposition_is_missing(clean_bwlaw):
    audit_claims.BWLAW = {}
    (claim,) = audit_claims._bandwidth_law_claim()
    assert claim.id == "method.bandwidth_law"
    assert claim.verdict == "MISLEADING"
    assert "bandwidth_law.json" in claim.evidence
    # the reason it is misleading is the interpretation, not the arithmetic
    assert "method effect" in claim.evidence


def test_the_claim_is_conditional_when_every_look_agrees(clean_bwlaw):
    audit_claims.BWLAW = _bwlaw_report()
    (claim,) = audit_claims._bandwidth_law_claim()
    assert claim.verdict == "TRUE BUT CONDITIONAL"
    assert "2 of 2" in claim.evidence
    assert "one card" in claim.evidence          # says what it is conditional on


def test_the_claim_reverts_to_misleading_when_a_look_disagrees(clean_bwlaw):
    """One kernel in which bandwidth does not predict is enough to withdraw the
    interpretation -- the whole point of the decomposition is that it can."""
    audit_claims.BWLAW = _bwlaw_report(shifts=(0.010, 0.007, 0.004, 0.001))
    (claim,) = audit_claims._bandwidth_law_claim()
    assert claim.verdict == "MISLEADING"


def test_the_claim_names_the_method_that_misfits(clean_bwlaw):
    audit_claims.BWLAW = _bwlaw_report()
    (claim,) = audit_claims._bandwidth_law_claim()
    assert "Where it misfits is" in claim.evidence


def test_the_excluded_method_is_named_as_excluded(clean_bwlaw):
    """A method with no bandwidth range must be reported as not counted, not
    silently dropped -- otherwise '6 of 6' reads as more agreement than it is."""
    audit_claims.BWLAW = _bwlaw_report()
    (claim,) = audit_claims._bandwidth_law_claim()
    assert "reported as excluded" in claim.falsification_attempted


def test_load_bandwidth_law_tolerates_a_missing_file(tmp_path):
    assert audit_claims.load_bandwidth_law(tmp_path / "nope.json") == {}


def test_load_bandwidth_law_round_trips(tmp_path):
    f = tmp_path / "bw.json"
    f.write_text(json.dumps(_bwlaw_report()), encoding="utf-8")
    assert audit_claims.load_bandwidth_law(f)["n_positive"] == 2


# ---------------------------------------------------------------------------
# The rival-predictor check
# ---------------------------------------------------------------------------


def _rows_with_time(spec) -> list[dict]:
    """(method, ctx, gb_s, base_ms, shift) -- `base_ms` is what lets the rival
    check derive bytes moved without a second input file."""
    return [{"method": m, "ctx": c, "gb_s": g, "base_ms": t, "shift": s}
            for m, c, g, t, s in spec]


def test_bandwidth_wins_when_bandwidth_is_the_truth():
    rows = _rows_with_time([("A", c, g, 1.0, g * 1e-4)
                            for c, g in zip((512, 2048, 8192, 16384),
                                            (10, 60, 120, 200))]
                           + [("B", c, g, 1.0, g * 1e-4)
                              for c, g in zip((512, 2048, 8192, 16384),
                                              (30, 90, 150, 240))])
    best = bandwidth_law.rival_predictors(rows)[0]
    assert best["predictor"] == "achieved GB/s"
    assert best["residual_sd_pp"] < 1e-6      # an exact line


def test_a_better_predictor_is_reported_when_there_is_one():
    """If the sensitivity really tracked time rather than bandwidth, this check
    has to say so -- otherwise it is decoration. Bandwidth is held nearly flat
    while time carries the signal."""
    gbs = [100, 101, 99, 102, 100, 101, 99, 102]
    times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    rows = _rows_with_time([("A" if i < 4 else "B", 512 * (i + 1), g, t, t * 1e-3)
                            for i, (g, t) in enumerate(zip(gbs, times))])
    ranked = bandwidth_law.rival_predictors(rows)
    assert ranked[0]["predictor"] != "achieved GB/s"
    names = [d["predictor"] for d in ranked]
    assert names.index("time") < names.index("achieved GB/s")


def test_a_rival_that_wins_everywhere_is_named():
    gbs = [100, 101, 99, 102, 100, 101, 99, 102]
    times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    rows = _rows_with_time([("A" if i < 4 else "B", 512 * (i + 1), g, t, t * 1e-3)
                            for i, (g, t) in enumerate(zip(gbs, times))])
    rep = bandwidth_law.build({"bandwidth_sensitivity":
                               {p: {"rows": rows} for p in ("p1", "p2")},
                               "telemetry": []})
    assert rep["rival_beats_bandwidth_everywhere"]


def test_a_rival_winning_under_one_protocol_only_is_not_adopted():
    """One protocol out of three is a coin landing heads, and the report has to
    treat it that way."""
    true_bw = _rows_with_time([("A", c, g, 1.0, g * 1e-4)
                               for c, g in zip((512, 2048, 8192, 16384),
                                               (10, 60, 120, 200))]
                              + [("B", c, g, 1.0, g * 1e-4)
                                 for c, g in zip((512, 2048, 8192, 16384),
                                                 (30, 90, 150, 240))])
    # one protocol where a squared term happens to fit better
    curved = _rows_with_time([(r["method"], r["ctx"], r["gb_s"], r["base_ms"],
                               (r["gb_s"] ** 2) * 1e-7) for r in true_bw])
    rep = bandwidth_law.build({"bandwidth_sensitivity":
                               {"p1": {"rows": true_bw}, "p2": {"rows": true_bw},
                                "p3": {"rows": curved}},
                               "telemetry": []})
    assert rep["rival_beats_bandwidth_everywhere"] == []
    assert rep["rival_wins"]                       # but it is still reported
    md = bandwidth_law.render(rep)
    assert "Reported, not adopted." in md


def test_the_rival_table_lists_every_predictor_for_every_protocol():
    rows = _rows_with_time([("A", c, g, 1.0, g * 1e-4)
                            for c, g in zip((512, 2048, 8192, 16384),
                                            (10, 60, 120, 200))]
                           + [("B", c, g, 2.0, g * 1e-4)
                              for c, g in zip((512, 2048, 8192, 16384),
                                              (30, 90, 150, 240))])
    rep = bandwidth_law.build({"bandwidth_sensitivity":
                               {p: {"rows": rows} for p in ("p1", "p2")},
                               "telemetry": []})
    md = bandwidth_law.render(rep)
    for name in ("achieved GB/s", "bytes moved", "log time"):
        assert name in md


def test_the_rival_check_degrades_without_base_ms():
    """A report written before `base_ms` was recorded still gets the
    bandwidth-shape comparison rather than an exception."""
    rows = [{"method": "A", "ctx": c, "gb_s": g, "shift": g * 1e-4}
            for c, g in zip((512, 2048, 8192, 16384), (10, 60, 120, 200))]
    names = [d["predictor"] for d in bandwidth_law.rival_predictors(rows)]
    assert "achieved GB/s" in names
    assert "time" not in names and "bytes moved" not in names
    # ...and with it, the time-derived rivals come back
    for r, t in zip(rows, (1.0, 2.0, 3.0, 4.0)):
        r["base_ms"] = t
    names = [d["predictor"] for d in bandwidth_law.rival_predictors(rows)]
    assert "time" in names and "bytes moved" in names


# ---------------------------------------------------------------------------
# The scale-free power-law fit
# ---------------------------------------------------------------------------


def test_the_power_law_recovers_a_planted_exponent():
    # scaled so every row clears SHIFT_FLOOR_PCT -- below it the rows are
    # dropped and there is nothing left to recover an exponent from
    rows = [{"method": "A", "ctx": i, "gb_s": g, "shift": (g ** 1.5) * 1e-4}
            for i, g in enumerate((10, 40, 90, 160, 250))]
    pl = bandwidth_law.power_law(rows)
    assert pl["exponent"] == pytest.approx(1.5, abs=1e-6)
    assert pl["r"] == pytest.approx(1.0, abs=1e-6)


def test_rows_at_the_measurement_floor_are_dropped_and_counted():
    """Their logarithm says more about the sampler's resolution than about
    bandwidth, so they are excluded -- and the exclusion is reported, not silent."""
    rows = [{"method": "A", "ctx": i, "gb_s": g, "shift": s}
            for i, (g, s) in enumerate([(10, 0.0000001), (40, 0.004),
                                        (90, 0.009), (160, 0.016),
                                        (250, 0.025)])]
    pl = bandwidth_law.power_law(rows)
    assert pl["n_total"] == 5 and pl["n_used"] == 4 and pl["n_below_floor"] == 1


def test_the_power_law_declines_when_too_little_survives_the_floor():
    rows = [{"method": "A", "ctx": i, "gb_s": g, "shift": 1e-9}
            for i, g in enumerate((10, 40, 90, 160))]
    pl = bandwidth_law.power_law(rows)
    assert pl["exponent"] is None
    assert pl["n_used"] == 0


def test_the_power_law_names_the_method_that_misfits():
    # `bad` must be *scattered*, not merely offset: a uniform multiplier is
    # absorbed by the fit's intercept and misfits nothing
    clean = [{"method": "good", "ctx": i, "gb_s": g, "shift": g * 1e-3}
             for i, g in enumerate((10, 40, 90, 160))]
    off = [{"method": "bad", "ctx": i, "gb_s": g, "shift": s}
           for i, (g, s) in enumerate([(20, 0.200), (60, 0.005),
                                       (110, 0.400), (200, 0.010)])]
    pl = bandwidth_law.power_law(clean + off)
    assert pl["worst_method"] == "bad"


def test_the_report_says_when_the_misfit_does_not_survive_the_refit():
    """A method worst under the linear fit but not under the scale-free one must
    be described as not robust, not quietly kept."""
    rows_a = [{"method": "A", "ctx": i, "gb_s": g, "base_ms": 1.0,
               "shift": g * 1e-4} for i, g in enumerate((10, 40, 90, 160))]
    rows_b = [{"method": "B", "ctx": i, "gb_s": g, "base_ms": 1.0,
               "shift": g * 1e-4 * 4} for i, g in enumerate((20, 60, 110, 200))]
    flipped = ([dict(r, shift=r["shift"] * 4) for r in rows_a]
               + [dict(r, shift=r["shift"] / 4) for r in rows_b])
    rep = bandwidth_law.build({"bandwidth_sensitivity": {
        "p1": {"rows": rows_a + rows_b}, "p2": {"rows": rows_a + rows_b},
        "p3": {"rows": flipped}}, "telemetry": []})
    md = bandwidth_law.render(rep)
    assert "Is the straight line the problem?" in md
    assert "not robust to the choice of fit" in md or "in all 3 protocols" in md


def test_the_bandwidth_figure_declines_without_its_input():
    """`make_session_plots.py` draws whatever exists; a payload with no
    bandwidth block must be skipped and said to be skipped, not crash the deck."""
    import make_session_plots as msp

    assert msp.plot_bandwidth_law({}) is False
    assert msp.plot_bandwidth_law({"bandwidth_sensitivity": {}}) is False


# ---------------------------------------------------------------------------
# thermal_check -- is temperature big enough to move a P-state?
# ---------------------------------------------------------------------------


def _thermal_payload(spec) -> dict:
    """`spec` is (method, ctx, regime, mem_mhz, temp_c) tuples."""
    byrow: dict = {}
    for m, c, reg, mem, t in spec:
        byrow.setdefault((m, c), {})[reg] = {
            "clocks": {"mem_mhz_mean": mem, "temp_c_mean": t,
                       "power_w_mean": 75.0, "sm_mhz_mean": 2750.0}}
    return {"results": [{"method": m, "ctx": c, "clocks": regs}
                        for (m, c), regs in byrow.items()]}


def test_the_thermal_slope_is_recovered():
    runs = []
    for t, mem in ((70.0, 11000.0), (72.0, 10900.0), (74.0, 10800.0)):
        runs.append(_thermal_payload([("A", 8192, "cold", mem, t)]))
    rep = thermal_check.build({"p": runs})
    assert rep["pooled"]["slope_mhz_per_c"] == pytest.approx(-50.0, abs=1e-6)


def test_between_cell_variation_cannot_leak_into_the_fit():
    """The test this file turns on. Two cells that differ enormously in both
    temperature and clock, with *no* relationship inside either: a pooled fit on
    raw values would report a strong slope, and the within-cell fit must not."""
    runs = []
    for i, (ta, tb) in enumerate(((-1.0, -1.0), (0.0, 0.0), (1.0, 1.0))):
        runs.append(_thermal_payload([
            # cool cell, low clock; hot cell, high clock -- opposite to thermal
            ("cool", 512, "cold", 10000.0 + ta * 0.0, 60.0 + ta),
            ("hot", 16384, "cold", 11500.0 + tb * 0.0, 85.0 + tb),
        ]))
    rep = thermal_check.build({"p": runs})
    assert abs(rep["pooled"]["slope_mhz_per_c"]) < 1e-6


def test_a_cell_with_too_few_observations_is_skipped():
    runs = [_thermal_payload([("A", 8192, "cold", 11000.0, 70.0),
                              ("B", 512, "cold", 10000.0, 60.0)]),
            _thermal_payload([("A", 8192, "cold", 10900.0, 72.0)])]
    rep = thermal_check.build({"p": runs})
    # A has 2 observations, B has 1 -- neither reaches MIN_PER_CELL
    assert rep["per_protocol"]["p"]["n_cells"] == 0
    assert rep["pooled"]["slope_mhz_per_c"] is None


def test_a_window_without_telemetry_is_ignored():
    payload = {"results": [
        {"method": "A", "ctx": 8192,
         "clocks": {"cold": {"clocks": {"mem_mhz_mean": 11000.0, "temp_c_mean": 70.0}},
                    "graph": {"clocks": {"mem_mhz_mean": None, "temp_c_mean": 70.0}}}},
        {"method": "B", "ctx": 512, "clocks": {"cold": {"clocks": {}}}},
    ]}
    assert len(thermal_check.observations(payload)) == 1


def test_the_sufficiency_verdict_follows_the_arithmetic():
    """A steep enough slope over the same temperature span must flip the verdict
    -- otherwise the conclusion is baked in rather than computed."""
    # two protocols, because the span the verdict divides by is the spread of
    # mean temperature *between* protocols -- with one there is nothing to span
    def groups(per_c):
        return {
            "cool": [_thermal_payload([("A", 8192, "cold", 11000.0 - per_c * i,
                                        70.0 + i)]) for i in range(3)],
            "hot": [_thermal_payload([("A", 8192, "cold", 10000.0 - per_c * i,
                                       90.0 + i)]) for i in range(3)],
        }
    assert not thermal_check.build(groups(5.0))["thermal_mechanism_sufficient"]
    assert thermal_check.build(groups(500.0))["thermal_mechanism_sufficient"]


def test_the_report_states_the_extrapolation_caveat():
    runs = [_thermal_payload([("A", 8192, "cold", 11000.0 - 50 * i, 70.0 + i),
                              ("B", 512, "cold", 10000.0 - 50 * i, 80.0 + i)])
            for i in range(3)]
    md = thermal_check.render(thermal_check.build({"p": runs}))
    assert "extrapolation" in md
    assert "not independent" in md
    assert "a temperature sweep would have to achieve" in md


def test_thermal_check_refuses_a_missing_run(tmp_path):
    with pytest.raises(SystemExit):
        thermal_check.load_groups([f"p={tmp_path / 'nope.json'}"])


def test_the_decay_test_finds_a_planted_decay():
    """A warm group whose advantage is large early and gone late, with little
    scatter, must come out established."""
    cells = [("A", 512), ("B", 512), ("C", 2048), ("D", 2048), ("E", 8192), ("F", 8192), ("G", 16384), ("H", 16384)]
    warm, cold = [], []
    for _ in range(3):
        warm.append(_thermal_payload(
            [(m, c, "cold", 11000.0 + (300.0 if i < 4 else 0.0), 70.0)
             for i, (m, c) in enumerate(cells)]))
        cold.append(_thermal_payload(
            [(m, c, "cold", 11000.0, 70.0) for m, c in cells]))
    d = thermal_check.preload_decay(warm, cold)
    assert d["early_mhz"] == pytest.approx(300.0)
    assert d["late_mhz"] == pytest.approx(0.0)
    assert d["significant"]


def test_a_decay_buried_in_scatter_is_not_called_established():
    """The case the real data is in: the point estimates lean the right way and
    the error bars swallow them."""
    cells = [("A", 512), ("B", 512), ("C", 2048), ("D", 2048), ("E", 8192), ("F", 8192), ("G", 16384), ("H", 16384)]
    noise = [+400, -350, +380, -300, -330, +420, -360, +340]
    warm, cold = [], []
    for _ in range(3):
        warm.append(_thermal_payload(
            [(m, c, "cold", 11000.0 + noise[i], 70.0)
             for i, (m, c) in enumerate(cells)]))
        cold.append(_thermal_payload(
            [(m, c, "cold", 11000.0, 70.0) for m, c in cells]))
    d = thermal_check.preload_decay(warm, cold)
    assert not d["significant"]
    assert d["runs_per_protocol_needed"] is None or d["runs_per_protocol_needed"] > 3


def test_the_decay_test_declines_with_too_few_shared_cells():
    warm = [_thermal_payload([("A", 512, "cold", 11000.0, 70.0)])]
    cold = [_thermal_payload([("A", 512, "cold", 10900.0, 70.0)])]
    d = thermal_check.preload_decay(warm, cold)
    assert d["early_mhz"] is None


def test_measurement_order_is_read_from_the_results_list():
    """Position-in-run is only recoverable because benchmark.py writes results in
    measurement order; if that stopped being true this would be silently wrong."""
    payload = _thermal_payload([("first", 512, "cold", 11000.0, 70.0),
                                ("second", 2048, "cold", 11000.0, 70.0),
                                ("third", 8192, "cold", 11000.0, 70.0)])
    cells = thermal_check.ordered_cells([payload])
    assert cells[("first", 512, "cold")]["position"] == 0
    assert cells[("third", 8192, "cold")]["position"] == 2


def test_the_report_states_what_the_underpowered_test_would_need():
    cells = [("A", 512), ("B", 512), ("C", 2048), ("D", 2048), ("E", 8192), ("F", 8192), ("G", 16384), ("H", 16384)]
    noise = [+400, -350, +380, -300, -330, +420, -360, +340]
    warm, cold = [], []
    for k in range(3):
        warm.append(_thermal_payload(
            [(m, c, "cold", 11000.0 + noise[i] - 5 * k, 70.0 + k)
             for i, (m, c) in enumerate(cells)]))
        cold.append(_thermal_payload(
            [(m, c, "cold", 11000.0 - 10 * k, 70.0 + k) for m, c in cells]))
    rep = thermal_check.build({"warm": warm, "cold": cold}, [("warm", "cold")])
    md = thermal_check.render(rep)
    assert "The other arm" in md
    assert "runs/protocol needed" in md
    assert "measure the clock ramp directly" in md


# ---------------------------------------------------------------------------
# clock_ramp -- the time constant, and the bug that hid in the summary
# ---------------------------------------------------------------------------


def _ramp_rows(spec):
    """(t, mem_mhz, sm_mhz, util_pct) tuples."""
    return [{"t": t, "mem_mhz": mem, "sm_mhz": sm, "util_pct": u,
             "temp_c": 60.0, "power_w": 70.0} for t, mem, sm, u in spec]


def test_the_ceiling_is_the_sustained_level_not_the_boost():
    """The bug this function had. A card that boosts for a few seconds and then
    settles must be measured against the level it settles at -- taking the peak
    answers 'time to the peak' when the question is 'time to the state the
    benchmark runs in'."""
    rows = _ramp_rows(
        [(0.1 * i, 12001.0, 2700.0, 99.0) for i in range(50)]        # 5 s boost
        + [(5.0 + 0.1 * i, 11001.0, 2700.0, 99.0) for i in range(500)])  # 50 s settled
    got = clock_ramp.time_to_ceiling(rows)
    assert got["ceiling"] == pytest.approx(11001.0)
    assert got["peak"] == pytest.approx(12001.0)
    assert got["peak_is_transient"]
    assert got["peak_seconds"] == pytest.approx(4.9, abs=0.2)


def test_arrival_is_dated_from_all_samples_not_only_loaded_ones():
    """The clock leaves idle while utilization is still climbing, so filtering
    the arrival search by utilization dates the ramp later than it happened."""
    rows = _ramp_rows(
        [(0.0, 405.0, 400.0, 2.0), (0.1, 405.0, 400.0, 2.0),
         (0.2, 11001.0, 600.0, 39.0)]                                 # low util, clock up
        + [(0.3 + 0.1 * i, 11001.0, 2700.0, 99.0) for i in range(200)])
    got = clock_ramp.time_to_ceiling(rows)
    assert got["first_at_ceiling_s"] == pytest.approx(0.2)


def test_a_single_touch_of_the_ceiling_does_not_end_the_ramp():
    rows = _ramp_rows(
        [(0.0, 11001.0, 2700.0, 99.0)]                     # one sample, then falls
        + [(0.1 + 0.1 * i, 9001.0, 2700.0, 99.0) for i in range(100)]
        + [(10.2 + 0.1 * i, 11001.0, 2700.0, 99.0) for i in range(200)])
    got = clock_ramp.time_to_ceiling(rows)
    assert got["first_at_ceiling_s"] == pytest.approx(10.2, abs=0.05)


def test_no_loaded_samples_means_no_ceiling():
    rows = _ramp_rows([(0.1 * i, 405.0, 400.0, 2.0) for i in range(50)])
    got = clock_ramp.time_to_ceiling(rows)
    assert got["n_loaded"] == 0 and got["ceiling"] is None


def test_the_verdict_turns_on_the_pre_registered_threshold():
    """H1/H2 was committed as 10% of the shortest protocol, so the verdict has to
    move at exactly that line and not somewhere convenient."""
    fast = {"mem_mhz": {"first_at_ceiling_s": 20.0}}
    slow = {"mem_mhz": {"first_at_ceiling_s": 21.0}}
    assert not clock_ramp.verdict(fast, 205.0)["warm_up_arm_viable"]
    assert clock_ramp.verdict(slow, 205.0)["warm_up_arm_viable"]


def test_a_clock_that_never_holds_is_reported_as_such():
    rows = _ramp_rows([(0.1 * i, 9001.0 if i % 2 else 11001.0, 2700.0, 99.0)
                       for i in range(200)])
    v = clock_ramp.verdict({"mem_mhz": clock_ramp.time_to_ceiling(rows)}, 205.0)
    assert not v["settled"]
    assert "does not simply" in v["note"]


def test_groups_differing_only_in_context_order_are_not_called_the_same_protocol():
    """`design_coords` reads method set and preload, which are identical for two
    runs that differ only in measurement order -- so the collision branch used to
    report them as the same protocol, which is false. The 2x2 still cannot hold
    the factor; it just has to say why."""
    a = _payload()
    b = _payload()
    b["contexts"] = list(reversed(a["contexts"] + [4096]))
    groups = {"forward": [("f", Bench(a))], "reversed": [("r", Bench(b))]}
    cells, levels, note = compare_protocols.design_cells(groups)
    assert cells is None and levels is None
    assert "different orders" in note
    assert "the 2x2 does not model" in note
    assert "are the same protocol" not in note


def test_genuinely_identical_protocols_still_say_so():
    groups = {"a": [("a", Bench(_payload()))], "b": [("b", Bench(_payload()))]}
    _, _, note = compare_protocols.design_cells(groups)
    assert "are the same protocol" in note


def _telemetry_payload(spec) -> dict:
    """(method, ctx, regime, sm, mem, temp, power) tuples."""
    byrow: dict = {}
    for m, c, reg, sm, mem, t, pw in spec:
        byrow.setdefault((m, c), {})[reg] = {
            "clocks": {"sm_mhz_mean": sm, "mem_mhz_mean": mem,
                       "temp_c_mean": t, "power_w_mean": pw}}
    return {"results": [{"method": m, "ctx": c, "clocks": regs}
                        for (m, c), regs in byrow.items()]}


def test_telemetry_stability_recovers_a_planted_spread():
    runs = [_telemetry_payload([("A", 8192, "cold", 2700.0 + d, 11000.0,
                                 70.0, 75.0)])
            for d in (-10.0, 0.0, 10.0)]
    out = thermal_check.telemetry_stability(runs)
    assert out["n_cells"] == 1
    assert out["per_key"]["sm_mhz"]["mean_within_cell_sd"] == pytest.approx(10.0)
    assert out["per_key"]["mem_mhz"]["mean_within_cell_sd"] == pytest.approx(0.0)


def test_telemetry_stability_skips_thin_cells():
    runs = [_telemetry_payload([("A", 8192, "cold", 2700.0, 11000.0, 70.0, 75.0)]),
            _telemetry_payload([("A", 8192, "cold", 2710.0, 11000.0, 70.0, 75.0)])]
    assert thermal_check.telemetry_stability(runs)["n_cells"] == 0


def test_telemetry_stability_tolerates_a_missing_variable():
    """An older run may not carry every field; the ones it does carry still
    count, rather than the whole comparison failing."""
    runs = []
    for d in (-10.0, 0.0, 10.0):
        pl = _telemetry_payload([("A", 8192, "cold", 2700.0 + d, 11000.0,
                                  70.0, 75.0)])
        del pl["results"][0]["clocks"]["cold"]["clocks"]["power_w_mean"]
        runs.append(pl)
    out = thermal_check.telemetry_stability(runs)
    assert out["per_key"]["sm_mhz"]["mean_within_cell_sd"] == pytest.approx(10.0)
    assert out["per_key"]["power_w"]["mean_within_cell_sd"] is None


def test_the_report_shows_the_stability_table_only_when_comparing():
    one = {"p": [_telemetry_payload([("A", 8192, "cold", 2700.0 + d, 11000.0,
                                      70.0 + d / 10, 75.0)])
                 for d in (-10.0, 0.0, 10.0)]}
    assert "Does the telemetry know" not in thermal_check.render(
        thermal_check.build(one))
    two = dict(one)
    two["q"] = [_telemetry_payload([("A", 8192, "cold", 2700.0 + d, 11000.0,
                                     70.0 + d / 10, 75.0)])
                for d in (-2.0, 0.0, 2.0)]
    assert "Does the telemetry know" in thermal_check.render(
        thermal_check.build(two))


def test_the_stability_table_survives_a_failed_thermal_fit():
    """The two calculations are independent, so a report must not drop the
    stability table because there was no temperature variation to fit."""
    flat = {g: [_telemetry_payload([("A", 8192, "cold", 2700.0 + d, 11000.0,
                                     70.0, 75.0)]) for d in (-10.0, 0.0, 10.0)]
            for g in ("p", "q")}
    rep = thermal_check.build(flat)
    assert rep["pooled"]["slope_mhz_per_c"] is None
    md = thermal_check.render(rep)
    assert "Not enough temperature variation" in md
    assert "Does the telemetry know" in md


# ---------------------------------------------------------------------------
# Protocol ranges over usable runs only
# ---------------------------------------------------------------------------


def test_a_rejected_run_does_not_set_a_protocols_range():
    """The finding this was written for: a protocol can look unsteady purely
    because one of its runs produced rows the gate rejects."""
    good = [Bench(_payload(1.0, seed=i)) for i in range(2)]
    # a third run whose fused row is far off and gate-rejected
    bad = _payload(1.30, seed=9, quotable=False)
    # ratio_ranges needs two groups to have anything to compare
    groups = {"p": [(f"r{i}", b) for i, b in enumerate(good)] + [("r2", Bench(bad))],
              "q": [(f"s{i}", Bench(_payload(1.0, seed=i + 5))) for i in range(3)]}
    recs = compare_protocols.ratio_ranges(groups, [CTX])
    r = _find(recs, "quant_cold")
    g = r["groups"]["p"]
    assert len(g["values"]) == 3
    assert g["n_usable"] < 3
    # the full range is wider than the usable one
    assert (g["max"] - g["min"]) > (g["usable_max"] - g["usable_min"])


def test_usable_range_equals_full_range_when_nothing_is_rejected():
    groups = {"p": [(f"r{i}", Bench(_payload(1.0, seed=i))) for i in range(3)],
              "q": [(f"s{i}", Bench(_payload(1.02, seed=i + 5))) for i in range(3)]}
    g = _find(compare_protocols.ratio_ranges(groups, [CTX]), "quant_cold")["groups"]["p"]
    assert g["n_usable"] == 3
    assert g["usable_min"] == g["min"] and g["usable_max"] == g["max"]


def test_the_usable_section_appears_only_when_a_run_was_lost():
    clean = {"p": [(f"r{i}", Bench(_payload(1.0, seed=i))) for i in range(3)],
             "q": [(f"s{i}", Bench(_payload(1.02, seed=i + 5))) for i in range(3)]}
    recs = compare_protocols.ratio_ranges(clean, [CTX])
    telem = compare_protocols.telemetry_agreement(clean, [CTX], ["fused_triton_4b"])
    cells, levels, note = compare_protocols.design_cells(clean)
    md = compare_protocols.render(recs, telem, clean, fac=[], fac_note=note,
                                  design=cells, levels=levels)
    assert "over usable runs only" not in md


# ---------------------------------------------------------------------------
# Duration vs excursion: the same within-group test, returning a null
# ---------------------------------------------------------------------------


def _dur_rows(spec):
    """(method, ctx, regime, duration_ms, is_excursion) tuples."""
    return [{"method": m, "ctx": c, "regime": g, "duration_ms": d,
             "is_excursion": e} for m, c, g, d, e in spec]


def _dur_dataset(rate_for, methods=("A", "B"), reps=6):
    """Each (method, ctx) cell gets `reps` observations; `rate_for(method, dur)`
    decides how many of them excurse."""
    spec = []
    for m in methods:
        base = 0.005 if m == "A" else 0.5
        for i, ctx in enumerate((512, 2048, 8192, 16384)):
            dur = base * (2 ** i)
            k = int(round(rate_for(m, i) * reps))
            for j in range(reps):
                spec.append((m, ctx, "cold", dur, j < k))
    return _dur_rows(spec)


def test_a_real_duration_effect_survives_the_within_method_test():
    """Longer rows excurse less, inside every method as well as between them."""
    rows = _dur_dataset(lambda m, i: [0.8, 0.6, 0.4, 0.2][i])
    d = clock_excursions.duration_effect(rows, n_buckets=4)
    assert d["n_methods_negative"] == d["n_methods_scored"]
    assert d["survives_within_method"]
    assert d["sign_test_p"] == pytest.approx(0.5 ** d["n_methods_scored"])


def test_a_between_method_effect_alone_is_reported_as_a_null():
    """The case the real data is in: the short-kernel method excurses a lot and
    the long-kernel one not at all, with no relationship inside either. Pooled
    this looks like a duration law; within method it is nothing."""
    rows = _dur_dataset(lambda m, i: 0.5 if m == "A" else 0.0)
    d = clock_excursions.duration_effect(rows, n_buckets=4)
    assert d["pooled_r"] is not None and d["pooled_r"] < -0.5
    assert not d["survives_within_method"]
    assert d["sign_test_p"] is None


def test_rows_without_a_duration_are_excluded():
    rows = _dur_dataset(lambda m, i: 0.5)
    for r in rows[:10]:
        r["duration_ms"] = None
    d = clock_excursions.duration_effect(rows, n_buckets=4)
    assert d["n"] == len(rows) - 10


def test_the_duration_report_says_which_way_it_came_out():
    real = clock_excursions.render_duration(
        clock_excursions.duration_effect(
            _dur_dataset(lambda m, i: [0.8, 0.6, 0.4, 0.2][i]), n_buckets=4))
    assert "about duration and not about kernel identity" in "\n".join(real)
    null = clock_excursions.render_duration(
        clock_excursions.duration_effect(
            _dur_dataset(lambda m, i: 0.5 if m == "A" else 0.0), n_buckets=4))
    assert "Recorded as a **null**" in "\n".join(null)


def test_the_duration_report_is_empty_without_enough_data():
    assert clock_excursions.render_duration({}) == []
    assert clock_excursions.duration_effect(_dur_rows([]))["pooled_r"] is None
