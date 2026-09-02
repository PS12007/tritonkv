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
import clock_excursions
import compare_protocols
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
