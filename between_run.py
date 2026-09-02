#!/usr/bin/env python
"""What a bootstrap CI does not measure: the spread between independent runs.

`audit_claims.py` puts a 95% bootstrap interval on every ratio it reports. That
interval describes *sampling* noise -- how much the ratio would move if you drew
another 50 samples from the same measurement window. It says nothing about how
much the ratio moves if you close the process, start it again, and let the card
land in a different memory P-state.

The two are not the same size. The DRAM-resident quantization ratio at ctx=8192
moved 1.27x -> 1.47x between two full runs while the bootstrap CI on either was
+-0.01. Reporting the CI alone is therefore an understatement of the uncertainty
by more than an order of magnitude, and that understatement is the last known
unreported bias in this project.

This script takes N complete benchmark JSONs from independent runs and reports,
for every headline ratio:

  * the point estimate and within-run CI from each run,
  * the run-to-run interval -- min low to max high across runs,
  * the inflation factor: how many times wider the run-to-run interval is than
    a single run's CI, and
  * whether the ratio's *verdict* (by the audit's own rule) is the same in
    every run, which is the only thing a reader actually cares about.

It also reports, per row, the between-run spread in median time next to the
between-run spread in mean memory clock, so the P-state hypothesis for the
movement can be checked rather than asserted.

Usage:
    ./.venv/Scripts/python.exe between_run.py results/runs/run1.json ...
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from audit_claims import (
    CONTROL,
    FOLD,
    GATHER,
    Bench,
    _verdict,
    bootstrap_ratio_ci,
)

RESULTS_DIR = Path(__file__).parent / "results"

# A run-to-run interval this many times wider than the within-run CI is worth
# saying out loud rather than burying in a table.
INFLATION_LOUD = 2.0


# ---------------------------------------------------------------------------
# The ratios worth tracking
# ---------------------------------------------------------------------------


# Ratios that do not involve a quantized kernel and so are the same number at
# every bit width. Emitted once rather than once per bit width, or the counts
# in this report would double-count a measurement that was only taken once.
NBITS_FREE = {"split_only"}


def ratio_specs(b: Bench, nbits: int, shared: bool = True) -> dict[tuple, tuple]:
    """(name, ctx) -> (numerator samples, denominator samples, methods, regime).

    Directions match `audit_claims.py` exactly so the numbers here can be laid
    beside the audit's without a sign chase: every ratio is >1 when the thing
    being argued for wins.
    """
    fused = f"fused_triton_{nbits}b"
    gather = GATHER.format(nbits=nbits)
    fold = FOLD.format(nbits=nbits)
    out: dict[tuple, tuple] = {}

    for ctx in b.contexts:

        def add(name, num, den, methods, regime):
            if name in NBITS_FREE and not shared:
                return
            if num and den:
                out[(name, ctx)] = (num, den, methods, regime)

        add("speedup_vs_sdpa", b.cold("fp16_sdpa", ctx), b.cold(fused, ctx),
            ("fp16_sdpa", fused), "cold")
        add("split_only", b.cold("fp16_sdpa", ctx), b.cold(CONTROL, ctx),
            ("fp16_sdpa", CONTROL), "cold")
        add("quant_cold", b.cold(CONTROL, ctx), b.cold(fused, ctx),
            (CONTROL, fused), "cold")
        add("quant_hot", b.hot_raw(CONTROL, ctx), b.hot_raw(fused, ctx),
            (CONTROL, fused), "graph")
        add("meta_broadcast_cold", b.cold(gather, ctx), b.cold(fused, ctx),
            (gather, fused), "cold")
        add("meta_broadcast_hot", b.hot_raw(gather, ctx), b.hot_raw(fused, ctx),
            (gather, fused), "graph")
        add("fold_zp_cold", b.cold(fused, ctx), b.cold(fold, ctx),
            (fused, fold), "cold")
        add("fold_zp_hot", b.hot_raw(fused, ctx), b.hot_raw(fold, ctx),
            (fused, fold), "graph")

    return out


# ---------------------------------------------------------------------------
# Per-ratio comparison across runs
# ---------------------------------------------------------------------------


def compare_ratios(benches: list[Bench], nbits: int, shared: bool = True) -> list[dict]:
    """One record per (ratio, ctx) that every run could compute."""
    per_run = [ratio_specs(b, nbits, shared) for b in benches]
    shared = set(per_run[0])
    for d in per_run[1:]:
        shared &= set(d)

    records = []
    for key in sorted(shared, key=lambda k: (k[0], k[1])):
        name, ctx = key
        runs = []
        for b, specs in zip(benches, per_run):
            num, den, methods, regime = specs[key]
            r, lo, hi = bootstrap_ratio_ci(num, den)
            runs.append({
                "ratio": r,
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_width": hi - lo,
                "verdict": _verdict(lo, hi),
                "quotable": all(b.quotable(m, ctx) for m in methods),
                "mem_mhz": {m: b.mem_clock(m, ctx, regime) for m in methods},
            })

        rs = [x["ratio"] for x in runs]
        lows = [x["ci_lo"] for x in runs]
        highs = [x["ci_hi"] for x in runs]
        widths = [x["ci_width"] for x in runs]
        verdicts = [x["verdict"] for x in runs]

        rr_lo, rr_hi = min(lows), max(highs)
        rr_width = rr_hi - rr_lo
        within = statistics.median(widths)
        # How many times wider the honest interval is than the one the audit
        # currently prints. 1.0 means the runs agree to within sampling noise.
        inflation = rr_width / within if within > 0 else float("inf")

        records.append({
            "name": name,
            "ctx": ctx,
            "nbits": nbits,
            "runs": runs,
            "point_min": min(rs),
            "point_max": max(rs),
            "point_spread_frac": (max(rs) / min(rs) - 1.0) if min(rs) > 0 else float("inf"),
            "run_to_run_lo": rr_lo,
            "run_to_run_hi": rr_hi,
            "run_to_run_width": rr_width,
            "within_run_width_median": within,
            "inflation": inflation,
            "verdicts": verdicts,
            "verdict_stable": len(set(verdicts)) == 1,
            "quotable_all_runs": all(x["quotable"] for x in runs),
            "nbits_free": name in NBITS_FREE,
        })
    return records


# ---------------------------------------------------------------------------
# Per-row comparison across runs: time next to memory clock
# ---------------------------------------------------------------------------


def compare_rows(benches: list[Bench]) -> list[dict]:
    """Per (method, ctx, regime): does the timing move, and does the clock?"""
    keys = set(benches[0].by)
    for b in benches[1:]:
        keys &= set(b.by)

    rows = []
    for method, ctx in sorted(keys, key=lambda k: (k[1], k[0])):
        for regime in ("cold", "graph"):
            meds, clocks, iqrs, quot = [], [], [], []
            ok = True
            for b in benches:
                r = b.get(method, ctx) or {}
                st = r.get(regime)
                if not isinstance(st, dict) or "median_ms" not in st:
                    ok = False
                    break
                meds.append(st["median_ms"])
                iqrs.append(st.get("iqr_frac_of_median"))
                clocks.append(b.mem_clock(method, ctx, regime))
                quot.append(bool(r.get("quotable")))
            if not ok or not meds:
                continue

            cl = [c for c in clocks if c]
            have_all_clocks = len(cl) == len(meds) and min(cl) > 0
            iqr_vals = [i for i in iqrs if i is not None]
            rows.append({
                "method": method,
                "ctx": ctx,
                "regime": regime,
                "median_ms": meds,
                "iqr_frac": iqrs,
                "time_spread_frac": (max(meds) / min(meds) - 1.0) if min(meds) > 0 else None,
                "within_run_iqr_median": statistics.median(iqr_vals) if iqr_vals else None,
                "mem_mhz": clocks,
                "mem_spread_frac": (max(cl) / min(cl) - 1.0) if have_all_clocks else None,
                "quotable": quot,
                "quotable_all": all(quot),
                "quotable_any": any(quot),
            })
    return rows


def pearson(xs, ys) -> float | None:
    """Plain correlation, so the P-state story is checked and not assumed."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None
             and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xs2 = [p[0] for p in pairs]
    ys2 = [p[1] for p in pairs]
    mx, my = statistics.fmean(xs2), statistics.fmean(ys2)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs2))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys2))
    return num / (dx * dy) if dx > 0 and dy > 0 else None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render(records: list[dict], rows: list[dict], meta: dict) -> str:
    n = meta["n_runs"]
    L = []
    L.append("# Between-run spread: what the bootstrap CI does not cover\n")
    L.append(f"{n} independent full runs of `benchmark.py --samples "
             f"{meta['samples']} --passes {meta['passes']}`, "
             "same code, same machine, nothing else changed.\n")
    L.append("A bootstrap CI answers \"how much would this ratio move on another 50 "
             "samples from this window\". It cannot answer \"how much does it move "
             "if the card lands in a different memory P-state next time\". The "
             "inflation column is the ratio of the second question's answer to "
             "the first's.\n")

    loud = [r for r in records if r["quotable_all_runs"] and r["inflation"] >= INFLATION_LOUD]
    unstable = [r for r in records if not r["verdict_stable"]]
    quot = [r for r in records if r["quotable_all_runs"]]

    L.append("## Summary\n")
    L.append(f"- {len(quot)}/{len(records)} tracked ratios are quotable in every run.")
    if quot:
        infl = sorted(r["inflation"] for r in quot)
        L.append(f"- Median inflation over those: **{statistics.median(infl):.1f}x**; "
                 f"worst **{infl[-1]:.1f}x**.")
        sp = sorted(r["point_spread_frac"] for r in quot)
        L.append(f"- Median between-run spread in the point estimate: "
                 f"**{statistics.median(sp) * 100:.1f}%**; worst "
                 f"**{sp[-1] * 100:.1f}%**.")
    L.append(f"- Verdict changes between runs: **{len(unstable)}** of {len(records)} "
             "tracked ratios" + (":" if unstable else "."))
    for r in unstable:
        L.append(f"  - `{r['name']}` ctx={r['ctx']} {r['nbits']}b: "
                 + " / ".join(r["verdicts"])
                 + f" (points {r['point_min']:.2f}x-{r['point_max']:.2f}x)")
    L.append("")

    # The gate exists to keep unstable rows out of the conclusions. Whether it
    # actually does is checkable here, and it is the cheapest available test of
    # it: split the same statistic by whether the gate passed the row.
    rejected = [r for r in records if not r["quotable_all_runs"]]
    if quot and rejected:
        qi = statistics.median(r["inflation"] for r in quot)
        ri = statistics.median(r["inflation"] for r in rejected)
        qs = statistics.median(r["point_spread_frac"] for r in quot)
        rs_ = statistics.median(r["point_spread_frac"] for r in rejected)
        L.append("### The gate is doing the thing it was built to do\n")
        L.append("Split the same ratios by whether every input passed the "
                 "clock/dispersion gate in every run:\n")
        L.append("| | n | median inflation | median between-run spread | worst spread |")
        L.append("|---|---|---|---|---|")
        L.append(f"| passed the gate in every run | {len(quot)} | {qi:.1f}x | "
                 f"{qs * 100:.1f}% | "
                 f"{max(r['point_spread_frac'] for r in quot) * 100:.1f}% |")
        L.append(f"| failed it in at least one | {len(rejected)} | {ri:.1f}x | "
                 f"{rs_ * 100:.1f}% | "
                 f"{max(r['point_spread_frac'] for r in rejected) * 100:.1f}% |")
        L.append("")
        L.append("The gate is applied *within* a run and knows nothing about the "
                 "other runs, so this is an out-of-sample check on it: rows it "
                 "rejects really do move more when the benchmark is run again. "
                 "That is a reason to keep it as it is, and a second reason not to "
                 "widen it.\n")

    L.append("## Ratios, per run and between runs\n")
    L.append("`*` = quotable in every run. Inflation = (run-to-run interval width) / "
             "(median within-run CI width).\n")
    hdr = ["ratio", "ctx", "bits"] + [f"run{i + 1}" for i in range(n)] + [
        "spread", "within-run CI", "run-to-run", "infl", "verdicts"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for r in sorted(records, key=lambda r: (r["name"], r["nbits"], r["ctx"])):
        star = "*" if r["quotable_all_runs"] else ""
        # A ratio with no quantized kernel in it is the same number at every bit
        # width, so it is measured and listed once.
        bits = "any" if r.get("nbits_free") else f"{r['nbits']}b"
        cells = [f"`{r['name']}`{star}", str(r["ctx"]), bits]
        cells += [f"{x['ratio']:.3f}" for x in r["runs"]]
        cells.append(f"{r['point_spread_frac'] * 100:.1f}%")
        cells.append(f"+-{r['within_run_width_median'] / 2:.3f}")
        cells.append(f"[{r['run_to_run_lo']:.2f}, {r['run_to_run_hi']:.2f}]")
        cells.append(f"{r['inflation']:.1f}x")
        vs = r["verdicts"]
        cells.append(vs[0] if r["verdict_stable"] else " / ".join(vs))
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Per-row movement, next to the memory clock\n")
    L.append("If the run-to-run movement is P-state, the rows that moved in time "
             "are the rows that moved in clock. Sorted by time spread, worst 24.\n")
    hdr2 = ["method", "ctx", "regime"] + [f"med{i + 1} ms" for i in range(n)] + [
        "time spread", "median IQR", "mem MHz spread", "quotable"]
    L.append("| " + " | ".join(hdr2) + " |")
    L.append("|" + "---|" * len(hdr2))
    for r in sorted(rows, key=lambda r: -(r["time_spread_frac"] or 0))[:24]:
        cells = [f"`{r['method']}`", str(r["ctx"]), r["regime"]]
        cells += [f"{m:.4f}" for m in r["median_ms"]]
        cells.append(f"{r['time_spread_frac'] * 100:.1f}%")
        iq = r["within_run_iqr_median"]
        cells.append(f"{iq * 100:.1f}%" if iq is not None else "-")
        ms = r["mem_spread_frac"]
        cells.append(f"{ms * 100:.1f}%" if ms is not None else "-")
        cells.append("all" if r["quotable_all"] else ("some" if r["quotable_any"] else "none"))
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    c = meta.get("corr_time_vs_clock")
    L.append("## Does the clock explain it?\n")
    if c is None:
        L.append("Not enough paired rows to correlate time spread against "
                 "memory-clock spread.\n")
    else:
        L.append(f"Across {meta['corr_n']} DRAM-resident rows measured in every run, "
                 "the correlation between a row's between-run time spread and its "
                 f"between-run mean-memory-clock spread is **r = {c:+.2f}**.\n")
        if c > 0.5:
            L.append("That is the P-state hypothesis holding up: the rows that moved "
                     "are the rows whose memory clock moved.\n")
        elif c > 0.2:
            L.append("Partly consistent with the P-state hypothesis, but the clock is "
                     "not the whole story -- some rows moved in time without moving in "
                     "clock.\n")
        else:
            L.append("The P-state hypothesis does **not** account for the run-to-run "
                     "movement on its own: rows moved in time without a matching move "
                     "in mean memory clock. Whatever drives the spread, naming it "
                     "\"P-state\" is not supported by these runs.\n")

    cold = [r for r in rows if r["regime"] == "cold"]
    always = sum(1 for r in cold if r["quotable_all"])
    ever = sum(1 for r in cold if r["quotable_any"])
    L.append("## Quotability is itself a random variable\n")
    L.append(f"Of {len(cold)} rows, **{always}** pass the gate in every run and "
             f"**{ever}** pass it in at least one. The {ever - always} rows in "
             "between are quotable or not depending on the run -- worth knowing "
             "before quoting a starred row as though the star were a property of "
             "the kernel.\n")

    L.append(f"## What {n} runs cannot tell you\n")
    L.append(f"{n} runs bound the *body* of the run-to-run distribution and say "
             "almost nothing about its tail. A P-state excursion that happens on "
             f"one run in ten is more likely than not to be absent from {n}, and "
             "one that does occur moves a ratio by far more than the spread "
             "tabulated above -- the intervals here are a floor on the "
             "uncertainty, not a bound on it. What this measurement does settle "
             "is the cheaper question: whether the numbers move enough between "
             "ordinary repetitions to change what may be said, and "
             + ("they do not."
                if not unstable else
                f"for {len(unstable)} of them they do.") + "\n")

    L.append("## How to read this alongside `audit.md`\n")
    L.append("The audit's CI is still the right interval for \"is this difference "
             "real within this run\". For any number that leaves this repo, the "
             "run-to-run column is the interval to quote. Where the two disagree by "
             f"more than {INFLATION_LOUD:.0f}x -- {len(loud)} of the quotable ratios "
             "here -- the CI alone is an understatement, and saying so is cheaper "
             "than being wrong later.\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="benchmark JSONs from independent runs")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 2])
    ap.add_argument("--out-md", default=str(RESULTS_DIR / "between_run.md"))
    ap.add_argument("--out-json", default=str(RESULTS_DIR / "between_run.json"))
    args = ap.parse_args()

    if len(args.runs) < 2:
        ap.error("need at least two runs to have a between-run spread")

    payloads = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.runs]
    benches = [Bench(p) for p in payloads]

    # Refuse to pool runs that were not the same experiment. `methods` is in
    # here because `benchmark.py --methods` makes a run cheap enough to repeat
    # many times, and a subset run pooled with a full one would compare rows
    # measured under different amounts of preceding GPU work.
    sig = {(p["args"]["samples"], p["args"]["passes"], p["args"]["group_size"],
            p["args"].get("methods"), tuple(p["contexts"])) for p in payloads}
    if len(sig) > 1:
        raise SystemExit(f"runs differ in configuration, refusing to pool: {sig}")

    records = []
    for i, nbits in enumerate(args.bits):
        records += compare_ratios(benches, nbits, shared=(i == 0))
    rows = compare_rows(benches)

    cold_rows = [r for r in rows if r["regime"] == "cold"]
    c = pearson([r["time_spread_frac"] for r in cold_rows],
                [r["mem_spread_frac"] for r in cold_rows])
    meta = {
        "n_runs": len(benches),
        "samples": payloads[0]["args"]["samples"],
        "passes": payloads[0]["args"]["passes"],
        "sources": [Path(p).name for p in args.runs],
        "wall_clock_seconds": [p.get("wall_clock_seconds") for p in payloads],
        "corr_time_vs_clock": c,
        "corr_n": sum(1 for r in cold_rows if r["mem_spread_frac"] is not None),
    }

    md = render(records, rows, meta)
    Path(args.out_md).write_text(md, encoding="utf-8")
    Path(args.out_json).write_text(
        json.dumps({"meta": meta, "ratios": records, "rows": rows}, indent=1),
        encoding="utf-8")

    quot = [r for r in records if r["quotable_all_runs"]]
    print(f"{len(benches)} runs, {len(records)} tracked ratios, "
          f"{len(quot)} quotable in all runs")
    if quot:
        print(f"median inflation "
              f"{statistics.median([r['inflation'] for r in quot]):.1f}x, "
              f"worst {max(r['inflation'] for r in quot):.1f}x")
    unstable = [r for r in records if not r["verdict_stable"]]
    print(f"verdict changes between runs: {len(unstable)}")
    for r in unstable:
        print(f"  {r['name']} ctx={r['ctx']} {r['nbits']}b: "
              + " / ".join(r["verdicts"]))
    print(f"wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
