#!/usr/bin/env python
"""Do two measurement protocols produce the same numbers?

`between_run.py` asks how much a ratio moves between repetitions of *one*
protocol. This asks the prior question: whether two protocols are measuring the
same thing at all. They are different questions and conflating them is how a
shortcut gets adopted -- a cheap protocol whose runs agree tightly with each
other can still disagree with the expensive one it replaced.

The test is deliberately blunt. For each ratio, take each group's range over its
own runs, and ask whether the ranges overlap. Overlap is not proof of agreement,
but *disjointness* on a ratio whose within-group spread is 0.6% is proof of
disagreement, and that is the finding worth acting on. No p-values: with three
runs per group the honest statement is "these ranges do not touch", not a test
that pretends to more resolution than three points support.

It also reports, for every row, whether the monitored variables agree -- SM
clock, memory clock, power, temperature, sample count. A protocol difference that
shows up in the timings *and* in the telemetry has an explanation available. One
that shows up only in the timings does not, and that gap is worth naming rather
than absorbing into "noise".

Usage:
    ./.venv/Scripts/python.exe compare_protocols.py \
        --label full=results/runs/run1.json,results/runs/run2.json,results/runs/run3.json \
        --label subset=results/tail/sub1.json,results/tail/sub2.json,results/tail/sub3.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from audit_claims import CONTROL, Bench, bootstrap_ratio_ci
from clock_excursions import load_groups

RESULTS_DIR = Path(__file__).parent / "results"

# (name, numerator, denominator, regime). Restricted to the attribution chain,
# because a filtered run only has those rows and this script exists to compare
# a filtered protocol against a full one.
RATIOS = (
    ("quant_cold", CONTROL, "fused_triton_4b", "cold"),
    ("quant_hot", CONTROL, "fused_triton_4b", "graph"),
    ("split_only", "fp16_sdpa", CONTROL, "cold"),
    ("speedup_vs_sdpa", "fp16_sdpa", "fused_triton_4b", "cold"),
)

# Telemetry compared per row, as (label, path into the clock summary).
TELEMETRY = (
    ("SM MHz", "sm_mhz_mean"),
    ("mem MHz", "mem_mhz_mean"),
    ("power W", "power_w_mean"),
    ("temp C", "temp_c_mean"),
    ("samples", "n_samples"),
)


def _samples(b: Bench, method: str, ctx: int, regime: str):
    return b.cold(method, ctx) if regime == "cold" else b.hot_raw(method, ctx)


def ratio_ranges(groups, contexts):
    """Per (ratio, ctx): each group's min/max point estimate over its runs."""
    out = []
    for name, num, den, regime in RATIOS:
        for ctx in contexts:
            per_group = {}
            ok = True
            for gname, entries in groups.items():
                vals = []
                for _, b in entries:
                    x, y = _samples(b, num, ctx, regime), _samples(b, den, ctx, regime)
                    if not (x and y):
                        ok = False
                        break
                    vals.append(bootstrap_ratio_ci(x, y)[0])
                if not ok:
                    break
                per_group[gname] = vals
            if not ok or len(per_group) < 2:
                continue
            names = list(per_group)
            base = names[0]
            rec = {
                "name": name, "ctx": ctx, "regime": regime,
                "groups": {g: {"min": min(v), "max": max(v),
                               "median": statistics.median(v), "values": v}
                           for g, v in per_group.items()},
            }
            # Disjoint from the first group, which is the reference protocol.
            for g in names[1:]:
                a, c = rec["groups"][base], rec["groups"][g]
                rec.setdefault("disjoint_from_base", {})[g] = (
                    a["max"] < c["min"] or c["max"] < a["min"])
                rec.setdefault("shift_from_base", {})[g] = (
                    c["median"] / a["median"] - 1.0) if a["median"] else None
            out.append(rec)
    return out


def telemetry_agreement(groups, contexts, methods):
    """Per row: how far apart the groups' monitored variables are.

    The point of this table is the rows where everything here agrees and the
    timings do not.
    """
    out = []
    base = list(groups)[0]
    for method in methods:
        for ctx in contexts:
            for regime in ("cold", "graph"):
                rec = {"method": method, "ctx": ctx, "regime": regime, "telemetry": {}}
                med = {}
                ok = True
                for gname, entries in groups.items():
                    vals = []
                    for _, b in entries:
                        row = b.get(method, ctx) or {}
                        clk = ((row.get("clocks") or {}).get(regime) or {}).get("clocks")
                        st = row.get(regime)
                        if not clk or not isinstance(st, dict):
                            ok = False
                            break
                        vals.append((clk, st["median_ms"] * 1e3))
                    if not ok:
                        break
                    med[gname] = vals
                if not ok or len(med) < 2:
                    continue
                for label, key in TELEMETRY:
                    per_g = {g: statistics.median([c[key] for c, _ in v])
                             for g, v in med.items() if all(key in c for c, _ in v)}
                    if len(per_g) < 2:
                        continue
                    lo, hi = min(per_g.values()), max(per_g.values())
                    rec["telemetry"][label] = {
                        "per_group": per_g,
                        "spread_frac": (hi / lo - 1.0) if lo else None,
                    }
                times = {g: statistics.median([t for _, t in v]) for g, v in med.items()}
                rec["median_us"] = times
                rec["time_shift_from_base"] = {
                    g: (t / times[base] - 1.0) for g, t in times.items() if g != base}
                out.append(rec)
    return out


def bandwidth_sensitivity(groups, contexts, methods, other=None):
    """Does a row's protocol sensitivity track the bandwidth it actually pulls?

    The predictive form of the finding. A row that pulls little DRAM bandwidth
    cannot be much affected by what state the memory subsystem is in; a row that
    saturates it is entirely at its mercy. If that is the mechanism, |shift|
    should rise with achieved GB/s -- and which rows are protocol-sensitive
    becomes something you can say in advance rather than discover.

    ``other`` names the group compared against the reference. It used to be
    implicitly the *last* group passed, which meant adding a fourth protocol on
    the command line silently repointed an already-published correlation at a
    different pair of runs. It is now explicit, and the caller computes one
    correlation per non-reference group.
    """
    names = list(groups)
    base = names[0]
    other = other or names[-1]
    rows = []
    for method in methods:
        for ctx in contexts:
            per = {}
            ok = True
            for gname in (base, other):
                vals = []
                for _, b in groups[gname]:
                    r = b.get(method, ctx) or {}
                    st = r.get("cold")
                    if not isinstance(st, dict) or "median_ms" not in st:
                        ok = False
                        break
                    vals.append((st["median_ms"], r.get("cache_bytes_1layer") or 0))
                if not ok:
                    break
                per[gname] = vals
            if not ok:
                continue
            f = statistics.median([t for t, _ in per[base]])
            p = statistics.median([t for t, _ in per[other]])
            nbytes = per[base][0][1]
            if not (f > 0 and nbytes):
                continue
            rows.append({
                "method": method, "ctx": ctx,
                "base_ms": f, "other_ms": p,
                "gb_s": nbytes / (f * 1e-3) / 1e9,
                "shift": p / f - 1.0,
            })
    xs = [r["gb_s"] for r in rows]
    ys = [abs(r["shift"]) for r in rows]
    corr = None
    if len(rows) >= 3:
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
        dy = math.sqrt(sum((b - my) ** 2 for b in ys))
        corr = num / (dx * dy) if dx > 0 and dy > 0 else None
    return rows, corr, (base, other)


# ---------------------------------------------------------------------------
# The 2x2: separating run length from recent saturation
# ---------------------------------------------------------------------------
#
# Three protocols leave the two candidate explanations confounded. `full` times
# 12 methods and takes ~800 s; `subset` times 3 and takes ~205 s; `preloaded`
# times 3 after 300 s of ramp workload. Every protocol with more methods is also
# a longer run, so "the card is in a different state because 800 s of work
# preceded this row" and "the card is in a different state because a lot of
# bandwidth was pulled recently" cannot be told apart. The missing cell is
# 12 methods *with* the preload, and adding it makes the design a 2x2.


def design_coords(b) -> tuple[str, float]:
    """The protocol coordinates a run recorded about itself.

    Read out of the run's own ``args`` rather than off its ``--label``, because
    the label is a name this script was handed on the command line and the args
    are what the benchmark actually did. A mislabelled run is then a detected
    error instead of a silently wrong cell.
    """
    a = b.p.get("args") or {}
    return (a.get("methods") or "all", float(a.get("preload") or 0.0))


def design_cells(groups):
    """Locate the labelled groups on the (method set x preload) grid.

    Returns ``(cells, levels, note)``. ``cells`` maps a coordinate pair to the
    label sitting there, ``levels`` is ``(method_sets, preloads)``. On anything
    that is not a complete 2x2 it returns ``(None, None, why)`` -- the factorial
    section is then skipped with the reason printed, rather than computed on a
    design that cannot support it.
    """
    coords, mixed = {}, []
    for g, entries in groups.items():
        cs = {design_coords(b) for _, b in entries}
        if len(cs) != 1:
            mixed.append(g)
            continue
        coords[g] = cs.pop()
    if mixed:
        return None, None, ("groups whose runs do not share one protocol: "
                            + ", ".join(mixed))
    cells = {}
    for g, c in coords.items():
        if c in cells:
            return None, None, (f"`{g}` and `{cells[c]}` are the same protocol "
                                f"{c}; a 2x2 needs four distinct cells")
        cells[c] = g
    # Ordered by how much each level times, not alphabetically: the whole
    # point is that ms[0] is the small method set and ms[1] the full one,
    # and "all" < "attribution" in string order would silently swap them.
    ms = sorted({m for m, _ in cells}, key=lambda m: (m == "all", m))
    ps = sorted({p for _, p in cells})
    if len(ms) != 2 or len(ps) != 2:
        return None, None, (f"design is {len(ms)}x{len(ps)}, not 2x2 "
                            f"(method sets {ms}, preloads {ps})")
    missing = [(m, p) for m in ms for p in ps if (m, p) not in cells]
    if missing:
        return None, None, ("incomplete 2x2; no runs at "
                            + ", ".join(str(c) for c in missing))
    return cells, (ms, ps), None


def factorial_effects(per_cell, levels):
    """Main effects, simple effects and the interaction, in log space.

    Log space because everything here is a ratio or a time and the questions are
    multiplicative ("does the preload cost 2%"). Differences of logs is also the
    only scale on which "the two factors simply add" is a well-posed claim at
    all, so the interaction term measures something rather than recording that
    the axis was chosen badly.

    There is no p-value, for the same reason as the rest of this file: three
    runs per cell support "this effect is larger than anything a cell varies by
    internally" and do not support a test claiming finer resolution. The largest
    within-cell range travels with every effect as exactly that yardstick.
    """
    ms, ps = levels
    m0, m1 = ms
    p0, p1 = ps
    lg = {c: math.log(statistics.median(v)) for c, v in per_cell.items()}
    simple = {
        "preload_at_few": lg[(m0, p1)] - lg[(m0, p0)],
        "preload_at_many": lg[(m1, p1)] - lg[(m1, p0)],
        "methods_at_none": lg[(m1, p0)] - lg[(m0, p0)],
        "methods_at_preload": lg[(m1, p1)] - lg[(m0, p1)],
    }
    main_preload = 0.5 * (simple["preload_at_few"] + simple["preload_at_many"])
    main_methods = 0.5 * (simple["methods_at_none"] + simple["methods_at_preload"])
    interaction = simple["preload_at_many"] - simple["preload_at_few"]
    spreads = {c: (max(v) / min(v) - 1.0)
               for c, v in per_cell.items() if min(v) > 0}
    noise = max(spreads.values()) if spreads else None
    eff = {
        "main_preload": math.expm1(main_preload),
        "main_methods": math.expm1(main_methods),
        "interaction": math.expm1(interaction),
    }
    return {
        "simple": {k: math.expm1(v) for k, v in simple.items()},
        **eff,
        "within_cell_spread": {f"{m}|{pp:g}": v for (m, pp), v in spreads.items()},
        "noise": noise,
        "resolved": {k: (abs(v) > noise) if noise is not None else None
                     for k, v in eff.items()},
        "medians": {f"{m}|{pp:g}": statistics.median(v)
                    for (m, pp), v in per_cell.items()},
    }


def factorial_ratios(groups, cells, levels, contexts):
    """The 2x2 decomposition of every attribution ratio."""
    out = []
    for name, num, den, regime in RATIOS:
        for ctx in contexts:
            per_cell, ok = {}, True
            for coord, gname in cells.items():
                vals = []
                for _, b in groups[gname]:
                    x = _samples(b, num, ctx, regime)
                    y = _samples(b, den, ctx, regime)
                    if not (x and y):
                        ok = False
                        break
                    vals.append(bootstrap_ratio_ci(x, y)[0])
                if not ok:
                    break
                per_cell[coord] = vals
            if not ok or len(per_cell) != 4:
                continue
            rec = factorial_effects(per_cell, levels)
            rec.update({"name": name, "ctx": ctx, "regime": regime})
            out.append(rec)
    return out


def factorial_rows(groups, cells, levels, contexts, methods):
    """The same decomposition on raw per-row times, plus each row's bandwidth.

    The ratios are what the repo quotes, but a ratio inherits the behaviour of
    two rows and cannot say which of them moved. This is the per-row view, and
    it carries achieved DRAM bandwidth so the preload effect can be tested
    against the bandwidth law directly instead of through a quotient.
    """
    ms, ps = levels
    ref = (ms[1], ps[0])  # many methods, no preload: the shipped protocol
    out = []
    for method in methods:
        for ctx in contexts:
            for regime in ("cold", "graph"):
                per_cell, ok, nbytes = {}, True, 0
                for coord, gname in cells.items():
                    vals = []
                    for _, b in groups[gname]:
                        r = b.get(method, ctx) or {}
                        st = r.get(regime)
                        if not isinstance(st, dict) or "median_ms" not in st:
                            ok = False
                            break
                        vals.append(st["median_ms"])
                        nbytes = nbytes or (r.get("cache_bytes_1layer") or 0)
                    if not ok:
                        break
                    per_cell[coord] = vals
                if not ok or len(per_cell) != 4:
                    continue
                rec = factorial_effects(per_cell, levels)
                ref_ms = statistics.median(per_cell[ref])
                rec.update({
                    "method": method, "ctx": ctx, "regime": regime,
                    "ref_ms": ref_ms,
                    "gb_s": (nbytes / (ref_ms * 1e-3) / 1e9)
                            if (regime == "cold" and ref_ms > 0 and nbytes) else None,
                })
                out.append(rec)
    return out


def corr_of(pairs):
    """Pearson r over (x, y) pairs, and the n it was computed on."""
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return (num / (dx * dy) if dx > 0 and dy > 0 else None), len(pairs)


def render(ratios, telem, groups, bw_by_group=None, fac=None,
           fac_rows=None, fac_note=None, design=None, levels=None) -> str:
    """`design` is the 2x2 grid, not to be confused with the local `cells`
    used for table rows below -- which is what the parameter was first
    called, and it silently shadowed it."""
    names = list(groups)
    base = names[0]
    L = []
    L.append(f"# {len(names)} protocols, compared\n")
    L.append("Reference protocol: **`" + base + "`**. A ratio is flagged when a "
             "group's range over its own runs does not touch the reference "
             "group's range at all.\n")
    for g, entries in groups.items():
        L.append(f"- `{g}`: {len(entries)} runs — " + ", ".join(n for n, _ in entries))
    L.append("")

    disj = [r for r in ratios if any((r.get("disjoint_from_base") or {}).values())]
    L.append("## Verdict\n")
    L.append(f"**{len(disj)} of {len(ratios)} ratios have a group whose range "
             f"misses `{base}`'s entirely.**"
             + (" The protocols are not interchangeable.\n" if disj
                else " Nothing here shows the protocols disagreeing.\n"))

    L.append("## Ratios\n")
    hdr = ["ratio", "ctx"] + [f"`{g}`" for g in names] + ["shift", "disjoint"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for r in ratios:
        cells = [f"`{r['name']}`", str(r["ctx"])]
        for g in names:
            d = r["groups"][g]
            cells.append(f"{d['min']:.3f}–{d['max']:.3f}")
        sh = r.get("shift_from_base") or {}
        cells.append(", ".join(f"{v:+.1%}" for v in sh.values() if v is not None) or "—")
        dj = [g for g, v in (r.get("disjoint_from_base") or {}).items() if v]
        cells.append("**" + ", ".join(dj) + "**" if dj else "")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Where the timings move and the telemetry does not\n")
    L.append("Rows whose median time differs from the reference by more than 1% "
             "while every monitored variable agrees to within 1%. These are the "
             "ones with no available explanation.\n")
    hdr2 = ["method", "ctx", "regime", "time shift"] + [lbl for lbl, _ in TELEMETRY]
    L.append("| " + " | ".join(hdr2) + " |")
    L.append("|" + "---|" * len(hdr2))
    n_unexplained = 0
    for t in telem:
        shifts = [abs(v) for v in t["time_shift_from_base"].values()]
        if not shifts or max(shifts) < 0.01:
            continue
        spreads = {lbl: (t["telemetry"].get(lbl) or {}).get("spread_frac")
                   for lbl, _ in TELEMETRY}
        quiet = all(s is not None and s < 0.01 for s in spreads.values())
        if not quiet:
            continue
        n_unexplained += 1
        cells = [f"`{t['method']}`", str(t["ctx"]), t["regime"],
                 ", ".join(f"{v:+.1%}" for v in t["time_shift_from_base"].values())]
        cells += [f"{spreads[lbl]:.1%}" for lbl, _ in TELEMETRY]
        L.append("| " + " | ".join(cells) + " |")
    if not n_unexplained:
        L.append("| — | | | *(none: every timing shift over 1% came with a "
                 "telemetry difference)* | | | | | |")
    L.append("")
    L.append(f"{n_unexplained} such rows. A protocol difference visible in the "
             "timings and invisible in the telemetry is not noise — it is a "
             "channel the instrumentation does not cover, and it should be "
             "named as one.\n")

    for label, (rows, rcorr) in (bw_by_group or {}).items():
        L.append(f"## Sensitivity tracks achieved bandwidth — `{label}`\n")
        L.append(f"Each row's DRAM-resident time under `{base}` gives the "
                 f"bandwidth it actually pulls; the shift is `{label}` against "
                 f"`{base}`. A row that barely touches DRAM cannot care what "
                 "state the memory subsystem is in; a row that saturates it is "
                 "entirely at its mercy.\n")
        L.append(f"| method | ctx | {base} (µs) | achieved GB/s | shift |")
        L.append("|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: -r["gb_s"]):
            L.append(f"| `{r['method']}` | {r['ctx']} | {r['base_ms'] * 1e3:.2f} | "
                     f"{r['gb_s']:.0f} | {r['shift']:+.1%} |")
        L.append("")
        if rcorr is not None:
            L.append(f"**Correlation between achieved bandwidth and |shift|: "
                     f"r = {rcorr:+.2f}** over {len(rows)} rows.\n")
            if rcorr > 0.6:
                L.append("That makes the finding predictive rather than "
                         "descriptive: the rows a change of protocol will move "
                         "are the rows pulling the most bandwidth, and they can "
                         "be named in advance. It also says which *ratios* are "
                         "exposed — any ratio dividing a high-bandwidth row by a "
                         "low-bandwidth one inherits the difference.\n")

    L += _render_factorial(fac, fac_rows, fac_note, design, levels)
    return "\n".join(L)



def _render_factorial(fac, fac_rows, note, cells, levels) -> list[str]:
    """The 2x2 section: which of the two confounded factors actually moved."""
    L = ["## The 2x2: run length or recent saturation?\n"]
    if not fac:
        L.append("*Not computed: " + (note or "no complete 2x2 in the runs "
                                      "supplied") + ".*\n")
        return L
    ms, ps = levels
    grid = {c: cells[c] for c in cells}
    L.append("Two things vary together across the three original protocols: how "
             "many methods a run times (and so how long it is) and whether the "
             "memory system was recently saturated. A fourth protocol — the full "
             "method set *with* the preload — makes the design complete, so the "
             "two can be separated instead of argued about.\n")
    L.append("| | no preload | preload |")
    L.append("|---|---|---|")
    for m in ms:
        label = "12 methods" if m == "all" else f"`{m}` only"
        L.append(f"| **{label}** | `{grid[(m, ps[0])]}` | `{grid[(m, ps[1])]}` |")
    L.append("")
    L.append("Effects are differences of logs, reported as percentages. "
             "**Main preload** is the average effect of adding the preload; "
             "**main methods** the average effect of timing the full method set; "
             "**interaction** is how much the preload's effect differs between "
             "the two method sets — zero would mean the two factors simply add. "
             "Each is called *resolved* only when it exceeds the largest range "
             "any one cell shows across its own three runs, which is the column "
             "marked *cell noise*.\n")

    hdr = ["ratio", "ctx", "main preload", "main methods", "interaction",
           "cell noise", "resolved"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for r in fac:
        res = [k.replace("main_", "").replace("_", " ")
               for k, v in (r["resolved"] or {}).items() if v]
        L.append("| `{}` | {} | {:+.1%} | {:+.1%} | {:+.1%} | {:.1%} | {} |".format(
            r["name"], r["ctx"], r["main_preload"], r["main_methods"],
            r["interaction"], r["noise"] or 0.0,
            "**" + ", ".join(res) + "**" if res else "—"))
    L.append("")

    L.append("### Simple effects\n")
    L.append("The same numbers split by level, which is what maps onto the "
             "hypotheses: if run length is the channel the preload should do "
             "little at either method count; if recent saturation is, the "
             "preload should do the same thing at both.\n")
    hdr2 = ["ratio", "ctx", "preload @ few", "preload @ 12", "methods @ none",
            "methods @ preload"]
    L.append("| " + " | ".join(hdr2) + " |")
    L.append("|" + "---|" * len(hdr2))
    for r in fac:
        s = r["simple"]
        L.append("| `{}` | {} | {:+.1%} | {:+.1%} | {:+.1%} | {:+.1%} |".format(
            r["name"], r["ctx"], s["preload_at_few"], s["preload_at_many"],
            s["methods_at_none"], s["methods_at_preload"]))
    L.append("")

    if fac_rows:
        cold = [r for r in fac_rows if r["regime"] == "cold" and r.get("gb_s")]
        r_pre, n_pre = corr_of([(r["gb_s"], abs(r["main_preload"])) for r in cold])
        r_met, n_met = corr_of([(r["gb_s"], abs(r["main_methods"])) for r in cold])
        L.append("### Which factor the bandwidth law is about\n")
        L.append("The bandwidth law was measured against a protocol pair that "
                 "changed *both* factors at once, so it could not say which one "
                 "it described. Correlating each main effect separately against "
                 "achieved bandwidth answers that.\n")
        L.append("| effect | r vs achieved GB/s | rows |")
        L.append("|---|---|---|")
        L.append(f"| preload | {r_pre:+.2f} | {n_pre} |" if r_pre is not None
                 else "| preload | — | |")
        L.append(f"| methods | {r_met:+.2f} | {n_met} |" if r_met is not None
                 else "| methods | — | |")
        L.append("")
        L.append("| method | ctx | GB/s | main preload | main methods | interaction |")
        L.append("|---|---|---|---|---|---|")
        for r in sorted(cold, key=lambda r: -r["gb_s"]):
            L.append("| `{}` | {} | {:.0f} | {:+.1%} | {:+.1%} | {:+.1%} |".format(
                r["method"], r["ctx"], r["gb_s"], r["main_preload"],
                r["main_methods"], r["interaction"]))
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", action="append", required=True,
                    help="name=path[,path...]; the first --label is the reference")
    ap.add_argument("--out-md", default=str(RESULTS_DIR / "compare_protocols.md"))
    ap.add_argument("--out-json", default=str(RESULTS_DIR / "compare_protocols.json"))
    args = ap.parse_args()

    groups = load_groups(args.label)
    if len(groups) < 2:
        ap.error("need at least two --label groups to compare")

    # Contexts and methods every group measured.
    contexts, methods = None, None
    for entries in groups.values():
        for _, b in entries:
            cs, ms = set(b.contexts), {m for m, _ in b.by}
            contexts = cs if contexts is None else (contexts & cs)
            methods = ms if methods is None else (methods & ms)
    contexts, methods = sorted(contexts or []), sorted(methods or [])

    ratios = ratio_ranges(groups, contexts)
    telem = telemetry_agreement(groups, contexts, methods)

    # One bandwidth correlation per non-reference protocol. A single one taken
    # against "whichever group came last on the command line" changed meaning
    # the moment a fourth protocol was added, which is how a published r quietly
    # stops describing the pair it was published about.
    bw_by_group = {}
    for other in list(groups)[1:]:
        rows, corr, _pair = bandwidth_sensitivity(groups, contexts, methods, other)
        if rows:
            bw_by_group[other] = (rows, corr)

    cells, levels, fac_note = design_cells(groups)
    fac = factorial_ratios(groups, cells, levels, contexts) if cells else []
    fac_rows = factorial_rows(groups, cells, levels, contexts, methods) if cells else []

    md = render(ratios, telem, groups, bw_by_group, fac, fac_rows, fac_note,
                cells, levels)
    Path(args.out_md).write_text(md, encoding="utf-8")
    Path(args.out_json).write_text(json.dumps({
        "groups": {k: [n for n, _ in v] for k, v in groups.items()},
        "contexts": contexts, "methods": methods,
        "ratios": ratios, "telemetry": telem,
        "bandwidth_sensitivity": {
            g: {"rows": rows, "corr": corr, "pair": [list(groups)[0], g]}
            for g, (rows, corr) in bw_by_group.items()},
        "design": {
            "note": fac_note,
            "cells": ({f"{m}|{p:g}": g for (m, p), g in cells.items()}
                      if cells else None),
            "levels": [list(levels[0]), list(levels[1])] if levels else None,
        },
        "factorial_ratios": fac,
        "factorial_rows": fac_rows,
    }, indent=1), encoding="utf-8")

    disj = [r for r in ratios if any((r.get("disjoint_from_base") or {}).values())]
    print(f"{len(ratios)} ratios over {len(groups)} protocols; "
          f"{len(disj)} disjoint from `{list(groups)[0]}`")
    for r in disj:
        gs = ", ".join(g for g, v in r["disjoint_from_base"].items() if v)
        base = r["groups"][list(groups)[0]]
        print(f"  {r['name']} ctx={r['ctx']}: {base['min']:.3f}-{base['max']:.3f} "
              f"vs {gs}")
    for g, (rows, corr) in bw_by_group.items():
        if corr is not None:
            print(f"bandwidth vs |shift| ({g} vs {list(groups)[0]}): "
                  f"r = {corr:+.2f} over {len(rows)} rows")
    if cells:
        print(f"2x2 design complete: {len(fac)} ratios decomposed")
        for r in fac:
            res = ", ".join(k for k, v in r["resolved"].items() if v) or "nothing"
            print(f"  {r['name']}@{r['ctx']}: preload {r['main_preload']:+.1%}, "
                  f"methods {r['main_methods']:+.1%}, "
                  f"interaction {r['interaction']:+.1%} "
                  f"(cell noise {r['noise']:.1%}; resolved: {res})")
    else:
        print(f"no 2x2: {fac_note}")
    print(f"wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
