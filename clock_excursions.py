#!/usr/bin/env python
"""How often does a row drop a memory P-state, and does anything catch it?

`between_run.py` answers "how much does a ratio move between ordinary runs" and
found the answer reassuring: a median 2.4x the printed CI, no verdict moved. What
it explicitly could not answer is the *tail* -- the excursion that produced a
1.27x where four later runs read 1.47x. Three runs bound a body; a rate needs a
denominator.

This script supplies one. Across N runs it finds the median mean memory clock for
every (method, ctx, regime) cell, flags every observation that sits materially
below it, and reports whether the quotability gate rejected that row.
The last column is the one that matters: the gate is built on SM clock and timing
dispersion and says nothing about the memory clock directly, so whether it
happens to catch a P-state drop is an empirical question and not a design claim.

Runs may be tagged into groups (`--label name=glob`), because the reason this
script exists is that the two groups measured here behave differently: a run that
times 3 methods per context rather than 12 is far more excursion-prone than a
full one. Less sustained work precedes each row, and the memory subsystem is
free to sit lower.

Usage:
    ./.venv/Scripts/python.exe clock_excursions.py \
        --label full=results/runs/run1.json,results/runs/run2.json,results/runs/run3.json \
        --label subset=results/tail/validate.json,results/tail/sub2.json,results/tail/sub3.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from audit_claims import Bench

RESULTS_DIR = Path(__file__).parent / "results"

# A memory clock this far below its own cell's median is an excursion
# rather than jitter. The observed P-state steps on this part are ~350-1100 MHz
# out of ~11000, i.e. 3-10%, so 3% sits just under one step.
EXCURSION_FRAC = 0.03

# Only these regimes are bandwidth-bound enough for a memory P-state to move the
# number much; the L2-resident regime is reported too, and the gap between them
# is itself part of the answer.
REGIMES = ("cold", "graph")



# ---------------------------------------------------------------------------
# Does a row's duration predict whether it excurses?
# ---------------------------------------------------------------------------
#
# The fastest kernels in this benchmark excurse most and the slowest never do,
# which suggests a mechanism: a 5 us kernel leaves the memory system idle for a
# larger fraction of its measurement loop than a 1.9 ms one, so the subsystem
# sees gaps and drops a P-state. Pooled over every observation the correlation is
# strong.
#
# It is also exactly the shape that fooled nobody only because this project
# already learned to check: duration is almost perfectly confounded with *which
# kernel it is*, since the fused Triton rows are the short ones and the SDPA rows
# are the long ones. The same within-group decomposition that rescued the
# bandwidth law is applied here, and here it returns a null.


def _log_median(vals):
    return math.log(statistics.median(vals))


def duration_effect(rows: list[dict], n_buckets: int = 6) -> dict:
    """Excursion rate against row duration, pooled and then within each method.

    `rows` are per-observation dicts carrying `method`, `duration_ms` and `is_excursion`.
    """
    usable = [r for r in rows if r.get("duration_ms")]
    if len(usable) < n_buckets * 4:
        return {"n": len(usable), "pooled_r": None}

    ordered = sorted(usable, key=lambda r: r["duration_ms"])
    per = len(ordered) // n_buckets
    buckets = []
    for b in range(n_buckets):
        lo = b * per
        hi = (b + 1) * per if b < n_buckets - 1 else len(ordered)
        sl = ordered[lo:hi]
        buckets.append({
            "min_ms": sl[0]["duration_ms"], "max_ms": sl[-1]["duration_ms"],
            "n": len(sl),
            "rate": sum(1 for r in sl if r["is_excursion"]) / len(sl),
        })
    pooled = _pearson([_log_median([r["duration_ms"] for r in ordered[b * per:(b + 1) * per
                                   if b < n_buckets - 1 else len(ordered)]])
                       for b in range(n_buckets)],
                      [x["rate"] for x in buckets])

    within = []
    for method in sorted({r["method"] for r in usable}):
        sel = [r for r in usable if r["method"] == method]
        cells: dict = {}
        for r in sel:
            cells.setdefault((r["ctx"], r["regime"]), []).append(r)
        if len(cells) < 3:
            continue
        xs = [_log_median([o["duration_ms"] for o in v]) for v in cells.values()]
        ys = [sum(1 for o in v if o["is_excursion"]) / len(v) for v in cells.values()]
        r = _pearson(xs, ys)
        within.append({"method": method, "cells": len(cells), "r": r,
                       "min_ms": min(o["duration_ms"] for o in sel),
                       "max_ms": max(o["duration_ms"] for o in sel)})
    scored = [w for w in within if w["r"] is not None]
    n_neg = sum(1 for w in scored if w["r"] < 0)
    return {
        "n": len(usable),
        "buckets": buckets,
        "pooled_r": pooled,
        "within": within,
        "n_methods_scored": len(scored),
        "n_methods_negative": n_neg,
        # One-sided sign test against "the sign is a coin flip". Only a clean
        # sweep counts, for the same reason as in bandwidth_law.py.
        "sign_test_p": (0.5 ** len(scored) if scored and n_neg == len(scored) else None),
        "survives_within_method": bool(scored and n_neg == len(scored)),
    }


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def render_duration(d: dict) -> list[str]:
    if not d or d.get("pooled_r") is None:
        return []
    L = ["", "## Does a row's duration predict whether it excurses?", "",
         "The fastest kernels here excurse most and the slowest never do. Pooled "
         "over every observation that is a strong relationship; the question is "
         "whether it is about duration or about which kernel it is.", "",
         "| duration bucket (us) | observations | excursion rate |", "|---|---|---|"]
    for b in d["buckets"]:
        L.append(f"| {b['min_ms'] * 1000:.1f} - {b['max_ms'] * 1000:.1f} | "
                 f"{b['n']} | {b['rate'] * 100:.1f}% |")
    L += ["", f"**Pooled r = {d['pooled_r']:+.3f}** over {len(d['buckets'])} buckets.", ""]
    L += ["Holding the method fixed and varying only context and regime:", "",
          "| method | cells | duration range (us) | r |", "|---|---|---|---|"]
    for w in d["within"]:
        r = "n/a" if w["r"] is None else f"{w['r']:+.3f}"
        L.append(f"| `{w['method']}` | {w['cells']} | "
                 f"{w['min_ms'] * 1000:.1f}-{w['max_ms'] * 1000:.0f} | {r} |")
    L.append("")
    if d["survives_within_method"]:
        L.append(f"**{d['n_methods_negative']} of {d['n_methods_scored']}** methods "
                 f"agree in sign (p = {d['sign_test_p']:.3f}), so the relationship is "
                 f"about duration and not about kernel identity.")
    else:
        L.append(f"**{d['n_methods_negative']} of {d['n_methods_scored']}** methods "
                 f"come out negative, which is not a sweep. The pooled correlation "
                 f"does **not** survive holding the method fixed: duration is "
                 f"confounded with which kernel a row is -- the fused Triton rows "
                 f"are the short ones and the SDPA rows the long ones -- and this "
                 f"data cannot separate them. Recorded as a **null**, by the same "
                 f"test that the bandwidth law passed.")
    return L + [""]


def load_groups(specs: list[str]) -> dict[str, list[tuple[str, Bench]]]:
    groups: dict[str, list[tuple[str, Bench]]] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--label wants name=path[,path...], got {spec!r}")
        name, paths = spec.split("=", 1)
        entries = []
        for path in paths.split(","):
            p = Path(path.strip())
            if not p.exists():
                raise SystemExit(f"no such run: {p}")
            entries.append((p.stem, Bench(json.loads(p.read_text(encoding="utf-8")))))
        groups[name] = entries
    return groups


def cells(groups) -> list[tuple[str, int, str]]:
    """(method, ctx, regime) measured by every run in every group.

    Intersected rather than unioned: a cell only one group timed cannot be
    compared across groups, and a baseline clock taken over a mixed population
    would be a different statistic on different cells.
    """
    common = None
    for entries in groups.values():
        for _, b in entries:
            keys = {(m, c) for m, c in b.by}
            common = keys if common is None else (common & keys)
    return [(m, c, r) for (m, c) in sorted(common or [], key=lambda k: (k[0], k[1]))
            for r in REGIMES]


def find_excursions(groups, frac: float = EXCURSION_FRAC):
    """One record per (cell, run) that sits `frac` below the cell's median clock."""
    out = []
    per_cell = {}
    for method, ctx, regime in cells(groups):
        obs = []
        for gname, entries in groups.items():
            for run, b in entries:
                mhz = b.mem_clock(method, ctx, regime)
                if mhz:
                    obs.append((gname, run, mhz, bool(b.quotable(method, ctx))))
        if len(obs) < 3:
            continue
        # Baseline = the median observation, not the mode.
        #
        # The mode is the tempting choice, since P-states are discrete and the
        # mode names the state a cell normally sits in. It is wrong here: on
        # several cells all six observations are distinct, so every count is 1,
        # every value ties for "most common", and whichever tie-break you pick
        # decides the answer. Breaking toward the highest clock reported 4 of 6
        # observations on one cell as excursions -- an "excursion" rate of 67%
        # against a baseline that one single run reached once.
        #
        # The median needs no tie-break, and it bounds the damage: a cell split
        # evenly between two P-states puts the baseline between them, so at
        # most half the observations can be flagged and never the majority.
        # (Half still can be, when the two states are more than 2*frac apart.
        # That is the intended reading -- a cell that sits 10% lower in half its
        # runs is genuinely bimodal, and the table should say so -- but it is
        # worth knowing the statistic does not mean "departs from the usual
        # state" on a cell that has no usual state.)
        vals = [m for _, _, m, _ in obs]
        baseline = statistics.median(vals)
        per_cell[(method, ctx, regime)] = {
            "baseline_mhz": baseline,
            "n_states": len({round(v) for v in vals}),
            "n_obs": len(vals),
        }
        for gname, run, mhz, quotable in obs:
            drop = 1.0 - mhz / baseline
            if drop >= frac:
                out.append({
                    "group": gname, "run": run, "method": method, "ctx": ctx,
                    "regime": regime, "baseline_mhz": baseline, "seen_mhz": mhz,
                    "drop_frac": drop, "row_quotable": quotable,
                })
    return out, per_cell


def all_observations(groups, per_cell, frac: float = EXCURSION_FRAC) -> list[dict]:
    """Every observation, flagged, with the row's duration attached.

    `find_excursions` returns only the flagged ones, which is what its own table
    needs. The duration question needs the denominator too -- the rows that did
    *not* excurse are most of the evidence.
    """
    out = []
    for (method, ctx, regime), cell in per_cell.items():
        baseline = cell["baseline_mhz"]
        for gname, entries in groups.items():
            for run, b in entries:
                mhz = b.mem_clock(method, ctx, regime)
                if not mhz:
                    continue
                row = b.get(method, ctx) or {}
                stats = row.get(regime) or {}
                out.append({
                    "group": gname, "run": run, "method": method, "ctx": ctx,
                    "regime": regime, "mem_mhz": mhz,
                    "duration_ms": stats.get("median_ms"),
                    "is_excursion": (1.0 - mhz / baseline) >= frac,
                })
    return out


def render(exc, per_cell, groups, frac) -> str:
    L = []
    L.append("# Memory P-state excursions, and what catches them\n")
    n_runs = sum(len(v) for v in groups.values())
    multi = sum(1 for v in per_cell.values() if v["n_states"] > 1)
    L.append(f"{n_runs} runs across {len(groups)} group(s), "
             f"{len(per_cell)} comparable (method, ctx, regime) cells. An "
             f"*excursion* is an observation whose mean memory clock sits at "
             f"least {frac:.0%} below the median clock that cell reached over "
             "these runs.\n")
    L.append(f"{multi} of {len(per_cell)} cells visited more than one distinct "
             "memory clock at all; the rest sat in a single state in every "
             "run.\n")

    by_group = defaultdict(list)
    for e in exc:
        by_group[e["group"]].append(e)

    L.append("## Rate, by run group\n")
    L.append("| group | runs | cells | observations | excursions | rate | of those, gate-rejected |")
    L.append("|---|---|---|---|---|---|---|")
    for gname, entries in groups.items():
        n_obs = len(per_cell) * len(entries)
        g = by_group.get(gname, [])
        rej = sum(1 for e in g if not e["row_quotable"])
        L.append(f"| `{gname}` | {len(entries)} | {len(per_cell)} | {n_obs} | "
                 f"{len(g)} | {len(g) / n_obs:.1%} | {rej}/{len(g) if g else 0} |")
    L.append("")

    L.append("## The same split, by regime\n")
    L.append("A memory P-state drop only costs time where the measurement is "
             "bandwidth-bound. Splitting by regime says how much of the rate "
             "above is a threat to a published number and how much is not.\n")
    L.append("| group | regime | excursions | median drop | worst | gate-rejected |")
    L.append("|---|---|---|---|---|---|")
    for gname in groups:
        for reg in REGIMES:
            g = [e for e in by_group.get(gname, []) if e["regime"] == reg]
            if not g:
                L.append(f"| `{gname}` | {reg} | 0 | — | — | — |")
                continue
            drops = [e["drop_frac"] for e in g]
            rej = sum(1 for e in g if not e["row_quotable"])
            L.append(f"| `{gname}` | {reg} | {len(g)} | "
                     f"{statistics.median(drops):.1%} | {max(drops):.1%} | "
                     f"{rej}/{len(g)} |")
    L.append("")

    L.append("## Every excursion\n")
    L.append("| run | group | method | ctx | regime | median MHz | seen | drop | row quotable |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for e in sorted(exc, key=lambda e: -e["drop_frac"]):
        L.append(f"| `{e['run']}` | {e['group']} | `{e['method']}` | {e['ctx']} | "
                 f"{e['regime']} | {e['baseline_mhz']:.0f} | {e['seen_mhz']:.0f} | "
                 f"{e['drop_frac']:.1%} | "
                 f"{'yes' if e['row_quotable'] else '**no — rejected**'} |")
    L.append("")

    rej = sum(1 for e in exc if not e["row_quotable"])
    L.append("## What the gate does and does not do\n")
    L.append(f"{rej} of {len(exc)} excursions landed on a row the quotability "
             "gate rejected. The gate tests the SM clock and the timing's own "
             "dispersion; it never looks at the memory clock, because gating on "
             "memory-clock stability was measured and rejected — it discards "
             "every DRAM-resident row, including rows with a 0.4% timing IQR. So "
             "any excursion it catches, it catches through the *dispersion* the "
             "excursion caused, not through the clock itself.\n")
    L.append("Read that as a bound, not a guarantee: a row that sits steadily in "
             "a lower P-state for its whole window has a tight IQR and passes.\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", action="append", required=True,
                    help="name=path[,path...]; repeat for several groups")
    ap.add_argument("--frac", type=float, default=EXCURSION_FRAC,
                    help="drop below the cell's median clock that counts as an "
                         f"excursion (default {EXCURSION_FRAC})")
    ap.add_argument("--out-md", default=str(RESULTS_DIR / "clock_excursions.md"))
    ap.add_argument("--out-json", default=str(RESULTS_DIR / "clock_excursions.json"))
    args = ap.parse_args()

    groups = load_groups(args.label)
    exc, per_cell = find_excursions(groups, args.frac)
    obs = all_observations(groups, per_cell, args.frac)
    dur = duration_effect(obs)
    md = render(exc, per_cell, groups, args.frac)
    md += "\n" + "\n".join(render_duration(dur))
    Path(args.out_md).write_text(md, encoding="utf-8")
    Path(args.out_json).write_text(json.dumps({
        "frac": args.frac,
        "groups": {k: [n for n, _ in v] for k, v in groups.items()},
        "n_cells": len(per_cell),
        "cells": {f"{m}|{c}|{r}": v for (m, c, r), v in per_cell.items()},
        "excursions": exc,
        "duration_effect": dur,
    }, indent=1), encoding="utf-8")

    for gname, entries in groups.items():
        g = [e for e in exc if e["group"] == gname]
        n_obs = len(per_cell) * len(entries)
        cold = [e for e in g if e["regime"] == "cold"]
        print(f"{gname}: {len(g)}/{n_obs} observations are excursions "
              f"({len(g) / n_obs:.1%}), {len(cold)} of them DRAM-resident")
    rej = sum(1 for e in exc if not e["row_quotable"])
    print(f"gate rejected {rej} of {len(exc)} excursions")
    print(f"wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
