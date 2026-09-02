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


def bandwidth_sensitivity(groups, contexts, methods):
    """Does a row's protocol sensitivity track the bandwidth it actually pulls?

    The predictive form of the finding. A row that pulls little DRAM bandwidth
    cannot be much affected by what state the memory subsystem is in; a row that
    saturates it is entirely at its mercy. If that is the mechanism, |shift|
    should rise with achieved GB/s -- and which rows are protocol-sensitive
    becomes something you can say in advance rather than discover.
    """
    names = list(groups)
    base, other = names[0], names[-1]
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


def render(ratios, telem, groups, bw=None, bw_corr=None, bw_pair=None) -> str:
    names = list(groups)
    base = names[0]
    L = []
    L.append("# Two protocols, compared\n")
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

    if bw:
        a, b = bw_pair
        L.append("## Sensitivity tracks achieved bandwidth\n")
        L.append(f"Each row's DRAM-resident time under `{a}` gives the bandwidth "
                 f"it actually pulls; the shift is `{b}` against `{a}`. A row "
                 "that barely touches DRAM cannot care what state the memory "
                 "subsystem is in; a row that saturates it is entirely at its "
                 "mercy.\n")
        L.append(f"| method | ctx | {a} (µs) | achieved GB/s | shift |")
        L.append("|---|---|---|---|---|")
        for r in sorted(bw, key=lambda r: -r["gb_s"]):
            L.append(f"| `{r['method']}` | {r['ctx']} | {r['base_ms'] * 1e3:.2f} | "
                     f"{r['gb_s']:.0f} | {r['shift']:+.1%} |")
        L.append("")
        if bw_corr is not None:
            L.append(f"**Correlation between achieved bandwidth and |shift|: "
                     f"r = {bw_corr:+.2f}** over {len(bw)} rows.\n")
            if bw_corr > 0.6:
                L.append("That makes the finding predictive rather than "
                         "descriptive: the rows a change of protocol will move "
                         "are the rows pulling the most bandwidth, and they can "
                         "be named in advance. It also says which *ratios* are "
                         "exposed — any ratio dividing a high-bandwidth row by a "
                         "low-bandwidth one inherits the difference.\n")
    return "\n".join(L)


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
    bw, bw_corr, bw_pair = bandwidth_sensitivity(groups, contexts, methods)
    md = render(ratios, telem, groups, bw, bw_corr, bw_pair)
    Path(args.out_md).write_text(md, encoding="utf-8")
    Path(args.out_json).write_text(json.dumps({
        "groups": {k: [n for n, _ in v] for k, v in groups.items()},
        "contexts": contexts, "methods": methods,
        "ratios": ratios, "telemetry": telem,
        "bandwidth_sensitivity": {"rows": bw, "corr": bw_corr,
                                  "pair": list(bw_pair) if bw_pair else None},
    }, indent=1), encoding="utf-8")

    disj = [r for r in ratios if any((r.get("disjoint_from_base") or {}).values())]
    print(f"{len(ratios)} ratios over {len(groups)} protocols; "
          f"{len(disj)} disjoint from `{list(groups)[0]}`")
    for r in disj:
        gs = ", ".join(g for g, v in r["disjoint_from_base"].items() if v)
        base = r["groups"][list(groups)[0]]
        print(f"  {r['name']} ctx={r['ctx']}: {base['min']:.3f}-{base['max']:.3f} "
              f"vs {gs}")
    if bw_corr is not None:
        print(f"bandwidth vs |protocol shift|: r = {bw_corr:+.2f} over {len(bw)} rows")
    print(f"wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
