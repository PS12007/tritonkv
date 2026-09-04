#!/usr/bin/env python
"""A second reporting tier: gate-failed, but the median is pinned anyway.

    ./.venv/Scripts/python.exe dispersion_tier.py     # reads results/benchmark.json

`analyze_dispersion.py` established the fact this script acts on. The quotability
gate rejects a row when its per-sample IQR exceeds 5% of its median -- but IQR is
a property of the *sample distribution*, not of the *estimate*, and those two come
apart badly on the L2-resident rows, which are tens of microseconds long and
jitter freely around a median that barely moves. On run 3, eight of the ten
rejected measurements pin their medians to +-0.05-1.31% while the gate is
rejecting them for a 5.6-11.5% IQR.

The temptation is to widen `MAX_IQR_FRAC`. That is the move this project has
refused three times, and refuses again here: widening the gate would admit the
two rejected measurements that are *genuinely* badly pinned (+-2.33% and
+-2.68%) along with the eight that are not. Dispersion and precision are
different questions, and the fix is to ask the second one separately rather than
to loosen the first.

So: a third verdict rather than a wider gate.

* **tier 1, quotable** -- what the gate already certifies. Untouched. A starred
  row means exactly what it meant before this file existed.
* **tier 2, pinned** -- clock-verified, failed the IQR gate, but every regime's
  median is pinned at least as well as the worst number the gate already
  accepts. Usable with an explicit qualifier, never a star.
* **tier 3, rejected** -- everything else.

**The bar is not a new free parameter.** It is read off the instrument's own
accepted behaviour: the worst median CI halfwidth among the measurements inside
rows the gate certifies as quotable. On run 3 that is +-1.700%
(`fused_gather_meta_4b@512`, DRAM-resident, IQR 3.9% -- comfortably inside the
gate). A tier-2 row is therefore, by construction, one whose headline number is
pinned *at least as well as something this repo already prints with a star*. It
is impossible for this tier to admit a number less certain than one the gate
blesses, and that is the property which makes it a report rather than a loophole.

Two deliberate restrictions:

* **A clock-rejected row is never eligible.** The gate is not a P-state filter
  (it tests SM clock and timing dispersion, never the memory clock), so a row
  that failed on clocks failed for a reason this tier cannot see and does not
  address.
* **Tier 2 is admissible per claim, not in general.** A row pinned to +-1.3% has
  no business supporting a 2% effect. Each tier-2 row carries `min_effect_frac`
  -- `EFFECT_MULTIPLE` times its own median CI halfwidth -- and a consumer is
  expected to check the effect it is claiming against it. The attribution rows
  clear this by two orders of magnitude (+-0.06/0.11/0.37% against effects of
  20-30%), which is the whole reason this tier is worth having.

Nothing here re-measures anything and nothing here touches `benchmark.py`. The
raw per-sample timings are already in the results JSON, so the tier applies
retroactively to every run ever recorded -- including all four protocols -- and
the instrument that produced them is left exactly as it was. Changing the
instrument while protocols were in flight was the confound under test; this
avoids it by construction rather than by timing.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analyze_dispersion import MAX_IQR_FRAC, REGIMES, describe

RESULTS_DIR = Path(__file__).parent / "results"

# How much larger than a row's own median uncertainty an effect must be before
# that row may be quoted in support of it. 5x is a deliberately blunt bar: the
# rows this tier exists for clear it by 50-500x, so nothing here turns on the
# exact multiple, and a value that had to be tuned to admit the rows it was
# written for would be the thing this file is arguing against.
EFFECT_MULTIPLE = 5.0

TIER_QUOTABLE = 1
TIER_PINNED = 2
TIER_REJECTED = 3

TIER_NAMES = {
    TIER_QUOTABLE: "quotable",
    TIER_PINNED: "pinned",
    TIER_REJECTED: "rejected",
}


def measure_row(row: dict) -> dict:
    """Per-regime dispersion and median precision for one benchmark row."""
    out = {}
    for regime_key, raw_key, regime_label in REGIMES:
        xs = row.get(raw_key)
        if not xs:
            continue
        d = describe(xs)
        if d is None:
            continue
        out[regime_key] = {
            "regime": regime_label,
            "iqr_frac": d["iqr_frac"],
            "median_ci_halfwidth_frac": d["median_ci_halfwidth_frac"],
            "tight": d["iqr_frac"] <= MAX_IQR_FRAC,
        }
    return out


def calibration_bar(rows: list[dict], per_row: dict) -> dict:
    """The worst median CI halfwidth the gate already accepts.

    Restricted to measurements inside rows the gate certifies as *quotable* --
    not merely to measurements whose own IQR happens to pass -- because the
    claim being made is "at least as well pinned as something this repo prints
    with a star", and a star is a row-level property that includes the clocks.
    """
    accepted = []
    for row in rows:
        if not row.get("quotable"):
            continue
        for m in per_row.get((row["method"], row["ctx"]), {}).values():
            if m["median_ci_halfwidth_frac"] is not None:
                accepted.append((m["median_ci_halfwidth_frac"], row["method"],
                                 row["ctx"], m["regime"], m["iqr_frac"]))
    if not accepted:
        # No starred row in this run means there is nothing to calibrate
        # against, so the tier declines to exist rather than inventing a bar.
        return {"bar_frac": None, "n_accepted": 0, "worst": None,
                "median_accepted_frac": None}
    accepted.sort()
    frac, method, ctx, regime, iqr = accepted[-1]
    return {
        "bar_frac": frac,
        "n_accepted": len(accepted),
        "median_accepted_frac": statistics.median(a[0] for a in accepted),
        "worst": {"method": method, "ctx": ctx, "regime": regime,
                  "median_ci_halfwidth_frac": frac, "iqr_frac": iqr},
    }


def assign_tier(row: dict, measurements: dict, bar_frac: float | None) -> dict:
    """Sort one row into a tier, and say why in the terms of the decision."""
    clock_ok = bool(row.get("clock_verified"))
    loose = [m for m in measurements.values() if not m["tight"]]
    pins = [m["median_ci_halfwidth_frac"] for m in measurements.values()
            if m["median_ci_halfwidth_frac"] is not None]
    worst_pin = max(pins) if pins else None

    rec = {
        "method": row["method"],
        "ctx": row["ctx"],
        "quotable": bool(row.get("quotable")),
        "clock_verified": clock_ok,
        "n_regimes": len(measurements),
        "n_loose_regimes": len(loose),
        "worst_iqr_frac": (max(m["iqr_frac"] for m in measurements.values())
                           if measurements else None),
        "worst_median_ci_halfwidth_frac": worst_pin,
        "regimes": dict(measurements),
    }

    if row.get("quotable"):
        rec.update(tier=TIER_QUOTABLE, tier_name=TIER_NAMES[TIER_QUOTABLE],
                   reason="passes the gate")
    elif not clock_ok:
        # Deliberately checked before the pin: a row whose clocks were not
        # verified failed for a reason this tier says nothing about.
        rec.update(tier=TIER_REJECTED, tier_name=TIER_NAMES[TIER_REJECTED],
                   reason="clock-rejected (not a dispersion question)")
    elif not measurements or worst_pin is None:
        rec.update(tier=TIER_REJECTED, tier_name=TIER_NAMES[TIER_REJECTED],
                   reason="no usable raw samples")
    elif bar_frac is None:
        rec.update(tier=TIER_REJECTED, tier_name=TIER_NAMES[TIER_REJECTED],
                   reason="no quotable row in this run to calibrate against")
    elif worst_pin <= bar_frac:
        rec.update(tier=TIER_PINNED, tier_name=TIER_NAMES[TIER_PINNED],
                   reason=(f"IQR {rec['worst_iqr_frac'] * 100:.1f}% fails the gate, "
                           f"median pinned to +-{worst_pin * 100:.2f}%"))
    else:
        rec.update(tier=TIER_REJECTED, tier_name=TIER_NAMES[TIER_REJECTED],
                   reason=(f"median pinned only to +-{worst_pin * 100:.2f}%, worse "
                           f"than the gate's own worst accepted "
                           f"+-{bar_frac * 100:.2f}%"))

    rec["min_effect_frac"] = (EFFECT_MULTIPLE * worst_pin
                              if rec["tier"] == TIER_PINNED and worst_pin is not None
                              else None)
    return rec


def build(payload: dict) -> dict:
    rows = payload["results"]
    per_row = {(r["method"], r["ctx"]): measure_row(r) for r in rows}
    cal = calibration_bar(rows, per_row)
    recs = [assign_tier(r, per_row[(r["method"], r["ctx"])], cal["bar_frac"])
            for r in rows]
    counts = {name: sum(1 for r in recs if r["tier_name"] == name)
              for name in TIER_NAMES.values()}
    return {
        "max_iqr_frac": MAX_IQR_FRAC,
        "effect_multiple": EFFECT_MULTIPLE,
        "calibration": cal,
        "counts": counts,
        "rows": recs,
    }


def usable_for(rec: dict, effect_frac: float) -> bool:
    """May this row be quoted in support of an effect of this size?

    Tier 1 rows are usable wherever the gate already allows them. Tier 2 rows are
    usable only where the effect dwarfs the row's own median uncertainty; tier 3
    rows are never usable. `effect_frac` is the size of the effect as a fraction
    -- 0.30 for a 1.30x ratio.
    """
    if rec["tier"] == TIER_QUOTABLE:
        return True
    if rec["tier"] != TIER_PINNED:
        return False
    floor = rec.get("min_effect_frac")
    return floor is not None and abs(effect_frac) >= floor


def by_row(report: dict) -> dict:
    """Index a report's rows by (method, ctx), the key the rest of the repo uses."""
    return {(r["method"], r["ctx"]): r for r in report["rows"]}


def render(report: dict) -> str:
    cal = report["calibration"]
    lines = ["# Dispersion tiers", ""]
    if cal["bar_frac"] is None:
        lines += ["No quotable row in this run, so there is no bar to calibrate "
                  "against and every gate-failed row stays rejected.", ""]
    else:
        w = cal["worst"]
        lines += [
            f"The gate accepts {cal['n_accepted']} measurements. The worst-pinned "
            f"of them is `{w['method']}@{w['ctx']}` ({w['regime']}), whose median is "
            f"pinned to **+-{w['median_ci_halfwidth_frac'] * 100:.2f}%** at an IQR "
            f"of {w['iqr_frac'] * 100:.1f}%. That is the bar: a gate-failed row "
            f"joins tier 2 only if it is pinned at least that well, in every "
            f"regime it was measured in.",
            "",
            f"(Median across accepted measurements: "
            f"+-{cal['median_accepted_frac'] * 100:.2f}%.)",
            "",
        ]
    c = report["counts"]
    lines += [
        f"**{c['quotable']} quotable / {c['pinned']} pinned / {c['rejected']} "
        f"rejected** of {len(report['rows'])} rows.",
        "",
        "## Tier 2: gate-failed, median pinned",
        "",
    ]
    tier2 = [r for r in report["rows"] if r["tier"] == TIER_PINNED]
    if not tier2:
        lines += ["None in this run.", ""]
    else:
        lines += ["| row | worst IQR | median pinned to | usable for effects above |",
                  "|---|---|---|---|"]
        for r in sorted(tier2, key=lambda r: -r["worst_median_ci_halfwidth_frac"]):
            lines.append(
                f"| `{r['method']}@{r['ctx']}` "
                f"| {r['worst_iqr_frac'] * 100:.1f}% "
                f"| +-{r['worst_median_ci_halfwidth_frac'] * 100:.2f}% "
                f"| {r['min_effect_frac'] * 100:.1f}% |")
        lines.append("")
    lines += ["## Still rejected, and why", ""]
    tier3 = [r for r in report["rows"] if r["tier"] == TIER_REJECTED]
    if not tier3:
        lines += ["Nothing.", ""]
    else:
        for r in sorted(tier3, key=lambda r: (r["method"], r["ctx"])):
            lines.append(f"- `{r['method']}@{r['ctx']}` -- {r['reason']}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(RESULTS_DIR / "benchmark.json"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "dispersion_tier.json"))
    ap.add_argument("--md", default=str(RESULTS_DIR / "dispersion_tier.md"))
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = build(payload)
    cal = report["calibration"]

    if cal["bar_frac"] is None:
        print("no quotable row in this run -- nothing to calibrate the tier against")
    else:
        w = cal["worst"]
        print(f"calibration bar: +-{cal['bar_frac'] * 100:.2f}%  "
              f"(worst accepted: {w['method']}@{w['ctx']} {w['regime']}, "
              f"IQR {w['iqr_frac'] * 100:.1f}%)")
    print()

    print(f"{'row':<34}{'tier':<10}{'IQR%':>7}{'med+-%':>9}  reason")
    for r in sorted(report["rows"], key=lambda r: (r["tier"], r["method"], r["ctx"])):
        name = f"{r['method']}@{r['ctx']}"
        iqr = r["worst_iqr_frac"]
        pin = r["worst_median_ci_halfwidth_frac"]
        iqr_s = "    n/a" if iqr is None else f"{iqr * 100:7.1f}"
        pin_s = "      n/a" if pin is None else f"{pin * 100:9.2f}"
        print(f"{name:<34}{r['tier_name']:<10}{iqr_s}{pin_s}  {r['reason']}")

    c = report["counts"]
    print()
    print(f"{c['quotable']} quotable / {c['pinned']} pinned / {c['rejected']} "
          f"rejected  of {len(report['rows'])} rows")
    tier2 = [r for r in report["rows"] if r["tier"] == TIER_PINNED]
    if tier2:
        worst = max(tier2, key=lambda r: r["worst_median_ci_halfwidth_frac"])
        print(f"the {len(tier2)} tier-2 rows are all pinned to "
              f"+-{worst['worst_median_ci_halfwidth_frac'] * 100:.2f}% or better; "
              f"the loosest is {worst['method']}@{worst['ctx']}, usable only for "
              f"effects above {worst['min_effect_frac'] * 100:.1f}%")

    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    Path(args.md).write_text(render(report), encoding="utf-8")
    print(f"\nwrote {args.out} and {args.md}")


if __name__ == "__main__":
    main()
