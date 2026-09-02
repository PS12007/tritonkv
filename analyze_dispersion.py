#!/usr/bin/env python
"""Why do rows fail the dispersion half of the gate? Decompose it before fixing it.

    python analyze_dispersion.py                 # reads results/benchmark.json

23 of 48 rows in the last run fail `IQR <= 5% of median`, and `next_steps.md`
records two candidate fixes that point in opposite directions: shorten the
sampling window so the 80 W part drifts less inside it, or lengthen it so the
drift averages out. Choosing between them by taste is exactly the move this
project keeps having to undo, so this decomposes the dispersion first.

The distinction that matters:

* **Drift** -- a trend across the sample index. The samples disagree because the
  GPU was in a different state at the end than at the start. Shorter windows
  reduce it; longer ones do not, and a longer window makes it worse.
* **White jitter** -- no trend, no memory between samples. More samples do *not*
  shrink the IQR, because IQR is a property of the sample distribution rather
  than of the estimate. It shrinks the *median's* uncertainty and nothing else.
  If a row fails this way, no window length fixes it and the gate is measuring
  the wrong thing for that regime.
* **Heavy tail** -- a handful of interrupted samples dragging the quartiles.
  Neither window length helps; this is what a trimmed statistic is for.

So each row gets: the raw IQR fraction, the IQR fraction after removing a linear
trend in sample index, the lag-1 autocorrelation, and how much of the spread the
worst 5% of samples own. Those four separate the three causes, and the answer is
allowed to be different per regime -- which is the hypothesis `next_steps.md`
already suspects.

Nothing here re-measures anything: it is all from the raw per-sample timings
already recorded in `results/benchmark.json`.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
MAX_IQR_FRAC = 0.05  # must match benchmark.MAX_IQR_FRAC

REGIMES = (("cold", "cold_raw_ms", "DRAM-resident"),
           ("graph", "graph_raw_ms", "L2-resident"))


def iqr_frac(x: np.ndarray) -> float:
    med = float(np.median(x))
    if med <= 0:
        return float("inf")
    p25, p75 = np.percentile(x, [25, 75])
    return float(p75 - p25) / med


def block_bootstrap_median_ci(a: np.ndarray, n_boot=4000, ci=0.95,
                              seed=0) -> tuple[float, float]:
    """CI for the median under serial correlation, via a moving-block bootstrap.

    Several of these series have lag-1 autocorrelation of 0.3-0.7 -- the card
    wanders rather than jittering independently -- and an ordinary bootstrap
    assumes independence, so it would report an interval too narrow by exactly
    the amount the correlation matters. Blocks of length ~n**(1/3) preserve the
    short-range structure while still resampling.
    """
    n = a.size
    L = max(2, int(round(n ** (1 / 3))))
    n_blocks = -(-n // L)
    starts_hi = max(1, n - L + 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, starts_hi, size=(n_boot, n_blocks))
    offs = np.arange(L)
    draws = a[(idx[:, :, None] + offs[None, None, :]).reshape(n_boot, -1)[:, :n]]
    meds = np.median(draws, axis=1)
    lo, hi = np.percentile(meds, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def describe(x: list[float]) -> dict | None:
    """Split a sample series into trend, memory, and tail."""
    a = np.asarray(x, dtype=np.float64)
    n = a.size
    if n < 8:
        return None
    t = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(t, a, 1)
    fit = slope * t + intercept
    resid = a - fit + float(np.median(a))
    # Fraction of total variance the trend explains. A drifting row has most of
    # its spread here; a jittery one has almost none.
    ss_tot = float(((a - a.mean()) ** 2).sum())
    r2 = float(1.0 - ((a - fit) ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0
    c = a - a.mean()
    denom = float((c * c).sum())
    lag1 = float((c[:-1] * c[1:]).sum() / denom) if denom > 0 else 0.0
    med = float(np.median(a))
    # What the IQR would be if the worst 5% of samples were dropped -- i.e. how
    # much of the failure is a few interrupted measurements rather than the bulk.
    keep = a[a <= np.percentile(a, 95)]
    # Significance of the trend, not just how much variance it explains. A -20%
    # decline buried in loud noise has a low r2 and is still a real ramp; an r2
    # gate would file the single largest systematic effect in the run as jitter.
    se_slope = float(np.sqrt(((a - fit) ** 2).sum() / max(n - 2, 1)
                             / max(((t - t.mean()) ** 2).sum(), 1e-12)))
    t_slope = float(slope / se_slope) if se_slope > 0 else 0.0
    detrended_trimmed = resid[resid <= np.percentile(resid, 95)]
    lo, hi = block_bootstrap_median_ci(a)
    return {
        "n": n,
        "median_ms": med,
        "iqr_frac": iqr_frac(a),
        "iqr_frac_detrended": iqr_frac(resid),
        "iqr_frac_trimmed95": iqr_frac(keep) if keep.size >= 8 else None,
        "iqr_frac_detrended_trimmed95": (iqr_frac(detrended_trimmed)
                                         if detrended_trimmed.size >= 8 else None),
        "trend_r2": r2,
        "trend_t": t_slope,
        "trend_pct_over_window": float(slope * (n - 1) / med * 100.0) if med else None,
        "lag1_autocorr": lag1,
        "p95_over_median": float(np.percentile(a, 95) / med) if med else None,
        # What a reader of the tables actually depends on: how well pinned is the
        # number being quoted? This shrinks with more samples even when the
        # per-sample IQR does not.
        "median_ci_halfwidth_frac": float((hi - lo) / 2 / med) if med else None,
    }


def classify(d: dict) -> str:
    """Name the dominant cause, in the terms the two candidate fixes are stated in.

    Deliberately keyed on what each fix would *do* rather than on a statistic:
    "drift" means a shorter window would help, "wander" and "white jitter" mean
    neither window length would, "tail" means a trimmed statistic would. A label
    nobody can act on is not a diagnosis.

    Significance, not r-squared, decides whether a trend counts. The fp16
    control's DRAM-resident rows fall 19-23% across their window with an r2 of
    0.09-0.16, because the noise around that decline is loud -- an r2 gate files
    the single largest systematic effect in the run as jitter.
    """
    if d["iqr_frac"] <= MAX_IQR_FRAC:
        return "passes"
    trend_real = abs(d["trend_t"]) >= 3.0 and abs(d["trend_pct_over_window"]) >= 5.0
    detrend_helps = d["iqr_frac_detrended"] <= 0.75 * d["iqr_frac"]
    if trend_real and detrend_helps:
        return ("drift (shorter windows)"
                if d["iqr_frac_detrended"] <= MAX_IQR_FRAC
                else "drift + floor (shorter windows, not enough)")
    if (d["iqr_frac_trimmed95"] or 9.9) <= MAX_IQR_FRAC:
        return "tail (trimmed stat)"
    if d["lag1_autocorr"] >= 0.25:
        return "wander (no window length helps)"
    return "white jitter (no window length helps)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(RESULTS_DIR / "benchmark.json"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "dispersion.json"))
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload["results"]

    out = []
    print(f"{'method':<26}{'ctx':>6}  {'regime':<14}"
          f"{'IQR%':>7}{'detr%':>7}{'trim%':>7}{'trend%':>8}{'t':>7}{'lag1':>7}"
          f"{'med+-%':>8}  cause")
    for r in rows:
        for regime_key, raw_key, regime_label in REGIMES:
            xs = r.get(raw_key)
            if not xs:
                continue
            d = describe(xs)
            if d is None:
                continue
            d.update(method=r["method"], ctx=r["ctx"], regime=regime_label,
                     row_quotable=r.get("quotable"))
            d["cause"] = classify(d)
            out.append(d)
            print(f"{r['method']:<26}{r['ctx']:>6}  {regime_label:<14}"
                  f"{d['iqr_frac'] * 100:>6.1f} {d['iqr_frac_detrended'] * 100:>6.1f} "
                  f"{(d['iqr_frac_trimmed95'] or float('nan')) * 100:>6.1f} "
                  f"{d['trend_pct_over_window']:>8.1f}{d['trend_t']:>7.1f}"
                  f"{d['lag1_autocorr']:>7.2f}"
                  f"{d['median_ci_halfwidth_frac'] * 100:>8.2f}  {d['cause']}")

    print()
    fails = [d for d in out if d["cause"] != "passes"]
    print(f"{len(fails)} of {len(out)} measurements fail IQR <= {MAX_IQR_FRAC:.0%}")
    for cause in ("drift (shorter windows)",
                  "drift + floor (shorter windows, not enough)",
                  "tail (trimmed stat)",
                  "wander (no window length helps)",
                  "white jitter (no window length helps)"):
        sel = [d for d in fails if d["cause"] == cause]
        if not sel:
            continue
        by_ctx: dict[int, int] = {}
        for d in sel:
            by_ctx[d["ctx"]] = by_ctx.get(d["ctx"], 0) + 1
        spread = ", ".join(f"ctx={c}:{n}" for c, n in sorted(by_ctx.items()))
        print(f"  {len(sel):>3}  {cause:<40} {spread}")

    for regime_label in ("DRAM-resident", "L2-resident"):
        sel = [d for d in fails if d["regime"] == regime_label]
        if sel:
            print(f"  {regime_label}: median |trend| over window "
                  f"{statistics.median(abs(d['trend_pct_over_window']) for d in sel):.1f}%, "
                  f"median lag-1 {statistics.median(d['lag1_autocorr'] for d in sel):.2f}")

    # A row can have a 10% per-sample IQR and still pin its median to a fraction
    # of a percent. If that is generally true then the gate is rejecting rows for
    # a property that does not affect any conclusion drawn from them -- which is
    # a finding about the gate, not a licence to widen it.
    print()
    if fails:
        worst = max(fails, key=lambda d: d["median_ci_halfwidth_frac"])
        print("median precision on the measurements that FAIL the IQR gate:")
        print(f"  worst: {worst['method']} ctx={worst['ctx']} {worst['regime']}"
              f" -- per-sample IQR {worst['iqr_frac'] * 100:.1f}%,"
              f" median pinned to +-{worst['median_ci_halfwidth_frac'] * 100:.2f}%")
        med_ci = statistics.median(d["median_ci_halfwidth_frac"] for d in fails)
        pass_ci = [d["median_ci_halfwidth_frac"] for d in out if d["cause"] == "passes"]
        print(f"  median across the {len(fails)} failing measurements:"
              f" +-{med_ci * 100:.2f}%  (moving-block bootstrap, 95%)")
        if pass_ci:
            print(f"  for comparison, the {len(pass_ci)} passing measurements:"
                  f" +-{statistics.median(pass_ci) * 100:.2f}%")

    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
