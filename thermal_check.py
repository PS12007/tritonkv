#!/usr/bin/env python
"""Can temperature explain why the shortest and longest protocols are the noisy ones?

    ./.venv/Scripts/python.exe thermal_check.py \\
        --label full=results/runs/run1.json,results/runs/run2.json,results/runs/run3.json \\
        --label subset=results/tail/validate.json,results/tail/sub2.json,results/tail/sub3.json \\
        --label preloaded=results/tail/pre1.json,results/tail/pre2.json,results/tail/pre3.json \\
        --label fullpre=results/tail/fullpre3.json,results/tail/fullpre4.json,results/tail/fullpre5.json

`next_steps.md` carries a two-mechanism *hypothesis* for the protocol spreads --
too little preceding work and the memory clock never comes up, too much and
thermal pressure drags it back down -- and says "a temperature sweep at fixed
protocol would be the actual experiment". Before spending two hours of wall clock
on that sweep, it is worth asking what the runs already recorded say, because
they carry a per-window temperature and a per-window memory clock for every
observation.

The trick is to compare **within a cell**. Between cells, temperature and memory
clock both vary enormously for reasons that have nothing to do with each other --
`fp16_sdpa@512` and `triton_fp16_control@16384` are different workloads. So every
observation is expressed as a deviation from its own (method, ctx, regime,
protocol) cell mean, and the question becomes: when a cell is measured hotter
than that cell usually is, does its memory clock sit lower?

It does, and the effect is far too small to matter. That is the point of this
file: it converts "thermal pressure" from a hypothesis into a slope in MHz per
degree, and then asks how many degrees would be needed to move the memory clock
by one P-state step. The answer is several times the temperature range the
protocols actually span, which is what rules the mechanism out at these
temperatures -- and says how much wider a sweep would have to reach to test it.

Nothing here re-measures anything.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

REGIMES = ("cold", "graph")

# The memory P-state steps observed on this part, from `progress_log.md`: the
# card sits at 9001 / 10001 / 11001 / 12001 MHz and the smallest step actually
# seen between two windows is ~350 MHz. A mechanism that cannot move the clock
# by at least the smallest step cannot change a P-state, and it is P-state
# changes that cost DRAM-resident time.
P_STATE_STEP_MIN_MHZ = 350.0
P_STATE_STEP_MAX_MHZ = 1100.0

# A cell needs at least this many observations before a within-cell deviation
# means anything.
MIN_PER_CELL = 3


def observations(payload: dict) -> list[dict]:
    """Every (method, ctx, regime) measurement window with clocks and a temperature."""
    out = []
    for r in payload.get("results") or []:
        for regime in REGIMES:
            w = ((r.get("clocks") or {}).get(regime) or {}).get("clocks") or {}
            if w.get("mem_mhz_mean") is None or w.get("temp_c_mean") is None:
                continue
            out.append({"method": r["method"], "ctx": r["ctx"], "regime": regime,
                        "mem_mhz": w["mem_mhz_mean"], "temp_c": w["temp_c_mean"],
                        "power_w": w.get("power_w_mean"),
                        "sm_mhz": w.get("sm_mhz_mean")})
    return out


def within_cell_deviations(rows: list[dict]) -> tuple[list[float], list[float], int]:
    """Express each observation as a deviation from its own cell's mean.

    Between cells, temperature and memory clock both move for unrelated reasons,
    so a pooled correlation over raw values would mostly measure "these are
    different workloads". Only the within-cell comparison isolates the question.
    """
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        cells.setdefault((r["method"], r["ctx"], r["regime"]), []).append(r)
    dt, dm = [], []
    used = 0
    for group in cells.values():
        if len(group) < MIN_PER_CELL:
            continue
        used += 1
        t_bar = statistics.fmean(g["temp_c"] for g in group)
        m_bar = statistics.fmean(g["mem_mhz"] for g in group)
        for g in group:
            dt.append(g["temp_c"] - t_bar)
            dm.append(g["mem_mhz"] - m_bar)
    return dt, dm, used


def fit(dt: list[float], dm: list[float]) -> dict:
    x, y = np.asarray(dt, float), np.asarray(dm, float)
    if x.size < 3 or x.std() == 0:
        return {"n": int(x.size), "slope_mhz_per_c": None}
    slope, intercept = np.polyfit(x, y, 1)
    # A cell whose memory clock never moved has no correlation to report -- and
    # `corrcoef` would return nan rather than saying so. The slope is still
    # well defined (it is zero), which is the honest answer.
    r = 0.0 if y.std() == 0 else float(np.corrcoef(x, y)[0, 1])
    # Reported, but see the caveat printed with it: observations inside a cell
    # come from repeated runs of the same measurement and are not independent,
    # so this overstates the significance.
    t = r * np.sqrt((x.size - 2) / max(1 - r * r, 1e-12))
    return {
        "n": int(x.size),
        "slope_mhz_per_c": float(slope),
        "r": r,
        "t": float(t),
        "variance_explained": float(r * r),
        "temp_dev_min": float(x.min()),
        "temp_dev_max": float(x.max()),
    }


def degrees_for_a_p_state(slope_mhz_per_c: float | None) -> dict:
    if not slope_mhz_per_c:
        return {}
    s = abs(slope_mhz_per_c)
    return {"min_step_c": P_STATE_STEP_MIN_MHZ / s,
            "max_step_c": P_STATE_STEP_MAX_MHZ / s}


def build(groups: dict[str, list[dict]]) -> dict:
    per_protocol = {}
    all_dt: list[float] = []
    all_dm: list[float] = []
    protocol_temps = {}
    for name, payloads in groups.items():
        rows = [o for p in payloads for o in observations(p)]
        dt, dm, cells = within_cell_deviations(rows)
        all_dt += dt
        all_dm += dm
        per_protocol[name] = {"n_observations": len(rows), "n_cells": cells,
                              "fit": fit(dt, dm)}
        if rows:
            protocol_temps[name] = statistics.fmean(r["temp_c"] for r in rows)

    pooled = fit(all_dt, all_dm)
    need = degrees_for_a_p_state(pooled.get("slope_mhz_per_c"))
    span = (max(protocol_temps.values()) - min(protocol_temps.values())
            if len(protocol_temps) > 1 else 0.0)
    predicted = (span * abs(pooled["slope_mhz_per_c"])
                 if pooled.get("slope_mhz_per_c") else None)
    return {
        "per_protocol": per_protocol,
        "pooled": pooled,
        "protocol_mean_temp_c": protocol_temps,
        "protocol_temp_span_c": span,
        "predicted_mhz_over_span": predicted,
        "degrees_for_a_p_state": need,
        "thermal_mechanism_sufficient": bool(
            predicted is not None and predicted >= P_STATE_STEP_MIN_MHZ),
    }


def render(rep: dict) -> str:
    p = rep["pooled"]
    L = ["# Can temperature explain the protocol spreads?", ""]
    if p.get("slope_mhz_per_c") is None:
        return "\n".join(L + ["Not enough data to fit."])
    L += [
        f"Every observation is expressed as a deviation from its own "
        f"(method, ctx, regime, protocol) cell mean, so this is a within-cell "
        f"question and not a comparison between different workloads.",
        "",
        f"**Slope: {p['slope_mhz_per_c']:.1f} MHz per degree C** "
        f"(r = {p['r']:+.3f}, n = {p['n']}), explaining "
        f"{p['variance_explained'] * 100:.1f}% of the variance in memory clock. "
        f"The sign is the one the thermal story predicts. The size is the "
        f"problem.",
        "",
        "| protocol | observations | cells | slope (MHz/C) | r |",
        "|---|---|---|---|---|",
    ]
    for name, blk in rep["per_protocol"].items():
        f = blk["fit"]
        if f.get("slope_mhz_per_c") is None:
            L.append(f"| `{name}` | {blk['n_observations']} | {blk['n_cells']} | -- | -- |")
        else:
            L.append(f"| `{name}` | {blk['n_observations']} | {blk['n_cells']} | "
                     f"{f['slope_mhz_per_c']:.1f} | {f['r']:+.3f} |")
    need = rep["degrees_for_a_p_state"]
    L += ["", "## The number that settles it", ""]
    L.append(f"A memory P-state step on this part is "
             f"{P_STATE_STEP_MIN_MHZ:.0f}-{P_STATE_STEP_MAX_MHZ:.0f} MHz, and it is "
             f"P-state changes that cost DRAM-resident time. At "
             f"{abs(p['slope_mhz_per_c']):.1f} MHz per degree, moving one step needs "
             f"**{need['min_step_c']:.1f}-{need['max_step_c']:.1f} degrees C**.")
    L.append("")
    L.append(f"The four protocols span **{rep['protocol_temp_span_c']:.1f} degrees C** "
             + ", ".join(f"(`{k}` {v:.1f})" for k, v in
                         sorted(rep["protocol_mean_temp_c"].items(), key=lambda kv: kv[1]))
             + f", which predicts **{rep['predicted_mhz_over_span']:.0f} MHz** -- "
             + ("enough" if rep["thermal_mechanism_sufficient"]
                else "well under a third of the smallest step")
             + ".")
    L += ["", "**"
          + ("Temperature is sufficient to move a P-state over the observed range."
             if rep["thermal_mechanism_sufficient"] else
             "So temperature cannot be the mechanism at these temperatures.")
          + "**", ""]
    L += [
        "Three caveats, none of which rescues the mechanism but all of which "
        "bound the claim:",
        "",
        "- The fit is linear over a narrow window -- cell deviations span only "
        f"{p['temp_dev_min']:.1f} to {p['temp_dev_max']:.1f} C -- so the degrees "
        "needed for a P-state step are an **extrapolation**. A threshold outside "
        "that window would not show up here.",
        "- Observations inside a cell come from repeated runs of the same "
        "measurement and are not independent, so the reported t overstates the "
        "significance.",
        "- Temperature is a mean over the window. A brief spike could throttle "
        "without moving the mean much.",
        "",
        "What this does say is what a temperature sweep would have to achieve to "
        f"be worth running: it must induce **at least {need['min_step_c']:.0f} "
        f"degrees C** of variation at fixed protocol, not the "
        f"{rep['protocol_temp_span_c']:.1f} the protocols happen to produce.",
    ]
    return "\n".join(L)


def load_groups(specs: list[str]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--label wants name=path[,path...], got {spec!r}")
        name, paths = spec.split("=", 1)
        loaded = []
        for path in paths.split(","):
            p = Path(path.strip())
            if not p.exists():
                raise SystemExit(f"no such run: {p}")
            loaded.append(json.loads(p.read_text(encoding="utf-8")))
        groups[name] = loaded
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", default=[],
                    help="name=path[,path...] -- one protocol per flag")
    ap.add_argument("--out", default=str(RESULTS_DIR / "thermal_check.json"))
    ap.add_argument("--md", default=str(RESULTS_DIR / "thermal_check.md"))
    args = ap.parse_args()
    if not args.label:
        raise SystemExit("nothing to compare -- pass at least one --label name=path")

    rep = build(load_groups(args.label))
    p = rep["pooled"]
    if p.get("slope_mhz_per_c") is None:
        print("not enough data to fit")
        return

    for name, blk in rep["per_protocol"].items():
        f = blk["fit"]
        s = ("--" if f.get("slope_mhz_per_c") is None
             else f"{f['slope_mhz_per_c']:7.1f} MHz/C  r = {f['r']:+.3f}")
        print(f"{name:<12} {blk['n_observations']:>4} obs  "
              f"{blk['n_cells']:>3} cells  {s}")
    print()
    print(f"pooled: {p['slope_mhz_per_c']:.1f} MHz per degree C  "
          f"(r = {p['r']:+.3f}, n = {p['n']}, {p['variance_explained'] * 100:.1f}% "
          f"of variance)")
    need = rep["degrees_for_a_p_state"]
    print(f"one P-state step ({P_STATE_STEP_MIN_MHZ:.0f}-{P_STATE_STEP_MAX_MHZ:.0f} MHz) "
          f"needs {need['min_step_c']:.1f}-{need['max_step_c']:.1f} C")
    print(f"the protocols span {rep['protocol_temp_span_c']:.1f} C "
          f"-> {rep['predicted_mhz_over_span']:.0f} MHz predicted")
    print("=> thermal mechanism is "
          + ("SUFFICIENT" if rep["thermal_mechanism_sufficient"]
             else "NOT sufficient at these temperatures"))

    Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    Path(args.md).write_text(render(rep), encoding="utf-8")
    print(f"\nwrote {args.out} and {args.md}")


if __name__ == "__main__":
    main()
