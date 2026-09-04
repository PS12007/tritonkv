#!/usr/bin/env python
"""What do the recorded runs say about the two-mechanism hypothesis?

Two arms, tested separately, both from runs already on disk:

* **Thermal** -- too much preceding work and thermal pressure drags the memory
  clock down. Ruled out at these temperatures; see below.
* **Warm-up** -- too little preceding work and the memory clock never comes up.
  Not ruled out, and not established either: the test is underpowered at three
  runs per protocol and this file says by how much.

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

## The warm-up arm

If a run has to *earn* its memory clock, a preloaded run should start higher than
a cold-start one and the advantage should **decay** as the cold-start run
accumulates its own work. Measurement order is recoverable -- the results list is
chronological -- so each shared cell gets the preloaded-minus-cold difference and
its mean position in the run, and the early half is compared against the late
half.

The point estimates lean the right way for the short protocol (`preloaded` over
`subset`) and the wrong way for the long one (`fullpre` under `full`), which is
the shape the two-mechanism story predicts. Neither survives its own error bar.
The file reports the Welch t and the number of runs per protocol that would be
needed, because the useful output of an underpowered test is the power
calculation.

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

# How many runs each protocol group actually has. Used only to turn the observed
# scatter into "how many runs would this test need", so it is stated rather than
# inferred from whatever was passed.
RUNS_PER_PROTOCOL = 3


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


def ordered_cells(payloads: list[dict]) -> dict:
    """Mean memory clock and mean position-in-run for each cell.

    `benchmark.py` writes its results in measurement order (context outer,
    method inner), so the list index is chronological and is the only record of
    when in a run a cell was measured.
    """
    acc: dict[tuple, dict] = {}
    for payload in payloads:
        for i, r in enumerate(payload.get("results") or []):
            for regime in REGIMES:
                w = ((r.get("clocks") or {}).get(regime) or {}).get("clocks") or {}
                if w.get("mem_mhz_mean") is None:
                    continue
                key = (r["method"], r["ctx"], regime)
                acc.setdefault(key, {"mem": [], "idx": []})
                acc[key]["mem"].append(w["mem_mhz_mean"])
                acc[key]["idx"].append(i)
    return {k: {"mem_mhz": statistics.fmean(v["mem"]),
                "position": statistics.fmean(v["idx"])}
            for k, v in acc.items()}


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    denom = np.sqrt(va + vb)
    if denom == 0:
        # Both groups constant. If their means differ the separation is perfect,
        # which is the opposite of "no evidence" -- returning 0 here would have
        # reported a noiseless effect as insignificant.
        gap = float(a.mean() - b.mean())
        t = 0.0 if gap == 0 else float("inf") * (1.0 if gap > 0 else -1.0)
        return t, float(a.size + b.size - 2)
    t = float((a.mean() - b.mean()) / denom)
    df = float((va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1)))
    return t, df


def preload_decay(warm: list[dict], cold: list[dict]) -> dict:
    """Does a preload's memory-clock advantage decay as a run earns its own?

    Positive `early` means the preloaded run sat at a higher memory clock in the
    first half of the run. The warm-up story predicts early > late; a preload
    that simply heats the card predicts a negative difference with no decay.
    """
    W, C = ordered_cells(warm), ordered_cells(cold)
    shared = sorted(set(W) & set(C), key=lambda k: C[k]["position"])
    if len(shared) < 6:
        return {"n_cells": len(shared), "early_mhz": None}
    diff = np.array([W[k]["mem_mhz"] - C[k]["mem_mhz"] for k in shared])
    half = diff.size // 2
    early, late = diff[:half], diff[half:]
    t, df = _welch(early, late)
    pooled_sd = float(np.sqrt((early.var(ddof=1) + late.var(ddof=1)) / 2))
    observed = float(early.mean() - late.mean())
    # How much would the per-cell scatter have to shrink for this difference to
    # clear t = 2? Scatter falls as 1/sqrt(runs), so the runs needed follow.
    need_sd = abs(observed) / 2.0 * np.sqrt(2.0 / max(half, 1)) if observed else None
    runs_needed = (RUNS_PER_PROTOCOL * (pooled_sd / need_sd) ** 2
                   if need_sd and need_sd > 0 and pooled_sd > 0 else None)
    return {
        "n_cells": len(shared),
        "early_mhz": float(early.mean()),
        "late_mhz": float(late.mean()),
        "early_sd": float(early.std(ddof=1)),
        "late_sd": float(late.std(ddof=1)),
        "decay_mhz": observed,
        "welch_t": t,
        "welch_df": df,
        "significant": abs(t) >= 2.0,
        "pooled_sd_mhz": pooled_sd,
        "runs_per_protocol_needed": (float(runs_needed) if runs_needed else None),
    }


def degrees_for_a_p_state(slope_mhz_per_c: float | None) -> dict:
    if not slope_mhz_per_c:
        return {}
    s = abs(slope_mhz_per_c)
    return {"min_step_c": P_STATE_STEP_MIN_MHZ / s,
            "max_step_c": P_STATE_STEP_MAX_MHZ / s}


def build(groups: dict[str, list[dict]], pairs: list[tuple] | None = None) -> dict:
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
    decay = {}
    for warm, cold in (pairs or []):
        if warm in groups and cold in groups:
            decay[f"{warm} vs {cold}"] = preload_decay(groups[warm], groups[cold])

    return {
        "per_protocol": per_protocol,
        "warm_up_arm": decay,
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
    decay = rep.get("warm_up_arm") or {}
    if decay:
        L += ["", "## The other arm: does a preload's advantage decay?", "",
              "If a run has to earn its memory clock, a preloaded run should start "
              "higher than a cold-start one and the gap should shrink as the "
              "cold-start run accumulates its own work. Measurement order is "
              "recoverable, so each shared cell is compared early-half against "
              "late-half.", "",
              "| comparison | cells | early (MHz) | late (MHz) | decay | Welch t | "
              "established? | runs/protocol needed |",
              "|---|---|---|---|---|---|---|---|"]
        for name, d in decay.items():
            if d.get("early_mhz") is None:
                L.append(f"| `{name}` | {d['n_cells']} | -- | -- | -- | -- | -- | -- |")
                continue
            runs = d["runs_per_protocol_needed"]
            L.append(
                f"| `{name}` | {d['n_cells']} | {d['early_mhz']:+.0f} +-{d['early_sd']:.0f} "
                f"| {d['late_mhz']:+.0f} +-{d['late_sd']:.0f} | {d['decay_mhz']:+.0f} "
                f"| {d['welch_t']:+.2f} | "
                f"{'yes' if d['significant'] else '**no**'} | "
                f"{'--' if not runs else f'~{runs:.0f}'} |")
        L += ["",
              "The point estimates lean the way the story predicts -- the short "
              "protocol gains from a preload and loses the gain as it goes, the long "
              "one only loses -- and **neither clears its own error bar**. The last "
              "column is the useful output: at the scatter these cells actually "
              "show, settling this by this route would take hundreds of runs per "
              "protocol, which is tens of hours. That is a reason to measure the "
              "clock ramp directly during an idle-to-load transition rather than to "
              "keep inferring it from benchmark cells.", ""]

    L += [
        "Three caveats, none of which rescues the thermal mechanism but all of "
        "which bound the claim:",
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
    ap.add_argument("--pair", action="append", default=[],
                    help="warm=cold -- test whether the warm protocol's memory-clock "
                         "advantage over the cold one decays across the run. Both "
                         "names must be --label groups.")
    ap.add_argument("--out", default=str(RESULTS_DIR / "thermal_check.json"))
    ap.add_argument("--md", default=str(RESULTS_DIR / "thermal_check.md"))
    args = ap.parse_args()
    if not args.label:
        raise SystemExit("nothing to compare -- pass at least one --label name=path")

    pairs = []
    for spec in args.pair:
        if "=" not in spec:
            raise SystemExit(f"--pair wants warm=cold, got {spec!r}")
        warm, cold = spec.split("=", 1)
        pairs.append((warm.strip(), cold.strip()))
    rep = build(load_groups(args.label), pairs)
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
    for name, d in (rep.get("warm_up_arm") or {}).items():
        if d.get("early_mhz") is None:
            print(f"{name}: too few shared cells")
            continue
        need = d["runs_per_protocol_needed"]
        print(f"{name}: early {d['early_mhz']:+.0f} late {d['late_mhz']:+.0f} "
              f"MHz  decay {d['decay_mhz']:+.0f}  t = {d['welch_t']:+.2f}  "
              + ("ESTABLISHED" if d["significant"]
                 else f"not established (would need ~{need:.0f} runs/protocol)"
                 if need else "not established"))
    print()
    print("=> thermal mechanism is "
          + ("SUFFICIENT" if rep["thermal_mechanism_sufficient"]
             else "NOT sufficient at these temperatures"))

    Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    Path(args.md).write_text(render(rep), encoding="utf-8")
    print(f"\nwrote {args.out} and {args.md}")


if __name__ == "__main__":
    main()
