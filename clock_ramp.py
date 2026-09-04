#!/usr/bin/env python
"""How long does the memory clock take to come up? Measure it directly.

    ./.venv/Scripts/python.exe clock_ramp.py --idle 90 --load 180

`thermal_check.py` tested the two-mechanism hypothesis for the protocol spreads
against the runs already on disk. The thermal arm is ruled out at these
temperatures. The warm-up arm -- too little preceding work and the memory clock
never comes up -- leans the right way and does not clear its error bar, and the
power calculation says settling it by repeating protocols would take **~200 runs
per protocol**, tens of hours.

That is the wrong experiment. The hypothesis is a claim about a *time constant*:
if a 205 s protocol is noisy because the memory clock has not finished rising,
then the clock must take a substantial fraction of 205 s to rise. That is
directly observable in a few minutes, without running the benchmark at all.

So: let the GPU idle until its clocks fall, then apply the same DRAM-saturating
load `benchmark.py` uses for its ramp, and watch the memory clock with
`nvidia-smi` throughout. The output is a time constant.

**What each answer would mean.** If the memory clock reaches its loaded ceiling
in a couple of seconds, the warm-up arm cannot explain a 205 s protocol behaving
differently from a 775 s one, and it is dead. If it takes tens of seconds or
creeps for minutes, the arm survives and the number is the thing to design the
next protocol around. Either way this is a measurement rather than an argument,
and it costs one run instead of two hundred.

The load is deliberately the *same* one `benchmark.py` ramps with -- a
cache-resident GEMM alongside a DRAM-sized copy -- because the question is about
that ramp's behaviour, not about a workload invented here. A GEMM alone drives
the SM clock and asks the memory system for nothing, which is the mistake this
project already made once and corrected.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from benchmark import ClockMonitor, _ramp_buffers, max_sm_clock

RESULTS_DIR = Path(__file__).parent / "results"

# A sample counts as "under load" above this utilization -- the same threshold
# benchmark.py uses to decide which samples define the reachable ceiling.
LOADED_UTIL_PCT = 50.0

# The clock is "at ceiling" once it is within this fraction of the highest value
# seen under load. The card steps between P-states rather than sliding, so this
# only has to be tighter than one step (350 MHz of ~11000, i.e. ~3%).
CEILING_FRAC = 0.99

# ...and it has to stay there this long before the arrival counts, so that a
# single sample touching the ceiling on its way past does not end the ramp.
HOLD_SECONDS = 3.0


def saturating_load(seconds: float, monitor: ClockMonitor) -> None:
    """The ramp workload, run continuously for a fixed wall-clock duration."""
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    src, dst = _ramp_buffers()
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(8):
            dst.copy_(src, non_blocking=True)
            torch.mm(a, b)
        torch.cuda.synchronize()


def series(monitor: ClockMonitor, t0: float, t1: float) -> list[dict]:
    """Samples in [t0, t1], with time expressed relative to t0."""
    return [{"t": ts - t0, **s} for ts, s in monitor.samples if t0 <= ts <= t1]


def time_to_ceiling(rows: list[dict], key: str = "mem_mhz") -> dict:
    """When did `key` first reach, and then hold, its **sustained** loaded level?

    The first version of this took the ceiling to be the maximum over loaded
    samples, and that was wrong in a way the summary hid. On this part the memory
    clock boosts to 12001 MHz for the first few seconds of a load and then settles
    to 11001 for the rest -- 96% of a 180 s window. Taking the maximum makes the
    transient the target, so the answer becomes "time to the peak" when the
    question is "time to the state a benchmark actually runs in".

    The ceiling is therefore the **median of the loaded samples**, which is the
    level the card holds. The peak is reported alongside with how long it lasted,
    because a boost that decays is a real feature of the ramp and not noise.

    Arrival is measured against *all* samples in the window rather than only the
    loaded ones: the clock here leaves idle while utilization is still climbing
    through 39%, so a utilization filter would date the ramp later than it
    happened. Utilization decides what the ceiling *is*, not when it was reached.
    """
    loaded = [r for r in rows if r.get("util_pct", 0.0) >= LOADED_UTIL_PCT]
    if not loaded:
        return {"n_loaded": 0, "ceiling": None}
    values = [r[key] for r in loaded]
    sustained = float(statistics.median(values))
    peak = float(max(values))
    at_peak = [r for r in loaded if r[key] >= peak * CEILING_FRAC]
    peak_seconds = (at_peak[-1]["t"] - at_peak[0]["t"]) if len(at_peak) > 1 else 0.0

    threshold = sustained * CEILING_FRAC
    first = None
    arrived = None
    for r in rows:
        if r[key] < threshold:
            first = None
            continue
        if first is None:
            first = r["t"]
        if r["t"] - first >= HOLD_SECONDS:
            arrived = first
            break
    if arrived is None and first is not None and rows[-1][key] >= threshold:
        arrived = first
    return {
        "n_loaded": len(loaded),
        "ceiling": sustained,
        "sustained_frac_of_loaded": float(
            sum(1 for v in values if v >= threshold) / len(values)),
        "peak": peak,
        "peak_seconds": float(peak_seconds),
        "peak_is_transient": bool(peak > sustained * 1.01),
        "threshold": float(threshold),
        "first_at_ceiling_s": (float(arrived) if arrived is not None else None),
        "idle_exit_s": next((r["t"] for r in rows if r[key] >= threshold), None),
        "first_loaded_value": float(loaded[0][key]),
        "last_value": float(loaded[-1][key]),
    }


def build(idle_rows: list[dict], load_rows: list[dict]) -> dict:
    out = {"n_idle_samples": len(idle_rows), "n_load_samples": len(load_rows)}
    for key in ("mem_mhz", "sm_mhz"):
        idle_vals = [r[key] for r in idle_rows]
        out[key] = {
            "idle_median": statistics.median(idle_vals) if idle_vals else None,
            "idle_min": min(idle_vals) if idle_vals else None,
            "idle_max": max(idle_vals) if idle_vals else None,
            **time_to_ceiling(load_rows, key),
        }
    for key in ("temp_c", "power_w"):
        vals = [r[key] for r in load_rows if key in r]
        out[key] = {"load_first": vals[0] if vals else None,
                    "load_last": vals[-1] if vals else None,
                    "load_max": max(vals) if vals else None}
    return out


def verdict(rep: dict, protocol_seconds: float) -> dict:
    """Can a ramp this fast explain a protocol this long behaving differently?"""
    t = rep["mem_mhz"].get("first_at_ceiling_s")
    if t is None:
        return {"settled": False,
                "note": "the memory clock never held its loaded ceiling in this "
                        "window, which is itself the answer: it does not simply "
                        "rise and stay"}
    frac = t / protocol_seconds if protocol_seconds else None
    return {
        "settled": True,
        "seconds_to_ceiling": t,
        "shortest_protocol_seconds": protocol_seconds,
        "fraction_of_shortest_protocol": frac,
        "warm_up_arm_viable": bool(frac is not None and frac >= 0.10),
    }


def render(rep: dict, v: dict, args) -> str:
    m = rep["mem_mhz"]
    L = ["# How long does the memory clock take to come up?", "",
         f"Idle {args.idle:.0f} s, then {args.load:.0f} s of the same "
         f"DRAM-saturating load `benchmark.py` ramps with. "
         f"{rep['n_idle_samples']} idle samples, {rep['n_load_samples']} under load.",
         ""]
    L += ["| | memory clock | SM clock |", "|---|---|---|"]
    s = rep["sm_mhz"]
    L.append(f"| idle median | {m['idle_median']:.0f} MHz | {s['idle_median']:.0f} MHz |")
    L.append(f"| loaded ceiling | {m['ceiling']:.0f} MHz | {s['ceiling']:.0f} MHz |"
             if m["ceiling"] and s["ceiling"] else "| loaded ceiling | -- | -- |")
    for name, blk in (("memory", m), ("SM", s)):
        t = blk.get("first_at_ceiling_s")
        L.append(f"| time to {name} ceiling | "
                 + (f"**{t:.1f} s**" if t is not None else "never held")
                 + " | |")
    if m.get("peak_is_transient"):
        L += ["",
              f"The memory clock **boosts to {m['peak']:.0f} MHz for the first "
              f"{m['peak_seconds']:.1f} s** and then settles to "
              f"{m['ceiling']:.0f} MHz, which it holds for "
              f"{m['sustained_frac_of_loaded'] * 100:.0f}% of the loaded window. "
              f"The sustained figure is the one a benchmark runs in, so it is what "
              f"the ramp time is measured against; taking the peak as the ceiling "
              f"would answer a different question."]
    L += ["", "## What it means", ""]
    if not v["settled"]:
        L.append(v["note"])
        return "\n".join(L)
    L.append(
        f"The memory clock reaches and holds its loaded ceiling "
        f"**{v['seconds_to_ceiling']:.1f} s** after the load starts — "
        f"{v['fraction_of_shortest_protocol'] * 100:.1f}% of the shortest protocol "
        f"in this repo ({v['shortest_protocol_seconds']:.0f} s).")
    L.append("")
    if v["warm_up_arm_viable"]:
        L.append("That is a large enough fraction that the warm-up arm survives: a "
                 "short protocol really can spend a meaningful part of itself with "
                 "the memory clock still rising. The number is the thing to design "
                 "the next protocol around.")
    else:
        L.append("**That kills the warm-up arm as an explanation of the protocol "
                 "spreads.** A clock that is up and holding within a small fraction "
                 "of the shortest protocol cannot explain why that protocol behaves "
                 "differently from one four times longer. Whatever separates them, "
                 "it is not the time the memory clock takes to rise from idle.")
    L += ["", "Caveats: one card, one measurement, and the load here is continuous "
              "whereas a benchmark alternates work with allocation and Python. A "
              "ramp measured under continuous load is a *lower bound* on how long "
              "an intermittent workload would take.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", type=float, default=90.0,
                    help="seconds to leave the GPU alone before loading it")
    ap.add_argument("--load", type=float, default=180.0,
                    help="seconds of saturating load to watch")
    ap.add_argument("--shortest-protocol", type=float, default=205.0,
                    help="wall seconds of the shortest protocol in the repo, which "
                         "is what the ramp time is judged against")
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--from-json", default=None,
                    help="re-analyse a saved clock_ramp.json instead of measuring. "
                         "The raw series is recorded, so a correction to the "
                         "analysis never needs the GPU again.")
    ap.add_argument("--out", default=str(RESULTS_DIR / "clock_ramp.json"))
    ap.add_argument("--md", default=str(RESULTS_DIR / "clock_ramp.md"))
    args = ap.parse_args()

    if args.from_json:
        prior = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        load_rows = prior["load_series"]
        idle_rows = prior.get("idle_series") or []
        rep = build(idle_rows, load_rows)
        # the idle summary is not recoverable from the load series alone
        for key in ("mem_mhz", "sm_mhz"):
            for f in ("idle_median", "idle_min", "idle_max"):
                rep[key][f] = prior[key][f]
        rep["n_idle_samples"] = prior["n_idle_samples"]
        v = verdict(rep, args.shortest_protocol)
        rep["verdict"] = v
        rep["args"] = vars(args)
        rep["load_series"] = load_rows
        rep["idle_series"] = idle_rows
        _report(rep, v, args)
        return

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")

    monitor = ClockMonitor(interval_ms=args.interval_ms, max_sm=max_sm_clock())
    if not monitor.start():
        raise SystemExit("nvidia-smi unavailable -- this measurement is the sampler")

    try:
        print(f"idling {args.idle:.0f} s (leave the machine alone)...", flush=True)
        t_idle0 = time.time()
        time.sleep(args.idle)
        t_idle1 = time.time()

        print(f"loading {args.load:.0f} s...", flush=True)
        t_load0 = time.time()
        saturating_load(args.load, monitor)
        t_load1 = time.time()
    finally:
        monitor.stop()

    idle_rows = series(monitor, t_idle0, t_idle1)
    load_rows = series(monitor, t_load0, t_load1)
    rep = build(idle_rows, load_rows)
    v = verdict(rep, args.shortest_protocol)
    rep["verdict"] = v
    rep["args"] = vars(args)
    rep["load_series"] = load_rows

    rep["idle_series"] = idle_rows
    _report(rep, v, args)


def _report(rep: dict, v: dict, args) -> None:
    m, s = rep["mem_mhz"], rep["sm_mhz"]
    print()
    print(f"idle:   mem {m['idle_median']:.0f} MHz   sm {s['idle_median']:.0f} MHz")
    if m["ceiling"]:
        print(f"loaded: mem sustained {m['ceiling']:.0f} MHz   "
              f"sm sustained {s['ceiling']:.0f} MHz")
        if m.get("peak_is_transient"):
            print(f"        (memory boosts to {m['peak']:.0f} MHz for the first "
                  f"{m['peak_seconds']:.1f} s, then settles)")
        t = m.get("first_at_ceiling_s")
        print(f"memory clock reached and held its ceiling: "
              + (f"{t:.1f} s after load start" if t is not None else "never"))
        ts = s.get("first_at_ceiling_s")
        print(f"SM clock reached and held its ceiling:     "
              + (f"{ts:.1f} s after load start" if ts is not None else "never"))
    print()
    if v["settled"]:
        print(f"=> {v['fraction_of_shortest_protocol'] * 100:.1f}% of the shortest "
              f"protocol ({args.shortest_protocol:.0f} s); warm-up arm "
              + ("SURVIVES" if v["warm_up_arm_viable"] else "is NOT viable"))
    else:
        print("=> " + v["note"])

    Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    Path(args.md).write_text(render(rep, v, args), encoding="utf-8")
    print(f"\nwrote {args.out} and {args.md}")


if __name__ == "__main__":
    main()
