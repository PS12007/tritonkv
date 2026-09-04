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
    """When did `key` first reach, and then hold, its loaded ceiling?

    Only samples taken under load define the ceiling: at idle this part reports a
    *higher* memory clock (12001 MHz) than it ever sustains under load, so an
    idle-inclusive maximum would set a target the measurement can never reach --
    the exact bug `benchmark.py` had to fix in its ramp.
    """
    loaded = [r for r in rows if r.get("util_pct", 0.0) >= LOADED_UTIL_PCT]
    if not loaded:
        return {"n_loaded": 0, "ceiling": None}
    ceiling = max(r[key] for r in loaded)
    threshold = ceiling * CEILING_FRAC
    first = None
    arrived = None
    for i, r in enumerate(loaded):
        if r[key] < threshold:
            first = None
            continue
        if first is None:
            first = r["t"]
        if r["t"] - first >= HOLD_SECONDS:
            arrived = first
            break
    # A ramp that reaches the ceiling and holds it to the end of the window
    # counts, even if the window ended before HOLD_SECONDS elapsed.
    if arrived is None and first is not None and loaded[-1][key] >= threshold:
        arrived = first
    return {
        "n_loaded": len(loaded),
        "ceiling": float(ceiling),
        "threshold": float(threshold),
        "first_at_ceiling_s": (float(arrived) if arrived is not None else None),
        "held_to_end": bool(arrived is not None
                            and loaded[-1][key] >= threshold),
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
    ap.add_argument("--out", default=str(RESULTS_DIR / "clock_ramp.json"))
    ap.add_argument("--md", default=str(RESULTS_DIR / "clock_ramp.md"))
    args = ap.parse_args()

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

    m, s = rep["mem_mhz"], rep["sm_mhz"]
    print()
    print(f"idle:   mem {m['idle_median']:.0f} MHz   sm {s['idle_median']:.0f} MHz")
    if m["ceiling"]:
        print(f"loaded: mem ceiling {m['ceiling']:.0f} MHz   "
              f"sm ceiling {s['ceiling']:.0f} MHz")
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
