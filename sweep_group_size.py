#!/usr/bin/env python
"""Group-size sweep on both metadata paths, as a mechanism test.

    python sweep_group_size.py                    # full sweep -> results/gs_sweep.json
    python sweep_group_size.py --quick            # one context, fewer samples

**Why this exists.** The README used to say: *"Group size barely moves it
(25.4 / 25.6 / 26.6 us at gs = 16/32/64), so the scale+zero tile loads are not
the cost."* The measurement was fine; the inference was wrong. That sweep ran on
the **gather** path, where the metadata tile is indexed by ``d // GS`` across the
full head dim -- so it issues ``BLOCK_N * D`` loads no matter what ``GS`` is.
Group size changed how many *distinct* values were read and never how many
instructions were issued. A flat sweep is exactly what the expensive version
predicts, so the flat sweep was evidence for nothing.

On the **broadcast** path ``GS`` genuinely sets the load count
(``BLOCK_N * D // GS``), so the same sweep should now be *sloped*.

Running both paths in one session on one card turns a timing anecdote into a
mechanism test with a falsifiable prediction made in advance:

    broadcast: sloped, and monotone in D // GS
    gather:    flat

If the broadcast path comes out flat too, the metadata-load story is wrong and
the 1.16-1.48x measured for the broadcast change is coming from somewhere else.
If the *gather* path comes out sloped, then group size is reaching the kernel
through some channel other than the metadata loads -- bytes moved, say -- and
the same caveat that sinks the original claim would sink this one.

Static counts (registers, PTX instructions, shared-memory traffic) are read off
the compiled kernel in the same run, because they are the mechanism and they cost
nothing to collect.

Everything is measured through ``benchmark.py``'s primitives -- the same clock
ramp, the same clock window, the same quotability gate -- so a row here is
quotable on the same terms as a row there.

**Outcome (2026-09-01, ctx = 512 / 2048 / 8192, 4-bit, block_n=32, 2 warps).**
The prediction above was half right, and the half that was wrong is the useful
half.

*Gather came out flat*, as predicted, at gs = 16/32/64 -- and then fell off a
cliff at gs=128, which the prediction did not anticipate at all. See below.

*Broadcast came out nearly flat too*, which refutes the stated prediction.
Across an 8x range of metadata loads (gs=16 -> gs=128 is 256 -> 32 loads per
tile) the L2-resident time moves 1.07-1.17x, and monotonically in the predicted
direction, but barely. Put next to the step the broadcast change itself made,
the shape is **saturation**, not proportionality (L2-resident, ctx=8192):

    gather, gs=32     4096 loads/tile   22.0 us
    broadcast, gs=16   256 loads/tile   17.0 us   <- 16x fewer loads buys 1.29x
    broadcast, gs=128   32 loads/tile   15.9 us   <- a further 8x buys 1.07x

So metadata loads are a real cost, and they stop being the binding cost once
they are roughly an order of magnitude down. This is the honest correction to
"loading them at their real width is what made the kernel fast": that step was
worth 1.29x because it crossed the saturation point, and there is almost nothing
left below it.

**The gs=128 cliff on the gather path.** 1.95x / 2.88x / 3.52x slower than gs=64
at ctx = 512 / 2048 / 8192 (L2-resident), with an IQR of ~1%, clock-verified at
2048 and 8192. This is the same outlier the old README recorded as "93 us at
gs=128" and never explained. It is not a load-count effect -- that cell issues
*fewer* PTX instructions than gs=64 (2415 vs 2989) and the same number of global
loads. What changes is shared memory: ``st.shared`` goes 30 -> 142, reproducibly,
across all nine (block_n, num_warps) combinations checked in ``probe_gs128.py``.

The cause is the degenerate index. At GS == head_dim, ``tl.arange(0, D) // GS``
folds to all-zeros, and Triton responds by giving the loaded tile a layout that
has to be converted through shared memory before it can feed the dequantize
path. The redundant-load form is bad; the redundant-load form with a *constant*
index is much worse. The shipped broadcast path is unaffected (it is the
*fastest* cell at gs=128), so this is a fact about the control, not about the
kernel -- but it is the explanation for a number that sat unexplained in this
repo from the first week.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from benchmark import (
    ClockMonitor,
    L2Flusher,
    bench_graph,
    bench_graph_rotating,
    env_info,
    replicas_for,
    stats,
    timing_is_tight,
    warm_clocks,
)
from configs import DEFAULT_MODEL, load_model_config
from quantize import dequantize_groupwise, quantize_kv
from reference import make_random_kv, reference_decode_attention

RESULTS_DIR = Path(__file__).parent / "results"

# 128 == head_dim: one group spanning the whole vector, i.e. the fewest possible
# metadata loads. It is included because it is the end of the axis, and because
# the old sweep reported a 93 us outlier there that nobody explained.
GROUP_SIZES = (16, 32, 64, 128)
PATHS = (("broadcast", True), ("gather", False))


def compiled_stats(kernel_module, before: set) -> dict:
    """Registers, PTX instruction count, and the shared-memory profile.

    The shared-memory counts are here because they turned out to be the whole
    story for the gs=128 gather cliff: that cell issues *fewer* PTX instructions
    than gs=64 and runs 2.8x slower, which is not a load-count signature. See
    ``probe_gs128.py``.
    """
    cache = kernel_module._fused_decode_attn_split.device_caches[0][0]
    new = [k for k in cache if k not in before]
    if not new:
        return {}
    ks = [cache[k] for k in new]
    out = {"n_regs": max(k.n_regs for k in ks), "n_spills": max(k.n_spills for k in ks)}
    ptx = None
    for k in ks:
        asm = getattr(k, "asm", None) or {}
        if "ptx" in asm:
            ptx = asm["ptx"]
            break
    if ptx is not None:
        # Count real instructions: lines ending in ';' that are not directives,
        # labels or predicate-only declarations. This is a proxy for issue cost,
        # not a SASS count, but it is measured the same way for every cell so
        # the *differences* are meaningful even if the absolute number is not.
        n = 0
        for line in ptx.splitlines():
            s = line.strip()
            if not s or s.startswith(("//", ".", "$", "{", "}")) or not s.endswith(";"):
                continue
            if s.startswith(".reg") or s.startswith(".param"):
                continue
            n += 1
        out["ptx_instructions"] = n
        try:
            from probe_gs128 import ptx_profile
            out["ptx_ops"] = ptx_profile(ptx)
        except Exception:
            pass
    md = getattr(ks[0], "metadata", None)
    out["shared_bytes"] = getattr(md, "shared", None)
    return out


def measure_cell(ctx, gs, bcast, cfg, shape, batch, monitor, max_sm, l2_bytes,
                 samples, nbits=4):
    """One (context, group size, path) cell: static counts + both regimes."""
    import kernels.fused_decode_attn as K
    from kernels.fused_decode_attn import fused_decode_attention

    HQ, HKV, D = shape.num_q_heads, shape.num_kv_heads, shape.head_dim

    q, k, v = make_random_kv(batch, HQ, HKV, ctx, D, device="cuda", seed=1234)
    kq, vq = quantize_kv(k, v, nbits, gs)
    cache_bytes = kq.packed.numel() * kq.packed.element_size() * 2 + (
        kq.scale.numel() * kq.scale.element_size() * 4
    )
    n_rep = replicas_for(cache_bytes, l2_bytes)

    out_buf = torch.empty((batch, HQ, D), device="cuda", dtype=torch.float32)
    ws: dict = {}

    def call():
        return fused_decode_attention(
            q, kq, vq, out=out_buf, _workspace=ws, meta_bcast=bcast, **cfg
        )

    # --- static counts, before any timing -----------------------------------
    seen = set(K._fused_decode_attn_split.device_caches[0][0])
    call()
    torch.cuda.synchronize()
    static = compiled_stats(K, seen)

    # --- accuracy, since group size is an accuracy knob as much as a speed one.
    # Two different questions, and conflating them is easy: `vs_fp16_truth` is
    # how much the *quantization* costs (percent-level, and the thing group size
    # trades against), while `vs_dequant_ref` is whether the *kernel* computes
    # what the quantized cache says it should (should stay ~1e-4 whatever GS is).
    ref_true = reference_decode_attention(q, k, v)
    ref_deq = reference_decode_attention(
        q, dequantize_groupwise(kq, torch.float32), dequantize_groupwise(vq, torch.float32)
    )
    got = call().to(torch.float32)

    def _rel(a, b):
        return (torch.linalg.vector_norm(a - b)
                / torch.linalg.vector_norm(b).clamp_min(1e-12)).item()

    rel_l2 = _rel(got, ref_true)
    rel_l2_kernel = _rel(got, ref_deq)

    # --- replicas for the DRAM-resident regime ------------------------------
    reps = []
    for i in range(n_rep):
        _, ki, vi = make_random_kv(batch, HQ, HKV, ctx, D, device="cuda", seed=1234 + i)
        reps.append(quantize_kv(ki, vi, nbits, gs))
    bufs = [torch.empty((batch, HQ, D), device="cuda", dtype=torch.float32)
            for _ in reps]
    wss = [{} for _ in reps]
    fns = [
        (lambda kqi=kqi, vqi=vqi, b=b, w=w: fused_decode_attention(
            q, kqi, vqi, out=b, _workspace=w, meta_bcast=bcast, **cfg))
        for (kqi, vqi), b, w in zip(reps, bufs, wss)
    ]
    for f in fns:
        f()
    torch.cuda.synchronize()

    def timed(fn):
        holder = {}

        def ramp():
            if monitor is not None:
                holder["inner"] = warm_clocks(monitor, max_sm)

        r = warm_clocks(monitor, max_sm) if monitor is not None else None
        span = [None, None]
        t_a = time.time()
        val = fn(span, ramp)
        t_b = time.time()
        lo = span[0] if span[0] is not None else t_a
        hi = span[1] if span[1] is not None else t_b
        w = monitor.window(lo, hi) if monitor is not None else None
        return val, {"ramp": r, "ramp_inner": holder.get("inner"), "clocks": w,
                     "sampling_seconds": hi - lo}

    cold, cold_clk = timed(lambda sp, w: bench_graph_rotating(fns, samples, span=sp, warm=w))
    hot, hot_clk = timed(lambda sp, w: bench_graph(call, max(10, samples // 2), span=sp, warm=w))

    cold_st = stats(cold) if isinstance(cold, list) else None
    hot_st = stats(hot) if isinstance(hot, list) else None
    clock_ok = bool((cold_clk["clocks"] or {}).get("stable")) and \
        bool((hot_clk["clocks"] or {}).get("stable"))

    del reps, bufs, fns
    torch.cuda.empty_cache()

    return {
        "ctx": ctx,
        "nbits": nbits,
        "group_size": gs,
        "path": "broadcast" if bcast else "gather",
        "config": cfg,
        "n_groups": D // gs,
        "meta_loads_per_tile": cfg["block_n"] * (D // gs if bcast else D),
        "n_replicas": n_rep,
        "footprint_bytes": n_rep * cache_bytes,
        "footprint_over_l2": (n_rep * cache_bytes / l2_bytes) if l2_bytes else None,
        "rel_l2_vs_fp16_truth": rel_l2,
        "rel_l2_vs_dequant_ref": rel_l2_kernel,
        **static,
        "cold": cold_st,
        "cold_raw_ms": cold if isinstance(cold, list) else None,
        "hot": hot_st,
        "hot_raw_ms": hot if isinstance(hot, list) else None,
        "clocks": {"cold": cold_clk["clocks"], "hot": hot_clk["clocks"]},
        "clock_verified": clock_ok,
        "timing_tight": timing_is_tight(cold_st) and timing_is_tight(hot_st),
        "quotable": clock_ok and timing_is_tight(cold_st) and timing_is_tight(hot_st),
    }


def slope_report(rows: list[dict], regime: str) -> str:
    """Is each path's sweep sloped or flat, in its own terms?

    Reported as max/min across the group sizes for one path at one context: a
    ratio near 1.00 is a flat sweep, and anything well above it is a sloped one.
    Deliberately not a regression -- with four points and a mechanism that
    predicts a *specific ordering*, the spread and the ordering are the whole
    content of the result.
    """
    lines = []
    for ctx in sorted({r["ctx"] for r in rows}):
        for path in ("broadcast", "gather"):
            sel = sorted((r for r in rows if r["ctx"] == ctx and r["path"] == path),
                         key=lambda r: r["group_size"])
            vals = [(r["group_size"], (r[regime] or {}).get("median_ms"), r["quotable"])
                    for r in sel]
            vals = [(g, t, q) for (g, t, q) in vals if t]
            if len(vals) < 2:
                continue
            ts = [t for (_, t, _) in vals]
            spread = max(ts) / min(ts)
            allq = all(q for (_, _, q) in vals)
            body = "  ".join(f"gs={g}:{t * 1e3:.1f}us" + ("" if q else "*")
                             for (g, t, q) in vals)
            lines.append(f"  ctx={ctx:<6} {path:<9} {body}   max/min={spread:.2f}x"
                         + ("" if allq else "   (* = not clock/dispersion verified)"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-clock-monitor", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_DIR / "gs_sweep.json"))
    args = ap.parse_args()

    contexts = [2048] if args.quick else args.contexts
    samples = 10 if args.quick else args.samples

    shape, provenance = load_model_config(args.model)
    env = env_info()
    print(f"model : {shape.name} ({provenance})")
    print(f"gpu   : {env['gpu']} sm{env['compute_capability']} "
          f"({env['sm_count']} SMs, L2={(env['l2_cache_bytes'] or 0) / 1e6:.0f} MB)")

    # One fixed config for the whole sweep. Retuning per group size would let the
    # tuner absorb the very effect being measured -- the point is what GS does at
    # a fixed block size, not what the best kernel at each GS costs.
    bench = RESULTS_DIR / "benchmark.json"
    cfg = {"block_n": 32, "num_warps": 4, "num_stages": 2}
    if bench.exists():
        tuned = json.load(open(bench, encoding="utf-8")).get("tuned_config", {})
        pick = tuned.get(f"{contexts[-1]}_4bit")
        if pick:
            cfg = dict(pick)
    print(f"config: {cfg} (fixed across the sweep)")

    monitor = None
    if not args.no_clock_monitor:
        monitor = ClockMonitor(max_sm=env["max_sm_clock_mhz"])
        if not monitor.start():
            print("clocks: nvidia-smi unavailable -- rows are NOT clock-verified")
            monitor = None

    l2_bytes = L2Flusher().l2_bytes
    rows = []
    t0 = time.time()
    for ctx in contexts:
        print(f"\n  --- context {ctx} ---")
        for gs in GROUP_SIZES:
            for name, bcast in PATHS:
                r = measure_cell(ctx, gs, bcast, cfg, shape, args.batch, monitor,
                                 env["max_sm_clock_mhz"], l2_bytes, samples)
                rows.append(r)
                cold = (r["cold"] or {}).get("median_ms", float("nan")) * 1e3
                hot = (r["hot"] or {}).get("median_ms", float("nan")) * 1e3
                print(f"    gs={gs:<4} {name:<9} "
                      f"regs={r.get('n_regs', '?'):<4} "
                      f"ptx={r.get('ptx_instructions', '?'):<5} "
                      f"loads/tile={r['meta_loads_per_tile']:<5} "
                      f"DRAM={cold:7.1f}us  L2={hot:6.1f}us  "
                      f"smem_st={(r.get('ptx_ops') or {}).get('st.shared', '?'):<4} "
                      f"quantErr={r['rel_l2_vs_fp16_truth']:.2e} "
                      f"kernErr={r['rel_l2_vs_dequant_ref']:.1e}  "
                      f"{'quotable' if r['quotable'] else 'REJECTED'}")

    if monitor is not None:
        monitor.stop()

    payload = {
        "env": env,
        "model": {"name": shape.name, "head_dim": shape.head_dim},
        "args": vars(args),
        "config": cfg,
        "group_sizes": list(GROUP_SIZES),
        "rows": rows,
        "wall_clock_seconds": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)

    print("\nDRAM-resident (cold):")
    print(slope_report(rows, "cold"))
    print("\nL2-resident (hot):")
    print(slope_report(rows, "hot"))
    print(f"\nwrote {args.out}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
