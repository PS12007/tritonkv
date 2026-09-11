#!/usr/bin/env python
"""Move the working set across L2 instead of moving L2: the direct test.

    ./.venv/Scripts/python.exe -u l2_sweep.py              # ~6-8 min on the GPU
    ./.venv/Scripts/python.exe l2_sweep.py --from-json results/l2_sweep.json

The project's headline claim is conditional on L2 residency: quantization *costs*
~1.3x when the KV cache fits in L2 and *pays* once it does not. Every number
behind that claim comes from two regimes that `benchmark.py` builds on purpose --
a single cache replayed hot, and a rotating set of replicas ~3x L2 -- at four
context lengths whose fp16 caches (0.5-16.8 MB) all fit in this card's 33.6 MB
L2. So in the hot regime the fp16 control has **never been measured outside L2**.
The conditional has been inferred from two ends and never watched crossing.

This sweep watches it cross. It times the same two kernels the conditional is
built from -- the fp16 control and the fused 4-bit kernel -- on a grid of context
lengths chosen so the **fp16 cache is a fixed multiple of this card's own L2**
(`X_GRID`: 0.125x up to 12x). If the L2 story is right, the hot-regime
quantization ratio has a shape that can be written down before measuring:

* **zone A, both caches fit** (fp16 <= 0.75 L2): quantization costs, ratio < 1.
* **zone B, fp16 spills but 4-bit fits** (fp16 >= 1.5 L2, 4-bit <= 0.75 L2): the
  control reads DRAM while the fused kernel still reads L2. Ratio > 1, and
  *larger than the DRAM-resident ratio at the same context* -- a regime
  `benchmark.py` never produces on this card.
* **zone C, both spill** (4-bit >= 1.5 L2): the hot regime is no longer hot for
  either kernel, so the ratio converges on the DRAM-resident one.

The crossing through 1 should sit near fp16 = 1x L2 -- on *every* card, because
the grid is defined relative to each card's L2. That is what makes this the test
for the cross-GPU plan: the context length of the crossing should move in
proportion to L2 (ctx* ~ L2 / 1024 bytes for this model), while its position in
units of L2 stays put. A "quantization pays past context N" story with no L2 in
it predicts the opposite.

The predictions, their thresholds and the decision rules are pre-registered in
`docs/preregistration_l2.md`, committed before this script was first run on the
GPU. `analyze()` scores them; it does not choose them.

Protocol notes. This is a separate process timing 2 kernels, not 12, so its
absolute numbers at shared contexts need not match `benchmark.py` to better than
the ~5% the protocol study measured. Every prediction here is about a sign, an
ordering, or an effect of 20% or more. The clock ramp, the clock window, the
dispersion gate and the ratio statistic (`bootstrap_ratio_ci` over raw samples)
are the benchmark's own, imported rather than re-implemented.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# The fp16 cache size at each grid point, as a multiple of L2. Dense around 1x,
# where the crossing is predicted, and out far enough that the 4-bit cache spills
# too (4-bit = fp16 / 3.2 at group size 32, so 12x fp16 is 3.75x for 4-bit).
X_GRID = (0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)

# Zone boundaries, as a fraction of nominal L2. Nominal L2 is not all usable --
# associativity, the other tensors, and whatever the replacement policy does to a
# working set replayed in a loop -- so the zones leave a margin on both sides
# rather than pretending the boundary is sharp. Points in the margin are
# measured and shown but not scored.
FIT_FRAC = 0.75
SPILL_FRAC = 1.5

# Pre-registered decision thresholds. Changing any of these after data exists
# is exactly what the pre-registration forbids -- see docs/preregistration_l2.md.
CROSSING_WINDOW = (0.5, 2.0)     # H-L2: where the hot ratio must cross 1, in x_fp16
ZONE_C_TOL = 0.10                # zone C: |hot/cold - 1| at most this
HUMP_WINDOW = (1.0, 4.0)         # where the peak hot ratio must sit, in x_fp16

CONTROL = "triton_fp16_control"
FUSED = "fused_triton_4b"
NBITS = 4


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def fp16_bytes_per_token(hkv: int, d: int) -> int:
    """K and V, fp16, one layer, batch 1."""
    return 2 * hkv * d * 2


def contexts_for(l2_bytes: int, bytes_per_token: int, grid=X_GRID,
                 round_to: int = 128, min_ctx: int = 256) -> list[tuple[float, int]]:
    """(target x, ctx) per grid point, ctx rounded to a multiple of ``round_to``.

    Duplicates -- possible on a small-L2 card, where the lowest grid points all
    round to ``min_ctx`` -- are dropped, keeping the first.
    """
    out, seen = [], set()
    for x in grid:
        ctx = int(round(x * l2_bytes / bytes_per_token / round_to)) * round_to
        ctx = max(min_ctx, ctx)
        if ctx in seen:
            continue
        seen.add(ctx)
        out.append((x, ctx))
    return out


def zone(x16: float, x4: float) -> str:
    if x16 <= FIT_FRAC:
        return "A"
    if x16 >= SPILL_FRAC and x4 <= FIT_FRAC:
        return "B"
    if x4 >= SPILL_FRAC:
        return "C"
    return "-"


ZONE_TEXT = {
    "A": "both fit",
    "B": "fp16 spills, 4-bit fits",
    "C": "both spill",
    "-": "margin (not scored)",
}


# ---------------------------------------------------------------------------
# Measurement (GPU)
# ---------------------------------------------------------------------------


TUNE_GRID = tuple({"block_n": bn, "num_warps": nw, "num_stages": ns}
                  for bn in (32, 64, 128) for nw in (2, 4, 8) for ns in (2, 3))


def tune_fused(q, kq, vq, monitor=None, max_sm=None, passes: int = 2) -> tuple[dict, float, dict]:
    """Pick the fused kernel's config at this context, by launch-free timing.

    Same grid as ``benchmark.tune``, different timer, and the reason is measured.
    The first version of this sweep reused the benchmark's procedure -- one
    launch after an L2 flush, 7 samples -- and at ctx=8192 chose
    ``block_n=64``, which an A/B with the clocks ramped put **26% slower**
    DRAM-resident (26.7 vs 21.3 us) and 15% slower L2-resident, stable over three
    rounds. The benchmark itself picks ``block_n=32`` there in 9 of 9 saved
    runs, so the fault was not the grid: 18 fresh compilations leave the GPU
    idle long enough to drop its clocks, and a flush-then-launch sample at
    ~33 us is mostly not the kernel.

    So: compile every config first, ramp the clocks, then time each one by
    CUDA-graph replay -- the regime every sweep number is measured in -- over
    ``passes`` round-robin passes, keeping each config's best median. A mistuned
    fused kernel biases every ratio here against quantization, which is the
    direction that would *fake* a failure of P3.
    """
    import benchmark as B
    from kernels.fused_decode_attn import fused_decode_attention

    out_buf = torch_empty_out(q)
    fns = []
    for cfg in TUNE_GRID:
        ws: dict = {}
        fn = (lambda cfg=cfg, ws=ws: fused_decode_attention(
            q, kq, vq, out=out_buf, _workspace=ws, **cfg))
        try:
            fn()                       # compile + first-touch, outside any timing
            fns.append((cfg, fn))
        except Exception:
            continue
    import torch
    torch.cuda.synchronize()
    if monitor is not None:
        B.warm_clocks(monitor, max_sm)
    best_by = {}
    for _ in range(passes):
        for cfg, fn in fns:
            ts = B.bench_graph(fn, samples=5, iters=10, warmup=3, min_seconds=0.1)
            if not isinstance(ts, list):
                continue
            key = (cfg["block_n"], cfg["num_warps"], cfg["num_stages"])
            best_by[key] = min(best_by.get(key, float("inf")), statistics.median(ts))
    if not best_by:
        return {"block_n": 64, "num_warps": 4, "num_stages": 2}, float("inf"), {}
    key = min(best_by, key=best_by.get)
    cfg = {"block_n": key[0], "num_warps": key[1], "num_stages": key[2]}
    return cfg, best_by[key], {f"{k[0]}/{k[1]}/{k[2]}": v * 1e3 for k, v in best_by.items()}


def torch_empty_out(q):
    import torch
    B_, HQ, D = q.shape
    return torch.empty((B_, HQ, D), device="cuda", dtype=torch.float32)


def reference_grouped(q, k=None, v=None, kq=None, vq=None, chunk: int = 32768):
    """``reference.reference_decode_attention``'s arithmetic without its memory bill.

    The original expands K and V to all ``HQ`` heads in fp32 -- 4.8 GB at this
    card's largest grid point and ~11 GB on a 72 MB-L2 card, where the grid
    reaches ~880k tokens. Same math here: fp32 scores, fp32 softmax over the
    whole history, fp32 weighted sum. Two differences, both of layout only: the
    query is reshaped to ``(B, HKV, group, D)`` so K/V are never expanded, and
    the history is visited in chunks so at most one chunk is ever widened to
    fp32. ``test_l2_sweep.py`` checks it against the original on CPU.

    Takes either fp16 ``k``/``v`` or quantized ``kq``/``vq``; a quantized cache
    is dequantized one chunk at a time, in fp32, exactly as the original does.
    """
    import torch

    from quantize import QuantizedTensor, dequantize_groupwise

    def rows(t, s, e):
        if isinstance(t, QuantizedTensor):
            part = QuantizedTensor(packed=t.packed[:, :, s:e], scale=t.scale[:, :, s:e],
                                   zero=t.zero[:, :, s:e], nbits=t.nbits,
                                   group_size=t.group_size, head_dim=t.head_dim)
            return dequantize_groupwise(part, torch.float32)
        return t[:, :, s:e].float()

    K = kq if kq is not None else k
    V = vq if vq is not None else v
    B_, HQ, D = q.shape
    HKV = K.packed.shape[1] if kq is not None else K.shape[1]
    S = K.packed.shape[2] if kq is not None else K.shape[2]
    g = HQ // HKV
    qg = q.float().reshape(B_, HKV, g, D) / math.sqrt(D)
    bounds = [(s, min(S, s + chunk)) for s in range(0, S, chunk)]
    scores = torch.cat([torch.einsum("bhgd,bhsd->bhgs", qg, rows(K, s, e))
                        for s, e in bounds], dim=-1)
    probs = torch.softmax(scores, dim=-1)
    out = torch.zeros((B_, HKV, g, D), device=q.device, dtype=torch.float32)
    for s, e in bounds:
        out += torch.einsum("bhgs,bhsd->bhgd", probs[..., s:e], rows(V, s, e))
    return out.reshape(B_, HQ, D)


def cosine(a, b) -> float:
    import torch
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a[None], b[None]).item()


def measure_point(shape, ctx: int, l2_bytes: int, samples: int, monitor, env,
                  vram_budget: int) -> dict:
    """Both kernels, both regimes, at one context length."""
    import torch

    import benchmark as B
    from configs import DEFAULT_GROUP_SIZE
    from kernels.fp16_decode_attn import fp16_decode_attention
    from kernels.fused_decode_attn import fused_decode_attention
    from quantize import quantize_kv
    from reference import make_random_kv

    HQ, HKV, D = shape.num_q_heads, shape.num_kv_heads, shape.head_dim
    bpt = fp16_bytes_per_token(HKV, D)
    fp16_bytes = bpt * ctx

    r16 = B.replicas_for(fp16_bytes, l2_bytes)
    # 4-bit at group size 32 is 3.2x smaller (0.625 bytes per element against 2);
    # the budget check needs a number before the first quantization happens.
    r4 = B.replicas_for(int(fp16_bytes / 3.2), l2_bytes)
    need = r16 * fp16_bytes + r4 * int(fp16_bytes / 3.2) + 2 * fp16_bytes
    if need > vram_budget:
        return {"ctx": ctx, "skipped": f"needs ~{need / 1e9:.2f} GB, budget "
                                       f"{vram_budget / 1e9:.2f} GB"}

    reps = [make_random_kv(1, HQ, HKV, ctx, D, device="cuda", seed=5000 + i)
            for i in range(max(r16, 1))]
    qreps = []
    for i in range(r4):
        qq, kk, vv = reps[i] if i < len(reps) else make_random_kv(
            1, HQ, HKV, ctx, D, device="cuda", seed=7000 + i)
        kq, vq = quantize_kv(kk, vv, NBITS, DEFAULT_GROUP_SIZE)
        qreps.append((qq, kq, vq))
    kq0, vq0 = qreps[0][1], qreps[0][2]
    q4_bytes = kq0.nbytes() + vq0.nbytes()

    # Correctness at this length: the sweep goes far past anything the test
    # suite covers, and a timing of a wrong answer is not a timing.
    q0, k0, v0 = reps[0]
    ref16 = reference_grouped(q0, k0, v0)
    got16 = fp16_decode_attention(q0, k0, v0)
    refq = reference_grouped(qreps[0][0], kq=kq0, vq=vq0)
    cfg, tune_ms, tune_table = tune_fused(qreps[0][0], kq0, vq0, monitor,
                                          env["max_sm_clock_mhz"])
    gotq = fused_decode_attention(qreps[0][0], kq0, vq0, **cfg)
    correctness = {"control_cos_vs_ref": cosine(got16, ref16),
                   "fused_cos_vs_dequant_ref": cosine(gotq, refq)}
    del ref16, got16, refq, gotq

    o16 = torch_empty_out(q0)
    oq = torch_empty_out(q0)
    ws16: dict = {}
    wsq: dict = {}
    fns = {
        CONTROL: [(lambda t=t: fp16_decode_attention(t[0], t[1], t[2], out=o16,
                                                     _workspace=ws16)) for t in reps[:r16]],
        FUSED: [(lambda t=t: fused_decode_attention(t[0], t[1], t[2], out=oq,
                                                    _workspace=wsq, **cfg)) for t in qreps],
    }

    def timed(fn, mem_sensitive):
        holder = {}

        def ramp():
            if monitor is not None:
                holder["inner"] = B.warm_clocks(monitor, env["max_sm_clock_mhz"])

        if monitor is not None:
            B.warm_clocks(monitor, env["max_sm_clock_mhz"], mem_wait_s=0.0)
        span = [None, None]
        t_a = time.time()
        val = fn(span, ramp)
        t_b = time.time()
        lo = span[0] if span[0] is not None else t_a
        hi = span[1] if span[1] is not None else t_b
        w = monitor.window(lo, hi, mem_sensitive=mem_sensitive) if monitor else None
        return val, w

    rows = {}
    for method in (CONTROL, FUSED):
        rec = {}
        for regime in ("cold", "hot"):
            if regime == "cold":
                val, w = timed(lambda sp, wm, f=fns[method]: B.bench_graph_rotating(
                    f, samples, span=sp, warm=wm), mem_sensitive=True)
            else:
                val, w = timed(lambda sp, wm, f=fns[method][0]: B.bench_graph(
                    f, samples, span=sp, warm=wm), mem_sensitive=False)
            if isinstance(val, list):
                st = B.stats(val)
                rec[regime] = {
                    "raw_ms": val,
                    "median_ms": st["median_ms"],
                    "iqr_frac_of_median": st["iqr_frac_of_median"],
                    "tight": B.timing_is_tight(st),
                    "clock_stable": bool(w and w.get("stable")),
                    "sm_mhz_mean": (w or {}).get("sm_mhz_mean"),
                    "mem_mhz_mean": (w or {}).get("mem_mhz_mean"),
                    "n_clock_samples": (w or {}).get("n_samples"),
                }
                rec[regime]["quotable"] = bool(
                    rec[regime]["tight"] and (rec[regime]["clock_stable"] or monitor is None))
            else:
                rec[regime] = {"error": (val or {}).get("error", "no samples")}
        rows[method] = rec

    point = {
        "ctx": ctx,
        "fp16_bytes": fp16_bytes,
        "q4_bytes": q4_bytes,
        "x_fp16": fp16_bytes / l2_bytes,
        "x_q4": q4_bytes / l2_bytes,
        "n_replicas": {CONTROL: r16, FUSED: r4},
        "fused_config": cfg,
        "tune_best_us": tune_ms * 1e3,
        "tune_table_us": tune_table,
        "correctness": correctness,
        "methods": rows,
    }
    del reps, qreps, fns
    torch.cuda.empty_cache()
    return point


def run(args) -> dict:
    import torch

    import benchmark as B
    from configs import DEFAULT_MODEL, load_model_config
    from kernels.fused_decode_attn import triton_available

    ok, why = triton_available()
    if not ok:
        raise SystemExit(f"FATAL: {why}")
    shape, provenance = load_model_config(DEFAULT_MODEL)
    env = B.env_info()
    flusher = B.L2Flusher()
    l2 = flusher.l2_bytes
    if not l2:
        raise SystemExit("FATAL: torch reports no L2 size for this device; the grid "
                         "is defined relative to it, so the sweep cannot be built.")
    bpt = fp16_bytes_per_token(shape.num_kv_heads, shape.head_dim)
    grid = contexts_for(l2, bpt)
    if args.max_points:
        grid = grid[: args.max_points]
    free, total = torch.cuda.mem_get_info()
    budget = int(0.6 * free)

    print(f"gpu   : {env['gpu']}  L2 = {l2 / 1e6:.1f} MB  ({env['sm_count']} SMs)")
    print(f"grid  : fp16 cache = {', '.join(f'{x:g}' for x, _ in grid)} x L2")
    print(f"        ctx        = {', '.join(str(c) for _, c in grid)}")
    print(f"vram  : {free / 1e9:.1f} GB free, sweep budget {budget / 1e9:.1f} GB\n")

    monitor = None
    if not args.no_clock_monitor:
        monitor = B.ClockMonitor(max_sm=env["max_sm_clock_mhz"])
        if not monitor.start():
            print("clocks: nvidia-smi unavailable -- timings are NOT clock-verified")
            monitor = None

    t0 = time.time()
    points = []
    for x, ctx in grid:
        p = measure_point(shape, ctx, l2, args.samples, monitor, env, budget)
        p["x_target"] = x
        points.append(p)
        print(one_line(p), flush=True)
    elapsed = time.time() - t0
    bandwidth = B.dram_bandwidth_probe(l2, monitor)
    if monitor is not None:
        monitor.stop()
    return {
        "kind": "l2_sweep",
        "env": env,
        "bandwidth_probe": bandwidth,
        "model": shape.as_dict(),
        "model_provenance": provenance,
        "args": B.portable_args(args),
        "l2_bytes": l2,
        "x_grid": list(X_GRID),
        "fit_frac": FIT_FRAC,
        "spill_frac": SPILL_FRAC,
        "points": points,
        "wall_clock_seconds": elapsed,
    }


def one_line(p: dict) -> str:
    if p.get("skipped"):
        return f"  ctx={p['ctx']:<7} skipped: {p['skipped']}"
    m = p["methods"]

    def us(method, regime):
        r = m[method].get(regime, {})
        return f"{r['median_ms'] * 1e3:8.1f}" if "median_ms" in r else "     err"

    def ratio(regime):
        a, b = m[CONTROL].get(regime, {}), m[FUSED].get(regime, {})
        if "median_ms" in a and "median_ms" in b:
            return f"{a['median_ms'] / b['median_ms']:.3f}"
        return "  n/a"

    flags = ("ok" if all(m[k].get(r, {}).get("quotable") for k in m for r in ("cold", "hot"))
             else "gate-failed")
    return (f"  ctx={p['ctx']:<7} fp16 {p['x_fp16']:5.2f}xL2  4b {p['x_q4']:5.2f}xL2  "
            f"zone {zone(p['x_fp16'], p['x_q4'])}  "
            f"ctrl {us(CONTROL, 'hot')}/{us(CONTROL, 'cold')} us  "
            f"fused {us(FUSED, 'hot')}/{us(FUSED, 'cold')} us  "
            f"quant hot {ratio('hot')} cold {ratio('cold')}  {flags}")


# ---------------------------------------------------------------------------
# Analysis (CPU only -- runs on any saved sweep)
# ---------------------------------------------------------------------------


def _ratio(point: dict, regime: str):
    """(ratio, lo, hi, quotable) for control / fused in one regime, or None."""
    from audit_claims import bootstrap_ratio_ci

    m = point.get("methods") or {}
    a = (m.get(CONTROL) or {}).get(regime) or {}
    b = (m.get(FUSED) or {}).get(regime) or {}
    if not a.get("raw_ms") or not b.get("raw_ms"):
        return None
    r, lo, hi = bootstrap_ratio_ci(a["raw_ms"], b["raw_ms"])
    return r, lo, hi, bool(a.get("quotable") and b.get("quotable"))


def crossing(xs: list[float], ys: list[float]) -> float | None:
    """First x at which y rises through 1, interpolated in log x.

    ``None`` if y never crosses from below to at-or-above 1. If y already starts
    at or above 1 the crossing is below the grid, reported as ``-inf`` so that it
    fails any window rather than silently passing.
    """
    if not xs:
        return None
    if ys[0] >= 1.0:
        return float("-inf")
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 < 1.0 <= y1:
            if y1 == y0:
                return x1
            f = (1.0 - y0) / (y1 - y0)
            return math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0)))
    return None


def analyze(payload: dict) -> dict:
    """Score the pre-registered predictions against one sweep."""
    pts = []
    for p in payload["points"]:
        if p.get("skipped"):
            continue
        hot = _ratio(p, "hot")
        cold = _ratio(p, "cold")
        if hot is None or cold is None:
            continue
        pts.append({
            "ctx": p["ctx"],
            "x_fp16": p["x_fp16"],
            "x_q4": p["x_q4"],
            "zone": zone(p["x_fp16"], p["x_q4"]),
            "hot": {"ratio": hot[0], "lo": hot[1], "hi": hot[2], "quotable": hot[3]},
            "cold": {"ratio": cold[0], "lo": cold[1], "hi": cold[2], "quotable": cold[3]},
            "correctness": p.get("correctness"),
        })
    pts.sort(key=lambda r: r["x_fp16"])

    preds = []

    def verdict(ok: bool | None) -> str:
        return "untestable" if ok is None else ("HOLDS" if ok else "FAILS")

    # P1 -- zone A: quantization costs. CI entirely below 1.
    za = [r for r in pts if r["zone"] == "A"]
    p1 = None if not za else all(r["hot"]["hi"] < 1.0 for r in za)
    preds.append({"id": "P1", "name": "zone A: hot ratio < 1 (CI below 1)",
                  "n": len(za), "verdict": verdict(p1),
                  "detail": [(r["ctx"], round(r["hot"]["ratio"], 3)) for r in za]})

    # P2 -- the primary: the hot ratio crosses 1 inside the window.
    x_star = crossing([r["x_fp16"] for r in pts], [r["hot"]["ratio"] for r in pts])
    if x_star is None:
        p2 = False if pts else None
    else:
        p2 = CROSSING_WINDOW[0] <= x_star <= CROSSING_WINDOW[1]
    preds.append({"id": "P2", "name": f"hot ratio crosses 1 at fp16 in "
                                      f"[{CROSSING_WINDOW[0]}, {CROSSING_WINDOW[1]}] x L2",
                  "n": len(pts), "verdict": verdict(p2), "x_star": x_star,
                  "ctx_star": (x_star * payload["l2_bytes"] /
                               fp16_bytes_per_token(payload["model"]["num_kv_heads"],
                                                    payload["model"]["head_dim"])
                               if x_star not in (None, float("-inf")) else None)})

    # P3 -- zone B: hot > 1 and hot > cold at the same context, CIs disjoint.
    zb = [r for r in pts if r["zone"] == "B"]
    p3 = None if not zb else all(r["hot"]["lo"] > 1.0 and r["hot"]["lo"] > r["cold"]["hi"]
                                 for r in zb)
    preds.append({"id": "P3", "name": "zone B: hot ratio > 1 and > DRAM-resident ratio",
                  "n": len(zb), "verdict": verdict(p3),
                  "detail": [(r["ctx"], round(r["hot"]["ratio"], 3), round(r["cold"]["ratio"], 3))
                             for r in zb]})

    # P4 -- zone C: the hot regime has converged on the DRAM-resident one.
    zc = [r for r in pts if r["zone"] == "C"]
    p4 = None if not zc else all(abs(r["hot"]["ratio"] / r["cold"]["ratio"] - 1) <= ZONE_C_TOL
                                 for r in zc)
    preds.append({"id": "P4", "name": f"zone C: hot within {ZONE_C_TOL:.0%} of DRAM-resident",
                  "n": len(zc), "verdict": verdict(p4),
                  "detail": [(r["ctx"], round(r["hot"]["ratio"] / r["cold"]["ratio"], 3))
                             for r in zc]})

    # P5 -- the hump: the peak hot ratio sits between 1x and 4x L2.
    if pts:
        peak = max(pts, key=lambda r: r["hot"]["ratio"])
        p5 = HUMP_WINDOW[0] <= peak["x_fp16"] <= HUMP_WINDOW[1]
        preds.append({"id": "P5", "name": f"peak hot ratio at fp16 in "
                                          f"[{HUMP_WINDOW[0]}, {HUMP_WINDOW[1]}] x L2",
                      "n": len(pts), "verdict": verdict(p5),
                      "peak": {"ctx": peak["ctx"], "x_fp16": peak["x_fp16"],
                               "ratio": peak["hot"]["ratio"]}})
    else:
        preds.append({"id": "P5", "name": "peak position", "n": 0, "verdict": "untestable"})

    n_quot = sum(1 for r in pts if r["hot"]["quotable"] and r["cold"]["quotable"])
    return {
        "gpu": payload["env"]["gpu"],
        "l2_bytes": payload["l2_bytes"],
        "points": pts,
        "predictions": preds,
        "n_points": len(pts),
        "n_points_quotable": n_quot,
        "correctness_min_cos": min(
            (min(r["correctness"].values()) for r in pts if r.get("correctness")),
            default=None),
    }


def render(rep: dict) -> str:
    L = rep["l2_bytes"] / 1e6
    lines = [f"# L2 sweep -- {rep['gpu']} (L2 {L:.1f} MB)", ""]
    lines.append("Hot = one cache replayed (L2-resident only if it fits); cold = rotating "
                 "replicas >= 3x L2. Ratio = fp16 control / fused 4-bit, >1 means "
                 "quantization pays. `bootstrap_ratio_ci` over raw samples; `*` = a row "
                 "in that pair failed the gate.")
    lines.append("")
    lines.append("| ctx | fp16 x L2 | 4-bit x L2 | zone | quant hot [95% CI] | "
                 "quant cold [95% CI] | hot/cold |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rep["points"]:
        h, c = r["hot"], r["cold"]
        lines.append(
            f"| {r['ctx']} | {r['x_fp16']:.2f} | {r['x_q4']:.2f} | {r['zone']} "
            f"| {h['ratio']:.3f} [{h['lo']:.3f}, {h['hi']:.3f}]{'' if h['quotable'] else '*'} "
            f"| {c['ratio']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]{'' if c['quotable'] else '*'} "
            f"| {h['ratio'] / c['ratio']:.3f} |")
    lines.append("")
    lines.append(f"Zones: A = {ZONE_TEXT['A']} (fp16 <= {FIT_FRAC}x); B = {ZONE_TEXT['B']} "
                 f"(fp16 >= {SPILL_FRAC}x, 4-bit <= {FIT_FRAC}x); C = {ZONE_TEXT['C']} "
                 f"(4-bit >= {SPILL_FRAC}x); `-` = {ZONE_TEXT['-']}.")
    lines.append("")
    lines.append(f"Quotable pairs: {rep['n_points_quotable']} of {rep['n_points']}. "
                 f"Worst correctness cosine anywhere in the sweep: "
                 f"{rep['correctness_min_cos']:.7f}" if rep["correctness_min_cos"] is not None
                 else "No correctness data.")
    lines.append("")
    lines.append("## Pre-registered predictions (docs/preregistration_l2.md)")
    lines.append("")
    for p in rep["predictions"]:
        extra = ""
        if p["id"] == "P2":
            xs = p.get("x_star")
            if xs is None:
                extra = " -- no crossing found"
            elif xs == float("-inf"):
                extra = " -- ratio already >= 1 at the first grid point"
            else:
                extra = f" -- crossing at fp16 = {xs:.2f}x L2 (ctx ~ {p['ctx_star']:.0f})"
        elif p["id"] == "P5" and p.get("peak"):
            pk = p["peak"]
            extra = f" -- peak {pk['ratio']:.3f} at ctx {pk['ctx']} ({pk['x_fp16']:.2f}x L2)"
        elif p.get("detail"):
            extra = f" -- {p['detail']}"
        lines.append(f"- **{p['id']} {p['verdict']}** ({p['n']} pts): {p['name']}{extra}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--max-points", type=int, default=None,
                    help="stop after this many grid points (smoke testing only)")
    ap.add_argument("--no-clock-monitor", action="store_true")
    ap.add_argument("--from-json", default=None,
                    help="re-analyze a saved sweep without touching the GPU")
    ap.add_argument("--out", default=str(RESULTS_DIR / "l2_sweep.json"))
    ap.add_argument("--md", default=str(RESULTS_DIR / "l2_sweep.md"))
    args = ap.parse_args()

    if args.from_json:
        payload = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    else:
        payload = run(args)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}  ({payload['wall_clock_seconds']:.0f} s of timing)")

    rep = analyze(payload)
    md = render(rep)
    Path(args.md).write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
