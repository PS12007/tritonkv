"""Wall-clock and memory benchmark for the fused quantized-KV decode kernel.

    python benchmark.py                 # full suite, writes results/benchmark.json
    python benchmark.py --quick         # smoke run (2 context lengths, few samples)

Methodology notes, because the methodology is the deliverable here as much as
the number is:

**Every configuration is sampled 50 times** (``--samples``) and reported as
mean +- std over those samples, never as a single run or a hand-picked best.
The raw per-sample timings are written to the JSON so anyone can recompute the
statistics or check the distribution.

**L2 is flushed between samples ("cold" regime).** This turns out to matter more
than anything else in this benchmark. A 16k-token 4-bit KV cache for one layer
is about 5 MB, and this GPU's L2 is tens of MB -- so a naive back-to-back
benchmark loop leaves the entire cache resident in L2 and measures something
that cannot happen during real decoding, where 28 layers cycle through the cache
and evict each other. The "hot" regime is *also* measured and reported, because
the gap between the two is itself an honest finding about how easy it is to
overstate a memory-bound kernel's speedup.

**Launch overhead is separated from compute.** Each method is timed three ways:
single cold call (what a real decode step costs), pipelined back-to-back calls
(launch latency partly hidden), and CUDA-graph replay (launch latency removed).
``pipelined - graph`` is the exposed launch overhead.

**Scope.** This measures *one attention layer's decode step*, not end-to-end
model tokens/sec. A whole-model projection is derived from it and is clearly
labelled as a projection -- it assumes the non-attention work is unchanged,
which it is, since this kernel only replaces attention.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from configs import BIT_WIDTHS, CONTEXT_LENGTHS, DEFAULT_GROUP_SIZE, DEFAULT_MODEL, load_model_config
from kernels.fp16_decode_attn import fp16_decode_attention
from kernels.fused_decode_attn import fused_decode_attention, triton_available
from quantize import dequantize_groupwise, quantize_kv
from reference import (
    baseline_dequant_sdpa,
    baseline_dequant_sdpa_compiled,
    baseline_fp16_sdpa,
    make_random_kv,
    reference_decode_attention,
)

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# GPU clock monitoring
#
# This is a laptop RTX 5060: an 80 W part whose SM clock idles near 285 MHz and
# boosts to 3090 MHz. A kernel timed while the clocks are still ramping is timed
# on a different GPU than one timed at boost, and the difference is larger than
# most of the effects this benchmark is trying to measure. So: sample the clocks
# throughout the run, drive them up before every measurement, and record what
# they actually were for each row. A row whose clocks moved is marked unstable
# and must not be quoted.
# ---------------------------------------------------------------------------

SMI_QUERY = "clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu"
SMI_KEYS = ("sm_mhz", "mem_mhz", "temp_c", "power_w", "util_pct")


def smi_once() -> dict | None:
    """One nvidia-smi sample, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        return {k: float(v) for k, v in zip(SMI_KEYS, parts)}
    except Exception:
        return None


def max_sm_clock() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip().splitlines()[0]) if out.returncode == 0 else None
    except Exception:
        return None


class ClockMonitor:
    """Background nvidia-smi sampler.

    Runs a single long-lived ``nvidia-smi -lms`` process rather than one
    subprocess per sample, so the monitoring itself costs ~nothing and does not
    show up in the thing being monitored.
    """

    def __init__(self, interval_ms: int = 100, max_sm: float | None = None):
        self.interval_ms = interval_ms
        self.max_sm = max_sm
        self.samples: list[tuple[float, dict]] = []
        self.proc = None
        self._thread = None
        self._stop = threading.Event()
        self.available = False

    def start(self) -> bool:
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={SMI_QUERY}",
                 "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except Exception:
            self.proc = None
            return False
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        time.sleep(0.6)
        self.available = len(self.samples) > 0
        return self.available

    def _read(self):
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != len(SMI_KEYS):
                continue
            try:
                self.samples.append((time.time(), {k: float(v) for k, v in zip(SMI_KEYS, parts)}))
            except ValueError:
                continue

    def latest(self) -> dict | None:
        return self.samples[-1][1] if self.samples else None

    def window(self, t0: float, t1: float) -> dict | None:
        """Summarize the clocks observed while [t0, t1] was being measured."""
        vals = [s for (t, s) in self.samples if t0 <= t <= t1]
        if not vals:
            near = [s for (t, s) in self.samples if t0 - 1.0 <= t <= t1 + 1.0]
            if not near:
                return None
            vals = near
        summary = {"n_samples": len(vals)}
        for k in SMI_KEYS:
            xs = [v[k] for v in vals]
            summary[f"{k}_min"] = min(xs)
            summary[f"{k}_max"] = max(xs)
            summary[f"{k}_mean"] = statistics.fmean(xs)
        sm_mean = summary["sm_mhz_mean"] or 1.0
        summary["sm_spread_frac"] = (summary["sm_mhz_max"] - summary["sm_mhz_min"]) / sm_mean
        summary["mem_clock_constant"] = summary["mem_mhz_min"] == summary["mem_mhz_max"]
        # "Boosted" is the condition that matters, not "perfectly constant". A
        # boosting GPU dithers by a few percent from one nvidia-smi sample to the
        # next even under steady load; what invalidates a measurement is running
        # it at *idle* clocks (285 MHz here, 9x slower than boost). So the gate is
        # a floor on the minimum observed clock, and the timing dispersion below
        # is what actually certifies that the number is reproducible.
        summary["boosted"] = bool(
            self.max_sm and summary["sm_mhz_min"] >= CLOCK_BOOST_FLOOR_FRAC * self.max_sm
        )
        summary["thermal_headroom"] = summary["temp_c_max"] < CLOCK_TEMP_LIMIT_C
        # A window summarized from one or two nvidia-smi samples is not evidence
        # that the GPU held its clocks -- it is evidence that the measurement was
        # shorter than the sampler's period. The fast kernels are exactly the ones
        # that finish inside a single 100 ms tick, and they are also the ones a
        # clock artefact flatters most, so "not enough samples to tell" has to be
        # a distinct failure from "clocks sagged" rather than a silent pass.
        summary["enough_samples"] = summary["n_samples"] >= MIN_CLOCK_SAMPLES
        summary["stable"] = bool(
            summary["boosted"] and summary["thermal_headroom"] and summary["enough_samples"]
        )
        return summary

    def stop(self):
        self._stop.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass


CLOCK_BOOST_FLOOR_FRAC = 0.70   # every sample must be >= this fraction of max SM clock
CLOCK_TEMP_LIMIT_C = 87.0       # above this, assume thermal throttling
CLOCK_TARGET_FRAC = 0.80        # ramp to this fraction of max SM clock before timing
MAX_IQR_FRAC = 0.05             # a quotable timing's IQR must be <= 5% of its median
MIN_CLOCK_SAMPLES = 4           # a clock window built from fewer samples proves nothing
# nvidia-smi -lms 100 delivers ~9 Hz in practice (measured: 109 ms median gap),
# so a window has to stay open well past a second to collect a usable number of
# clock samples. 1.5 s buys ~14.
MIN_SAMPLING_SECONDS = 1.5


def timing_is_tight(st) -> bool:
    """Did the samples themselves agree, independent of what the clocks did?"""
    return bool(
        isinstance(st, dict)
        and st.get("iqr_frac_of_median") is not None
        and st["iqr_frac_of_median"] <= MAX_IQR_FRAC
    )


def warm_clocks(monitor: ClockMonitor | None, max_sm: float | None,
                target_frac: float = CLOCK_TARGET_FRAC, timeout_s: float = 20.0,
                min_spin_s: float = 0.4) -> dict:
    """Spin the GPU until the SM clock boosts, so timings start at boost clocks.

    ``nvidia-smi -lgc`` would pin the clocks outright, but it needs
    administrator rights, so the portable substitute is to give the GPU enough
    work that its own governor ramps up, then measure immediately afterwards.
    """
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    t0 = time.time()
    target = (max_sm * target_frac) if max_sm else None
    reached = False
    while True:
        for _ in range(24):
            a = (a @ b) * 0.0009765625
        torch.cuda.synchronize()
        el = time.time() - t0
        cur = monitor.latest() if (monitor and monitor.available) else None
        if target and cur and cur["sm_mhz"] >= target:
            reached = True
            if el >= min_spin_s:
                break
        if el >= timeout_s or (target is None and el >= min_spin_s):
            break
    del a, b
    cur = monitor.latest() if (monitor and monitor.available) else None
    return {
        "spin_seconds": time.time() - t0,
        "sm_mhz_after": cur["sm_mhz"] if cur else None,
        "target_mhz": target,
        "reached_target": reached,
    }


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


def mark(span: list | None, i: int):
    """Record the wall-clock edge of a sampling loop, for clock attribution."""
    if span is not None:
        span[i] = time.time()


class L2Flusher:
    """Writes a buffer several times larger than L2 to evict everything."""

    def __init__(self):
        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            l2 = getattr(props, "L2_cache_size", 0) or 0
        except Exception:
            l2 = 0
        nbytes = max(int(64e6), int(4 * l2))
        self.buf = torch.empty(nbytes, dtype=torch.int8, device="cuda")
        self.nbytes = nbytes
        self.l2_bytes = l2

    def flush(self):
        self.buf.zero_()


def bench_cold(fn, flusher: L2Flusher, samples: int, warmup: int = 15,
               span: list | None = None, warm=None) -> list[float]:
    """Per-call latency in ms, with L2 flushed before every sample.

A ``span`` list, if given, is filled with the wall-clock ``[start, end]`` of the
*sampling loop only*. The clock monitor uses it so that warmup, graph capture
and other untimed setup -- during which the GPU is free to drop back to idle
clocks -- are not mistaken for a throttled measurement.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    if warm is not None:
        warm()
    mark(span, 0)
    for i in range(samples):
        flusher.flush()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    mark(span, 1)
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def bench_pipelined(fn, samples: int, iters: int = 50, warmup: int = 15,
                    span: list | None = None, warm=None) -> list[float]:
    """Per-call latency in ms for back-to-back calls with a hot cache."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    out = []
    if warm is not None:
        warm()
    mark(span, 0)
    for _ in range(samples):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / iters)
    mark(span, 1)
    return out


def bench_graph(fn, samples: int, iters: int = 50, warmup: int = 15,
                span: list | None = None, warm=None,
                min_seconds: float = MIN_SAMPLING_SECONDS):
    """Per-call latency in ms via CUDA-graph replay (launch overhead removed).

    Returns ``None`` if the method cannot be captured (e.g. it allocates, or
    torch.compile emitted something capture-hostile) -- that is reported as
    such rather than silently skipped.
    """
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
        torch.cuda.synchronize()

        # Ramp the clocks *after* capture, then prime. Graph capture is a long
        # CPU-side stretch with the GPU nearly idle, so a ramp done before it
        # has decayed by the time the sampling loop opens -- which is exactly
        # how the fastest methods came to fail the boost gate while the slow
        # ones passed. Priming after the ramp also puts the cache back, since
        # the ramp's GEMM evicts it.
        if warm is not None:
            warm()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()

        out = []
        mark(span, 0)
        t_start = time.time()
        while True:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            out.append(s.elapsed_time(e) / iters)
            elapsed = time.time() - t_start
            if len(out) >= samples and (
                elapsed >= min_seconds or elapsed >= MAX_SAMPLING_SECONDS
            ):
                break
        mark(span, 1)
        del g
        return out
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


MAX_REPLICAS = 512
MAX_REPLICA_BYTES = 400_000_000
# Ceiling on how long a stretched measurement may run. The stretch is bounded in
# *time*, not in sample count: a count cap silently binds first for the fastest
# kernels -- the ones that need the stretch -- because their samples are cheap,
# which is how a 0.26 s window ended up being asked to yield 4 clock samples at
# 9 Hz.
MAX_SAMPLING_SECONDS = 6.0


def replicas_for(cache_bytes: int, l2_bytes: int) -> int:
    """How many independent copies of the cache to rotate through.

    Enough that the whole set cannot sit in L2, capped so a 16k fp16 cache does
    not eat the 8 GB card. The achieved footprint is reported alongside the
    timing so a case that failed to reach the target is visible rather than
    quietly optimistic.
    """
    target = max(3 * (l2_bytes or 0), 96_000_000)
    if cache_bytes <= 0:
        return 1
    want = -(-target // cache_bytes)
    by_bytes = MAX_REPLICA_BYTES // cache_bytes
    return int(max(1, min(MAX_REPLICAS, want, by_bytes)))


def bench_graph_rotating(fns, samples: int, warmup: int = 3, span: list | None = None,
                         warm=None, min_seconds: float = MIN_SAMPLING_SECONDS):
    """Per-call GPU time over a working set too large for L2.

    This replaces the obvious "flush L2, time one call" loop, which on Windows
    measures the WDDM submission path waking an idle GPU (~300-600 us) rather
    than the kernel. Here every call in the captured graph touches a *different*
    copy of the cache, so the data is genuinely cold, while CUDA-graph replay
    keeps launch overhead out of the number. That is also what decoding actually
    looks like: 28 layers cycling through caches that evict each other.
    """
    R = len(fns)
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                for f in fns:
                    f()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for f in fns:
                f()
        torch.cuda.synchronize()
        if warm is not None:
            warm()
        for _ in range(2):
            g.replay()
        torch.cuda.synchronize()

        out = []
        mark(span, 0)
        t_start = time.time()
        while True:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            out.append(s.elapsed_time(e) / R)
            elapsed = time.time() - t_start
            if len(out) >= samples and (
                elapsed >= min_seconds or elapsed >= MAX_SAMPLING_SECONDS
            ):
                break
        mark(span, 1)
        del g
        return out
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def stats(xs: list[float]) -> dict:
    """Mean/std plus median and IQR.

    On a thermally-limited laptop GPU the sample distribution is skewed by the
    occasional throttled sample, so the median and IQR are the honest summary
    and the mean is kept only for comparability.
    """
    s = sorted(xs)
    n = len(s)

    def q(f: float) -> float:
        if n == 1:
            return s[0]
        i = f * (n - 1)
        lo = int(i)
        hi = min(lo + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (i - lo)

    med = statistics.median(s)
    p25, p75 = q(0.25), q(0.75)
    return {
        "mean_ms": statistics.fmean(s),
        "std_ms": statistics.pstdev(s) if n > 1 else 0.0,
        "median_ms": med,
        "p25_ms": p25,
        "p75_ms": p75,
        "iqr_ms": p75 - p25,
        "iqr_frac_of_median": (p75 - p25) / med if med else None,
        "min_ms": s[0],
        "max_ms": s[-1],
        "n": n,
    }


# ---------------------------------------------------------------------------
# Benchmark case construction
# ---------------------------------------------------------------------------


@dataclass
class Case:
    method: str
    ctx: int
    nbits: int | None
    group_size: int | None
    fn: object = field(repr=False, default=None)
    fns: list = field(repr=False, default_factory=list)
    n_replicas: int = 1
    footprint_bytes: int = 0
    cache_bytes: int = 0
    effective_bits: float = 16.0
    config: dict = field(default_factory=dict)


def build_cases(
    shape, ctx: int, batch: int, group_size: int, tuned: dict, l2_bytes: int
) -> list[Case]:
    """One Case per method, each carrying N independent copies of the cache.

    Replicas are shared between every method that reads the same cache format,
    so the eager, compiled and fused 4-bit cases all rotate over the *same*
    tensors and are cold in exactly the same way.
    """
    B = batch
    HQ, HKV, D = shape.num_q_heads, shape.num_kv_heads, shape.head_dim
    cases: list[Case] = []

    # --- fp16 reference cache -------------------------------------------
    probe_q, probe_k, probe_v = make_random_kv(B, HQ, HKV, ctx, D, device="cuda", seed=1234)
    fp16_bytes = probe_k.numel() * 2 * 2  # K and V
    r_fp16 = replicas_for(fp16_bytes, l2_bytes)
    fp16_reps = [
        make_random_kv(B, HQ, HKV, ctx, D, device="cuda", seed=1234 + i) for i in range(r_fp16)
    ]
    cases.append(
        Case(
            method="fp16_sdpa",
            ctx=ctx,
            nbits=None,
            group_size=None,
            fn=lambda t=fp16_reps[0]: baseline_fp16_sdpa(*t),
            fns=[(lambda t=t: baseline_fp16_sdpa(*t)) for t in fp16_reps],
            n_replicas=r_fp16,
            footprint_bytes=r_fp16 * fp16_bytes,
            cache_bytes=fp16_bytes,
            effective_bits=16.0,
        )
    )

    # The control: same kernel shape, same split, unquantized fp16 K/V. Without
    # it the fused kernel's speedup silently absorbs the flash-decoding win that
    # has nothing to do with quantization.
    ws16: dict = {}
    o16 = torch.empty((B, HQ, D), device="cuda", dtype=torch.float32)
    cases.append(
        Case(
            method="triton_fp16_control",
            ctx=ctx, nbits=None, group_size=None,
            fn=lambda t=fp16_reps[0], w=ws16, o=o16: fp16_decode_attention(
                t[0], t[1], t[2], out=o, _workspace=w),
            fns=[
                (lambda t=t, w=ws16, o=o16: fp16_decode_attention(
                    t[0], t[1], t[2], out=o, _workspace=w))
                for t in fp16_reps
            ],
            n_replicas=r_fp16, footprint_bytes=r_fp16 * fp16_bytes,
            cache_bytes=fp16_bytes, effective_bits=16.0,
        )
    )

    for nbits in BIT_WIDTHS:
        kq0, vq0 = quantize_kv(probe_k, probe_v, nbits, group_size)
        cbytes = kq0.nbytes() + vq0.nbytes()
        ebits = kq0.effective_bits_per_element()
        r_q = replicas_for(cbytes, l2_bytes)

        # Quantize from the fp16 replicas we already have, extending if needed.
        qreps = []
        for i in range(r_q):
            if i < len(fp16_reps):
                qq, kk, vv = fp16_reps[i]
            else:
                qq, kk, vv = make_random_kv(B, HQ, HKV, ctx, D, device="cuda", seed=9000 + i)
            kqi, vqi = quantize_kv(kk, vv, nbits, group_size)
            qreps.append((qq, kqi, vqi))

        cases.append(
            Case(
                method=f"dequant_sdpa_eager_{nbits}b",
                ctx=ctx, nbits=nbits, group_size=group_size,
                fn=lambda t=qreps[0]: baseline_dequant_sdpa(*t),
                fns=[(lambda t=t: baseline_dequant_sdpa(*t)) for t in qreps],
                n_replicas=r_q, footprint_bytes=r_q * cbytes,
                cache_bytes=cbytes, effective_bits=ebits,
            )
        )
        cases.append(
            Case(
                method=f"dequant_sdpa_compiled_{nbits}b",
                ctx=ctx, nbits=nbits, group_size=group_size,
                fn=lambda t=qreps[0]: baseline_dequant_sdpa_compiled(*t),
                fns=[(lambda t=t: baseline_dequant_sdpa_compiled(*t)) for t in qreps],
                n_replicas=r_q, footprint_bytes=r_q * cbytes,
                cache_bytes=cbytes, effective_bits=ebits,
            )
        )

        cfg = tuned.get((ctx, nbits), {"block_n": 64, "num_warps": 4, "num_stages": 2})
        ws: dict = {}
        out_buf = torch.empty((B, HQ, D), device="cuda", dtype=torch.float32)
        cases.append(
            Case(
                method=f"fused_triton_{nbits}b",
                ctx=ctx, nbits=nbits, group_size=group_size,
                fn=lambda t=qreps[0], cfg=cfg, ws=ws, o=out_buf: fused_decode_attention(
                    t[0], t[1], t[2], out=o, _workspace=ws, **cfg
                ),
                fns=[
                    (lambda t=t, cfg=cfg, ws=ws, o=out_buf: fused_decode_attention(
                        t[0], t[1], t[2], out=o, _workspace=ws, **cfg))
                    for t in qreps
                ],
                n_replicas=r_q, footprint_bytes=r_q * cbytes,
                cache_bytes=cbytes, effective_bits=ebits,
                config=dict(cfg),
            )
        )

        # Control for the metadata-load change: the same kernel with the same
        # tuned config, differing only in how the per-group scale/zero tile is
        # fetched. It is kept as a permanent row rather than a one-off A/B for
        # the same reason the fp16 control is: without it, a later reader
        # cannot tell how much of the fused kernel's speed is the dequant
        # arithmetic and how much was an avoidable gather. The two produce
        # bitwise-identical output (test_correctness.py), so any timing gap
        # between them is pure issue cost.
        ws_g: dict = {}
        out_g = torch.empty((B, HQ, D), device="cuda", dtype=torch.float32)
        cases.append(
            Case(
                method=f"fused_gather_meta_{nbits}b",
                ctx=ctx, nbits=nbits, group_size=group_size,
                fn=lambda t=qreps[0], cfg=cfg, ws=ws_g, o=out_g: fused_decode_attention(
                    t[0], t[1], t[2], out=o, _workspace=ws, meta_bcast=False, **cfg
                ),
                fns=[
                    (lambda t=t, cfg=cfg, ws=ws_g, o=out_g: fused_decode_attention(
                        t[0], t[1], t[2], out=o, _workspace=ws, meta_bcast=False, **cfg))
                    for t in qreps
                ],
                n_replicas=r_q, footprint_bytes=r_q * cbytes,
                cache_bytes=cbytes, effective_bits=ebits,
                config=dict(cfg),
            )
        )

        # The zero-point-folded score path: same tuned config again, but the
        # scores come from a per-group dot against the raw codes instead of a
        # dequantized K tile. Carried as its own row because the effect is
        # regime-dependent -- it is not simply better -- and a single "best of"
        # number would hide exactly the conditional that makes it interesting.
        ws_f: dict = {}
        out_f = torch.empty((B, HQ, D), device="cuda", dtype=torch.float32)
        cases.append(
            Case(
                method=f"fused_fold_zp_{nbits}b",
                ctx=ctx, nbits=nbits, group_size=group_size,
                fn=lambda t=qreps[0], cfg=cfg, ws=ws_f, o=out_f: fused_decode_attention(
                    t[0], t[1], t[2], out=o, _workspace=ws, fold_zp=True, **cfg
                ),
                fns=[
                    (lambda t=t, cfg=cfg, ws=ws_f, o=out_f: fused_decode_attention(
                        t[0], t[1], t[2], out=o, _workspace=ws, fold_zp=True, **cfg))
                    for t in qreps
                ],
                n_replicas=r_q, footprint_bytes=r_q * cbytes,
                cache_bytes=cbytes, effective_bits=ebits,
                config=dict(cfg),
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Autotuning (small, explicit, and recorded)
# ---------------------------------------------------------------------------


def tune(shape, contexts, batch, group_size, verbose=True) -> dict:
    """Pick BLOCK_N / num_warps / num_stages per (ctx, nbits) by measuring.

    Kept deliberately small and explicit rather than using ``triton.autotune``
    so the chosen configuration can be written into the results file and the
    README -- an untuned baseline compared against a tuned kernel would be an
    unfair comparison, and so would a tuned kernel whose config nobody records.
    """
    B = batch
    HQ, HKV, D = shape.num_q_heads, shape.num_kv_heads, shape.head_dim
    flusher = L2Flusher()
    grid = [
        {"block_n": bn, "num_warps": nw, "num_stages": ns}
        for bn in (32, 64, 128)
        for nw in (2, 4, 8)
        for ns in (2, 3)
    ]
    chosen = {}
    for ctx in contexts:
        q, k, v = make_random_kv(B, HQ, HKV, ctx, D, device="cuda", seed=99)
        for nbits in BIT_WIDTHS:
            kq, vq = quantize_kv(k, v, nbits, group_size)
            out_buf = torch.empty((B, HQ, D), device="cuda", dtype=torch.float32)
            best, best_t = None, float("inf")
            for cfg in grid:
                ws: dict = {}
                try:
                    fn = lambda cfg=cfg, ws=ws: fused_decode_attention(
                        q, kq, vq, out=out_buf, _workspace=ws, **cfg
                    )
                    ts = bench_cold(fn, flusher, samples=7, warmup=5)
                except Exception:
                    continue
                t = statistics.median(ts)
                if t < best_t:
                    best, best_t = cfg, t
            if best is None:
                best = {"block_n": 64, "num_warps": 4, "num_stages": 2}
            chosen[(ctx, nbits)] = best
            if verbose:
                print(f"  tuned ctx={ctx:<6} {nbits}-bit -> {best}  ({best_t * 1e3:.1f} us)")
    return chosen


# ---------------------------------------------------------------------------
# Correctness snapshot (so audit_claims doesn't have to re-derive it)
# ---------------------------------------------------------------------------


def correctness_snapshot(shape, contexts, batch, group_size, seeds=(0, 1, 2)) -> list[dict]:
    HQ, HKV, D = shape.num_q_heads, shape.num_kv_heads, shape.head_dim
    rows = []
    for ctx in contexts:
        for nbits in BIT_WIDTHS:
            per_seed = []
            for seed in seeds:
                q, k, v = make_random_kv(batch, HQ, HKV, ctx, D, device="cuda", seed=seed)
                kq, vq = quantize_kv(k, v, nbits, group_size)
                k_deq = dequantize_groupwise(kq, torch.float32)
                v_deq = dequantize_groupwise(vq, torch.float32)

                ref_true = reference_decode_attention(q, k, v)
                ref_deq = reference_decode_attention(q, k_deq, v_deq)
                got = fused_decode_attention(q, kq, vq)
                base = baseline_dequant_sdpa(q, kq, vq)

                def m(a, b):
                    a = a.float().flatten()
                    b = b.float().flatten()
                    d = a - b
                    return {
                        "cosine": torch.nn.functional.cosine_similarity(a[None], b[None]).item(),
                        "max_abs_err": d.abs().max().item(),
                        "rel_l2": (d.norm() / b.norm().clamp_min(1e-12)).item(),
                    }

                per_seed.append(
                    {
                        "seed": seed,
                        "kernel_vs_dequant_ref": m(got, ref_deq),
                        "kernel_vs_fp16_truth": m(got, ref_true),
                        "baseline_vs_fp16_truth": m(base, ref_true),
                        "worst_elem_rel_to_mean_abs": (
                            (got - ref_deq).abs().max() / ref_deq.abs().mean().clamp_min(1e-9)
                        ).item(),
                    }
                )
            rows.append(
                {
                    "ctx": ctx,
                    "nbits": nbits,
                    "group_size": group_size,
                    "seeds": per_seed,
                    "agg": {
                        "min_cosine_vs_dequant_ref": min(
                            s["kernel_vs_dequant_ref"]["cosine"] for s in per_seed
                        ),
                        "max_rel_l2_vs_dequant_ref": max(
                            s["kernel_vs_dequant_ref"]["rel_l2"] for s in per_seed
                        ),
                        "mean_rel_l2_vs_fp16_truth": statistics.fmean(
                            s["kernel_vs_fp16_truth"]["rel_l2"] for s in per_seed
                        ),
                        "mean_baseline_rel_l2_vs_fp16_truth": statistics.fmean(
                            s["baseline_vs_fp16_truth"]["rel_l2"] for s in per_seed
                        ),
                    },
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def env_info() -> dict:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        import triton

        tv = triton.__version__
    except Exception:
        tv = None
    try:
        import importlib.metadata as md

        tw = md.version("triton-windows")
    except Exception:
        tw = None
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_bytes": props.total_memory,
        "l2_cache_bytes": getattr(props, "L2_cache_size", None),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": tv,
        "triton_windows": tw,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "max_sm_clock_mhz": max_sm_clock(),
        "clocks_at_start": smi_once(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--no-clock-monitor", action="store_true",
                    help="skip nvidia-smi clock sampling and the pre-measurement clock ramp")
    ap.add_argument("--out", default=str(RESULTS_DIR / "benchmark.json"))
    args = ap.parse_args()

    ok, why = triton_available()
    if not ok:
        print(f"FATAL: {why}", file=sys.stderr)
        return 1

    contexts = (512, 2048) if args.quick else CONTEXT_LENGTHS
    samples = 8 if args.quick else args.samples

    shape, provenance = load_model_config(args.model)
    print(f"model : {shape.name}  ({provenance})")
    print(f"        layers={shape.num_layers} HQ={shape.num_q_heads} "
          f"HKV={shape.num_kv_heads} D={shape.head_dim} gqa_group={shape.gqa_group}")
    env = env_info()
    print(f"gpu   : {env['gpu']} sm{env['compute_capability']} "
          f"({env['sm_count']} SMs, L2={(env['l2_cache_bytes'] or 0) / 1e6:.0f} MB)")
    print(f"stack : torch {env['torch']} / cuda {env['cuda']} / triton {env['triton']}")
    print()

    t0 = time.time()

    tuned = {}
    if not args.no_tune:
        print("tuning fused kernel:")
        tuned = tune(shape, contexts, args.batch, args.group_size)
        print()

    print("correctness snapshot:")
    correctness = correctness_snapshot(shape, contexts, args.batch, args.group_size)
    for row in correctness:
        a = row["agg"]
        print(
            f"  ctx={row['ctx']:<6} {row['nbits']}-bit  "
            f"cos(kernel,dequant-ref) >= {a['min_cosine_vs_dequant_ref']:.7f}  "
            f"relL2 <= {a['max_rel_l2_vs_dequant_ref']:.2e}  |  "
            f"vs fp16 truth: kernel {a['mean_rel_l2_vs_fp16_truth']:.3e} / "
            f"baseline {a['mean_baseline_rel_l2_vs_fp16_truth']:.3e}"
        )
    print()

    monitor = None
    if not args.no_clock_monitor:
        monitor = ClockMonitor(max_sm=env["max_sm_clock_mhz"])
        if monitor.start():
            c0 = monitor.latest() or {}
            print(f"clocks: monitoring on; idle {c0.get('sm_mhz', 0):.0f} MHz SM / "
                  f"{c0.get('mem_mhz', 0):.0f} MHz mem, max SM "
                  f"{env['max_sm_clock_mhz'] or 0:.0f} MHz, {c0.get('temp_c', 0):.0f} C")
        else:
            print("clocks: nvidia-smi unavailable -- timings are NOT clock-verified")
            monitor = None

    flusher = L2Flusher()
    l2_bytes = flusher.l2_bytes
    rows = []
    unstable = []
    print(f"\ntiming (samples={samples}, L2 flush buffer = {flusher.nbytes / 1e6:.0f} MB):")
    for ctx in contexts:
        cases = build_cases(shape, ctx, args.batch, args.group_size, tuned, l2_bytes)
        print(f"\n  --- context {ctx} ---")
        for c in cases:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            c.fn()
            torch.cuda.synchronize()
            transient = torch.cuda.max_memory_allocated() - before

            def timed(fn):
                """Run one measurement with the clocks ramped first, and record them.

                Each of the four measurements gets its own ramp and its own clock
                window, because they are minutes apart and the GPU drops back to
                idle clocks in between. Bracketing the whole row instead would
                report the ramp itself as instability and hide a genuinely
                throttled measurement inside a wide average.

                The ramp is handed *into* the timing function rather than only
                run here, because everything between this point and the opening
                of the clock window -- warmup, CUDA-graph capture, priming
                replays -- is CPU-bound with the GPU near idle, and long enough
                for the boost to decay. Ramping only out here systematically
                penalised the fastest methods: the slow PyTorch baselines re-boost
                themselves within their own first sample, while a 14 us kernel
                never does, so the fp16 control failed the boost gate for being
                fast rather than for being throttled.
                """
                ramp_holder = {}

                def ramp():
                    if monitor is not None:
                        ramp_holder["inner"] = warm_clocks(
                            monitor, env["max_sm_clock_mhz"]
                        )

                r = warm_clocks(monitor, env["max_sm_clock_mhz"]) if monitor is not None else None
                span = [None, None]
                t_a = time.time()
                val = fn(span, ramp)
                t_b = time.time()
                lo = span[0] if span[0] is not None else t_a
                hi = span[1] if span[1] is not None else t_b
                w = monitor.window(lo, hi) if monitor is not None else None
                return val, {"ramp": r, "ramp_inner": ramp_holder.get("inner"),
                             "clocks": w,
                             "sampling_seconds": hi - lo, "wall_seconds": t_b - t_a}

            # the hot-regime numbers get bootstrapped too, so they need enough
            # samples to bootstrap: half the cold count, not a fifth
            sub = max(10, samples // 2)
            cold, cold_clk = timed(
                lambda sp, w: bench_graph_rotating(c.fns, samples, span=sp, warm=w))
            cold_naive, cold_naive_clk = timed(
                lambda sp, w: bench_cold(c.fn, flusher, sub, span=sp, warm=w))
            pipe, pipe_clk = timed(lambda sp, w: bench_pipelined(c.fn, sub, span=sp, warm=w))
            graph, graph_clk = timed(lambda sp, w: bench_graph(c.fn, sub, span=sp, warm=w))
            clocks = cold_clk["clocks"]

            row = {
                "method": c.method,
                "ctx": ctx,
                "batch": args.batch,
                "nbits": c.nbits,
                "group_size": c.group_size,
                "effective_bits": c.effective_bits,
                "cache_bytes_1layer": c.cache_bytes,
                "cache_bytes_model": int(
                    shape.kv_bytes_per_token(c.effective_bits) * ctx * args.batch
                ),
                "transient_alloc_bytes": int(transient),
                "config": c.config,
                "n_replicas": c.n_replicas,
                "footprint_bytes": c.footprint_bytes,
                "footprint_over_l2": (c.footprint_bytes / l2_bytes) if l2_bytes else None,
                "cold": stats(cold) if isinstance(cold, list) else cold,
                "cold_raw_ms": cold if isinstance(cold, list) else None,
                "cold_naive_single_call": stats(cold_naive) if isinstance(cold_naive, list) else None,
                "pipelined": stats(pipe) if isinstance(pipe, list) else pipe,
                "graph": stats(graph) if isinstance(graph, list) else graph,
                # raw hot-regime samples too: the L2-conditional finding needs a
                # bootstrap CI in *both* regimes, not just the cold one
                "graph_raw_ms": graph if isinstance(graph, list) else None,
                "pipelined_raw_ms": pipe if isinstance(pipe, list) else None,
                "clocks": {
                    "cold": cold_clk,
                    "cold_naive_single_call": cold_naive_clk,
                    "pipelined": pipe_clk,
                    "graph": graph_clk,
                },
                "clock_verified": bool(
                    cold_clk["clocks"] and cold_clk["clocks"]["stable"]
                    and graph_clk["clocks"] and graph_clk["clocks"]["stable"]
                ),
            }
            # quotability needs the summarized timings, so it is settled here
            # rather than inside the literal above
            row["timing_tight"] = bool(
                timing_is_tight(row["cold"]) and timing_is_tight(row["graph"])
            )
            row["quotable"] = bool(row["clock_verified"] and row["timing_tight"])
            if isinstance(pipe, list) and isinstance(graph, list):
                row["launch_overhead_ms"] = row["pipelined"]["mean_ms"] - row["graph"]["mean_ms"]
            else:
                row["launch_overhead_ms"] = None
            rows.append(row)

            lo = row["launch_overhead_ms"]
            if isinstance(cold, list):
                cold_s = (f"{row['cold']['median_ms'] * 1e3:8.1f} us "
                          f"(IQR {row['cold']['iqr_ms'] * 1e3:5.1f})")
            else:
                cold_s = f"  {cold.get('error', 'n/a')[:24]:<24}"
            gclk = graph_clk["clocks"]
            if clocks is None or gclk is None:
                clk = "clk n/a"
            elif row["quotable"]:
                clk = f"clk {clocks['sm_mhz_mean']:4.0f} MHz ok"
            elif not row["clock_verified"]:
                bad = clocks if not clocks["stable"] else gclk
                if not bad.get("enough_samples", True):
                    clk = f"only {bad['n_samples']} clock samples TOO-SHORT"
                    unstable.append(f"{c.method}@{ctx} (too few clock samples)")
                else:
                    clk = f"clk {bad['sm_mhz_min']:4.0f} MHz min NOT-BOOSTED"
                    unstable.append(f"{c.method}@{ctx} (clocks)")
            else:
                clk = f"IQR {row['cold']['iqr_frac_of_median']:.1%} TOO-NOISY"
                unstable.append(f"{c.method}@{ctx} (dispersion)")
            hot = row["graph"]["median_ms"] * 1e3 if isinstance(graph, list) else float("nan")
            print(
                f"  {c.method:<28} cold {cold_s}   "
                f"hot {hot:7.1f} us   "
                f"launch {'n/a' if lo is None else f'{lo * 1e3:6.1f} us'}   "
                f"cache {c.cache_bytes / 1e6:6.2f} MB x{c.n_replicas:<4} "
                f"= {c.footprint_bytes / 1e6:6.0f} MB "
                f"({(c.footprint_bytes / l2_bytes if l2_bytes else 0):4.1f}x L2)   {clk}"
            )

    elapsed = time.time() - t0
    clock_summary = {
        "monitored": monitor is not None,
        "max_sm_clock_mhz": env["max_sm_clock_mhz"],
        "boost_floor_frac": CLOCK_BOOST_FLOOR_FRAC,
        "max_iqr_frac": MAX_IQR_FRAC,
        "temp_limit_c": CLOCK_TEMP_LIMIT_C,
        "ramp_target_frac": CLOCK_TARGET_FRAC,
        "min_clock_samples": MIN_CLOCK_SAMPLES,
        "min_sampling_seconds": MIN_SAMPLING_SECONDS,
        "clocks_at_end": smi_once(),
        "n_rows": len(rows),
        "n_rows_clock_verified": sum(1 for r in rows if r.get("clock_verified")),
        "n_rows_quotable": sum(1 for r in rows if r.get("quotable")),
        "rejected_rows": unstable,
    }
    if monitor is not None:
        monitor.stop()
    payload = {
        "env": env,
        "clock_monitoring": clock_summary,
        "model": shape.as_dict(),
        "model_provenance": provenance,
        "args": vars(args),
        "contexts": list(contexts),
        "bit_widths": list(BIT_WIDTHS),
        "tuned_config": {f"{c}_{b}bit": v for (c, b), v in tuned.items()},
        "correctness": correctness,
        "results": rows,
        "wall_clock_seconds": elapsed,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}  ({elapsed:.1f}s total)")
    if monitor is not None:
        ok_n = clock_summary["n_rows_quotable"]
        print(f"clocks: {ok_n}/{len(rows)} rows quotable "
              f"(all samples >= {CLOCK_BOOST_FLOOR_FRAC:.0%} of max SM clock, "
              f"timing IQR <= {MAX_IQR_FRAC:.0%} of median)")
        if unstable:
            print("  rejected, do not quote: " + ", ".join(unstable))
    else:
        print("clocks: not monitored -- these timings are not quotable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
