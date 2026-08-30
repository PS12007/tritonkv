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
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from configs import BIT_WIDTHS, CONTEXT_LENGTHS, DEFAULT_GROUP_SIZE, DEFAULT_MODEL, load_model_config
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
# Timing primitives
# ---------------------------------------------------------------------------


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


def bench_cold(fn, flusher: L2Flusher, samples: int, warmup: int = 15) -> list[float]:
    """Per-call latency in ms, with L2 flushed before every sample."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for i in range(samples):
        flusher.flush()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def bench_pipelined(fn, samples: int, iters: int = 50, warmup: int = 15) -> list[float]:
    """Per-call latency in ms for back-to-back calls with a hot cache."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    out = []
    for _ in range(samples):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / iters)
    return out


def bench_graph(fn, samples: int, iters: int = 50, warmup: int = 15):
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

        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()

        out = []
        for _ in range(samples):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            out.append(s.elapsed_time(e) / iters)
        del g
        return out
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


MAX_REPLICAS = 512
MAX_REPLICA_BYTES = 400_000_000


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


def bench_graph_rotating(fns, samples: int, warmup: int = 3):
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
        for _ in range(2):
            g.replay()
        torch.cuda.synchronize()

        out = []
        for _ in range(samples):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            out.append(s.elapsed_time(e) / R)
        del g
        return out
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def stats(xs: list[float]) -> dict:
    return {
        "mean_ms": statistics.fmean(xs),
        "std_ms": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "median_ms": statistics.median(xs),
        "min_ms": min(xs),
        "max_ms": max(xs),
        "n": len(xs),
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
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-tune", action="store_true")
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

    flusher = L2Flusher()
    l2_bytes = flusher.l2_bytes
    rows = []
    print(f"timing (samples={samples}, L2 flush buffer = {flusher.nbytes / 1e6:.0f} MB):")
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

            cold = bench_graph_rotating(c.fns, samples)
            cold_naive = bench_cold(c.fn, flusher, max(5, samples // 5))
            pipe = bench_pipelined(c.fn, max(5, samples // 5))
            graph = bench_graph(c.fn, max(5, samples // 5))

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
            }
            if isinstance(pipe, list) and isinstance(graph, list):
                row["launch_overhead_ms"] = row["pipelined"]["mean_ms"] - row["graph"]["mean_ms"]
            else:
                row["launch_overhead_ms"] = None
            rows.append(row)

            lo = row["launch_overhead_ms"]
            if isinstance(cold, list):
                cold_s = (f"{row['cold']['mean_ms'] * 1e3:8.1f} +- "
                          f"{row['cold']['std_ms'] * 1e3:4.1f} us")
            else:
                cold_s = f"  {cold.get('error', 'n/a')[:24]:<24}"
            hot = row["graph"]["mean_ms"] * 1e3 if isinstance(graph, list) else float("nan")
            print(
                f"  {c.method:<28} cold {cold_s}   "
                f"hot {hot:7.1f} us   "
                f"launch {'n/a' if lo is None else f'{lo * 1e3:6.1f} us'}   "
                f"cache {c.cache_bytes / 1e6:6.2f} MB x{c.n_replicas:<4} "
                f"= {c.footprint_bytes / 1e6:6.0f} MB "
                f"({(c.footprint_bytes / l2_bytes if l2_bytes else 0):4.1f}x L2)"
            )

    elapsed = time.time() - t0
    payload = {
        "env": env,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
