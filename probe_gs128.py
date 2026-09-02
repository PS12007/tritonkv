#!/usr/bin/env python
"""Static probe: what does the compiler emit for each (group size, path)?

The timing sweep turned up an anomaly that the load-count story cannot explain.
On the *gather* path the sweep is flat at gs = 16/32/64 (9.4 us) and then jumps
to 26.7 us at gs=128 -- 2.8x slower with *fewer* PTX instructions (1416 vs 1687),
no spills, and an IQR of 1%, so it is reproducible rather than noise. That is the
same cliff the old README sweep recorded as "93 us at gs=128" and left
unexplained.

Fewer instructions and much more time is not a load-count signature. It is what a
layout conversion looks like: a handful of operations that each cost a
shared-memory round trip. So this counts, per cell, the things that would show
that -- shared memory reserved, ld.shared/st.shared/bar.sync in the PTX -- next
to the global loads the load-count story is about.

No timing here, so it needs no clock gate: these are compile-time facts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import torch

import kernels.fused_decode_attn as K
from configs import DEFAULT_MODEL, load_model_config
from kernels.fused_decode_attn import fused_decode_attention
from quantize import quantize_kv
from reference import make_random_kv

OPS = ("ld.global", "st.global", "ld.shared", "st.shared", "bar.sync",
       "cvt.", "shfl.sync", "mma.sync", "ld.param")


def ptx_profile(ptx: str) -> dict:
    c = Counter()
    for line in ptx.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        for op in OPS:
            if s.startswith(op) or re.match(rf"@%p\d+\s+{re.escape(op)}", s):
                c[op] += 1
                break
    return dict(c)


def compile_one(ctx, gs, bcast, cfg, shape, nbits=4):
    HQ, HKV, D = shape.num_q_heads, shape.num_kv_heads, shape.head_dim
    q, k, v = make_random_kv(1, HQ, HKV, ctx, D, device="cuda", seed=7)
    kq, vq = quantize_kv(k, v, nbits, gs)
    seen = set(K._fused_decode_attn_split.device_caches[0][0])
    fused_decode_attention(q, kq, vq, meta_bcast=bcast, **cfg)
    torch.cuda.synchronize()
    cache = K._fused_decode_attn_split.device_caches[0][0]
    new = [cache[kk] for kk in cache if kk not in seen]
    if not new:
        return None
    # The split kernel is the one that reads the cache; the combine kernel is a
    # separate jit function, so everything in `new` here belongs to the split.
    best = max(new, key=lambda x: x.n_regs)
    asm = getattr(best, "asm", None) or {}
    prof = ptx_profile(asm["ptx"]) if "ptx" in asm else {}
    md = getattr(best, "metadata", None)
    return {
        "group_size": gs,
        "path": "broadcast" if bcast else "gather",
        "n_regs": best.n_regs,
        "n_spills": best.n_spills,
        "shared_bytes": getattr(md, "shared", None),
        "num_warps": getattr(md, "num_warps", None),
        **prof,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--block-n", type=int, default=32)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--num-stages", type=int, default=2)
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--nbits", type=int, default=4)
    args = ap.parse_args()

    shape, _ = load_model_config(DEFAULT_MODEL)
    cfg = {"block_n": args.block_n, "num_warps": args.num_warps,
           "num_stages": args.num_stages}
    rows = []
    for gs in args.group_sizes:
        for name, bcast in (("broadcast", True), ("gather", False)):
            r = compile_one(args.ctx, gs, bcast, cfg, shape, args.nbits)
            if r:
                rows.append(r)

    cols = ["group_size", "path", "n_regs", "n_spills", "shared_bytes"] + list(OPS)
    widths = {c: max(len(c), *(len(str(r.get(c, 0))) for r in rows)) for c in cols}
    print(f"ctx={args.ctx} {args.nbits}-bit  {cfg}")
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, 0)).ljust(widths[c]) for c in cols))
    print()
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
