#!/usr/bin/env python
"""Score the cross-GPU predictions in docs/preregistration_l2.md, Part B.

    ./.venv/Scripts/python.exe cross_gpu.py \\
        --card laptop5060=results/benchmark.json --sweep laptop5060=results/l2_sweep.json \\
        --card friend3070=incoming/3070/benchmark.json --sweep friend3070=incoming/3070/l2_sweep.json

Written, and committed, before any volunteer's file existed, for the reason the
pre-registration gives: an analysis built after the data arrives gets to choose
what counts. Every threshold here is imported from ``l2_sweep.py`` or from the
audit, or is written in the pre-registration; none is set in this file.

Per card:

* **C1** -- the DRAM half. ``quant_cold`` (fp16 control / fused 4-bit, rotating
  set) must get the audit verdict TRUE at ctx 8192 and 16384.
* **C2** -- the L2 half. ``quant_hot`` must have CI upper bound < 1 at every
  benchmark context whose fp16 cache is <= ``FIT_FRAC`` of the card's L2.
* **C3** -- the sweep's P1-P5, scored by ``l2_sweep.analyze`` unchanged.

Across cards:

* **C4** -- every crossing ``x*`` in the pre-registered window; and for every
  pair of cards whose L2 differs by >= 3x, the ratio of their crossing contexts
  within a factor of 2 of the ratio of their L2 sizes.
* **M1** -- ``quant_cold@16384`` against compute-to-bandwidth balance
  (SM count x max SM clock / measured copy bandwidth), and against L2, as rank
  correlations with n stated and no p-value.

Exclusions, as pre-registered: rows must be quotable, or tier 2 and admissible
for the effect (``dispersion_tier.usable_for``); a DRAM-resident cell is dropped
if either row's ``footprint_over_l2`` is below 2 (the 512-replica cap).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import dispersion_tier
import l2_sweep
from audit_claims import _verdict, bootstrap_ratio_ci

RESULTS_DIR = Path(__file__).parent / "results"

CONTROL = "triton_fp16_control"
FUSED = "fused_triton_4b"
C1_CONTEXTS = (8192, 16384)
MIN_COLD_FOOTPRINT = 2.0          # x L2; see the 512-replica cap
C4_PAIR_MIN_L2_RATIO = 3.0
C4_CTX_RATIO_TOL = 2.0
M1_CTX = 16384


# ---------------------------------------------------------------------------
# One card
# ---------------------------------------------------------------------------


def _rows(payload: dict) -> dict:
    return {(r["method"], r["ctx"]): r for r in payload["results"]}


def _tiers(payload: dict) -> dict:
    return dispersion_tier.by_row(dispersion_tier.build(payload))


def _usable(tiers: dict, method: str, ctx: int, effect_frac: float) -> bool:
    rec = tiers.get((method, ctx))
    return bool(rec and dispersion_tier.usable_for(rec, effect_frac))


def balance(env: dict, probe: dict | None) -> float | None:
    """SM count x max SM clock (GHz) / measured copy bandwidth (GB/s)."""
    gbps = (probe or {}).get("copy_gbps_median")
    sms, mhz = env.get("sm_count"), env.get("max_sm_clock_mhz")
    if not (gbps and sms and mhz):
        return None
    return sms * (mhz / 1e3) / gbps


def score_card(label: str, bench: dict, sweep: dict | None = None) -> dict:
    env = bench["env"]
    l2 = env.get("l2_cache_bytes") or 0
    rows = _rows(bench)
    tiers = _tiers(bench)
    # Benchmark files from before 2026-09-10 carry no probe; the sweep, run on the
    # same card, does.
    probe = bench.get("bandwidth_probe") or (sweep or {}).get("bandwidth_probe")
    out = {"label": label, "gpu": env.get("gpu"), "l2_bytes": l2,
           "sm_count": env.get("sm_count"), "max_sm_clock_mhz": env.get("max_sm_clock_mhz"),
           "copy_gbps": (probe or {}).get("copy_gbps_median"),
           "balance": balance(env, probe),
           "platform": env.get("platform"), "driver": (env.get("smi_static") or {}).get(
               "driver_version")}

    # C1 -- DRAM half
    c1 = []
    for ctx in C1_CONTEXTS:
        a, b = rows.get((CONTROL, ctx)), rows.get((FUSED, ctx))
        rec = {"ctx": ctx}
        if not a or not b or not a.get("cold_raw_ms") or not b.get("cold_raw_ms"):
            rec.update(status="missing")
            c1.append(rec)
            continue
        r, lo, hi = bootstrap_ratio_ci(a["cold_raw_ms"], b["cold_raw_ms"])
        rec.update(ratio=r, lo=lo, hi=hi, verdict=_verdict(lo, hi))
        foot = min(a.get("footprint_over_l2") or 0, b.get("footprint_over_l2") or 0)
        if foot < MIN_COLD_FOOTPRINT:
            rec.update(status=f"excluded: footprint {foot:.2f}x L2 < {MIN_COLD_FOOTPRINT}")
        elif not (_usable(tiers, CONTROL, ctx, r - 1) and _usable(tiers, FUSED, ctx, r - 1)):
            rec.update(status="excluded: row not usable")
        else:
            rec.update(status="scored")
        c1.append(rec)
    scored = [x for x in c1 if x["status"] == "scored"]
    if any(x["verdict"] != "TRUE" for x in scored):
        c1_v = "FAILS"
    elif len(scored) == len(C1_CONTEXTS):
        c1_v = "HOLDS"
    else:
        c1_v = "untestable"
    out["C1"] = {"verdict": c1_v, "cells": c1}

    # C2 -- L2 half
    c2 = []
    for ctx in bench["contexts"]:
        a, b = rows.get((CONTROL, ctx)), rows.get((FUSED, ctx))
        if not a or not b or not l2:
            continue
        x16 = (a.get("cache_bytes_1layer") or 0) / l2
        if x16 > l2_sweep.FIT_FRAC:
            continue
        if not a.get("graph_raw_ms") or not b.get("graph_raw_ms"):
            c2.append({"ctx": ctx, "x_fp16": x16, "status": "missing"})
            continue
        r, lo, hi = bootstrap_ratio_ci(a["graph_raw_ms"], b["graph_raw_ms"])
        usable = (_usable(tiers, CONTROL, ctx, 1 - r) and _usable(tiers, FUSED, ctx, 1 - r))
        c2.append({"ctx": ctx, "x_fp16": x16, "ratio": r, "lo": lo, "hi": hi,
                   "status": "scored" if usable else "excluded: row not usable"})
    scored2 = [x for x in c2 if x["status"] == "scored"]
    c2_v = ("untestable" if not scored2 else
            "HOLDS" if all(x["hi"] < 1.0 for x in scored2) else "FAILS")
    out["C2"] = {"verdict": c2_v, "cells": c2}

    # C3 -- the sweep
    if sweep is not None:
        rep = l2_sweep.analyze(sweep)
        out["C3"] = {p["id"]: p["verdict"] for p in rep["predictions"]}
        p2 = next(p for p in rep["predictions"] if p["id"] == "P2")
        out["x_star"] = p2.get("x_star")
        out["ctx_star"] = p2.get("ctx_star")
    else:
        out["C3"] = None
        out["x_star"] = out["ctx_star"] = None

    # M1 input
    m1 = next((x for x in c1 if x["ctx"] == M1_CTX and x["status"] == "scored"), None)
    out["quant_cold_m1"] = m1["ratio"] if m1 else None
    return out


# ---------------------------------------------------------------------------
# Across cards
# ---------------------------------------------------------------------------


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else None


def score_across(cards: list[dict]) -> dict:
    # C4
    with_x = [c for c in cards if c.get("x_star") not in (None, float("-inf"))]
    window = l2_sweep.CROSSING_WINDOW
    in_window = [window[0] <= c["x_star"] <= window[1] for c in with_x]
    pairs = []
    for i, a in enumerate(with_x):
        for b in with_x[i + 1:]:
            big, small = (a, b) if a["l2_bytes"] >= b["l2_bytes"] else (b, a)
            l2r = big["l2_bytes"] / small["l2_bytes"]
            if l2r < C4_PAIR_MIN_L2_RATIO:
                continue
            ctxr = big["ctx_star"] / small["ctx_star"]
            ok = (l2r / C4_CTX_RATIO_TOL) <= ctxr <= (l2r * C4_CTX_RATIO_TOL)
            pairs.append({"big": big["label"], "small": small["label"], "l2_ratio": l2r,
                          "ctx_ratio": ctxr, "ok": ok})
    if not with_x:
        c4 = "untestable"
    elif not all(in_window) or any(not p["ok"] for p in pairs):
        c4 = "FAILS"
    elif not pairs:
        c4 = "partial: every crossing in window, no pair of cards >= 3x apart in L2"
    else:
        c4 = "HOLDS"

    # M1
    m = [c for c in cards if c.get("quant_cold_m1") is not None and c.get("balance")]
    rho_bal = spearman([c["balance"] for c in m], [c["quant_cold_m1"] for c in m])
    rho_l2 = spearman([c["l2_bytes"] for c in m], [c["quant_cold_m1"] for c in m])
    return {
        "C4": {"verdict": c4, "pairs": pairs,
               "crossings": [(c["label"], c["x_star"], c["ctx_star"]) for c in with_x]},
        "M1": {"n": len(m), "rho_balance": rho_bal, "rho_l2": rho_l2,
               "descriptive_only": len(m) < 5},
    }


def render(cards: list[dict], across: dict) -> str:
    L = ["# Cross-GPU scorecard (docs/preregistration_l2.md, Part B)", ""]
    L.append("| card | GPU | L2 MB | SMs | copy GB/s | balance | C1 | C2 | C3 (P1-P5) | x* | ctx* |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cards:
        c3 = " ".join(f"{k}:{v[0]}" for k, v in (c["C3"] or {}).items()) or "no sweep"
        L.append(" | ".join([
            f"| {c['label']}", str(c["gpu"]), f"{c['l2_bytes'] / 1e6:.1f}", str(c["sm_count"]),
            _fmt(c.get("copy_gbps"), ".0f"), _fmt(c.get("balance"), ".3f"),
            c["C1"]["verdict"], c["C2"]["verdict"], c3,
            _fmt(c.get("x_star"), ".2f"), _fmt(c.get("ctx_star"), ".0f") + " |"]))
    L.append("")
    for c in cards:
        L.append(f"**{c['label']}** C1 cells: " + "; ".join(
            f"ctx {x['ctx']}: " + (f"{x['ratio']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] "
                                  f"{x.get('verdict', '')} ({x['status']})"
                                  if "ratio" in x else x["status"])
            for x in c["C1"]["cells"]))
        L.append(f"**{c['label']}** C2 cells: " + ("; ".join(
            f"ctx {x['ctx']} ({x['x_fp16']:.2f}x L2): " +
            (f"{x['ratio']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] ({x['status']})"
             if "ratio" in x else x["status"])
            for x in c["C2"]["cells"]) or "no context has fp16 <= 0.75 L2"))
        L.append("")
    c4, m1 = across["C4"], across["M1"]
    L.append(f"**C4:** {c4['verdict']}")
    for p in c4["pairs"]:
        L.append(f"- {p['big']} / {p['small']}: L2 ratio {p['l2_ratio']:.2f}, crossing-ctx "
                 f"ratio {p['ctx_ratio']:.2f} -> {'ok' if p['ok'] else 'OUTSIDE factor 2'}")
    L.append("")
    L.append(f"**M1** (n = {m1['n']}{', descriptive only' if m1['descriptive_only'] else ''}): "
             f"rank corr. of quant_cold@{M1_CTX} with balance = {_fmt(m1['rho_balance'], '+.2f')}, "
             f"with L2 = {_fmt(m1['rho_l2'], '+.2f')}. No p-value, by pre-registration.")
    L.append("")
    return "\n".join(L)


def _fmt(v, spec: str) -> str:
    if v is None or (isinstance(v, float) and math.isinf(v)):
        return "n/a"
    return format(v, spec)


def _pairs(specs: list[str]) -> dict:
    out = {}
    for s in specs or []:
        label, _, path = s.partition("=")
        if not path:
            raise SystemExit(f"expected label=path, got {s!r}")
        out[label] = json.loads(Path(path).read_text(encoding="utf-8"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--card", action="append", required=True, metavar="LABEL=benchmark.json")
    ap.add_argument("--sweep", action="append", default=[], metavar="LABEL=l2_sweep.json")
    ap.add_argument("--out-md", default=str(RESULTS_DIR / "cross_gpu.md"))
    ap.add_argument("--out-json", default=str(RESULTS_DIR / "cross_gpu.json"))
    args = ap.parse_args()
    benches, sweeps = _pairs(args.card), _pairs(args.sweep)
    unknown = set(sweeps) - set(benches)
    if unknown:
        raise SystemExit(f"--sweep labels with no --card: {sorted(unknown)}")
    cards = [score_card(k, v, sweeps.get(k)) for k, v in benches.items()]
    across = score_across(cards)
    md = render(cards, across)
    Path(args.out_md).write_text(md, encoding="utf-8")
    Path(args.out_json).write_text(json.dumps({"cards": cards, "across": across}, indent=1,
                                              default=str), encoding="utf-8")
    print(md)
    print(f"wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
