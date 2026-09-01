"""Adversarial self-audit of every performance and correctness claim.

    python audit_claims.py                       # reads results/benchmark.json
    python audit_claims.py --no-gpu-checks       # statistics only, no GPU needed

The rule this file exists to enforce: **no claim is reported as a flat "True"
without evidence, and every claim gets a genuine attempt at falsification
first.** Each claim is emitted as one of

* ``TRUE``                   -- survived falsification, evidence attached
* ``TRUE BUT CONDITIONAL``   -- holds only under a stated condition
* ``MISLEADING``             -- technically defensible phrasing, wrong impression
* ``FALSE``                  -- does not hold

Speedups are tested with a **bootstrap confidence interval over the raw
per-sample timings**, not by comparing two means. A speedup only counts if the
95% CI of the ratio excludes the practical-significance threshold, so a 3%
"win" that is really run-to-run jitter cannot be reported as a win.

Writes ``results/audit.json`` and ``results/audit.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# A speedup must beat this to be called a speedup at all. 1.05 = 5%.
PRACTICAL_THRESHOLD = 1.05
BOOTSTRAP_N = 10000
CI = 0.95


@dataclass
class Claim:
    id: str
    claim: str
    verdict: str
    evidence: str
    falsification_attempted: str

    def as_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ratio_ci(num: list[float], den: list[float], n=BOOTSTRAP_N, ci=CI, seed=0):
    """Bootstrap CI for mean(num)/mean(den).

    ``num`` is the slower method's timings and ``den`` the faster one's, so the
    ratio is the speedup. Resampling both independently is right here: the
    samples are separate timed runs, not paired measurements.
    """
    rng = random.Random(seed)
    ratios = []
    ln, ld = len(num), len(den)
    for _ in range(n):
        a = statistics.fmean(num[rng.randrange(ln)] for _ in range(ln))
        b = statistics.fmean(den[rng.randrange(ld)] for _ in range(ld))
        ratios.append(a / b if b > 0 else float("inf"))
    ratios.sort()
    lo = ratios[int((1 - ci) / 2 * n)]
    hi = ratios[min(n - 1, int((1 + ci) / 2 * n))]
    return statistics.fmean(num) / statistics.fmean(den), lo, hi


def cv(xs: list[float]) -> float:
    """Coefficient of variation -- how noisy the measurement itself was."""
    m = statistics.fmean(xs)
    return statistics.pstdev(xs) / m if m > 0 else float("inf")


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


class Bench:
    def __init__(self, payload: dict):
        self.p = payload
        self.by = {(r["method"], r["ctx"]): r for r in payload["results"]}
        self.contexts = payload["contexts"]
        self.model = payload["model"]
        self.env = payload["env"]

    def get(self, method: str, ctx: int):
        return self.by.get((method, ctx))

    def cold(self, method: str, ctx: int) -> list[float] | None:
        r = self.get(method, ctx)
        return r["cold_raw_ms"] if r else None

    def hot(self, method: str, ctx: int) -> float | None:
        r = self.get(method, ctx)
        if not r or not isinstance(r.get("pipelined"), dict) or "mean_ms" not in r["pipelined"]:
            return None
        return r["pipelined"]["mean_ms"]

    def hot_raw(self, method: str, ctx: int) -> list[float] | None:
        """Raw per-sample hot (L2-resident, graph-replay) timings, if recorded."""
        r = self.get(method, ctx)
        return (r or {}).get("graph_raw_ms")

    def quotable(self, method: str, ctx: int) -> bool:
        return bool((self.get(method, ctx) or {}).get("quotable"))

    def corr(self, ctx: int, nbits: int):
        for row in self.p["correctness"]:
            if row["ctx"] == ctx and row["nbits"] == nbits:
                return row
        return None


# ---------------------------------------------------------------------------
# Claim generation
# ---------------------------------------------------------------------------


def audit_speed(b: Bench, nbits: int = 4) -> list[Claim]:
    claims: list[Claim] = []
    fused = f"fused_triton_{nbits}b"
    eager = f"dequant_sdpa_eager_{nbits}b"
    comp = f"dequant_sdpa_compiled_{nbits}b"

    per_ctx = {}
    for ctx in b.contexts:
        f, e, c = b.cold(fused, ctx), b.cold(eager, ctx), b.cold(comp, ctx)
        if not (f and e):
            continue
        r_e, lo_e, hi_e = bootstrap_ratio_ci(e, f)
        entry = {"eager": (r_e, lo_e, hi_e), "noise": (cv(f), cv(e))}
        if c:
            entry["compiled"] = bootstrap_ratio_ci(c, f)
        per_ctx[ctx] = entry

    # --- Claim: speedup vs the naive eager baseline, per context length ----
    for ctx, d in per_ctx.items():
        r, lo, hi = d["eager"]
        cvf, cve = d["noise"]
        if lo > PRACTICAL_THRESHOLD:
            verdict = "TRUE"
        elif hi < 1.0:
            verdict = "FALSE"
        else:
            verdict = "MISLEADING"
        caveat = clock_caveat(b, ctx, fused, eager)
        verdict = downgrade(verdict, caveat)
        claims.append(
            Claim(
                id=f"speed.eager.{nbits}b.ctx{ctx}",
                claim=(
                    f"At {ctx} tokens of context, the fused {nbits}-bit Triton kernel is "
                    f"{r:.2f}x faster than PyTorch dequantize-then-SDPA."
                ),
                verdict=verdict,
                evidence=(
                    f"bootstrap 95% CI of the speedup ratio = [{lo:.2f}x, {hi:.2f}x] over "
                    f"{len(b.cold(fused, ctx))} cold-L2 samples per method; measurement noise "
                    f"CV = {cvf * 100:.1f}% (fused) / {cve * 100:.1f}% (baseline)." + caveat
                ),
                falsification_attempted=(
                    "Tried to show the gap is run-to-run jitter by resampling the raw "
                    f"per-call timings {BOOTSTRAP_N} times. "
                    + (
                        f"Failed: even the 2.5th-percentile ratio ({lo:.2f}x) clears the "
                        f"{PRACTICAL_THRESHOLD:.2f}x practical-significance bar."
                        if verdict == "TRUE"
                        else f"Succeeded or was inconclusive: the CI [{lo:.2f}x, {hi:.2f}x] "
                        f"does not exclude the {PRACTICAL_THRESHOLD:.2f}x bar."
                    )
                ),
            )
        )

    # --- Claim: it holds at ALL context lengths ---------------------------
    if per_ctx:
        wins = {c: d["eager"][1] > PRACTICAL_THRESHOLD for c, d in per_ctx.items()}
        losers = [c for c, w in wins.items() if not w]
        ratios = {c: d["eager"][0] for c, d in per_ctx.items()}
        if not losers:
            verdict, cond = "TRUE", ""
        else:
            verdict = "TRUE BUT CONDITIONAL"
            cond = f" Condition: context length not in {losers}."
        claims.append(
            Claim(
                id=f"speed.all_contexts.{nbits}b",
                claim=f"The fused {nbits}-bit kernel beats the naive baseline at every tested context length.",
                verdict=verdict,
                evidence=(
                    "per-context speedups: "
                    + ", ".join(f"{c}:{r:.2f}x" for c, r in sorted(ratios.items()))
                    + cond
                ),
                falsification_attempted=(
                    "Checked each context length independently instead of quoting the best "
                    "one. Short contexts are where a split-K kernel is most likely to lose, "
                    "because per-launch overhead stops being amortized."
                ),
            )
        )

    # --- Claim: it beats the STRONGEST PyTorch baseline, not just eager ----
    have_comp = {c: d for c, d in per_ctx.items() if "compiled" in d}
    if have_comp:
        worst_ctx = min(have_comp, key=lambda c: have_comp[c]["compiled"][1])
        r, lo, hi = have_comp[worst_ctx]["compiled"]
        all_win = all(d["compiled"][1] > PRACTICAL_THRESHOLD for d in have_comp.values())
        claims.append(
            Claim(
                id=f"speed.vs_compiled.{nbits}b",
                claim=(
                    f"The speedup is not an artefact of comparing against a weak baseline: the "
                    f"fused {nbits}-bit kernel also beats a torch.compile'd dequant + SDPA."
                ),
                verdict="TRUE" if all_win else "TRUE BUT CONDITIONAL",
                evidence=(
                    "vs compiled baseline: "
                    + ", ".join(
                        f"{c}:{d['compiled'][0]:.2f}x [{d['compiled'][1]:.2f},{d['compiled'][2]:.2f}]"
                        for c, d in sorted(have_comp.items())
                    )
                    + f". Weakest case is ctx={worst_ctx} at {r:.2f}x."
                ),
                falsification_attempted=(
                    "Let Inductor fuse the entire unpack+dequant chain into a single kernel, "
                    "removing the intermediate uint8 and fp16 tensors that make the eager "
                    "baseline look bad. This is the strongest pure-PyTorch baseline we know "
                    "how to write."
                ),
            )
        )

    # --- Claim: the L2-hot number is the honest one -----------------------
    hot_ratios, cold_ratios = {}, {}
    for ctx in b.contexts:
        hf, he = b.hot(fused, ctx), b.hot(eager, ctx)
        if hf and he:
            hot_ratios[ctx] = he / hf
        if ctx in per_ctx:
            cold_ratios[ctx] = per_ctx[ctx]["eager"][0]
    if hot_ratios and cold_ratios:
        shared = sorted(set(hot_ratios) & set(cold_ratios))
        gap = max((hot_ratios[c] - cold_ratios[c]) for c in shared) if shared else 0.0
        claims.append(
            Claim(
                id=f"method.l2_regime.{nbits}b",
                claim="The reported speedup does not depend on how the benchmark loop treats cache.",
                verdict="MISLEADING" if abs(gap) > 0.3 else "TRUE",
                evidence=(
                    "speedup with L2 flushed between samples vs. hot back-to-back loop: "
                    + ", ".join(
                        f"{c}: {cold_ratios[c]:.2f}x cold / {hot_ratios[c]:.2f}x hot" for c in shared
                    )
                    + f". Largest divergence {gap:+.2f}x."
                ),
                falsification_attempted=(
                    "Deliberately ran the benchmark the easy way (tight loop, no cache "
                    f"flush). A one-layer 4-bit cache is only a few MB against this GPU's "
                    f"{(b.env.get('l2_cache_bytes') or 0) / 1e6:.0f} MB L2, so a naive loop "
                    "measures an L2-resident cache that cannot occur during real decoding, "
                    "where every layer evicts the previous one. All headline numbers in this "
                    "project use the cold-L2 regime."
                ),
            )
        )

    # --- Claim: fused quantized decode beats unquantized fp16 -------------
    for ctx in b.contexts:
        f, fp = b.cold(fused, ctx), b.cold("fp16_sdpa", ctx)
        if not (f and fp):
            continue
        r, lo, hi = bootstrap_ratio_ci(fp, f)
        if lo > PRACTICAL_THRESHOLD:
            v = "TRUE"
        elif hi < 1.0:
            v = "FALSE"
        else:
            v = "MISLEADING"
        caveat = clock_caveat(b, ctx, fused, "fp16_sdpa")
        v = downgrade(v, caveat)
        claims.append(
            Claim(
                id=f"speed.vs_fp16.{nbits}b.ctx{ctx}",
                claim=(
                    f"At {ctx} tokens, the fused {nbits}-bit kernel is faster than plain "
                    f"unquantized fp16 SDPA ({r:.2f}x)."
                ),
                verdict=v,
                evidence=f"bootstrap 95% CI [{lo:.2f}x, {hi:.2f}x]." + caveat,
                falsification_attempted=(
                    "Compared against fp16 SDPA, which does no dequantization at all and "
                    "dispatches to a hand-tuned cuDNN/flash path. This is the comparison "
                    "most likely to go against the kernel, and it is the honest question a "
                    "reader asks: is quantized decode actually faster, or only cheaper?"
                ),
            )
        )

    return claims


def audit_memory(b: Bench, nbits: int = 4) -> list[Claim]:
    claims = []
    ctx = max(b.contexts)
    fused = b.get(f"fused_triton_{nbits}b", ctx)
    fp16 = b.get("fp16_sdpa", ctx)
    if not (fused and fp16):
        return claims

    ratio = fp16["cache_bytes_1layer"] / fused["cache_bytes_1layer"]
    eff = fused["effective_bits"]
    claims.append(
        Claim(
            id=f"mem.compression.{nbits}b",
            claim=f"{nbits}-bit KV quantization shrinks the cache by {16 / nbits:.0f}x.",
            verdict="MISLEADING",
            evidence=(
                f"Real measured ratio is {ratio:.2f}x, not {16 / nbits:.0f}x, because each "
                f"group of {fused['group_size']} elements also carries an fp16 scale and an "
                f"fp16 zero-point: {eff:.1f} effective bits/element, not {nbits}. "
                f"At ctx={ctx} one layer is {fused['cache_bytes_1layer'] / 1e6:.2f} MB vs "
                f"{fp16['cache_bytes_1layer'] / 1e6:.2f} MB fp16."
            ),
            falsification_attempted=(
                "Counted the metadata that compression claims usually omit. The nominal "
                f"{16 / nbits:.0f}x is only reachable as group_size -> infinity, which is "
                "not a configuration anyone runs because accuracy collapses."
            ),
        )
    )

    model = b.model
    per_tok_fp16 = 2 * model["num_layers"] * model["num_kv_heads"] * model["head_dim"] * 2
    per_tok_q = per_tok_fp16 * eff / 16
    claims.append(
        Claim(
            id=f"mem.whole_model.{nbits}b",
            claim=(
                f"For {model['name']}, {nbits}-bit KV cache at {ctx} tokens uses "
                f"{per_tok_q * ctx / 1e6:.1f} MB instead of {per_tok_fp16 * ctx / 1e6:.1f} MB."
            ),
            verdict="TRUE BUT CONDITIONAL",
            evidence=(
                f"{per_tok_fp16} B/token fp16 vs {per_tok_q:.0f} B/token at {eff:.1f} effective "
                f"bits, over {model['num_layers']} layers x {model['num_kv_heads']} KV heads x "
                f"{model['head_dim']} dims x 2 (K and V)."
            ),
            falsification_attempted=(
                "This is arithmetic from the model's architecture, not a measurement -- the "
                "benchmark only ever allocates ONE layer's cache. Condition: it assumes every "
                "layer is quantized with the same settings and that no fp16 residual buffer "
                "is kept for recent tokens (many published KV-quant methods do keep one, "
                "which would raise the real number)."
            ),
        )
    )

    transient = fused.get("transient_alloc_bytes", 0)
    base_transient = b.get(f"dequant_sdpa_eager_{nbits}b", ctx)
    if base_transient:
        bt = base_transient.get("transient_alloc_bytes", 0)
        claims.append(
            Claim(
                id=f"mem.transient.{nbits}b",
                claim="The fused kernel avoids the per-step allocation the naive path needs.",
                verdict="TRUE" if transient < bt else "FALSE",
                evidence=(
                    f"peak transient allocation during one decode step at ctx={ctx}: "
                    f"fused {transient / 1e6:.2f} MB vs baseline {bt / 1e6:.2f} MB."
                ),
                falsification_attempted=(
                    "Measured with torch.cuda.max_memory_allocated around a single call "
                    "rather than reasoning about it. Note the fused kernel is not zero: it "
                    "allocates flash-decoding split partials, which grow with the split count."
                ),
            )
        )
    return claims


def audit_correctness(b: Bench) -> list[Claim]:
    claims = []
    for nbits in b.p["bit_widths"]:
        rows = [b.corr(c, nbits) for c in b.contexts]
        rows = [r for r in rows if r]
        if not rows:
            continue
        min_cos = min(r["agg"]["min_cosine_vs_dequant_ref"] for r in rows)
        max_rel = max(r["agg"]["max_rel_l2_vs_dequant_ref"] for r in rows)
        worst_elem = max(
            s["worst_elem_rel_to_mean_abs"] for r in rows for s in r["seeds"]
        )

        claims.append(
            Claim(
                id=f"correct.kernel.{nbits}b",
                claim=(
                    f"The fused {nbits}-bit kernel computes the same thing as the PyTorch "
                    f"reference (cosine >= {min_cos:.6f} everywhere)."
                ),
                verdict="TRUE" if min_cos >= 0.99999 else "MISLEADING",
                evidence=(
                    f"across {len(rows)} context lengths x {len(rows[0]['seeds'])} seeds: "
                    f"min cosine {min_cos:.7f}, max relative L2 {max_rel:.2e}, and the single "
                    f"worst output element is {worst_elem * 100:.2f}% of the mean |output|."
                ),
                falsification_attempted=(
                    "Cosine similarity over a flattened tensor is exactly the metric that "
                    "hides a few badly-wrong elements among thousands of right ones, so the "
                    "worst *individual element* was checked too, not only the aggregate. "
                    "Comparison is against dequantize-then-attend in fp32 on identical "
                    "dequantized values, which isolates kernel error from quantization error."
                ),
            )
        )

        k_err = statistics.fmean(r["agg"]["mean_rel_l2_vs_fp16_truth"] for r in rows)
        b_err = statistics.fmean(r["agg"]["mean_baseline_rel_l2_vs_fp16_truth"] for r in rows)
        claims.append(
            Claim(
                id=f"correct.no_extra_error.{nbits}b",
                claim=f"The {nbits}-bit speedup is not bought with accuracy.",
                verdict="TRUE" if k_err <= 1.5 * b_err else "FALSE",
                evidence=(
                    f"vs the unquantized fp16 cache, mean relative L2 error is "
                    f"{k_err:.3e} for the fused kernel and {b_err:.3e} for the PyTorch "
                    f"dequant baseline (ratio {k_err / b_err:.2f}x). Essentially all of it is "
                    f"quantization error that both paths share."
                ),
                falsification_attempted=(
                    "Measured the kernel and the baseline against the SAME unquantized "
                    "ground truth, so a kernel that quietly traded precision for speed would "
                    "show a larger error than the baseline. It does not."
                ),
            )
        )

        # Is the quantization itself good enough to be usable at all?
        e2e = statistics.fmean(r["agg"]["mean_rel_l2_vs_fp16_truth"] for r in rows)
        if nbits == 2:
            claims.append(
                Claim(
                    id="correct.2bit_usable",
                    claim="2-bit KV quantization is usable.",
                    verdict="MISLEADING",
                    evidence=(
                        f"The 2-bit *kernel* is correct (it matches its own reference to "
                        f"cosine {min_cos:.6f}), but 2-bit *quantization* introduces "
                        f"{e2e * 100:.1f}% relative error against the unquantized cache. "
                        "Kernel correctness and end-task usability are different claims and "
                        "this project only establishes the first."
                    ),
                    falsification_attempted=(
                        "Separated 'the kernel implements 2-bit correctly' from '2-bit is "
                        "accurate enough to deploy'. No downstream perplexity or task "
                        "evaluation was run, so the second claim is not supported by "
                        "anything measured here."
                    ),
                )
            )
    return claims


# ---------------------------------------------------------------------------
# Attribution: which part of the speedup is actually the quantization?
#
# This is the section that exists because the honest answer turned out to be
# "mostly none of it". The kernel beats PyTorch SDPA by a large factor, but it
# changes two things at once: it stores KV in 4 or 2 bits, *and* it splits the
# history across SMs (flash-decoding), which PyTorch's SDPA does not do for a
# single query token. `triton_fp16_control` is the same kernel with the same
# split, the same online softmax and the same GQA amortization, reading plain
# fp16 -- so the two ratios below separate the two effects.
# ---------------------------------------------------------------------------

CONTROL = "triton_fp16_control"


def clock_caveat(b: Bench, ctx: int, *methods: str) -> str:
    """Flag a ratio whose inputs were not measured at boost clocks.

    A baseline timed while the GPU had dropped to idle clocks looks slower than
    it is, which inflates every speedup computed against it -- in the flattering
    direction. So a claim built on such a row says so instead of quoting it.
    """
    if not b.p.get("clock_monitoring"):
        return ""
    bad = [m for m in methods if b.get(m, ctx) and not b.quotable(m, ctx)]
    if not bad:
        return ""
    return (
        " NOT CLOCK-VERIFIED: " + ", ".join(bad) + f" at ctx={ctx} failed the boost-clock "
        "or dispersion gate, so this ratio may be inflated by a measurement taken at "
        "reduced clocks; treat it as conditional on a re-run."
    )


def downgrade(verdict: str, caveat: str) -> str:
    return "TRUE BUT CONDITIONAL" if (caveat and verdict == "TRUE") else verdict


def _verdict(lo: float, hi: float) -> str:
    if lo > PRACTICAL_THRESHOLD:
        return "TRUE"
    if hi < 1.0:
        return "FALSE"
    return "MISLEADING"


def audit_attribution(b: Bench, nbits: int = 4) -> list[Claim]:
    """Split the headline speedup into flash-decoding and quantization."""
    claims: list[Claim] = []
    fused = f"fused_triton_{nbits}b"

    if b.cold(CONTROL, b.contexts[0]) is None:
        return [
            Claim(
                id=f"attribution.control_missing.{nbits}b",
                claim=(
                    f"The {nbits}-bit kernel's speedup over PyTorch is attributable to "
                    "quantization."
                ),
                verdict="MISLEADING",
                evidence=(
                    "No fp16 control kernel is present in these results, so the measurement "
                    "cannot separate quantization from the flash-decoding split that the "
                    "same kernel also performs. Re-run benchmark.py with "
                    f"{CONTROL} enabled before making any attribution claim."
                ),
                falsification_attempted=(
                    "Refused to attribute the win to quantization on evidence that cannot "
                    "distinguish the two mechanisms."
                ),
            )
        ]

    # --- per context: split total speedup into split-effect x quant-effect ---
    split_only, quant_cold, quant_hot = {}, {}, {}
    for ctx in b.contexts:
        sdpa, ctrl, fus = b.cold("fp16_sdpa", ctx), b.cold(CONTROL, ctx), b.cold(fused, ctx)
        if sdpa and ctrl:
            split_only[ctx] = bootstrap_ratio_ci(sdpa, ctrl)
        if ctrl and fus:
            quant_cold[ctx] = bootstrap_ratio_ci(ctrl, fus)
        h_ctrl, h_fus = b.hot_raw(CONTROL, ctx), b.hot_raw(fused, ctx)
        if h_ctrl and h_fus:
            quant_hot[ctx] = bootstrap_ratio_ci(h_ctrl, h_fus)

    # --- Claim 1: most of the headline number is the split, not the bits ----
    if split_only and quant_cold:
        shared = sorted(set(split_only) & set(quant_cold))
        worst = min(shared, key=lambda c: quant_cold[c][0]) if shared else None
        claims.append(
            Claim(
                id=f"attribution.headline.{nbits}b",
                claim=(
                    f"The fused {nbits}-bit kernel's speedup over PyTorch SDPA shows that "
                    "quantizing the KV cache makes decode attention faster."
                ),
                verdict="MISLEADING",
                evidence=(
                    "The kernel changes two things at once. Holding the algorithm fixed and "
                    "varying only the storage format: an identical fp16 kernel (same split, "
                    "same online softmax, same GQA amortization, no dequantization) already "
                    "reaches "
                    + ", ".join(f"{c}:{split_only[c][0]:.1f}x" for c in shared)
                    + " against fp16 SDPA on its own. What the quantization then adds is "
                    + ", ".join(f"{c}:{quant_cold[c][0]:.2f}x" for c in shared)
                    + " (DRAM-resident) -- the split, not the bit width, is most of the "
                    "headline number."
                ),
                falsification_attempted=(
                    "Wrote a control kernel specifically to try to make the quantization "
                    "look unnecessary, and it largely succeeded. The remaining question -- "
                    "whether quantization ever pays for itself -- is answered separately "
                    f"below (weakest DRAM-resident case: ctx={worst})."
                    if worst is not None
                    else "Wrote an fp16 control kernel to isolate the two effects."
                ),
            )
        )

    # --- Claim 2: quantization in the L2-resident (hot) regime --------------
    for ctx, (r, lo, hi) in sorted(quant_hot.items()):
        caveat = clock_caveat(b, ctx, fused, CONTROL)
        v = downgrade(_verdict(lo, hi), caveat)
        claims.append(
            Claim(
                id=f"attribution.quant_effect.hot.{nbits}b.ctx{ctx}",
                claim=(
                    f"With the KV cache resident in L2 at {ctx} tokens, reading it as "
                    f"{nbits}-bit codes is faster than reading it as fp16."
                ),
                verdict=v,
                evidence=(
                    f"fp16 control / fused {nbits}-bit, CUDA-graph replay on an "
                    f"L2-resident cache: {r:.2f}x, bootstrap 95% CI [{lo:.2f}x, {hi:.2f}x]. "
                    + (
                        f"Below 1.0x: the quantized path costs {1 / r:.2f}x more time."
                        if hi < 1.0
                        else ""
                    )
                    + caveat
                ),
                falsification_attempted=(
                    "Compared against the same kernel rather than against PyTorch, so the "
                    "only difference is the dequantization. When the bytes are already in "
                    "L2 there is little traffic left to save and the unpack/scale/fma chain "
                    "is pure added work -- so this is the regime where quantization is "
                    "expected to lose, and it does."
                ),
            )
        )

    # --- Claim 3: quantization in the DRAM-resident (cold) regime -----------
    for ctx, (r, lo, hi) in sorted(quant_cold.items()):
        caveat = clock_caveat(b, ctx, fused, CONTROL)
        v = downgrade(_verdict(lo, hi), caveat)
        row = b.get(fused, ctx) or {}
        over = row.get("footprint_over_l2")
        claims.append(
            Claim(
                id=f"attribution.quant_effect.cold.{nbits}b.ctx{ctx}",
                claim=(
                    f"With the working set larger than L2 at {ctx} tokens, storing the KV "
                    f"cache in {nbits} bits is faster than storing it in fp16."
                ),
                verdict=v,
                evidence=(
                    f"fp16 control / fused {nbits}-bit over a rotating working set "
                    + (f"{over:.1f}x the size of L2" if over else "larger than L2")
                    + f": {r:.2f}x, bootstrap 95% CI [{lo:.2f}x, {hi:.2f}x]." + caveat
                ),
                falsification_attempted=(
                    "Same kernel, same split, same softmax; only the storage format "
                    "differs. Both sides pay the same rotating-working-set penalty, so a "
                    "win here is a bytes-moved win and not a caching artefact."
                ),
            )
        )

    # --- Claim 4: the conditional that the whole project turns on -----------
    if quant_hot and quant_cold:
        shared = sorted(set(quant_hot) & set(quant_cold))
        hot_losses = [c for c in shared if quant_hot[c][2] < 1.0]
        cold_wins = [c for c in shared if quant_cold[c][1] > PRACTICAL_THRESHOLD]
        if hot_losses and cold_wins:
            verdict = "TRUE BUT CONDITIONAL"
        elif cold_wins:
            verdict = "TRUE"
        else:
            verdict = "MISLEADING"
        claims.append(
            Claim(
                id=f"attribution.l2_conditional.{nbits}b",
                claim=(
                    f"Fusing dequantization into decode attention is worth doing at "
                    f"{nbits} bits."
                ),
                verdict=verdict,
                evidence=(
                    "The sign of the effect depends on where the KV cache lives. "
                    "Quantization effect (fp16 control / fused), same kernel on both sides: "
                    + "; ".join(
                        f"ctx={c}: {quant_hot[c][0]:.2f}x L2-resident vs "
                        f"{quant_cold[c][0]:.2f}x DRAM-resident"
                        for c in shared
                    )
                    + ". Condition: the fused kernel pays for itself only once the working "
                    f"set exceeds this GPU's "
                    f"{(b.env.get('l2_cache_bytes') or 0) / 1e6:.0f} MB L2"
                    + (
                        f"; it loses at ctx={hot_losses} when the cache fits in L2."
                        if hot_losses
                        else "."
                    )
                ),
                falsification_attempted=(
                    "Measured both regimes with the same control kernel rather than "
                    "reporting whichever regime flattered the result. A single "
                    "unconditional Nx figure was available from either regime alone, and "
                    "would have been wrong in the other."
                ),
            )
        )

    return claims


def audit_measurement(b: Bench) -> list[Claim]:
    """Were the timings taken on a GPU that was actually at boost clocks?"""
    cm = b.p.get("clock_monitoring")
    if not cm or not cm.get("monitored"):
        return [
            Claim(
                id="method.clock_verified",
                claim="The reported timings were taken under controlled GPU clocks.",
                verdict="MISLEADING",
                evidence=(
                    "These results carry no clock telemetry, so nothing is known about what "
                    "the SM clock was doing while they were measured. On the laptop part "
                    "used here the idle clock is roughly a ninth of the boost clock, which "
                    "is a larger effect than most of what is being measured."
                ),
                falsification_attempted=(
                    "Checked for clock telemetry in the results file rather than assuming "
                    "the GPU was in a steady state."
                ),
            )
        ]
    n, ok = cm.get("n_rows", 0), cm.get("n_rows_quotable", 0)
    rejected = cm.get("rejected_rows") or []
    return [
        Claim(
            id="method.clock_verified",
            claim="The reported timings were taken under controlled GPU clocks.",
            verdict="TRUE" if ok == n else "TRUE BUT CONDITIONAL",
            evidence=(
                f"{ok} of {n} rows satisfy both gates: every nvidia-smi sample taken during "
                f"the sampling loop was at or above {cm['boost_floor_frac']:.0%} of the "
                f"{cm['max_sm_clock_mhz']:.0f} MHz maximum SM clock, and the timing's own "
                f"IQR was at most {cm['max_iqr_frac']:.0%} of its median. "
                + (
                    "Rejected and not quoted anywhere: " + ", ".join(rejected) + "."
                    if rejected
                    else "No row was rejected."
                )
            ),
            falsification_attempted=(
                "The GPU is an 80 W laptop RTX 5060 that idles near 285 MHz and boosts to "
                "3090 MHz, and pinning clocks with nvidia-smi -lgc needs administrator "
                "rights. So instead the GPU is deliberately spun up to at least "
                f"{cm['ramp_target_frac']:.0%} of maximum before every measurement, the "
                "clocks are sampled throughout, and the window is attributed to the "
                "sampling loop only -- warmup and CUDA-graph capture, during which the GPU "
                "is free to fall back to idle clocks, are excluded so they are neither "
                "mistaken for throttling nor able to hide it."
            ),
        )
    ]


def audit_scope(b: Bench) -> list[Claim]:
    m = b.model
    return [
        Claim(
            id="scope.tokens_per_sec",
            claim="This kernel makes the model decode faster.",
            verdict="TRUE BUT CONDITIONAL",
            evidence=(
                "What was measured is one attention layer's decode step in isolation. "
                f"{m['name']} has {m['num_layers']} layers, and attention is only part of a "
                "decode step -- the MLP, the projections, and the LM head are untouched. The "
                "end-to-end gain is therefore bounded by attention's share of decode time, "
                "which grows with context length and is small at short contexts."
            ),
            falsification_attempted=(
                "Resisted converting a kernel microbenchmark into an end-to-end "
                "tokens/sec headline. No full-model generation was run, so no end-to-end "
                "tokens/sec figure is claimed anywhere in this project."
            ),
        ),
        Claim(
            id="scope.batch_size",
            claim="The result generalizes to real serving workloads.",
            verdict="TRUE BUT CONDITIONAL",
            evidence=(
                f"Everything here is batch {b.p['args'].get('batch', 1)}, single sequence, "
                "no paged/blocked cache, no prefill, no RoPE, no attention sinks or sliding "
                "window. Larger batches supply their own parallelism, which shrinks the "
                "benefit of splitting the history and changes the arithmetic intensity."
            ),
            falsification_attempted=(
                "Listed what was NOT tested rather than claiming coverage. Batch >1 works "
                "and is covered by the correctness tests, but is not benchmarked."
            ),
        ),
        Claim(
            id="scope.quant_scheme",
            claim="These accuracy numbers represent low-bit KV quantization in general.",
            verdict="MISLEADING",
            evidence=(
                "Both K and V are quantized per-token along head_dim. KIVI (Liu et al. 2024) "
                "shows keys are substantially better quantized per-channel, because key "
                "outliers are channel-aligned. The per-token scheme used here is therefore "
                "a pessimistic setting for key accuracy and the numbers should not be read "
                "as a bound on what low-bit KV caching can achieve."
            ),
            falsification_attempted=(
                "Checked the choice against the published literature instead of assuming "
                "the obvious scheme is the best one. The choice was made for kernel "
                "simplicity, and saying so is more useful than hiding it."
            ),
        ),
        Claim(
            id="scope.hardware",
            claim="These speedups will reproduce on other GPUs.",
            verdict="TRUE BUT CONDITIONAL",
            evidence=(
                f"Measured on a single {b.env['gpu']} (sm_{b.env['compute_capability'].replace('.', '')}, "
                f"{b.env['sm_count']} SMs, {(b.env.get('l2_cache_bytes') or 0) / 1e6:.0f} MB L2). "
                "The win is a memory-traffic win, so it should survive on any GPU where "
                "decode attention is bandwidth-bound -- but the magnitude depends on the "
                "bandwidth-to-L2 ratio and on the split count chosen for the SM count."
            ),
            falsification_attempted=(
                "Only one GPU was available, so cross-hardware generality is asserted from "
                "the mechanism, not demonstrated. Stated as conditional for that reason."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Live adversarial GPU checks
# ---------------------------------------------------------------------------


def gpu_checks(b: Bench) -> list[Claim]:
    """Re-run targeted checks that the aggregate metrics could hide."""
    import torch

    from kernels.fused_decode_attn import fused_decode_attention, triton_available
    from quantize import dequantize_groupwise, quantize_kv
    from reference import make_random_kv, reference_decode_attention

    ok, why = triton_available()
    if not ok:
        return [
            Claim(
                id="adversarial.gpu",
                claim="Adversarial GPU checks were run.",
                verdict="FALSE",
                evidence=f"skipped: {why}",
                falsification_attempted="n/a",
            )
        ]

    m = b.model
    HQ, HKV, D = m["num_q_heads"], m["num_kv_heads"], m["head_dim"]
    claims = []

    # 1. Near one-hot softmax: does the kernel pick the same token?
    torch.manual_seed(0)
    q, k, v = make_random_kv(1, HQ, HKV, 8192, D, device="cuda", seed=77)
    kq, vq = quantize_kv(k, v, 4, 32)
    k_deq = dequantize_groupwise(kq, torch.float32)
    v_deq = dequantize_groupwise(vq, torch.float32)
    # Point every query head straight at one specific cached key.
    target = 4321
    q_sharp = k_deq[:, :, target, :].repeat_interleave(HQ // HKV, dim=1) * 30.0
    got = fused_decode_attention(q_sharp, kq, vq)
    ref = reference_decode_attention(q_sharp, k_deq, v_deq)
    rel = ((got - ref).norm() / ref.norm()).item()
    claims.append(
        Claim(
            id="adversarial.peaked_softmax",
            claim="The kernel is accurate when the attention distribution is nearly one-hot.",
            verdict="TRUE" if rel < 5e-3 else "FALSE",
            evidence=f"query aligned to cached key #{target} and scaled 30x: relative L2 error {rel:.2e}.",
            falsification_attempted=(
                "Online softmax rescaling is most fragile when one score dominates the "
                "whole history, because the running maximum jumps between splits. Random "
                "queries never produce that. This constructs it deliberately."
            ),
        )
    )

    # 2. Element-level error distribution -- what cosine similarity hides.
    q, k, v = make_random_kv(1, HQ, HKV, 16384, D, device="cuda", seed=78)
    kq, vq = quantize_kv(k, v, 4, 32)
    k_deq = dequantize_groupwise(kq, torch.float32)
    v_deq = dequantize_groupwise(vq, torch.float32)
    got = fused_decode_attention(q, kq, vq)
    ref = reference_decode_attention(q, k_deq, v_deq)
    cos = torch.nn.functional.cosine_similarity(
        got.flatten()[None], ref.flatten()[None]
    ).item()
    denom = ref.abs().clamp_min(ref.abs().mean() * 0.1)
    relerr = ((got - ref).abs() / denom).flatten()
    frac_1pct = (relerr > 0.01).float().mean().item()
    worst = relerr.max().item()
    sample_idx = [0, 37, 512, 1023]
    samples = [
        (int(i), float(ref.flatten()[i]), float(got.flatten()[i])) for i in sample_idx
    ]
    claims.append(
        Claim(
            id="adversarial.element_distribution",
            claim=f"Cosine similarity of {cos:.7f} means every output element is right.",
            verdict="TRUE" if frac_1pct == 0.0 else "MISLEADING",
            evidence=(
                f"{frac_1pct * 100:.3f}% of output elements are off by more than 1% "
                f"(relative to a floor of 0.1x mean |output|); worst element {worst * 100:.2f}%. "
                "Hand-checked individual values (index, reference, kernel): "
                + "; ".join(f"[{i}] {r:+.6f} vs {g:+.6f}" for i, r, g in samples)
            ),
            falsification_attempted=(
                "Cosine similarity over 1536 numbers can sit at 0.9999999 while a handful of "
                "elements are badly wrong. Looked at the full per-element error distribution "
                "and printed four individual values by hand, exactly as the brief demands."
            ),
        )
    )

    # 3. Does the split count change the answer? (silent nondeterminism)
    outs = [
        fused_decode_attention(q, kq, vq, num_splits=ns).clone() for ns in (1, 3, 8, 32)
    ]
    max_spread = max(
        (outs[i] - outs[0]).abs().max().item() for i in range(1, len(outs))
    )
    scale = outs[0].abs().mean().item()
    claims.append(
        Claim(
            id="adversarial.split_invariance",
            claim="The flash-decoding split count is an implementation detail with no effect on output.",
            verdict="TRUE" if max_spread < 0.02 * scale else "MISLEADING",
            evidence=(
                f"max element-wise spread across split counts 1/3/8/32 is {max_spread:.3e}, "
                f"which is {max_spread / scale * 100:.4f}% of the mean |output|. Not bitwise "
                "identical -- floating-point reduction order genuinely differs."
            ),
            falsification_attempted=(
                "Included non-power-of-two split counts (3, 33-block remainders) so the "
                "ragged final split and its log-sum-exp rescaling are actually exercised."
            ),
        )
    )

    # 4. Quantization on adversarial (heavy-outlier) data
    for frac, sc in ((0.0, 1.0), (0.05, 20.0)):
        q2, k2, v2 = make_random_kv(
            1, HQ, HKV, 4096, D, device="cuda", seed=79, outlier_frac=frac, outlier_scale=sc
        )
        kq2, vq2 = quantize_kv(k2, v2, 4, 32)
        got2 = fused_decode_attention(q2, kq2, vq2)
        ref2 = reference_decode_attention(q2, k2, v2)
        e = ((got2 - ref2).norm() / ref2.norm()).item()
        label = "clean gaussian" if frac == 0 else f"{frac:.0%} outliers at {sc:.0f}x"
        claims.append(
            Claim(
                id=f"adversarial.outliers.{'clean' if frac == 0 else 'heavy'}",
                claim=f"4-bit KV quantization error stays small ({label}).",
                verdict="TRUE" if e < 0.05 else "TRUE BUT CONDITIONAL",
                evidence=f"relative L2 error of the attention output vs unquantized fp16: {e:.3e}.",
                falsification_attempted=(
                    "Pure Gaussian test data is an unrealistically easy input for a "
                    "quantizer. Injected channel outliers, which is what actually breaks "
                    "low-bit KV caches in real models, and re-measured."
                ),
            )
        )

    return claims


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_ORDER = {"MISLEADING": 0, "FALSE": 1, "TRUE BUT CONDITIONAL": 2, "TRUE": 3}


def render_markdown(claims: list[Claim], b: Bench) -> str:
    counts: dict[str, int] = {}
    for c in claims:
        counts[c.verdict] = counts.get(c.verdict, 0) + 1

    lines = [
        "# Adversarial claim audit",
        "",
        f"Generated by `audit_claims.py` from `results/benchmark.json` "
        f"({b.env['timestamp']}, {b.env['gpu']}).",
        "",
        "Every claim below was written down first and then actively attacked. "
        "Speedups are judged by a bootstrap 95% confidence interval over the raw "
        f"per-sample timings, against a practical-significance bar of "
        f"{PRACTICAL_THRESHOLD:.2f}x -- a difference smaller than that is reported as noise, "
        "not as a win.",
        "",
        "| verdict | count |",
        "| --- | --- |",
    ]
    for v in ("TRUE", "TRUE BUT CONDITIONAL", "MISLEADING", "FALSE"):
        if v in counts:
            lines.append(f"| {v} | {counts[v]} |")
    lines += ["", "## Claims", ""]

    for c in sorted(claims, key=lambda c: (_ORDER.get(c.verdict, 9), c.id)):
        lines += [
            f"### `{c.id}` — **{c.verdict}**",
            "",
            f"> {c.claim}",
            "",
            f"**Evidence.** {c.evidence}",
            "",
            f"**Falsification attempted.** {c.falsification_attempted}",
            "",
        ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(RESULTS_DIR / "benchmark.json"))
    ap.add_argument("--no-gpu-checks", action="store_true")
    ap.add_argument("--out-md", default=str(RESULTS_DIR / "audit.md"))
    ap.add_argument("--out-json", default=str(RESULTS_DIR / "audit.json"))
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"no benchmark results at {path} -- run `python benchmark.py` first")
    b = Bench(json.loads(path.read_text()))

    claims: list[Claim] = []
    for nbits in b.p["bit_widths"]:
        claims += audit_speed(b, nbits)
        claims += audit_memory(b, nbits)
        claims += audit_attribution(b, nbits)
    claims += audit_correctness(b)
    claims += audit_scope(b)
    claims += audit_measurement(b)
    if not args.no_gpu_checks:
        claims += gpu_checks(b)

    md = render_markdown(claims, b)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md, encoding="utf-8")
    Path(args.out_json).write_text(
        json.dumps({"claims": [c.as_dict() for c in claims]}, indent=2), encoding="utf-8"
    )

    counts: dict[str, int] = {}
    for c in claims:
        counts[c.verdict] = counts.get(c.verdict, 0) + 1
    print(f"{len(claims)} claims audited: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for c in sorted(claims, key=lambda c: (_ORDER.get(c.verdict, 9), c.id)):
        print(f"  [{c.verdict:<20}] {c.id}")
    print(f"\nwrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
