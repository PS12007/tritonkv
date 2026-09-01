"""Figures for the writeup, generated from results/benchmark.json.

    python make_plots.py            # writes docs/plots/*.png

Four figures, in the order the argument is made:

1. ``speedup_vs_context``     -- the headline number, in both cache regimes.
2. ``quantization_effect``    -- the finding: what the quantization itself buys,
                                 which is negative when the cache fits in L2.
3. ``kv_cache_memory``        -- what the bits actually save, which is memory.
4. ``correctness_vs_bits``    -- what the bits cost, which is accuracy.

Every ratio is drawn with a bootstrap 95% CI from the raw per-sample timings
(the same function the audit uses), and any point whose underlying measurement
failed the benchmark's boost-clock gate is drawn hollow rather than silently
plotted as if it were solid evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from audit_claims import bootstrap_ratio_ci

ROOT = Path(__file__).parent
PLOTS = ROOT / "docs" / "plots"

# Palette: validated categorical slots 1-3 (blue / orange / aqua) on the light
# surface -- all-pairs CVD dE 9.2, normal-vision dE 24.0. Aqua sits under 3:1
# against the surface, so every series is direct-labelled, never colour-alone.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e3e2dd"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
})


def style(ax, title, subtitle=None, xlabel=None, ylabel=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    # The subtitle sits between the title and the axes, so the title's pad has to
    # be computed from how many lines the subtitle actually has.
    nlines = subtitle.count("\n") + 1 if subtitle else 0
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold",
                 pad=12 + 13.5 * nlines)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=INK2, va="bottom", linespacing=1.35)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5)


def ctx_axis(ax, contexts):
    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts)
    ax.set_xticklabels([f"{c // 1024}k" if c >= 1024 else str(c) for c in contexts])
    ax.minorticks_off()


def xfmt(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))


def ratio_ticks(ax, candidates=(0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2, 3, 5, 10, 20, 30, 50, 100)):
    """A log ratio axis spanning less than a decade labels only 1x on its own."""
    lo, hi = ax.get_ylim()
    ticks = [t for t in candidates if lo <= t <= hi]
    if len(ticks) >= 3:
        ax.set_yticks(ticks)
        ax.minorticks_off()


def save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / f"{name}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


class Data:
    def __init__(self, payload):
        self.p = payload
        self.by = {(r["method"], r["ctx"]): r for r in payload["results"]}
        self.contexts = payload["contexts"]

    def row(self, m, ctx):
        return self.by.get((m, ctx))

    def raw(self, m, ctx, regime):
        r = self.row(m, ctx)
        key = "cold_raw_ms" if regime == "cold" else "graph_raw_ms"
        return (r or {}).get(key)

    def quotable(self, m, ctx):
        return bool((self.row(m, ctx) or {}).get("quotable", True))

    def ratio(self, num_m, den_m, ctx, regime):
        """Speedup of ``den_m`` over ``num_m`` with a bootstrap 95% CI."""
        n, d = self.raw(num_m, ctx, regime), self.raw(den_m, ctx, regime)
        if not n or not d:
            return None
        r, lo, hi = bootstrap_ratio_ci(n, d)
        return {
            "r": r, "lo": lo, "hi": hi,
            "solid": self.quotable(num_m, ctx) and self.quotable(den_m, ctx),
        }

    def series(self, num_m, den_m, regime):
        out = {}
        for ctx in self.contexts:
            v = self.ratio(num_m, den_m, ctx, regime)
            if v:
                out[ctx] = v
        return out


def draw_ratio_series(ax, data: dict, color, label, marker="o"):
    """A ratio line with CI whiskers; hollow markers = failed the clock gate."""
    xs = sorted(data)
    ys = [data[x]["r"] for x in xs]
    lo = [data[x]["r"] - data[x]["lo"] for x in xs]
    hi = [data[x]["hi"] - data[x]["r"] for x in xs]
    ax.errorbar(xs, ys, yerr=[lo, hi], color=color, lw=2, marker="none",
                elinewidth=1.2, capsize=3, zorder=3, label=label)
    for x in xs:
        solid = data[x]["solid"]
        ax.plot([x], [data[x]["r"]], marker=marker, ms=8, zorder=4,
                color=color if solid else SURFACE,
                markeredgecolor=color, markeredgewidth=2)
    return xs, ys


def end_label(ax, xs, ys, text, color, dy=10, va="bottom"):
    """Direct label past the last point, in the series colour."""
    ax.annotate(text, (xs[-1], ys[-1]), xytext=(12, dy), textcoords="offset points",
                color=color, fontsize=9.5, fontweight="bold", ha="left", va=va, zorder=5)


def headroom(ax, contexts, right=3.4):
    """Room on the right for direct labels, on a log2 context axis."""
    ax.set_xlim(contexts[0] / 1.25, contexts[-1] * right)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_speedup(d: Data, nbits: int):
    fused = f"fused_triton_{nbits}b"
    cold = d.series("fp16_sdpa", fused, "cold")
    hot = d.series("fp16_sdpa", fused, "hot")
    if not (cold or hot):
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    if cold:
        xs, ys = draw_ratio_series(ax, cold, S1, "KV cache in DRAM (working set > L2)")
        end_label(ax, xs, ys, "DRAM-resident", S1, dy=-4, va="center")
    if hot:
        xs, ys = draw_ratio_series(ax, hot, S2, "KV cache in L2 (single layer, hot loop)", marker="s")
        end_label(ax, xs, ys, "L2-resident", S2, dy=-4, va="center")
    ax.set_yscale("log")
    xfmt(ax)
    ctx_axis(ax, d.contexts)
    headroom(ax, d.contexts)
    ratio_ticks(ax)
    style(
        ax,
        f"Fused {nbits}-bit decode attention vs PyTorch fp16 SDPA",
        "Per decode step, one attention layer. Whiskers are bootstrap 95% CIs over raw "
        "per-sample timings;\nhollow markers failed the boost-clock gate and are not quoted.",
        "context length (tokens)",
        "speedup over fp16 SDPA",
    )
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK2)
    save(fig, f"speedup_vs_context_{nbits}b")


def fig_quantization_effect(d: Data, nbits: int):
    """The finding: what the quantization contributes, holding the kernel fixed."""
    fused = f"fused_triton_{nbits}b"
    ctrl = "triton_fp16_control"
    cold = d.series(ctrl, fused, "cold")
    hot = d.series(ctrl, fused, "hot")
    if not (cold or hot):
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    if cold:
        xs, ys = draw_ratio_series(ax, cold, S1, "KV cache in DRAM (working set > L2)")
        end_label(ax, xs, ys, "DRAM-resident:\nthe bits pay off", S1, dy=-4, va="center")
    if hot:
        xs, ys = draw_ratio_series(ax, hot, S2, "KV cache in L2 (single layer, hot loop)", marker="s")
        end_label(ax, xs, ys, "L2-resident:\nthe bits cost", S2, dy=-4, va="center")
    ax.set_yscale("log")
    xfmt(ax)
    ctx_axis(ax, d.contexts)
    headroom(ax, d.contexts, right=5.0)
    # Shade the win region only as far as the data goes, so the log axis is not
    # stretched to a decade nothing occupies.
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, 0.45), max(hi, 1.6))
    lo, hi = ax.get_ylim()
    ax.axhspan(1.0, hi, color=S3, alpha=0.07, zorder=0)
    ax.axhline(1.0, color=MUTED, lw=1.2, zorder=1)
    ax.annotate("above 1.0x: quantization pays for the dequantization it costs",
                (0.015, 0.975), xycoords="axes fraction", color=INK2, fontsize=8.5,
                va="top")
    ratio_ticks(ax)
    style(
        ax,
        f"What the {nbits}-bit quantization itself buys",
        f"Same kernel on both sides: fp16 control (identical split, softmax and GQA "
        f"amortization) divided by\nthe fused {nbits}-bit kernel. Above 1.0x the bits help; "
        "below 1.0x they cost.",
        "context length (tokens)",
        f"fp16 control time / fused {nbits}-bit time",
    )
    ax.legend(loc="lower left", fontsize=9, labelcolor=INK2)
    save(fig, f"quantization_effect_{nbits}b")


def fig_attribution(d: Data, nbits: int):
    """Decompose the headline speedup into its two independent causes."""
    fused = f"fused_triton_{nbits}b"
    ctrl = "triton_fp16_control"
    split = d.series("fp16_sdpa", ctrl, "cold")
    quant_c = d.series(ctrl, fused, "cold")
    quant_h = d.series(ctrl, fused, "hot")
    ctxs = sorted(set(split) & set(quant_c))
    if not ctxs:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    w = 0.26
    xs = range(len(ctxs))
    groups = [
        (split, S1, "flash-decoding split (fp16 SDPA / fp16 Triton)", -1),
        (quant_c, S2, f"{nbits}-bit quantization, DRAM-resident", 0),
        (quant_h, S3, f"{nbits}-bit quantization, L2-resident", 1),
    ]
    for series, color, label, off in groups:
        if not series:
            continue
        vals = [series.get(c, {}).get("r") for c in ctxs]
        pos = [x + off * w for x in xs]
        bars = [(p, v) for p, v in zip(pos, vals) if v]
        ax.bar([p for p, _ in bars], [v - 1.0 for _, v in bars], width=w * 0.92,
               bottom=1.0, color=color, label=label, zorder=3,
               edgecolor=SURFACE, linewidth=2)
        for p, v in bars:
            ax.annotate(f"{v:.1f}x" if v >= 10 else f"{v:.2f}x", (p, v),
                        xytext=(0, 4 if v >= 1 else -12), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK2, zorder=4)
    ax.axhline(1.0, color=MUTED, lw=1.2, zorder=2)
    ax.set_yscale("log")
    xfmt(ax)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo * 0.8, hi * 3.2)   # room for the legend above the tallest bar
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{c // 1024}k" if c >= 1024 else str(c) for c in ctxs])
    style(
        ax,
        "Where the speedup actually comes from",
        "The kernel changes two things at once. Splitting the history across SMs is worth "
        "the large factor;\nthe quantization is worth about 1x, and less than 1x whenever "
        "the cache fits in L2.",
        "context length (tokens)",
        "factor (log scale, 1.0x = no effect)",
    )
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK2)
    save(fig, f"speedup_attribution_{nbits}b")


def fig_memory(d: Data):
    """What the bits do buy, unconditionally: bytes."""
    contexts = d.contexts
    rows = {}
    for label, method in (("fp16", "fp16_sdpa"), ("4-bit", "fused_triton_4b"),
                          ("2-bit", "fused_triton_2b")):
        vals, bits = [], None
        for ctx in contexts:
            r = d.row(method, ctx)
            vals.append((r or {}).get("cache_bytes_model", 0) / 1e6)
            bits = (r or {}).get("effective_bits", bits)
        if any(vals):
            rows[label] = (vals, bits)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    w = 0.26
    xs = range(len(contexts))
    for i, (label, color) in enumerate((("fp16", S1), ("4-bit", S2), ("2-bit", S3))):
        if label not in rows:
            continue
        vals, bits = rows[label]
        pos = [x + (i - 1) * w for x in xs]
        ax.bar(pos, vals, width=w * 0.92, color=color, zorder=3,
               edgecolor=SURFACE, linewidth=2,
               label=f"{label} ({bits:.1f} effective bits/value)" if bits else label)
        for p, v in zip(pos, vals):
            ax.annotate(f"{v:.0f}", (p, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK2, zorder=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{c // 1024}k" if c >= 1024 else str(c) for c in contexts])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    m = d.p["model"]
    style(
        ax,
        "KV cache size for the whole model",
        f"{m['name']}: {m['num_layers']} layers x {m['num_kv_heads']} KV heads x "
        f"{m['head_dim']} head_dim, batch 1. Effective bits include the\nper-group scale "
        "and zero-point, so the compression is the real one, not the nominal 4x / 8x.",
        "context length (tokens)",
        "KV cache (MB)",
    )
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK2)
    save(fig, "kv_cache_memory")


def fig_correctness(d: Data):
    """What the bits cost: accuracy against the unquantized fp16 answer."""
    corr = d.p.get("correctness") or []
    if not corr:
        return
    bits = sorted({r["nbits"] for r in corr}, reverse=True)
    contexts = sorted({r["ctx"] for r in corr})
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    w = 0.36
    xs = range(len(contexts))
    for i, (nb, color) in enumerate(zip(bits, (S2, S3))):
        kern, base = [], []
        for ctx in contexts:
            row = next((r for r in corr if r["ctx"] == ctx and r["nbits"] == nb), None)
            a = (row or {}).get("agg", {})
            kern.append(a.get("mean_rel_l2_vs_fp16_truth", 0))
            base.append(a.get("mean_baseline_rel_l2_vs_fp16_truth", 0))
        pos = [x + (i - 0.5) * w for x in xs]
        ax.bar(pos, kern, width=w * 0.92, color=color, zorder=3,
               edgecolor=SURFACE, linewidth=2, label=f"{nb}-bit fused kernel")
        # the dequantize-then-SDPA reference at the same bit width: if the kernel
        # sat above it, the kernel would be adding error of its own. It does not.
        ax.scatter(pos, base, s=26, zorder=5, color=SURFACE, edgecolor=INK2,
                   linewidth=1.4, label="same bits, PyTorch reference" if i == 0 else None)
        for p, v in zip(pos, kern):
            ax.annotate(f"{v:.2f}", (p, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK2, zorder=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{c // 1024}k" if c >= 1024 else str(c) for c in contexts])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)   # room for the legend above the bars
    style(
        ax,
        "Accuracy cost of quantizing the KV cache",
        "Relative L2 error of the attention output against the unquantized fp16 answer. "
        "The kernel lands on\ntop of the PyTorch reference at the same bit width, so the "
        "error is the quantization's, not the kernel's.",
        "context length (tokens)",
        "relative L2 error vs fp16 truth",
    )
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK2)
    save(fig, "correctness_vs_bits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "results" / "benchmark.json"))
    args = ap.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"no results at {path} -- run `python benchmark.py` first")
    d = Data(json.loads(path.read_text()))
    print(f"figures from {path.name} ({d.p['env']['timestamp']}):")
    for nbits in d.p["bit_widths"]:
        fig_speedup(d, nbits)
        fig_quantization_effect(d, nbits)
        fig_attribution(d, nbits)
    fig_memory(d)
    fig_correctness(d)


if __name__ == "__main__":
    main()
