"""Figures for the 2026-09-01 evening findings: the metadata broadcast, the
variants that were rejected, and the second repair to the clock gate.

Kept separate from ``make_plots.py`` because these are argument figures about
*this project's own process* rather than the standard result set. Style is
imported from ``make_plots`` so the two sets look like one deck.

Everything here is derived live -- from ``results/benchmark.json``, from the
Triton compiler, or from a GPU run -- except the two values named in
``RECORDED``, which come from measurements logged in ``docs/progress_log.md``
and are labelled as recorded wherever they appear.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_plots import (
    GRID, INK, INK2, MUTED, PLOTS, ROOT, S1, S2, S3, SURFACE, save, style,
)

# Measured in this session and written down in the progress log. The register
# counts below are recomputed live; these two are not, because the instruction
# count comes from a Triton compiler remark on stderr and the third variant was
# reverted, so neither can be recompiled from the tree as it now stands.
RECORDED = {
    "instructions": {"gather": 2245, "broadcast": 1653},
    "code_bcast_regs": 223,
    # clock samples per measurement window, full run before the min-sample fix
    "clock_hist_before": {1: 28, 2: 5, 3: 2, 4: 4, 8: 2, 9: 19, 10: 14,
                          11: 1, 12: 2, 13: 1, 14: 2, 17: 5, 18: 3,
                          21: 1, 22: 4, 28: 3},
}

CTX_LABEL = {512: "512", 2048: "2k", 8192: "8k", 16384: "16k"}


def figure_header(fig, title, subtitle, title_size=14, sub_size=9.5, gap=0.035):
    """Stack a figure title above a multi-line subtitle without them colliding.

    ``fig.suptitle`` takes a y in figure coordinates, but the subtitle's height
    depends on its line count and on the figure's height in inches, so a fixed y
    overlaps as soon as the subtitle grows a line. Compute it instead.
    """
    nlines = subtitle.count("\n") + 1
    line_h = (sub_size * 1.35) / 72.0 / fig.get_figheight()
    fig.text(0.005, 1.0, subtitle, ha="left", va="bottom",
             fontsize=sub_size, color=INK2, linespacing=1.35)
    fig.text(0.005, 1.0 + nlines * line_h + gap, title, ha="left", va="bottom",
             fontsize=title_size, fontweight="bold", color=INK)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def by_method(payload: dict) -> dict:
    return {(r["method"], r["ctx"]): r for r in payload["results"]}


def median_us(row: dict, regime: str):
    s = row.get(regime) if row else None
    return s["median_ms"] * 1e3 if isinstance(s, dict) and "median_ms" in s else None


# ---------------------------------------------------------------------------


def plot_broadcast_speedup(payload: dict):
    """gather / broadcast, both regimes, both bit widths."""
    by = by_method(payload)
    contexts = payload["contexts"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for ax, nbits in zip(axes, payload["bit_widths"]):
        rows = {"L2-resident": [], "DRAM-resident": []}
        quot = {"L2-resident": [], "DRAM-resident": []}
        for ctx in contexts:
            g = by.get((f"fused_gather_meta_{nbits}b", ctx))
            b = by.get((f"fused_triton_{nbits}b", ctx))
            for label, regime in (("L2-resident", "graph"), ("DRAM-resident", "cold")):
                gu, bu = median_us(g, regime), median_us(b, regime)
                rows[label].append(gu / bu if gu and bu else np.nan)
                quot[label].append(bool(g and b and g["quotable"] and b["quotable"]))

        x = np.arange(len(contexts))
        w = 0.36
        for i, (label, colour) in enumerate((("L2-resident", S1), ("DRAM-resident", S2))):
            vals = rows[label]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, color=colour, label=label,
                          edgecolor=colour)
            # A bar whose inputs did not both pass the gate is drawn hollow, so
            # the eye cannot pick it up as evidence.
            for bar, ok in zip(bars, quot[label]):
                if not ok:
                    bar.set_facecolor("none")
                    bar.set_hatch("///")
                    bar.set_linewidth(1.1)
            for xi, v, ok in zip(x + (i - 0.5) * w, vals, quot[label]):
                if not np.isnan(v):
                    ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                            fontsize=8.5, color=INK if ok else MUTED)

        ax.axhline(1.0, color=INK2, lw=1.1, ls="--", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([CTX_LABEL[c] for c in contexts])
        ax.set_ylim(0.9, max(1.55, np.nanmax(rows["L2-resident"]) + 0.09))
        style(ax, f"{nbits}-bit",
              xlabel="context length (tokens)",
              ylabel="speedup over the gather path" if nbits == payload["bit_widths"][0] else None)
        if nbits == payload["bit_widths"][0]:
            ax.legend(loc="upper left", fontsize=9)

    figure_header(
        fig,
        "Loading the per-group scale/zero at its real width",
        "Same kernel, same tuned config, bitwise-identical output — the only\n"
        "difference is whether each scale/zero is re-read group_size times.\n"
        "Hatched bars had an input that failed the clock/dispersion gate.",
    )
    save(fig, "meta_broadcast_speedup")


def plot_inner_loop_cost(regs: dict):
    """Why it got faster: fewer instructions, and half the registers."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    ax = axes[0]
    ins = RECORDED["instructions"]
    bars = ax.bar(["gather", "broadcast"], [ins["gather"], ins["broadcast"]],
                  width=0.55, color=[MUTED, S1])
    for b, v in zip(bars, (ins["gather"], ins["broadcast"])):
        ax.text(b.get_x() + b.get_width() / 2, v + 40, f"{v}", ha="center",
                fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(0, ins["gather"] * 1.18)
    style(ax, "Instructions in the kernel",
          subtitle="Triton compiler remark, 4-bit (recorded)", ylabel="instructions")

    ax = axes[1]
    names = ["gather", "broadcast", "code\nbroadcast"]
    vals = [regs["gather"], regs["broadcast"], RECORDED["code_bcast_regs"]]
    colours = [MUTED, S1, S2]
    bars = ax.bar(names, vals, width=0.55, color=colours)
    # The third bar is the rejected variant: same idea, applied to the codes.
    bars[2].set_facecolor("none")
    bars[2].set_edgecolor(S2)
    bars[2].set_hatch("///")
    bars[2].set_linewidth(1.1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v}", ha="center",
                fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(0, max(vals) * 1.2)
    style(ax, "Registers per thread",
          subtitle="live from Triton, except the rejected variant (recorded)",
          ylabel="registers")
    ax.text(2, RECORDED["code_bcast_regs"] * 0.42, "rejected\nslower",
            ha="center", va="center", fontsize=9, color=S2, fontweight="bold",
            linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                      edgecolor="none"))

    figure_header(
        fig,
        "The same trick, applied twice, with opposite results",
        "Narrowing the scale/zero load halves the live register footprint.\n"
        "Narrowing the packed-code load pushes it back up — the codes are needed\n"
        "at full width anyway, so it adds a live tile without removing one.",
    )
    fig.subplots_adjust(wspace=0.28)
    save(fig, "inner_loop_cost")


def plot_clock_samples(payload: dict):
    """How many nvidia-smi samples each measurement window actually contained."""
    after = {}
    for r in payload["results"]:
        for regime in ("cold", "graph"):
            c = ((r.get("clocks") or {}).get(regime) or {}).get("clocks") or {}
            n = c.get("n_samples")
            if n is not None:
                after[n] = after.get(n, 0) + 1
    before = RECORDED["clock_hist_before"]

    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    hi = max(max(before), max(after)) + 1
    edges = np.arange(0, hi + 2)
    for data, label, colour, off in ((before, "before", S2, -0.2),
                                     (after, "after", S1, 0.2)):
        xs = np.array(sorted(data))
        ys = np.array([data[k] for k in xs])
        ax.bar(xs + off, ys, 0.4, color=colour, label=label, edgecolor=colour)

    ax.axvspan(-0.5, 3.5, color=S2, alpha=0.07, zorder=0)
    ax.axvline(4, color=INK2, lw=1.1, ls="--")
    ax.text(4.35, ax.get_ylim()[1] * 0.93, "gate now requires ≥ 4",
            fontsize=9, color=INK2)
    ax.text(1.6, ax.get_ylim()[1] * 0.62,
            f"{sum(v for k, v in before.items() if k <= 3)} windows\njudged on ≤ 3 samples",
            ha="center", fontsize=9.5, color=S2, fontweight="bold", linespacing=1.35)
    ax.set_xlim(-0.6, hi)
    style(ax, "The clock gate was answering from noise",
          subtitle="nvidia-smi samples inside each measurement window, "
                   "96 windows per full run\n"
                   "The sampler runs at 9.2 Hz, so a 30 ms measurement cannot "
                   "observe the clocks at all.",
          xlabel="nvidia-smi samples in the window",
          ylabel="measurement windows")
    ax.legend(loc="upper right", fontsize=9.5)
    save(fig, "clock_samples_per_window")


def plot_fold_accuracy():
    """The rejected-for-speed variant is the accurate one."""
    import torch

    from kernels.fused_decode_attn import fused_decode_attention
    from quantize import dequantize_groupwise, quantize_kv
    from reference import make_random_kv, reference_decode_attention

    contexts = [512, 2048, 8192, 16384]
    series = {"dequantize K first (shipped)": [], "fold the zero-point out": []}
    for ctx in contexts:
        q, k, v = make_random_kv(1, 12, 2, ctx, 128, device="cuda", seed=5)
        kq, vq = quantize_kv(k, v, 4, 32)
        ref = reference_decode_attention(
            q, dequantize_groupwise(kq, torch.float32),
            dequantize_groupwise(vq, torch.float32))
        for label, fold in (("dequantize K first (shipped)", False),
                            ("fold the zero-point out", True)):
            got = fused_decode_attention(q, kq, vq, fold_zp=fold)
            series[label].append(((got - ref).norm() / ref.norm()).item())

    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    for (label, ys), colour, marker in zip(series.items(), (S2, S3), ("o", "s")):
        ax.plot(contexts, ys, marker=marker, color=colour, lw=2, ms=7, label=label)
        ax.text(contexts[-1] * 1.06, ys[-1], f"{ys[-1]:.1e}", color=colour,
                fontsize=9.5, va="center", fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(contexts)
    ax.set_xticklabels([CTX_LABEL[c] for c in contexts])
    ax.minorticks_off()
    ax.set_xlim(contexts[0] * 0.82, contexts[-1] * 2.4)
    ax.set_yticks([2e-4, 3e-4, 5e-4, 1e-3])
    ax.set_yticklabels(["2e-4", "3e-4", "5e-4", "1e-3"])
    style(ax, "The variant I rejected for speed is the accurate one",
          subtitle="kernel error against the dequantize-then-attend reference, "
                   "4-bit\nFolding never rounds a dequantized K value to fp16, "
                   "so its error does not grow with context.",
          xlabel="context length (tokens)", ylabel="relative L2 error")
    ax.legend(loc="upper left", fontsize=9.5)
    save(fig, "fold_accuracy")


# ---------------------------------------------------------------------------
# Group-size sweep (results/gs_sweep.json, written by sweep_group_size.py)
# ---------------------------------------------------------------------------


def _sweep(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _cell(rows, ctx, gs, path_name, regime="hot"):
    for r in rows:
        if r["ctx"] == ctx and r["group_size"] == gs and r["path"] == path_name:
            st = r.get(regime)
            return r, (st["median_ms"] * 1e3 if st else None)
    return None, None


def plot_gs_saturation(sweep: dict):
    """Time against metadata loads per tile: the cost saturates, it is not linear.

    The prediction going in was that the broadcast path's sweep would be sloped,
    because there ``GS`` really does set the load count. It is nearly flat. What
    that flatness means only becomes visible with the gather point on the same
    axis: the expensive step is the first order of magnitude, and there is very
    little left below it.
    """
    rows = sweep["rows"]
    contexts = [c for c in (2048, 8192) if any(r["ctx"] == c for r in rows)]
    fig, axes = plt.subplots(1, len(contexts), figsize=(5.6 * len(contexts), 4.2))
    axes = np.atleast_1d(axes)

    for ax, ctx in zip(axes, contexts):
        xs, ys, labels, oks = [], [], [], []
        for gs in sweep["group_sizes"]:
            r, us = _cell(rows, ctx, gs, "broadcast")
            if us:
                xs.append(r["meta_loads_per_tile"])
                ys.append(us)
                labels.append("gs=%d" % gs)
                oks.append(r["quotable"])
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        labels = [labels[i] for i in order]
        oks = [oks[i] for i in order]
        ax.plot(xs, ys, "-", color=S1, lw=1.6, zorder=2)
        for x, y, lab, ok in zip(xs, ys, labels, oks):
            ax.plot([x], [y], "o", ms=7, color=S1 if ok else "none",
                    markeredgecolor=S1, markeredgewidth=1.4, zorder=3)
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", va="bottom", fontsize=8.5, color=INK2)

        # The gather path sits at a single x -- BLOCK_N * D loads whatever GS is,
        # which is the whole point of the original mistake -- so it is drawn as
        # the one point it really is. gs=128 is excluded: that cell is a
        # different effect (see plot_gs128_cliff) and would read as part of a
        # trend it has nothing to do with.
        gvals = []
        for gs in (16, 32, 64):
            _, us = _cell(rows, ctx, gs, "gather")
            if us:
                gvals.append(us)
        gref = _cell(rows, ctx, 32, "gather")[0]
        if gvals and gref:
            gx = gref["meta_loads_per_tile"]
            gy = float(np.median(gvals))
            ax.plot([gx], [gy], "s", ms=8, color=MUTED, markeredgecolor=MUTED,
                    zorder=3)
            ax.annotate("gather (gs=16/32/64)", (gx, gy),
                        textcoords="offset points", xytext=(-10, -3), ha="right",
                        va="center", fontsize=8.5, color=INK2)
            lo, hi = min(float(ys.min()), gy), max(float(ys.max()), gy)
            pad = (hi - lo) * 0.16
            ax.set_ylim(lo - pad, hi + pad)
            # The step the broadcast change actually made, drawn so the eye reads
            # it as one move rather than as two unrelated clusters.
            ax.plot([gx, xs[-1]], [gy, ys[-1]], ls=":", lw=1.3, color=MUTED,
                    zorder=1)
            ax.text(0.30, 0.70,
                    "%dx fewer loads: %.2fx faster\na further %dx: %.2fx"
                    % (gx // xs[-1], gy / ys[-1], xs[-1] // xs[0], ys[-1] / ys[0]),
                    transform=ax.transAxes, fontsize=9, color=INK, va="center",
                    bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID))

        ax.set_xscale("log", base=2)
        style(ax, "ctx = %s" % CTX_LABEL.get(ctx, ctx),
              subtitle="L2-resident, CUDA-graph replay",
              xlabel="per-group scale/zero loads per tile",
              ylabel="median latency (us)")

    figure_header(
        fig,
        "Metadata loads are a real cost that stops binding",
        "Same kernel, same config; only the metadata load count changes.\n"
        "Hollow markers failed the clock/dispersion gate.",
    )
    save(fig, "gs_saturation")


def plot_gs128_cliff(sweep: dict):
    """The unexplained gs=128 outlier, and the shared-memory traffic behind it."""
    rows = sweep["rows"]
    contexts = [c for c in (512, 2048, 8192) if any(r["ctx"] == c for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    x = np.arange(len(contexts))
    w = 0.27
    series = (("gather, gs=64", 64, "gather", MUTED),
              ("gather, gs=128", 128, "gather", S2),
              ("broadcast, gs=128", 128, "broadcast", S1))
    top = 0.0
    for i, (label, gs, pname, colour) in enumerate(series):
        vals, oks = [], []
        for ctx in contexts:
            r, us = _cell(rows, ctx, gs, pname)
            vals.append(us if us else np.nan)
            oks.append(bool(r and r["quotable"]))
        top = max(top, np.nanmax(vals))
        bars = ax.bar(x + (i - 1) * w, vals, w, color=colour, edgecolor=colour,
                      label=label)
        for bar, ok in zip(bars, oks):
            if not ok:
                bar.set_facecolor("none")
                bar.set_hatch("///")
                bar.set_linewidth(1.1)
        for xi_, v in zip(x + (i - 1) * w, vals):
            if not np.isnan(v):
                ax.text(xi_, v + top * 0.015, "%.0f" % v, ha="center",
                        va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([CTX_LABEL.get(c, str(c)) for c in contexts])
    ax.set_ylim(0, top * 1.18)
    ax.legend(loc="upper left", fontsize=9)
    style(ax, "The cliff is only on the gather path",
          subtitle="L2-resident; hatched bars failed the gate",
          xlabel="context length (tokens)", ylabel="median latency (us)")

    ax = axes[1]
    ctx = contexts[-1]
    cells = (("gather\ngs=64", 64, "gather", MUTED),
             ("gather\ngs=128", 128, "gather", S2),
             ("broadcast\ngs=128", 128, "broadcast", S1))
    names, st_vals, ld_vals, colours = [], [], [], []
    for label, gs, pname, colour in cells:
        r, _ = _cell(rows, ctx, gs, pname)
        ops = (r or {}).get("ptx_ops") or {}
        names.append(label)
        st_vals.append(ops.get("st.shared", 0))
        ld_vals.append(ops.get("ld.shared", 0))
        colours.append(colour)
    xi = np.arange(len(names))
    ax.bar(xi - 0.19, st_vals, 0.36, color=colours, label="st.shared")
    ax.bar(xi + 0.19, ld_vals, 0.36, color=colours, alpha=0.45, label="ld.shared")
    hi = max(st_vals + ld_vals) or 1
    for a, v in zip(xi - 0.19, st_vals):
        ax.text(a, v + hi * 0.02, str(v), ha="center", fontsize=9,
                fontweight="bold", color=INK)
    for a, v in zip(xi + 0.19, ld_vals):
        ax.text(a, v + hi * 0.02, str(v), ha="center", fontsize=9, color=INK2)
    ax.set_xticks(xi)
    ax.set_xticklabels(names)
    ax.set_ylim(0, hi * 1.25)
    ax.legend(loc="upper left", fontsize=9)
    style(ax, "...and it is shared memory, not loads",
          subtitle="PTX counts; the three cells issue identical global loads",
          ylabel="instructions")

    figure_header(
        fig,
        "When group_size == head_dim, the index folds to a constant",
        "Triton then gives the loaded tile a layout it has to convert through shared\n"
        "memory. The slow cell issues FEWER instructions than gs=64 and the same number\n"
        "of global loads; what it adds is a round trip, inside the loop.",
    )
    save(fig, "gs128_cliff")


# ---------------------------------------------------------------------------
# Dispersion gate (results/dispersion.json, written by analyze_dispersion.py)
# ---------------------------------------------------------------------------


CAUSE_SHORT = {
    "drift (shorter windows)": "drift\n(shorter window helps)",
    "drift + floor (shorter windows, not enough)": "drift + floor\n(helps, not enough)",
    "tail (trimmed stat)": "tail\n(trimmed stat helps)",
    "wander (no window length helps)": "wander\n(no window length helps)",
    "white jitter (no window length helps)": "white jitter\n(no window length helps)",
}


def plot_dispersion(rows: list[dict]):
    """What the IQR gate rejects, against what the rejected rows actually cost.

    Two things in one figure because they only mean something together: the gate
    rejects a quarter of all measurements, and the rejected measurements still
    pin their medians an order of magnitude finer than the effects they are used
    to establish.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.6),
                             gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.42})

    ax = axes[0]
    for label, colour, sel in (
        ("passes the gate", MUTED, [d for d in rows if d["cause"] == "passes"]),
        ("fails the gate", S2, [d for d in rows if d["cause"] != "passes"]),
    ):
        ax.scatter([d["iqr_frac"] * 100 for d in sel],
                   [d["median_ci_halfwidth_frac"] * 100 for d in sel],
                   s=26, color=colour, alpha=0.8, edgecolor="none", label=label)
    ax.axvline(5.0, color=INK2, lw=1.2, ls="--")
    ax.text(5.3, ax.get_ylim()[1] * 0.93, "gate: IQR = 5%", fontsize=8.5, color=INK2)
    # The band the conclusions actually live in. Nothing in this plot comes close
    # to it, which is the point.
    ax.axhspan(10, 50, color=S1, alpha=0.10, zorder=0)
    ax.text(ax.get_xlim()[1] * 0.98, 18, "effects reported: 10-50%",
            fontsize=8.5, color=S1, ha="right")
    ax.set_yscale("log")
    ax.legend(loc="lower right", fontsize=9)
    style(ax, "Rejected rows still pin their medians",
          subtitle="one point per measurement; y from a block bootstrap",
          xlabel="per-sample IQR (% of median)",
          ylabel="median uncertainty (+-% of median)")

    ax = axes[1]
    fails = [d for d in rows if d["cause"] != "passes"]
    counts: dict[str, int] = {}
    for d in fails:
        counts[d["cause"]] = counts.get(d["cause"], 0) + 1
    order = sorted(counts, key=lambda c: counts[c])
    # Blue for the causes a shorter window would help, orange for the ones no
    # window length touches -- the distinction the whole exercise was about.
    colours = [S1 if c.startswith("drift") else (S3 if c.startswith("tail") else S2)
               for c in order]
    y = np.arange(len(order))
    ax.barh(y, [counts[c] for c in order], 0.62, color=colours)
    for yi, c in zip(y, order):
        ax.text(counts[c] + 0.14, yi, str(counts[c]), va="center", fontsize=9.5,
                fontweight="bold", color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([CAUSE_SHORT.get(c, c) for c in order], fontsize=8.2,
                       linespacing=1.25)
    ax.set_xlim(0, max(counts.values()) * 1.25)
    style(ax, f"...and only {sum(n for c, n in counts.items() if c.startswith('drift'))}"
              f" of {len(fails)} failures are drift",
          subtitle="both fixes this repo proposed assumed drift",
          xlabel="measurements")

    figure_header(
        fig,
        "Auditing the quality gate instead of the kernel",
        "IQR describes how the card behaved during the window. It does not describe\n"
        "how well the reported number is pinned, and here the two come apart.",
    )
    save(fig, "dispersion_gate")


def measure_registers() -> dict:
    """Compile both shipped paths and read the register count back."""
    import torch

    import kernels.fused_decode_attn as K
    from kernels.fused_decode_attn import fused_decode_attention
    from quantize import quantize_kv
    from reference import make_random_kv

    q, k, v = make_random_kv(1, 12, 2, 8192, 128, device="cuda", seed=1)
    kq, vq = quantize_kv(k, v, 4, 32)
    out = {}
    seen = set(K._fused_decode_attn_split.device_caches[0][0])
    for name, bcast in (("gather", False), ("broadcast", True)):
        fused_decode_attention(q, kq, vq, meta_bcast=bcast)
        cache = K._fused_decode_attn_split.device_caches[0][0]
        new = [kk for kk in cache if kk not in seen]
        seen.update(new)
        out[name] = max(cache[kk].n_regs for kk in new)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "results" / "benchmark.json"))
    ap.add_argument("--no-gpu", action="store_true",
                    help="skip the two figures that need a live GPU")
    args = ap.parse_args()

    payload = load(Path(args.input))
    print(f"figures -> {PLOTS}")
    plot_broadcast_speedup(payload)
    plot_clock_samples(payload)
    disp = _sweep(ROOT / "results" / "dispersion.json")
    if disp:
        plot_dispersion(disp)
    else:
        print("  (skipped dispersion_gate: no results/dispersion.json --"
              " run analyze_dispersion.py)")
    sweep = _sweep(ROOT / "results" / "gs_sweep.json")
    if sweep:
        plot_gs_saturation(sweep)
        plot_gs128_cliff(sweep)
    else:
        print("  (skipped gs_saturation and gs128_cliff: no results/gs_sweep.json)")
    if args.no_gpu:
        print("  (skipped inner_loop_cost and fold_accuracy: --no-gpu)")
        return
    regs = measure_registers()
    print(f"  registers measured live: {regs}")
    plot_inner_loop_cost(regs)
    plot_fold_accuracy()


if __name__ == "__main__":
    main()
