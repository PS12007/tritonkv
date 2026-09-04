#!/usr/bin/env python
"""Does achieved bandwidth predict protocol sensitivity, or is it a proxy for the method?

    ./.venv/Scripts/python.exe bandwidth_law.py   # reads results/compare_protocols.json

`compare_protocols.py` reports **r = +0.84** between a row's achieved DRAM
bandwidth and how far a change of measurement protocol moves it, over 12 rows,
and this repo has been quoting that as the law that says which rows care. Twelve
points, three methods, four contexts. There is an obvious way for such a
correlation to be true and mean nothing: the three methods pull *very* different
bandwidth (11 / 88 / 214 GB/s on average) and are three different kernels, so a
pooled correlation can be entirely a **method** effect wearing bandwidth's
clothes. "The fp16 control moves more than SDPA" would then be the whole content,
and "bandwidth" only the label on it.

The test that separates them is to hold the method fixed and vary only the
context, which varies bandwidth by 3-4x inside a single kernel. If bandwidth is
doing real work the correlation survives there too; if it was a method label, it
does not.

This script runs that test, and two others the same data supports:

* **Leave-one-out.** A 12-point correlation can rest on one point. Every row is
  dropped in turn and `r` recomputed.
* **Who actually misfits.** `next_steps.md` has carried "`fused_triton_4b@16k`
  fits neither story" as an open item. Residuals against the fitted line say
  which rows really are the misfits, which turns out not to be that one.
* **Is the memory clock really constant across protocols?** The protocol finding
  was reported with "SM clock and memory clock are identical across all three
  protocols", which made the timing shift look like a channel the telemetry could
  not see. That claim is checked here per row rather than assumed.

Nothing here re-measures anything: it reads the JSON `compare_protocols.py`
already wrote.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

# A method whose achieved bandwidth barely varies across contexts cannot test a
# bandwidth law -- there is nothing to correlate against. `fp16_sdpa` sits at
# 11-12 GB/s at every context because it never splits the history, so it is
# reported and then excluded from the within-method test, with that said out loud
# rather than quietly.
MIN_GB_S_RANGE = 2.0

# Two memory P-states differing by less than this are the same state read through
# a sampler; the observed steps on this part are 350-1100 MHz.
MEM_SAME_MHZ = 50.0


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def leave_one_out(rows: list[dict]) -> dict:
    """Recompute r with each row dropped in turn."""
    gb = [r["gb_s"] for r in rows]
    sh = [abs(r["shift"]) for r in rows]
    full = pearson(gb, sh)
    out = []
    for i, r in enumerate(rows):
        out.append({
            "dropped": f"{r['method']}@{r['ctx']}",
            "r": pearson(gb[:i] + gb[i + 1:], sh[:i] + sh[i + 1:]),
        })
    worst = min(out, key=lambda d: d["r"]) if out else None
    return {"r": full, "per_row": out, "worst": worst,
            "min_r": worst["r"] if worst else float("nan")}


def decompose(rows: list[dict]) -> dict:
    """Split the pooled correlation into a within-method and a between-method part."""
    methods = sorted({r["method"] for r in rows})
    within = []
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        gb = [r["gb_s"] for r in sel]
        rng = max(gb) - min(gb) if gb else 0.0
        within.append({
            "method": m,
            "n": len(sel),
            "gb_s_min": min(gb) if gb else None,
            "gb_s_max": max(gb) if gb else None,
            "gb_s_range": rng,
            "r": pearson(gb, [abs(r["shift"]) for r in sel]),
            "testable": rng >= MIN_GB_S_RANGE,
        })
    between = []
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        between.append({
            "method": m,
            "mean_gb_s": statistics.fmean(r["gb_s"] for r in sel),
            "mean_abs_shift": statistics.fmean(abs(r["shift"]) for r in sel),
        })
    between.sort(key=lambda d: d["mean_gb_s"])
    monotone = all(a["mean_abs_shift"] < b["mean_abs_shift"]
                   for a, b in zip(between, between[1:]))
    return {"within": within, "between": between, "between_monotone": monotone}


# Everything here is derived from `gb_s` and `base_ms`, which are both already in
# `compare_protocols.json` -- bytes moved is bandwidth times time. Keeping the
# script single-input means it cannot silently disagree with the file it is
# auditing about what a row's footprint was.
def _predictors(rows: list[dict]) -> dict:
    gb = np.array([r["gb_s"] for r in rows], dtype=float)
    out = {
        "achieved GB/s": gb,
        "GB/s squared": gb ** 2,
        "log GB/s": np.log(gb),
    }
    # `base_ms` is what makes the time-derived rivals possible. A report written
    # before it was recorded still gets the bandwidth-shape comparison rather
    # than an exception -- the check degrades, it does not fail.
    if all("base_ms" in r for r in rows):
        ms = np.array([r["base_ms"] for r in rows], dtype=float)
        by = gb * ms  # proportional to bytes moved; the constant does not affect r
        out.update({
            "bytes moved": by,
            "log bytes moved": np.log(by),
            "time": ms,
            "log time": np.log(ms),
        })
    return out


def rival_predictors(rows: list[dict]) -> list[dict]:
    """Does anything else predict protocol sensitivity better than bandwidth?

    Bandwidth was the *hypothesis*, chosen before this comparison existed, so
    this is a robustness check and not a selection procedure -- which matters,
    because picking the best of seven predictors on twelve points is exactly how
    a spurious one gets chosen. A rival that wins under one protocol and loses
    under the other two is what a chance win looks like, and is reported that way
    rather than adopted.
    """
    P = _predictors(rows)
    y = np.array([abs(r["shift"]) for r in rows], dtype=float) * 100.0
    out = []
    for name, x in P.items():
        if x.std() == 0:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        out.append({"predictor": name,
                    "abs_r": abs(pearson(x, y)),
                    "residual_sd_pp": float(resid.std())})
    out.sort(key=lambda d: d["residual_sd_pp"])
    return out


def residuals(rows: list[dict]) -> list[dict]:
    """Per-row residual against the fitted |shift| ~ bandwidth line."""
    gb = np.array([r["gb_s"] for r in rows], dtype=float)
    sh = np.array([abs(r["shift"]) for r in rows], dtype=float) * 100.0
    slope, intercept = np.polyfit(gb, sh, 1)
    pred = slope * gb + intercept
    return [{"method": r["method"], "ctx": r["ctx"], "gb_s": r["gb_s"],
             "observed_pct": float(s), "predicted_pct": float(p),
             "residual_pp": float(s - p)}
            for r, s, p in zip(rows, sh, pred)]


def mem_clock_constancy(telemetry: list[dict], protocols: list[str]) -> dict:
    """Is the memory clock really the same under every protocol, per row?"""
    rows = []
    for t in telemetry:
        pg = ((t.get("telemetry") or {}).get("mem MHz") or {}).get("per_group") or {}
        vals = [pg[p] for p in protocols if p in pg]
        if len(vals) < 2:
            continue
        spread = max(vals) - min(vals)
        rows.append({
            "method": t["method"], "ctx": t["ctx"], "regime": t["regime"],
            "mem_mhz": {p: pg[p] for p in protocols if p in pg},
            "spread_mhz": spread,
            "constant": spread < MEM_SAME_MHZ,
        })
    varying = [r for r in rows if not r["constant"]]
    return {
        "n_rows": len(rows),
        "n_varying": len(varying),
        "rows": rows,
        "worst": max(rows, key=lambda r: r["spread_mhz"]) if rows else None,
    }


def build(payload: dict) -> dict:
    bw = payload["bandwidth_sensitivity"]
    protocols = list(bw)
    per_protocol = {}
    for proto, block in bw.items():
        rows = block["rows"]
        per_protocol[proto] = {
            "n": len(rows),
            "loo": leave_one_out(rows),
            "decomposition": decompose(rows),
            "residuals": residuals(rows),
            "rivals": rival_predictors(rows),
        }

    # The sign test: every (method, protocol) pair whose bandwidth actually
    # varies is one independent look at "does bandwidth predict, within a
    # kernel". Individually each has n=4 and proves nothing; the question is
    # whether they agree.
    looks = []
    for proto, block in per_protocol.items():
        for w in block["decomposition"]["within"]:
            if w["testable"] and not np.isnan(w["r"]):
                looks.append({"protocol": proto, "method": w["method"],
                              "r": w["r"], "n": w["n"]})
    n_pos = sum(1 for L in looks if L["r"] > 0)
    # one-sided sign test against "the sign is a coin flip"
    p_sign = 0.5 ** len(looks) if n_pos == len(looks) and looks else None

    # Which rows misfit, averaged over protocols
    mean_abs: dict[tuple, list] = {}
    for block in per_protocol.values():
        for r in block["residuals"]:
            mean_abs.setdefault((r["method"], r["ctx"]), []).append(abs(r["residual_pp"]))
    misfit = sorted(
        ({"method": k[0], "ctx": k[1],
          "mean_abs_residual_pp": statistics.fmean(v)} for k, v in mean_abs.items()),
        key=lambda d: -d["mean_abs_residual_pp"])
    by_method: dict[str, list] = {}
    for d in misfit:
        by_method.setdefault(d["method"], []).append(d["mean_abs_residual_pp"])

    # A rival only counts as beating bandwidth if it does so under every
    # protocol. One protocol out of three is a coin landing heads.
    rival_wins: dict[str, int] = {}
    for block in per_protocol.values():
        best = block["rivals"][0]["predictor"]
        if best != "achieved GB/s":
            rival_wins[best] = rival_wins.get(best, 0) + 1
    n_protocols = len(per_protocol)

    return {
        "protocols": protocols,
        "rival_wins": rival_wins,
        "rival_beats_bandwidth_everywhere": [
            k for k, v in rival_wins.items() if v == n_protocols],
        "per_protocol": per_protocol,
        "within_method_looks": looks,
        "n_looks": len(looks),
        "n_positive": n_pos,
        "sign_test_p": p_sign,
        "misfit_rows": misfit,
        "misfit_by_method": {m: statistics.fmean(v) for m, v in by_method.items()},
        "mem_clock": mem_clock_constancy(payload.get("telemetry") or [], protocols),
    }


def render(rep: dict) -> str:
    L = ["# Is the bandwidth law a law, or a method label?", ""]
    L.append("`compare_protocols.py` correlates a row's achieved DRAM bandwidth "
             "against how far a change of protocol moves it, pooled over 12 rows. "
             "Three methods pulling 11 / 88 / 214 GB/s on average is exactly the "
             "shape that lets a **method** effect masquerade as a bandwidth one, "
             "so this file holds the method fixed and varies only the context.")
    L += ["", "## Within a single kernel", ""]
    L += ["| protocol | method | GB/s range | n | r |", "|---|---|---|---|---|"]
    for proto, block in rep["per_protocol"].items():
        for w in block["decomposition"]["within"]:
            note = "" if w["testable"] else "  *(no range -- not testable)*"
            L.append(f"| `{proto}` | `{w['method']}` | "
                     f"{w['gb_s_min']:.0f}-{w['gb_s_max']:.0f} | {w['n']} | "
                     f"{w['r']:+.3f}{note} |")
    L.append("")
    if rep["n_looks"]:
        L.append(f"**{rep['n_positive']} of {rep['n_looks']}** (method x protocol) "
                 f"pairs with any bandwidth range to correlate against come out "
                 f"positive."
                 + (f" Under a null that the sign is a coin flip that is "
                    f"p = {rep['sign_test_p']:.3f}." if rep["sign_test_p"] else "")
                 + " Each is only n=4 and proves nothing alone; the evidence is "
                   "that they agree.")
    L += ["", "## Between methods", ""]
    for proto, block in rep["per_protocol"].items():
        dec = block["decomposition"]
        chain = " -> ".join(f"{d['mean_gb_s']:.0f} GB/s: {d['mean_abs_shift'] * 100:.2f}%"
                            for d in dec["between"])
        L.append(f"- `{proto}`: {chain}"
                 + ("  (monotone)" if dec["between_monotone"] else "  (**not** monotone)"))
    L += ["", "## Leave-one-out", ""]
    L += ["| protocol | r | lowest r with one row dropped | that row |",
          "|---|---|---|---|"]
    for proto, block in rep["per_protocol"].items():
        loo = block["loo"]
        L.append(f"| `{proto}` | {loo['r']:+.3f} | {loo['min_r']:+.3f} | "
                 f"`{loo['worst']['dropped']}` |")
    L += ["", "## Which rows actually misfit", "",
          "Mean |residual| against the fitted line, averaged over protocols.", "",
          "| row | mean abs residual (pp) |", "|---|---|"]
    for d in rep["misfit_rows"]:
        L.append(f"| `{d['method']}@{d['ctx']}` | {d['mean_abs_residual_pp']:.2f} |")
    L.append("")
    L.append("By method: " + ", ".join(
        f"`{m}` {v:.2f} pp" for m, v in sorted(rep["misfit_by_method"].items(),
                                               key=lambda kv: -kv[1])) + ".")
    L += ["", "## Does anything predict better than bandwidth?", "",
          "Residual spread around a straight-line fit, lower is better. Bandwidth "
          "was the hypothesis, fixed before this table existed, so this is a "
          "robustness check rather than a search -- best-of-seven on twelve points "
          "is how a spurious predictor gets picked.", ""]
    heads = list(rep["per_protocol"])
    L.append("| predictor | " + " | ".join(f"`{h}`" for h in heads) + " |")
    L.append("|---" * (len(heads) + 1) + "|")
    order = [d["predictor"] for d in
             next(iter(rep["per_protocol"].values()))["rivals"]]
    for name in order:
        cells = []
        for h in heads:
            hit = next((d for d in rep["per_protocol"][h]["rivals"]
                        if d["predictor"] == name), None)
            cells.append(f"{hit['residual_sd_pp']:.2f}" if hit else "--")
        mark = "**" if name == "achieved GB/s" else ""
        L.append(f"| {mark}{name}{mark} | " + " | ".join(cells) + " |")
    L.append("")
    beats = rep["rival_beats_bandwidth_everywhere"]
    if beats:
        L.append(f"**{', '.join(beats)} beats achieved bandwidth under every "
                 f"protocol.** That is not a chance win and the law should be "
                 f"restated in those terms.")
    elif rep["rival_wins"]:
        L.append("Rivals that win under *some* protocol but not all: "
                 + ", ".join(f"`{k}` ({v}/{len(heads)})"
                             for k, v in rep["rival_wins"].items())
                 + ". Reported, not adopted.")
    else:
        L.append("Achieved bandwidth has the smallest residual spread under every "
                 "protocol.")

    mc = rep["mem_clock"]
    L += ["", "## Is the memory clock constant across protocols?", ""]
    L.append(f"**{mc['n_varying']} of {mc['n_rows']}** measurement rows have a "
             f"memory clock that differs by {MEM_SAME_MHZ:.0f} MHz or more between "
             f"protocols.")
    if mc["worst"]:
        w = mc["worst"]
        L.append("")
        L.append(f"Worst: `{w['method']}@{w['ctx']}` ({w['regime']}), spread "
                 f"**{w['spread_mhz']:.0f} MHz** -- "
                 + ", ".join(f"{p} {v:.0f}" for p, v in w["mem_mhz"].items()) + ".")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(RESULTS_DIR / "compare_protocols.json"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "bandwidth_law.json"))
    ap.add_argument("--md", default=str(RESULTS_DIR / "bandwidth_law.md"))
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"no protocol comparison at {path} -- "
                         f"run `compare_protocols.py` first")
    rep = build(json.loads(path.read_text(encoding="utf-8")))

    for proto, block in rep["per_protocol"].items():
        loo = block["loo"]
        print(f"{proto:<10} r = {loo['r']:+.3f}   worst leave-one-out "
              f"{loo['min_r']:+.3f} (drop {loo['worst']['dropped']})")
    print()
    print("within a single kernel:")
    for L in rep["within_method_looks"]:
        print(f"   {L['protocol']:<10} {L['method']:<22} r = {L['r']:+.3f}  (n={L['n']})")
    print(f"   -> {rep['n_positive']}/{rep['n_looks']} positive"
          + (f", sign-test p = {rep['sign_test_p']:.3f}" if rep["sign_test_p"] else ""))
    print()
    print("worst-fitting rows (mean |residual| over protocols):")
    for d in rep["misfit_rows"][:4]:
        print(f"   {d['method']:<22}@{d['ctx']:<6} {d['mean_abs_residual_pp']:.2f} pp")
    print("   by method: " + ", ".join(
        f"{m} {v:.2f}" for m, v in sorted(rep["misfit_by_method"].items(),
                                          key=lambda kv: -kv[1])))
    print()
    print("rival predictors (residual sd, pp -- lower is better):")
    for proto, block in rep["per_protocol"].items():
        top = block["rivals"][:3]
        print(f"   {proto:<10} " + "  ".join(
            f"{d['predictor']} {d['residual_sd_pp']:.2f}" for d in top))
    beats = rep["rival_beats_bandwidth_everywhere"]
    print("   -> " + (f"{', '.join(beats)} beats bandwidth under every protocol"
                      if beats else
                      "nothing beats achieved bandwidth under every protocol"))

    mc = rep["mem_clock"]
    print()
    print(f"memory clock varies across protocols on {mc['n_varying']}/{mc['n_rows']} rows"
          + (f"; worst {mc['worst']['method']}@{mc['worst']['ctx']} "
             f"({mc['worst']['regime']}) {mc['worst']['spread_mhz']:.0f} MHz"
             if mc["worst"] else ""))

    Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    Path(args.md).write_text(render(rep), encoding="utf-8")
    print(f"\nwrote {args.out} and {args.md}")


if __name__ == "__main__":
    main()
