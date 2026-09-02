# Where to pick this up

Rewritten 2026-09-01 (late) after the group-size sweep and the audit repairs.
Read `progress_log.md` first for *why* things are the way they are; this file is
only *what is left*.

## State: what works right now

Everything below is verified on this machine, not assumed.

- Environment is healthy. `.venv` has `torch 2.12.0+cu130`, `triton-windows
  3.8.0.post28`, matplotlib, pytest, transformers. CUDA is live on an
  **RTX 5060 Laptop, sm_120, 26 SMs, 8 GB, 33.6 MB L2, 80 W**.
  Run things as `./.venv/Scripts/python.exe ...` — no activation needed.
- `python -m pytest test_correctness.py -q` → **106 passed in ~89 s**.
- `python benchmark.py --samples 50` → ~7.5 min, clock-monitored, writes
  `results/benchmark.json`. Last run: **25/48 rows quotable**, and every
  rejection is dispersion — there are no clock rejections left.
- `python make_plots.py` → 8 figures in `docs/plots/`;
  `python make_session_plots.py` → 6 more (2 need `results/gs_sweep.json`).
- `python sweep_group_size.py --contexts 512 2048 8192` → ~3.5 min, 24 cells,
  writes `results/gs_sweep.json`. `python probe_gs128.py` → static PTX counts,
  no timing, seconds.
- README, `docs/key_numbers.md`, `docs/thread_outline.md`.

**The headline finding, in its current form:** the fused kernel's win over
PyTorch SDPA is almost entirely the flash-decoding split (10–26× on quotable
rows), not the quantization. Quantization *costs* ~1.1–1.35× when the KV cache
is L2-resident and pays 1.22–1.47× when the working set exceeds L2. At ctx=2048
the sign flip is now fully clock-verified with every input passing the gate:
**0.90× L2-resident, 1.22× DRAM-resident.**

## Do this first

Nothing is blocking. `results/audit.{md,json}` are current (regenerated
2026-09-01 22:30, **68 claims: 21 TRUE / 25 CONDITIONAL / 10 MISLEADING /
12 FALSE**), and the audit now takes ~20 s rather than ~13 min, so re-run it
after any benchmark run without thinking about it:

```
./.venv/Scripts/python.exe audit_claims.py
./.venv/Scripts/python.exe make_plots.py            # slow, minutes per figure
./.venv/Scripts/python.exe make_session_plots.py    # fast, needs gs_sweep.json
```

## Open work, in order

1. **Decide what to do about the dispersion rejects, now that they are
   diagnosed.** `analyze_dispersion.py` decomposed all 96 measurements: only
   8 of 25 failures carry a significant trend (so shorter windows help 8 rows,
   not 23), 13 have neither trend nor tail, and the failing rows pin their
   medians to +-0.69% anyway. The two fixes this file used to propose are
   therefore mostly answering a question the data does not ask. What is left is
   a **presentation** decision, not a measurement one, and it should be made
   deliberately:

   - report the median's CI alongside the IQR in `benchmark.py`, so a starred
     row carries the number that actually bears on the conclusion (needs a
     re-run to take effect, and does not change the gate); and/or
   - add a *second* tier -- "gate-failed but median pinned to <1%" -- so the
     attribution tables can use those rows with an explicit qualifier instead
     of dropping them.

   Do **not** widen `MAX_IQR_FRAC`. The four rows whose failure is a tail would
   be fixed by a trimmed statistic, which is a defensible change on its own
   merits; the seven "drift + floor" rows would genuinely benefit from a shorter
   window, and ctx=512 is where four of them are.

2. **Where the metadata-load curve actually bends.** `sweep_group_size.py` ran
   the sweep on both paths and the prediction ("broadcast sloped, gather flat")
   was refuted: broadcast is flat too, 1.07-1.17x across an 8x range of load
   counts. The honest shape is *saturation* -- 16x fewer loads bought 1.29x, a
   further 8x bought 1.07x -- so the knee is somewhere between 256 and 4096
   loads per tile and nothing measures it yet. Filling that gap needs a
   measurement-only knob that varies the metadata load count while holding
   everything else fixed (a redundancy constexpr on the broadcast path, R=1
   being the shipped code). Worth doing only if the knee matters to a reader;
   the current claim ("the win crossed the knee") does not depend on where
   exactly it is.

3. **The attribution is only fully quotable at ctx ≤ 2048.** The 8192 and 16384
   rows still lose either `fp16_sdpa` or `triton_fp16_control` to dispersion in
   any given run, so the long-context sign flip is reported but leans on rows
   that did not all pass simultaneously. Fixing (1) fixes this.

4. **2-bit still deserves a decision, not a table row.** Numerically unusable
   (rel L2 ≈ 0.7) under per-token grouping along `head_dim`. KIVI's result is
   that keys want per-channel grouping. Either implement per-channel keys and
   re-measure, or state plainly that 2-bit is out of scope and stop benchmarking
   it. Right now it is carried at full cost in every table without being a
   candidate for use. (Note it is also where `fold_zp` is *worst*, 0.66×.)

5. **Cross-check on a non-laptop GPU if one becomes available.** Everything here
   is one card with a 9× clock range and a 33.6 MB L2. The L2-residency
   conditional is stated in terms of L2 size precisely because that is the axis
   expected to move; nothing tests that yet. The metadata-broadcast win should
   also shrink on a part that is less issue-bound.

## Things that are done and should not be redone

- Clock monitoring, the quotability gate, the inner ramp, and the minimum
  clock-sample requirement (`benchmark.py`).
- The fp16 control kernel, the `fused_gather_meta_*` control, and the
  `fused_fold_zp_*` row.
- The metadata broadcast itself, and the bitwise-identity tests that pin it.
- Median/IQR reporting, raw hot-regime samples in the JSON.
- The group-size sweep on both paths (`sweep_group_size.py`) and the static PTX
  probe behind it (`probe_gs128.py`). The gs=128 gather cliff is **explained**:
  at `group_size == head_dim` the gather index folds to all-zeros and Triton
  converts the tile through shared memory (`st.shared` 30 -> 142), inside the
  loop, which is why it worsens with context. It is a fact about the control
  row; the shipped broadcast path is the *fastest* cell at gs=128.
- The audit's speed: `bootstrap_ratio_ci` is vectorized (13 min -> 20 s), with
  the stdlib version kept as the definition and `--check-bootstrap` comparing
  them on real rows (worst disagreement 2.3% of a CI width).
- The audit's **statistics**: the resample is circular-block, not i.i.d., because
  the series are serially correlated (lag-1 up to 0.72) and i.i.d. intervals were
  up to 1.95x too narrow. `_verdict` also requires the deciding CI endpoint to
  clear its bar by 10% of the interval's own width, so a claim cannot be settled
  by a rounding difference. Both together changed **no** verdict.
- The dispersion decomposition itself (`analyze_dispersion.py`) and the
  `method.dispersion_gate` claim that reports it against this project's own gate.

## Things that were tried and rejected — do not redo without new evidence

- **Zero-point folding for speed.** Implemented and kept (`fold_zp=True`) for
  its *accuracy*, but it is not faster at any context in either regime. An early
  ad-hoc A/B suggested otherwise; the harness disagreed and the harness wins.
- **Narrow-load + register broadcast for the packed codes.** Bitwise identical,
  measurably slower at long context (registers 128 → 223). Reverted.

## Session hygiene note

A Claude Code session in this directory once survived its terminal being closed
and kept writing files while a second session worked in the same tree. If files
appear that nobody in the current session wrote, check for orphaned
`claude.exe` processes before assuming the working tree is yours alone.
