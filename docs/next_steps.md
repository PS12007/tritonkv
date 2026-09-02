# Where to pick this up

Rewritten 2026-09-02 after the three-run between-run measurement.
Read `progress_log.md` first for *why* things are the way they are; this file is
only *what is left*.

## State: what works right now

Everything below is verified on this machine, not assumed.

- Environment is healthy. `.venv` has `torch 2.12.0+cu130`, `triton-windows
  3.8.0.post28`, matplotlib, pytest, transformers. CUDA is live on an
  **RTX 5060 Laptop, sm_120, 26 SMs, 8 GB, 33.6 MB L2, 80 W**.
  Run things as `./.venv/Scripts/python.exe ...` — no activation needed.
- `python -m pytest test_correctness.py -q` → **106 passed in ~89 s**.
- `python benchmark.py --samples 50` → ~13 min, clock-monitored, writes
  `results/benchmark.json`. Three back-to-back runs on 2026-09-02 gave
  **42 / 41 / 39 of 48 rows quotable** (was 25/48 before the ramp fix), and
  every rejection is dispersion — there are no clock rejections left. Run it
  with `python -u` or the log stays buffered.
- `python between_run.py results/runs/run*.json` → seconds, writes
  `results/between_run.{md,json}`. `audit_claims.py` picks that file up
  automatically and reports the run-to-run interval next to every CI.
  `python -m pytest test_between_run.py -q` → 16 CPU-only tests, ~2 s.
- The canonical `results/benchmark.json` is **run 3 of 3** — "the last one", a
  selection rule that cannot be gamed after the fact, and incidentally the least
  flattering of the three. The other two are in `results/runs/`, and the older
  interleaved (`--passes 2`) run is kept as `results/benchmark_interleaved.json`.
- `python make_plots.py` → 8 figures in `docs/plots/`;
  `python make_session_plots.py` → 6 more (2 need `results/gs_sweep.json`).
- `python sweep_group_size.py --contexts 512 2048 8192` → ~3.5 min, 24 cells,
  writes `results/gs_sweep.json`. `python probe_gs128.py` → static PTX counts,
  no timing, seconds.
- README, `docs/key_numbers.md`, `docs/thread_outline.md`.

**The headline finding, in its current form:** the fused kernel's win over
PyTorch SDPA is almost entirely the flash-decoding split (10.5–68× on quotable
rows), not the quantization. Quantization *costs* 1.23–1.37× when the KV cache
is L2-resident and pays 1.18–1.48× when the working set exceeds L2 and the
context is 2k or more. Ranges below are **across three independent runs**, not
within-run CIs.

**ctx=8192 is the only context where every input of the attribution chain passes
the gate in all three runs**: 0.800–0.813× L2-resident, 1.469–1.478×
DRAM-resident. ctx=512 (0.717–0.796× L2 / 0.905–0.928× DRAM — quantization loses
in *both* regimes at short context) and ctx=2048 (0.856–0.910× / 1.188–1.197×,
the sign flip itself) clear the gate in one run and not the others, so they are
reported with that qualifier rather than starred as though the star were a
property of the kernel.

## Do this first

Nothing is blocking. `results/audit.{md,json}` are current (regenerated
2026-09-02 against run 3 with the between-run data loaded, **69 claims: 26 TRUE /
20 CONDITIONAL / 11 MISLEADING / 12 FALSE**), and the audit takes ~20 s, so
re-run it after any benchmark run without thinking about it:

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

3. **Characterise the tail of the run-to-run distribution, not just its body.**
   Three runs settled the question this file used to lead with, and the answer
   was reassuring: no verdict moved, and the honest interval is ~2.4× the
   printed CI rather than an order of magnitude. But the 1.27× that motivated
   the whole exercise did **not** reappear — `fused_triton_4b@8192` sat at
   11001 MHz in all four subsequent runs, and the three new runs agree to 0.6%
   at that cell. So the run-to-run distribution has a body of about ±1% and a
   tail that moves a headline ratio by 15%, and three runs measure the body
   while saying nothing about the tail.

   What would actually settle it: five to ten runs, unattended, recording only
   the per-row mean memory clock and the four headline ratios (a `--clocks-only`
   fast path, since the full run costs 13 min and most of it is not needed).
   The question is not "what is the mean" — it is "how often does a row drop a
   P-state", and that is a rate, so it needs a denominator. Worth doing before
   any number here is quoted with a hard interval; not worth doing to move any
   verdict, since none moved.

   Also unresolved and now measurable: `between_run.py` reports the correlation
   between a row's between-run movement in time and in mean memory clock as
   **r = +0.71** over 48 DRAM-resident rows. That confirms the P-state story
   but leaves ~half the variance elsewhere. Nothing identifies what.

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
- The bandwidth-aware clock ramp: a DRAM-sized copy alongside the GEMM, a
  stopping rule that waits for the memory clock to stop changing (bounded, and
  reported rather than enforced), and a ceiling learned only from samples taken
  while the GPU is busy. The old ramp drove the SM clock and asked memory for
  nothing, which is why the P-state moved inside the measurement.

## Things that were tried and rejected — do not redo without new evidence

- **Zero-point folding for speed.** Implemented and kept (`fold_zp=True`) for
  its *accuracy*, but it is not faster at any context in either regime. An early
  ad-hoc A/B suggested otherwise; the harness disagreed and the harness wins.
- **Narrow-load + register broadcast for the packed codes.** Bitwise identical,
  measurably slower at long context (registers 128 → 223). Reverted.
- **Gating on memory-clock stability.** The obvious companion to the ramp fix,
  and wrong: it rejects every DRAM-resident row, including rows with 0.4–2%
  timing IQR. The ramp fix that cut L2-resident median |trend| from 2.4% to 0.2%
  also made the memory-clock spread go from mostly 0% to ~19%, so the gate would
  have discarded precisely the measurements the fix improved. Oscillation inside
  a window is noise both methods sit in equally; the bias is *between* rows, and
  that is where the check now lives.
- **Interleaving the methods** (`--passes 2`). Implemented, measured over a full
  run, and it changed nothing: 39/48 quotable either way, mismatches over 3%
  went 12/20 → 14/20, and the run took 1520 s instead of 861 s. The premise was
  wrong — a row's two passes agree to 0.14%, so there is no drift to average, and
  a method's mean memory clock reproduces across independent runs to 86 MHz out
  of ~10,500. The clock is set by the workload, not by the schedule. The flag
  remains, defaulting to 1.
- **Widening `MAX_IQR_FRAC`.** Not tried, and deliberately not tried. See
  `method.dispersion_gate` in the audit for what the gate does and does not
  measure — the answer to a gate that rejects too much is to fix the
  measurement, which is what the ramp work did.

## Session hygiene note

A Claude Code session in this directory once survived its terminal being closed
and kept writing files while a second session worked in the same tree. If files
appear that nobody in the current session wrote, check for orphaned
`claude.exe` processes before assuming the working tree is yours alone.
