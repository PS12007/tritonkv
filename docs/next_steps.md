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
- `python benchmark.py --samples 50` → ~14 min, clock-monitored, writes
  `results/benchmark.json`. Last run: **39/48 rows quotable** (was 25/48
  before the ramp fix), and every rejection is dispersion — there are no
  clock rejections left. Run it with `python -u` or the log stays buffered.
- `python make_plots.py` → 8 figures in `docs/plots/`;
  `python make_session_plots.py` → 6 more (2 need `results/gs_sweep.json`).
- `python sweep_group_size.py --contexts 512 2048 8192` → ~3.5 min, 24 cells,
  writes `results/gs_sweep.json`. `python probe_gs128.py` → static PTX counts,
  no timing, seconds.
- README, `docs/key_numbers.md`, `docs/thread_outline.md`.

**The headline finding, in its current form:** the fused kernel's win over
PyTorch SDPA is almost entirely the flash-decoding split (10–68× on quotable
rows), not the quantization. Quantization *costs* 1.11–1.58× when the KV cache
is L2-resident and pays 1.17–1.48× when the working set exceeds L2 and the
context is 2k or more. Two contexts now have every input passing the gate
simultaneously: **ctx=512 → 0.72× L2 / 0.91× DRAM** (quantization loses in
*both* regimes at short context) and **ctx=2048 → 0.90× L2 / 1.17× DRAM**
(the sign flip itself).

## Do this first

Nothing is blocking. `results/audit.{md,json}` are current (regenerated
2026-09-01 23:35 against the post-ramp-fix benchmark, **68 claims: 28 TRUE /
18 CONDITIONAL / 10 MISLEADING / 12 FALSE**), and the audit now takes ~20 s rather than ~13 min, so re-run it
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

3. **Re-check the long-context attribution against the new run.** The 8192 and
   16384 rows used to lose `triton_fp16_control` to dispersion, and the cause is
   now known and fixed: the ramp never touched the memory system, so the memory
   P-state stepped up *during* those bandwidth-bound measurements (−19.3% and
   −22.6% across their own windows). With the ramp warming DRAM too, those rows
   should hold. Confirm it against the run that produced the current
   `results/benchmark.json`, and re-check whether the long-context sign flip is
   now quotable with every input passing simultaneously — it previously leaned on
   rows that did not.

   The **between-row** memory-clock mismatch is now understood and is not
   fixable on this machine: each kernel induces its own memory P-state on a
   power-shared part (reproducible to 86 MHz across independent runs), and
   interleaving was measured and changed nothing. What is left is to decide how
   loudly to say it. The systematic direction is favourable — the fp16 control
   has the highest mean memory clock of any method, which *understates* the
   quantization benefit — so the honest move is to state the bound rather than
   to keep flagging each instance.

   Note also that the DRAM-resident quantization ratio at ctx=8192 moved
   1.27× → 1.47× between two runs, because one row's memory P-state differed
   between them, while the bootstrap CI on either is ±0.01. **The CI describes
   sampling noise within a run and nothing about which P-state that run landed
   in.** Reporting a run-to-run interval alongside it, from two or three full
   runs, would be the honest fix and is the single most valuable remaining
   measurement in this file.

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
