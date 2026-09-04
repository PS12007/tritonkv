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
- `python compare_protocols.py --label full=... --label subset=... --label
  preloaded=... --label fullpre=...` → ~4 min (bootstrap over 4 groups),
  writes `results/compare_protocols.{md,json}`: per-protocol ranges, the
  disjointness flag, the bandwidth correlation per protocol, and the 2x2
  decomposition. Reads each run's protocol from its own recorded `args`.
- `python clock_excursions.py --label full=... --label subset=...` → seconds,
  writes `results/clock_excursions.{md,json}`: the per-run-group rate of memory
  P-state excursions and whether the gate rejected each one.
- `python thermal_check.py --label name=path[,path...]` (one flag per protocol)
  → seconds, writes `results/thermal_check.{md,json}`: the within-cell slope of
  memory clock against temperature, and how many degrees a P-state step would
  need. With `--pair warm=cold` it also tests the warm-up arm — whether a
  preload's clock advantage decays across a run — and reports how many runs
  that test would need. Re-measures nothing.
- `python clock_ramp.py --idle 90 --load 180` → ~4.5 min **on the GPU**, writes
  `results/clock_ramp.{md,json}` including the raw 10 Hz series.
  `--from-json results/clock_ramp.json` re-runs the analysis without the GPU.
- `python bandwidth_law.py` → seconds, writes `results/bandwidth_law.{md,json}`
  (figure: `docs/plots/bandwidth_law.png`, from `make_session_plots.py`):
  the within-method decomposition of the bandwidth law, leave-one-out, per-row
  residuals, and the per-row memory-clock constancy check. Reads
  `results/compare_protocols.json`; re-measures nothing.
- `python dispersion_tier.py` → seconds, writes
  `results/dispersion_tier.{md,json}`: the three-tier verdict per row, the
  calibration bar it was judged against, and each promoted row's
  `min_effect_frac`. Post-hoc from the raw samples, so it runs on any results
  JSON ever recorded and never touches `benchmark.py`. Run 3: **39 quotable /
  7 pinned / 2 rejected**.
- `python -m pytest test_between_run.py -q` → **139** CPU-only tests, ~18 s,
  covering `between_run.py`, `clock_excursions.py`, `compare_protocols.py` and
  `dispersion_tier.py` (including the 2x2 arithmetic, the design reader, and the
  tier's calibration bar and per-claim admissibility).
- `python benchmark.py --methods attribution` → 210 s instead of 775 s, times
  only the three rows the conditional is built from. **Not a substitute for a
  full run** — see open item 3; it is measurably noisier and shifted.
- The canonical `results/benchmark.json` is **run 3 of 3** — "the last one", a
  selection rule that cannot be gamed after the fact, and incidentally the least
  flattering of the three. The other two are in `results/runs/`, and the older
  interleaved (`--passes 2`) run is kept as `results/benchmark_interleaved.json`.
- `python make_plots.py` → 8 figures in `docs/plots/`;
  `python make_session_plots.py` → 7 more (2 need `results/gs_sweep.json`,
  and `protocol_factorial` needs a complete 2x2 in
  `results/compare_protocols.json`).
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
2026-09-02 against run 3 with the between-run and protocol data loaded,
**72 claims: 26 TRUE / 22 CONDITIONAL / 12 MISLEADING / 12 FALSE**), and the
audit takes ~20 s, so
re-run it after any benchmark run without thinking about it:

```
./.venv/Scripts/python.exe audit_claims.py
./.venv/Scripts/python.exe make_plots.py            # slow, minutes per figure
./.venv/Scripts/python.exe make_session_plots.py    # fast, needs gs_sweep.json
```

## Open work, in order

1. **DONE -- the second dispersion tier is implemented.** `dispersion_tier.py`
   adds a third verdict rather than widening `MAX_IQR_FRAC`, which stays at 5%
   and is not to be touched.

   * **tier 1, quotable** -- the gate's own verdict, unchanged. A star means
     what it always meant.
   * **tier 2, pinned** -- clock-verified, failed the IQR gate, but every regime
     pins its median at least as well as *the worst number the gate already
     accepts* (run 3: +-1.700%, from `fused_gather_meta_4b@512`). Usable with an
     explicit qualifier, never a star.
   * **tier 3, rejected** -- everything else.

   The bar is read off the instrument per run, so it is not a new free
   parameter, and across the nine full runs it is stable at 1.43-1.96%. A
   clock-rejected row is never eligible (the gate is not a P-state filter). Each
   tier-2 row carries `min_effect_frac` = 5x its own median uncertainty, and
   `usable_for(rec, effect)` is the per-claim test.

   Run 3: **39 quotable / 7 pinned / 2 rejected** of 48. The two that stay
   rejected are the two genuinely unpinned ones (+-2.33%, +-2.68%) — which is
   the reason the gate was not widened instead.

   Attribution chain complete, over all **nine** full-method runs (three
   protocols):

   | ctx | gate only | with tier 2 |
   |---|---|---|
   | 512 | 5/9 | **9/9** |
   | 2048 | 5/9 | **9/9** |
   | 8192 | 7/9 | 7/9 |
   | 16384 | 2/9 | **7/9** |

   ctx=512 and ctx=2048 — the two that carry the sign flip and that this file
   used to report with a "clears in one run and not the others" qualifier — are
   now complete in every run. ctx=8192 is untouched because its one incomplete
   run fails on a median genuinely pinned to only +-1.91%, not on dispersion.

   Applied post-hoc from the raw per-sample timings already in every results
   JSON, so it covers all nine full runs and the subset runs retroactively and
   `benchmark.py` is untouched. `python dispersion_tier.py` -> seconds, writes
   `results/dispersion_tier.{md,json}`.

   **The audit consumes it** (`--dispersion-tier`, absent is fine). Evidence
   lines now separate `*` (not usable) from `~` (gate-failed, median pinned,
   effect at least 5x that pin), where both used to be `*`. **No verdict moved**;
   70 claims became 71 with `method.dispersion_tier`, the companion to
   `method.dispersion_gate`, and 72 with `method.bandwidth_law`.
   **26 TRUE / 22 CONDITIONAL / 12 MISLEADING / 12 FALSE.**

   README and `key_numbers.md` are updated: both timing tables mark those rows
   `~`, the qualifier is replaced by the six-run table, and `key_numbers.md`
   carries it as **correction 10**. Figure: `docs/plots/dispersion_tier.png`
   (`make_session_plots.py`).

   Note that **promotion is a property of the run**, exactly as quotability is:
   no row is promoted in all nine full runs, and the most any manages is four.
   Report tier-2 coverage over runs, never from a single one.

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

3. **The tail rate: answered, but only for the shipped protocol.**
   `clock_excursions.py` puts a rate on P-state excursions across six runs. Under
   the full protocol: **2 of 72 observations, and 0 of them DRAM-resident** —
   which is the only regime where a memory P-state drop costs meaningful time.
   That is why three full runs agree to 0.6% at the cell that once read 1.27×:
   sustained work holds the memory clock up.

   The cheap denominator this file used to ask for **does not exist**, and the
   reason is a result rather than an obstacle. `benchmark.py --methods
   attribution` is 3.7× faster (210 s vs 775 s) but is a *different and noisier
   experiment*: 2 of 12 ratios land outside the full-run range entirely
   (`quant_cold@8192` reads 1.277–1.445 against 1.469–1.478), and its excursion
   rate is 11.1% against 2.8%. Take three quarters of the preceding work away
   and the excursions come back — `sub3` reproduced the historical 1.277× exactly,
   at 10334 MHz. The flag stays because it is a good excursion *generator* and
   because `between_run.py` refuses to pool a filtered run with a full one.

   What is left here is genuinely optional: five to ten more **full** runs would
   tighten "0 of 72" to a smaller upper bound on the DRAM-resident excursion
   rate. That is 2 hours of wall clock to narrow a bound that already supports
   everything this repo says. Do it before quoting a hard interval externally,
   not before drawing any conclusion here.

   **The channel is power, not clock — corrected 2026-09-02.** This item used
   to say the protocol gap was invisible in the telemetry. It is not; the figure
   being compared was a whole-run power average, which is flat by construction
   because the card sits at its limit most of the time. `compare_protocols.py`
   compares per row and finds **r = −0.57** between a row's power shift and its
   time shift across 24 shared rows: more power, less time, at identical SM and
   memory clocks. `triton_fp16_control@8192` runs 3.9% faster on 3.1% more
   power with the memory clock identical at 11001 MHz. (That row's memory clock
   really is 11001 under all four protocols. It does **not** generalize: 14 of 24
   rows differ across the original three protocols, 17 of 24 across all four —
   see `bandwidth_law.py`.)

   **And the protocol effect is now characterised, 2026-09-02 (late).**
   `--preload 300` was the probe: if integrated load explained the gap, a
   preloaded subset run should converge on the full run. It did the opposite —
   moved further in the same direction — so integrated load is **refuted** as
   the explanation. What predicts the shift instead is **achieved bandwidth**:
   `compare_protocols.py` finds **r = +0.84** between a row's DRAM bandwidth and
   the size of its protocol shift, over 12 rows. `fp16_sdpa` at 11 GB/s moves
   ±0.3%; `triton_fp16_control` at 257 GB/s moves 7.3%. That is predictive, and
   it explains why the quantization ratio is the most exposed number here — it
   divides the highest-bandwidth row by a much lower one.

   **The cost, stated plainly: the shipped protocol reports the most favourable
   `quant_cold@8192` of the three** (1.475 full, 1.422 subset, 1.397 preloaded)
   against a 0.6% between-run spread. The honest range for that cell across
   everything measured is **1.28–1.48**. No verdict changes and the conditional
   survives, but the interval this repo quotes for that cell is too narrow and
   the README now says so.

   **The fourth protocol has been run, and the answer is "recent saturation"
   (2026-09-02, night).** `benchmark.py --samples 50 --preload 300` with the full
   method set, three clean runs, completes the 2x2. At `quant_cold@8192` the
   cells read `subset` 1.4217 / `preloaded` 1.3968 / `full` 1.4755 / `fullpre`
   **1.3944** — 0.2% from the pre-registered H3, 5.5% from H1.

   The simple effects are the finding: method count is worth **+3.8%** with no
   preload and **−0.2%** after one. Saturating the memory system for 300 s does
   everything 800 s of preceding measurement was doing, so run length was never
   the channel — only a proxy for recently-pulled bandwidth. The shipped `full`
   protocol's 1.4755 is what a run reads when the memory subsystem has *not*
   recently been saturated, and it is the only one of the four in that state.

   Reported honestly: **no effect is "resolved"** against this file's own
   yardstick (the largest within-cell range, 13.2%, set by `subset`). The
   pre-registered point prediction was hit to 0.2% and the conservative interval
   test does not clear it; both are true, and the yardstick was fixed before the
   data and has not been moved since. What does separate them is the usual test —
   `fullpre`'s range misses `full`'s entirely. Bandwidth law survives repointing:
   r = +0.70 for `fullpre` vs `full`.

   Still open, and now sharper:

   - **Why the shortest and the longest protocols are the noisy ones — the
     thermal half of the answer is now RULED OUT at these temperatures
     (2026-09-03, `thermal_check.py`).** Every observation across all four
     protocols, expressed as a deviation from its own (method, ctx, regime,
     protocol) cell mean: temperature moves the memory clock by **−30.4 MHz per
     degree C** (r = −0.143, n = 720, 2.0% of the variance). The sign is the one
     the thermal story predicts; the size is the problem. A memory P-state step
     on this part is 350–1100 MHz, so moving one needs **11.5–36 degrees C** —
     and the four protocols span only **3.3 degrees**, worth 99 MHz. That is
     under a third of the smallest step.

     So a temperature sweep is only worth running if it **induces at least
     ~12 degrees C** at fixed protocol, which the protocols themselves do not.
     Caveats in `results/thermal_check.md`: the fit is linear over ±3.9 C so the
     degrees-per-step figure is an extrapolation and a threshold outside that
     window would not show up; observations inside a cell are not independent so
     the reported t overstates significance; and temperature is a window mean, so
     a brief spike could throttle without moving it.

     The *other* arm — too little preceding work and the clock never comes up —
     is **live but cannot be settled this way.** `thermal_check.py --pair` tests
     its prediction directly: a preloaded run should start at a higher memory
     clock and the advantage should decay as the cold-start run earns its own.
     The point estimates lean correctly (`preloaded` over `subset`: **+124 MHz
     early, +43 late**; `fullpre` under `full`: −38 and −31, i.e. a preload that
     only costs) and **neither clears its error bar** (Welch t = +1.46 and −0.19).
     At the scatter these cells show, settling it by this route needs
     **~200 runs per protocol** — tens of hours.

     **DONE, and the arm is dead (2026-09-03, `clock_ramp.py`, pre-registered in
     `05c1be3`, result `b29ff18`).** Idle 90 s, then 180 s of the same
     DRAM-saturating load `benchmark.py` ramps with, sampling at 10 Hz. The
     memory clock goes 405 MHz idle → **11001 MHz sustained, reached and held
     0.4 s after the load starts** — 0.2% of the shortest protocol. Pre-registered
     H1 ≥ 20.5 s / H2 < 20.5 s; predicted "H2, 1–4 s"; observed 0.4 s. A 205 s run
     spends 99.8% of itself with the clock up, so the warm-up arm cannot explain
     it behaving differently from a 775 s run.

     Two by-products: the memory clock **leads the SM clock by 18×** (0.4 s vs
     7.4 s), the opposite of the intuition behind the original ramp bug; and
     there is a real **6.5 s boost transient** at 12001 MHz before the card
     settles to 11001.

     **And the cooling loop does not explain the dispersion either — H-context,
     pre-registered, landed (2026-09-04, `results/tail/reversed1.json`).** One
     run with `--contexts 16384 8192 2048 512`. Failures by context went
     512:2/2048:7/8192:1/16384:0 (normal) → 16384:1/8192:1/2048:1/512:4
     (reversed): they follow the **context**, not the position. The correlation
     between position and IQR **flipped sign**, −0.226 → +0.231. H-position
     wanted ≥4 failures on ctx=16384/8192 and got 2. Short-context rows are noisy
     because they are 3–5 µs long, not because of when they are measured.

     Two by-products. The **L2-conditional survives** a protocol it was never
     measured under (`quant_hot` < 1 everywhere; `quant_cold` 0.904 → 1.158 →
     1.484 → 1.512, same crossing point). And **`quant_cold@16384` read 1.512
     against 1.416–1.469 from three full runs** — on a tier-2 row, so usable, and
     consistent with the bandwidth law: `control@16384` at 305 GB/s is the most
     protocol-sensitive row here and reversal moves it from last to first.
     **WITHDRAWN after three reversed runs (2026-09-04).**
     `compare_protocols.py` over `full` (3) vs `reversed` (3): **1 of 16 ratios
     disjoint**, and it is `speedup_vs_sdpa@16384`, not a quantization ratio.
     `quant_cold@16384` is full 1.416–1.469 vs reversed 1.433–1.508 — +5.0% on
     **overlapping** ranges, so not a shift. The three reversed values were
     1.5125 / 1.4113 / 1.5151.

     What reversing does is make the protocol **worse**: at the headline cell
     `full` spans 0.6% and `reversed` spans **11%** (1.332–1.480), with a 7.8%
     P-state excursion rate against 4.9%. Reversed is a diagnostic, never a
     default. Bandwidth law survives repointing weakly, r = +0.42 against +0.70
     to +0.85 for the other pairs.

     **Both arms of the two-mechanism hypothesis are now dead** — thermal on
     effect size, warm-up on time constant. What survives is the observation
     itself: `subset` → `preloaded` improves spread 13.2% → 0.6% for 300 s of
     preload, and nothing here explains why.

     **And nothing in the telemetry explains it either (2026-09-04).**
     `thermal_check.py` reports the mean within-cell run-to-run SD of every
     monitored variable beside each protocol's ratio spread: `subset` (13.2%) and
     `full` (0.6%) differ 20x in spread while sitting within 6% of each other on
     temperature SD and 10% on memory-clock SD, and `preloaded` is as steady as
     `full` in its ratios with the second-worst SM-clock reproducibility. Nothing
     tracks it.

     Candidate status: the **power governor** is weakened (power pins at 79.8 W
     in 10 s, then flat); the **fan curve is untestable on this part** —
     `nvidia-smi --query-gpu=fan.speed` is `[N/A]` because a laptop GPU does not
     control the chassis fan, so that is unavailable rather than refuted;
     **allocator or driver state** is untouched.

     **Do not run more protocol repetitions.** What would settle this is a
     variable the current sampler does not record, and the obvious one cannot be
     read on this hardware.

   - **AND THE SPREADS THAT MOTIVATED ALL OF THIS WERE MOSTLY REJECTED RUNS
     (2026-09-04).** A protocol's spread is the range of its runs' point
     estimates and nothing checked whether those runs were quotable. At
     `quant_cold@8192`, over runs that survive the gate: `full` 0.6% (3/3),
     `subset` n=1 (**1/3**), `preloaded` 0.6% (3/3), `fullpre` **0.3%**
     (**2/3**). **`fullpre`'s 8.4% was one rejected run** — over usable runs it
     is the *tightest* protocol measured. `subset` cannot be given a range at
     all. The spread ranking and the excursion ranking are the same ranking
     because one is largely made of the other.

     So the open question is now **why do the shortest and longest protocols
     produce more P-state excursions** — a rate over 72–288 observations rather
     than a range over three runs, and `fullpre` drops out of it.
     `compare_protocols.py` prints both ranges whenever they differ.

     **First candidate for that, tested and NULL: row duration.** Excursion rate
     falls hard with duration pooled — 3.5 / 8.3 / 11.1 / 6.9 / 1.4 / **0.0%**
     across six buckets over 864 observations, r = −0.686, and the slowest bucket
     has zero excursions in 144 observations. But duration is confounded with
     kernel identity (fused Triton rows are short, SDPA rows long), and holding
     the method fixed only **7 of 11** come out negative. Not a sweep, so it is a
     null. `clock_excursions.py` carries `duration_effect` and prints the verdict
     either way. Do not revive this without a design that varies duration within
     a kernel.

     **Second candidate, also null: the PRECEDING row's duration.** The best idea
     available, because it explains the protocol difference directly (`subset`
     times 3 methods so its fast kernels follow each other; `full` times 12 so a
     fast kernel is usually preceded by a slow DRAM-heavy one). Excursion rate by
     predecessor duration over 978 observations: 4.1 / 6.2 / 5.6 / 4.6 / 4.5% —
     **no gradient at all**. Within method 8 of 12 negative. Dead.

     **STOP TESTING HYPOTHESES AGAINST THIS DATASET.** Five have now been tried
     on the same ~1000 observations (temperature, clock warm-up, the four
     telemetry variables, row duration, predecessor duration). Continuing is
     p-hacking: the within-method decomposition protects against confounding,
     not multiplicity, and nothing here protects against multiplicity. **The
     excursion mechanism is not identifiable from the runs on disk.** What is
     known is a rate (2.8 / 12.5 / 1.4 / 5.6 / 4.2% by protocol) and that the
     gate catches the excursions that matter, which protects the published
     numbers whatever the cause. Anything further needs **data collected for the
     question** — a design that varies one candidate while holding the others
     fixed.

   - Original framing, kept for the record. Spreads
     at `quant_cold@8192` are `full` 0.6%, `subset` 13.2%, `preloaded` 0.6%,
     `fullpre` 8.4%; excursion rates 2.8% / 12.5% / 2.8% / 6.9%. That is not
     monotone in load or in temperature (69.3 / 69.5 / 70.8 / 72.1 C — the
     coolest protocol and the hottest are the two that misbehave). A
     two-mechanism story fits (too little preceding work and the clock never
     comes up; too much and thermal pressure pulls it down) but four protocols
     and two free parameters is a description, not a test. A temperature sweep
     at fixed protocol would be the actual experiment.
   - **CORRECTED (2026-09-03): the misfit is the control, not the fused
     kernel.** This line used to read "`fused_triton_4b@16k` still fits neither
     story". Against the fitted |shift| ~ bandwidth line that row is the
     *fourth-best fit of twelve* (mean |residual| 0.45 pp). The misfits are all
     four `triton_fp16_control` rows: 1.74 pp against 0.41 pp for everything
     else, worst at `@8k` (2.32) and `@16k` (2.28). `bandwidth_law.py` measures
     it. Why the control specifically is now the open question — and it is
     weaker than it first looked: under a scale-free log-log fit (exponent
     0.82/0.84/0.98, so the relationship is essentially linear) the control is
     worst under only **2 of 3** protocols. Ruled out so far: bytes moved,
     footprint, time, curvature of the fit, and "its shifts are just bigger"
     (relative residuals are not flat either). The memory clock does **not**
     answer it — `control@8k` is the worst-fitting row of the
     twelve and its memory clock is constant at 11001 MHz under all four
     protocols.
   - Bandwidth is also the **best of seven candidate predictors** (its own
     square and log, bytes moved, log bytes, time, log time): smallest residual
     spread under `preloaded` and `fullpre`, second under `subset`, and nothing
     beats it under every protocol. So the control's misfit is not explained by
     bytes moved, by footprint, or by time either.
   - The law itself survived the test that could have killed it: within a single
     kernel, r is positive in **6 of 6** (method x protocol) pairs that have any
     bandwidth range (sign test p = 0.016), the between-method means are
     monotone under all three protocols, and leave-one-out never takes r below
     +0.644. So bandwidth is not a proxy for method identity.
   - The r values leave ~30-50% of the variance unaccounted.

   Do **not** re-run the confounded comparison: the 2x2 supersedes it.

4. **2-bit still deserves a decision, not a table row.** Numerically unusable
   (rel L2 ≈ 0.7; the audit carries it as `correct.2bit_usable`, MISLEADING —
   the *kernel* is correct to cosine 1.000000, the *quantization* is 72% off)
   under per-token grouping along `head_dim`. KIVI's result is that keys want
   per-channel grouping. Either implement per-channel keys and re-measure, or
   state plainly that 2-bit is out of scope. (Note it is also where `fold_zp` is
   *worst*, 0.66×.)

   **What that costs, measured 2026-09-03 — and "just stop benchmarking it" is
   not free.** 2-bit is **5 of the 12 methods** per context
   (`fused_triton_2b`, `fused_gather_meta_2b`, `fused_fold_zp_2b`,
   `dequant_sdpa_eager_2b`, `dequant_sdpa_compiled_2b`) and **43% of the run's
   measured wall clock** (284 s of 661 s in clock windows; 775 s end to end).

   Dropping it takes the shipped protocol from **12 methods per context to 7** —
   a move along *exactly* the axis the 2x2 measured. At `quant_cold@8192` the
   simple effect of method count with no preload is **+3.78%** (3 methods 1.4217
   → 12 methods 1.4755), and by that sign a shorter run reads *lower*. So
   removing 2-bit is a **protocol change, not a documentation change**: the
   headline cell would have to be re-measured under the new protocol, not merely
   re-rendered, and the repo's own finding is what says so.

   That does not argue for keeping it — it argues that the tidy-up option has a
   price and the price is known. A third option exists and is cheaper than
   either: keep timing it (protocol unchanged) and mark it in the tables as a
   research row rather than a candidate, which is what the accuracy claim
   already says in prose.

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

**Do not run analysis on the CPU while `benchmark.py` is timing.** Learned the
expensive way on 2026-09-02, during the fourth-protocol runs. Numpy bootstraps
(`audit_claims.py`, `compare_protocols.py`), `pytest` and `analyze_dispersion.py`
were run alongside a benchmark; the run came back with a memory P-state
excursion on `fused_triton_4b@8192` (10144 MHz against a cell median of 11001)
that pulled `quant_cold@8192` down to 1.278 -- below every hypothesis the
experiment had pre-registered.

The mechanism is the one this repo already knows about, arriving from a new
direction: the timing loop has to keep the GPU saturated, and a multi-core
bootstrap that deschedules the submitting thread lets the GPU idle long enough
to drop a P-state. The DRAM-resident rows are exactly the ones that care. What
makes it worse than noise is that it is *indistinguishable from the effect under
test* -- the whole question was whether the protocol changes the memory
subsystem's state.

Two runs were discarded and re-measured; they are kept in
`results/tail/contaminated/` because they are a clean demonstration of the
effect. While a run is in flight, restrict yourself to file reads, greps and
markdown edits.

Two Windows process facts from the same incident: stopping a background *task*
kills the tracked shell and not its descendants -- the `sh.exe` driver loop and
its `benchmark.py` child kept running and launched the next run -- and
`Stop-Process` is refused by the auto-mode classifier, so there may be no way to
end a run early. Check with `Get-CimInstance Win32_Process -Filter
"Name='python.exe' or Name='sh.exe'"` before assuming a kill took, and prefer
letting a run finish over starting anything that would overlap it.


A Claude Code session in this directory once survived its terminal being closed
and kept writing files while a second session worked in the same tree. If files
appear that nobody in the current session wrote, check for orphaned
`claude.exe` processes before assuming the working tree is yours alone.
