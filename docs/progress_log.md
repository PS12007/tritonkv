# Progress log

Written as it happened, not reconstructed afterwards. Each entry: what was
attempted, what actually happened, and one concrete number where there is one.
Timestamps are local (UTC+?, machine clock).

---

## 2026-08-30 19:18 — Environment triage

**Attempted:** find out what hardware this project actually has to run on before
designing anything.

**What happened:** `nvidia-smi` reports an **RTX 5060 Laptop, 8 GB, driver
610.47, CUDA UMD 13.3** — Blackwell, compute capability sm_120. The installed
Python had `torch 2.12.0+cpu`, i.e. no CUDA at all, so nothing could have run.

Two things had to be true before a single line of Triton could execute:

1. A CUDA torch build new enough for sm_120 (CUDA ≥ 12.8).
2. Triton on **Windows** — upstream Triton ships no Windows wheels, so this
   needs the `triton-windows` community build, which in turn needs an MSVC
   toolchain to compile.

`vswhere` found **Visual Studio 2022 Community with MSVC 14.44** already
installed, so the MSVC requirement was already satisfied.

**Number:** 8151 MiB of VRAM, 838 MiB already consumed by the desktop → about
**7.1 GB usable**. That comfortably rules out anything requiring a large model
and confirms the brief's "keep it cheap" constraint is the right call.

**Decision:** project-local venv with `torch==2.12.0+cu130` and
`triton-windows==3.8.0.post28`, so the global interpreter is left alone.

---

## 2026-08-30 19:20 — Reference implementation first (no Triton yet)

**Attempted:** per the brief, build the PyTorch ground truth before any kernel.

**What happened:** wrote `quantize.py` (group-wise 2/4-bit asymmetric quant) and
`reference.py` (fp32 ground-truth attention + the baselines). Two design calls
that turned out to matter, both made before writing any kernel:

**1. Metadata is rounded to fp16 *before* the codes are computed.** The obvious
implementation computes `scale` in fp32, quantizes with it, then stores `scale`
as fp16. That leaves the reference and the kernel dequantizing with *different*
scales, and the resulting mismatch looks exactly like a kernel bug. Rounding
first makes the two bit-identical, so any later disagreement is real.

**2. A "split-P" packing layout instead of the obvious interleaved one.** The
natural layout puts dims `2j, 2j+1` in byte `j`. That forces a de-interleave
inside the kernel. Instead byte `j` holds dims `j, j+DP, j+2·DP, …`, so a kernel
that loads a contiguous run of bytes gets `P` already-aligned sub-vectors of the
head dimension for free.

**Number:** 4-bit with `group_size=32` and fp16 scale+zero is **5.0 effective
bits per element**, not 4. That is 3.2× compression over fp16, not 4×. Every
memory claim in this project quotes the effective number.

---

## 2026-08-30 19:35 — Kernel design: the unpack problem

**Attempted:** write the fused kernel.

**What happened:** hit the real design problem immediately. To feed `tl.dot`,
the kernel needs a dense `(BLOCK_N, D)` tile of dequantized K. But the load
produces `(BLOCK_N, D/P)` bytes, and Triton has no way to slice-assign a tile,
so building the full-width tile wants either a `tl.join`+`tl.reshape` (which
costs a layout conversion through shared memory) or `P` separate accumulators
held in an unrolled Python list (which depends on fragile constexpr-unrolling
behaviour).

**The way out:** don't reconstruct the tile at all — *address* it. Build an
index vector over the full head dim and load byte `d % DP` with shift
`(d // DP) · nbits`. Every byte gets loaded `P` times, but all `P` loads hit the
same cache line, so DRAM traffic is unchanged and the result is a dense
`(BLOCK_N, D)` tile with no reshape, no transpose, and no unrolled accumulators.
The whole unpack is four lines.

**Number:** the arithmetic that motivates the project. Per cached element-row,
the naive path moves `0.5·D` (read packed) + `2·D` (write fp16) + `2·D` (read it
back) = **4.5·D bytes**; the fused kernel moves **0.5·D**. Ceiling on the
speedup for a purely memory-bound decode: **~9×** against the eager baseline,
less against a fused/compiled one.

---

<!-- entries below this line are appended as results come in -->

## 2026-08-30 19:40 — Environment fixed, kernel runs, first correctness pass

**Attempted:** get the kernel to actually execute.

**What happened:** the `torch 2.12.0+cu130` install had been interrupted (DLL
load failure on `caffe2_nvrtc.dll`); it completed and now imports cleanly.
First compile of the kernel failed outright:

> `NameError: Cannot access global variable LOG2E from within @jit'ed function.`

Triton 3.8 only captures globals already declared `tl.constexpr`. One-line fix.

**Number:** **66/66 correctness tests pass in 25.8 s** on an RTX 5060 Laptop
(sm_120, 26 SMs, 8 GB). 2-bit passes the same kernel-vs-dequant-reference
thresholds as 4-bit, so the stretch goal is numerically live, not abandoned.

---

## 2026-08-30 19:45 — The baselines were strawmen, and the "cold" timing was fiction

**Attempted:** believe the first benchmark output. Failed to.

**What happened:** the first run reported `hot` slower than `cold` and launch
overhead larger than total runtime. Two separate problems, both inflating the
kernel's win.

**1. The fp16 baseline was crippled.** Every baseline went through SDPA with
`enable_gqa=True`. Measured at S=8192 by CUDA-graph replay:

| strategy | time | effective |
|---|---|---|
| `enable_gqa=True` (what we used) | 990 us | 8.5 GB/s |
| dispatcher's choice, expanded KV | 746 us | 11.2 GB/s |
| mem-efficient backend | 746 us | 11.3 GB/s |
| **cuDNN backend** | **318 us** | **26.4 GB/s** |
| hand-written fp16 bmm | 382 us | 22.0 GB/s |

FlashAttention was never a candidate: this Windows torch build reports *"Torch
was not compiled with flash attention"*. `reference.py` now probes all six
strategies once per shape and caches the winner. Ranking has to be done by
CUDA-graph replay too — WDDM's 50–300 us per-call submission cost had the probe
preferring a 3.3 GB/s path over a 24.7 GB/s one at ctx=512.

**2. "Cold" was measuring Windows, not the GPU.** `flush L2, time one call`
reported 395 us for 43 us of GPU work: on an idle GPU that number is the WDDM
submission path waking up. Replaced with a rotating working set — N independent
copies of the cache, sized to exceed L2, replayed from a single CUDA graph.
Cold and hot now agree to a few percent with sub-microsecond std.

**A real bug, caught by the baseline getting *better*:** caching the probe's
*closure* instead of its strategy *name* pinned the probe's own tensors, so
every later call returned the first call's answer. Baseline error against fp16
truth jumped 0.121 → 0.900. The test suite did not catch it, because the only
assertion involving the baseline requires the kernel to be no *worse* than it.

---

## 2026-08-30 19:50 — The headline speedup is mostly not what it looks like

**Attempted:** falsify the project's own thesis before writing it up.

**What happened:** the fused kernel beats fp16 SDPA by 9–36x. But that compares
two things at once: 4-bit storage *and* a flash-decoding split across the
history, which PyTorch's SDPA simply does not do for `q_len == 1`. So
`kernels/fp16_decode_attn.py` was written as a **control** — same split, same
online softmax, same GQA amortization, same combine kernel, K/V read as plain
fp16. The only difference is the dequantization.

Hot (L2-resident), CUDA-graph replay, us per decode step:

| ctx | SDPA fp16 | Triton fp16 (control) | fused 4-bit | flash-decode | quantization |
|---|---|---|---|---|---|
| 512 | 46.1 | 3.2 | 5.2 | 14.6x | **0.61x** |
| 2048 | 173.9 | 6.5 | 10.1 | 26.7x | **0.64x** |
| 8192 | 732.1 | 13.5 | 25.7 | 54.2x | **0.53x** |
| 16384 | 1452.6 | 20.5 | 40.3 | 70.9x | **0.51x** |

**Number: quantization makes the kernel ~2x SLOWER, not faster** — when the
cache fits in L2. Nearly all of the apparent win is the split, not the fusion.
Group size barely moves it (25.4 / 25.6 / 26.6 us at gs=16/32/64), so the
scale+zero tile loads are not the cost; the shift/mask/convert/fma chain is.
At 16k the fp16 path runs at ~410 GB/s (pure L2) while the 4-bit path reaches
~64 GB/s — the fused kernel is issue-bound, not bandwidth-bound.

**Then the sign flips.** With a working set 4x L2 (134 MB), so the cache
genuinely comes from DRAM:

| ctx | Triton fp16 (control) | fused 4-bit | quantization |
|---|---|---|---|
| 512 | 4.64 us | 6.34 us | 0.73x |
| 2048 | 11.26 us | 59.03 us | 0.19x |
| 8192 | 362.36 us | 32.60 us | **11.11x** |
| 16384 | 795.11 us | 51.02 us | **15.58x** |

So the real claim is conditional and much more interesting than "Nx faster":
**the fused kernel wins exactly when the KV cache does not fit in L2, and loses
when it does.** That is the thing worth writing up.

**Caveat, not yet resolved:** those cold numbers carry +-50-100 us of variance
and the 2048 row is not credible. Suspected laptop power/thermal throttling
(80 W cap; idle clocks observed at 180 MHz SM / 810 MHz mem). Must be re-run
with clock monitoring before any of it is quoted.

---

## 2026-09-01 11:45 — The clock-monitored re-run overturns the headline

**Attempted:** the task at the top of `next_steps.md` — re-run the cold
benchmark with clock monitoring, on the theory that the +-50-100us variance and
the incredible ctx=2048 row were power/thermal throttling.

**What happened:** they were. And the throttling was not adding noise to the
result, it was *creating* it.

`benchmark.py` now runs a background `nvidia-smi -lms 100` sampler, spins the GPU
to >=80% of max SM clock before every measurement, and attributes clock samples
to the sampling loop only. Three iterations were needed to get the gate right:

1. Bracketing the whole row and requiring <5% SM-clock spread rejected 14/16
   rows. Wrong gate: a boosting GPU dithers several percent between samples
   under steady load, while the timings themselves had <2% IQR.
2. The gate that matters is **boosted, not constant** — every sample >=70% of the
   3090 MHz max — plus the measurement's own IQR <=5% of its median. Two
   independent gates: was the GPU at speed, and did the samples agree.
3. Even then, slow baselines failed. The clock window was covering warmup and
   CUDA-graph capture, during which the GPU idles back down. Added a `span`
   out-parameter to each `bench_*` so the window covers the **sampling loop
   only**. 13/16 on the smoke run, 22/32 on the full run.

**The number that changed.** Previously reported DRAM-resident quantization
effect (fp16 control / fused 4-bit):

| ctx | before | after |
|---|---|---|
| 2048 | 0.19x | 1.14x |
| 8192 | **11.11x** | 1.20x |
| 16384 | **15.58x** | 1.32x |

The fp16 control at 16k was measured at 795us. It is 63.4us. A 12.5x error, and
the SM clock range on this part is 285 -> 3090 MHz, which is 10.8x. The control
kernel is fast enough that its whole measurement fit inside the GPU's spin-up.
The old benchmark could not see this because it never recorded the clocks.

**So the L2-conditional finding survives, but at a tenth of the size.** The sign
still flips — quantization costs 1.4-2x when the cache is L2-resident and pays
1.14-1.32x when it is not — and that is still the interesting claim, because it
is a claim about *when*, not about a factor. The 11-15x was never real.

**Also done in this session:**

- `audit_claims.py` gained the attribution section it was missing: flash-decoding
  effect, quantization effect in both regimes, the L2 conditional, and a
  `method.clock_verified` claim. It now also refuses to quote a ratio whose
  baseline failed the clock gate — a de-boosted baseline inflates every speedup
  measured against it, always in the flattering direction. 63 claims:
  24 TRUE / 20 CONDITIONAL / 9 MISLEADING / **10 FALSE**, all ten of them ours.
- `stats()` reports median/p25/p75/IQR alongside mean/std; the console and the
  README now lead with median (IQR).
- Raw per-sample hot-regime timings (`graph_raw_ms`) are stored, so the
  L2-resident regime gets bootstrap CIs too rather than a bare mean.
- `make_plots.py` + `docs/plots/` (8 figures), `docs/key_numbers.md`,
  `docs/thread_outline.md`, and a rewritten README results section.

**The lesson worth keeping:** the earlier session did the right adversarial
thing — build a control kernel to falsify your own thesis — and still got a
wrong answer, because the control was the *fastest* thing being measured and so
was the most sensitive to a slow GPU. Falsification does not help if the
measurement apparatus is what is lying. Check the apparatus first.

---

## 2026-09-01 18:30 — The metadata gather was most of the inner loop

**Attempted:** `next_steps.md` item 2 — attack the issue-bound
shift/mask/convert/fma chain, on the theory that folding the zero-point out of
the inner loop (`q·(code·scale + zero) = scale·(q·code) + zero·Σq`) was the way
in. That is not what turned out to matter.

**What happened.** Before writing the segmented dot, I looked again at what the
inner loop actually issues, and found a much cheaper target. The per-group scale
and zero live in memory as `(BLOCK_N, D // group_size)` — 4 values per token at
`gs=32`. The kernel was loading them as `(BLOCK_N, D)` by indexing with
`d // group_size`, i.e. **re-reading every parameter 32 times**, four times per
iteration (K and V, scale and zero). Loading them at their real width and
expanding in registers is one `tl.broadcast_to` + `tl.reshape`:

| | instructions | registers | spills |
|---|---|---|---|
| gather (`d // GS`) | 2245 | 244 | 0 |
| broadcast | **1653** | **128** | 0 |

Both paths are **bitwise identical** — asserted over 40 cases of
(S × nbits × group_size) in `test_correctness.py`, not merely close — so the
whole gap is issue cost.

**Measured (clock-gated, `fused_gather_meta_*` is carried as a permanent
control row so this stays auditable):**

| regime | 4-bit | 2-bit |
|---|---|---|
| L2-resident | 1.16 / 1.27 / 1.32 / 1.48× | 1.15 / 1.27 / 1.40 / 1.48× |
| DRAM-resident | 1.15 / 1.05 / 1.13 / 1.24× | 1.14 / 1.06 / 1.22 / 1.32× |

(ctx = 512 / 2048 / 8192 / 16384.) Bigger where the kernel is issue-bound rather
than bandwidth-bound, which is the right signature.

**The README claim this refutes.** It said: *"Group size barely moves it
(25.4 / 25.6 / 26.6 µs at gs = 16/32/64), so the scale+zero tile loads are not
the cost."* The measurement was right and the inference was wrong. In the gather
path the load is indexed by `d // GS` over the **full** head dim, so it issues
`BLOCK_N * D` loads *whatever `GS` is* — group size changes how many distinct
values are read, never how many instructions are issued. The experiment varied
metadata **bytes** and concluded about metadata **instructions**. A flat sweep
was exactly what the expensive version predicts.

**Two things that did *not* work, and are worth as much as the one that did.**

1. *Zero-point folding* (the item next_steps predicted "could change the
   conclusion of the whole project"). Implemented, kept behind `fold_zp=`. It is
   **not** a speed win: 0.74–1.10× (4-bit) and 0.66–0.94× (2-bit) against the
   broadcast path, clearing no practical-significance bar at any context in
   either regime. An early ad-hoc A/B *did* show it winning 1.04–1.09× in DRAM;
   the harness disagreed once both variants shared cache replicas under the
   gate, and the disagreement is the finding — the effect is inside measurement
   variation. It is kept because it is **more accurate**: it never rounds a
   dequantized K value to fp16, so kernel error stays flat at 1.5e-4 instead of
   drifting 2.3e-4 → 7.7e-4 as context grows.
2. *The same narrow-load trick applied to the packed codes.* Bitwise identical,
   and a **loss** (0.69–0.96× at ctx ≥ 8192). Registers go 128 → 223: the codes
   are needed at full width regardless, so expanding them from a narrow load
   adds a live tile without removing one. Reverted. The lesson generalizes less
   than it first appears — the win is specific to loads whose expanded form is
   redundant.

---

## 2026-09-01 18:50 — The clock gate was passing rows on a single sample

**Attempted:** `next_steps.md` item 1 — close the unquotable rows. The stated
theory was that the *slow* PyTorch baselines were sagging. It was the opposite.

**What happened.** `triton_fp16_control` — the *fastest* method — kept failing
the boost gate. The cause is ordering: `warm_clocks()` ran in the driver, then
`bench_graph` did warmup, CUDA-graph capture and priming replays before opening
the clock window. That stretch is CPU-bound with the GPU near idle and long
enough for the boost to decay, so the ramp was spent before the measured window
began. A slow baseline re-boosts inside its own first sample; a 14 µs kernel
never does. **The gate was penalising methods for being fast.** Fixed by handing
the ramp *into* the timing function, run after capture and before the priming
replays (which then put the cache back, since the ramp's GEMM evicts it).

That worked — and immediately exposed something worse. With the control now
passing, I checked *how* it passed: `n_samples: 1`. **28 of 96 measurement
windows were being judged on a single `nvidia-smi` sample**, including the rows
the whole attribution rests on. "The GPU was boosted" verified once at 9 Hz is
not verification; it is a measurement shorter than the sampler's period.

Fixes, both from measurement rather than guesswork:

- The sampler really does run at **9.2 Hz** (109 ms median gap, measured), so a
  window must stay open well past a second to earn samples. Measurements now
  keep sampling to `MIN_SAMPLING_SECONDS = 1.5`.
- The gate requires `MIN_CLOCK_SAMPLES = 4`, reported as its own failure mode
  ("too few clock samples") rather than silently passing.
- The first attempt bounded the stretch by *sample count* (`samples * 60`),
  which silently bound first for exactly the fast kernels that needed it — their
  samples are cheap, so 600 of them is 0.26 s. Re-bounded in **time**.

**Result:** windows with ≤ 1 clock sample went **28 → 0**; the minimum is now 13.
Every remaining rejection is dispersion — there are no clock rejections left.
And the attribution is now fully quotable at ctx = 512 **and** 2048 in *both*
regimes, with all three methods passing the gate at the same context:

| ctx | regime | split | quantization |
|---|---|---|---|
| 512 | L2-resident | 13.3× | 0.74× |
| 512 | DRAM-resident | 10.0× | 0.97× |
| 2048 | L2-resident | 26.2× | **0.90×** |
| 2048 | DRAM-resident | 14.5× | **1.22×** |

That is the first time the **sign flip itself** has had every input
clock-verified at a single context. Previously it rested on rows where the fp16
control had failed the gate.

**The lesson, which is the same one as 2026-09-01 11:45 one level down:** the
apparatus had a second failure mode underneath the one already fixed. Adding the
clock monitor caught throttling; it did not catch *not having looked long enough
to tell*, and a gate that answers "yes" from one sample reads identically to one
that answers from twenty.

---

## 2026-09-01 21:20 — The audit's own new section had never run

**Attempted:** `next_steps.md` item 0 — regenerate the stale
`results/audit.{md,json}`.

**What happened.** It crashed. `audit_optimizations()` — the section added
earlier the same evening to adjudicate this session's two kernel changes against
their own controls — indexed `bootstrap_ratio_ci`'s return value as
`ratios_cold[c][1][0]`, but that function returns a *flat* `(mean, lo, hi)`
triple, not `(mean, (lo, hi))`. Every other caller in the file unpacks it
correctly. So the section had been written, committed, and described in the
README, and had **never once executed**.

Worth saying plainly: the reason it went unnoticed is that the audit takes
~13 minutes, so it does not get run casually. A tool that is expensive to run is
a tool whose failures are discovered late.

**Which is why the second fix was to make it cheap.** `bootstrap_ratio_ci` was
10,000 resamples of two ~600-sample rows in a Python loop — about 12M
interpreter-level draws per call, dozens of calls per audit. Vectorized into two
numpy index matrices: **13 min → 20 s.** The stdlib version is kept as
`_bootstrap_ratio_ci_py` and `--check-bootstrap` runs both on the real rows and
reports the disagreement, because swapping an estimator's RNG is a change to the
measuring apparatus and this project has already been bitten twice by those.
Worst endpoint disagreement over 8 real ratios: **2.3% of the CI width.**

**And the third fix was a verdict the section got wrong.**
`optimization.zero_point_fold.4b` came out `TRUE BUT CONDITIONAL` on a
DRAM-resident 1.08× at ctx=8192 whose bootstrap CI cleared the 1.05× bar. But
neither the baseline nor the folded row passed the clock/dispersion gate at that
context — the neighbouring `meta_broadcast` claim applies that gate and this one
did not. Gated, it reads **FALSE**, which is what the 2-bit claim already said
and what the progress log has said since the fold was measured.

Current counts: **67 claims — 21 TRUE / 25 CONDITIONAL / 9 MISLEADING /
12 FALSE.**

---

## 2026-09-01 21:45 — The group-size sweep, run properly, refuted the prediction

**Attempted:** `next_steps.md` item 2. The old sweep (25.4 / 25.6 / 26.6 µs at
gs = 16/32/64) ran on the gather path, where group size cannot affect the load
count, so it was evidence for nothing. On the broadcast path `GS` genuinely sets
the count (`BLOCK_N * D/GS`), so `sweep_group_size.py` runs **both** paths in one
session with the prediction written down first:

> broadcast: sloped, monotone in `D // GS`. gather: flat.

**What happened.** Gather was flat. Broadcast was flat too — 1.07–1.17× across
an 8× range of metadata loads. The prediction was wrong.

The right shape is **saturation**, and it only shows up when the broadcast
change's own step is put in the same table (L2-resident, ctx=8192, every row
clock-verified):

| path | metadata loads / tile | median |
|---|---|---|
| gather, gs=32 | 4096 | 22.0 µs |
| broadcast, gs=16 | 256 | 17.0 µs (16× fewer loads → **1.29×**) |
| broadcast, gs=128 | 32 | 15.9 µs (a further 8× → **1.07×**) |

So metadata loads are a real cost that stops binding about an order of magnitude
below where the gather path sat. The broadcast change was worth 1.29× because it
crossed that point — not because time is proportional to load count. Note how
close this came to being the *same* mistake in reverse: a project that had run
only the bottom two rows would have concluded metadata loads were free, on a
flat sweep, exactly as the first version did from the other side.

**The gs=128 outlier turned out to be the interesting result.** The old sweep
recorded 93 µs at gs=128 against ~26 µs elsewhere and left it alone. It
reproduces: on the gather path gs=128 is **1.95× / 2.88× / 3.52×** slower than
gs=64 at ctx = 512 / 2048 / 8192 (L2-resident, IQR ≈ 1%, clock-verified at 2048
and 8192).

It is not a load-count effect. That cell issues *fewer* PTX instructions than
gs=64 (2415 vs 2989) and the same number of global loads. `probe_gs128.py`
compiles every cell and counts what the PTX actually contains:

| cell | ld.global | st.shared | ld.shared | regs | spills |
|---|---|---|---|---|---|
| gather, gs=64 | 130 | 16 | 38 | 124 | 0 |
| gather, gs=128 | 130 | **72** | **98** | 121 | 0 |
| broadcast, gs=128 | 10 | 16 | 38 | 108 | 0 |

(block_n=32, 4 warps; the same jump appears in all nine (block_n, num_warps)
combinations checked, e.g. 40 → 264 at block_n=128.)

**Cause.** When `group_size == head_dim`, `tl.arange(0, D) // GS` folds to
all-zeros. Triton then gives the loaded tile a layout that has to be converted
through shared memory before it can feed the dequantize path, and the conversion
is inside the loop — which is why the penalty *grows* with context rather than
amortizing. The redundant-load form is slow; the redundant-load form with a
constant index is much slower.

The shipped broadcast path is unaffected: at gs=128 it is the fastest cell in the
whole table. So this is a fact about the control row, not about the kernel — but
it closes a number that had been sitting in this repo unexplained since the first
week, and it is a second example of the session's theme, that **fewer
instructions is not the same thing as less work.**

---

## 2026-09-01 22:20 — The dispersion rejects, and the CI that was too narrow

**Attempted:** `next_steps.md` item 1 — close the 23 of 48 rows failing the
dispersion half of the gate. That file offered two fixes pointing opposite ways
(shorter windows to reduce drift, longer ones to average it out) and said the
choice should be made by measuring the drift. So `analyze_dispersion.py`
decomposes every rejected series before touching anything, into the three things
that would call for different fixes: a **trend** across the sample index (a
shorter window helps), a **tail** of interrupted samples (a trimmed statistic
helps), and **wander or white jitter** with no trend at all (no window length
helps, because IQR is a property of the sample distribution and does not shrink
with more samples).

**What happened.** Across 96 measurements, 25 fail:

| cause | count | where |
|---|---|---|
| drift — a shorter window fixes it | 1 | ctx=16384 |
| drift, but not enough on its own | 7 | ctx=512 ×4, 8192 ×2, 16384 ×1 |
| tail — a trimmed statistic fixes it | 4 | spread |
| wander (lag-1 ≥ 0.25), no fix by window length | 6 | ctx=8192 ×2, 16384 ×4 |
| white jitter, no fix by window length | 7 | ctx=512 ×3, 2048 ×1, 16384 ×3 |

Only **8 of 25** carry a statistically significant trend; **13 of 25** have
neither a trend nor a tail. So the premise of both proposed fixes is mostly
wrong, and the thermal-drift story that motivated them describes one row.

Two things worth recording about the method. First, significance rather than
r-squared decides what counts as a trend: the fp16 control's DRAM-resident rows
fall **19–23% across their window** with an r² of 0.09–0.16, because the noise
around that decline is loud. An r²-based test filed the largest systematic
effect in the run as jitter. Second, the classification is keyed on *what each
fix would do*, not on a statistic, because a label nobody can act on is not a
diagnosis.

**Then the useful part.** If the spread is not drift, how badly does it actually
hurt? A moving-block bootstrap of each rejected row's median says: not at all.
The failing measurements pin their medians to **±0.69%** (median; worst
±3.36%), against ±0.16% for the passing ones — while the effects these tables
report are 10–50%. A starred row means *the card was restless*, not *the number
is unknown*.

**The gate is deliberately left exactly as it was.** Loosening a gate because it
is inconvenient is how the numbers this project exists to avoid get published.
What changed instead is that the auditor now says all of this out loud, as
`method.dispersion_gate` — a MISLEADING verdict against the claim that a rejected
row's median cannot be trusted.

**And the analysis found a real defect one level up.** The rejected rows are not
just dispersed, they are *serially correlated* — lag-1 autocorrelation up to
**0.72**. Every confidence interval in `audit_claims.py` was an i.i.d.
bootstrap, which assumes exactly the independence the data does not have. On
these rows the i.i.d. interval is up to **1.95× too narrow**, and every verdict
here is decided by whether an interval clears a bar. Too narrow means too
confident, in the flattering direction. Fixed with a circular-block resample.

**Two corrections inside that fix, both caught by checking rather than by
reasoning:**

1. The first version used *moving* blocks. Samples near either end of a series
   can only be reached by the few blocks that overlap them, so the ends are
   under-weighted and the resampled mean drifts toward the middle of the run —
   which shifts the interval's **centre**, not just its width. It promoted one
   claim from CONDITIONAL to TRUE: a reweighting artefact dressed as a
   statistical correction. Circular blocks weight every sample equally.
2. That did not fix it, which was the more interesting outcome. The claim was
   sitting on a knife edge: its CI low moved from **1.04973 to 1.05019** against
   a 1.05 bar — five parts in a hundred thousand, on a pair of series with no
   measurable autocorrelation. `_verdict` was a step function evaluated exactly
   at the threshold, so any rounding decision anywhere became a verdict. It now
   requires the deciding endpoint to clear the bar by at least 10% of the
   interval's own width, and says so in the evidence when it does not.

**Result:** with the block bootstrap *and* the margin, **zero verdicts change**.
The correlation correction did not move a single conclusion; the one apparent
change was the missing margin all along. 68 claims: **21 TRUE / 25 CONDITIONAL /
10 MISLEADING / 12 FALSE**.

---

## 2026-09-01 23:10 — I was warming up the wrong half of the machine

**Attempted:** `next_steps.md` item 3 — the attribution is only fully quotable at
ctx ≤ 2048 because `triton_fp16_control` keeps failing at 8192 and 16384. The
dispersion decomposition said those two rows fall **19.3% and 22.6% across their
own measurement window**, the largest systematic effect anywhere in the run. A
19% decline is not jitter. Something was still ramping.

**It was not the SM clock.** Those windows pass that gate — SM min 2445–2490 MHz
against a 2472 MHz floor. It was the **memory** clock. The P-states on this part
are 405 MHz at deep idle, **12001 MHz at light idle**, and **9001 or 11001 MHz
under load**: an 80 W laptop chip shares power between the domains, so the memory
clock comes *down* when the SMs start working, and then moves between those two
states — a 20% swing. A bandwidth-bound measurement spanning that is two
measurements averaged together.

The monitor had been computing `mem_clock_constant` since the clock work began.
It had never been used.

| DRAM-resident windows | n | median trend | median IQR | fail IQR ≤ 5% |
|---|---|---|---|---|
| memory clock changed | 22 | **−5.6%** | 3.8% | **9** |
| memory clock held | 26 | −0.9% | 1.7% | 1 |

**The cause, and it is embarrassing in a useful way.** The pre-measurement ramp
was a 2048×2048 fp16 GEMM. That is compute-bound and works out of cache: it
drives the SM clock hard and asks the memory system for essentially nothing, so
the governor had no reason to move the memory clock until the *measurement*
started touching DRAM. Every ramp in this project has been warming the half of
the machine that was already fine.

The ramp now runs a DRAM-sized copy alongside the GEMM, and waits for the memory
clock to stop changing rather than for the SM clock to cross a line.

**Two bugs inside that fix, both caught by running it rather than reading it.**

1. The wait was unbounded, and under load the clock oscillates rather than
   settling, so it is never satisfied: every ramp ran to its 20 s timeout and a
   two-context smoke run stopped finishing. Bounded at 2.5 s, with the outcome
   *reported* rather than enforced.
2. It targeted the highest memory clock ever seen — which is the **idle** 12001
   MHz, a value no measurement will ever run at. The ramp was waiting for a clock
   the measurement cannot reach. The ceiling is now learned only from samples
   taken while the GPU is busy.

**And then the change I was most confident about, measured and rejected.** The
obvious companion to the ramp fix was to put memory-clock stability into the
gate, the way SM clock stability already is. It rejects **every** DRAM-resident
row — rows whose timing IQRs are 0.4–2%. A gate that refuses a measurement that
tight is not measuring measurement quality.

The decisive evidence is sharper than that, and it points the opposite way from
the intuition. Comparing the same 48 measurements before and after the ramp fix:

| | median \|trend\| | median IQR | IQR > 5% | memory-clock spread |
|---|---|---|---|---|
| L2-resident, old ramp | 2.4% | 1.6% | 5/24 | mostly 0% |
| L2-resident, new ramp | **0.2%** | **0.7%** | **3/24** | ~19% |
| DRAM-resident, old ramp | 2.0% | 1.8% | 5/24 | 0% or ~21% |
| DRAM-resident, new ramp | 1.9% | 1.8% | **2/24** | ~19% |

**The ramp that made the measurements better made the memory clock move more.**
Gating on memory-clock movement would have thrown out precisely the measurements
the fix improved. (Caveat on that table: the "new" numbers come from a
`--quick` run at 8 samples against a full run at 50, so the trend estimates are
noisier; the clean comparison is the full re-run this entry is waiting on. The
direction is not subtle, though — the largest old drifts, −13% to −16% at
ctx=512, are gone.)

What the memory clock was, all along, was a **ramp** problem wearing the costume
of a gate problem. Warm the memory system first and the drift loses its
direction; what remains is oscillation that both methods sit in equally, which is
noise in a ratio rather than bias, and which the dispersion gate and the median's
own CI already account for.

**The bias case is real, and it lives between rows.** Two rows measured minutes
apart can each be internally fine and still average different P-states, and in
the DRAM-resident regime that is a 20% bandwidth difference landing entirely on
one side of a ratio. `audit_claims.py` now checks exactly that, and on the old
data it fires on four claims — including the two carrying the project's headline
conditional:

```
attribution.quant_effect.cold.4b.ctx8192   fused 9144 MHz vs control 10287 MHz (12%)
attribution.quant_effect.cold.4b.ctx16384  fused 9770 MHz vs control 10287 MHz  (5%)
attribution.quant_effect.cold.2b.ctx8192   fused 9001 MHz vs control 10287 MHz (14%)
attribution.quant_effect.cold.2b.ctx16384  fused 9858 MHz vs control 10287 MHz  (4%)
```

The direction deserves stating precisely rather than dramatically. The ratio is
`control_time / fused_time`, and it is the **control** that ran at the higher
memory clock — so the control was faster than it would have been alongside the
fused kernel, the numerator is smaller, and the reported quantization benefit is
**understated** rather than inflated. The long-context sign flip is not an
artefact of this. But "the bias happens to point away from my thesis" is a fact
about one run, not a property of the measurement.


---

## 2026-09-01 23:45 — The re-run, which is the only thing that settles it

`benchmark.py --samples 50` against the bandwidth-aware ramp, 861 s.

**Quotable rows: 25/48 → 39/48.** Every remaining rejection is dispersion; there
are no clock rejections at all. Running the dispersion decomposition again:
**9 of 96 measurements fail, down from 25** — 4 drift, 2 wander, 3 white jitter.
The failing rows now pin their medians to ±1.60% against ±0.14% for the passing
ones.

**Audit: 68 claims, 28 TRUE / 18 CONDITIONAL / 10 MISLEADING / 12 FALSE**
(was 21 / 25 / 10 / 12). Seven claims moved from conditional to established, and
not one of them moved because a threshold was relaxed — the measurement got
better.

**The attribution, recomputed.** `×` is `fp16_sdpa / triton_fp16_control` for the
split and `triton_fp16_control / fused_triton_4b` for the quantization, so above
1.00 means the thing helps:

| ctx | split, L2 | split, DRAM | quant, L2 | quant, DRAM | all three rows quotable |
|---|---|---|---|---|---|
| 512 | 14.1× | 10.4× | 0.72× | 0.91× | **yes** |
| 2048 | 26.1× | 15.9× | **0.90×** | **1.17×** | **yes** |
| 8192 | 53.3× | 23.4× | 0.83× | 1.27× | no |
| 16384 | 68.3× | 24.8× | 0.63× | 1.36× | no |

The headline is unchanged and better supported: the win is the flash-decoding
split, and quantization is a rounding error on top of it whose *sign* depends on
whether the working set fits in L2. Two contexts now have every input passing the
gate simultaneously rather than one, and ctx=512 says something the old data
could not — at short context quantization loses in **both** regimes, which is a
cleaner statement of the conditional than "it pays above L2".

**What is left, and it is the last known bias.** The audit's memory-clock check
still fires on five DRAM-resident claims, because the methods are measured in
sequence and rows minutes apart average different P-states:

```
cold.4b.ctx512    fused 10430 vs control 11358 MHz   (9%)
cold.4b.ctx2048   fused 11358 vs control 11001 MHz   (3%)
cold.4b.ctx8192   fused  9934 vs control 11001 MHz  (11%)
cold.2b.ctx512    fused 10068 vs control 11358 MHz  (13%)
cold.2b.ctx2048   fused 11334 vs control 11001 MHz   (3%)
```

Note the direction is not consistent: at ctx=2048 the *fused* kernel had the
faster memory, which inflates the 1.17×; at 8192 the control did, which deflates
the 1.27×. So this is not a bias with a convenient sign, it is a bias with no
sign at all — which is worse, because it cannot be argued away in either
direction. The fix is structural: measure the methods **interleaved** rather than
in sequence, so a slow drift in clock state lands on all of them equally.


---

## 2026-09-02 00:20 — Interleaving the methods, measured and thrown away

**Attempted:** the last known bias. The audit's between-row check kept firing on
the DRAM-resident quantization claims: two rows that get divided by each other had
averaged different memory P-states, up to 13% of bandwidth on one side. The
obvious cause is that the methods are measured one to completion, so the two rows
in a ratio are minutes apart, and the obvious fix is to measure them round-robin
so any slow drift lands on all of them equally.

Implemented as `--passes`, with the per-pass sample series pooled and the per-pass
clock windows merged (`merge_clock_records`, combining verdicts with `all` rather
than averaging — one throttled pass is a throttled measurement). Full run at two
passes: **1520 s against 861 s.**

**It bought nothing.**

| | sequential | interleaved |
|---|---|---|
| quotable rows | 39/48 | 39/48 |
| ratios with > 3% memory-clock mismatch | 12/20 | 14/20 |
| median mismatch | 5.5% | 4.9% |
| DRAM-resident median IQR | 1.53% | 1.49% |

**The number that explains why** is the agreement between a row's own two passes:
a median of **0.14%**. There is no slow drift for interleaving to average away.

And then the check that settles it. Comparing the two independent full runs, a
method's mean memory clock reproduces to a median of **86 MHz out of ~10,500** —
0.8%, with only 3 of 48 rows moving more than 400 MHz. The memory clock a row
runs at is a property of *the method and context*, not of when it was measured:

```
triton_fp16_control       11169 MHz        fused_fold_zp_4b       10488 MHz
fused_triton_2b           10919 MHz        dequant_sdpa_eager_4b  10477 MHz
fused_triton_4b           10773 MHz        fused_gather_meta_4b   10193 MHz
fp16_sdpa                 10624 MHz        fused_gather_meta_2b   10163 MHz
```

On a power-shared 80 W part **the kernel is one of the things that sets the
clock**. A bandwidth-hungry baseline pulls the memory clock up; a 5 µs kernel does
not. So "hold the clock fixed and vary only the kernel" is not available here
without the administrator rights `nvidia-smi -lgc` needs, and no amount of
scheduling substitutes for it.

**What is actually being compared, then**, is kernel A at the clock A induces
against kernel B at the clock B induces. For a latency question that is arguably
the honest comparison — it is what a user gets — but it is not a controlled
experiment, and the audit now says which way each instance leans instead of only
that it exists. The systematic part is worth stating plainly: `triton_fp16_control`
has the **highest** mean memory clock of any method, in both runs. It is the
denominator's competitor in every quantization ratio, so this bias makes the
reported quantization benefit **understated** — 4 of the 5 flagged claims lean
that way.

`--passes` defaults back to 1. The flag stays so the negative result can be
re-run, and because someone on a desktop card with pinnable clocks may find it
does something there.

**One reporting bug found on the way.** The rejection line printed the *cold* IQR
whatever failed, so a row rejected for its L2-resident dispersion was reported as
`IQR 1.1% TOO-NOISY` — a number that comfortably passes the gate it was being
blamed for. It now names the regime that failed.

**Where this leaves the run.** The interleaved run is kept as the current
`results/benchmark.json` (`passes: 2` is recorded in it). 39/48 quotable, and the
audit reads **68 claims: 29 TRUE / 17 CONDITIONAL / 10 MISLEADING / 12 FALSE**.

One thing not to gloss over: the DRAM-resident quantization ratio at ctx=8192
moved **1.27× → 1.47×** between the two runs, because `fused_triton_4b@8192` is
one of the three rows whose memory clock did shift between them (9934 → 11001
MHz). The bootstrap CI on either number is ±0.01. The CI is describing sampling
noise inside one run and says nothing about which P-state that run happened to
land in, and the long-context numbers should be read with that gap in mind.

---

## 2026-09-02 — the interval that was never measured

The last entry ended on a number that did not have an interval: the DRAM-resident
quantization ratio at ctx=8192 had read **1.27×** in one run and **1.47×** in the
next, on bootstrap CIs of ±0.01 each. That gap is not a statistical subtlety, it
is a category error — every CI in `audit_claims.py` resamples the timings of a
*single run*, so it answers "how much would this move on another 50 samples from
this window" and cannot answer "how much does it move if the process exits and
the card lands in a different memory P-state". Every verdict in this repo turns
on whether an interval clears a bar, and the interval being used was the wrong
one.

The fix is not clever, it is expensive: run the whole benchmark again, more than
once, and look. Three back-to-back runs at `--samples 50 --passes 1`, 807 s /
773 s / 775 s, nothing changed between them.

### What came out

`between_run.py` compares N complete runs on 60 tracked ratios, reporting for
each: every run's point estimate and CI, the union of those CIs, the **inflation
factor** (union width ÷ median single-run CI width), and whether the *verdict*
moved.

| | n | median inflation | median between-run spread | worst spread |
|---|---|---|---|---|
| passed the gate in every run | 22 | **2.4×** | 0.7% | 2.0% |
| failed it in at least one | 38 | **5.0×** | 2.9% | 44.0% |

**No verdict changed, on any of the 60 ratios.** The honest interval is about
2.4× the printed one — the CI was too narrow, but by a factor, not by the order
of magnitude the 1.27×/1.47× pair implied. The conditional finding survives
intact: quantization costs 0.717–0.813× L2-resident at every context, and pays
1.188–1.197× at 2k, 1.469–1.478× at 8k, 1.416–1.469× at 16k.

### Three things that were not the question

**The gate scores well out of sample.** It is applied inside a run and knows
nothing about the other two, yet the rows it rejects are exactly the rows that
move when the benchmark is re-run: 5.0× against 2.4× inflation, 2.9% against 0.7%
spread. Every previous argument for the gate was internal to a run. This one is
not, and it is the third time the rule "fix the measurement rather than widen the
gate" has paid.

**The P-state story is now measured.** Across the 48 DRAM-resident rows, the
correlation between a row's between-run movement in time and its between-run
movement in mean memory clock is **r = +0.71**. That had been asserted from a
mechanism argument and two anecdotes; it is now a number. It also leaves about
half the variance unexplained, and nothing here identifies what that is.

**The 1.27× was a tail event, not the typical spread — which is worse news.**
`fused_triton_4b@8192` sat at 11001 MHz in all four runs since; the 9934 MHz
window has not recurred, and the three new runs agree to 0.6% at that cell. So
the run-to-run distribution has a body of about ±1% and a tail that moves a
headline ratio by 15%. **Three runs measure the body and say nothing about the
tail**, and the report says so in those words rather than presenting the union of
three runs as a bound.

A fourth, smaller finding: **quotability is itself a random variable.** The
per-run counts were 42, 41 and 39 of 48; 35 rows pass in all three and 46 in at
least one. So 11 rows are starred or not depending on the run. A star has been
read in this repo as a property of the kernel, and it is a property of the run.

### What changed in the code

- `between_run.py`, new. Refuses to pool runs whose configurations differ, since
  averaging a 50-sample run with a 10-sample one would put a number in the report
  that no run produced.
- `audit_claims.py` loads `results/between_run.json` when it exists. Every
  per-context claim prints its run-to-run interval next to its CI; a ratio whose
  verdict moved between runs is downgraded automatically (`TRUE` →
  `TRUE BUT CONDITIONAL`, `FALSE` → `MISLEADING`); and a new claim,
  `method.between_run_spread`, audits the audit's own intervals. With no
  between-run data present it reads **MISLEADING**, which is the state every
  previous version of this repo was in.
- `test_between_run.py`, new: 16 CPU-only tests against synthetic runs with known
  answers — identical runs must not manufacture spread, a 20% shift must read as
  20%, a shift across the bar must register as a verdict change, and the pooling
  guard must be an error rather than a warning. One of them caught a formatting
  bug immediately: at two decimals a real 0.005× spread on a 0.73× ratio printed
  as "0.73x-0.73x".
- `make_session_plots.py` grows `between_run_spread`, two panels split by gate
  status. The x scales differ by an order of magnitude, and that gap *is* the
  gate's out-of-sample score.

### Housekeeping

`results/benchmark.json` is now **run 3 of the 3**, by the rule "the last one" —
the only selection rule that cannot be gamed after the fact, and here also the
least flattering (39/48 quotable against 42 and 41). The interleaved run it
replaced is kept as `results/benchmark_interleaved.json`.

Two claims in the README were stale against the new run and have been corrected
rather than left: the sign flip is fully clock-verified at **ctx=8192 in all
three runs**, not at ctx=2048 (which clears the gate in one run of three); and
"the 8k and 16k SDPA baselines are not clock-verified" is no longer true — the
bandwidth-aware ramp fixed it, and those rows now pass with a timing IQR of
0.1–0.4%.

Audit: **69 claims — 26 TRUE / 20 TRUE BUT CONDITIONAL / 11 MISLEADING /
12 FALSE.** 106 kernel tests and 16 between-run tests pass.

---

## 2026-09-02 (later) — the cheap denominator, and why there isn't one

The between-run study left one thing open, and it was the honest kind of open: three
runs bound the *body* of the run-to-run distribution and say nothing about its
tail. The 1.27x that started the whole exercise never reappeared in four
subsequent runs. A tail rate needs a denominator, and a denominator needs many
runs, and a full run is 13 minutes.

So the plan was a fast path: `benchmark.py --methods attribution` times only the
three rows the conditional is built from (`fp16_sdpa`, `triton_fp16_control`,
`fused_triton_4b`) and skips the other nine. Filtering happens *after*
`build_cases`, so every replica is still allocated and the GPU sits in the same
memory state — the intent was a faster run of the same experiment. 210 s against
775 s.

**It is not the same experiment, and the validation said so before it was used
for anything.** Three subset runs against the three full ones:

| ratio | ctx | full runs | subset runs |
|---|---|---|---|
| `quant_cold` | 8192 | 1.469–1.478 | **1.277–1.445** |
| `split_only` | 8192 | 22.585–23.046 | **23.299–23.710** |

Two of twelve ratios have ranges that miss the full-run range entirely, and the
subset spread at the headline cell is 13% against 0.6%. Not a small effect on a
number whose whole interval is 0.6% wide.

**Nothing in the telemetry explains it.** SM clocks agree to 0.4%, mean power to
0.3 W, mean temperature to 1 °C, sample counts per window are identical, and at
ctx=8192 cold the memory clock is 11001 MHz in both. The rows still differ:
`triton_fp16_control@8192` reads 31.4–32.1 µs against 32.6–32.8. Whatever the
mechanism is, the clock monitor cannot see it, which is worth stating plainly
rather than filing under "noise".

### The excursion the study was looking for, on demand

`sub3` produced **1.277x** at ctx=8192 — the historical number, to three
decimals — with `fused_triton_4b@8192` sitting at **10334 MHz** instead of 11001.
That is the same mechanism recorded when the 1.27x first appeared (9934 MHz
then). The excursion is real, reproducible, and evidently made *more likely* by
shortening the run.

`clock_excursions.py` puts a rate on it. Across six runs it takes each
(method, ctx, regime) cell's median memory clock and flags every observation
sitting ≥3% below it:

| group | runs | observations | excursions | rate | DRAM-resident ones |
|---|---|---|---|---|---|
| full | 3 | 72 | 2 | **2.8%** | **0** |
| subset | 3 | 72 | 8 | **11.1%** | 1 |

The last column is the one that matters. A memory P-state drop only costs time
where the measurement is bandwidth-bound, and **in three full runs there were no
DRAM-resident excursions at all**. That is why the three full runs agree to 0.6%
at the cell that once read 1.27x: sustained work holds the clock up, and the
shipped protocol supplies sustained work. Take three quarters of it away and the
excursions come back.

**The gate is not a P-state filter, and should not be described as one.** It
rejected 4 of 10 excursions — including, usefully, the one that produced the
1.277x. But it tests the SM clock and the timing's own dispersion and never
looks at the memory clock, so it catches an excursion only through the
dispersion the excursion happens to cause. A row that sits steadily in a lower
P-state all window has a tight IQR and passes. (Gating on memory-clock stability
was measured and rejected months of work ago, because it discards every
DRAM-resident row.)

### What this changes

`--methods` stays. It is honest about itself — the JSON records it and
`between_run.py` refuses to pool a filtered run with a full one — and it has
turned out to be a good *excursion generator*, which is a more useful thing than
the fast path it was written to be. What it cannot do is stand in for a full run,
and the docstring now says so.

The open item it was meant to serve is closed differently than expected. The
tail rate under the shipped protocol is not "unknown pending ten more runs"; it
is 0 of 72 DRAM-resident observations over three runs, with a mechanism that
explains both the zero and the historical exception. Ten more runs would tighten
that bound. They would not change what may be said today.

### One near-miss worth recording

The first version of `clock_excursions.py` used the *mode* as each cell's
baseline, since P-states are discrete and the mode names the state a cell
normally sits in. On several cells all six observations are distinct, so every
count ties at one, and the tie-break toward the highest clock reported **4 of 6
observations as excursions** against a baseline one run reached once — an
"excursion rate" of 67% on a cell that was merely drifting. Switching to the
median fixed it and dropped the totals from 15 excursions to 10.

Caught by reading the output table and finding a 15.0% "drop" that looked
implausible, not by a test. There is now a test named for it.

`clock_excursions.py`, `--methods`, and eight new tests (24 total in
`test_between_run.py`). 106 kernel tests pass.

---

## 2026-09-02 (late) — the protocol is a variable, and bandwidth says which rows care

The previous entry left the protocol gap named but not explained, and then a
correction had to be made to it: the claim that no telemetry accounted for the
gap was wrong, because the power figure being compared was a *whole-run* average.
On an 80 W part pinned at its limit most of the time that average is flat by
construction. Per row, `compare_protocols.py` found **r = −0.57** between a row's
power shift and its time shift. Better, but only a third of the variance.

### The experiment

`--preload SECONDS` tests one prediction. If the gap is *total* sustained load,
a short run given 300 s of saturating work beforehand should converge on the long
run. Position within a run was already ruled out for free: the interleaved run in
the tree moves a row's mean memory clock by a median of **+0.00%** between its two
passes, direction split 15/48 up.

Three runs per protocol. **The prediction was wrong.** Preloading did not move the
short runs toward the full runs — it moved them further in the same direction.

| protocol | fp16 control @8k | power | `quant_cold` @8k |
|---|---|---|---|
| full (12 methods/ctx) | 32.61–32.78 µs | 74.2–74.6 W | 1.469–1.478 |
| subset (3 methods/ctx) | 31.42–32.14 µs | 75.4–76.8 W | 1.277–1.445 |
| subset + 300 s preload | 30.27–31.13 µs | 74.6–76.9 W | 1.395–1.402 |

SM clock 2761–2772 MHz in all three; memory clock 11001 MHz in all three. The
control still moves 7.3%.

### What does predict it

Achieved bandwidth — each row's own DRAM-resident bytes over its own
DRAM-resident time:

| row | achieved GB/s | shift under preload |
|---|---|---|
| `triton_fp16_control` @8k | 257 | −7.3% |
| `fused_triton_4b` @8k | 118 | −2.2% |
| `fused_triton_4b` @512 | 32 | +0.0% |
| `fp16_sdpa` (all contexts) | 11–12 | +0.0 to −0.3% |

**r = +0.84** over 12 rows. A row that barely touches DRAM cannot care what state
the memory subsystem is in; a row that saturates it is entirely at its mercy.
That is the predictive form — which rows a protocol change will move can now be
said in advance rather than discovered.

And it names the exposed *ratios*. The quantization ratio divides the
highest-bandwidth row in the benchmark (the control, 257 GB/s) by a much lower one
(the fused kernel, 118 GB/s), so it inherits the whole difference.
`speedup_vs_sdpa` divides an 11 GB/s row by a 118 GB/s row and moves ±1.6%. The
fp16 control exists to make the comparison fair, and the property that makes it a
good control — same algorithm, 4× the bytes — is exactly what makes it the most
protocol-sensitive row in the file.

### What it costs

**The shipped protocol reports the most favourable number of the three.**
`quant_cold@8192` is 1.475 full, 1.422 subset, 1.397 preloaded, against a 0.6%
between-run spread within the full protocol. The honest range for that cell across
everything measured is **1.28–1.48**, not 1.469–1.478. That is a flattering bias
larger than the interval this repo has been quoting, and the README now says so in
those words.

**No verdict changes.** `quant_cold@8k` is ≥1.39 under every protocol,
`quant_hot` ≤0.82 under every protocol, and the L2-resident half is nearly
protocol-immune (+2.3%) for exactly the reason the bandwidth law predicts:
neither of its rows pulls much DRAM bandwidth. The conditional survives; its
magnitude is softer than the CI suggested.

So "hold everything fixed and vary only the kernel" now has a third thing that
will not hold still. The clock was the first, the power state the second, the
measurement schedule the third — and unlike the other two the schedule is fully
under the experimenter's control, which makes it the one worth reporting rather
than lamenting.

### Code

`compare_protocols.py` is new: labelled run groups, each ratio's range within its
group, a flag when a group's range misses the reference's entirely, a per-row
telemetry comparison, and the bandwidth-versus-sensitivity table with its
correlation. Deliberately blunt — with three runs per group the honest statement
is "these ranges do not touch", not a test pretending to more resolution than
three points support. It reproduced the earlier ad-hoc subset-vs-full result
exactly before being used for anything new.

`benchmark.py --preload SECONDS`, recorded in the JSON; `between_run.py` refuses
to pool runs that differ in it, because it is the variable under test.

`test_between_run.py`: 25 → 32 tests. 106 kernel tests pass. Audit unchanged at
69 claims.

## The fourth protocol: a pre-registration (2026-09-02, later)

Written and committed **before** the runs it describes, so the prediction cannot
be fitted to the result afterwards. The data does not exist at the time of this
commit; `compare_protocols.py` will say so, naming the missing cell.

### The confound

Three protocols exist, and two things vary together across all of them:

| protocol    | methods timed | preload | wall  |
|-------------|---------------|---------|-------|
| `full`      | 12            | 0 s     | ~800 s|
| `subset`    | 3             | 0 s     | ~205 s|
| `preloaded` | 3             | 300 s   | ~499 s|

Every protocol that times more methods is also a longer run. So the two
available explanations for the protocol shift — "the card is in a different
state because 800 s of measured work preceded this row" and "the card is in a
different state because a lot of bandwidth was pulled recently" — cannot be told
apart by anything measured so far. `--preload` was built to test the second and
gave a result that fits neither cleanly: it moved the subset runs *further* from
the full runs rather than toward them.

The missing cell is 12 methods **with** the preload. Adding it makes the design
a complete 2×2 and the two factors separable:

|             | no preload  | preload     |
|-------------|-------------|-------------|
| 3 methods   | `subset`    | `preloaded` |
| 12 methods  | `full`      | `fullpre` ← new |

`fullpre` is `benchmark.py --samples 50 --preload 300`, three runs, identical to
`full` in every other respect.

### Predictions, at `quant_cold@8192`

`full` reads 1.469–1.478 there (median 1.475) and a protocol's three runs agree
to 0.6%, so these outcomes are distinguishable rather than rhetorical.

- **H1 — run length is the channel, and the preload is inert once a run is long
  enough.** `fullpre` ≈ 1.475, within the between-run spread of `full`.
- **H2 — the two factors are separable and additive in log space.**
  `fullpre` ≈ 1.475 × (1.397/1.422) ≈ 1.449, i.e. −1.8%, with an interaction
  term near zero.
- **H3 — recent saturation dominates and swamps run length.** `fullpre` ≈
  `preloaded` ≈ 1.397, −5.3%, with the main effect of method count near zero.

Secondary, also fixed in advance: the bandwidth correlation is recomputed with
the preload main effect alone, and is predicted to stay above +0.6. If the
bandwidth law is about the memory subsystem's state rather than about run
length, it should attach to the preload factor and not to the method-count one.

No prediction is offered for `fused_triton_4b@16k`, which fits neither the power
story nor the bandwidth one and is on the open list for that reason.

### What each outcome would mean

H1 says the shipped protocol's favourable reading is a property of it being
*long*, and no preload can substitute for that. H3 says the shipped 1.475 is an
artefact of not having recently saturated the memory system, and the honest
centre of that cell is nearer 1.40. H2 says report the interval as the span of
the whole 2×2 rather than of any one protocol. All three are publishable; H1 and
H3 are the ones that would change what the README says.

### Code, in this commit and before the data

`compare_protocols.py` grew the 2×2: `design_cells` reads each run's protocol
coordinates out of its own recorded `args` rather than off its `--label`, so a
mislabelled run is a detected error and not a silently wrong cell; it refuses
anything that is not a complete 2×2 and names the cell it is missing.
`factorial_effects` reports both main effects, the four simple effects and the
interaction, all as differences of logs — the only scale on which "the two
factors add" is a well-posed claim — with the largest within-cell range carried
alongside as the yardstick for calling an effect resolved. No p-values, for the
same reason as the rest of that file.

One bug fixed on the way, and it is the kind this repo keeps finding in its own
apparatus rather than in the kernel: `bandwidth_sensitivity` compared the
reference against *whichever group came last on the command line*. Adding a
fourth protocol would therefore have silently repointed the published r = +0.84
at a different pair of runs while the text around it still claimed to describe
the old one. The compared group is now explicit and the script computes one
correlation per protocol. Re-run on the existing three, it reproduces
`preloaded` vs `full` at r = +0.84 exactly, and reports `subset` vs `full` at
r = +0.85 — a pair that had never been scored, and which strengthens the law by
holding on it too.

`test_between_run.py`: 32 → 47 tests. The new ones pin the log-space arithmetic
against hand-built cells (additive factors give zero interaction; an effect
confined to one level shows up as one; a doubling and a halving cancel), the
design reader against mislabelled and mixed groups, and the "resolved" rule
against an effect smaller than its own cell noise. One of them caught a real
shadowing bug: `render` already used `cells` as a local for table rows, so the
new parameter of that name was clobbered before it reached the section that
needed it.

## The fourth protocol: the answer is H3 (2026-09-02, night)

The runs are in. The pre-registration above was committed as `a00d747`, before
any of this existed, and the prediction it named is the one that happened.

### The 2x2, at `quant_cold` ctx=8192

|             | no preload        | 300 s preload     |
|-------------|-------------------|-------------------|
| 3 methods   | `subset` **1.4217** | `preloaded` **1.3968** |
| 12 methods  | `full` **1.4755**   | `fullpre` **1.3944**   |

Predicted: H1 1.4755, H2 1.4496, **H3 1.3968**. Observed: **1.3944** — 0.2% from
H3, 3.8% from H2, 5.5% from H1.

**Recent saturation dominates; run length does not.** The simple effects say it
more precisely than the headline does:

| effect | value |
|---|---|
| preload, at 3 methods | −1.8% |
| preload, at 12 methods | **−5.5%** |
| method count, with no preload | **+3.8%** |
| method count, after the preload | **−0.2%** |

Read the last two together. Without the preload, going from 3 methods to 12 is
worth +3.8% — which is the whole of the original `full`-vs-`subset` gap and the
reason "longer runs read higher" looked like the explanation. **After the
preload, method count is worth −0.2%: it stops mattering entirely.** Saturating
the memory system for 300 s does everything that 800 s of preceding measurement
was doing, and then some. Run length was never the channel; it was a proxy for
how much bandwidth had recently been pulled.

So `full`'s 1.4755 is not what a long run reads. It is what a run reads when the
memory subsystem has *not* recently been saturated — and the shipped protocol is
the only one of the four in that state by the time it reaches this row.

### What this does not establish

The effects are **not resolved** against this file's own yardstick, and that has
to be said plainly rather than buried. `factorial_effects` calls an effect
resolved only when it exceeds the largest range any single cell shows across its
own three runs, and at this cell that yardstick is **13.2%** — set by `subset`,
whose three runs span 1.2770–1.4451. Every effect in the table is smaller than
that. By the conservative test, nothing here is resolved.

Both statements are true at once: the pre-registered point prediction was hit to
0.2%, and the conservative interval test does not clear it. The yardstick was
chosen before the data and is not being changed after it — that is the entire
value of having fixed it in advance. What *does* separate the protocols is the
test this file has used all along: **`fullpre`'s range misses `full`'s
entirely** at this cell, as do `subset`'s and `preloaded`'s.

The pre-registered secondary prediction also held: the bandwidth law survives
being repointed at the new pair, **r = +0.70** for `fullpre` vs `full`, against
+0.84 for `preloaded` and +0.85 for `subset`.

### The surprise: which protocols are noisy

The within-cell spreads at this cell are not what "more sustained load is
steadier" predicts:

| protocol | methods | preload | wall | spread | excursion rate | DRAM-resident |
|---|---|---|---|---|---|---|
| `full`      | 12 | 0 s   | 785 s  | 0.6%  | 2.8%  | 0 |
| `subset`    | 3  | 0 s   | 205 s  | 13.2% | 12.5% | 2 |
| `preloaded` | 3  | 300 s | 502 s  | 0.6%  | 2.8%  | 0 |
| `fullpre`   | 12 | 300 s | 1080 s | 8.4%  | 6.9%  | 1 |

The two tight protocols are the 785 s one and the 502 s one. The two noisy ones
are the *shortest* (205 s) and the *longest* (1080 s). That is not monotone in
load, and it is not monotone in temperature either — mean temperature on the
attribution rows runs 69.3 C (`subset`), 69.5 C (`full`), 70.8 C (`preloaded`),
72.1 C (`fullpre`), so the coolest protocol and the hottest are the two that
misbehave.

A two-mechanism reading fits — too little preceding work and the memory clock
never comes up, too much and thermal pressure pulls it back down — and it is
worth writing down as a hypothesis. It is **not** established here: four
protocols, one card, and a story with two free parameters is a description, not
a test. It is on the open list, not in the README.

### A methodological correction, and what it cost

Two `fullpre` runs were discarded and re-measured because CPU-heavy analysis
(numpy bootstraps, `pytest`) had been run *concurrently with the timing loop*.
The suspicion was that descheduling the submitting thread let the GPU idle into
a lower memory P-state, which would have forged exactly the signal under test.

The re-measurement settled it, and **the suspicion was wrong**: the discarded
`fullpre1` read 1.2783 at 10144 MHz, and the clean `fullpre3` reads 1.2901 at
10232 MHz. The contaminated pair sits inside the clean distribution. The
excursions are a property of the `fullpre` protocol, not of the contention. The
runs are kept in `results/tail/contaminated/` rather than deleted, because a
discarded measurement that turns out to agree is evidence about the discarding
rule. Re-running was still right: at the time the two could not be told apart,
and an hour of wall clock is cheap against publishing a number whose cause is
unknown.

### Code and record

`compare_protocols.py` renders the 2x2, the simple effects and a per-protocol
bandwidth correlation. `clock_excursions.py` now has four groups.
`audit_claims.py` gained `method.protocol_choice` — the counterpart to
`method.between_run_spread`, reading MISLEADING when the protocol comparison is
absent, and MISLEADING now because it is present and says the protocols
disagree. **70 claims: 26 TRUE / 20 CONDITIONAL / 12 MISLEADING / 12 FALSE.**
New figure `docs/plots/protocol_factorial.png`. `test_between_run.py`: 47 tests.

**What the repo should quote for `quant_cold@8192`: 1.28–1.48**, unchanged as a
range — but the centre of it has moved. Three of the four protocols put the cell
at 1.39–1.42. Only `full`, the shipped one, says 1.475.


## The second dispersion tier: a third verdict, not a wider gate

`next_steps.md` had carried this as open item 1 since the dispersion decomposition
diagnosed it, deliberately deferred: it touches how rows are reported, and
changing the instrument while four protocols were in flight was the exact
confound those protocols were built to measure. The protocols are done, so it
was safe to act.

**The fact.** The gate rejects a row when its per-sample IQR exceeds 5% of its
median. But IQR is a property of the *sample distribution* and the tables quote
the *median*, and on this part those two come apart badly. The L2-resident rows
are tens of microseconds long and jitter freely around a median that barely
moves. On run 3, eight of the ten rejected measurements pin their medians to
+-0.05-1.31% while being rejected for IQRs of 5.6-11.5%.

**The refused fix.** Widening `MAX_IQR_FRAC` would admit the two rejected
measurements that are genuinely badly pinned (+-2.33%, +-2.68%) along with the
eight that are not. Dispersion and precision are different questions; the fix is
to ask the second one separately.

**The bar, and why it is not a new free parameter.** A promoted row must pin its
median at least as well as *the worst number the gate already accepts* — on run 3
that is +-1.700%, from `fused_gather_meta_4b@512`, a row this repo already prints
with a star. The bar is read off the instrument's own accepted behaviour, per
run, so the tier cannot by construction admit a number less certain than one the
gate blesses. That is what makes it a report rather than a loophole. Across the
six full runs the bar lands at 1.43-1.96%, so it is a property of the instrument
and not of the run.

**Two restrictions that bite.** A clock-rejected row is never eligible — the gate
is not a P-state filter, so a clock failure is a question this tier cannot see.
And a promoted row is admissible *per claim*: each carries `min_effect_frac`,
five times its own median uncertainty, and a consumer is expected to check the
effect it is claiming against it. The attribution rows clear it by 4-25x.

**What it buys, measured across all six full runs rather than argued from one.**
The attribution chain (`fp16_sdpa`, `triton_fp16_control`, `fused_triton_4b`)
complete at each context, gate-only vs gate+tier-2:

| ctx | gate only | with tier 2 |
|---|---|---|
| 512 | 3/6 | **6/6** |
| 2048 | 2/6 | **6/6** |
| 8192 | 5/6 | 5/6 |
| 16384 | 2/6 | 4/6 |

The two contexts that carry the sign flip — 512, where quantization loses in both
regimes, and 2048, where the flip happens — go from a minority of runs to all
six. That is the qualifier `next_steps.md` wanted to remove: those contexts were
being reported as "clears in one run and not the others" for a reason that turns
out not to bear on the number quoted.

ctx=8192 is **unchanged**, which is the restriction visible in the data: the one
run where its chain is incomplete (`fullpre3`) fails because `fused_triton_4b`'s
median is pinned only to +-1.91% against that run's +-1.49% bar. The tier
declines it on its own merits rather than rounding it up.

**Promotion is a property of the run, exactly as quotability is.** No row is
promoted in all six full runs; the most any row manages is four. That is the same
finding as "quotability is a random variable", not a new problem, and it is why
the table above is stated over six runs instead of one.

**Code.** `dispersion_tier.py`, applied post-hoc — the raw per-sample timings are
already in every results JSON, so the tier applies retroactively to all six full
runs and the three subset runs, and `benchmark.py` is untouched. No re-run was
needed and no measurement changed. `test_between_run.py`: 47 -> **62** tests.


### The audit consumes the tier

The tier was computed but nothing read it, which is half a feature.
`audit_claims.py` now loads `results/dispersion_tier.json` (`--dispersion-tier`,
absent is fine) and the evidence lines distinguish two things they used to
conflate under a single `*`:

- `*` -- not usable: rejected, or pinned too loosely for *this* effect.
- `~` -- failed the per-sample IQR gate, but pins its median at least as well as
  the worst row the gate itself accepts, against an effect at least 5x that pin.

That distinction is the whole point. Before this, `optimization.meta_broadcast.4b`
starred ctx=512 and ctx=2048 identically to a row pinned to +-2.68%, when they
are pinned to +-0.11% and +-0.37% against effects of 15% and 16%.

**No verdict moved.** 70 claims became 71 -- the new one is
`method.dispersion_tier`, the companion to `method.dispersion_gate`, reading
MISLEADING when the tier file is absent in the same shape as
`method.between_run_spread` and `method.protocol_choice`. Counts:
**26 TRUE / 21 CONDITIONAL / 12 MISLEADING / 12 FALSE**. The four changed claims
(`meta_broadcast` and `zero_point_fold`, both bit widths) changed evidence only.

Two properties worth keeping:

- **It collapses.** With no tier file, `tier_mark` returns exactly the old
  two-way split and the audit reads as it did before any of this was written.
  That is a test, not a hope.
- **The effect floor is armed but does not currently bind.** No promoted row is
  being asked to support an effect smaller than 5x its own median uncertainty --
  every one clears comfortably. The guard exists for the claim that has not been
  written yet, and the fact that it is slack today is a statement about the
  claims, not about the guard.

One detail that was a bug for about a minute: the marker for a multi-row ratio is
the *worst* of its rows, and `max()` on the strings gets that backwards. Hence
`worst_mark`, and a test for it.

`test_between_run.py`: 62 -> **74** tests.


### A figure for it

`docs/plots/dispersion_tier.png` (`make_session_plots.py`). Left: the same
scatter as `dispersion_gate`, recoloured by the three-way verdict, with the gate
drawn as a vertical line and the calibration bar as a horizontal one. The two
lines are perpendicular, which is the argument in one picture -- the gate cuts on
per-sample IQR and every number these tables quote lives on the median-precision
axis. Right: the six-run chain coverage, gate alone against gate plus tier.

Colour is the *row's* verdict while position is the *measurement's*, so a
promoted point can sit left of the gate line: that measurement was fine and its
partner regime was not. A row is judged on its worse regime, and the figure shows
that rather than hiding it.

`_chain_coverage` is the function that produces the numbers quoted in the README
and in `key_numbers.md`, so it has its own tests rather than being trusted
because a figure looked right. `test_between_run.py`: 74 -> **77**.


## Is the bandwidth law a law, or a method label?

The protocol finding rests on **r = +0.84** between a row's achieved DRAM
bandwidth and how far a change of protocol moves it, pooled over 12 rows. Twelve
points, three methods, four contexts — and the three methods pull 11 / 88 /
214 GB/s on average. That is precisely the shape in which a **method** effect
poses as a bandwidth one: "the fp16 control moves more than SDPA" would be the
whole content, and "bandwidth" only the label on it. The correlation would be
real and the interpretation empty.

The test that separates them is to hold the method fixed and vary only the
context, which moves bandwidth 3–4× inside one kernel. `bandwidth_law.py` runs
it, and the law survives:

| protocol | `fused_triton_4b` | `triton_fp16_control` |
|---|---|---|
| `subset` | +0.605 | +0.920 |
| `preloaded` | +0.998 | +0.761 |
| `fullpre` | +0.939 | +0.426 |

**6 of 6 positive**, sign test p = 0.016. Each look is n=4 and settles nothing on
its own; that they agree is the evidence. `fp16_sdpa` is excluded and said to be
excluded — 11–12 GB/s at every context is no range at all, and a method that
cannot test the law should not be counted as though it had. Between-method means
are monotone under all three protocols, and leave-one-out never drops r below
+0.644, so the pooled figure does not rest on one point either.

The test was written so it could fail: `test_a_method_label_masquerading_as_a_law_is_caught`
feeds it data where bandwidth separates two methods perfectly and predicts
nothing inside either, and checks that the pooled r is high while the
within-method r is not. A decomposition that could not return that answer would
not be evidence when it returns the other one.

### Two corrections that fell out of it

**The named misfit was the wrong row.** `next_steps.md` had carried
"`fused_triton_4b@16k` still fits neither story" as an open item. Against the
fitted line that row is the **fourth-best fit of twelve** (mean |residual| 0.45
pp). The misfits are all four `triton_fp16_control` rows — 1.74 pp against 0.41
pp for everything else — worst at `@8k` (2.32) and `@16k` (2.28). Why the control
specifically is the open question now, and it is a better one: the control is the
only method reading unquantized fp16, so it is the only one whose bytes are not
what its context length suggests.

**"The clocks are identical across protocols" was true where it was measured and
false as a generalization.** `triton_fp16_control@8192`, DRAM-resident, reads
11001 MHz under all four protocols — that line needs no correction. But 14 of 24
measurement rows differ across the original three protocols (6 of 12
DRAM-resident), and 17 of 24 across all four. The row that matters is
`triton_fp16_control@16384`: highest bandwidth in the benchmark at 305 GB/s,
largest protocol shift here at +10.1%, the only row positive under all three —
and its memory clock is **11401 MHz under `full` against 11001 under `subset` and
`preloaded`**, a 400 MHz step its shift orders with (+10.1 / +3.9 / +1.05%). That
row is not an exception to the story; it is the one place the P-state channel is
visibly open.

It does **not** explain the control's misfit in general, and saying so is the
point: `control@8k` is the worst-fitting row of the twelve and its memory clock
is flat at 11001 MHz under all four. Two rows of the same method, one with the
clock moving and one without, and both misfit.

Nothing here re-measures anything — it all comes out of the JSON
`compare_protocols.py` had already written. `test_between_run.py`: 77 → **87**.


### The audit carries it

`method.bandwidth_law`, the fourth of these method claims after
`between_run_spread`, `protocol_choice` and `dispersion_tier`, and the same
shape: MISLEADING when `results/bandwidth_law.json` is absent, because a pooled
correlation over three methods pulling 11 / 88 / 214 GB/s does not by itself
distinguish a bandwidth law from a method label, and nothing then checks.

Present and holding, it reads TRUE BUT CONDITIONAL and lists all six
within-method looks, the leave-one-out floor, which method misfits, and that the
memory clock is not constant across protocols. The conditional is named rather
than implied: one card, one kernel family, four contexts per method.

The claim **reverts to MISLEADING if a single look disagrees**, and there is a
test that feeds it exactly that. A verdict that could only come out one way would
not be worth printing.

**72 claims: 26 TRUE / 22 CONDITIONAL / 12 MISLEADING / 12 FALSE.** No existing
verdict moved. `test_between_run.py`: 87 → **94**.


### And bandwidth is the best of the obvious predictors

The natural follow-up to "why does the control misfit" is "is bandwidth even the
right axis". `bandwidth_law.py` fits |shift| against seven candidates derivable
from the same file — bandwidth, its square and its log, bytes moved, log bytes,
time, log time. Bandwidth has the smallest residual spread under `preloaded`
(1.12 pp) and `fullpre` (1.02) and is second under `subset` (1.44). Bytes moved
(2.25 / 1.95 / 1.41), time and their logs are all clearly worse. **Nothing beats
bandwidth under every protocol.**

`GB/s squared` wins under `subset` alone — 0.95 against 1.44 — and loses under
the other two. On twelve points that is what a coin landing heads looks like, so
it is reported and **not adopted**; the report says so in those words.

The framing matters more than the table. Bandwidth was the *hypothesis*, fixed
before this comparison existed, so this is a robustness check and not a selection
procedure — picking the best of seven predictors on twelve points is precisely
how a spurious one gets chosen, and a table like this can manufacture a finding
if it is read the other way round.

So the control's misfit is not bytes moved, not footprint, and not time. It stays
open, and it is a better-posed question than the one it replaced.

The rival check **degrades rather than fails** when `base_ms` is absent: an older
`compare_protocols.json` gets the bandwidth-shape comparison and loses only the
time-derived rivals. There is a test for that. `test_between_run.py`: 94 →
**100**.


### How solid is "the control misfits"? Less than it looked

Two follow-ups, and the second cost the finding some of its strength.

**Is the straight line the problem?** If the true relationship curved, a linear
fit would dump its residuals onto the highest-bandwidth rows, which are all
control rows — the misfit would be an artefact of the fit. Fitting
`|shift| ~ GB/s**k` in log-log gives **k = 0.82 / 0.84 / 0.98**. Essentially
linear, so no.

**Is it just that the control's shifts are bigger?** A constant-variance fit
under-weights rows with large values, so a method with 3.13 pp mean shifts will
show larger absolute residuals than one with 1.15 pp for no interesting reason.
Relative residuals are not flat either (control 1.66, fused 0.69), so not that.

**But under the scale-free fit the control is worst in only 2 of 3 protocols**,
against all three under the linear fit. Under `fullpre` the fused kernel is worst.
So the misfit is real under the fit this repo uses and **not robust to the choice
of fit**, and with four rows per method it is weak evidence either way. The report
says exactly that rather than quoting the linear result alone.

Ruled out so far, for why the fp16 control fits worst: bytes moved, footprint
over L2, time, curvature, and shift magnitude. The memory clock does not explain
it either — `control@8k` is the worst-fitting row and its clock is flat at
11001 MHz under all four protocols. It stays open, better characterised than
before, and it is now clear it is a small effect measured on four rows rather
than a structural fact waiting to be named.

`test_between_run.py`: 100 → **105**.


## Dropping 2-bit would be a protocol change, not a tidy-up

Open item 4 has read "either implement per-channel keys and re-measure, or state
plainly that 2-bit is out of scope and stop benchmarking it" for some time, with
the second option carrying the tone of the cheap one. It is not.

2-bit is **5 of the 12 methods** timed at every context and **43% of the run's
measured wall clock** — 284 s of 661 s inside clock windows, against 775 s end to
end. Removing it takes the shipped protocol from 12 methods per context to 7.

Method count is not a free parameter here. It is one of the two factors in the
2x2, and at `quant_cold@8192` its simple effect with no preload is **+3.78%**
(3 methods 1.4217 → 12 methods 1.4755). A 7-method run sits between those
endpoints, on the axis the repo has already measured as moving its headline
number, in the direction that reads lower.

So "stop benchmarking 2-bit" means the headline cell must be **re-measured under
the new protocol**, not re-rendered from the old JSON. That is three clean runs,
not an edit.

None of which argues for keeping 2-bit. It argues that the option that looked
free is the one with a measurable price, and that the price is known because of
work already done. A third option is cheaper than both: keep timing it, leaving
the protocol untouched, and mark it in the tables as a research row rather than
a candidate — which is what `correct.2bit_usable` already says in prose and what
no table currently reflects.

This is a decision about scope rather than a measurement, so it is written down
here rather than taken.


## Temperature cannot be the mechanism, and the runs already said so

`next_steps.md` carried a two-mechanism *hypothesis* for the protocol spreads —
too little preceding work and the memory clock never comes up, too much and
thermal pressure drags it back down — and asked for "a temperature sweep at fixed
protocol" as the actual experiment. Two hours of wall clock. Before spending it,
the runs already recorded were asked, because every measurement window carries a
mean temperature and a mean memory clock.

The comparison has to be **within a cell**. Between cells, temperature and memory
clock both move enormously for unrelated reasons — `fp16_sdpa@512` and
`triton_fp16_control@16384` are different workloads — so a pooled fit on raw
values would mostly measure "these are different things". Every observation is
therefore expressed as a deviation from its own (method, ctx, regime, protocol)
cell mean.

**The slope is −30.4 MHz per degree C** (r = −0.143, n = 720, **2.0%** of the
variance in memory clock). The sign is the one the thermal story predicts. The
size is what settles it.

A memory P-state step on this part is 350–1100 MHz, and it is P-state changes
that cost DRAM-resident time. At 30.4 MHz per degree, moving one step needs
**11.5–36 degrees C**. The four protocols span **3.3 degrees** (subset 70.2 →
fullpre 73.4), which predicts **99 MHz** — under a third of the smallest step.

So temperature is not the mechanism at these temperatures, and the sweep is only
worth running if it deliberately induces **~12 degrees C or more** at fixed
protocol. The protocols do not produce that by themselves, which is exactly why
the four-protocol data could never have answered the question and why a sweep was
the right instinct. It just needs a much wider range than anyone was going to get
by varying the protocol.

Per protocol the slope is −37.3 (`full`), −32.2 (`fullpre`), −9.3 (`subset`) and
**+15.5** (`preloaded`) MHz/C — the last of the wrong sign, on 24 cells. That
disagreement is itself a reason not to lean on the pooled figure too hard.

Three caveats, stated in the generated report and none of which rescues the
mechanism: the fit is linear over a ±3.9 C window so the degrees-per-step figure
is an **extrapolation** and a threshold outside it would not show up here;
observations inside a cell come from repeated runs of the same measurement and
are **not independent**, so the reported t = −3.9 overstates the significance;
and temperature is a **window mean**, so a brief spike could throttle without
moving it.

The other arm of the hypothesis — too little preceding work and the clock never
comes up — is untouched by any of this and remains the live one. It is also the
arm the `subset` → `preloaded` comparison already supports (13.2% spread → 0.6%
after a 300 s preload).

The test that makes this worth trusting is
`test_between_cell_variation_cannot_leak_into_the_fit`: two cells differing
enormously in both temperature and clock with no relationship inside either,
where a naive pooled fit would report a strong slope and the within-cell fit must
report none. `test_between_run.py`: 106 → **113**.


## Pre-registration: the clock-ramp measurement, before it is run

`thermal_check.py` said settling the warm-up arm by repeating protocols would
take ~200 runs per protocol. That is the wrong experiment. The hypothesis is a
claim about a **time constant** — if a 205 s protocol is noisy because the memory
clock has not finished rising, the clock must take a meaningful fraction of 205 s
to rise. That is directly observable in minutes.

`clock_ramp.py`: idle the GPU 90 s, then apply the same DRAM-saturating load
`benchmark.py` ramps with (cache-resident GEMM alongside a DRAM-sized copy) for
180 s, sampling `nvidia-smi` at 10 Hz throughout. Output: seconds until the
memory clock reaches and **holds** (3 s) 99% of its loaded ceiling. Only loaded
samples define the ceiling — at idle this part reports 12001 MHz, higher than it
ever sustains under load, and targeting that is a bug `benchmark.py` already had
to fix.

**Written down before the run, and the decision rule with it.**

- **H1 — the warm-up arm survives.** The memory clock takes **≥ 20.5 s** (10% of
  the shortest protocol) to reach and hold its ceiling. A short protocol then
  really does spend a meaningful part of itself with the clock still rising.
- **H2 — the warm-up arm dies.** It arrives in **< 20.5 s**. A clock that is up
  and holding within a small fraction of the shortest protocol cannot explain
  that protocol behaving differently from one four times longer.

**The prediction is H2, and more specifically 1–4 s.** The reason is already in
the tree: `benchmark.py`'s own per-row ramp uses `MEM_SETTLE_SECONDS = 0.55` and
gives up after `MEM_MAX_WAIT_SECONDS = 2.5`, and it mostly succeeds rather than
timing out. A ramp that usually settles inside 2.5 s is not a ramp with a
200-second time constant.

If H2 lands, the `subset` → `preloaded` improvement (13.2% spread → 0.6%) still
needs an explanation, and it will have to be something other than "the clock had
not come up yet" — thermal steady state, or the power governor integrating over a
longer window, are the obvious candidates and neither is tested here.

The threshold, the hold rule, and both hypotheses are committed before the
measurement exists. This is the same discipline as the fourth protocol, which is
the only reason that result was worth anything.


## The result: H2, and the warm-up arm is dead

Pre-registered H1 ≥ 20.5 s / H2 < 20.5 s, prediction "H2, and specifically
1–4 s". **Observed: 0.4 s.** H2 by a factor of 50, and the point prediction was
wrong in the direction of caution — the ramp is 2.5× faster than predicted.

| | idle | sustained under load | time to reach and hold |
|---|---|---|---|
| memory clock | 405 MHz | **11001 MHz** | **0.4 s** |
| SM clock | 435 MHz | 2640 MHz | 7.4 s |

The memory clock is at its sustained operating point **0.4 s** after the load
starts — 0.2% of the shortest protocol in this repo. A 205 s run spends 99.8% of
itself with the memory clock already up. **The warm-up arm cannot explain why a
205 s protocol behaves differently from a 775 s one.** Both arms of the
two-mechanism hypothesis are now dead: the thermal one on effect size, this one
on time constant.

Two things worth keeping from the series:

- **The memory clock leads the SM clock by 18×.** Memory is at its sustained
  level in 0.4 s; the SM clock takes 7.4 s. That is the opposite of the
  intuition the original ramp bug came from — the first ramp drove the SM clock
  and assumed memory would follow, and in fact memory arrives first and by a
  long way.
- **There is a boost transient.** The memory clock runs at 12001 MHz for the
  first 6.5 s and then settles to 11001 for the remaining 96% of the window.
  Real, reproducible in the trace, and irrelevant at these timescales — but it
  is what caused the bug below.

### A correction to this measurement, found by reading the series

The first version of `time_to_ceiling` took the ceiling to be the **maximum**
over loaded samples. That made the 12001 MHz boost the target, so the reported
answer was "0.8 s to the peak" when the question is "how long until the card is
in the state a benchmark actually runs in". The summary looked fine; the series
did not. The ceiling is now the **median** of loaded samples, with the peak
reported separately alongside how long it lasted.

A second, smaller one in the same function: arrival was being searched only among
samples with utilization ≥ 50%, but the memory clock leaves idle at t = 0.41 s
while utilization is still climbing through 39%. Utilization decides what the
ceiling *is*; it should not decide when the clock got there.

Neither correction changes the verdict — 0.8 s and 0.4 s are both far inside H2 —
which is the only reason it is comfortable to report them. The raw series is
written into `results/clock_ramp.json` and `--from-json` re-runs the analysis on
it, so both corrections were applied **without touching the GPU again**.

### What this leaves open, and it is now sharper

`subset` → `preloaded` is a real improvement: spread at `quant_cold@8192` goes
13.2% → 0.6% and the excursion rate 12.5% → 2.8%, for 300 s of preload. That is
not in dispute. What is now excluded is the explanation: it is **not** that the
memory clock had not come up, because the memory clock comes up in 0.4 s.

Candidates that survive and are untested here: the power governor integrating
over a window much longer than the clock ramp; thermal steady state in the sense
of a settled fan curve rather than a temperature; and allocator or driver state
that a preload happens to warm. The measurement that would separate them is not
obvious, and inventing one is a better use of the next session than another
protocol repetition.

`test_between_run.py`: 118 → **124**, including a test that plants a boost
transient and requires the sustained level to be chosen over it.

## The slow process is the cooling loop, not the clock

The clock-ramp trace was recorded to answer one question and answered a second
one for free. Over 180 s of continuous saturating load, sampled at 10 Hz:

| | 0-10 s | 10-30 s | 30-60 s | 60-120 s | 120-180 s |
|---|---|---|---|---|---|
| power (W) | 72.8 | 79.8 | 79.8 | 79.8 | 79.9 |
| temperature (C) | 56.3 | 65.4 | 72.5 | 73.9 | 70.9 |
| memory clock (MHz) | 11322 | 11001 | 11001 | 11001 | 11001 |
| SM clock (MHz) | 2507 | 2689 | 2636 | 2624 | 2643 |

**Power pins at its limit in ~10 s and never moves again** — drift over the last
120 s is +0.03 W. That weakens "the power governor integrates over a long window"
as the explanation for what a preload does, which was one of the three candidates
left standing.

**Temperature does not settle for two minutes, and it overshoots.** Smoothed over
2 s it rises to a peak of **76.2 C at t = 71 s**, falls back to 71.0 C by
t ≈ 120 s, and is flat from there to 180 s. The SM clock mirrors it exactly —
2608 MHz at peak temperature, recovering to 2641 and holding. That is a cooling
loop with a lag: heat rises, the fan responds late, temperature comes back down
and settles.

So there **is** a slow process on this card, with a time constant around
**120 s** — 300x the memory clock's 0.4 s. It is the thermal/fan loop, it is
visible in temperature and in the SM clock, and it is not visible in the memory
clock at all (11001 MHz flat from t = 10 s).

That is the right order of magnitude to matter here. The shortest protocol is
205 s, so **a `subset` run spends its entire measurement inside the thermal
transient**; `full` at 775 s spends about 15% of itself there; and a 300 s
preload absorbs the whole transient, which is exactly the shape of "preloading
fixes the short protocol" (13.2% spread -> 0.6%).

### The prediction it makes, and the confound that blocks it

If the thermal transient is what hurts, the rows measured early in a run should
be the ones with poor timing dispersion. They are:

| quarter of the run | rows | dispersion failures |
|---|---|---|
| 1st (rows 0-11) | 24 measurements | 2 (8%) |
| 2nd (rows 12-23) | 24 | **7 (29%)** |
| 3rd (rows 24-35) | 24 | 1 (4%) |
| 4th (rows 36-47) | 24 | **0** |

Nine of ten failures are in the first half, none in the last quarter, and
correlation between position-in-run and IQR is r = -0.226: later is tighter.

**And it proves nothing**, because in this benchmark the context loop is the
outer one. Position 0-11 *is* ctx=512, 12-23 *is* ctx=2048, and so on. Measurement
order and context length are perfectly confounded in the default ordering, so
this table is equally well explained by "short contexts are noisier", which is
already known to be true for unrelated reasons (they are tens of microseconds
long).

### The experiment that separates them

Run the benchmark with the context order **reversed**. If dispersion failures
follow *position*, they move to ctx=16384 and ctx=8192. If they follow *context*,
they stay with ctx=512 and ctx=2048. One run, ~13 minutes, and the two
explanations make opposite predictions about which rows fail.

`benchmark.py --contexts` was added for this: it sets the order as well as the
content, records itself in the results, and `between_run.py` already refuses to
pool runs whose context tuples differ, so a reordered run cannot be mixed into
the canonical set by accident.

**Pre-registered, before the run exists:**

- **H-position.** The thermal transient is what hurts. Failures concentrate in
  the first half of the *reversed* run — i.e. on ctx=16384 and ctx=8192, which
  fail 1 of 48 measurements in the normal ordering. Predicted: **>= 4** failures
  among them, and ctx=512/2048 improve.
- **H-context.** Short contexts are simply noisier. Failures stay on ctx=512 and
  ctx=2048 regardless of when they are measured. Predicted: ctx=16384 stays at or
  near **0** failures, and the overall failure pattern looks like the normal
  ordering.

**The prediction is H-context**, and not by much: the L2-resident rows at short
context are 3-5 microseconds long, where a fixed amount of jitter is a much
larger fraction of the median, and that mechanism does not need position to
explain anything. H-position would be the more interesting result, which is a
reason to be suspicious of wanting it.

A mixed outcome is possible and is not a failure of the design: failures moving
partly is evidence for both mechanisms operating, and the split is the number to
report.

## The result: H-context, and position was a proxy all along

One reversed run, 790 s, 41/48 quotable, `results/tail/reversed1.json`.

| | context order | failures by context | by quarter of the run | corr(position, IQR) |
|---|---|---|---|---|
| normal | 512 → 16384 | 512:2 2048:7 8192:1 16384:0 | 2 / 7 / 1 / 0 | **−0.226** |
| reversed | 16384 → 512 | 16384:1 8192:1 2048:1 512:4 | 1 / 1 / 1 / 4 | **+0.231** |

**The correlation with position flipped sign while the association with context
stayed.** Failures sat on the short contexts when those were measured first, and
on the short contexts when those were measured last. Position was a proxy for
context and nothing more.

Pre-registered H-position wanted **≥ 4** failures on ctx=16384/8192 in the
reversed run; it got **2**. H-context wanted ctx=16384 to stay near zero and the
pattern to be unchanged; it got 1, and the pattern is unchanged. **H-context
lands**, which is the side that was predicted — worth saying, because H-position
was the more interesting result and wanting it was a reason to distrust it.

So the cooling loop's ~120 s time constant is real and does **not** show up as
timing dispersion by position. The short-context rows are noisy because they are
3–5 µs long, where a fixed amount of jitter is a large fraction of the median.
That mechanism never needed position to explain anything.

### Two by-products of the reversed run

**The L2-conditional survives a protocol it was never measured under.**
`quant_hot` is below 1 at every context (0.853 / 0.895 / 0.797 / 0.733 — the bits
cost, L2-resident) and `quant_cold` crosses 1 between 512 and 2048 (0.904 →
1.158 → 1.484 → 1.512). Same sign, same crossing point, a different measurement
order. This is the first time the conditional has been checked against a protocol
change that was not designed around it.

**But `quant_cold@16384` moved +4.9%, and it should be watched.** 1.442 in run 3
against **1.512** reversed, which is outside the 1.416–1.469 that three full runs
gave. It is not a rejected row: `triton_fp16_control@16384` is tier-2 in this run
(pinned to ±1.25%, floor 6.2%, against a 51% effect), so the number is usable.

The obvious reading is that this is the bandwidth law again. `control@16384` at
305 GB/s is the most protocol-sensitive row in the benchmark, and reversing the
order moves it from being measured *last* — after ten minutes of work — to being
measured *first*, before the machine has settled. `quant_cold@8192`, whose
control pulls 257 GB/s, moved +0.3%. The most exposed row moved most.

That is **one run**, so it is an observation and not a shift. Recording it as
such: context order is a candidate protocol variable, the 16k cell is where it
would show, and establishing it needs three reversed runs rather than one. The
headline `quant_cold@8192` is unaffected either way.

### Correction: the 16k observation does not replicate cleanly

A second reversed run (`reversed2`, 776 s, 42/48 quotable) was run before the
previous section's observation could harden into a belief, and it is worth
recording what it did to it.

| | `quant_cold@8192` | `quant_cold@16384` |
|---|---|---|
| run1 / run2 / run3 (normal) | 1.4766 / 1.4766 / 1.4801 | 1.3796 / 1.4246 / 1.4424 |
| reversed1 | 1.4844 | **1.5125** |
| reversed2 | 1.3174 *(not usable)* | **1.4113** |

**At 8192 the gate did its job.** `reversed2` reads 1.3174, far below anything
else here — and `fused_triton_4b@8192` is tier-3 in that run, so the ratio is not
usable and is not quoted. The usable reversed value is `reversed1`'s 1.4844,
which sits with the normal runs. This is the machinery working: the outlier was
caught by a rule written long before this experiment.

**At 16384 both runs are usable and they disagree by 7.2%.** 1.5125 and 1.4113,
against a normal three-run range of 1.3796–1.4424. So the earlier reading — "the
most protocol-sensitive row moved, consistent with the bandwidth law" — is *not
supported by two runs*. Both reversed values are at or above the normal maximum,
which keeps the direction alive, but the spread *within* the reversed protocol
(7.2%) is larger than the spread between normal runs (4.5%), and two points
cannot separate a shift from a noisy protocol.

The hedge in the previous section was the right one and is now the operative
statement: it was an observation, not a shift. A third reversed run is in flight;
whatever it says, the honest summary of the pair is **"the reversed protocol is
noisier, and the 16k cell has not been shown to move."**

Independent support for the noise half: `clock_excursions.py` over the two
reversed runs gives a memory P-state excursion rate of **7.8%** (15 of 192)
against **4.9%** (14 of 288) for the three normal runs.
