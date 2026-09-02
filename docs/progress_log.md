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
