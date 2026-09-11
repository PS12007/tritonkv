# Pre-registration: does the quantization effect follow L2?

Written 2026-09-10. **Committed before `l2_sweep.py` was run beyond the two-point
smoke tests disclosed below, and before any volunteer's results file exists.**
The commit that adds this file is the timestamp. Nothing in it is to be edited
after data arrives except the "Outcome" sections at the bottom, which are
appended, never rewritten.

The four-protocol experiment was pre-registered the same way and landed to 0.2%
of its prediction. The reason to do it again is the same: deciding what counts
as a result after seeing the data would throw away the only thing that makes a
single laptop's measurements worth anything to anyone else.

---

## The claim under test

The README's headline, in its current form: quantization **costs** ~1.23–1.37×
when the KV cache is L2-resident and **pays** ~1.19–1.51× once the working set
exceeds L2, from ctx=2048 up.

That statement names a mechanism, L2 residency, but it was never tested as one.
It was inferred from two regimes built on purpose at four contexts, and all four
fp16 caches (0.5–16.8 MB) fit in this card's 33.6 MB L2. So:

* in the "hot" regime **the fp16 control has never been measured outside L2**, and
* on one card, L2 size cannot be separated from everything else about that card.

Two experiments, one on this machine now and one across volunteers' machines.

---

## Part A — `l2_sweep.py` on this card (RTX 5060 Laptop, L2 = 33.55 MB)

**Design.** Time the fp16 control and the fused 4-bit kernel on a grid where the
fp16 cache is a fixed multiple of L2: `x = fp16 bytes / L2` ∈ {0.125, 0.25, 0.5,
0.75, 1, 1.25, 1.5, 2, 3, 4, 6, 8, 12}. Here that is ctx = 4096 … 393216. At each
point: both kernels, both regimes (hot = one cache replayed by CUDA graph; cold =
rotating replicas ≥ 3× L2), the benchmark's clock ramp, clock window and
dispersion gate, the fused config tuned at that context by launch-free timing,
and a correctness check against a chunked fp32 reference. The 4-bit cache is
`x / 3.2` of L2 at every point.

**Zones**, fixed in `l2_sweep.py` as `FIT_FRAC = 0.75`, `SPILL_FRAC = 1.5`:

| zone | condition | grid points here |
|---|---|---|
| A — both fit | fp16 ≤ 0.75 L2 | x = 0.125, 0.25, 0.5, 0.75 |
| margin | neither A nor B nor C | x = 1, 1.25, 3, 4 |
| B — fp16 spills, 4-bit fits | fp16 ≥ 1.5 L2 and 4-bit ≤ 0.75 L2 | x = 1.5, 2 |
| C — both spill | 4-bit ≥ 1.5 L2 | x = 6, 8, 12 |

Margin points are measured and shown, never scored. The margins exist because
nominal L2 is not all usable and nothing here knows the replacement policy.

**Predictions.** The statistic is `bootstrap_ratio_ci` (circular block, 10 000
resamples) over raw per-sample timings, ratio = control / fused, so > 1 means
quantization pays.

| id | prediction | decision rule | why the L2 story predicts it |
|---|---|---|---|
| **P1** | zone A: quantization costs | every zone-A point has hot-ratio CI **upper** bound < 1 | both kernels read L2; the fused kernel does strictly more work per byte |
| **P2** *(primary)* | the hot ratio rises through 1 near fp16 = 1× L2 | log-interpolated first crossing `x*` ∈ **[0.5, 2.0]**; no crossing, or already ≥ 1 at the first point, **fails** | the control's bytes leave L2; the fused kernel's are still 3.2× smaller |
| **P3** | zone B: quantization pays *more* than it does from DRAM | every zone-B point has hot CI lower bound > 1 **and** > the cold CI upper bound at the same ctx | control reads DRAM, fused reads L2 — a regime `benchmark.py` never produces on this card |
| **P4** | zone C: hot converges on cold | every zone-C point has \|hot / cold − 1\| ≤ **0.10** | "hot" is no longer hot for either kernel |
| **P5** | the hot ratio has a hump | the grid point with the largest hot ratio has x ∈ **[1, 4]** | rises as the control spills, falls back as the 4-bit cache spills too |

**What refutes the mechanism on this card:** P2 failing. P1 failing *at x = 0.75
only* would say the usable L2 is smaller than nominal. That is a correction to the
margin, not a refutation, and it will be reported as exactly that. P3, P4 and P5
are secondary: each can fail for a reason the L2 story does not rule out (for
example, if the replacement policy lets a working set of 1.5–2× L2 keep a large
hit rate, P3 weakens), and each failure will be reported with that reason
checked, not assumed.

**Point guesses, for the record and not scored:** crossing at `x*` ≈ 1.0
(ctx ≈ 32k); zone-B hot ratio ≈ 1.8–2.2 (the benchmark's own
control-DRAM ÷ fused-L2 at ctx=8192 is 32.8 / 17.1 = 1.92); zone-C hot ratio ≈
the cold ratio ≈ 1.4–1.5.

**Smoke tests already run, disclosed in full.** Two runs of `--max-points 2`,
which measure x = 0.125 and 0.25 only (zone A, ctx 4096 and 8192, a region
`benchmark.py` already covers):

| run | fused config chosen | hot ratio @4096 / @8192 | cold ratio @4096 / @8192 |
|---|---|---|---|
| 1 — benchmark's tuner procedure | 64/4/2 (mistuned) | 0.749 / 0.722 | 1.022 / 1.130 |
| 2 — launch-free tuner | 32/2/2 | 0.813 / 0.820 | 1.159 / 1.336 |

Run 1 exposed a tuner fault: `block_n=64` was chosen, and an A/B with the clocks
ramped put it 26% slower DRAM-resident and 15% slower L2-resident than
`block_n=32`, over three rounds. The fix is in `tune_fused`'s docstring. No
point at x ≥ 0.5 has been measured by anything. Both smoke outputs live in the
session scratchpad, not in `results/`.

**Protocol caveat, stated in advance.** The sweep times 2 kernels, not 12.
`compare_protocols.py` measured method count as worth up to +3.8% at the
headline cell, so the sweep's numbers at shared contexts are not expected to
match `benchmark.py` to better than ~5%. No prediction above turns on an effect
that small.

---

## Part B — across volunteers' GPUs (`docs/RUN_ON_YOUR_GPU.md`)

Each volunteer sends `results/benchmark.json` and, if they have eight more
minutes, `results/l2_sweep.json`. Both files are self-describing: GPU name,
L2 size, SM count, driver, and (from 2026-09-10 on) a measured DRAM bandwidth.

### What must hold on every card

| id | prediction | data | decision rule |
|---|---|---|---|
| **C1** | the DRAM half: quantization pays at long context | `benchmark.json`, `quant_cold` at ctx 8192 and 16384 | audit verdict TRUE (CI low > 1.05 with the audit's margin rule) at **both** contexts |
| **C2** | the L2 half: quantization costs while everything fits | `benchmark.json`, `quant_hot` at every context whose fp16 cache ≤ 0.75 L2 | CI upper bound < 1 at every such context |
| **C3** | the sweep's shape | `l2_sweep.json` | P1–P5 as in Part A, same thresholds, scored per card |
| **C4** *(the transfer test)* | the crossing moves with L2 | every card's `x*` from C3 | every `x*` ∈ [0.5, 2]; and for any two cards whose L2 differs ≥ 3×, their `ctx*` ratio is within a factor of 2 of their L2 ratio |

C4 is the reason for all of this. A story in which quantization simply "pays
past context N" predicts the same `ctx*` on every card. The L2 story predicts
`ctx*` ∝ L2, i.e. `ctx*` ≈ L2 / 1024 bytes for this model. A 6 MB-L2 card should
cross near ctx ≈ 6k and a 72 MB card near ctx ≈ 70k, against ~32k here.

### What the size of the effect should track

**M1.** In the DRAM-resident regime the control is bandwidth-bound and the fused
kernel is mostly load-issue-bound (established here: 16× fewer metadata loads
bought 1.29×, 14% fewer instructions bought nothing). So `quant_cold` at long
context should scale with the card's **compute-to-bandwidth balance**:

    balance = SM count × max SM clock (GHz) / measured copy bandwidth (GB/s)

This card: 26 × 3.09 / 342 = **0.235**, `quant_cold@16384` = 1.42–1.47.
Prediction: across cards, `quant_cold@16384` rank-correlates **positively** with
balance, and L2 size adds nothing once balance is known (the DRAM regime is
built relative to each card's L2, so L2 should not enter it).

**The risk this makes explicit, before any data:** a low-balance card fails C1
first. By spec sheet an RTX 3090 is ~82 × 1.7 / ~850 ≈ 0.16. If quant_cold
scaled in proportion, that is ≈ 1.0 at 16k, which would fail C1. **If that
happens, C1 fails and the README's DRAM-half claim is restated as
card-dependent.** It does not get reclassified as a success for M1. Both
statements get reported: the prediction that failed and the model that
anticipated it.

With fewer than five cards M1 is descriptive: a rank correlation over three
points can only be ±1, ±0.5 or 0. It will be reported with its n and without a
p-value.

### No prediction, reported anyway

* **The size of the L2-resident cost** (`quant_hot` in zone A). It depends on
  L2 bandwidth relative to issue rate, which nothing here measures.
* **Anything relative to PyTorch SDPA.** The baseline picks the fastest SDPA
  backend available, and Windows builds of torch ship without flash attention
  (the smoke log says so). So `split_only` and `speedup_vs_sdpa` are
  platform-dependent and not compared across machines.

### Exclusions, fixed now

* A card whose `test_correctness.py` does not pass contributes **no** data.
* Rows are used only if quotable, or tier 2 and admissible for the effect in
  question (`dispersion_tier.usable_for`). Clock-rejected rows never.
* A DRAM-resident cell is excluded if either row's recorded `footprint_over_l2`
  is below **2**. `benchmark.py` caps a context at 512 replicas, which leaves the
  4-bit set at ctx=512 at 2.5× L2 here and ~1.2× on a 72 MB card. That is not
  cold. The cap stays, because changing it is a protocol change on this card.
* One run per card, **the last one received**, the same selection rule as
  `results/benchmark.json`, because it is the one rule that cannot be gamed after
  the fact.
* **Every card that returns a file is reported**, including cards that fail
  predictions and cards whose runs are mostly gate-rejected. A card is never
  dropped after its numbers are seen.

### What changes in the README, decided now

* **C1 fails on any card:** "pays ~1.19–1.51× once the working set exceeds L2"
  becomes a per-card statement with the measured range, and the headline stops
  implying generality.
* **P2 or C4 fails:** the L2-residency mechanism is withdrawn from the headline.
  The README says what was observed instead.
* **All hold:** the README says "tested on N cards spanning an X× range of L2",
  with the cards named, and the magnitude claim stays qualified by M1.

---

## Outcome — Part A

*(appended after the run; nothing above this line changes)*

## Outcome — Part B

*(appended as cards arrive)*
