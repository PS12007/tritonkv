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

**Run:** 2026-09-10, `results/l2_sweep.json`, 13 points, ~11 min, every clock
window stable, worst correctness cosine 0.9999998, copy bandwidth 343 GB/s.
**By the committed scoring (`l2_sweep.analyze`, all points): all five hold.**

| | prediction | outcome | written-down guess |
|---|---|---|---|
| **P1** | zone A hot CI < 1 | **HOLDS**: 0.824 / 0.813 / 0.739 / 0.735 | — |
| **P2** | crossing in [0.5, 2] x L2 | **HOLDS**: **x\* = 1.04**, ctx ≈ 34 100 | ≈ 1.0, ctx ≈ 32k |
| **P3** | zone B hot > 1 and > cold | **HOLDS**: 1.914 vs 1.435; 1.905 vs 1.426 | hot ≈ 1.8–2.2 |
| **P4** | zone C \|hot/cold − 1\| ≤ 0.10 | **HOLDS**: 0.921 / 0.989 / 0.926 | hot ≈ cold ≈ 1.4–1.5 |
| **P5** | peak at x ∈ [1, 4] | **HOLDS**: 1.914 at x = 1.5 | — |

**The shape is sharper than the zones assumed.** At x = 1.00, with the fp16 cache
exactly the nominal L2 size, the control's hot per-token time is still only 7%
above x = 0.75 (ratio 0.812). At x = 1.25 its hot time is 93% of its DRAM time
(ratio 1.843). The cliff is between 1.00x and 1.25x. The fused kernel falls off
its own cliff between x = 3 and x = 4, i.e. with *its* cache between 0.94x and
1.25x L2, which is exactly 3.2x further out. The margins in `FIT_FRAC` /
`SPILL_FRAC` were wider than this card needed. They are not narrowed now; they
are what volunteers' cards will be scored against.

**Five things that have to be said alongside that table:**

1. **The hump rests on measurements the gate rejects.** 8 of 13 pairs have a
   gate failure, all of them dispersion (every clock window passed). On the hump
   (x = 1.25–3) the failing measurement is the **fp16 control's hot timing**:
   IQR 6–11%, with the median pinned to ±0.1–2.7%. It happens only where a
   working set 1.25–3x L2 is replayed in a loop, which fits an erratic partial
   hit rate. Rescored with filters the pre-registration did not specify for
   Part A:

   | scoring | P1 | P2 | P3 | P4 | P5 |
   |---|---|---|---|---|---|
   | committed (all points) | HOLDS | HOLDS (1.04) | HOLDS | HOLDS | HOLDS |
   | gate-quotable pairs only | HOLDS | HOLDS (1.78, only because the gate removes every point from 1.25x to 6x) | untestable | HOLDS (1 pt) | **FAILS** (peak among survivors is at x = 8) |
   | quotable + tier 2 (bar ±1.07%) | HOLDS | HOLDS (1.04) | untestable | HOLDS (2 pts) | HOLDS |

   **P1 and P2, the primary, survive every filter.** P3 and P5 hold by the
   committed scoring and depend on control-hot measurements pinned to only
   ±1.1–2.7%. That is worse than the tier-2 bar, but 30–80x smaller than the
   +90% effect they carry.
2. **Mean vs median.** The pre-registered statistic is `bootstrap_ratio_ci`
   (ratio of means). A median ratio would put ctx = 196608 at hot/cold 0.892,
   outside P4's ±10%. By the registered statistic it is 0.921, inside. It is
   reported, not re-scored.
3. **The sweep's DRAM ratios are biased low by ~7%, and the bias favours P3.**
   The sweep tunes on the hot regime and mostly picks `num_warps=2`. An A/B with
   the clocks ramped, three rounds each, puts that config 7–8% slower
   DRAM-resident than the benchmark's `num_warps=4` (23.2 vs 21.5 µs at 8k,
   141 vs 132 µs at 64k) and no slower hot. Correcting it would move zone B's cold
   ratio from ~1.43 toward ~1.53 against a hot ratio of ~1.9, so P3 would still
   hold by ~25%.
4. **A statement in this file did not hold.** The protocol caveat above says
   the sweep should match `benchmark.py` to within ~5% at shared contexts. The
   hot ratios do: 0.813 vs 0.800 at 8k, 0.739 vs 0.729 at 16k. The DRAM ratio at
   8k does not: 1.243 against 1.469–1.478, a 16% miss. Item 3 accounts for about
   7%. The fused cold measurement at 8k is itself gate-rejected (IQR 7.4%, pinned
   only ±2.9%) and plausibly accounts for more, but "plausibly" is not measured.
   No prediction uses that cell.
5. **The smoke-test caveat stands.** Two zone-A points were measured before this
   file was committed. They are consistent with the full run (0.813 / 0.820 then,
   0.824 / 0.813 now).

**Consequence for Part B:** nothing in Part B is changed by this outcome. It
does sharpen one expectation, recorded here as a note rather than a new
prediction: on this card the crossing sits at 1.04x L2, so a volunteer card's
`x*` well away from 1 would be informative even inside the [0.5, 2] window.

## Outcome — Part B

*(appended as cards arrive)*
