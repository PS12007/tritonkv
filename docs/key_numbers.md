# Key numbers

Every number here comes from one clock-verified run of `benchmark.py --samples 50`
(2026-09-02, `results/benchmark.json`, which is run 3 of three back-to-back runs)
or from `test_correctness.py`. Numbers that did not pass the benchmark's
clock-verification gate are marked `*` and are not used to support any
conclusion.

**Where a range is given "across runs", it comes from `results/between_run.md`
and is the union of three independent full runs' intervals — not a within-run
bootstrap CI, which is a median 2.4× narrower and does not cover the memory
P-state the card happens to land in.**

**Machine.** RTX 5060 Laptop GPU, sm_120, 26 SMs, 8 GB, **33.6 MB L2**, 80 W,
SM clock 285 MHz idle / 3090 MHz max. Windows 11, torch 2.12.0+cu130,
triton-windows 3.8.0.post28.

**Shape.** Qwen2.5-1.5B-Instruct attention: `HQ=12, HKV=2, D=128`, GQA group 6,
batch 1, one attention layer, `group_size=32`, decode step (`q_len=1`).

---

## The three numbers that matter

| | value | what it means |
|---|---|---|
| **Flash-decoding split** | **10.5–26×** (DRAM), 14–68× (L2) | fp16 SDPA ÷ fp16 Triton control. Nothing to do with quantization. |
| **Quantization, L2-resident** | **0.717–0.813×** across runs | fp16 control ÷ fused 4-bit. The bits *cost* 1.23–1.37×, at every context. |
| **Quantization, DRAM-resident** | **0.905–0.928×** at 512, **1.188–1.478×** at 2k–16k | Same ratio, working set 3× L2. The bits pay off from ~2k tokens. |

The headline "up to 70× faster than PyTorch" is the product of the first row and
a number near 1. Quoting it as a quantization result is the error this project
exists to avoid.

**The sign flip is fully clock-verified at ctx=8192, in all three runs.** SDPA,
the fp16 control and the fused kernel all pass the gate there every time:
**0.800–0.813× L2-resident, 1.469–1.478× DRAM-resident.**

ctx=512 and ctx=2048 were previously quoted with "clears the gate in one run and
not the others". Measured over **nine** full-method runs spanning three
protocols and admitting the pinned tier, all three rows of the chain survive at
**9/9** at both contexts (against 5/9 on the gate alone). ctx=8192 is unchanged
at 7/9 and ctx=16384 goes 2/9 → **7/9**. Promotion is still a property of the
run — no row is promoted in more than four of the nine —
so this is stated with a denominator, not as a star. Earlier versions of this
claim rested on rows where the fp16 control had failed the gate.

---

## Timing, µs per decode step

**L2-resident (CUDA-graph replay, median of ≥25 samples × 50 calls):**

| ctx | SDPA fp16 | Triton fp16 control | fused 4-bit | fused 2-bit | gather-meta 4-bit | split | quant |
|---|---|---|---|---|---|---|---|
| 512 | 46.4 | 3.3 | 4.6~ | 5.9~ | 5.3 | 14.0× | 0.73× |
| 2048 | 174.9 | 6.7~ | 8.2~ | 7.9~ | 9.9~ | 26.2× | 0.81× |
| 8192 | 734.7 | 13.5 | 17.1 | 16.8 | 23.6 | 54.3× | **0.79×** |
| 16384 | 1456.9 | 21.3 | 29.2 | 29.3 | 43.0 | 68.4× | **0.73×** |

**DRAM-resident (rotating working set 3× L2 = 101 MB, median of ≥50 samples):**

| ctx | SDPA fp16 | Triton fp16 control | fused 4-bit | fused 2-bit | gather-meta 4-bit | split | quant |
|---|---|---|---|---|---|---|---|
| 512 | 49.0 | 4.6 | 5.1~ | 6.4~ | 5.9 | 10.5× | 0.90× |
| 2048 | 179.6 | 11.8~ | 9.7~ | 9.3~ | 11.4~ | 15.3× | 1.21× |
| 8192 | 735.6 | 32.8 | 22.2 | 20.7 | 27.5 | 22.4× | **1.48×** |
| 16384 | 1464.1 | 55.6 | 38.6 | 35.8 | 48.7 | 26.3× | **1.44×** |

`gather-meta` is the same kernel with the metadata gather instead of the
broadcast — carried as a permanent control row, see below.

**Clock verification: 39 of 48 rows quotable in this run** (42 and 41 in the
other two — the count itself moves between runs; 35 rows pass in all three and
46 in at least one.) A row is quotable only if
every `nvidia-smi` sample during its sampling loop was ≥ 70% of the 3090 MHz
maximum SM clock, its own IQR was ≤ 5% of its median, **and its clock window
holds ≥ 4 samples**. Every rejection in this run is dispersion; there are no
clock rejections left.

**`~` is the second tier, not a star.** All nine rejections in this run are
dispersion, and `dispersion_tier.py` asks the question the gate does not: how
well is the *median* pinned, which is the number these tables quote? Seven of the
nine pin every regime at least as well as the worst row the gate already accepts
(±1.70%, itself an unstarred row here), so they are marked `~` and used for
effects at least 5× their own pin. The `~` rows above are pinned to ±0.20–1.04%
against quantization effects of 10–27%. Two rows are **not** promoted — pinned
only to ±2.33% and ±2.68% — which is why the gate was not widened instead.

---

## The metadata-broadcast change

The per-group scale and zero are `(BLOCK_N, D/group_size)` in memory. The kernel
used to load them as `(BLOCK_N, D)` by indexing with `d // group_size`,
re-reading each parameter `group_size` times, four times per loop iteration.
Loading them at their real width and expanding in registers is bitwise
identical:

| | instructions | registers | spills |
|---|---|---|---|
| gather (`d // GS`) | 2245 | 244 | 0 |
| broadcast | **1653** | **128** | 0 |

**Speedup (gather ÷ broadcast), ctx = 512 / 2048 / 8192 / 16384**, as the range
over three independent runs:

| regime | 4-bit | 2-bit |
|---|---|---|
| L2-resident | 1.12–1.23 / 1.22–1.30 / 1.39 / 1.47× | 1.15–1.17 / 1.19–1.24 / 1.41 / 1.48× |
| DRAM-resident | 1.14 / 1.16–1.18 / 1.24 / 1.26–1.28× | 1.11–1.15 / 1.11–1.16 / 1.27 / 1.34× |

The change is `TRUE` at every context, in both regimes, at both bit widths, in
every run — it is the most reproducible result in this file, and the only one
where the run-to-run spread is smaller than the effect at every cell.

**What this refutes.** An earlier version of this file and the README said
*"group size barely moves it, so the scale+zero tile loads are not the cost."*
The measurement was right; the inference was wrong. In the gather path the load
is indexed by `d // GS` over the full head dim, so it issues `BLOCK_N * D` loads
**whatever `GS` is**. Group size changes how many distinct values are read, never
how many instructions are issued — the experiment varied metadata *bytes* and
concluded about metadata *instructions*.

**Two rejected variants**, kept here because they bound the claim:

- **Zero-point fold** (`fold_zp=True`): 0.74–1.10× (4-bit), 0.66–0.94× (2-bit).
  Not a speed win at any context in either regime. Kept as an option because it
  is *more accurate* — it never rounds a dequantized K value to fp16, so kernel
  error stays flat at **1.5e-4** instead of drifting 2.3e-4 → 7.7e-4 with
  context.
- **The same trick on the packed codes**: bitwise identical and a **loss**
  (0.69–0.96× at ctx ≥ 8192), registers 128 → 223. The codes are needed at full
  width regardless, so expanding them from a narrow load adds a live tile
  without removing one. Reverted.

---

## Memory (exact, from the format — not measured)

| format | effective bits/element | 1 layer @ 16k | whole model @ 16k | vs fp16 |
|---|---|---|---|---|
| fp16 | 16.0 | 16.8 MB | 470 MB | 1.0× |
| 4-bit, gs=32 | **5.0** | 5.2 MB | 147 MB | **3.2×** |
| 2-bit, gs=32 | **3.0** | 3.1 MB | 88 MB | **5.3×** |

4-bit with an fp16 scale and zero per 32 elements is 5.0 bits/element, not 4, so
the compression is 3.2×, not 4×.

**Transient allocation at 16k, 4-bit:** fused kernel **0.32 MB**, versus
**117 MB** for dequantize-then-SDPA, which must materialize the fp16 cache. This
is the one advantage that holds in every regime.

---

## Accuracy

`pytest test_correctness.py -q` → **106 passed** in ~89 s (66 before this
session; the 40 new ones assert the metadata gather and broadcast paths are
bitwise identical across S × nbits × group_size). `pytest test_between_run.py -q`
→ **16 passed** in ~2 s, CPU only: the between-run machinery checked against
synthetic runs whose answers are known in advance.

| ctx | bits | cosine vs dequant ref | rel L2 vs dequant ref | kernel vs fp16 truth | PyTorch baseline vs fp16 truth |
|---|---|---|---|---|---|
| 512 | 4 | ≥ 0.9999999 | ≤ 4.18e-04 | 0.121 | 0.121 |
| 2048 | 4 | ≥ 0.9999999 | ≤ 4.75e-04 | 0.140 | 0.140 |
| 8192 | 4 | ≥ 0.9999988 | ≤ 1.59e-03 | 0.125 | 0.125 |
| 16384 | 4 | ≥ 0.9999992 | ≤ 1.28e-03 | 0.151 | 0.151 |
| 512 | 2 | ≥ 0.9999998 | ≤ 5.98e-04 | 0.656 | 0.656 |
| 2048 | 2 | ≥ 0.9999996 | ≤ 8.71e-04 | 0.811 | 0.811 |
| 8192 | 2 | ≥ 0.9999996 | ≤ 8.44e-04 | 0.659 | 0.659 |
| 16384 | 2 | ≥ 0.9999995 | ≤ 9.01e-04 | 0.756 | 0.756 |

The kernel column and the baseline column agree to three decimals at every row:
**the error is the quantizer's, not the kernel's.** 4-bit costs ~0.13 relative
L2; 2-bit costs ~0.7 and is not usable, despite passing every kernel-correctness
test.

Inputs carry 1% heavy-tailed outliers at 8× magnitude — Gaussian noise is an
unrealistically easy input for a quantizer.

---

## The corrections, in order

1. **11–15× → 1.14–1.32× (2026-09-01 morning).** The DRAM-resident quantization
   effect was measured with the fp16 control still at idle clocks — a 9× clock
   range on an 80 W laptop part. Nothing in that benchmark could detect it,
   because it never recorded the clocks.
2. **The clock gate was passing rows on one sample (2026-09-01 evening).** After
   the gate was added, 28 of 96 measurement windows were being judged on a
   single `nvidia-smi` sample, because the measurement was shorter than the
   sampler's 109 ms period. Windows are now held open ≥ 1.5 s and must carry
   ≥ 4 samples; the count is now **0 of 96**.

3. **An audit section that had never executed (2026-09-01 evening).** The
   per-optimization claims crashed on a tuple-indexing bug and were silently
   absent from every report, so the two kernel changes this project made had
   never been adjudicated against their own controls.
4. **A claim reading a starred row (2026-09-01 evening).** The zero-point-fold
   claim cleared its bar at ctx=8192 on a row where neither input had passed the
   gate, because it checked its own gate and not its neighbour's. Gated, it is
   `FALSE` at both bit widths.
5. **The confidence intervals were up to 1.95× too narrow (2026-09-01 late).**
   The timing series are serially correlated (lag-1 to 0.72), so the i.i.d.
   bootstrap was too confident in the flattering direction. Now circular-block.
   A companion bug: `_verdict` was a step function at the threshold, and one
   claim turned on 1.04973 against a 1.05 bar. It now needs a margin of 10% of
   the interval's own width. Together these changed **no** verdict.
6. **The warm-up was warming the wrong half of the machine (2026-09-01 late).**
   The ramp was a cache-resident GEMM, so it drove the SM clock and asked memory
   for nothing; the memory P-state then stepped up *during* bandwidth-bound
   measurements (−19.3% and −22.6% across their own windows). With a DRAM-sized
   copy alongside the GEMM, quotable rows went 25/48 → 39–42/48.
7. **The rejection line named the wrong regime (2026-09-01 late).** A row
   rejected for L2-resident dispersion printed the *cold* IQR — a number that
   passes the gate it was being blamed for.
8. **The confidence interval was the wrong interval (2026-09-02).** Every CI in
   the audit is a bootstrap over one run's samples, and covers sampling noise
   inside that run only. Three independent full runs put the honest interval at
   a median **2.4× wider** (worst 5.8× among gated ratios, 28× among rejected
   ones). No verdict moved, which is the reassuring half; the unreassuring half
   is that the 1.27× excursion that motivated the check did not recur in four
   subsequent runs, so the distribution has a tail that three runs cannot see.
   `between_run.py` now measures this and `audit_claims.py` reports it on every
   per-context claim.

9. **The measurement protocol was a free variable (2026-09-02).** Even the
   run-to-run interval is an interval over repetitions of *one* protocol. Run
   the same benchmark under four — 3 or 12 methods per context, crossed with
   0 or 300 s of saturating preload — and `quant_cold@8192` reads 1.4755 /
   1.4217 / 1.3968 / 1.3944. The shipped protocol reports the highest of the
   four, and its range misses the other three entirely. The 2x2 says what the
   channel is: method count is worth +3.8% with no preload and **−0.2%** after
   one, so run length was only ever a proxy for recently-pulled bandwidth. This
   one was **pre-registered** — the three candidate outcomes and their values
   were committed before the runs — and the prediction that landed was hit to
   0.2%. `compare_protocols.py` measures it; `audit_claims.py` carries it as
   `method.protocol_choice`.

10. **The gate tested a statistic nobody quotes (2026-09-03).** A row is
    rejected when its *per-sample IQR* exceeds 5% of its median, but every table
    here quotes the *median*, and on the short L2-resident rows those two come
    apart completely — eight of run 3's ten rejected measurements pin their
    medians to ±0.05–1.31% while scattering 5.6–11.5% per sample. The fix is not
    a wider gate, which would also admit the two rows pinned only to ±2.33% and
    ±2.68%; it is a second verdict whose bar is **the worst-pinned row the gate
    already accepts** (±1.70%), so it cannot admit anything less certain than a
    number already printed unstarred. Over nine full runs this takes the
    attribution chain at ctx=512 from 5/9 runs to 9/9 and at ctx=2048 from 5/9 to
    9/9 — the two contexts carrying the sign flip, previously quoted with an
    apologetic qualifier. `dispersion_tier.py` measures it; `audit_claims.py`
    carries it as `method.dispersion_tier` and marks promoted rows `~`.

**All ten are apparatus. None is a bug in the kernel.** That count is the most
transferable thing in this repo: on a thermally-limited consumer part the
measurement is harder than the optimization, and every one of these corrections
moved a number that had already been written down as a result.
