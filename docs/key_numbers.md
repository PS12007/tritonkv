# Key numbers

Every number here comes from one clock-verified run of `benchmark.py --samples 50`
(2026-09-01, `results/benchmark.json`) or from `test_correctness.py`. Numbers
that did not pass the benchmark's clock-verification gate are marked `*` and are
not used to support any conclusion.

**Machine.** RTX 5060 Laptop GPU, sm_120, 26 SMs, 8 GB, **33.6 MB L2**, 80 W,
SM clock 285 MHz idle / 3090 MHz max. Windows 11, torch 2.12.0+cu130,
triton-windows 3.8.0.post28.

**Shape.** Qwen2.5-1.5B-Instruct attention: `HQ=12, HKV=2, D=128`, GQA group 6,
batch 1, one attention layer, `group_size=32`, decode step (`q_len=1`).

---

## The three numbers that matter

| | value | what it means |
|---|---|---|
| **Flash-decoding split** | **10–26×** (quotable rows) | fp16 SDPA ÷ fp16 Triton control. Nothing to do with quantization. |
| **Quantization, L2-resident** | **0.74–0.90×** | fp16 control ÷ fused 4-bit. The bits *cost* 1.1–1.35×. |
| **Quantization, DRAM-resident** | **0.97–1.22×** (quotable), 1.26–1.47× beyond | Same ratio, working set 3× L2. The bits pay off from ~2k tokens. |

The headline "up to 70× faster than PyTorch" is the product of the first row and
a number near 1. Quoting it as a quantization result is the error this project
exists to avoid.

**The sign flip is now fully clock-verified at a single context.** At ctx=2048,
with all three methods passing the gate: **0.90× L2-resident, 1.22×
DRAM-resident.** Earlier versions of this claim rested on rows where the fp16
control had failed the gate.

---

## Timing, µs per decode step

**L2-resident (CUDA-graph replay, median of ≥25 samples × 50 calls):**

| ctx | SDPA fp16 | Triton fp16 control | fused 4-bit | fused 2-bit | gather-meta 4-bit | split | quant |
|---|---|---|---|---|---|---|---|
| 512 | 46.7 | 3.5 | 4.7 | 4.8* | 5.5* | 13.3× | 0.74× |
| 2048 | 178.8 | 6.8 | 7.6 | 7.7 | 9.7 | 26.2× | **0.90×** |
| 8192 | 747.3 | 13.7* | 16.8 | 16.9* | 22.1* | 54.4× | 0.82× |
| 16384 | 1493.7* | 21.2* | 28.9* | 29.2* | 42.7* | 70.4× | 0.74× |

**DRAM-resident (rotating working set 3× L2 = 101 MB, median of ≥50 samples):**

| ctx | SDPA fp16 | Triton fp16 control | fused 4-bit | fused 2-bit | gather-meta 4-bit | split | quant |
|---|---|---|---|---|---|---|---|
| 512 | 50.1 | 5.0 | 5.2 | 5.2* | 6.0* | 10.0× | 0.97× |
| 2048 | 184.1 | 12.7 | 10.4 | 9.9 | 11.0 | 14.5× | **1.22×** |
| 8192 | 756.4 | 31.9* | 25.3 | 21.8* | 28.5* | 23.7× | 1.26× |
| 16384 | 1501.6* | 58.8* | 40.1* | 36.2* | 49.5* | 25.5× | 1.47× |

`gather-meta` is the same kernel with the metadata gather instead of the
broadcast — carried as a permanent control row, see below.

**Clock verification: 25 of 48 rows quotable.** A row is quotable only if
every `nvidia-smi` sample during its sampling loop was ≥ 70% of the 3090 MHz
maximum SM clock, its own IQR was ≤ 5% of its median, **and its clock window
holds ≥ 4 samples**. Every rejection in this run is dispersion; there are no
clock rejections left.

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

**Speedup (gather ÷ broadcast), ctx = 512 / 2048 / 8192 / 16384:**

| regime | 4-bit | 2-bit |
|---|---|---|
| L2-resident | 1.16 / 1.27 / 1.32 / 1.48× | 1.15 / 1.27 / 1.40 / 1.48× |
| DRAM-resident | 1.15 / 1.05 / 1.13 / 1.24× | 1.14 / 1.06 / 1.22 / 1.32× |

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

`pytest test_correctness.py -q` → **106 passed** in ~123 s (66 before this
session; the 40 new ones assert the metadata gather and broadcast paths are
bitwise identical across S × nbits × group_size).

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
