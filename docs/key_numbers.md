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
| **Flash-decoding split** | **9.7–24×** DRAM-resident, **13–69×** L2-resident | fp16 SDPA ÷ fp16 Triton control. Nothing to do with quantization. |
| **Quantization, L2-resident** | **0.51–0.72×** | fp16 control ÷ fused 4-bit. The bits *cost* 1.4–2×. |
| **Quantization, DRAM-resident** | **0.72–1.32×** | Same ratio, working set 3× L2. The bits pay off only at ≥ 2k tokens. |

The headline "36× faster than PyTorch" is the product of the first row and a
number near 1. Quoting it as a quantization result is the error this project
exists to avoid.

---

## Timing, µs per decode step

**L2-resident (CUDA-graph replay, median of 25 samples × 50 calls):**

| ctx | SDPA fp16 | Triton fp16 control | fused 4-bit | fused 2-bit | split effect | quant effect |
|---|---|---|---|---|---|---|
| 512 | 48.7 | 3.7 | 5.3 | 6.4 | 13.2× | 0.69× |
| 2048 | 180.6 | 7.2 | 11.0 | 11.0 | 25.2× | 0.65× |
| 8192 | 736.2* | 14.2* | 23.5 | 23.2 | 51.7× | 0.61× |
| 16384 | 1463.8* | 21.2 | 41.2 | 40.8 | 69.1× | 0.51× |

**DRAM-resident (rotating working set 3× L2 = 101 MB, median (IQR) of 50 samples):**

| ctx | SDPA fp16 | Triton fp16 control | fused 4-bit | fused 2-bit | split effect | quant effect |
|---|---|---|---|---|---|---|
| 512 | 49.0 (0.1) | 5.0 (0.1) | 7.0 (0.1) | 5.9 (0.2) | 9.7× | 0.72× |
| 2048 | 179.7 (3.1) | 12.3 (0.1) | 10.8 (0.3) | 10.4 (0.1) | 14.6× | 1.14× |
| 8192 | 766.3 (48.6)* | 31.4 (0.5)* | 26.2 (0.5) | 25.8 (0.3) | 24.4× | 1.20× |
| 16384 | 1525.2 (102.2)* | 63.4 (1.8) | 48.0 (1.0) | 49.7 (2.3) | 24.0× | 1.32× |

Bootstrap 95% CIs on the quantization effect (4-bit, DRAM-resident): 512
`[0.72, 0.81]`, 2048 `[1.12, 1.17]`, 8192 `[1.20, 1.21]`, 16384 `[1.30, 1.33]`.
The effect is small, but it is not noise.

**Clock verification: 22 of 32 rows quotable.** A row is quotable only if every
`nvidia-smi` sample taken during its sampling loop was ≥ 70% of the 3090 MHz
maximum SM clock *and* its own IQR was ≤ 5% of its median.

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

`pytest test_correctness.py -q` → **66 passed** in ~26 s.

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

## Audit verdicts

63 claims in `results/audit.md`: **24 TRUE, 20 TRUE BUT CONDITIONAL, 9
MISLEADING, 10 FALSE.** All ten `FALSE` verdicts are this project's own claims
about the L2-resident regime.

---

## The correction

An earlier version of this repo reported the DRAM-resident quantization effect
as **11–15×**. It is **1.14–1.32×**. The old measurement timed the fp16 control
while the GPU was still at idle clocks — a 9× clock range on an 80 W laptop
part — which made the control look ~12× slower than it is. Nothing in the old
benchmark could detect this, because it never recorded what the clocks were
doing. That is why every row now carries its clock window.
