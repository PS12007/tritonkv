# Fused Triton kernel for quantized-KV decode attention

A Triton kernel that computes one decode step of attention **directly on a
packed 2/4-bit KV cache**, without ever materializing a full-precision copy of
the history.

**Status: the kernel is correct, well tested, and slower than an unquantized
kernel of the same shape whenever the KV cache fits in L2.** That is the result,
and it is not the one the project set out to find. Every timing below is
clock-verified on a laptop GPU whose clock range is wider than the effects being
measured — see [Results](#results) and [What is not true](#what-is-not-true-yet).
No number here should be quoted without the condition attached to it.

---

## The problem

A naive quantized KV cache does this on every decode step:

```python
K_fp16 = dequantize(K_packed)     # writes S x D fp16 to DRAM
V_fp16 = dequantize(V_packed)     # writes S x D fp16 to DRAM
out    = attention(q, K_fp16, V_fp16)   # reads them straight back
```

Decode attention is memory-bound — one query row against the whole history — so
that round trip through DRAM is close to *all* of the cost. Per cached
element-row the naive path moves `0.5·D` (read packed) + `2·D` (write fp16) +
`2·D` (read it back) = **4.5·D bytes**, where a fused kernel moves **0.5·D**.

Why it is non-trivial: to feed `tl.dot` the kernel needs a dense `(BLOCK_N, D)`
tile of dequantized K, but a load of packed codes gives `(BLOCK_N, D/P)` bytes,
and Triton cannot slice-assign into a tile. The two obvious ways out — a
`tl.join`+`tl.reshape`, or `P` unrolled accumulators — cost a shared-memory
layout conversion or depend on fragile constexpr unrolling.

**The way around it** is to not reconstruct the tile at all, but *address* it.
Codes are packed "split-P", so byte `j` holds dims `j, j+DP, j+2·DP, …`. The
kernel builds an index vector over the full head dim and loads byte `d % DP`
with shift `(d // DP)·nbits`. Each byte is loaded `P` times, but those loads hit
the same cache line, so DRAM traffic is unchanged and the result is a dense
`(BLOCK_N, D)` tile with no reshape, no transpose, and no unrolled accumulators.

The kernel is flash-decoding shaped: the history is split across programs, each
computing a partial online-softmax result, reduced by a second tiny kernel. One
program owns one KV head and *all* the query heads sharing it, so the unpack is
paid once per GQA group rather than once per query head.

---

## Results

Measured on the machine described under [Reproducing](#reproducing). Shapes are
Qwen2.5-1.5B-Instruct's attention (`HQ=12, HKV=2, D=128`, GQA group 6), batch 1,
one attention layer, `group_size=32`.

### The honest headline

The fused kernel beats PyTorch's fp16 SDPA by 7–36×. **That number is
misleading and should not be used.** It changes two things at once: the cache is
4-bit *and* the work is split across the history. PyTorch's SDPA does no split
for `q_len == 1`, so a fused-vs-SDPA comparison silently credits the
quantization for a parallelization win.

`kernels/fp16_decode_attn.py` exists to separate them. It is the same kernel —
same split, same online softmax, same GQA amortization, same combine kernel —
reading plain fp16. The only difference is the dequantization.

![Where the speedup comes from](docs/plots/speedup_attribution_4b.png)

**Splitting the history is worth 9–25×. The quantization is worth 0.5–1.3×.**

**Hot regime (cache fits in L2), µs per decode step, CUDA-graph replay, median
of 25 samples × 50 calls:**

| ctx | SDPA fp16 | Triton fp16 (control) | fused 4-bit | fused 2-bit | flash-decode effect | quantization effect |
|---|---|---|---|---|---|---|
| 512 | 48.7 | 3.7 | 5.3 | 6.4 | 13.2× | **0.69×** |
| 2048 | 180.6 | 7.2 | 11.0 | 11.0 | 25.2× | **0.65×** |
| 8192 | 736.2* | 14.2* | 23.5 | 23.2 | 51.7× | **0.61×** |
| 16384 | 1463.8* | 21.2 | 41.2 | 40.8 | 69.1× | **0.51×** |

**Quantization makes the kernel 1.4–2× slower here, not faster.** Nearly the
whole apparent win is the split. At 16k the fp16 path runs at ~410 GB/s (pure
L2) while the 4-bit path reaches only ~64 GB/s — the fused kernel is issue-bound
in the shift/mask/convert/fma chain, not bandwidth-bound. Group size barely moves
it (25.4 / 25.6 / 26.6 µs at gs = 16/32/64), so the scale+zero tile loads are
not the cost.

**Cold regime (rotating working set, 3× L2 = 101 MB), µs per decode step,
median (IQR) of 50 samples:**

| ctx | SDPA fp16 | Triton fp16 (control) | fused 4-bit | fused 2-bit | flash-decode effect | quantization effect |
|---|---|---|---|---|---|---|
| 512 | 49.0 (0.1) | 5.0 (0.1) | 7.0 (0.1) | 5.9 (0.2) | 9.7× | 0.72× |
| 2048 | 179.7 (3.1) | 12.3 (0.1) | 10.8 (0.3) | 10.4 (0.1) | 14.6× | **1.14×** |
| 8192 | 766.3 (48.6)* | 31.4 (0.5)* | 26.2 (0.5) | 25.8 (0.3) | 24.4× | **1.20×** |
| 16384 | 1525.2 (102.2)* | 63.4 (1.8) | 48.0 (1.0) | 49.7 (2.3) | 24.0× | **1.32×** |

`*` = did not pass the clock-verification gate (see below) and is not quoted as
evidence anywhere; every conclusion here rests on unstarred rows.

The sign flips: once the cache genuinely comes from DRAM, 4-bit leads — but by
**1.14–1.32×, not by an order of magnitude**.

![What the quantization itself buys](docs/plots/quantization_effect_4b.png)

**So the real claim is conditional: the fused kernel pays for itself only when
the KV cache does not fit in L2, and costs ~1.5–2× when it does.** At 512 tokens
it loses in both regimes — there is not enough history to amortize anything.

### Every timing on this page is clock-verified

An earlier version of this README reported the cold regime as an **11–15×** win
for quantization. That was wrong, and the way it was wrong is worth stating.

This is an 80 W laptop GPU that idles at 285 MHz and boosts to 3090 MHz — a
9× range, larger than most effects being measured here. The fp16 control kernel
is fast enough that its measurement finished while the GPU was still at idle
clocks, so the control looked ~12× slower than it is, and the quantization
looked ~12× better than it is. Nothing in the old benchmark could see this,
because it never asked what the clocks were doing.

`benchmark.py` now runs a background `nvidia-smi` sampler, deliberately spins the
GPU up to ≥ 80% of maximum before *every* measurement, and attributes the clock
samples to the sampling loop only — warmup and CUDA-graph capture are excluded,
so they neither look like throttling nor hide it. A row is **quotable** only if

1. every clock sample during its sampling loop was ≥ 70% of the 3090 MHz
   maximum, and
2. the timing's own IQR was ≤ 5% of its median.

The last full run: **22 of 32 rows quotable**. The 10 rejects are named in
`results/benchmark.json` under `clock_monitoring.rejected_rows` and are starred
in the tables above. Most of them are the slow PyTorch baselines, whose long
per-sample gaps let the clocks sag — which would have *inflated* the speedups
reported against them, so rejecting them is the conservative choice.

### Memory

Exact, not measured — these follow from the format.

| format | effective bits/element | cache @ ctx=512 (1 layer) | vs fp16 |
|---|---|---|---|
| fp16 | 16.0 | 0.52 MB | 1.0× |
| 4-bit, gs=32 | **5.0** | 0.16 MB | **3.2×** |
| 2-bit, gs=32 | **3.0** | 0.10 MB | **5.3×** |

4-bit with an fp16 scale and zero per 32 elements is **5.0** bits/element, not
4. Every memory claim here quotes the effective number, so the compression is
3.2×, not 4×.

![KV cache size for the whole model](docs/plots/kv_cache_memory.png)

This is the one unconditional win: at 16k tokens the whole-model KV cache is
470 MB in fp16, 147 MB at 4-bit, 88 MB at 2-bit. The fused kernel also
allocates nothing to get there — 0.3 MB of transient workspace at 16k, against
117 MB for dequantize-then-SDPA, which has to materialize the fp16 cache.

### Correctness

`python -m pytest test_correctness.py -q` → **66 passed in ~26 s**.

Two different errors are measured, and the distinction is the whole point:

- **Kernel error** — fused kernel vs. dequantize-then-attend in fp32 on the
  *same* dequantized values. Any difference is the kernel's own arithmetic.
- **End-to-end error** — vs. attention on the unquantized fp16 cache. This is
  dominated by the quantizer, not the kernel.

| ctx | bits | cosine (kernel vs dequant ref) | rel L2 | kernel vs fp16 truth | baseline vs fp16 truth |
|---|---|---|---|---|---|
| 512 | 4 | ≥ 0.9999999 | ≤ 4.18e-04 | 1.207e-01 | 1.208e-01 |
| 2048 | 4 | ≥ 0.9999999 | ≤ 4.75e-04 | 1.396e-01 | 1.397e-01 |
| 8192 | 4 | ≥ 0.9999988 | ≤ 1.59e-03 | 1.251e-01 | 1.249e-01 |
| 16384 | 4 | ≥ 0.9999992 | ≤ 1.28e-03 | 1.509e-01 | 1.514e-01 |
| 512 | 2 | ≥ 0.9999998 | ≤ 5.98e-04 | 6.558e-01 | 6.560e-01 |
| 2048 | 2 | ≥ 0.9999996 | ≤ 8.71e-04 | 8.110e-01 | 8.105e-01 |
| 8192 | 2 | ≥ 0.9999996 | ≤ 8.44e-04 | 6.589e-01 | 6.585e-01 |
| 16384 | 2 | ≥ 0.9999995 | ≤ 9.01e-04 | 7.560e-01 | 7.557e-01 |

Thresholds are asserted, not eyeballed: `cosine ≥ 0.99999`, `rel L2 ≤ 5e-3`, and
the kernel's end-to-end error must be within 1.5× of the PyTorch baseline's.
The suite also covers pack/unpack roundtrip exactness, a half-quantization-step
reconstruction bound, GQA and MHA shapes, group sizes 16–128, block sizes
16–128, split counts 1/2/5/17/64 (uneven on purpose), S=1, extreme scores, and a
worst-single-element check that aggregate cosine would hide.

![Accuracy cost of quantizing the KV cache](docs/plots/correctness_vs_bits.png)

Test inputs inject 1% heavy-tailed outliers at 8× magnitude, because pure
Gaussian noise is an unrealistically easy input for a quantizer and would
flatter these numbers.

### Self-audit

`audit_claims.py` writes down every claim this project could make, then attacks
it. Speedups are judged by a bootstrap 95% CI over the raw per-sample timings
against a 1.05× practical-significance bar, so a difference that is really
run-to-run jitter cannot be reported as a win.

| verdict | count |
|---|---|
| TRUE | 24 |
| TRUE BUT CONDITIONAL | 20 |
| MISLEADING | 9 |
| FALSE | 10 |

The ten `FALSE` verdicts are all this project's own claims about the L2-resident
regime, and the audit is what puts them there. Full text in `results/audit.md`
after a run.

---

## What is not true (yet)

Be specific about what is *not* solved:

1. **"The fused kernel is Nx faster than PyTorch."** Misleading. Most of that
   ratio is flash-decoding, which has nothing to do with quantization. Use the
   decomposition table.
2. **"Low-bit KV makes decode faster."** False in the L2-resident regime — it is
   1.4–2× *slower* there. True only when the working set exceeds L2, and then
   only by 1.14–1.32×, and only at 2k tokens and above. At 512 tokens it loses
   in both regimes.
3. **2-bit is not usable**, despite passing. The kernel reproduces the
   dequantized values to cosine ≥ 0.9999996, but the *quantizer* loses far too
   much: rel L2 of **0.66–0.81** against the fp16 cache, versus 0.12–0.14 for
   4-bit. 2-bit is a correct implementation of a scheme that does not preserve
   the cache. 4-bit is the only configuration worth using.
4. **No end-to-end model integration.** Everything here is one attention layer.
   There is no tokens/sec claim, and none should be inferred.
5. **FlashAttention is not in the comparison** — this Windows torch build
   reports "not compiled with flash attention", so the strongest fp16 baseline
   available was cuDNN. On Linux with FA2 the SDPA baseline would be much
   stronger and the flash-decode effect much smaller.
6. **Per-channel key quantization is not implemented.** KIVI shows keys are
   better quantized per-channel; this uses per-token grouping along `head_dim`
   for both K and V, which is simpler but leaves accuracy on the table.
7. **One GPU, one clock regime, one driver.** The clock-verification gate makes
   these numbers reproducible *on this machine*; it says nothing about how the
   attribution shifts on a desktop part with a bigger L2 or a fixed power
   budget. The conditional is stated in terms of L2 residency precisely because
   that is the axis expected to move.
8. **The 8k and 16k SDPA baselines are not clock-verified.** Their timings
   scatter by 6–7% run to run, above the 5% gate, so the "36× vs PyTorch"
   figure at long context is reported but not leaned on.

---

## Reproducing

**Hardware actually used:** NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell,
sm_120, 26 SMs, 8 GB, 34 MB L2), Windows 11, driver 610.47. This is a
thermally-limited 80 W laptop part sharing the GPU with the desktop compositor,
idling at 285 MHz and boosting to 3090 MHz — which is exactly why every timing
is gated on clock verification. `nvidia-smi -lgc` would pin the clocks outright
but needs administrator rights; the benchmark spins the GPU up instead and
rejects any row it could not verify.

**Stack:** `torch 2.12.0+cu130`, `triton-windows 3.8.0.post28`, CUDA 13.0,
`transformers 5.16.1`. Full pins in `requirements.txt`.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe -m pytest test_correctness.py -q   # ~26 s
.venv/Scripts/python.exe benchmark.py --quick                # ~75 s smoke run
.venv/Scripts/python.exe benchmark.py --samples 50           # full suite, ~4.5 min
.venv/Scripts/python.exe audit_claims.py                     # reads results/benchmark.json
.venv/Scripts/python.exe make_plots.py                       # regenerates docs/plots/
```

On Windows, Triton needs an MSVC toolchain (Visual Studio 2022, MSVC 14.4x).
On Linux, swap `triton-windows` for `triton==3.8.0`.

`results/` is gitignored — benchmark output is meant to be regenerated, not
committed.

---

## Layout

| file | what it is |
|---|---|
| `quantize.py` | group-wise 2/4-bit asymmetric quantization, split-P packing. Pure PyTorch, the correctness ground truth. |
| `reference.py` | fp32 ground truth + the baselines. Probes six fp16 attention strategies per shape and caches the fastest, so the baseline is not a strawman. |
| `kernels/fused_decode_attn.py` | the fused kernel. |
| `kernels/fp16_decode_attn.py` | the control: identical shape, unquantized. Isolates the flash-decoding effect. |
| `test_correctness.py` | 66 tests, explicit asserted thresholds. |
| `benchmark.py` | timing + memory. Rotating working set for the cold regime, CUDA-graph replay for the hot one. |
| `audit_claims.py` | adversarial self-audit: 63 claims, bootstrap CIs over raw timings, attribution against the fp16 control, and a clock-verification gate. |
| `make_plots.py` | the figures in `docs/plots/`, regenerated from `results/benchmark.json`. |
| `docs/progress_log.md` | what was tried and what broke, written as it happened. |
| `docs/next_steps.md` | ordered list of what is left. |

## Methodology notes

Two measurement bugs were found and fixed; both had been inflating the kernel's
win, and both are worth knowing about if you benchmark decode attention:

- **The baselines were accidental strawmen.** Every baseline went through SDPA
  with `enable_gqa=True`, which runs at 8.5 GB/s here. Explicitly expanding KV
  gets 11.2 GB/s; cuDNN gets 26.4 GB/s. `reference.py` now probes and caches the
  winner per shape.
- **"Cold" timing measured Windows, not the GPU.** `flush L2, time one call`
  reported 395 µs for 43 µs of GPU work — on an idle GPU that is the WDDM
  submission path waking up. Replaced with N independent cache copies sized to
  exceed L2, replayed from one CUDA graph.

Ranking candidates by wall clock is also unusable on Windows: WDDM's 50–300 µs
per-call cost had the backend probe preferring a 3.3 GB/s path over a 24.7 GB/s
one. The probe ranks by graph replay.
