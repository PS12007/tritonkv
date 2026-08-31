# Fused Triton kernel for quantized-KV decode attention

A Triton kernel that computes one decode step of attention **directly on a
packed 2/4-bit KV cache**, without ever materializing a full-precision copy of
the history.

**Status: work in progress.** The kernel is correct and well tested. The
performance story is not finished, and the headline result is not the one the
project set out to find — see [Results](#results) and
[What is not true](#what-is-not-true-yet). No timing number here should be
quoted without the condition attached to it.

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

The fused kernel beats PyTorch's fp16 SDPA by 9–36×. **That number is
misleading and should not be used.** It changes two things at once: the cache is
4-bit *and* the work is split across the history. PyTorch's SDPA does no split
for `q_len == 1`, so a fused-vs-SDPA comparison silently credits the
quantization for a parallelization win.

`kernels/fp16_decode_attn.py` exists to separate them. It is the same kernel —
same split, same online softmax, same GQA amortization, same combine kernel —
reading plain fp16. The only difference is the dequantization.

**Hot regime (cache fits in L2), µs per decode step, CUDA-graph replay,
median of 15×200 calls, sd < 1%:**

| ctx | SDPA fp16 | Triton fp16 (control) | fused 4-bit | flash-decode effect | quantization effect |
|---|---|---|---|---|---|
| 512 | 46.1 | 3.2 | 5.2 | 14.6× | **0.61×** |
| 2048 | 173.9 | 6.5 | 10.1 | 26.7× | **0.64×** |
| 8192 | 732.1 | 13.5 | 25.7 | 54.2× | **0.53×** |
| 16384 | 1452.6 | 20.5 | 40.3 | 70.9× | **0.51×** |

**Quantization makes the kernel ~2× slower here, not faster.** Nearly the whole
apparent win is the split. At 16k the fp16 path runs at ~410 GB/s (pure L2)
while the 4-bit path reaches only ~64 GB/s — the fused kernel is issue-bound in
the shift/mask/convert/fma chain, not bandwidth-bound. Group size barely moves
it (25.4 / 25.6 / 26.6 µs at gs = 16/32/64), so the scale+zero tile loads are
not the cost.

**Cold regime (working set 4× L2 = 134 MB) — PROVISIONAL, see caveat:**

| ctx | Triton fp16 (control) | fused 4-bit | fused 2-bit | quantization effect |
|---|---|---|---|---|
| 512 | 4.64 ± 0.04 | 6.34 ± 0.01 | 5.95 ± 0.22 | 0.73× |
| 2048 | 11.26 ± 0.09 | 59.03 ± 20.86 | 12.05 ± 0.55 | 0.19× |
| 8192 | 362.36 ± 50.69 | 32.60 ± 2.54 | 151.28 ± 49.05 | **11.11×** |
| 16384 | 795.11 ± 100.48 | 51.02 ± 4.02 | 48.80 ± 4.81 | **15.58×** |

The sign flips: once the cache genuinely comes from DRAM, 4-bit leads by 11–15×
at long context. **So the real claim is conditional — the fused kernel wins
exactly when the KV cache does not fit in L2, and loses ~2× when it does.**

⚠️ **These cold numbers are not yet quotable.** Variance is ±50–100 µs and the
ctx=2048 row (59 µs, slower than ctx=8192) is not credible. Suspected
power/thermal throttling on an 80 W laptop part. A clock-monitored re-run is the
first task in `docs/next_steps.md`.

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
| 512 | 2 | ≥ 0.9999998 | ≤ 5.98e-04 | 6.558e-01 | 6.560e-01 |
| 2048 | 2 | ≥ 0.9999996 | ≤ 8.71e-04 | 8.110e-01 | 8.105e-01 |

Thresholds are asserted, not eyeballed: `cosine ≥ 0.99999`, `rel L2 ≤ 5e-3`, and
the kernel's end-to-end error must be within 1.5× of the PyTorch baseline's.
The suite also covers pack/unpack roundtrip exactness, a half-quantization-step
reconstruction bound, GQA and MHA shapes, group sizes 16–128, block sizes
16–128, split counts 1/2/5/17/64 (uneven on purpose), S=1, extreme scores, and a
worst-single-element check that aggregate cosine would hide.

Test inputs inject 1% heavy-tailed outliers at 8× magnitude, because pure
Gaussian noise is an unrealistically easy input for a quantizer and would
flatter these numbers.

---

## What is not true (yet)

Be specific about what is *not* solved:

1. **"The fused kernel is Nx faster than PyTorch."** Misleading. Most of that
   ratio is flash-decoding, which has nothing to do with quantization. Use the
   decomposition table.
2. **"Low-bit KV makes decode faster."** False in the L2-resident regime — it is
   ~2× slower there. True only when the working set exceeds L2, and that
   measurement is not yet stable enough to publish.
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
7. **The adversarial audit is incomplete.** `audit_claims.py` does bootstrap
   CIs over raw per-sample timings, but does not yet know the fp16 control
   exists, so it cannot see the finding above.

---

## Reproducing

**Hardware actually used:** NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell,
sm_120, 26 SMs, 8 GB, 34 MB L2), Windows 11, driver 610.47. This is a
thermally-limited 80 W laptop part sharing the GPU with the desktop compositor —
which is exactly why the cold-regime variance is flagged above.

**Stack:** `torch 2.12.0+cu130`, `triton-windows 3.8.0.post28`, CUDA 13.0,
`transformers 5.16.1`. Full pins in `requirements.txt`.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe -m pytest test_correctness.py -q   # ~26 s
.venv/Scripts/python.exe benchmark.py --quick                # ~37 s
.venv/Scripts/python.exe benchmark.py --samples 50           # full suite
.venv/Scripts/python.exe audit_claims.py                     # reads results/benchmark.json
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
| `audit_claims.py` | adversarial self-audit with bootstrap CIs. Incomplete — see above. |
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
