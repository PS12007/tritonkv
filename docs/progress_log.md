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
