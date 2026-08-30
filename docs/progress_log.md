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
