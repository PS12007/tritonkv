# Thread outline

A writeup of this project, structured as posts. The arc is deliberately not
"I made a fast kernel" — it is "I made a fast kernel, then found out the fast
part wasn't the part I was writing about." Numbers cited here are the
clock-verified ones in [`key_numbers.md`](key_numbers.md).

---

## 1. The hook

> I wrote a Triton kernel that does decode attention directly on a 4-bit KV
> cache. It's 36× faster than PyTorch. Almost none of that is the 4 bits.

Post the attribution figure (`plots/speedup_attribution_4b.png`) immediately.
Lead with the correction, not the headline — the headline is the setup.

## 2. The problem worth solving

A naive quantized KV cache round-trips through DRAM on every decode step:
dequantize K and V into fp16, then read them straight back into attention.
Per cached element-row that is 4.5·D bytes moved where a fused kernel moves
0.5·D. Decode attention is memory-bound, so that round trip is close to all of
the cost.

## 3. The interesting kernel bit

`tl.dot` needs a dense `(BLOCK_N, D)` tile, but a load of packed codes gives
`(BLOCK_N, D/P)` bytes and Triton cannot slice-assign into a tile. The usual
escapes — `tl.join` + `tl.reshape`, or P unrolled accumulators — cost a
shared-memory layout conversion or lean on fragile constexpr unrolling.

**The way around it is to not reconstruct the tile, but address it.** Codes are
packed split-P, so byte `j` holds dims `j, j+D/P, j+2D/P, …`. Build an index
vector over the full head dim, load byte `d % (D/P)` and shift by
`(d // (D/P))·nbits`. Each byte is loaded P times, but those loads hit the same
cache line — DRAM traffic is unchanged and the tile comes out dense, with no
reshape, no transpose, no unrolled accumulators.

Good standalone post even if the perf story went the other way.

## 4. The turn: build the control

The kernel is flash-decoding shaped — it splits the history across SMs. PyTorch's
SDPA does not, for `q_len == 1`. So "fused 4-bit vs SDPA" compares two changes at
once and credits the quantization for a parallelization win.

The fix is an experiment, not an argument: `kernels/fp16_decode_attn.py`, the
same kernel with the same split, the same online softmax, the same GQA
amortization, reading plain fp16. The only difference is dequantization.

**Result: the split is worth 9–25×. The quantization is worth 0.51–1.32×,
depending on where the cache lives.** Post `plots/quantization_effect_4b.png`.

## 5. The second turn: the first answer was wrong too

This is the best post in the thread, and it is about measurement.

The first version of that control experiment said quantization won by **11–15×**
once the working set exceeded L2. It was wrong. The GPU is an 80 W laptop part
that idles at 285 MHz and boosts to 3090 MHz — a 9× range, wider than the
effects being measured. The fp16 control is fast enough that its measurement
*finished while the GPU was still spinning up*, so the control looked ~12×
slower than it is.

Nothing in the benchmark could catch this, because it never asked what the
clocks were doing.

The fix, which is the transferable part:

- sample `nvidia-smi` in the background throughout the run;
- spin the GPU to ≥ 80% of max clock before **every** measurement (`-lgc` would
  pin them, but needs admin);
- attribute the clock window to the **sampling loop only** — warmup and
  CUDA-graph capture are excluded, so they neither look like throttling nor hide
  it;
- reject any row whose clocks sagged below 70% of max or whose own IQR exceeded
  5% of its median.

22 of 32 rows survived. The real answer is **1.14–1.32×**, and the honest version
of the finding is smaller and more interesting than the wrong one.

## 6. What is actually true

- **L2-resident: quantization costs 1.4–2×.** The unpack/shift/mask/fma chain is
  pure added work when there is no DRAM traffic left to save. At 16k the fp16
  path runs ~410 GB/s (L2) and the 4-bit path ~64 GB/s — issue-bound, not
  bandwidth-bound.
- **DRAM-resident: quantization wins 1.14–1.32×**, and only from 2k tokens up.
- **Memory is the unconditional win**: 470 MB → 147 MB for the whole-model cache
  at 16k, and 0.3 MB of transient workspace against 117 MB for
  dequantize-then-SDPA.
- **The kernel is exact**: it matches dequantize-then-attend to cosine
  ≥ 0.999999. All end-to-end error is the quantizer's — the kernel column and
  the PyTorch column agree to three decimals.
- **2-bit is not usable** — rel L2 ≈ 0.7 against the fp16 cache — even though it
  passes every kernel-correctness test. Correct implementation, unusable scheme.

## 7. The closing argument

The project ships an adversarial audit (`audit_claims.py`) that writes down every
claim and attacks it with bootstrap CIs over raw per-sample timings: 63 claims,
**10 of them FALSE**, all of them this project's own.

> The measurement infrastructure is the deliverable. The kernel is 700 lines;
> the reason to trust any number about it is the other 2000.

---

## What not to say

- No unqualified "Nx faster." Every timing carries its regime.
- No end-to-end tokens/sec — this is one attention layer, and nothing about
  whole-model throughput was measured.
- No claim about other GPUs. One card, one driver, one clock regime. The
  conditional is stated in terms of L2 residency because that is the axis
  expected to move.
- Don't quote the starred rows in `key_numbers.md`; they failed clock
  verification and exist to show what was rejected.
