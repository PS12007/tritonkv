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

**Splitting the history is worth 10–26×. The quantization is worth 0.74–1.47×.**

**Hot regime (cache fits in L2), µs per decode step, CUDA-graph replay:**

| ctx | SDPA fp16 | Triton fp16 (control) | fused 4-bit | fused 2-bit | flash-decode effect | quantization effect |
|---|---|---|---|---|---|---|
| 512 | 46.7 | 3.5 | 4.7 | 4.8* | 13.3× | 0.74× |
| 2048 | 178.8 | 6.8 | 7.6 | 7.7 | 26.2× | **0.90×** |
| 8192 | 747.3 | 13.7* | 16.8 | 16.9* | 54.4× | 0.82× |
| 16384 | 1493.7* | 21.2* | 28.9* | 29.2* | 70.4× | 0.74× |

**Quantization makes the kernel ~1.1–1.35× slower here, not faster.** Nearly the
whole apparent win is the split.

**Cold regime (rotating working set, 3× L2 = 101 MB), µs per decode step:**

| ctx | SDPA fp16 | Triton fp16 (control) | fused 4-bit | fused 2-bit | flash-decode effect | quantization effect |
|---|---|---|---|---|---|---|
| 512 | 50.1 | 5.0 | 5.2 | 5.2* | 10.0× | 0.97× |
| 2048 | 184.1 | 12.7 | 10.4 | 9.9 | 14.5× | **1.22×** |
| 8192 | 756.4 | 31.9* | 25.3 | 21.8* | 23.7× | 1.26× |
| 16384 | 1501.6* | 58.8* | 40.1* | 36.2* | 25.5× | 1.47× |

`*` = did not pass the clock-verification gate (see below) and is not quoted as
evidence anywhere; every conclusion here rests on unstarred rows.

The sign flips: once the cache genuinely comes from DRAM, 4-bit leads — but by
**1.22–1.47×, not by an order of magnitude**.

**At ctx = 2048 the sign flip is now fully clock-verified**, with SDPA, the fp16
control and the fused kernel all passing the gate at the same context: **0.90×
L2-resident, 1.22× DRAM-resident**. Earlier versions of this claim rested on
rows where the control had failed the gate.

![What the quantization itself buys](docs/plots/quantization_effect_4b.png)

**So the real claim is conditional: the fused kernel pays for itself only when
the KV cache does not fit in L2, and costs ~1.1–1.35× when it does.** At 512
tokens it roughly breaks even in the cold regime — there is not enough history
to amortize anything.

### The inner loop was mostly loading the same 4 numbers over and over

The per-group scale and zero are `(BLOCK_N, head_dim/group_size)` in memory — 4
values per token at `group_size=32`. The kernel used to load them as
`(BLOCK_N, head_dim)` by indexing with `d // group_size`, **re-reading each
parameter 32 times**, four times per loop iteration (K and V, scale and zero).
Loading them at their real width and expanding in registers is bitwise
identical — asserted over 40 cases of (S × nbits × group_size), not merely
close:

| | instructions | registers | spills |
|---|---|---|---|
| gather (`d // group_size`) | 2245 | 244 | 0 |
| broadcast | **1653** | **128** | 0 |

Worth 1.16–1.48× L2-resident and 1.05–1.32× DRAM-resident, biggest where the
kernel is issue-bound rather than bandwidth-bound. The old path is kept as a
permanent benchmark row (`fused_gather_meta_*`) so the attribution stays
auditable.

**This refutes a claim that used to be on this page.** It said: *"Group size
barely moves it (25.4 / 25.6 / 26.6 µs at gs = 16/32/64), so the scale+zero tile
loads are not the cost."* The measurement was right and the inference was wrong.
In the gather path the load is indexed by `d // group_size` over the **full**
head dim, so it issues `BLOCK_N × head_dim` loads *whatever the group size is* —
group size changes how many distinct values are read, never how many
instructions are issued. The experiment varied metadata **bytes** and concluded
about metadata **instructions**, and a flat sweep is exactly what the expensive
version predicts.

### …and then the same experiment, run properly, refuted half of *that*

`sweep_group_size.py` re-runs the sweep on **both** paths at once, with the
prediction written down first: broadcast should now be sloped, because there
`group_size` really does set the load count; gather should stay flat. Gather was
flat. Broadcast was **also nearly flat** — 1.07–1.17× across an 8× range of
metadata loads. The prediction was wrong, and the right shape is *saturation*
(L2-resident, ctx = 8192, all rows clock-verified):

| path | metadata loads per tile | median |
|---|---|---|
| gather, gs=32 | 4096 | 22.0 µs |
| broadcast, gs=16 | 256 | 17.0 µs — 16× fewer loads buys **1.29×** |
| broadcast, gs=128 | 32 | 15.9 µs — a further 8× buys **1.07×** |

So metadata loads are a real cost and they stop being the *binding* cost about
an order of magnitude below where the gather path sat. The broadcast change was
worth 1.29× because it crossed that point, not because load count and time are
proportional. A version of this project that had only run the second half of
that table would have concluded metadata loads were free — which is exactly the
mistake the first sweep made, from the other side.

### The gs=128 outlier was a shared-memory layout conversion

The old sweep had a loose end nobody chased: 93 µs at `gs=128`, against ~26 µs
everywhere else. It reproduces, and it is now explained.

On the gather path at `gs=128` the kernel is **1.95× / 2.88× / 3.52×** slower
than at `gs=64` (ctx = 512 / 2048 / 8192, L2-resident, IQR ≈ 1%). It issues
*fewer* PTX instructions than `gs=64` (2415 vs 2989) and exactly the same number
of global loads, so it is not a load-count effect. What moves is shared memory:
at the swept config (`block_n=32`, 2 warps) `st.shared` goes **30 → 142**. The
same jump appears in all nine (`block_n`, `num_warps`) combinations checked by
`probe_gs128.py` — 16 → 72 at 4 warps, 40 → 264 at `block_n=128` — so it is a
property of the index, not of one tuning.

The cause is the degenerate index. When `group_size == head_dim`,
`tl.arange(0, D) // group_size` folds to all-zeros, and Triton gives the loaded
tile a layout that must be converted through shared memory before it can feed
the dequantize path. The redundant-load form is slow; the redundant-load form
with a *constant* index is much slower, and it gets worse with context because
the conversion is inside the loop. The shipped broadcast path is unaffected — at
`gs=128` it is the *fastest* cell in the table — so this is a fact about the
control, not about the kernel.

Two variants were measured and **rejected**, which is what bounds the claim:

- **Folding the zero-point out of the inner loop** (`scale·(q·code) + zero·Σq`,
  a per-group dot against the raw codes). Not faster at any context in either
  regime: 0.72–1.08× at 4-bit, 0.67–0.94× at 2-bit. The only cell above the
  1.05× bar (4-bit, DRAM-resident, ctx=8192) sits on a row that failed the
  clock/dispersion gate, and the audit now says so instead of quoting it. Kept
  as an option
  (`fold_zp=True`) only because it is *more accurate* — it never rounds a
  dequantized K value to fp16, so kernel error stays flat at 1.5e-4 instead of
  drifting 2.3e-4 → 7.7e-4 as context grows.
- **The same narrow-load trick applied to the packed codes.** Bitwise identical
  and a **loss** (0.69–0.96× at ctx ≥ 8192); registers go 128 → 223. The codes
  are needed at full width regardless, so expanding them from a narrow load adds
  a live tile without removing one. Reverted. The lesson generalizes less than
  it first looks: the win is specific to loads whose expanded form is redundant.

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
   maximum,
2. the timing's own IQR was ≤ 5% of its median, and
3. **its clock window holds at least 4 samples.**

The last full run: **25 of 48 rows quotable**. The rejects are named in
`results/benchmark.json` under `clock_monitoring.rejected_rows` and are starred
in the tables above. **Every one of them is dispersion; there are no clock
rejections left.**

#### The gate had a second failure mode underneath the first one

Adding the clock monitor caught throttling. It did not catch *not having looked
long enough to tell*, and a gate that answers "the GPU was boosted" from one
sample reads exactly like one that answers from twenty.

Two things were wrong, and the fix for the first exposed the second:

- **The ramp was being spent before the measurement began.** `warm_clocks()` ran
  in the driver, but the timing function then did warmup, CUDA-graph capture and
  priming replays before opening its clock window — a long CPU-bound stretch
  with the GPU near idle. A slow PyTorch baseline re-boosts inside its own first
  sample; a 14 µs kernel never does, so **the gate was penalising methods for
  being fast**, which is the opposite of the failure it was built to catch. The
  ramp now runs *inside* the timing function, after capture.
- **28 of 96 measurement windows were being judged on a single `nvidia-smi`
  sample** — including rows the attribution rests on. `nvidia-smi -lms 100`
  actually delivers ~9 Hz (109 ms median gap, measured), so a 30 ms measurement
  cannot earn evidence about clocks at all. Windows are now held open ≥ 1.5 s and
  must carry ≥ 4 samples, reported as its own failure mode rather than passing
  silently. Windows with ≤ 1 sample: **28 → 0**; the minimum is now 13.

An earlier attempt bounded that stretch by *sample count* rather than time,
which silently bound first for exactly the fast kernels that needed it — their
samples are cheap, so 600 of them is 0.26 s. The unit mattered.

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

`python -m pytest test_correctness.py -q` → **106 passed in ~123 s**.

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

The audit now also adjudicates each kernel change against its **own** control
rather than against PyTorch — `optimization.meta_broadcast` (the metadata
broadcast, versus the same kernel with the gather) and
`optimization.zero_point_fold` (which it marks `FALSE` on speed, since it clears
the 1.05× bar at no context in either regime).

Current run against `results/benchmark.json` (2026-09-01 19:00):
**67 claims — 21 TRUE / 25 TRUE BUT CONDITIONAL / 9 MISLEADING / 12 FALSE.**
Regenerate with `./.venv/Scripts/python.exe audit_claims.py` (~20 s) and read
`results/audit.md`. Two of those verdicts moved this session for reasons that
were in the auditor rather than in the kernel:

- `optimization.*` crashed on a tuple-indexing bug, so the per-optimization
  claims had never actually been generated.
- `optimization.zero_point_fold.4b` came out `TRUE BUT CONDITIONAL` on a
  DRAM-resident 1.08× at ctx=8192 — on a row that had failed the clock and
  dispersion gate. That claim now applies the same gate as its neighbour and
  reads `FALSE`, like the 2-bit one always did.

Historically the `FALSE` verdicts have been this project's own claims about the
L2-resident regime, and the audit is what puts them there.

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
| `audit_claims.py` | adversarial self-audit: bootstrap CIs over raw timings, attribution against the fp16 control, per-optimization claims with their own controls, and a clock-verification gate. |
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
