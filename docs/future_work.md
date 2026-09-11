# Future work: what this project is worth, and what to do next

Written 2026-09-10, after the L2 sweep and before any volunteer data. This is a
plan for *later*. Nothing here is started. Read `next_steps.md` for what is
in flight now.

---

## Part 1 — What the project is, in plain words

When a chatbot writes a reply, it writes one word at a time. To pick each new
word it re-reads its **notes on everything written so far** (the "KV cache").
Those notes grow with the conversation, and re-reading them is the slow part.

The obvious idea is to **shrink the notes**. Store each number in 4 bits
instead of 16, so there's roughly 3× less to read. This project is a GPU
program (a "kernel") that reads the shrunk notes directly without unpacking them
first, plus a very careful measurement of whether that actually makes things
faster.

It found three things:

1. **The big speedup wasn't the shrinking.** The kernel is up to 38× faster than
   PyTorch, but almost all of that comes from *splitting the work across the
   whole GPU*. PyTorch doesn't do that for one word at a time. Shrinking the
   notes, measured fairly against an identical unshrunk kernel, is worth
   0.7–1.5×.
2. **Shrinking only helps when the notes don't fit in the GPU's small fast
   memory (L2).** Think of a desk and a filing cabinet. If the notes fit on your
   desk, compressing them just adds unzipping time, so it's 1.2–1.4× *slower*.
   Once they're in the filing cabinet across the room, carrying a third as much
   paper saves trips, so it's 1.2–1.9× *faster*. The flip was predicted in
   writing and measured at **1.04× the L2 size** (prediction: 1.0×).
3. **On a laptop GPU, measuring is harder than building.** The laptop's speed
   swings 9× on its own. Ten times the headline number turned out to be wrong,
   and every time the fault was in the measurement, never the kernel. Most of
   the repo is the tooling that caught those mistakes.

---

## Part 2 — Did this produce anything useful? An honest answer

**What is *not* new.** Kernels that read a compressed KV cache already exist
(research like KIVI, KVQuant and QServe; engines like vLLM and FlashInfer).
Splitting decode across the GPU ("flash-decoding") is known. "Compression helps
when you're limited by memory speed" is known in principle. This project didn't
invent a technique.

**What it adds, in order of how useful it is to someone else:**

| | contribution | who it's useful to | how useful |
|---|---|---|---|
| 1 | **The control-kernel attribution.** "N× faster than PyTorch" for a compressed-cache kernel mostly measures parallelism, shown with an identical unquantized kernel. | Anyone reading or writing a speedup claim for KV quantization. Comparing against an unsplit baseline is an easy mistake to make without noticing; this project made it first. | **High as a lesson**, and easy to reuse: build the control. |
| 2 | **When compression pays, located precisely.** The flip at ~1× L2; the "hump" (1.9×) where fp16 has spilled but 4-bit still fits; convergence to the DRAM ratio once both spill. Pre-registered and landed. | People deciding whether KV quantization is worth it for *speed* on a given GPU. Modern consumer GPUs have 24–96 MB of L2, versus 3–6 MB two generations ago, so this matters more than it used to. | **Medium.** A clean, clearly stated result, not a breakthrough. Becomes much stronger if other GPUs confirm it (the volunteer test). |
| 3 | **Honest benchmarking on a laptop GPU without admin rights.** Clock monitoring, warming up *memory* and not just compute, gates, between-run intervals, pre-registration. | Anyone benchmarking kernels on consumer hardware, which is most hobbyists and students. | **Medium, and currently locked inside this repo.** See Part 3, dead end C. |
| 4 | **The tile-addressing trick** for feeding packed codes to `tl.dot`. | Triton programmers. | Low–medium. A nice piece of craft. |

**Overall:**
- **As research:** a small, solid, carefully verified result. That's
  blog-post or workshop-note territory today, and more if the cross-GPU data holds.
- **As a portfolio piece:** strong. The rare thing it demonstrates is rigor:
  predicting results before measuring, catching your own errors, and reporting
  failures as loudly as wins. That is exactly what ML-systems and GPU-performance
  teams look for, and it is uncommon even in published work.
- **As a tool people can use today:** not yet. It's one attention layer, one
  prompt at a time, with no real model behind it. Direction A fixes that.

**The highlight, in one sentence:** *a pre-registered measurement showing that
KV-cache compression speeds up LLM decoding only once the cache outgrows the
GPU's L2, with the crossover landing at 1.04× L2 against a written prediction
of 1.0×, and with the popular "N× faster" framing shown to be mostly a
parallelism effect.*

---

## Part 3 — The dead ends, and what can be done about each

### Dead end A — "make the kernel faster" has stopped paying on this laptop

The last four ideas gained nothing: fp16 dequantization (80 fewer instructions,
0% faster), zero-point folding, the narrow-load trick on codes (slower), and a
wider tuner. The kernel is limited by how many **memory loads** it can issue,
not by arithmetic.

**What can be done:**
- **Per-channel key quantization (direction B) may break through it.** A side
  effect is that the key scales/zeros become *one row per tile* instead of one
  per token, which cuts metadata loads again. The last time metadata loads were
  cut, it bought 1.29×. This is the one kernel idea left with a mechanism
  behind it.
- **Wait for the volunteer data.** A desktop card with more SMs and different
  memory may not be load-bound in the same way. If the bottleneck moves, the
  "dead end" was laptop-specific.
- Otherwise, accept it. A kernel at its hardware's limit is a result, not a
  failure.

### Dead end B — why the GPU sometimes drops its memory speed

Five explanations were tested and all failed. The repo stopped on purpose,
because testing more ideas against the same data finds false patterns.

**What can be done:** only a *new experiment built for the question*, e.g. hold
everything fixed and vary one thing (idle gaps between kernels, power mode,
background load). It would be interesting, but it doesn't change any number the
project reports, because the quality gate already catches the bad cases.
**Recommendation: leave it parked.** Lowest value of anything here.

### Dead end C — ~11,000 lines of measurement tooling that only this repo can use

This is a dead end only in the sense that the value is trapped.

**What can be done:** extract it into a small standalone package, say
`honestbench`, that anyone can wrap around their own Triton or PyTorch kernel:

- the background `nvidia-smi` clock monitor
- the compute-*and*-memory warm-up (the non-obvious fix from this project)
- the per-row quotability gate and the dispersion tier
- CUDA-graph timing in L2-resident and DRAM-resident regimes, sized to the
  card's own L2
- the between-run comparison and the blocked-bootstrap ratio CI

That turns sunk cost into a thing other people can star, use and cite, and it
markets well ("I benchmarked my kernel on a laptop and caught myself being wrong
10 times; here's the tool"). Effort: 2–3 sessions, mostly moving code and
writing a README with one worked example.

### Dead end D — 2-bit is correct but useless (72% error)

This is direction B below.

---

## Part 4 — Direction A: run it inside a real model

### Why

Everything so far is **one attention layer, with made-up data, one prompt at a
time**. The obvious question from anyone on X, or in an interview, is: *"does it
make a real model faster?"* Right now the honest answer is "unknown". Answering
it turns a benchmark into a demo.

### The surprise that makes this worth doing: the "costs in L2" case probably never happens in a real model

In a real model, generating one token means reading **all the model's weights**
(Qwen2.5-1.5B: ~3.1 GB in fp16) plus **every layer's KV cache**. Streaming 3 GB
through a 33.6 MB L2 flushes it about 90 times per token. So by the time any
layer's attention runs, its KV cache cannot still be sitting in L2.

**So in real decoding, attention should always behave like the "DRAM-resident"
regime**, the one where 4-bit **wins** (1.2–1.5× from ~2k tokens). The "4-bit is
slower in L2" result would be an artefact of benchmarking one layer in
isolation.

If that holds, the practical message becomes simpler and better: *in a real
model, 4-bit KV speeds up attention from ~2k tokens of context; the L2 penalty
is a microbenchmark effect.* **Pre-register this before building anything.**

### How big the end-to-end win should be (estimate, to be pre-registered)

Per generated token, batch 1, on this laptop:

| piece | fp16 KV | 4-bit KV | where the number comes from |
|---|---|---|---|
| read the weights | ~9–10 ms | same | 3.1 GB at ~300–340 GB/s (measured copy: 343 GB/s) |
| attention, 28 layers @ 16k ctx | 28 × 55.6 µs = **1.56 ms** | 28 × 38.6 µs = **1.08 ms** | `key_numbers.md`, DRAM-resident |
| kernel launch overhead, unless CUDA graphs | ~28 × 60–95 µs ≈ **1.7–2.7 ms** | same | `benchmark.json` `launch_overhead_ms` |

So at 16k context, **4-bit saves ~0.5 ms out of ~11–13 ms, about 4%**. At 32k,
maybe 7–9%. That's small, and it has to be said up front. **The real
end-to-end benefit is memory, not speed.** After ~3 GB of weights, an 8 GB card
has ~4–5 GB left for KV, which is ~150k tokens in fp16 against ~480k in 4-bit.
That means 3.2× more context, or 3.2× more simultaneous conversations.

Two traps, flagged now:

1. **Launch overhead can eat the whole gain.** On Windows each Triton launch
   costs 60–95 µs, and the fused kernel is two launches per layer. Without CUDA
   graphs (a static cache plus `torch.compile(mode="reduce-overhead")` or manual
   graph capture), the 1.7–2.7 ms of launch overhead is **3.5–5.5× the 0.5 ms
   saved**. Graphs are not optional here.
2. **Against stock Hugging Face on Windows it will look enormous, and that's
   the split again.** Stock attention at 16k costs ~1.46 ms *per layer* here,
   ~41 ms per token over 28 layers, so the fused kernel could show ~4×
   end-to-end against stock `transformers`. Report it only next to the
   fp16-control-kernel number, the same rule the whole project runs on.

### Steps

1. **Dump real K/V from a real forward pass (cheap, do first).** Run Qwen2.5-1.5B
   on a few real long documents, save each layer's K and V at several context
   lengths. This one step also feeds direction B, because real keys have
   structure that the random test data doesn't.
2. **Accuracy on real data.** 4-bit vs fp16 KV: attention-output error per layer,
   then **perplexity** on real text (e.g. a slice of WikiText-2 or a long book
   chapter). Pre-register an acceptable perplexity increase, e.g. ≤ 1–2%.
3. **Plug the kernel into the model.** In `transformers` 5.x, register a custom
   attention function (`AttentionInterface`) and a custom `Cache` class that
   quantizes each new token as it's appended (easy with the current per-token
   grouping). Prefill stays on fp16 SDPA and quantizes the result; decode
   (`q_len == 1`) calls the fused kernel.
4. **Make decode graph-capturable:** a static-size cache, preallocated
   workspaces (the kernel already takes `_workspace=`), and no Python-side
   allocation per step.
5. **Measure tokens/sec** at contexts {512, 2k, 8k, 16k, 32k}, three ways (stock
   HF attention, the fp16 control kernel, fused 4-bit), with the clock gate and
   the between-run rule. Report the 4-bit gain against the **control**, and the
   memory saving.
6. **Test the pre-registered prediction:** inside the real model, the per-layer
   attention times should match the *DRAM-resident* benchmark rows, not the
   L2-resident ones, at every context.

### Success looks like

- Perplexity within the pre-registered bound.
- The "attention is always DRAM-resident in a real model" prediction holds (or
  fails loudly).
- An honest end-to-end table, e.g. "4-bit KV: +4% tokens/sec at 16k, +8% at 32k,
  3.2× more context in the same memory". The headline is the memory; the speed
  is a bonus.

**Effort:** 3–5 sessions. **Risk:** `transformers` API churn; CUDA-graph
capture on Windows; Qwen2.5-1.5B only supports 32k context natively.

**Marketing value: the highest of anything here.** "I put my kernel in a real
model; here's what actually changed" is the post people share.

---

## Part 5 — Direction B: make 2-bit usable (per-channel keys, the KIVI idea)

### Why 2-bit fails today

Right now each token's 128 numbers are split into groups of 32, and each group
gets one scale and zero-point ("per-token" grouping). With only 4 levels
(2 bits), one large number in a group sets the scale, and the small numbers all
round to the same level. Result: **66–81% error**. The kernel is correct, but
the compression throws away too much.

**The KIVI paper's observation:** in real models, **keys have "outlier
channels"**, i.e. certain dimensions that are large for *every* token. Grouping
per token puts those big channels in the same group as normal ones, which is the
worst case. Grouping **per channel** (along tokens, within one dimension) gives
each outlier channel its own scale. Values don't have this structure, so they
stay per-token.

### The catch, and how KIVI handles it

A per-channel group spans G tokens (e.g. 32). You can't finalize a group until 32
tokens have arrived. KIVI keeps the **most recent R tokens in fp16** (a
"residual window") and quantizes them in blocks of G once the window fills.

### What changes in the kernel

1. **Key metadata becomes one row per tile.** With `BLOCK_N == G == 32`, the
   key scale/zero for a whole tile is a single `(D,)` vector, broadcast down the
   tokens. That's even fewer metadata loads than the current broadcast path,
   and the last metadata-load cut bought **1.29×**. So this could make 4-bit
   *faster* too. It's the most promising fix for dead end A.
2. **Values stay exactly as today** (per-token).
3. **The fp16 residual tail** (the last < R tokens) goes through the existing
   fp16 control kernel as one more "split", and the existing combine kernel
   merges it. Both pieces already exist.
4. **Packing for keys** changes from split-P along `head_dim` to packing along
   tokens (or keep split-P and change only the metadata indexing; decide after
   measuring registers, remembering the 244 → 128 lesson).

### Steps

1. **Measure on real keys first (no kernel work).** Using the real K/V dumped in
   direction A step 1, compute per-token vs per-channel 2-bit key error in plain
   PyTorch. **Critical:** the current test data puts outliers at *random
   positions*, not in fixed channels, so per-channel grouping would *not* help
   on it, and a synthetic test would wrongly say the idea fails. Real keys only.
2. **Pre-register** the expected error reduction and a perplexity bound before
   looking at the real-key numbers.
3. Extend `quantize.py` with per-channel keys and the residual window (pure
   PyTorch; this stays the correctness ground truth).
4. Kernel: per-channel key metadata path, keep the old path behind a flag as a
   permanent control row (house rule), and add bitwise and tolerance tests.
5. Measure: accuracy (perplexity with direction A's harness) and speed with
   `benchmark.py` + `between_run.py`, both bit widths.

### Success looks like

- 2-bit with per-channel keys reaches perplexity close to 4-bit per-token. If so,
  that's 5.3× smaller than fp16 at usable quality, against 3.2× today.
- Bonus: per-channel metadata makes the kernel faster at 4-bit too. That's
  measurable, and it counts as a win only if it survives `between_run.py`.

**Effort:** 3–5 sessions, after direction A's step 1. **Risk:** the residual
window complicates graph capture (variable tail length); packing along tokens
may cost registers.

---

## Part 6 — Suggested order

1. **Now:** collect volunteer runs (passive; `docs/volunteer_message.md`).
2. **Direction A, step 1:** dump real K/V. It's cheap and unblocks both A and B.
3. **Direction B, step 1:** per-channel vs per-token 2-bit error on real keys,
   pre-registered. CPU-friendly, and quick to learn whether B is worth it.
4. **Direction A, steps 2–6:** the real-model integration.
5. **Direction B, kernel work**, if step 3 said yes.
6. **Dead end C:** extract `honestbench` whenever there's a quiet week.
7. Leave dead end B parked.

Every step that produces a number gets its prediction written down and committed
before the number exists. That habit is this project's best idea.
