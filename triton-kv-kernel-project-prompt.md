# Claude Code Prompt: Fused Triton Kernel for Quantized KV Cache Decode

Copy everything below into Claude Code as your project brief.

---

## Project Goal

Build and benchmark a fused Triton kernel that speeds up the **decode-time
dequantization + attention score computation** for a low-bit-quantized KV
cache, and prove the speedup with real, honestly-reported numbers on a
single consumer GPU.

This targets a specific, real, unsolved bottleneck: naive quantized-KV-cache
implementations dequantize the entire cached history back to float32 on
every decode step before computing attention. A fused kernel should
quantize-aware compute the attention score directly on packed low-bit data,
avoiding the full dequant pass.

## Constraints (keep this cheap and finishable)

- Single GPU only (whatever Claude Code has access to — even a T4/L4 on
  Colab, or a single consumer card). Do NOT design anything requiring
  multi-GPU tensor parallelism.
- One small dense model: Qwen2.5-1.5B-Instruct or Llama-3.2-1B-Instruct
  (both fit easily in <8GB VRAM in fp16).
- Context lengths to test: 512, 2k, 8k, 16k. Do not go past 16k — it's not
  needed to demonstrate the kernel win and burns GPU time for no benefit.
- Correctness/unit tests should run on CPU or the smallest available GPU in
  seconds, not minutes.
- Total expected GPU wall-clock time for the full benchmark suite: under
  30 minutes.

## Deliverables

1. **`quantize.py`** — group-wise 2-bit and 4-bit quantization/dequantization
   for KV cache tensors (key and value separately), with per-group scale and
   zero-point. Pure PyTorch reference implementation first — this is the
   correctness baseline everything else is checked against.

2. **`kernels/fused_decode_attn.py`** — a Triton kernel that, for a single
   decode step, computes attention scores directly from packed low-bit
   K/V without materializing a full-precision copy of the whole cache.
   Start with 4-bit (easier correctness), then attempt 2-bit as a stretch
   goal.

3. **`test_correctness.py`** — numerical correctness tests comparing the
   Triton kernel's output against the PyTorch reference (dequant-then-attend)
   using cosine similarity and max absolute error, at multiple context
   lengths and multiple random seeds. Report exact thresholds and pass/fail,
   not just "looks close."

4. **`benchmark.py`** — wall-clock and memory benchmark comparing:
   - Baseline: PyTorch reference (full dequant every decode step)
   - Fused kernel: Triton kernel
   at each context length (512/2k/8k/16k), reporting decode tokens/sec,
   peak VRAM, and kernel launch overhead separately from compute time.
   Run each configuration at least 5 times and report mean ± std — not a
   single run (this is what the TurboQuant repo got called out for: a
   speedup claim from N=1).

5. **`audit_claims.py`** — an adversarial self-audit, following this exact
   pattern:
   - List every performance claim the benchmark produces
   - For each, actively try to falsify it: is the speedup within noise?
     Does it hold at all context lengths or only some? Does correctness
     degrade in a way the cosine-similarity metric hides (e.g. check a
     few individual output tokens by hand, not just aggregate similarity)?
   - Report claims as: **True**, **True but conditional (state the
     condition)**, or **Misleading (explain why)** — no claim should be
     reported as a flat "True" without evidence.

6. **`README.md`** — structured like this:
   - What problem this solves and why it's non-trivial
   - Benchmark results table (mean ± std, all context lengths, both bit
     widths attempted)
   - Correctness results table
   - Adversarial audit summary table
   - Known limitations section — be specific about what's NOT solved
     (e.g. if 2-bit correctness fails, say so and explain why 4-bit was
     used for the main results)
   - How to reproduce (exact commands, exact GPU used, exact library
     versions)

## Process notes for Claude Code

- Build the PyTorch reference implementation and its correctness tests
  FIRST, before writing any Triton code. The reference is the ground truth
  the kernel is checked against.
- Get the Triton kernel correct before optimizing it. A correct-but-slow
  kernel is a fine intermediate milestone; a fast-but-wrong kernel is
  worthless.
- If the 2-bit kernel proves too hard to get numerically correct in
  reasonable time, stop and ship the 4-bit result with 2-bit documented as
  attempted-but-not-completed, rather than shipping incorrect 2-bit numbers.
- Pin exact library versions (torch, triton, transformers, CUDA) in a
  requirements file — the README's "how to reproduce" section must actually
  work for someone else.
- Do not use `gpu_memory_utilization`-style vLLM integration for this
  project — keep it a standalone kernel + benchmark harness so the whole
  thing stays small, understandable, and cheap to run. vLLM integration
  can be a documented "future work" item, not a requirement.

## Content generation (for documentation / X posts)

Alongside the code, generate raw material for a build-in-public thread and
a project writeup. Don't write the actual posts — draft the *material* so
a real narrative can be written from real numbers later, not invented
after the fact.

1. **`docs/progress_log.md`** — a running, timestamped log appended to at
   each major milestone (reference implementation working, kernel first
   passes correctness, first real speedup measured, audit complete, etc).
   Each entry: what was attempted, what happened, one concrete number if
   there is one. This becomes the source material for a "building X"
   thread — write it as it happens, not retroactively.

2. **`docs/plots/`** — generate actual chart images (matplotlib, saved as
   PNG) for: speedup vs. context length, VRAM usage comparison, and the
   correctness-vs-bit-width tradeoff. Charts are the highest-engagement
   content for a technical X post — prioritize getting these to look clean
   (labeled axes, clear baseline-vs-kernel comparison) over anything else
   in the docs folder.

3. **`docs/key_numbers.md`** — a short, plain-language pull-list of the
   3–5 single most postable facts from the whole project (e.g. "the fused
   kernel is Nx faster at 8k context, verified across 5 runs"). Every
   number here must trace back to a specific benchmark run in
   `benchmark.py` output — no rounding up, no cherry-picking the best of
   many runs without saying so.

4. **`docs/thread_outline.md`** — a bullet-point skeleton (not full prose)
   for a build-in-public thread: one bullet per tweet, in order (hook →
   problem → what was tried → what broke → final result → link to repo).
   Leave the actual writing/voice to be done by hand later — this is
   structure only, so nothing here should read like a finished post.

Keep all of this secondary to the actual code and correctness — if time
runs short, a working kernel with honest benchmark numbers in the README
is the deliverable that matters; the docs/ folder is what makes it easy to
turn into posts afterward, not a requirement for the project to be done.

## Definition of done

- 4-bit fused kernel passes correctness tests at all four context lengths
- Benchmark shows a real (not noise-level) decode speedup at at least one
  context length, honestly reported with error bars
- Adversarial audit is complete and included in the README, even if some
  entries say "misleading" or "condition not met"
- Everything runs end-to-end on a single GPU in under 30 minutes total
