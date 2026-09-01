# Where to pick this up

Rewritten 2026-09-01 after the clock-monitored re-run. Read `progress_log.md`
first for *why* things are the way they are; this file is only *what is left*.

## State: what works right now

Everything below is verified on this machine, not assumed.

- Environment is healthy. `.venv` has `torch 2.12.0+cu130`, `triton-windows
  3.8.0.post28`, matplotlib, pytest, transformers. CUDA is live on an
  **RTX 5060 Laptop, sm_120, 26 SMs, 8 GB, 33.6 MB L2, 80 W**.
  Run things as `./.venv/Scripts/python.exe ...` — no activation needed.
- `python -m pytest test_correctness.py -q` → **66 passed in ~26 s**.
- `python benchmark.py --samples 50` → ~4.5 min, clock-monitored, writes
  `results/benchmark.json`. Last run: **22/32 rows quotable**.
- `python audit_claims.py` → 63 claims, `results/audit.{md,json}`.
- `python make_plots.py` → 8 figures in `docs/plots/` (committed).
- README, `docs/key_numbers.md`, `docs/thread_outline.md` all reflect the
  corrected numbers.

**The headline finding, in its current form:** the fused kernel's win over
PyTorch SDPA is almost entirely the flash-decoding split (9–25×), not the
quantization (0.51–1.32×). Quantization *costs* 1.4–2× when the KV cache is
L2-resident and pays 1.14–1.32× when the working set exceeds L2. The earlier
"11–15×" was a clock artefact — see the 2026-09-01 progress-log entry before
trusting any pre-2026-09-01 number in this repo.

## Open work, in order

1. **Close the 10 unquotable rows.** Six failed the boost-clock gate (the slow
   PyTorch baselines, whose long per-sample gaps let the clocks sag) and four
   failed dispersion at 6–7% IQR (`fp16_sdpa` at 8k/16k, `triton_fp16_control`
   at 8k). Two candidate fixes: raise `--samples` for the slow methods only, or
   interleave a tiny clock-keeper kernel between samples and verify it does not
   perturb the timing. Until then the "36× vs PyTorch" figure at long context is
   reported but not leaned on.

2. **Attack the issue-bound inner loop.** The 4-bit path reaches ~64 GB/s while
   the fp16 control does ~410 GB/s out of L2 — the kernel is issue-bound in the
   shift/mask/convert/fma chain, not bandwidth-bound. That chain is exactly what
   costs the 1.4–2× in the L2-resident regime. The promising restructure is to
   fold the zero-point out of the inner loop:
   `q·(code·scale + zero) = scale·(q·code) + zero·Σq`, since `Σ_{d∈g} q[d]` is
   precomputable per (head, group). It needs a segmented dot rather than one
   `tl.dot`, so it is not a small change. If it lands, re-run the attribution —
   it could move the L2-resident regime from "costs 2×" to "free", which would
   change the conclusion of the whole project.

3. **`gs=128` is anomalous and unexplained.** 25.4 / 25.6 / 26.6 µs at
   gs=16/32/64, then 93 µs at gs=128. Worth a look; it may be a tile-shape or
   register-pressure cliff rather than anything about the format.

4. **2-bit deserves a decision, not a table row.** It is numerically unusable
   (rel L2 ≈ 0.7) under per-token grouping along `head_dim`. KIVI's result is
   that keys want per-channel grouping. Either implement per-channel keys and
   re-measure, or state plainly that 2-bit is out of scope and stop benchmarking
   it. Right now it is carried at full cost in every table without being a
   candidate for use.

5. **Cross-check on a non-laptop GPU if one becomes available.** Everything here
   is one card with a 9× clock range and a 33.6 MB L2. The L2-residency
   conditional is stated in terms of L2 size precisely because that is the axis
   expected to move; nothing tests that yet.

## Things that are done and should not be redone

- Clock monitoring and the quotability gate (`benchmark.py`).
- The fp16 control kernel and the attribution claims (`audit_claims.py`).
- Median/IQR reporting, raw hot-regime samples in the JSON.
- Plots, key numbers, thread outline, README.

## Session hygiene note

A Claude Code session in this directory once survived its terminal being closed
and kept writing files while a second session worked in the same tree. If files
appear that nobody in the current session wrote, check for orphaned
`claude.exe` processes before assuming the working tree is yours alone.
