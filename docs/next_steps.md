# Where to pick this up

Written 2026-08-30 ~19:55, mid-session. Read `progress_log.md` first for *why*
things are the way they are; this file is only *what is left*.

## State: what works right now

Everything below is verified on this machine, not assumed.

- Environment is healthy. `.venv` has `torch 2.12.0+cu130`, `triton-windows
  3.8.0.post28`, matplotlib, pytest, transformers. CUDA is live on an
  **RTX 5060 Laptop, sm_120, 26 SMs, 8 GB, 34 MB L2**.
  Run things as `./.venv/Scripts/python.exe ...` — no activation needed.
- `python -m pytest test_correctness.py -q` → **66 passed in ~26 s**. 4-bit and
  2-bit both pass.
- `python benchmark.py --quick` runs end to end (~37 s) and writes
  `results/benchmark.json`. `results/` is gitignored.
- `kernels/fp16_decode_attn.py` (the control) is correct: cosine ≥ 0.9999998
  against the fp32 reference at S = 1 … 16384.

## The one thing that must be done first

**Re-run the cold benchmark with clock monitoring, and do not quote any timing
number until it is stable.**

The cold/rotating numbers in the last progress-log entry carry ±50–100 µs of
variance, and the ctx=2048 row (fused 4-bit at 59 µs, slower than at 8192) is
not credible. Idle clocks were observed at 180 MHz SM / 810 MHz mem on an 80 W
laptop part, so power/thermal throttling is the prime suspect.

Concretely:

1. Sample `nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw
   --format=csv` alongside the run and record it into `results/benchmark.json`
   via `env_info()`. A run whose clocks moved is not a run worth quoting.
2. Close anything using the GPU (the desktop compositor and Chrome were both
   resident — 886 MiB and ~1 % utilization at idle).
3. Re-run `python benchmark.py --samples 50`. It was interrupted partway
   through its first full run; nothing from it was committed.
4. If variance stays high, raise the sample count and report **median + IQR**
   rather than mean ± std, and say plainly in the README that this is a
   thermally-limited laptop GPU.

## Then, in order

1. **Add attribution claims to `audit_claims.py`.** It is a good file (764
   lines, bootstrap CIs over raw per-sample timings) but it does not yet know
   `triton_fp16_control` exists, so it cannot see the project's most important
   finding. It needs claims for:
   - flash-decoding effect = `fp16_sdpa` ÷ `triton_fp16_control`
   - quantization effect = `triton_fp16_control` ÷ `fused_triton_4b`
   - the conditional: **the fused kernel wins only when the KV cache does not
     fit in L2, and loses ~2x when it does.** Any bare "Nx faster than PyTorch"
     claim should be emitted as `MISLEADING` on the evidence we already have.
2. **`docs/plots/`** — speedup vs context length, VRAM comparison,
   correctness-vs-bit-width. The speedup plot must show the hot and cold
   regimes as separate series; a single line would misrepresent the result.
3. **`docs/key_numbers.md`**, **`docs/thread_outline.md`**.
4. **`README.md`** — results table, correctness table, audit summary, and a
   "Known limitations" section that says the quantization *costs* ~2x when
   L2-resident, that FlashAttention was unavailable on this platform, and that
   this is one attention layer, not end-to-end tokens/sec.
5. **`requirements.txt`** — pin exactly what is in `.venv` (`pip freeze`), and
   note that `triton-windows` is Windows-only; on Linux the dependency is
   plain `triton`.

## Open questions worth a real answer

- **Why is the 4-bit path issue-bound at ~64 GB/s?** Group size barely changes
  it (25.4 / 25.6 / 26.6 µs at gs=16/32/64 for ctx=8192), so the scale+zero
  tile loads are not the cost — the shift/mask/convert/fma chain is. `gs=128`
  jumps to 93 µs, which is anomalous and unexplained; worth a look.
- A promising restructure: fold the zero-point out of the inner loop using
  `q·(code·scale + zero) = scale·(q·code) + zero·Σq`, since `Σ_{d∈g} q[d]` is
  precomputable per (head, group). It needs a segmented dot rather than one
  `tl.dot`, so it is not a small change — but it attacks exactly the chain that
  is costing the 2x.
- 2-bit is numerically fine and sometimes *faster* than 4-bit in the cold
  regime (48.8 µs vs 51.0 µs at ctx=16384). Worth confirming rather than
  assuming 4-bit is the headline configuration.

## Session hygiene note

A previous Claude Code session (PID 27180) survived its terminal being closed
and kept writing into this directory. It was killed at 19:38. If files ever
appear that nobody in the current session wrote, check for orphaned
`claude.exe` processes before assuming the working tree is yours alone.
