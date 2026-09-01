# Where to pick this up

Rewritten 2026-09-01 (evening) after the metadata-broadcast change and the
second round of clock-gate repairs. Read `progress_log.md` first for *why*
things are the way they are; this file is only *what is left*.

## State: what works right now

Everything below is verified on this machine, not assumed.

- Environment is healthy. `.venv` has `torch 2.12.0+cu130`, `triton-windows
  3.8.0.post28`, matplotlib, pytest, transformers. CUDA is live on an
  **RTX 5060 Laptop, sm_120, 26 SMs, 8 GB, 33.6 MB L2, 80 W**.
  Run things as `./.venv/Scripts/python.exe ...` — no activation needed.
- `python -m pytest test_correctness.py -q` → **106 passed in ~123 s**.
- `python benchmark.py --samples 50` → ~7.5 min, clock-monitored, writes
  `results/benchmark.json`. Last run: **25/48 rows quotable**, and every
  rejection is dispersion — there are no clock rejections left.
- `python make_plots.py` → 8 figures in `docs/plots/`.
- README, `docs/key_numbers.md`, `docs/thread_outline.md`.

**The headline finding, in its current form:** the fused kernel's win over
PyTorch SDPA is almost entirely the flash-decoding split (10–26× on quotable
rows), not the quantization. Quantization *costs* ~1.1–1.35× when the KV cache
is L2-resident and pays 1.22–1.47× when the working set exceeds L2. At ctx=2048
the sign flip is now fully clock-verified with every input passing the gate:
**0.90× L2-resident, 1.22× DRAM-resident.**

## Do this first

**`results/audit.{md,json}` are stale.** They were generated against the
previous `benchmark.json` and predate both the `fused_gather_meta_*` /
`fused_fold_zp_*` rows and the new `audit_optimizations()` section. Re-run:

```
./.venv/Scripts/python.exe audit_claims.py
```

Then re-check the claim counts quoted in the README ("63 claims: 24 TRUE / 20
CONDITIONAL / 9 MISLEADING / 10 FALSE") — those numbers are from the old run and
should not be quoted until regenerated. `make_plots.py` should be re-run too.

## Open work, in order

1. **Close the remaining dispersion rejects.** The clock half of the gate is
   solved; what is left is genuine timing dispersion, concentrated at ctx=512
   (measurements near the CUDA-event resolution floor, where a 5 µs median
   cannot easily hold IQR ≤ 5%) and at ctx=16384 (long windows in which the
   80 W part drifts thermally). Two honest options, and they point opposite
   ways: shorten windows to reduce drift, or lengthen them to average it out.
   Note the tension with the clock gate — `MIN_SAMPLING_SECONDS = 1.5` exists
   precisely to keep windows long enough to earn clock samples. A per-regime
   setting is probably the answer, and should be justified by measurement of the
   drift, not chosen by taste.

2. **`gs=128` and the group-size sweep need re-measuring.** The old sweep
   (25.4 / 25.6 / 26.6 µs at gs = 16/32/64, then 93 µs at gs=128) was run on the
   gather path, where group size cannot affect the metadata load count at all —
   so it could not have shown anything about metadata cost, and the flat result
   was misread as evidence that metadata was cheap (see the progress log). On
   the broadcast path `GS` genuinely sets the load count (`BLOCK_N * D/GS`), so
   the sweep should now be **sloped**, and the gs=128 outlier may or may not
   survive. A ready-to-run script is described in the progress log; this is the
   cheapest remaining experiment and it tests the mechanism directly.

3. **The attribution is only fully quotable at ctx ≤ 2048.** The 8192 and 16384
   rows still lose either `fp16_sdpa` or `triton_fp16_control` to dispersion in
   any given run, so the long-context sign flip is reported but leans on rows
   that did not all pass simultaneously. Fixing (1) fixes this.

4. **2-bit still deserves a decision, not a table row.** Numerically unusable
   (rel L2 ≈ 0.7) under per-token grouping along `head_dim`. KIVI's result is
   that keys want per-channel grouping. Either implement per-channel keys and
   re-measure, or state plainly that 2-bit is out of scope and stop benchmarking
   it. Right now it is carried at full cost in every table without being a
   candidate for use. (Note it is also where `fold_zp` is *worst*, 0.66×.)

5. **Cross-check on a non-laptop GPU if one becomes available.** Everything here
   is one card with a 9× clock range and a 33.6 MB L2. The L2-residency
   conditional is stated in terms of L2 size precisely because that is the axis
   expected to move; nothing tests that yet. The metadata-broadcast win should
   also shrink on a part that is less issue-bound.

## Things that are done and should not be redone

- Clock monitoring, the quotability gate, the inner ramp, and the minimum
  clock-sample requirement (`benchmark.py`).
- The fp16 control kernel, the `fused_gather_meta_*` control, and the
  `fused_fold_zp_*` row.
- The metadata broadcast itself, and the bitwise-identity tests that pin it.
- Median/IQR reporting, raw hot-regime samples in the JSON.

## Things that were tried and rejected — do not redo without new evidence

- **Zero-point folding for speed.** Implemented and kept (`fold_zp=True`) for
  its *accuracy*, but it is not faster at any context in either regime. An early
  ad-hoc A/B suggested otherwise; the harness disagreed and the harness wins.
- **Narrow-load + register broadcast for the packed codes.** Bitwise identical,
  measurably slower at long context (registers 128 → 223). Reverted.

## Session hygiene note

A Claude Code session in this directory once survived its terminal being closed
and kept writing files while a second session worked in the same tree. If files
appear that nobody in the current session wrote, check for orphaned
`claude.exe` processes before assuming the working tree is yours alone.
