# Current status

Last updated: 2026-11-06. Branch: `optimize/inblock-k`. Current evidence checkpoint: `0d8fec0`. Measured source/executable checkpoint: `2b7d4fc` (`Preserve aligned naive baseline code generation for fair comparisons`). Initial inblock: `654b5e2`. Gated scripts: `597b7bf`. Historical baseline: `8a68227`. Documentation commits may follow. Verify `git status` and `git log -1` before continuing.

## Implemented and verified

- `naive`: retained one-thread-per-output fused baseline with aligned/tail variants. Aligned code matches all 336 original SASS lines, including 168 instructions and scheduling control words, after display-whitespace normalization. Tail guards apply only when K is not divisible by eight.
- `inblock`: block `(32,8)`, eight contiguous packed-word K slices, 1024-byte FP32 shared reduction, one FP16 output write. No atomics/global partials. FP16 reconstruction is preserved.
- Original-weight cuBLAS, pre-dequantized cuBLAS, dequantization, and unfused paths remain available. Shared layout/packing supports partial final words/groups and arbitrary positive N. Group size remains a multiple of eight.
- Local CPU build/CTest passed. c58 CUDA sm_120 Release build, small-case GPU gate, and four zero-error Compute Sanitizer memcheck/synccheck runs passed for the repaired executable. All 48 final timing CSV rows pass, with bit-exact reconstruction of all 16,777,216 FP16 weights per main run. Main M=128 output checks sample 64 rows.

## E001 outcome

[E001](EXPERIMENTS.md#e001--in-block-k-splitting) is complete. Keep inblock separately selectable for the demonstrated M=1 case. Reject a universal naive replacement or automatic dispatch based on this experiment. Ratios below are retained-naive median divided by inblock median, so values above 1 favor inblock.

| M | Store-flush speedup | No-flush speedup |
|---|---:|---:|
| 1 | 3.853 | 2.886 |
| 8 | 0.883 | 0.992 |
| 32 | 0.812 | 0.807 |
| 128 | 0.755 | 0.748 |

M=1 store-flush latency falls from 206.496 to 53.600 us and no-flush from 110.192 to 38.176 us. Every larger M regresses in this final series. M=1 profiler active warps and issue activity increase without a material reduction in DRAM read bytes. At M=128 already-high activity does not prevent regression. Stall/instruction explanations remain hypotheses, not isolated causes.

Final evidence: `results/inblock_c58_20261106_final/`. Run `python3 results/inblock_c58_20261106_final/analyze.py` and `python3 results/inblock_c58_20261106_final/verify_sass.py` to reproduce artifact assertions and summaries. All required work for E001 is complete.

## Lessons and limits

The pilot `results/inblock_c58_20261106/` is preserved. Its inner-loop tail guard slowed aligned naive code and inflated pilot speedup ratios. A matched control isolated the guard regression, then the aligned baseline was repaired and the complete series rerun. See E001 for the failed attempt and evidence. Historical `results/baseline_*` and `results/ncu/` remain unchanged.

Only one GPU/node, seed, and final timing series were measured. Clocks were observed, not locked. Sanitizer covers small cases, M=128 correctness is sampled, and no steady-state warm Nsight profile was collected. The application store flush does not prove input eviction. Historical `naive_M1_hot` Nsight evidence remains inconclusive for a warm comparison. Modeled `eff_gbps` is not measured DRAM bandwidth.

Remote source: `/home/hpuvvad/csc548-w4a16-inblock-20261106`. Job 328410 on c58 was verified during this session. Revalidate ownership/state/time/GPU before reuse. Keep `/home/hpuvvad/csc548-w4a16` and the user's allocation intact.

## Single next task

Implement and evaluate the user-authorized [E002](EXPERIMENTS.md#e002--candidate-aligned-k-specialization), preserving E001 as a separately selectable comparison. Pre-implementation criteria are recorded in E002. Current repository is clean at 84afdbf before the experiment. Allocation 328410 was revalidated RUNNING on c58 with RTX 5060 Ti, driver 615.71.09, and more than five hours remaining on 2026-11-06. Revalidate before later reuse.
