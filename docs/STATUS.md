# Current status

Last updated: 2026-11-06. Branch: `optimize/inblock-k`. Current evidence checkpoint: `c7ea4b4`. Measured E002 code checkpoint: `91bd042`. E002 criteria checkpoint: `dc37ec6`. E001 evidence checkpoint: `0d8fec0`. E001 measured source/executable checkpoint: `2b7d4fc` (`Preserve aligned naive baseline code generation for fair comparisons`). Initial inblock: `654b5e2`. Gated scripts: `597b7bf`. Historical baseline: `8a68227`. Documentation commits may follow. Verify `git status` and `git log -1` before continuing.

## E002 outcome

[E002](EXPERIMENTS.md#e002--candidate-aligned-k-specialization) is complete and accepted. Pre-implementation criteria were committed in `dc37ec6`: at least 1.05 E001/E002 median speedup at M=32 and M=128 under both policies, no more than 5% M=1 latency regression, unchanged FP64 tolerance, bit-exact W_hat and E001/E002 outputs, and zero selected sanitizer errors. Every condition passes in the measured series.

`inblock_specialized` retains the eight-slice mapping with compile-time aligned/tail guards. E001 source remains unchanged. Rebuilt E001 and E002's tail variant each match all 480 saved E001 SASS lines including encodings/control words. The aligned variant emits 216 static instructions including padding versus E001's 240. Separate profiles measure 23.0878% fewer dynamic executed warp instructions at both M=1 and M=128. These are distinct static/dynamic measurements.

Local CPU/CTest, target CUDA build, expanded small-case checks, and six zero-error sanitizer runs pass. All 56 timing rows pass with zero reconstructed-weight mismatches. All eight main E001/E002 comparisons check every M*N output bit, including all 128 rows at M=128, and match exactly. The independent FP64 reference still samples 64 rows at M=128.

| M | E001/E002 store-flush speedup | E001/E002 no-flush speedup |
|---|---:|---:|
| 1 | 1.040 | 1.283 |
| 8 | 1.266 | 1.327 |
| 32 | 1.305 | 1.327 |
| 128 | 1.320 | 1.322 |

M=128 store-flush E001/E002 latency is 3452.928/2614.944 us, and no-flush is 3445.312/2605.632 us. E002 remains slightly slower than same-executable naive at M=128. It is accepted against E001's criterion, not as a universal winner. Preserve all selectors, with no automatic M-based kernel selection or further optimization. E002 retains its required K-divisibility dispatch.

Remote E002 source: `/home/hpuvvad/csc548-w4a16-e002-20261106`. Evidence: `results/e002_c58_20261106/`. Run its `analyze.py` and `verify_sass.py` for consistency/criteria and SASS assertions. `criteria.json`, `comparisons.csv`, and `profile_summary.csv` contain generated summaries. All required E002 work is complete.

## Implemented and verified E001

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

Both experiments used one GPU/node and seed, with one complete final timing series per experiment. Clocks were observed, not locked. Sanitizer covers small cases, M=128 correctness is sampled, and no steady-state warm Nsight profile was collected. The application store flush does not prove input eviction. Historical `naive_M1_hot` Nsight evidence remains inconclusive for a warm comparison. Modeled `eff_gbps` is not measured DRAM bandwidth.

Historical E001 remote source: `/home/hpuvvad/csc548-w4a16-inblock-20261106`. Job 328410 on c58 was verified during this session. Revalidate ownership/state/time/GPU before reuse. Keep `/home/hpuvvad/csc548-w4a16` and the user's allocation intact.

## Single next task

Await the user's next requested experiment. Reuse of weights across M rows is a candidate to inspect, motivated by persistent large-M L2 traffic and E002's near tie with naive, but it is not an isolated bottleneck conclusion or authorization to implement. Any new experiment must fix its hypothesis and criterion first. Revalidate allocation 328410 before later reuse and leave the user's allocation running.
