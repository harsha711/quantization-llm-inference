# Current status

Last updated: 2026-11-06. Branch: `optimize/inblock-k`. Current evidence checkpoint: `a16cbc5`. E003 measured source/scripts: `25b2291`, implementation: `5a7bcaf`, predeclared criteria: `53bb695`. E002 evidence/code/criteria: `c7ea4b4` / `91bd042` / `dc37ec6`. E001 evidence/measured code: `0d8fec0` / `2b7d4fc`. Historical baseline: `8a68227`. Documentation commits may follow. Verify actual `git status` and `git log -1` before continuing.

## E003 outcome

[E003](EXPERIMENTS.md#e003--reuse-reconstructed-weights-across-input-rows) is complete. Both `inblock_m2` and `inblock_m4` meet the predeclared larger-M criterion, with unchanged group=128, quantization/layout, and per-row FP32 arithmetic. Each unpacked FP16-reconstructed weight feeds two or four row accumulators. Grid y is ceil(M/Rows), with separate K/M-tail variants and shared partials of 2048/4096 bytes. Existing naive, E001, and one-row E002 sources remain unchanged and selectable. Do not add automatic row-count dispatch.

All 90 final timing rows pass. Twenty row-reuse/E002 comparisons check every output bit, including every row at M=128, and match. All reconstructed weights match, six memcheck/synccheck runs report zero errors, and local CPU/CTest plus target CUDA build/gate pass. Small validation covers M={1,2,3,4,5,7} with aligned/tail K, odd N, empty slices, and zero weights. Independent FP64 checks still sample 64 rows at M=128. Both naive/E002 control variants match saved instruction/control encodings exactly after normalizing build-specific namespace symbols and display whitespace.

Speedups below are same-executable E002 median divided by candidate median. M=5 is a supplemental incomplete-row-tile point, not part of the standard four-M acceptance series.

| M | Two rows store flush | Two rows no flush | Four rows store flush | Four rows no flush |
|---|---:|---:|---:|---:|
| 1 | 1.001 | 0.822 | 0.816 | 0.508 |
| 8 | 1.482 | 1.481 | 1.605 | 1.515 |
| 32 | 1.553 | 1.563 | 1.718 | 1.694 |
| 128 | 1.570 | 1.574 | 1.745 | 1.737 |
| 5 (tail) | 1.208 | 1.202 | 1.078 | 0.961 |

At M=128 store-flush, E002/two-row/four-row medians are 2621.968/1670.464/1502.208 us. No-flush medians are 2605.632/1655.040/1499.712 us. Four rows beats same-executable naive by 1.719–1.804 across M=32/128 and both policies. M=1 no-flush regresses for both candidates. The two-row cold M=1 near tie is inconclusive across repeats. Four rows also loses to two rows at M=5 and regresses versus E002 there without flushing. Accept explicit reuse selectors for larger M, reject a universal replacement.

Twenty separate cache-invalidated profiles cover naive/E002/both candidates at all five M points. At M=128 two/four rows reduce dynamic executed warp instructions by 38.106%/57.207% and L2 bytes by 44.313%/66.680% versus E002. Aligned candidates use 39 registers/thread versus 34 for E002, while active warps remain above 97%. DRAM reads stay around 9.7–9.9 MB. These observations support useful reuse, without isolating instruction cost, traffic cost, independent accumulators, and reduced block count from one another. M=1 instruction counts increase instead. No warm steady-state counter series was collected.

Evidence is `results/e003_c58_20261106/`. Run its `analyze.py` and `verify_sass.py` to reproduce assertions and generated `comparisons.csv`, `criteria.json`, `profile_summary.csv`, `repeat_comparison.csv`, and `sass_summary.json`. Two complete 90-row event series use the same executable. The first is preserved in `results/e003_c58_20261106_attempt1/` because profiling then failed to match target-specific template spelling. The corrected full rerun is the acceptance series. Both timing series meet the larger-M criterion. Diagnostics and the correction are documented in E003. Remote source: `/home/hpuvvad/csc548-w4a16-e003-20261106`.

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

E001 and E002 used one GPU/node and seed, with one complete final timing series per experiment. E003 adds a second complete same-binary timing series on that same GPU/node and seed. Clocks were observed, not locked. Sanitizer covers small cases, M=128 FP64 correctness is sampled, and no steady-state warm Nsight profile was collected. The application store flush does not prove input eviction. Historical `naive_M1_hot` Nsight evidence remains inconclusive for a warm comparison. Modeled `eff_gbps` is not measured DRAM bandwidth.

Historical E001 remote source: `/home/hpuvvad/csc548-w4a16-inblock-20261106`. Job 328410 on c58 was verified during this session. Revalidate ownership/state/time/GPU before reuse. Keep `/home/hpuvvad/csc548-w4a16` and the user's allocation intact.

Historical analyzers that assert source hashes require the corresponding recorded-source checkout. Do not rewrite old manifests to make E001/E002 analyses pass against E003 source. Current E003 analysis matches the present source and its recorded manifest.

## Single next task

Await the user's next requested experiment. E003 is fully implemented, validated, measured, and documented. Reuse at M=1 and four-row tails have explicit revisit conditions in E003. Shape-based dispatch, larger row tiles, layout changes, Tensor Cores, and asynchronous transfers are candidates requiring a separate request and new evidence-based criteria. Revalidate allocation 328410 and the GPU before any further use. Leave the user's allocation and all original/E001/E002/E003 directories intact.
