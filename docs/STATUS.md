# Current status

E005 is now authorized and in progress. E004 checkpoint `cf8a419` was verified clean. The single active task is the bounded A/B/C launch-resource experiment, with predeclared gates and criteria in E005 of EXPERIMENTS.md. No E005 candidate has been launched yet.

Last updated: 2026-11-06. Branch: `optimize/inblock-k`. E004 measured benchmark source: `209442c`, complete timing evidence: `640e9e5`, profiler/provenance evidence: `528c430`, predeclared gates: `670b32d`. Documentation and analyzer commits may follow. Previous complete E003 handoff: `c1c6423`, E003 implementation/source: `5a7bcaf` / `25b2291`, evidence: `a16cbc5`. Verify actual git state before continuing.

## E004 outcome

[E004](EXPERIMENTS.md#e004--maintained-marlin-integration-and-diagnostic-comparison) is complete. Direct dense INT4 W4A16 Marlin from vLLM v0.33.0, exact commit `4ba787ec9959e566d3b587a796dff07673324da5`, runs on c58's RTX 5060 Ti sm_120. Official wheel and loaded-library hashes, Apache-2.0 source/notice archive, dependency freeze, build settings and environment are preserved. The isolated Python 3.11/Torch 2.14 environment is `/tmp/e004-hpuvvad-20261106`; shared project is `/home/hpuvvad/csc548-w4a16-e004-20261106`. No serving engine, model, fallback, external kernel patch, or new custom kernel optimization. All E001–E003 implementations/results remain intact.

Lossless repacking preserves every code/scale bit and reconstructed FP16 element. Marlin uses uint4b8/group 128, FP16 inputs/scales/reconstruction/output, FP32 accumulation/global reduction, atomic reduction disabled. All 384 screening rows pass unchanged all-output FP64 checks, including all 128 rows at M=128. Both 64-row repeated series and both 12-row eager/graph diagnostics also pass. Six retained-project sanitizer runs pass. The original default cuBLAS mode failed 2/524288 outputs at square M=128. A preserved bounded diagnosis reproduced that failure and passed with explicit `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`. Every complete E004 comparison uses that setting, without changing the historical cuBLAS source or tolerance. The failed partial screen and two operational prerequisites plus a diagnostic pointer-order error are preserved honestly.

The complete map covers M={1,2,4,8,16,32,64,128}, all three requested 4096/11008 K/N pairs, 8 comparators and both cache policies, with 10 warmups/20 samples. No requested shape needs padding. Fresh common graph-event intervals remove host launch gaps and exclude store flush/setup. Do not combine these medians with historical timings. Marlin beats all retained fused variants in this screen. Repeated square no-flush M=64 instead favors reconstructed cuBLAS: Marlin 58.304 us twice versus 54.768/54.752 us. Under store flushing Marlin 61.440 us beats 90.112 us cuBLAS at that point. Eager no-flush M=1 Marlin 18.560 us versus 13.536 us graph demonstrates the host-gap sensitivity. The full compact map, raw samples, p90 and setup/capture costs are preserved.

Three square Marlin profiles cover M=8/16/64 after 10 warmups, cache-control none, 13 kernel-replay passes. All launch 36 blocks*256 threads and request 101376 B dynamic shared (102400 B allocated). M=8 uses 122 registers/thread, shared/register limits 1/2 blocks, active warps 17.257%, eligible warps 0.429, tensor-active 44.613%. Source layout needs 41984 B for its eight-row tile, versus the host estimator's 50176 B. M=16 actually needs 50176 B. M=64 uses 255 registers and is independently register-limited to one block. Replay DRAM reads 2.4–2.5 MB are not proof of fully resident warm operands. Low occupancy does not prove the bottleneck, and tensor-active percentage is not peak FLOP/s.

**Single recommendation, unimplemented:** E005 bounded one-versus-two-CTA ablation for M<=8. Correct the eight-row host shared-memory accounting and compare 72 CTAs/49664 B to the untouched 36 CTA/101376 B control, preserving the Tensor Core body, format and FP32 reduction. Prove lock/C_tmp bounds before launching. Require >=5% no-flush gain at M=1 and M=8 in two independent 100-sample series, <=5% store-flush regression, strict correctness/sanitizer checks, and unchanged M=16/M=64 fallback. Hold out M=2/M=4, both rectangles and seed 2. This is standard existing Marlin resource tuning, not a novelty claim or established speedup. Read the full seven-part E004 decision before proposing implementation.

Reproduce evidence assertions with `python3 scripts/e004/analyze.py results/e004_20261106`. It checks complete raw timing sets, aggregation, provenance and all three profiles, generating `analysis.json`, `performance_map.csv`, `repeat_comparison.csv`, `profile_summary.csv`. Evidence is committed, while the bridge binary/environment remain on c58 with hashes and reconstruction commands. No RTX PRO 6000, external-kernel sanitizer or end-to-end serving run. One GPU/seed, unlocked clocks, short screen beyond selected repeats. E004 is complete and E005 is not authorized for implementation yet.

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

Await the user's decision on the single E005 recommendation in E004. Do not implement it automatically. The next authorized implementation must first prove lock/C_tmp indexing and resource feasibility for 72 blocks while preserving an untouched Marlin control. Revalidate allocation 328410 (historical, c58, scheduled end 20:00:41 Eastern on2026-11-06), ownership/state/remaining time/GPU before any reuse. Leave all user-owned allocations and original/E001/E002/E003/E004 directories intact.
