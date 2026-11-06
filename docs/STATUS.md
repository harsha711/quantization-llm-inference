# Current status

**E005-R1 is complete. B/C are rejected under the predeclared performance criterion.** Starting state `b3abd3d` was verified clean. Criteria `2c0c4cd`, harness `f36a1bd`/`5379eb3`, complete raw evidence/analyzer `607a7f3`. Documentation commits follow. Original E005's stopped run and E006's correction evidence are unchanged. No push.

The same corrected E006 library (`e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7`) serves A36/101376 B, B36/49664 B and C72/49664 B. Source/binary/runtime/environment identity and all 113 E006 handoff hashes were verified before and after measurement. E006's 135 numerical/162 sanitizer rows were reused without rebuilding or repeating the investigation. Fresh runtime/launcher checks, 27 small A/B/C numerical/replay cases and retained-project GPU checks pass. Every timed/profile row also passes unchanged all-output FP64 and operand/workspace checks.

Two balanced primary and fallback series plus held-outs contain 84 rows/8400 samples, 10 warmups/100 samples each, both cache policies. C no-flush changes versus corrected A are −0.237%/0.000% at M1 and −2.653%/−2.765% at M8. Both miss the required 5% improvement. Reservation-only B is 2.844% slower at M1 no-flush in both series and effectively tied at M8. Store-flush constraints and M16/64 fallback regression constraints pass. Held-out seed 2 rectangular C gains reach 9.081%, but one rectangle's store-flush cases regress about 3.1%. Those results do not change the fixed primary criterion or authorize dispatch.

Three corrected M8 profiles show A/B about 1.00 active CTA per SM active cycle and C 1.76, with active warps 16.6%→29.0%. C also executes 9.248% more warp instructions and transfers 2.562% more L2 bytes. Higher concurrent activity did not produce the required square-case latency benefit. Replay counters/durations remain distinct from application timings. Clocks were unlocked. Only one GPU and two primary series were measured.

Read [E005-R1](EXPERIMENTS.md#e005-r1--corrected-body-reservation-and-grid-performance-ablation) and `scripts/e005_r1/README.md`. Raw evidence and regenerated comparisons/variance/profiles are in `results/e005_r1_20261106/`. Run `python3 scripts/e005_r1/analyze.py results/e005_r1_20261106`. The first retained-project prerequisite used an absent E005 path and failed 127. Its preserved retry from the verified E004 executable root passes before timing. No raw failure is overwritten, no historical VALIDATED is created and no wheel/original-body timing is a denominator.

ARC root is `/home/hpuvvad/csc548-w4a16-e005-r1-20261106`. Corrected binaries remain preserved in the E006 root. Allocation 328410/c58 remains running at final recorded check, with scheduled end 20:00:41 Eastern on 2026-11-06, and is left intact. Reverify before any future use. **Single next task: await the user's next experiment decision. Do not introduce another optimization or repeat unchanged E005-R1.**

The sections below preserve the earlier E006/E005/E004 handoffs. Their pending authorization statements describe those earlier checkpoints. The completed E005-R1 decision above is current.

E006 is complete as a **source-supported synchronization correction**, with the reported hazard absent in the tested corrected A/B/C body. Starting state `182d200` was verified clean. Criteria `cfcfab5`, source/reproducer `46d3e8d`, complete raw evidence/tooling `00c665a`. Documentation commits follow. All original E005 and earlier sources/results/binaries remain preserved. No push.

## E006 outcome

The inherited report comes from different warps reading B and overwriting the same shared region for reduction without a block-wide entry join. The existing async wait drains only the executing thread's copy groups. One barrier immediately after that drain orders the phase transition. Source dataflow and the compiled terminal branch also show that the last lookahead B value is unused in the analyzed small-M specialization. Neither incorrect original output nor an applicable documented tool limitation was established. The exact official-wheel PC/source mapping remains unknown.

Original A reproduces the report at M8/K4096/N4096 and the same specialization at K256/N2048. The matched rebuilt unpatched body still fails racecheck, and all 16 control SASS functions match E005 exactly. The first single-barrier patch passes the original case. All 135 regular numerical rows and 162 representative sanitizer rows pass. Coverage includes A/B/C, M1/5/8 tails, M16/64 fallback, square and both rectangles, mixed/zero operands, reused guarded locks/scratch and poisoned graph replays. Every output uses the unchanged FP64 tolerance and every code/scale/reconstruction bit is checked. All six external memcheck/racecheck/synccheck runs exit 0, with racecheck zero warnings. Numerical/replay checks do not establish that the original was race-free or had incorrect outputs.

Use `python3 scripts/e006/analyze.py results/e006_20261106` to verify complete evidence. Read [E006](EXPERIMENTS.md#e006--bounded-inherited-marlin-racecheck-investigation), `results/e006_20261106/ACCESS_ANALYSIS.md` and `scripts/e006/README.md` for exact commands, access ranges, uniformity, documentation/upstream history, failures and limits. Preserved failures include a non-reproducing block 0 filter and a 300-second detailed small-report timeout. A fresh concise-analysis run completed with the original hazard and valid numerical output. Neither is hidden or treated as a pass.

The analyzer also passes under ARC prun with explicit `--e005-results /home/hpuvvad/csc548-w4a16-e005-20261106/results/e005_20261106`. This is required in the isolated ARC root, where historical evidence stays in its original directory. See `analyzer-arc.log`.

Actual ARC root: `/home/hpuvvad/csc548-w4a16-e006-20261106`. Matched source trees are `external/control` and `external/patched`, archived locally as `matched-source.tar.gz`. Remote unpatched/patched `.so` SHA256 values are `51cfcf3a8c3758631e2824d0be8f0da4c6a4cf29021a769ccf864afb4d1d652a` / `e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7`. All three corrected host configurations use that same patched body. Small-target resources stay 126 registers,32 local bytes,0 static shared, with launch plans A36/101376 B, B36/49664 B, C72/49664 B. Predicted two-block capacity is not achieved residency. No E006 latency, held-out seed2 or Nsight series exists.

E005 may be reconsidered only in a separately requested corrected-body continuation. Its historical original-body failure, library and absent VALIDATED sentinel remain unchanged. No performance acceptance criterion has been evaluated. The user allocation 328410 on c58 was verified RUNNING at the final E006 check, scheduled end 20:00:41 Eastern on2026-11-06, and left intact. Revalidate before reuse. Local host-only build/CTest and unchanged E005 evidence assertions pass. Target validation is the recorded ARC matrix, not the CPU-only check.

The E005 account below describes the original-body record and its stopping rule. E006 above is the current resolution. Historical E005 is complete as a bounded **no-go at Gate 3**, with performance inconclusive. Criteria commit `bc94827`, implementation/source analysis `eb4f88c`, raw evidence/analyzer `1fb86e1`. Documentation commits follow. E004 checkpoint `cf8a419` was verified clean. All earlier implementations/results remain intact.

## E005 outcome (historical original body)

The host-only A/B/C experiment is implemented in `scripts/e005/`, with untouched pinned source, explicit patch, common local device code, build/SASS hashes and raw diagnostics in `results/e005_20261106/`. The small-M spatial layout needs 41984 B. Actual local attributes report 126 registers/thread, 32 local bytes, zero static shared. CUDA predicts A (36 blocks,101376 B) permits one block/SM and B/C (49664 B) permit two. C launches 72 blocks. These are launch plans and predicted capacity, not achieved residency or measured speedups. M16/M64 preserve original plans/entrypoints and checked output bits. The official wheel uses 122 registers, so historical wheel times were never used as A denominators.

All 180 small and 126 main correctness rows pass unchanged all-output FP64 tolerance, exact code/scale/reconstruction conversion, reused-lock reset and workspace guards. Small checks cover M1,2,4,5,8, mixed-sign/zero-A/zero-W and an actual candidate specialization plus upstream fallback. Main checks cover M1,2,4,5,8,16,64 across all three E004 K/N pairs, with five direct and five poisoned graph calls. External memcheck and synccheck pass on representative cases, including rectangular memcheck reaching C's largest scratch offset.

**Racecheck fails.** It reports 36 errors across small A/B/C checks, with B-fragment reads at template line 843 against overlay-reduction writes at 1237. A bounded square M8 diagnosis reports two errors in unchanged local A and two in the isolated official wheel, both exit 99. This is consistent with inherited reported behavior at the B-to-reduction phase transition. Neither harmlessness, a tool false positive, an output-affecting bug, nor exact wheel instruction/source equivalence is established. Spatial capacity and passing output comparisons do not override the racecheck gate. The independent source/progress argument and its temporal limit are recorded in `GATE1.md`.

No primary latency series, fallback latency regression series, held-out seed 2 run or Nsight counter profile was released. No performance acceptance decision is possible. `measure_arc.sh` is an unexecuted plan protected by the absent VALIDATED sentinel. Do not create that sentinel manually, suppress the report, loosen criteria, or repeat unchanged E005. A separate safety investigation would require user authorization, then a new matched build and all required gates before timing. Stop after E005.

Reproduce the stopped-gate assertions with `python3 scripts/e005/analyze.py results/e005_20261106`. Actual ARC project is `/home/hpuvvad/csc548-w4a16-e005-20261106`. Existing Python3.11/Torch2.14 environment and read-only E004 bridge were reused. User allocation 328410 on c58 was left RUNNING, scheduled end 20:00:41 Eastern on 2026-11-06. Revalidate before reuse. No push.

Last updated: 2026-11-06. Branch: `optimize/inblock-k`. E004 measured benchmark source: `209442c`, complete timing evidence: `640e9e5`, profiler/provenance evidence: `528c430`, predeclared gates: `670b32d`. Documentation and analyzer commits may follow. Previous complete E003 handoff: `c1c6423`, E003 implementation/source: `5a7bcaf` / `25b2291`, evidence: `a16cbc5`. Verify actual git state before continuing.

## E004 outcome

[E004](EXPERIMENTS.md#e004--maintained-marlin-integration-and-diagnostic-comparison) is complete. Direct dense INT4 W4A16 Marlin from vLLM v0.33.0, exact commit `4ba787ec9959e566d3b587a796dff07673324da5`, runs on c58's RTX 5060 Ti sm_120. Official wheel and loaded-library hashes, Apache-2.0 source/notice archive, dependency freeze, build settings and environment are preserved. The isolated Python 3.11/Torch 2.14 environment is `/tmp/e004-hpuvvad-20261106`; shared project is `/home/hpuvvad/csc548-w4a16-e004-20261106`. No serving engine, model, fallback, external kernel patch, or new custom kernel optimization. All E001–E003 implementations/results remain intact.

Lossless repacking preserves every code/scale bit and reconstructed FP16 element. Marlin uses uint4b8/group 128, FP16 inputs/scales/reconstruction/output, FP32 accumulation/global reduction, atomic reduction disabled. All 384 screening rows pass unchanged all-output FP64 checks, including all 128 rows at M=128. Both 64-row repeated series and both 12-row eager/graph diagnostics also pass. Six retained-project sanitizer runs pass. The original default cuBLAS mode failed 2/524288 outputs at square M=128. A preserved bounded diagnosis reproduced that failure and passed with explicit `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`. Every complete E004 comparison uses that setting, without changing the historical cuBLAS source or tolerance. The failed partial screen and two operational prerequisites plus a diagnostic pointer-order error are preserved honestly.

The complete map covers M={1,2,4,8,16,32,64,128}, all three requested 4096/11008 K/N pairs, 8 comparators and both cache policies, with 10 warmups/20 samples. No requested shape needs padding. Fresh common graph-event intervals remove host launch gaps and exclude store flush/setup. Do not combine these medians with historical timings. Marlin beats all retained fused variants in this screen. Repeated square no-flush M=64 instead favors reconstructed cuBLAS: Marlin 58.304 us twice versus 54.768/54.752 us. Under store flushing Marlin 61.440 us beats 90.112 us cuBLAS at that point. Eager no-flush M=1 Marlin 18.560 us versus 13.536 us graph demonstrates the host-gap sensitivity. The full compact map, raw samples, p90 and setup/capture costs are preserved.

Three square Marlin profiles cover M=8/16/64 after 10 warmups, cache-control none, 13 kernel-replay passes. All launch 36 blocks*256 threads and request 101376 B dynamic shared (102400 B allocated). M=8 uses 122 registers/thread, shared/register limits 1/2 blocks, active warps 17.257%, eligible warps 0.429, tensor-active 44.613%. Source layout needs 41984 B for its eight-row tile, versus the host estimator's 50176 B. M=16 actually needs 50176 B. M=64 uses 255 registers and is independently register-limited to one block. Replay DRAM reads 2.4–2.5 MB are not proof of fully resident warm operands. Low occupancy does not prove the bottleneck, and tensor-active percentage is not peak FLOP/s.

**Historical E004 recommendation, now implemented but stopped by E005’s racecheck gate:** E005 bounded one-versus-two-CTA ablation for M<=8. Correct the eight-row host shared-memory accounting and compare 72 CTAs/49664 B to the untouched 36 CTA/101376 B control, preserving the Tensor Core body, format and FP32 reduction. Prove lock/C_tmp bounds before launching. Require >=5% no-flush gain at M=1 and M=8 in two independent 100-sample series, <=5% store-flush regression, strict correctness/sanitizer checks, and unchanged M=16/M=64 fallback. Hold out M=2/M=4, both rectangles and seed 2. This is standard existing Marlin resource tuning, not a novelty claim or established speedup. Read the full seven-part E004 decision before proposing implementation.

Reproduce evidence assertions with `python3 scripts/e004/analyze.py results/e004_20261106`. It checks complete raw timing sets, aggregation, provenance and all three profiles, generating `analysis.json`, `performance_map.csv`, `repeat_comparison.csv`, `profile_summary.csv`. Evidence is committed, while the bridge binary/environment remain on c58 with hashes and reconstruction commands. No RTX PRO 6000, external-kernel sanitizer or end-to-end serving run. One GPU/seed, unlocked clocks, short screen beyond selected repeats. At E004 completion E005 was unimplemented. The E005 outcome above is the current handoff and now supersedes that historical next-task recommendation.

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

Await authorization for a separately recorded E005 continuation using the common E006-corrected A/B/C body. Require its validation prerequisite, two balanced primary/fallback timing series, held-out cases and targeted profiles under the original criteria. Do not resume timing automatically, enable the old failed E005 binary, splice old denominators, or implement another optimization. Preserve all controls, failed attempts and raw evidence. Revalidate the live allocation and leave user-owned allocations and all earlier directories intact.
