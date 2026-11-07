# Experiment and design decision log

Read the numerical/measurement [contract](CONTRACT.md) before interpreting these records. Preserve raw evidence, failed attempts, and inconclusive outcomes. An alternative that has not been implemented or measured is not a failed optimization. Historical entries below reconstruct only what the repository supports. Missing prior failure numbers or commit IDs remain unknown.

## E008 — Weight reuse distance with preserved A/C

Status: complete, meaningful C gains are confined to repeated single-matrix use under the fixed criterion. Authorized 2026-11-06 from clean `dd6b9a9`. Predeclared plan and checkpoints follow, then measured results and decision. No kernel or launch-policy change.

Problem and hypothesis: E005-R2 establishes repeatable reused-weight gains but also store-flush regressions. Increasing the number of independent weight matrices between reuses may remove C's benefit. The experiment tests whether extra ready work remains useful under controlled intervening weight traffic, without claiming an inference benchmark or guaranteed eviction.

Selected design and alternatives: Keep corrected E006 binary SHA256 `e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7`, A36/101376 B and C72/49664 B, M=4, G128 and the unchanged numerical contract. For each rectangle allocate four independent packed/scales pairs generated with fresh seeds 3, 4, 5, 6. Use the seed-3 activation for all four and recompute each full FP64 reference. Keep all four allocations alive, assert disjoint storage intervals and distinct packed/scale hashes, then cycle the first 1, 2, 4 pairs in that order on one stream. Each matrix has its own output and guarded locks and scratch, reused without per-call reset. Allocation, repacking, reference, poisoning and validation stay outside timing. First-writer scratch initialization, reduction and lock reset inside the corrected kernel remain within the complete-cycle event interval. No artificial flush or per-operation event is inserted in a cycle. The previously validated E004 graph-event timer is reused with one launch callback containing the complete cycle. Record cycle latency and cycle latency divided by matrix count, never treat per-matrix values as independently measured samples.

Method and predeclared decision: Two independent processes, both rectangles, 1/2/4 matrices, A/C, 10 complete-cycle warmups and 100 complete-cycle samples per row. Reverse shape, cycle-length and configuration traversal in series 2. Exactly 24 timing rows/2400 cycle samples, 5600 GEMMs. Match operand hashes across A/C and series. For each rectangle/count, meaningful C gain is `100*(1-C_cycle_median/A_cycle_median) >= 5` in BOTH series. The single-matrix control must meet this gate before claiming that rectangle reproduces the old opportunity. A count passing only one series is inconclusive. If the control passes and both multi-matrix counts pass, permit proposing restricted launch-policy validation on additional shapes/batches. If only the control passes, classify the opportunity as reuse-dependent, with no general decoding claim. If either longer distance reproducibly loses (C slower in both series), retain A for that pattern and close that workload's lead. Mixed or subthreshold positive results remain explicitly inconclusive for meaningful benefit. No automatic dispatch, partition rewrite, counter profiling or further optimization is authorized.

Working-set calculation: Each rectangle has K*N=45088768 codes. Packed weights use 22544384 B and G128 FP16 scales use 704512 B, total 23248896 B =22.171875 MiB per pair. Active 1/2/4-pair weight sets are 22.171875/44.34375/88.6875 MiB, compared with device-reported 32 MiB L2. This is allocated logical weight data, not measured DRAM traffic or proof of eviction. All four weight allocations remain alive for every count, so only the accessed prefix changes. Output/workspace/activation storage and event nodes are reported separately.

Validation plan: Verify current ARC documentation, live owner/RUNNING c10 allocation, one prun rank, topology/affinity/tools, pinned dependency freeze and preserved source/binary hashes. Reuse E005-R2 synchronization evidence only after identity checks. Test the new complete-cycle harness first on M=4/K=256/N=2048, all counts/A/C with three direct and three poisoned graph cycles. Inspect this prerequisite before larger work. Then validate both rectangles/all counts with all-output FP64, exact packing/scales/reconstruction, distinct allocations, guard/lock checks and matched output hashes. Run representative four-matrix cycles through memcheck/racecheck/synccheck on both rectangles/A/C. Run retained build/CTest/GPU checks before timing. Stop on any correctness/launcher error and preserve diagnostics. Analyzer must reject missing rows, aliased weight ranges and mismatched operands. Preserve prior records without rewriting their derived analysis.

Prerequisite checkpoint: Criteria `ec79eca`, harness `2ec17f6`. c10 allocation 328681 remains owned by hpuvvad and RUNNING, scheduled end 2026-11-07 02:50:31 Eastern. Current official ARC documentation still specifies salloc/sbatch allocation and prun execution. The same GPU UUID, driver 610.57.04, tool versions, preserved binaries and exact dependency freeze match E005-R2. All 113 E006 and 148 E005-R2 handoff hashes pass. One-rank prun and all six small complete-cycle A/C rows pass three direct and three poisoned graph cycles, exact conversion, all-output FP64 and disjoint storage checks. Full validation is pending at this checkpoint. Local host-only build/CTest, Python parsing, shell syntax and fixed-gate/alias self-check pass.

Full validation checkpoint: All 12 rectangular numerical rows and 12 four-matrix sanitizer rows pass. Each row checks three direct and three poisoned graph cycles, all active outputs against FP64, exact codes/scales/reconstruction and guarded reused workspace. Memcheck/synccheck report 0 errors. Racecheck reports 0 errors/0 warnings. Retained sm120 project build/CTest/GPU checks pass, including six clean small sanitizer checks. The ARC and local validation-stage analyzers pass before timing. No corrected-library rebuild or numerical tolerance change occurs.

Interpretation limits fixed before timing: Each weight pair has independent output and workspace, so their active reuse distance changes too. Four guarded scratch allocations occupy 2359808 B including guards, and output size depends on N. This storage is much smaller than the four-pair 88.6875 MiB weight set, but no measured traffic attribution isolates weights alone. All four allocations are alive in every count, and raw ranges distinguish allocated bytes from accessed bytes. Warmup/capture and per-sample event boundaries are retained from E004. The experiment is a steady cyclic-access test with one fixed activation, not a model layer sequence, an explicit cache eviction test or proof of general decoding speedup.

Revisit only if: A separately requested experiment changes workload scope or supplies new evidence about cache behavior. Do not repeat unchanged runs to seek better percentages. The user-created allocation stays running.

### E008 measurements, interpretation and decision

Status: complete. Criteria `ec79eca`, cycle harness `2ec17f6`, prerequisite `fa8e485`, fixed timing/evidence tooling `c37e115`, complete fresh validation `379b5d3`, complete evidence `fff3d48`. Both series finish on c10 with exactly 24 rows/2400 complete-cycle samples/5600 timed GEMMs. The corrected E006 library is unchanged. Source, binary and exact dependency identities match before and after the run. All 113 E006 and 148 E005-R2 handoff hashes still match. Only fresh E008 A is a denominator. No prior timings are spliced into these comparisons.

The table reports cycle median latency, lower is better. Amortized latency is cycle median divided by the number of matrices, not an independently timed kernel. C gain is 100*(1-C/A), so a negative value denotes a regression. No sample is trimmed.

| K / N | Matrices | Series | A cycle, us | C cycle, us | A / C amortized us per matrix | C gain, % |
|---|---:|---:|---:|---:|---:|---:|
| 4096 / 11008 | 1 | 1 | 27.872 | 24.128 | 27.872 / 24.128 | +13.433 |
| 4096 / 11008 | 2 | 1 | 124.336 | 120.384 | 62.168 / 60.192 | +3.178 |
| 4096 / 11008 | 4 | 1 | 245.024 | 241.936 | 61.256 / 60.484 | +1.260 |
| 11008 / 4096 | 1 | 1 | 25.824 | 24.128 | 25.824 / 24.128 | +6.568 |
| 11008 / 4096 | 2 | 1 | 120.176 | 119.904 | 60.088 / 59.952 | +0.226 |
| 11008 / 4096 | 4 | 1 | 236.832 | 238.624 | 59.208 / 59.656 | -0.757 |
| 4096 / 11008 | 1 | 2 | 27.856 | 24.128 | 27.856 / 24.128 | +13.383 |
| 4096 / 11008 | 2 | 2 | 124.112 | 121.904 | 62.056 / 60.952 | +1.779 |
| 4096 / 11008 | 4 | 2 | 246.880 | 240.976 | 61.720 / 60.244 | +2.391 |
| 11008 / 4096 | 1 | 2 | 25.872 | 24.128 | 25.872 / 24.128 | +6.741 |
| 11008 / 4096 | 2 | 2 | 119.872 | 119.952 | 59.936 / 59.976 | -0.067 |
| 11008 / 4096 | 4 | 2 | 237.600 | 238.624 | 59.400 / 59.656 | -0.431 |

Both single-matrix controls pass the predeclared 5% gate in each reversed series. The new harness reproduces the corrected-C opportunity on fresh seed-3 operands, at 13.433%/13.383% for K=4096/N=11008 and 6.568%/6.741% for the transpose. Every two/four-matrix case fails the meaningful-gain gate. The stronger rectangle retains small positive gains, 1.779–3.178% at two matrices and 1.260–2.391% at four. Those are measured subthreshold gains, not a demonstration that C is slower. The transpose is essentially tied at two matrices,+0.226%/−0.067%, and C is reproducibly slower at four by 0.757%/0.431%.

The absolute latency change is substantial for both configurations. For K=4096/N=11008, one-matrix A is 27.856–27.872 us per GEMM, while two/four-matrix amortized A is 61.256–62.168 us. C rises from 24.128 us to 60.192–60.952 us. The transpose has the same broad increase, around 24–26 us to 59–60 us. This is consistent with a strong reuse-distance/cache effect. The experiment does not measure eviction, per-layer inference behavior or DRAM traffic, and it does not isolate the share due to weights versus the smaller output/workspace access changes. The earlier replay profiles are not substituted for missing cycle-specific counters. The changed c58/c10 gain magnitude still cannot be assigned to one environmental difference.

Timing anomalies and limits: All raw samples, min/max/p90 and sample standard deviations are preserved in `sample_statistics.csv`. Series 2 transpose single-matrix C has a 3412.544 us sample at index 78 and A has a 7667.264 us sample at index 12. Their medians are 24.128/25.872 us and p90 values 26.176/27.616 us, but sample standard deviations rise to 338.795/764.108 us. Several two-matrix rows also have isolated maxima of 144.832–221.824 us against medians around 120–124 us. No cause was isolated, no outlier was removed and no alternative aggregation replaced the predeclared median. These data do not establish stable tail latency. Clocks are unlocked. Before series 1 the GPU is P8/180 MHz, afterward P0/2617 MHz, and after series 2 P0/2782 MHz, with active memory clock 14001 MHz. Boundary snapshots do not establish per-sample frequency. Both series use the same executable, GPU, runtime, rank/thread layout and inputs.

Validation: Six small complete-cycle smoke rows, 12 full rectangular numerical rows and 12 representative four-matrix sanitizer rows pass. The sanitizer rows cover 48 matrix outputs, each through three direct and three poisoned graph cycles. Memcheck/synccheck report 0 errors, racecheck 0 errors/0 warnings. Retained CUDA build/CTest/GPU checks pass. All 24 timing rows pass pre/post all-output FP64, guards and reused-lock checks. Maximum timing-row absolute error is 0.0018590828112792224 and maximum relative L2 is 0.00020788470978230626. Acceptance uses the unchanged elementwise magnitude/RMS threshold, not the absolute maximum alone. Packed/scales storage is genuinely independent, all four matrix hashes differ, activations are fixed, and A/C/series operand hashes agree. Full-output hashes for a given configuration/matrix also agree across cycle lengths and series. A/C output-bit equality is not required because their reduction partition differs.

Analysis checks: `python3 scripts/e008/analyze.py results/e008_c10_20261106` verifies complete orders, row/sample counts, fixed numerics/resources, storage disjointness, operand/output identities, raw statistics, prerequisite summaries and tooling hashes. `python3 scripts/e008/check_analysis.py results/e008_c10_20261106` accepts full evidence and rejects missing timing rows, aliased weights and mismatched operands in temporary copies. Both pass locally and under ARC prun. The fixed-gate/alias self-check also passes. Raw logs and exit codes are preserved. No failed numerical/launcher/sanitizer attempt occurred in this E008 run. No timing rerun or extra shape search was performed.

Final metadata attempt: Collecting retained host flags initially referenced nonexistent `w4a16_cpu.dir/flags.make` and produced preserved cp/tar diagnostics in `metadata-attempt1.log`. The target is `w4a16_host`. Correcting that path retrieves existing build flags without a rebuild or measurement rerun. This is a metadata-collection failure, not a failed numerical or sanitizer case.

Decision: The meaningful C advantage is **reuse-dependent in this setup**. Keep corrected A as the default, and retain A for the tested cycling workloads. Close the four-matrix transpose launch-policy lead because C loses in both orders. The stronger rectangle's positive 1–3% multi-matrix gains do not meet the predeclared continuation criterion. No multi-matrix result supports the next restricted-policy investigation under that criterion, and no general decoding improvement is claimed. Preserve C as the repeated-single-matrix experimental configuration. Do not introduce dispatch, a partition rewrite or another optimization. This is a completed bounded memory-access experiment, not a failed synchronization correction or a reason to repeat E006.

Revisit conditions: A separately requested workload with demonstrated short weight reuse could justify evaluating an explicitly restricted policy, including its tradeoff with larger reuse distances. New cycle-specific cache/traffic or phase evidence could justify a different concrete hypothesis. Do not rerun the same 24 rows to seek a better percentage. Review this result before selecting another task.

Reproduction and provenance: Shared root `/home/hpuvvad/csc548-w4a16-e008-20261106`, runtime `/tmp/e005-r2-hpuvvad-20261106`, raw root `results/e008_c10_20261106/`. Commands are the shell traces in each stage log. On verified allocation 328681, export absolute PROJECT_ROOT and node/allocation variables, then use `prepare_arc.sh`, inspect its prerequisite analyzer, `validate_arc.sh`, inspect its validation analyzer, then `measure_arc.sh`. All compute uses one prun rank with 16 OpenMP threads, portable retained host build flags and actual c10 topology/affinity. Corrected library/bridge/wheel are reused by exact hash, and only retained project checks are rebuilt. Tool/compiler/runtime/GPU/CPU metadata, dependency freeze, source/build flags, original manifests and final input checks are retained. The user-created allocation remains running, scheduled end 2026-11-07 02:50:31 Eastern. Final handoff verification is recorded separately. No push, additional profile, clock lock, automatic dispatch or full-inference benchmark.

## E005-R2 — Independent rectangular A/C confirmation

Status: complete, both rectangles pass the predeclared investigation gate. Authorized 2026-11-06 from clean `925be8a`. Criteria `192e33a`, harness `5412dbb`, verified prerequisite `f19d83a`, timing/gated profile plan `00b5146`, complete profile evidence/checkers `f86d463`. This is a bounded confirmation of the E005-R1 rectangular lead, not E008 implementation or a changed default.

Problem and hypothesis: One held-out c58 series suggests C helps M=4/K=4096/N=11008 more than its transpose. Existing corrected A/C launch configurations may deliver a reproducible geometry-dependent benefit. Execution order or cache/environment effects could instead explain the observation.

Selected design and alternatives: Reuse the exact E006 corrected library, source and A/C host policies without recompilation. A uses 36 blocks/101376 B and C uses 72 blocks/49664 B on a verified 36-SM RTX 5060 Ti. Keep G128 and all numerical rules. Use M=4, seed 2, both K/N orientations (4096,11008) and (11008,4096), both `no_flush` and `store_flush`, 10 warmups and 100 samples. Run two independent processes, eight timing rows each, reversing A/C and rectangular traversal order in series2. Reverse cache-policy traversal too. There are exactly 16 timing rows/1600 samples. Preserve raw samples, commands, operand hashes and observed variability. Broader tuning, a new partitioner and automatic dispatch are excluded.

Predeclared continuation gate: For a rectangle, require `100*(1-C_median/A_median) >= 5` under no-flush in **each** independent series. If neither rectangle passes, or the benefit follows execution order, stop this lead without profiling or policy changes. Report store-flush changes separately without folding them into this no-flush investigation gate. If either rectangle passes, profile A and C on the rectangle with the larger minimum no-flush gain across the two series. A tie selects K=4096/N=11008. Passing warrants investigation only and does not authorize dispatch or partition changes.

Validation and environment: The user reports that c58 expired and the new node is c10. Allocation 328681 is owned by hpuvvad, RUNNING, one node/16 CPUs, scheduled end 2026-11-07 02:50:31 cluster time. Verify live state again before launches. c10 has RTX 5060 Ti 16GB but driver 610.57.04 versus c58's 615.71.09. The c58 node-local Python environment is absent. Restore the exact pinned dependency versions in a separate c10 environment and verify operator, bridge, corrected-library and source hashes. Changed node/driver/runtime path require fresh relevant validation, not automatic reuse of E006's entire gate. First test one-rank prun/runtime and small A/C M=4 K=256/N=2048 correctness. Then run fresh A/C M=4 checks and memcheck/racecheck/synccheck on both actual rectangles, including repeated direct calls, poisoned graph replay, all-output FP64, exact codes/scales/reconstruction, workspace guards and lock reset. Run retained project build/CTest/GPU gate separately. Any correctness or launcher failure stops timing and preserves diagnostics. No need to repeat the original E006 source investigation.

Conditional profiles: Query supported target metrics first. Preserve the original replay method with no profiler cache/clock control, 10 warmups and one exact corrected local kernel. Collect matched A/C instruction counts, registers/resources, CTA/eligible/issue/Tensor activity, L2/DRAM traffic and throughput, barrier/scoreboard/wait metrics. Record replay passes and avoid treating profiler duration as application latency. Interpret against competing parallel-work, memory-service and coordination-cost explanations. No optimization is implemented after profiling in this task.

Operational prerequisites: The first staging attempt lacked the parent `results/` directory and stopped before launch. The retry verified all 113 E006 handoff hashes but then stopped because the historical four-binary manifest includes the expired c58 `/tmp` wheel path. Preserve `prepare-attempt1.log`, `prepare-attempt2.log` and the failed manifest output. Verify the three preserved shared binaries separately, restore the pinned wheel on c10, then verify its exact recorded SHA256 in the identity check. These are setup failures, not numerical failures or permission to reuse c58's environment gate. Local host-only build/CTest, shell syntax, Python parsing and continuation-gate self-check pass. None supplies target CUDA validation.

Verified prerequisite checkpoint: Criteria commit `192e33a`, harness `5412dbb`, prerequisite-record/analyzer correction `9a24996`. Restored Python 3.11/Torch 2.14.0+cu130 dependency freeze matches exactly. Corrected E006 library, bridge and official repacking wheel hashes match their historical values. One prun rank runs on c10, OpenMP 16 with workers bound across CPUs 0–15 on one NUMA node, AMD EPYC 9115. GPU UUID is `GPU-38a74fc5-9f81-89d1-4ad5-3f5796d39bf2`, driver 610.57.04. CUDA compiler 13.4.92, sanitizer 2026.3.0.0 and NCU 2026.3.1.0 match the saved tool versions. All six small A/C M=4 K=256/N=2048 mixed/zero-A/zero-W rows pass with three direct and three poisoned graph replays, maximum absolute error 0.0009750075405463576. This is a cheap prerequisite only. At this checkpoint full rectangular validation was pending. Its completed result follows.

### E005-R2 fresh validation and confirmation

All 12 fresh regular rectangular A/C rows pass, including mixed/zero-A/zero-W for both orientations. All 12 fresh sanitizer rows pass across memcheck/racecheck/synccheck. Every row has three direct and three poisoned graph calls with guarded reused workspace. Racecheck reports zero errors and zero warnings. The retained project builds for sm120, CTest passes and its GPU gate passes with six clean sanitizer checks. No corrected-kernel rebuild occurs. Validation precedes all confirmation timing.

Both independent eight-row series complete with exactly 1600 event samples. The analyzer passes on ARC and locally, requiring full row sets, reversed shape/configuration/cache-policy order, fixed inputs, exact resource plans, unchanged numerical checks and all source hashes. For K=4096/N=11008, A/C no-flush medians are 27.840/24.128 us in series1 and 27.824/24.128 us in series2, gains 13.333%/13.283%. For K=11008/N=4096 they are 26.176/24.128 us and 25.760/24.128 us, gains 7.824%/6.335%. Both orientations pass the predeclared 5% gate. The chosen profile rectangle is K=4096/N=11008 by its larger minimum gain.

Store-flush changes are separate: C regresses 5.477%/6.061% at K=4096/N=11008 and 0.046%/1.419% for the transpose. The stronger reused-input result does not erase these regressions or justify a universal launch replacement. Maximum absolute timing error is 0.0016377683277823962 and relative L2 is 0.0002078045406591234. The policy remains corrected A by default.

The first timing-evidence transfer overlapped the supported-metric query and tar refused a changing log. Preserve the diagnostic in this record: `tar: results/e005_r2_c10_20261106/metric-query.log: file changed as we read it`. A premature local analyzer call then found the incomplete synchronization missing `identity-final.exit`. Retry after the query completes transfers all evidence and the analyzer passes. Neither operation reruns measurements or changes numerical evidence. The approximately 135 MiB metric query is losslessly gzip-compressed locally with its uncompressed SHA256 recorded. All requested profile metrics are supported, including L2 throughput and not-selected/no-instruction waits, which were absent from the earlier square profiles. Profile only one A/C pair under the predeclared method.

### E005-R2 matched profiles and design decision

The selected K=4096/N=11008/M=4 A/C pair passes numerical/workspace checks and exact local symbol verification. Both profiles use 13 kernel-replay passes, 10 warmups, `--cache-control none` and `--clock-control none`. The library hash remains the corrected E006 hash. The following counters describe that replay, with normalization stated explicitly. They do not measure the unprofiled graph interval or the store-flush policy.

| Metric | Corrected A | Corrected C |
|---|---:|---:|
| Active CTAs per SM active cycle | 1.000 | 1.754 |
| Active warps, % peak during active cycles | 16.607 | 29.214 |
| Eligible warps per scheduler active cycle | 0.455 | 0.695 |
| Issue-active cycles, % scheduler active | 37.597 | 45.589 |
| Executed warp instructions | 3605631 | 3788084 |
| Predicated-on integer thread instructions | 38469786 | 41560208 |
| Tensor instructions | 176128 | 176128 |
| L2 traffic, B | 26586112 | 26886944 |
| L2 throughput, % of sustained peak over elapsed cycles | 24.394 | 26.833 |
| DRAM reads, B | 15618048 | 15089664 |
| DRAM throughput, % of sustained peak over elapsed cycles | 80.490 | 84.392 |
| Tensor-pipe active cycles, % elapsed | 35.126 | 38.453 |
| Barrier-stall metric, % of peak over elapsed cycles | 1.162 | 2.405 |
| Long/short scoreboard metrics, % of peak over elapsed cycles | 0.263 / 0.345 | 0.421 / 0.480 |
| Registers per thread, requested / allocated | 126 / 128 | 126 / 128 |
| Shared reservation, requested / allocated B | 101376 / 102400 | 49664 / 50688 |

C increases ready work and issue activity while adding 5.060% executed warp instructions, 8.033% predicated-on integer thread instructions and 1.132% L2 traffic. Tensor instruction work stays equal. This supports a benefit from additional concurrency despite extra control/partition work. Tensor instruction count is unchanged, and no dequantization-specific runtime reduction is established. Replay DRAM throughput is high, so memory service is relevant in this profiled case. L2 and Tensor activity do not establish their saturation. These are aggregate throughput/stall metrics, not disjoint wall-time fractions. DRAM throughput and byte/duration changes are related quantities, not independent causal tests. Use the [Nsight profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) for normalization and replay semantics.

Profiler durations 44.032/40.576 us differ substantially from application no-flush medians about 27.832/24.128 us. Do not use the replay's 7.849% duration change as the application gain, nor claim its memory state matches graph timing. No store-flush profile was requested or collected, so its regression mechanism remains unresolved. Lower replay DRAM reads do not prove less logical weight data or full residency. The new node/driver prevents attributing changed gain magnitude relative to c58 to one cause. Only fresh c10 A is the timing denominator.

The source-derived finite model, saved separately with its source hash, predicts 72 whole output tiles before A's 14 remaining tiles. C puts all 86 tiles into its stripe phase. Output tiles requiring global split reduction rise 14→68. This identifies coordination work that changes with the existing launch, but no phase time or source-PC attribution was collected. Higher barrier/wait/integer work is consistent with that overhead. It does not identify a concrete improved partition algorithm or establish how much latency eliminating it would save.

Decision: the existing-C opportunity is real within the tested M=4/seed 2 rectangles on c10 under reused inputs. Both no-flush gains pass 5% in both execution orders. The strongest rectangle has a repeatable store-flush regression above 5%, so keep corrected A as the default. Preserve E005-R1 and E007 as negative results under their own criteria. This confirmation selects no new kernel, launch policy or partitioner. Additional profiles, tuning, dispatch and E008 implementation require a separate request. Revisit this lead only through a separately predeclared validation of a restricted launch choice on independent inputs/workloads and an explicit cache-policy tradeoff, or through source-PC/phase evidence that identifies a concrete coordination change. Do not rerun the same confirmation merely to improve percentages.

Validation of analysis: `python3 scripts/e005_r2/analyze.py results/e005_r2_c10_20261106` verifies 16 timing rows/1600 samples and all fresh prerequisites. `python3 scripts/e005_r2/analyze_profiles.py results/e005_r2_c10_20261106` verifies the two profiles, normalized units, exact kernel and final identity. `python3 scripts/e005_r2/check_analysis.py results/e005_r2_c10_20261106` accepts complete evidence and rejects a missing timing row or unrelated kernel in temporary copies. `python3 scripts/e005_r2/analyze.py --self-test` checks the fixed 5% gate and order-following rejection. All three final evidence checks also pass under ARC prun, recorded in `analysis.log`, `profile-analysis.log` and `analysis-negative.log`. Timing/analyzer, actual prun, commands, dependency freeze, CPU/NUMA/affinity, tool/driver versions, build flags, source/binary hashes, raw samples, min/max/p90/stdev, compressed metric query and complete NCU reports remain preserved. All eight series1 operand hashes also match the historical seed 2 held-outs, without using any historical timing denominator.

ARC root is `/home/hpuvvad/csc548-w4a16-e005-r2-20261106`, runtime `/tmp/e005-r2-hpuvvad-20261106`. Post-profile checks confirm all 113 E006 handoff files and all five benchmark inputs unchanged. Historical E005/E006 VALIDATED files remain absent. Allocation 328681/c10 is still RUNNING at the final recorded check, scheduled end 2026-11-07 02:50:31 cluster time, and is left intact. Local host-only build/CTest also pass. Limits are two timing series, one seed/M, two rectangles, one new GPU/driver and one no-flush profile pair with unlocked clocks. No serving-level or automatic-policy claim.

## Pre-E008 — Saved-profile bottleneck assessment

Status: completed assessment, no new experiment implemented or measured. Requested after `ab54819`.

**Question and alternatives:** Determine whether the corrected kernel is constrained by available parallel work, memory service, or partition/reduction overhead before choosing E008. Preserve E006 corrected A, E005-R1 and E007. Do not infer a bottleneck from occupancy alone or treat a small instruction reduction as a runtime fraction.

**Evidence and validation:** Read the retained E005-R1/E007 profiles, application comparisons and finite stripe model. Run `python3 scripts/e005_r1/analyze.py <temporary-copy-of-results/e005_r1_20261106>` and `python3 scripts/e007/analyze.py <temporary-copy-of-results/e007_20261106>`. Both pass, confirming 8400/5600 samples and the original rejection decisions. Temporary copies prevent regenerated derived files from changing historical evidence. Recompute instruction/L2 deltas and rectangular timing changes from CSV values. No numerical, sanitizer or CUDA rerun applies to this documentation-only assessment.

**Findings and decision:** [The short assessment](BOTTLENECK_ASSESSMENT.md) separates observations from competing explanations. Corrected square profiles support limited ready work with added overhead when concurrency increases, without establishing a unique dominant pipeline or memory limit. The strongest lead is the single-series rectangular gain, not E007's terminal guard. Recommend a small independent A/C timing confirmation with the preserved corrected library before collecting rectangular profiles. No new algorithm, default or dispatch is selected.

**Next evidence required:** Two reversed M4/seed2 timing series for both rectangular orientations and both cache policies, with unchanged correctness/methodology, followed only conditionally by one matched profile pair. These measurements are proposed and unperformed. They do not reverse E005-R1's failed primary criterion. Await a user decision before execution or E008 implementation. E007's recorded revisit condition remains unchanged.

## H001 — Correct FP16 reconstruction before accumulation

Status: accepted historical correction, already present in baseline commit `8a68227`.

**Problem:** Accumulating with FP32 `(q-8)*s` differs from reconstructing FP16 W_hat and then accumulating. Comparing those operations can report numerical disagreement caused by different inputs rather than a reduction defect.

**Hypothesis:** All reconstructed-weight paths agree on their inputs when each product is rounded to FP16 before FP32 accumulation.

**Change and design decision:** Use `FP32(FP16(FP32(q-8)*FP32(s)))` in fused kernels. Preserve FP16 materialization on CPU/GPU and in the pre-dequantized cuBLAS baseline. Direct FP32 reconstruction was the numerical alternative, not an interchangeable implementation of this contract.

**Baseline and candidate:** Correction predates the available initial commit. Exact earlier candidate/failure commits and failure measurements are not recoverable from this repository. Evidence is the reconstruction in `src/quant.cpp`, `src/cuda/dequant.cu`, the v0 naive-kernel comment/implementation, README's original contract, and `results/baseline_c7.csv` already contained in `8a68227`. Do not assign `results/first_run_c7.csv` to the incorrect reconstruction without additional evidence.

**Validation and findings:** Historical baseline rows have `w_hat_mismatches=0` and `pass=1`. Main-shape naive relative L2 errors versus reconstructed FP64 are approximately 2.07e-4. Errors versus original weights are approximately 0.12 and include quantization error. The available data establishes agreement for the corrected implementation, not the magnitude of a missing earlier failure.

**Decision:** Keep FP16 reconstruction in every W_hat path. Never loosen output tolerance to hide a reconstruction mismatch.

**Revisit only if:** A separately requested numerical contract changes weight reconstruction precision and all references/baselines are rebuilt for that operation.

## H002 — Historical naive and comparison baselines

Status: measured historical baseline, preserved for context.

**Problem and hypothesis:** Establish correctness, latency, and the effect of increasing activation rows before choosing an optimization. Low launch parallelism is a candidate explanation for small-M naive latency, not the sole established cause.

**Change:** None in this record. Six comparison roles are now defined in the contract, while historical data contains the original five without inblock.

**Version and environment:** Baseline commit `8a68227` contains these records. Measurements were saved 2026-11-04 on c7, RTX 5060 Ti, CC 12.0, 36 SMs, L2=32 MiB, driver 610.57.04, CUDA toolkit 13.4.92, Nsight Compute 2026.3.1. Exact historical executable hash/build flags were not preserved in `baseline_meta.txt`. Do not manufacture them.

**Commands and raw results:** `scripts/run_baseline.sh` records the original procedure, and `results/baseline_c7.log` records runs. Do not rerun it in place, because it overwrites historical baseline files. Main sweep arguments were:

```bash
./build/w4a16_bench --M "$M" --N 4096 --K 4096 --group 128 --seed 1 \
  --warmup 10 --iters 100 --csv results/baseline_c7.csv
# M=1,8,32,128, application store flush enabled by default.
./build/w4a16_bench --M 1 --N 4096 --K 4096 --group 128 --seed 1 \
  --warmup 10 --iters 100 --kernels naive --no-flush-l2 \
  --csv results/baseline_c7.csv
```

**Validation:** Historical main rows pass the original FP64 threshold. W_hat mismatches are zero. Up to 64 rows are checked, so the M=128 sweep is sampled. Small tests and group=4096 runs are also in the raw CSV. Historical sanitizer evidence is not present.

**Measurement and findings:** The following are CUDA-event medians in microseconds from `results/baseline_c7.csv`, K=N=4096, group=128, seed=1, 10 warmups, 100 timed iterations, application store flush. Lower is better. Raw min/max/p90 remain in the CSV.

| M | naive | cublas original W | cublas_dq W_hat | dequant | unfused |
|---|---:|---:|---:|---:|---:|
| 1 | 204.448 | 94.208 | 94.208 | 120.480 | 178.528 |
| 8 | 202.752 | 86.016 | 86.016 | 121.184 | 186.016 |
| 32 | 696.640 | 86.368 | 86.368 | 120.480 | 172.016 |
| 128 | 2540.896 | 116.704 | 114.688 | 121.184 | 225.312 |

Historical M=1 no-flush naive median is 108.096 us, range 107.072–109.856 us, p90=108.128 us. The store-flush M=1 range is 202.272–206.528 us, p90=205.120 us. The difference demonstrates sensitivity to this application cache policy, not proof of complete eviction.

Nsight evidence in `results/ncu/summary.txt` reports naive M=1 grid=(16,1,1), block=256, 32 registers/thread, achieved active-warps metric 16.58%, SM throughput 9.16%, and DRAM reads 8.66 MB under profiler cache control. At M=128 the grid has 2048 blocks, active warps 95.79%, SM throughput 85.60%, and approximately 1.124 GB L2 traffic despite only 9.75 MB DRAM reads. These profile samples differ from event timing conditions. Low M=1 block count motivates E001. Large-M repeated cache traffic motivates a possible future reuse study but does not establish the next bottleneck.

**Decision:** Preserve these records. Rerun naive and candidate together on the same node/build for E001. Never divide new c58 candidate latency by old c7 latency to claim speedup.

**Revisit only if:** Historical provenance is needed or a controlled rerun changes the measurement method. Record new evidence in a new directory.

## H003 — Cache-flush and profiler warm-run limitations

Status: inconclusive as proof of physical cold/warm cache state. Existing timing policies remain usable when described precisely.

**Problem and hypothesis:** Packed weights/scales occupy 8.25 MiB and fit the observed 32 MiB L2. Repeated launches can use cache. Writing a larger buffer before timing was intended to reduce this reuse.

**Change and alternatives:** The application store kernel writes 64 MiB before the start event. `--no-flush-l2` omits that store. Nsight `--cache-control all` invalidates caches between replay passes and is a separate profiling method. These methods do not define identical measurements.

**Version, commands, and raw results:** Baseline `8a68227`, `scripts/ncu_baseline.sh`, `results/ncu/summary.txt`, `results/ncu/naive_M1_cold*`, and `results/ncu/naive_M1_hot*`. Historical warm-profiler launch used `--cache-control none --launch-count 1` and selected the first warmup.

**Validation and findings:** The application's store flush does not verify eviction of every input line and store traffic may affect subsequent DRAM work. The historical profiler hot run captured an initial launch, so its DRAM/time still looked cold. L2 hits on later replay passes do not repair that warm-run interpretation. `dram__bytes_read.sum` was unavailable on this target. Historical reports used supported `dram__bytes_op_read`/`dram__bytes_op_write` metrics instead.

**Decision:** Do not classify the historical profiler hot sample as a demonstrated steady-state warm comparison. Preserve it as inconclusive. Report application flush/no-flush event latency separately from profiler cache-controlled counters. Label `eff_gbps` as a traffic-model metric.

**Revisit only if:** A new cache experiment validates the actual cache state or profiles a correctly selected steady-state launch with documented replay/cache settings.

## E001 — In-block K splitting

Status: accepted as a separately selectable M=1 experiment. Rejected as a universal naive replacement. Final correctness, sanitizer, latency, and profiler evaluation are complete at code checkpoint `2b7d4fc`. Pilot speedup interpretation remains inconclusive because the tail guard slowed its naive denominator.

**Problem:** For N=4096, M=1 naive launches 16 blocks, one thread per output, each traversing the full K accumulation chain. Historical profiling shows low active-warps/SM throughput in that case. This motivates a parallelism experiment, not a proven attribution of every stall.

**Hypothesis:** Eight threads collaborating on each output, with distinct K slices, will reduce latency enough to offset shared reduction overhead by shortening per-thread accumulation and increasing total blocks.

**Success criterion recorded before timing:** All correctness checks must pass the original tolerance with zero W_hat bit mismatches and no sanitizer errors in the selected cases. The primary performance criterion is M=1 median speedup greater than 1.0 over same-executable naive under both application policies. Report all M={1,8,32,128} results, p90/ranges, and regressions. A borderline or inconsistent advantage requires repeat series and an inconclusive decision. No universal acceptance across shapes is implied.

**Change:** Block=(32,8), grid=(ceil(N/32), M). Each warp owns 32 columns and one of eight contiguous packed-word K intervals. For W=ceil(K/8), slice y consumes `[floor(W*y/8),floor(W*(y+1)/8))`. These intervals cover all words exactly once, even W<8. Each thread preserves FP16 reconstruction and FP32 FMA, writes its partial to `float partial[8][32]`, and reaches the barrier. Invalid N lanes/empty slices contribute zero. Warp zero adds eight partials in slice order and writes FP16 once. Shared storage is 1024 bytes. There are no atomics or global partials. The FP32 reduction order changes.

The shared layout originally truncated K tails and required K divisible by group. Correctness cases required extending host packing, scale counts, quantization, naive, and dequantization bounds to ceiling counts and valid tail elements. These changes preserve aligned data formats. Groups remain multiples of eight.

**Alternatives considered:**

| Design | Decision and reason |
|---|---|
| Keep one naive thread per output | Retain as baseline. Small-M block count and full-K chain remain. |
| Eight contiguous K slices and shared reduction | Implement. Adjacent lanes retain adjacent-column loads and interval coverage is straightforward to audit. Performance is not assumed. |
| Interleaved K words among warps | Untested alternative. Changes access order without evidence favoring it for the first experiment. |
| Warp shuffle reduction | Not selected. The eight partials of one output live in different warps. Shared memory directly supports cross-warp communication. |
| Cross-block split-K, atomics, or global partials | Excluded from this requested in-block experiment. Adds another reduction mechanism. |
| Tensor Cores, async copies, A/W tiling, new layout | Deferred and untested. Changes additional mechanisms, preventing attribution to E001. |
| Reject tails | Not selected because required small-K/partial-group checks need correct tail support. |
| Non-multiple-of-eight groups | Not implemented. Existing one-scale-per-word guarantee remains. |

**Baseline and candidate:** Historical baseline `8a68227`, initial candidate `654b5e2`, gated scripts `597b7bf`, repaired implementation `2b7d4fc`, branch `optimize/inblock-k`. E001's actual denominator is the retained naive in the candidate executable. Pilot build: Release, CUDA sm_120, nvcc 13.4.92. Host flags are `-O3 -DNDEBUG -std=gnu++17 -Wall -Wextra -fopenmp`. CUDA flags are `-O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_120,code=[compute_120,sm_120] -lineinfo`. Full source/executable hashes and flags are in `metadata.txt`, `source-build.sha256`, and `check.sha256` under the pilot result directory. No portable-host-ISA changes or `-march=native` are requested.

**Validation:** Local nvcc is absent. `bash scripts/build.sh`, `ctest --test-dir build --output-on-failure`, `git diff --check`, and `bash -n scripts/check_inblock.sh` passed locally. CPU tests include K={1,3,7,8,9,40,65,129,137,257}, N=33, group=128, scales/codes/reconstruction, zero weights, unused tail nibbles, and packed-word interval coverage for W=1 through 520. CUDA compilation and the complete small-case GPU gate passed on c58. Four Compute Sanitizer runs (memcheck and synccheck, each at K=3 and K=137) reported zero errors. All 48 pilot sweep rows passed, with zero bit mismatches among all 16,777,216 W_hat elements in each main-shape check. The repaired executable also passed the complete GPU gate, including four zero-error memcheck/synccheck invocations. This target evidence is distinct from the local host-only pass.

The GPU gate is `bash scripts/check_inblock.sh`. It runs small cases across all comparison paths, K tails, N={1,31,32,33,63}, zero weights, and every output row. When available, Compute Sanitizer memcheck and synccheck check naive/inblock/dequant at K={3,137}, M=3, N=33. Sanitizer unavailability must be recorded, not counted as a pass.

**Measurement environment:** On 2026-11-06, job 328410 was RUNNING on c58 in `rtx5060ti16g`, one node, 16 allocated CPUs, six-hour limit ending 20:00 America/Detroit (Eastern). The interactive allocation belongs to the user and must remain running. Revalidate rather than reuse these historical values blindly. c58 exposed RTX 5060 Ti 16 GB, driver 615.71.09, CUDA toolkit 13.4.92, Nsight Compute 2026.3.1, and an AMD EPYC 9115 16-Core Processor, one CPU NUMA domain, 16 logical CPUs, GCC 12.4.0, and Open MPI 4.1.6. Full hardware/topology/ISA/affinity/tool metadata is required.

Remote source is `/home/hpuvvad/csc548-w4a16-inblock-20261106`, preserving `/home/hpuvvad/csc548-w4a16`. Pilot remote/local result path is `results/inblock_c58_20261106/`. It contains `allocation.txt`, `build.log`, `check.log`, `check.sha256`, `metadata.txt`, `source-build.sha256`, `gpu_before.csv`, `gpu_after.csv`, `timings.log`, and `timings.csv`. Preserve the pilot. The complete repaired-baseline series uses `results/inblock_c58_20261106_final/`. Pilot controls are preserved as `control_commands.sh`, `control.log`, `control_original.csv`, `control_guard.csv`, and `control.sha256`. The repaired-baseline directory contains repeated `build.log`, `check.log`, `check.sha256`, `allocation.txt`, `naive_original.sass`, `naive_specialized.sass`, `verify_sass.py`, and `sass_comparison.txt`. `results/inblock_c58_20261106_final/prepare_commands.sh` preserves preparation commands. Its `measure_commands.sh` preserves the exact final measurement environment/procedure.

**Historical reproduction scope:** E001 used source checkpoint `2b7d4fc` and its recorded six-kernel/four-sanitizer scripts. Current scripts have expanded for E002 to seven kernels and six sanitizer runs. Reproduce E001 from its recorded source/artifacts. Exporting an old SOURCE_COMMIT on current code does not reproduce that source.

**Reproduction commands:** The supplied route is `ssh -i ~/username hpuvvad@arc-off-network`, then `ssh hpuvvad@c58`. Verify current documentation and `scontrol show job 328410` before applying the historical launcher environment below. These exports describe an existing allocation, not a resource request:

```bash
set -euo pipefail
export SLURM_JOB_ID=328410 SLURM_JOB_NODELIST=c58 SLURM_NTASKS=1
export SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node
export OMPI_MCA_hwloc_base_binding_policy=none
prun /bin/hostname
# Expected application output: one c58 line, plus launcher diagnostics.
cd /home/hpuvvad/csc548-w4a16-inblock-20261106
export RUN_DIR="$PWD/results/inblock_reproduce_$(date -u +%Y%m%dT%H%M%SZ)"
export SOURCE_COMMIT=2b7d4fc  # only for the verified measured source checkpoint
mkdir "$RUN_DIR"
prun bash scripts/build.sh > "$RUN_DIR/build.log" 2>&1
prun bash scripts/check_inblock.sh > "$RUN_DIR/check.log" 2>&1
# The successful gate writes check.sha256 automatically. Inspect its complete log.
prun bash scripts/run_inblock.sh > "$RUN_DIR/timings.log" 2>&1
prun bash scripts/ncu_inblock.sh > "$RUN_DIR/profile.log" 2>&1
# Profiling is separate from latency and queries metrics on the target.
```

Save complete stdout/stderr with pipefail enabled when tee is used. The timing script verifies executable hashes against the successful gate and refuses to overwrite timings.csv. Source/archive integrity, exact remote commands, allocation checks, and raw logs must remain alongside results. A future unattended batch submission must use an absolute exported project path and depend on its successful prerequisite job.

The timing script runs all six kernels, M={1,8,32,128}, K=N=4096, group=128, seed=1, 10 warmups, 100 timed iterations, both application store-flush and no-flush. Each kernel's launch mapping remains fixed across the series. Profile M=1 and M=128 naive/inblock separately after querying target metrics. Capture launch geometry, registers/thread, shared memory/block, occupancy, scheduler eligibility/issue activity, stalls including long scoreboard, SM/memory utilization, and supported DRAM read/write metrics. Nsight `--cache-control all` with application `--no-flush-l2` is its own cache method. Additional blocks are a launch property, not a guaranteed increase in achieved occupancy.

**Operational attempts and failures:**

1. Initial SSH failed locally with `socket: Operation not permitted`. Approved network escalation established gateway-to-c58 access. This was sandbox restriction, not cluster failure.
2. Branch creation required escalation because `.git` is read-only in the sandbox. Baseline content was preserved.
3. `/etc/sysconfig/prun` was absent on c58. An exploratory `set -e` command stopped reading it. Installed prun's default configuration subsequently launched one hostname through Open MPI. No launcher-policy change was needed.
4. An archive exclusion `build*` also excluded `scripts/build.sh`, causing the remote build to stop on a missing script. Explicit source-directory packaging corrected it. This was a packaging failure, not a CUDA compile failure.
5. SSH returned status 255 after the pilot sweep. Complete 48-row CSV, final log entries, and `gpu_after.csv` showed the sweep had completed. Preserve that diagnostic rather than rerun costly measurements solely because the connection exit was nonzero.
6. Adding the tail guard inside every naive packed-word loop slowed aligned inputs. The matched control below identified the regression. The pilot is preserved and the complete repaired-baseline series was rerun rather than cherry-picking replacement points.
7. The first literal SASS comparison failed because cuobjdump alignment whitespace differed with symbol-name length. Comparing normalized display whitespace then matched instructions and encoding/control lines. No kernel change was needed for this display-only mismatch.

**Pilot findings, not final speedups:** The first same-executable series completed all eight shape/policy combinations. Its naive denominator included an inner-loop tail guard even for aligned K=4096. Median event latencies below are in microseconds. They are preserved as evidence of the attempted implementation and must not be promoted to the final baseline comparison.

| M | Pilot store-flush naive | Pilot store-flush inblock | Pilot no-flush naive | Pilot no-flush inblock |
|---|---:|---:|---:|---:|
| 1 | 258.048 | 53.600 | 159.296 | 38.448 |
| 8 | 270.016 | 233.824 | 292.352 | 224.800 |
| 32 | 934.240 | 871.488 | 931.088 | 868.896 |
| 128 | 3464.352 | 3452.608 | 3431.680 | 3445.280 |

A same-node control compared the exact original `8a68227` naive source with the guarded implementation, retaining benchmark/host/dequantization objects and build conditions. Original versus guarded medians were 206.496 versus 258.048 us for store-flush M=1, 2606.400 versus 3464.544 us for store-flush M=128, 110.160 versus 159.296 us for no-flush M=1, and 2577.984 versus 3431.680 us for no-flush M=128. This control supports a real guard-related baseline regression. Merely comparing new c58 measurements with historical c7 data would not isolate it because driver/node conditions differ.

The chosen repair compiles `template<bool HasTail>` variants, using `if constexpr` for the final-word guard and dispatching by `K%8`. Full-word K with a partial final quantization group still uses the original aligned path. The `HasTail=false` emitted code matches the original naive code exactly after display-whitespace normalization: all 336 SASS lines, comprising 168 instructions plus continuation/control words, including addresses, opcodes, and scheduling control words. Reproduce with `python3 results/inblock_c58_20261106_final/verify_sass.py`. This repairs baseline preservation, not a new requested optimization. The repaired GPU checks passed. Every M/cache point and profile was rerun with that executable. Control/final hashes distinguish versions. Final evidence follows below.

**Final validation and artifacts:** All 48 final CSV rows pass. Each main-shape reconstruction check found zero bit mismatches among 16,777,216 weights. Small zero-weight cases produce exact zero. At M=128 the fused maximum absolute error is approximately 1.9462e-3 and relative L2 error approximately 2.0750e-4. The absolute maximum alone is not the tolerance test, which includes reference magnitude and RMS. Main M=128 validation checks 64 sampled rows, not every row. The repeated small-case gate and four sanitizer runs pass with zero reported errors.

Final raw evidence is in `results/inblock_c58_20261106_final/`: `timings.csv`, `timings.log`, `check.log`, `metadata.txt`, `source-build.sha256`, `check.sha256`, `gpu_before.csv`, `gpu_after.csv`, allocation/build/SASS records, and `ncu/`. Each final profile has its own log, stderr, `.ncu-repz`, and imported `_metrics.csv`. The target metric query is preserved compressed as `ncu/metric_names.txt.gz`. Profile commands are in `scripts/ncu_inblock.sh` and `profile.log`. Source/executable hashes identify code checkpoint `2b7d4fc`. Evidence was committed as `0d8fec0`. `analyze.py` asserts all 48 rows and matching comparison conditions, then generates `speedups.csv` and normalized `profile_summary.csv`. Both this script and `verify_sass.py` pass.

**Final latency:** The complete repaired-baseline series uses the contract's fixed node, executable, flags, shapes, seed, warmups, iterations, and cache policies. These are CUDA-event medians in microseconds, lower is better. Original-weight cuBLAS and W_hat cuBLAS use distinct references.

| Policy | M | naive | inblock | cublas | cublas_dq | dequant | unfused |
|---|---:|---:|---:|---:|---:|---:|---:|
| Store flush | 1 | 206.496 | 53.600 | 94.208 | 94.208 | 121.184 | 178.528 |
| Store flush | 8 | 206.496 | 233.824 | 86.048 | 86.048 | 120.480 | 186.048 |
| Store flush | 32 | 705.888 | 869.056 | 86.400 | 86.368 | 121.184 | 172.032 |
| Store flush | 128 | 2606.432 | 3452.928 | 118.784 | 116.736 | 120.512 | 229.344 |
| No flush | 1 | 110.192 | 38.176 | 83.520 | 83.520 | 115.680 | 136.768 |
| No flush | 8 | 222.688 | 224.512 | 52.800 | 52.800 | 116.288 | 142.912 |
| No flush | 32 | 701.696 | 869.344 | 30.304 | 30.272 | 115.680 | 148.704 |
| No flush | 128 | 2578.672 | 3445.312 | 106.048 | 106.048 | 116.288 | 193.824 |

M=1 inblock is 3.853 times faster than retained naive with the application store flush and 2.886 times faster without it, satisfying the pre-timing primary criterion. It is also faster than both cuBLAS paths at M=1 in this one series. At M=8 store-flush speedup is 0.883, and no-flush is 0.992. Both M=32 and M=128 regress under both policies. Therefore the benefit is shape-specific, not a replacement recommendation across the measured M range.

Within-run min/max/p90 in microseconds and median speedup quantify variation in the two fused paths. Speedup above 1 favors inblock. These statistics do not establish independent trial-to-trial reproducibility. Min/max/p90 for the other four baselines are also retained in the complete raw CSV.

| Policy | M | naive min–max | naive p90 | inblock min–max | inblock p90 | naive/inblock median |
|---|---:|---:|---:|---:|---:|---:|
| Store flush | 1 | 204.448–210.592 | 208.512 | 51.200–55.840 | 55.648 | 3.853 |
| Store flush | 8 | 205.472–211.392 | 208.576 | 232.768–235.936 | 234.848 | 0.883 |
| Store flush | 32 | 700.768–713.216 | 711.072 | 862.880–878.624 | 877.568 | 0.812 |
| Store flush | 128 | 2561.376–2609.600 | 2608.480 | 3451.552–3455.424 | 3454.592 | 0.755 |
| No flush | 1 | 110.048–113.952 | 112.192 | 37.760–41.344 | 40.192 | 2.886 |
| No flush | 8 | 220.416–226.688 | 222.784 | 223.776–235.552 | 225.536 | 0.992 |
| No flush | 32 | 695.520–711.488 | 704.768 | 867.872–870.112 | 869.952 | 0.807 |
| No flush | 128 | 2540.064–2580.736 | 2579.008 | 3444.256–3451.104 | 3446.336 | 0.748 |

**Final profiler evidence:** These are separate Nsight runs with `--cache-control all`, application `--no-flush-l2`, one selected launch, and queried target metrics. They are not warm/no-flush event measurements. Values below preserve profiler units, with decimal MB/GB distinct from MiB/GiB. Achieved active warps and issue active use their recorded active-cycle percentage denominators. SM throughput and wait stalls use elapsed-cycle percentage denominators.

| Metric | naive M=1 | inblock M=1 | naive M=128 | inblock M=128 |
|---|---:|---:|---:|---:|
| Blocks | 16 | 128 | 2048 | 16384 |
| Registers/thread | 32 | 32 | 32 | 32 |
| Static shared bytes/block | 0 | 1024 | 0 | 1024 |
| Dynamic shared bytes/block | 0 | 0 | 0 | 0 |
| Driver-reserved shared bytes/block | 1024 | 1024 | 1024 | 1024 |
| Total shared bytes/block | 1024 | 2048 | 1024 | 2048 |
| Active warps (% of sustained active peak) | 16.565 | 55.511 | 95.769 | 99.193 |
| Eligible warps/active cycle | 0.251 | 1.126 | 5.435 | 4.969 |
| Issue active (% of sustained active peak) | 21.075 | 63.679 | 85.784 | 86.391 |
| SM throughput (% of sustained elapsed peak) | 9.159 | 57.311 | 85.539 | 86.215 |
| Memory utilization (% of sustained elapsed peak) | 10.026 | 45.009 | 66.394 | 50.305 |
| DRAM throughput (% of sustained elapsed peak) | 10.026 | 45.009 | 0.814 | 0.616 |
| Long-scoreboard stall (% of sustained elapsed peak) | 5.335 | 25.574 | 15.987 | 14.204 |
| Wait stall (% of sustained elapsed peak) | 0.796 | 11.730 | 7.650 | 18.122 |
| DRAM reads (MB) | 8.663808 | 8.664832 | 9.704192 | 9.705216 |
| L2 traffic | 9.041024 MB | 9.035968 MB | 1.125170 GB | 1.153074 GB |

M=1's increased eligible warps, issue activity, and active warps are consistent with better utilization from the changed mapping. DRAM read bytes remain nearly equal, so the improvement is not explained by a reduction in those measured bytes. DRAM write counters report zero in these selected profiles even though each kernel writes C. Output data can remain cached, so zero observed DRAM write bytes does not mean no output stores occurred. Long-scoreboard percentage increases while elapsed time falls, illustrating why one normalized stall counter alone does not determine latency.

At M=128 both kernels already have many active warps and similar SM throughput, yet inblock is slower. Its wait-stall metric rises from 7.650% to 18.122%, eligible warps decline, and L2 traffic rises modestly. Additional runtime tail checks, shared reduction, and altered instruction scheduling are candidate explanations. This experiment does not isolate their individual costs. Higher active-warps percentage does not establish better performance.

**Limits:** One RTX 5060 Ti node, one seed, and one complete final timing series were measured. There are 100 per-kernel samples but no independently repeated final series or confidence intervals. Clocks were observed, not locked. Sanitizer evidence covers selected small shapes, main M=128 correctness is sampled, and profiling used invalidated caches rather than a validated steady-state warm profile. The store flush remains an intended policy, and `eff_gbps` remains modeled traffic. SASS parity is verified only for this compiler/architecture. These results do not establish end-to-end inference performance or portability to other GPUs.

**Decision:** Keep inblock separately selectable for the demonstrated M=1 use case. Reject replacing naive across shapes or adding automatic dispatch from this experiment. Preserve all pilot/control/final evidence. No additional optimization is implemented.

**Revisit only if:** Do not reintroduce the guarded aligned naive loop unless compiler/machine-code evidence and a matched control show that its regression is absent. Reconsider a universal inblock replacement only after an isolated change removes measured M>=8 regressions under consistent reruns of every point. A new GPU, compiler, or workload requires a fresh controlled series rather than transferring this result.

**Optional additional validation:** This was not required for E001's sampled-row methodology. To check every M=128 output row on the verified allocation, keeping the original tolerance:

```bash
prun ./build/w4a16_bench --M 128 --N 4096 --K 4096 --group 128 --seed 1 \
  --warmup 0 --iters 1 --check-rows 128 --kernels naive,inblock --no-flush-l2
```

**Optional independent timing replication:** No additional measurements are needed to complete E001. To assess between-series variability later, revalidate the allocation and use a fresh directory. Do not overwrite final evidence or combine new points selectively with this series. Run from the remote project root with the verified launcher environment above:

```bash
set -euo pipefail
export RUN_DIR="$PWD/results/inblock_replicate_$(date -u +%Y%m%dT%H%M%SZ)"
export SOURCE_COMMIT=2b7d4fc  # verify the actual source checkpoint first
mkdir "$RUN_DIR"
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/allocation.txt"
prun bash scripts/check_inblock.sh > "$RUN_DIR/check.log" 2>&1
```

Inspect the complete successful prerequisite log before the expensive sweep. Then run every point using the same executable and methodology:

```bash
# The successful gate has already created check.sha256.
prun bash scripts/run_inblock.sh > "$RUN_DIR/timings.log" 2>&1
```

The script captures fresh metadata/hashes and both cache policies across all four M values. Replication does not imply locked clocks. Record any new clock control or changed conditions as a separate series.

## E002 — Candidate aligned-K specialization

Status: accepted on 2026-11-06 at code checkpoint `91bd042`. The predeclared criterion is met in the completed series. CUDA build, expanded GPU/sanitizer gate, full timing sweep, output-bit comparisons, and separate profiling are complete. The user authorized E002 while preserving E001. The hypothesis, alternatives, and acceptance criteria below were committed before implementation in `dc37ec6`. Do not interpret the local CPU pass as CUDA validation.

**Problem:** E001 retains a runtime final-word guard in the candidate inner loop even at aligned K=4096. The isolated naive control demonstrated that such a guard can change code generation enough to slow a kernel, but this does not establish the effect on inblock.

**Hypothesis:** Compile-time aligned/tail variants in inblock may reduce aligned-case instruction overhead and improve its larger-M latency while preserving the existing layout, numerical contract, and tail correctness. The magnitude and affected shapes are unknown.

**Change and alternatives chosen before implementation:** Keep `src/cuda/w4a16_inblock.cu` and its `inblock` selector unchanged. Add `inblock_specialized` in a separate source file with `template<bool HasTail>`. Dispatch `HasTail=false` for K divisible by eight and `true` otherwise. Guard final-word bounds with `if constexpr`. Preserve the mapping, eight packed-word intervals, FP16 reconstruction, FP32 accumulation/reduction, and FP16 output. A shared refactor of E001 was considered but would change its compilation unit and baseline provenance. Copying this small experimental kernel keeps the original source intact. No tiling or layout change is included.

**Success criterion fixed before implementation:** Pass the unchanged FP64 tolerance, zero reconstructed-weight mismatches, all checked E002 outputs bit-identical to E001, and zero errors in selected sanitizer cases covering both aligned and tail dispatch. Require at least 1.05 median speedup over same-executable E001 for both M=32 and M=128 under both application cache policies, with no more than a 5% M=1 latency regression. The 5% threshold distinguishes a useful improvement from a small timing difference in this experiment. Report every M point and p90, including regressions. A small or inconsistent change remains inconclusive and requires complete independent repeat series before a performance acceptance claim.

**Evidence required:** Inspect aligned and tail candidate SASS, compare retained E001 SASS with its saved version, repeat the full correctness/sanitizer gate, and rerun every M/cache point with E001 and candidate in the same executable. Profile M=1 and M=128 separately with queried metrics. Keep all E001 records. Use a distinct E002 remote directory and result path. Do not change timing events or flushing.

**Implementation and baseline preservation:** `src/cuda/w4a16_inblock_specialized.cu` contains the separate template kernel. `src/cuda/w4a16_inblock.cu` remains unchanged. The benchmark exposes seven selectors, including `inblock` and `inblock_specialized`. Its untimed specialized validation also launches E001 and compares all M*N FP16 output bits, regardless of the FP64 row-sampling limit. Any mismatch stops measurements. This comparison is additional to the original FP64 threshold and all-weight reconstruction check. Layout, mapping, reduction order, event placement, and flush policy are unchanged.

**Versions and build:** Starting repository checkpoint `dc37ec6` follows E001's final documentation checkpoint `84afdbf`. E001 measured source is `2b7d4fc`. E002 implementation checkpoint is `91bd042`, and evidence checkpoint is `c7ea4b4`. No kernel code changed after measurement. Metadata's `baseline_commit=8a68227` field identifies the historical naive origin. The actual E002 comparison baseline is retained E001 inblock from the preserved `2b7d4fc`/`84afdbf` project state, rerun in the same `91bd042` executable. Exact executable/source hashes are recorded with the build/measurement artifacts. Same-executable `inblock` is E002's primary latency denominator. Historical E001 timings are context, not that denominator. The measured build uses Release, CUDA sm_120, nvcc 13.4.92, GCC 12.4.0, Open MPI 4.1.6, and existing portable host flags. One prun-launched process uses 16 OpenMP threads on c58. Exact flags, binding, versions, and hashes are in `results/e002_c58_20261106/metadata.txt` and `source-build.sha256`.

**Validation plan and current evidence:** Local `bash scripts/build.sh`, `ctest --test-dir build --output-on-failure`, and shell syntax checks passed. Independent static review identified no issue. The c58 CUDA build and complete small-case gate passed. Every E002-versus-E001 small-case output-bit comparison reports zero mismatches. All six Compute Sanitizer runs report zero errors. The small GPU gate covers K={1,3,7,8,9,40,65,128,129,136,137,257}, N=33, M=3, group=128, seed=1, plus N={1,31,32,63}, K=137, M=2, seed=7, and a zero-weight case. K=128 and K=136 exercise aligned dispatch, including a partial final quantization group at K=136. K=3 and K=137 exercise tails. Every small output row is checked. Six Compute Sanitizer invocations cover memcheck and synccheck at each K={3,136,137}, checking naive, E001, E002, and dequantization.

**Environment and evidence paths:** ARC job 328410 was revalidated RUNNING on c58, RTX 5060 Ti, driver 615.71.09. This is the existing user-owned allocation, not a new resource request. Revalidate state/remaining time before later reuse and leave it running. Remote source is `/home/hpuvvad/csc548-w4a16-e002-20261106`, isolated from E001. Remote/local result directory is `results/e002_c58_20261106/`. Full allocation, CPU/ISA/topology/NUMA/affinity, compiler/MPI/GPU/tool versions, source/executable hashes, flags, commands, and raw logs must accompany measurements.

**Reproduction procedure:** Use the user-supplied gateway-to-c58 route and the verified launcher environment recorded in E001, after rechecking the live allocation. From the isolated E002 project root, export an absolute RUN_DIR and the actual SOURCE_COMMIT. Run the build, `prun /bin/hostname`, and `prun bash scripts/check_inblock.sh`, saving logs. The successful gate writes `check.sha256`. Inspect it before `prun bash scripts/run_inblock.sh`. The run script verifies the executable marker, refuses to overwrite timings.csv, and sweeps all seven kernels at every M={1,8,32,128}, K=N=4096, group=128, seed=1, 10 warmups, 100 timed iterations under both application policies. Query metrics on the GPU and run `prun bash scripts/ncu_inblock.sh` separately for naive/E001/E002 at M=1 and M=128 with profiler cache invalidation. Exact executed commands are retained as `prepare_commands.sh` and `measure_commands.sh` in the result directory. Build/gate logs are `build.log` and `check.log`, launcher output is `launcher.log`, timing output is `timings.log`, and profiling output is `profile.log`. Both source/build hashes and executable prerequisite markers are preserved.

**Attempts and failures:** No correctness, launcher, CUDA-build, or sanitizer failure was observed in E002. All predeclared conditions passed in this series. The small 4.0% store-flush M=1 gain is not evidence of independent repeatability. Preserve E001's guard-related pilot failure as the motivating control rather than reporting a new E002 failure that did not occur.

**Machine-code findings:** E001 source remains unchanged and its rebuilt device code matches all 480 saved SASS lines, including instruction encodings and scheduling control words, after display-whitespace normalization. E002's tail variant also matches those same 480 E001 lines exactly. `verify_sass.py` reproduces these assertions from `e001_saved.sass`, `e001_rebuilt.sass`, and `e002.sass`, and writes `sass_summary.json`.

| Static emitted-code property | E001 | E002 aligned | E002 tail |
|---|---:|---:|---:|
| Instructions including padding | 240 | 216 | 240 |
| BRA occurrences | 11 | 4 | 11 |
| ISETP.GE.AND occurrences | 12 | 4 | 12 |
| FFMA occurrences | 8 | 8 | 8 |

These are static code counts, including padding, not dynamic executed instructions or operation counts. They establish an aligned code-generation difference while retaining the unrolled FFMA count. They do not alone establish a latency benefit. Separate profiling queries `smsp__inst_executed.sum` to measure dynamic executed warp instructions.

**Final validation and artifact checks:** All 56 timing rows pass the original tolerance with zero W_hat mismatches. Each main shape confirmed bit-exact reconstruction of all 16,777,216 FP16 weights. All eight main E001/E002 comparisons found zero mismatches across every M*N output, including all 128 rows at M=128. The original FP64 reference check at M=128 still samples 64 rows. Bitwise agreement with E001 and sampled FP64 validation are distinct checks. The expanded small-case gate and six sanitizer runs pass, including aligned K=136 with a partial group, tail cases, odd N, and exact-zero weights. At M=128, both E001 and E002 have maximum absolute error 1.9462e-3 and relative L2 error 2.0750e-4 under each policy.

`python3 results/e002_c58_20261106/analyze.py` validates the 56-row series, cache/shape/build consistency, main full-output comparisons, selected sanitizers, and declared criteria. It writes `comparisons.csv`, `criteria.json`, and normalized `profile_summary.csv`. Every condition in `criteria.json` is true. `python3 results/e002_c58_20261106/verify_sass.py` validates retained E001 machine-code parity and generates `sass_summary.json`. Exact source/executable hashes, flags, metadata, allocation/launcher logs, pre/post GPU observations, SASS, raw profiles, and complete commands remain in the same result directory. No historical E001 records or source were replaced.

**Final latency:** These event medians are in microseconds, lower is better. All seven kernels were rerun in one executable on c58, with K=N=4096, group=128, seed=1, 10 warmups, and 100 timed iterations under both application policies. The primary E002 denominator is E001 in this executable, not a historical E001 run.

| Policy | M | naive | E001 inblock | E002 specialized | cublas | cublas_dq | dequant | unfused |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Store flush | 1 | 206.496 | 53.600 | 51.520 | 94.208 | 94.208 | 121.152 | 178.512 |
| Store flush | 8 | 206.496 | 233.824 | 184.672 | 86.048 | 86.608 | 120.512 | 186.688 |
| Store flush | 32 | 704.976 | 876.224 | 671.408 | 86.480 | 86.368 | 121.184 | 172.000 |
| Store flush | 128 | 2606.400 | 3452.928 | 2614.944 | 118.784 | 116.736 | 120.496 | 229.344 |
| No flush | 1 | 110.144 | 38.432 | 29.952 | 83.520 | 83.392 | 115.696 | 136.768 |
| No flush | 8 | 221.824 | 224.512 | 169.216 | 53.904 | 52.848 | 116.288 | 142.912 |
| No flush | 32 | 703.712 | 868.896 | 654.912 | 30.368 | 30.272 | 115.680 | 148.704 |
| No flush | 128 | 2578.544 | 3445.312 | 2605.632 | 108.096 | 106.048 | 116.288 | 193.824 |

E002 exceeds 1.05 E001/E002 median speedup for both M=32 and M=128 under both policies, and neither M=1 policy regresses. It therefore meets the fixed acceptance criterion. Large-M speedups are approximately 1.305–1.327 in this series. M=1 store-flush speedup is only 1.040, while no-flush speedup is 1.283. Treat the smaller difference as tentative without independent series replication.

The following fused-path ranges/p90 values are in microseconds. Ratios above 1 favor E002. Min/max/p90 for every other baseline remain in the raw CSV.

| Policy | M | E001 min–max | E001 p90 | E002 min–max | E002 p90 | E001/E002 median | naive/E002 median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Store flush | 1 | 51.200–55.840 | 55.616 | 49.120–53.600 | 51.584 | 1.040 | 4.008 |
| Store flush | 8 | 233.472–236.032 | 235.872 | 182.656–186.784 | 184.736 | 1.266 | 1.118 |
| Store flush | 32 | 862.880–878.592 | 878.240 | 670.368–674.208 | 672.448 | 1.305 | 1.050 |
| Store flush | 128 | 3451.552–3455.008 | 3453.952 | 2612.896–2623.488 | 2622.464 | 1.320 | 0.997 |
| No flush | 1 | 38.016–41.184 | 40.192 | 28.000–31.040 | 29.952 | 1.283 | 3.677 |
| No flush | 8 | 223.808–235.456 | 225.536 | 168.224–177.728 | 169.536 | 1.327 | 1.311 |
| No flush | 32 | 859.680–874.208 | 869.952 | 653.856–663.424 | 655.936 | 1.327 | 1.075 |
| No flush | 128 | 3444.096–3451.328 | 3446.336 | 2598.592–2607.296 | 2606.656 | 1.322 | 0.990 |

At M=128, E002 is near naive but slightly slower: 2614.944 versus 2606.400 us with store flushing and 2605.632 versus 2578.544 us without it. Acceptance against E001 is not a claim of universal superiority over naive or cuBLAS. Original-weight and W_hat cuBLAS continue to have distinct reference operands.

**Profiler evidence:** Six separate Nsight runs cover naive/E001/E002 at M=1 and M=128. They use target-queried metrics, `--cache-control all`, and application `--no-flush-l2`. They do not measure the application's steady-state no-flush policy. Dynamic executed-instruction totals below are `smsp__inst_executed.sum` warp-instruction counter values, distinct from the static SASS counts above. Percentage denominators are shown with each metric.

| Metric | E001 M=1 | E002 M=1 | E001 M=128 | E002 M=128 |
|---|---:|---:|---:|---:|
| Dynamic executed warp instructions | 9,043,456 | 6,955,520 | 1,157,562,368 | 890,306,560 |
| Registers/thread | 32 | 34 | 32 | 34 |
| Static shared bytes/block | 1024 | 1024 | 1024 | 1024 |
| Dynamic shared bytes/block | 0 | 0 | 0 | 0 |
| Driver-reserved shared bytes/block | 1024 | 1024 | 1024 | 1024 |
| Total shared bytes/block | 2048 | 2048 | 2048 | 2048 |
| Active warps (% of sustained active peak) | 55.515 | 56.020 | 99.195 | 99.064 |
| Eligible warps/active cycle | 1.122 | 1.221 | 4.969 | 6.056 |
| Issue active (% of sustained active peak) | 63.333 | 63.455 | 86.392 | 88.138 |
| Wait stall (% of sustained elapsed peak) | 11.676 | 5.083 | 18.121 | 7.950 |
| Long-scoreboard stall (% of sustained elapsed peak) | 25.486 | 31.901 | 14.182 | 12.275 |
| Memory utilization (% of sustained elapsed peak) | 44.895 | 58.630 | 50.301 | 66.722 |
| DRAM throughput (% of sustained elapsed peak) | 44.895 | 58.630 | 0.616 | 0.814 |
| DRAM reads (decimal MB) | 8.664832 | 8.664576 | 9.705728 | 9.705216 |
| L2 traffic | 9.033888 MB | 9.020608 MB | 1.153263 GB | 1.152521 GB |

Dynamic executed warp instructions fall by 23.0878% at both profiled M values. Wait-stall percentages decline, while M=128 active warps remain approximately 99% and registers increase from 32 to 34. Measured DRAM/L2 bytes remain nearly unchanged. These observations support reduced instruction/control overhead as an explanation for the specialization benefit. The isolated source change also changes compiler scheduling and register allocation, so this experiment does not apportion their separate contributions. M=128's memory-utilization metric is much higher than its DRAM-throughput metric, so they must not be conflated with each other or with modeled `eff_gbps`.

Observed DRAM write bytes are zero in these selected profiles even though every kernel writes C. Cached output stores need not reach DRAM during the measured kernel. The zero counter is not proof of absent output writes. Raw reports, imported wide metric CSVs, queried metric names, and normalized summaries remain under the E002 result directory.

**Limits:** One GPU/node, seed, compiler configuration, and complete E002 event series were measured. No independent final repetitions, confidence intervals, locked clocks, or steady-state warm profiler runs were added. Within-run samples/ranges do not measure between-series variability. Sanitizer covers selected small cases. All-row E001/E002 bitwise agreement does not expand the sampled FP64 reference to all rows. Application flushing remains an intended cache policy rather than demonstrated eviction. The result establishes this specialization on the measured setup, not other GPUs or end-to-end LLM inference.

**Decision:** Accept the separately selectable `inblock_specialized` implementation for the declared E002 criterion. Retain `inblock` unchanged as the E001 comparison, preserve naive and other baselines, and add no automatic M-based kernel selection or further optimization. The required K-divisibility dispatch remains. All required E002 work is complete.

**Revisit only if:** Reevaluate on a changed GPU/toolchain/workload or new independent measurements that challenge the observed benefit. Repeating the same accepted configuration without a new question is unnecessary. Any further mechanism, including reuse across M rows, requires a separately requested experiment with its own hypothesis and criterion. High L2 traffic can motivate inspecting row reuse but does not establish it as the isolated bottleneck.

## Template for the next requested experiment

```text
## E00X — Name
Status: proposed / implemented / measured / accepted / rejected / inconclusive
Problem:
Hypothesis and success criterion, recorded before implementation:
Change and alternatives considered:
Baseline and candidate: commits, kernel names, build settings
Validation: shapes, numerical error, sanitizer outcomes
Measurement: GPU, cache policy, commands, raw paths, variability
Findings: measured results and profiler evidence, with limits
Decision: keep / reject / investigate, and why
Revisit only if: new evidence or changed conditions
```

## E003 — Reuse reconstructed weights across input rows

Status: accepted for the measured larger-M cases on 2026-11-06. Both row counts meet the predeclared criterion. Authorized after E002 completion at `c188b92`. The plan below was committed before implementation in `53bb695`. Implementation is `5a7bcaf`, final measurement-script/source checkpoint is `25b2291`, and evidence checkpoint is `a16cbc5`. The one-row E002 control remains unchanged and selectable.

**Problem:** E002 repeats packed-weight loads, scale loads, unpacking, and FP16 reconstruction for every input row. Its M=128 invalidated-cache profile records about 1.15 GB of L2 traffic. This is evidence of repeated work, not proof of the critical latency bottleneck.

**Hypothesis:** Reconstructing each weight once for two or four input rows will reduce repeated weight-loading and dequantization work enough to outweigh extra registers and reduced block count.

**Design and alternatives:** Preserve the E002 source and `inblock_specialized` one-row selector, plus naive and E001. Add a separate row-reuse source with compile-time Rows={2,4}, exposed as `inblock_m2` and `inblock_m4`. Keep block=(32,8), contiguous packed-word slice intervals, group=128, packing, FP16 reconstruction, and each row's FP32 FMA/reduction order. Each unpacked reconstructed weight feeds Rows independent accumulators. Shared partials become [Rows][8][32]. Grid y becomes ceil(M/Rows). All threads publish partials and reach the barrier, including invalid columns and rows. Compile-time K-tail and M-tail variants preserve the bounds-free aligned path, while incomplete tiles guard activation reads and output writes. A shared refactor of E002 was considered and rejected because it would alter control provenance. Cross-block reuse, a changed packed format, group-size tuning, automatic dispatch between row counts, and Tensor Cores are outside this experiment.

**Success criterion fixed before implementation:** Unchanged FP64 tolerance, exact reconstructed weights, every E003 output bit equal to E002 outside timing, and zero selected sanitizer errors. Evaluate each candidate separately. Accept as a useful larger-M selector only if same-executable E002/candidate median latency is at least 1.05 at both M=32 and M=128 under both application policies. Report M=1/8 and supplemental M=5 tail behavior, including every regression, without a small-M non-regression requirement or a universal-replacement claim. If a decision depends on a borderline or inconsistent effect, repeat the entire timing series independently before accepting. A candidate that fails is retained for comparison and recorded as rejected or inconclusive, with revisit conditions.

**Validation and measurement plan:** Extend the GPU gate to both row counts, aligned/tail K, incomplete M tiles including M={1,2,3,4,5,7}, odd N, empty K slices, and zero weights. Use full-output bitwise E002 comparisons even at M=128. Run memcheck and synccheck on aligned and tail K with incomplete row tiles. Preserve numerical tolerances, seed=1, K=N=4096, group=128, 10 warmups, 100 event iterations, and both existing application cache policies. Rerun all M={1,8,32,128} points in the new executable with naive, E002, and both candidates. Add M=5 under both policies as a separately identified tail point. Collect independent cache-invalidated Nsight profiles for naive/E002/both candidates at those shapes, querying executed warp instructions, L2 bytes, registers, shared memory, occupancy/issue/stalls, and DRAM bytes. Keep profiling separate from event timing. Verify E002 and naive device code against saved controls. Preserve raw commands, logs, hashes, profiles, and metadata in a distinct E003 directory. No group/layout/traffic-model change is intended.

**Implementation and prerequisite:** Added `src/cuda/w4a16_inblock_rows.cu` and both selectors without editing any existing kernel source. The benchmark performs full-output E002 bitwise comparisons before candidate timing, independently of FP64 row sampling. The small GPU gate includes M={1,2,3,4,5,7}, aligned K=136 and tail K=137, plus the existing short-K/odd-N/zero-weight cases. Six memcheck/synccheck runs use incomplete M=5 tiles at K={3,136,137}. Local CPU build/CTest and shell syntax checks pass. The isolated c58 CUDA build, smoke launch, complete GPU gate, and all six sanitizer runs pass. Saved/rebuilt naive and E002 SASS compare exactly, including instruction encodings/control words after display-whitespace normalization. All eight E003 row-count/K-tail/M-tail device specializations are emitted.

**Review correction before profiling:** Independent read-only review found that template-argument Nsight filters require `--kernel-name-base demangled`. The default function-name basis drops template arguments. Corrected the script before any profiling, as documented by the official Nsight CLI reference. No failed profiler run or performance claim came from the uncorrected filter. Initial progress inspection reached the build before check.log existed and reported that missing file. The actual prerequisite continued successfully. This was an inspection timing issue, not a failed correctness run.

**Control-check diagnostic:** The first E003 SASS assertion compared complete symbol names and falsely reported a control change because CUDA embeds a build-specific anonymous-namespace hash. Matched each specialization by its kernel/template suffix instead, retaining exact comparisons of every instruction/encoding/control line. Both naive variants (400/336 lines) and E002 variants (480/432 lines) match their saved controls exactly. No device code was changed in response. The initial shell inspection did not stop after that analysis assertion, so its subsequent gate-success message alone was insufficient evidence of SASS parity. The corrected check was rerun with fail-fast shell handling before measurements.

**First measurement attempt and profiler failure:** The first 90-row event sweep completed with passing correctness. Nsight then stopped while importing the first row-reuse report because no kernel matched `w4a16_inblock_rows_kernel<2,`. The target demangler spells this argument `<(int)2,`, despite the documented demangled basis being selected. The available-kernel diagnostic is preserved. Changed the filter to the common `w4a16_inblock_rows_kernel` name, since each profiling invocation selects exactly one benchmark kernel, and assert the exported row-count-specific name in analysis. Preserve the first complete timings and partial profiles under `results/e003_c58_20261106_attempt1/`. Rerun the entire timing/profile series with the corrected measurement script, retaining the exact same already-validated executable, flags, topology, seed, group, event placement, and cache policies. This is a measurement-script failure, not a kernel correctness failure. No partial profile result supports acceptance.

**Archive guard during restart:** Sending the saved original profiler script created the intended attempt-archive directory before the restart script's nonexistence guard, so the first restart stopped before measurements. Moved that single script into the original result directory and removed the now-empty archive directory with `rmdir`, then reran the guarded move. No evidence was overwritten or deleted. The original full timing/partial profile attempt is preserved separately. The final prerequisite reuses the unchanged executable only after checking its saved SHA 256 gate.

**Versions, execution, and reproduction:** Final evidence is `results/e003_c58_20261106/`, evidence checkpoint `a16cbc5`. The remote source is `/home/hpuvvad/csc548-w4a16-e003-20261106`. Implementation checkpoint `5a7bcaf` and final measurement-script checkpoint `25b2291` use the same executable, SHA 256 `5ee05fc2046d6e821cdd6f7c9b944541eb7f9e00e9f5e7a50b0e61a7887a232e`. Metadata's `baseline_commit=8a68227` identifies the historical naive origin, not the E003 denominator. The actual controls are same-executable naive and E002. Exact commands are in `prepare_commands.sh`, `restart_commands.sh`, and `measure_commands.sh`. `resource_usage.txt` records output of the separate post-measurement cuobjdump command retained in `measure_commands.sh`. The first timing series and interrupted profiles are retained in `results/e003_c58_20261106_attempt1/`, including its original profiler script. Attempt-local analysis scripts are frozen snapshots, not acceptance tools for an incomplete profile set. Use the final directory's analysis scripts.

ARC documentation and job 328410 were revalidated before launch. The user-owned allocation was RUNNING on c58, RTX 5060 Ti 16 GB, sm_120, 36 SMs, 32 MiB L2. One `prun` process used 16 OpenMP threads on the allocated AMD EPYC 9115, with explicit rank/thread/binding environment in the command files. Driver 615.71.09, CUDA 13.4.92, GCC 12.4.0, Open MPI 4.1.6, Nsight Compute 2026.3.1, and Compute Sanitizer 2026.3.0 were retained. Release CUDA flags are `-O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_120,code=[compute_120,sm_120] -lineinfo`. Host flags are portable, with no `-march=native`. Metadata, CPU ISA/topology/NUMA/affinity, versions, source/build hashes, storage path, launcher log, and pre/post GPU observations accompany the results. Clocks were observed, not locked. The final pre-sweep observation was idle 180 MHz SM / 405 MHz memory, and the post-sweep observation was 2737 MHz SM / 14001 MHz memory. These observations do not establish per-launch clock stability. No new allocation was submitted, and the user's allocation was left running.

**Validation:** Local CPU/CTest and shell syntax checks pass. The target CUDA build, one-rank launcher smoke test, full small GPU gate, and six zero-error Compute Sanitizer runs pass. All 90 final CSV rows pass with zero mismatches across all 16,777,216 reconstructed FP16 weights per main-shaped invocation. Twenty E003/E002 comparisons cover every M*N FP16 output, including all 128 rows at M=128. All match exactly. Ten E002/E001 full-output comparisons also match. The separate FP64 threshold remains unchanged and samples 64 rows at M=128, with exact row indices logged. M=5 and all other small M check every row. At M=128, both row variants retain E002's `max_abs_err=1.9462e-3` and `rel_l2_err=2.0750e-4`. Every bitwise comparison is outside event timing. All eight emitted row/K-tail/M-tail variants have zero local-memory and stack allocation in cuobjdump resource output. Saved/rebuilt naive and E002 instruction/control lines match exactly.

**Latency:** The following values are final-series median/p90 microseconds, lower is better. All four selectors share the same executable, inputs, seed=1, K=N=4096, group=128, 10 warmups, and 100 iterations. Store-flush and no-flush are the application policies in the contract. M=5 is the supplemental incomplete-row-tile point and is excluded from the four-point standard-series acceptance criterion.

| Policy | M | Naive median/p90 us | E002 median/p90 us | Two rows median/p90 us | Four rows median/p90 us |
|---|---:|---:|---:|---:|---:|
| Store flush | 1 | 206.528/208.544 | 51.520/51.584 | 51.488/52.928 | 63.136/63.488 |
| Store flush | 8 | 206.496/207.552 | 184.640/184.704 | 124.576/124.928 | 115.040/115.136 |
| Store flush | 32 | 704.864/708.928 | 671.424/672.448 | 432.448/432.480 | 390.816/391.840 |
| Store flush | 128 | 2606.400/2608.448 | 2621.968/2622.144 | 1670.464/1670.560 | 1502.208/1505.952 |
| Store flush | 5 | 204.448/204.800 | 123.232/125.280 | 102.048/102.432 | 114.336/114.688 |
| No flush | 1 | 110.144/112.192 | 29.952/29.984 | 36.416/36.480 | 58.944/58.976 |
| No flush | 8 | 222.672/222.784 | 169.216/169.536 | 114.240/114.272 | 111.728/112.128 |
| No flush | 32 | 697.600/706.112 | 654.912/655.936 | 419.072/419.392 | 386.624/387.648 |
| No flush | 128 | 2578.656/2579.008 | 2605.632/2606.656 | 1655.040/1656.064 | 1499.712/1500.736 |
| No flush | 5 | 161.344/163.360 | 107.776/108.096 | 89.632/89.664 | 112.192/112.256 |

Speedup is same-policy control median divided by candidate median, so values above one favor reuse. Both row counts exceed 1.05 at M=32 and M=128 under both policies. The two-row E002 speedups there are 1.553–1.574, and the four-row speedups are 1.694–1.745. Four rows also beats same-executable naive by 1.719–1.804 at those points. M=8 improves against both controls. Full ratios for every shape and both denominators are in `comparisons.csv`.

At M=1, no-flush latency increases from E002's 29.952 us to 36.416 us for two rows and 58.944 us for four rows. Store-flush two-row latency is effectively tied with E002 at 51.488 versus 51.520 us in the final run, while the first run measured 52.800 versus 51.520 us. This small cold difference is inconclusive. Four rows regresses to 63.136 us under store flushing. M=1 offers no row reuse and still pays the tail/resource overhead.

At the M=5 tail point, two rows improves by 1.208 store-flush and 1.202 no-flush versus E002. Four rows improves by 1.078 under store flushing but regresses to 112.192 versus 107.776 us without flushing. Four rows therefore loses to two rows on this tail despite reconstructing weights fewer times. Every output still matches E002, and memcheck/synccheck exercise M=5 with both K-tail paths.

**Repeated-series evidence:** The first complete 90-row timing series preceded the profiler failure. The second complete series uses the identical executable and event conditions, with only the profiler filter corrected. Both series meet the larger-M criterion for both candidates. `repeat_comparison.csv` reports every same-key median pair. Across naive/E002/two-row/four-row points, the largest median decrease is 2.485% for two-row M=1 store-flush, and the largest increase is 1.438% for four-row M=8 no-flush. This supports the much larger M=32/128 improvements but does not establish broad device/seed or long-term reproducibility. The first series' profiling remains incomplete and is not combined with the final counter series.

**Profiler evidence:** Twenty final profiles cover naive, E002, two rows, and four rows at M={1,8,32,128,5}. Each uses `--cache-control all`, `--warmup 0 --iters 1 --nocheck --no-flush-l2`, one matching kernel, and queried supported metrics. These independently invalidated-cache profiles do not represent application steady-state no-flush cache residency. Profile replay duration is separate from the event latency table. Raw `.ncu-repz`, exported metrics, logs, stderr, and compressed supported-metric names are preserved. The table below reports M=128. Instruction counts are dynamic executed warp instructions, not scalar FLOPs. L2 and DRAM counters use decimal MB, not MiB. Static shared storage excludes the driver's additional 1024 bytes per block.

| Kernel | Blocks | Registers/thread | Static shared B/block | Warp instructions, million | L2 MB | DRAM read MB | Active warps % | Issue active % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive | 2048 | 32 | 0 | 864.944128 | 1123.981 | 9.710 | 95.795 | 85.754 |
| E002 | 16384 | 34 | 1024 | 890.306560 | 1151.328 | 9.852 | 99.064 | 88.141 |
| Two rows | 8192 | 39 | 2048 | 551.043072 | 641.144 | 9.712 | 98.653 | 85.985 |
| Four rows | 4096 | 39 | 4096 | 380.989440 | 383.622 | 9.708 | 97.854 | 65.624 |

Relative to E002 at M=128, two rows reduces executed warp instructions by 38.106% and L2 bytes by 44.313%. Four rows reduces them by 57.207% and 66.680%. At M=8 and M=32 the instruction reductions are the same, with L2 reductions in the same direction. Both aligned row variants use 39 registers/thread versus E002's 34. Active warps remain above 97% at M=128 in both candidates, although four-row issue activity is lower. The measured instruction/traffic reductions and faster event latency support the reuse hypothesis for these larger shapes. They do not isolate instruction reduction from the effects of additional independent accumulators, compiler scheduling, or reduced block count. DRAM reads remain around 9.7–9.9 MB, so the gain is not evidence of a comparable DRAM-traffic reduction.

At M=1, two/four rows instead execute 25.670%/74.941% more warp instructions than E002, with slightly higher L2 bytes. Both retain 128 blocks, since a partial row tile cannot reduce block count below one tile per column block. Tail variants use 36/38 registers rather than the aligned variants' 39, and static shared memory remains 2048/4096 bytes. At M=5, two/four rows reduce warp instructions by 24.580%/29.996% and L2 bytes by 35.247%/55.809%, yet four rows remains slower under no-flush timing. This is direct evidence that fewer loads/instructions alone does not determine latency. Reduced parallelism, row guards, and resources change together in this experiment, and their individual costs were not isolated.

**Decision and designs that did not work universally:** Accept both explicit row-reuse selectors for the measured M=32/128 criterion. Four rows is fastest among these fused variants at M=8/32/128 in both application policies. Retain two rows for comparison and its better M=5 behavior. Preserve E002 as the one-row control and the better M=1 option. Reject a universal replacement by either reuse variant and do not add automatic shape dispatch. Unchanged quantization/group/layout and per-row arithmetic satisfy the contract. The failed template-specific profiler filter and false SASS symbol comparison are operational failures, not evidence against weight reuse. No other optimization was implemented.

**Revisit only if:** Revisit reuse at M=1 only if new measured evidence identifies a way to avoid unused-row/tail overhead without losing the control's parallelism, or the hardware/compiler changes. Revisit four-row tails only with evidence that their guard/unused-slot or parallelism cost can be reduced enough to overcome the observed M=5 no-flush loss. Revisit row-count dispatch only after a separately authorized experiment establishes stable decisions across more M, K, N, seeds, and repeated runs. More rows, split-tail launches, layout changes, Tensor Cores, and asynchronous transfers remain untested candidates, not failed implementations.

**Limits and continuation:** Two complete timing series, one GPU/node/seed, fixed main K/N/group, observed unlocked clocks, and one final invalidated-cache profile series. No independent steady-state warm profile, exhaustive FP64 M=128 check, or full cross-shape tuning was performed. The application store flush does not prove eviction. Modeled `eff_gbps` remains unchanged and is not measured DRAM bandwidth. Recompute summaries with `python3 results/e003_c58_20261106/analyze.py` and `python3 results/e003_c58_20261106/verify_sass.py`. Analysis checks all 90 final rows, exact profile-shape/selector set, full-output comparisons, sanitizer summaries, source hashes, baseline code parity, repeat settings, and equal binary hashes. Await the user's next requested experiment. Revalidate the live allocation before any reuse and keep it intact.

## E004 — Maintained Marlin integration and diagnostic comparison

Status: complete. Integration and all four gates passed. Recommend one bounded small-M launch-resource ablation, with no established speedup or novelty claim. E003 checkpoint `c1c6423` was clean before work. External kernels and retained project kernels remain unchanged.

**Objective and hypothesis:** Determine whether maintained dense INT4 W4A16 Marlin exposes a reproducible, tractable limitation on the actual RTX 5060 Ti. A clear no-go is acceptable. Faster external execution alone is not an optimization opportunity or novelty claim.

**Design and preservation:** Inspect and pin vLLM source, start from its kernel tests/benchmarks, directly invoke the GEMM operator, and losslessly repack existing codes/scales. Prefer an isolated environment and adapter over changing this project's build dependencies. Preserve all E001–E003 sources, selectors, evidence, and user allocations. No serving engine, model download, external kernel patch, or E005 implementation.

**Gate 1 — prerequisites and compatibility:** Verify live allocation/GPU, Python/CUDA/runtime and build requirements. Pin source commit, dependencies, license, build flags and hashes. Inspect integer encoding, tile packing, scale permutation/indexing, matrix orientation, reconstruction rounding, intermediate reductions, padding and workspace. Validate conversion and small valid shapes before M=1/32, K=N=4096. Existing tolerance remains unchanged. Distinct numerical operations require separate labels/references. Confirm executed kernel, reject fallback. A concrete integration blocker ends dependent gates after bounded diagnosis and independent documentation/tooling.

**Gate 2 — performance screening:** M={1,2,4,8,16,32,64,128}, (K, N)={(4096,4096),(4096,11008),(11008,4096)}, group=128, seed=1, both existing store-flush/no-flush policies. Unsupported shapes are explicit. Compare Marlin, original/reconstructed-weight cuBLAS, retained naive/E002/E003, and inexpensive unfused context using identical operands. CUDA events exclude setup/repacking/tuning, which are recorded separately. Start with a short complete screen, repeat promising/surprising cases independently. Capture per-launch samples, warmups/iterations, medians/p90, clocks, exact environment, launch/stream method and hashes. Diagnose wrapper launch gaps before claims. Never combine profiler-replay and event timing.

**Gate 3 — evidence:** Select at most three representative cases for supported-metric profiling. Inspect actual kernel/geometry, registers/shared memory/occupancy, tensor/scalar activity, eligibility/stalls, L2/DRAM and extra work. Separate observations from hypotheses and record what explanations are excluded.

**Gate 4 — decision criterion:** Recommend exactly one next experiment only with an affected workload, suspected mechanism, concrete proposed change, measurable criterion, tradeoffs/regressions, held-out cases, and prior-work classification. Otherwise recommend one bounded ablation or document the integration no-go. Do not search or patch indefinitely.

**Initial gate record (before execution):** Validation and measurement had not started at declaration. Final results and failures are recorded below. New records live under `results/e004_20261106/`. ARC job 328410 is RUNNING, owned by hpuvvad on c58, verified at runtime 01:16:34 of 06:00:00. This does not authorize reuse without checking remaining time again as needed. Current ARC documentation still specifies prun inside allocations.

**Revisit only if:** A documented blocker changes, or a separately requested experiment tests the final evidence-based recommendation.

**Pre-measurement integration decisions:** Selected released vLLM v0.33.0, peeled source `4ba787ec9959e566d3b587a796dff07673324da5`, with official wheel SHA 256 `4b6bdec4ab269990bf70d45a3c1e94c3760a27064513760193b26d6f0da883aa` and embedded `g4ba787ec9`. Initial moving main was inspected and recorded but not benchmarked. Release changes relative to main are explicit, not silently mixed. Source subset/license archive SHA 256 `79e67e69caff6099a6346a90406b7168dee18704c1441e28b21e4b948eb7b33f`. Python 3.11 isolated at `/tmp/e004-hpuvvad-20261106`, shared project `/home/hpuvvad/csc548-w4a16-e004-20261106`. No original dependencies changed. Initial Python 3.6 default was unsuitable, but installed Python 3.11 and glibc 2.28 satisfy the selected wheel tags. Torch 2.14.0 installed successfully. The retained-project CUDA build and prerequisite had not yet run at this point. Their completion is recorded below.

**Pre-measurement review corrections (not performance failures):** Static review found that PyTorch's implicit graph capture stream differs from an outer stream hardcoded in ctypes closures. Explicit capture on `torch.cuda.current_stream()` now makes all launches/events/flushes use the same stream. Added NaN output poisoning before each candidate check and graph replay to reject stale results or empty captures. Removed dequantization from public timed selector choices because it produces W_hat rather than C, while retaining internal bit-check materialization. Hash only source files, excluding generated Python cache directories. These issues were corrected before adapter measurement. The graph capture/setup cost is now recorded separately. Small grouped smoke uses K=256, N=128 to avoid the different per-column path inferred from a single scale row at K=128.

**Prerequisite attempt 1:** The initial target Release project and bridge builds succeeded, but the bridge's post-build source hash glob encountered `scripts/e004/__pycache__` and exited 1. The file-only correction arrived after that shell had started. No timing followed. `bridge-build-attempt1.log` and `bridge-source-attempt1.sha256` preserve the failed gate. `resume_prerequisite.sh` records preservation and rehashing of the unchanged binary, then runs the required project correctness/sanitizer gate. This is a concrete failed prerequisite, not a failed Marlin kernel integration or performance result.

**Prerequisite attempt 2:** The retained project correctness gate completed with six zero-error sanitizer runs, and checksum-pinned vLLM 0.33.0 installed successfully. ARC `prun` flattened the quoted `python -c` argument, producing `SyntaxError: invalid syntax` on the token `import`. The prerequisite failed immediately. The correction uses `scripts/e004/check_runtime.py` as a real file, avoiding compound interpreter arguments. Preserve `torch-smoke-attempt1.log`. This launcher quoting failure occurred before any Marlin measurement, and does not indicate a CUDA or numerical failure.

**Gate 1 passed:** Target runtime loads official vLLM 0.33.0 with Torch 2.14.0+cu130, CUDA 13.0 package runtime and native Torch sm_120 support on RTX 5060 Ti. Project build/CPU tests/small correctness and six Compute Sanitizer runs pass. Small K=256, N=128, M={1,5,32} and main K=N=4096, M={1,32} pass lossless inverse packing/scale-bit checks, all-element FP16 reconstruction checks, full E001/E002/E003 control comparisons, strict all-row FP64 checks for all eight timed comparators, and poisoned-output graph replay checks under both policies. Torch GPU traces show one actual dense `marlin::Marlin<...>` kernel for each direct operator call, not a fallback. Raw smoke logs/jsonl/traces preserve selection and numerical errors. Smoke timing uses only 2 warmups/3 events and is not acceptance performance evidence. First repack calls may include lazy initialization and host gaps in their setup event interval, so wall/event setup costs are labeled, not claimed as isolated GPU repack throughput. The next complete screen uses 10 warmups/20 events; targeted repeats use 10/100. The same bridge, operands, graph events, node and cache policies apply to every comparator in each stage. Per-selector first validation-call wall time and capture costs are saved separately. The isolated wheel cuBLAS 13.1.1.3 is loaded in-process, as shown in process maps, so new cuBLAS comparisons must not use historical system-library medians.

**Screen attempt 1 stopped on correctness:** The first screen produced 122 passing rows then original-weight cuBLAS at M=128, K=N=4096 failed the unchanged all-row threshold on 2/524288 outputs. Failed row/column pairs are (5,789) and (91,677). Relative L2 is 2.950522e-4, and maximum absolute error is 2.415464e-3. No failing timing row was emitted and the process stopped. Preserve the partial screen as `screen_attempt1*`; do not treat it as a complete performance map or combine it with rerun points. Historical M=128 checks sampled 64 rows, excluding both failed rows, and used a different cuBLAS library. They were not retroactively invalidated by claiming identical coverage/environment.

**Bounded numerical diagnosis:** `diagnose_cublas.py` compares default math mode 0 and `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` mode 16 with identical operands, FP32 compute type, and original/reconstructed-weight FP64 references, two repetitions each. Default original-weight failures reproduce exactly. Mode 16 has zero bad outputs for both paths, relative L2 2.070527e-4 original and 2.076626e-4 reconstructed. The flag forces reductions to use compute precision, according to NVIDIA's cuBLAS 13.0 documentation. It may also change internal algorithm selection, so the ablation does not isolate every internal cause. E004 will explicitly set mode 16 on its own handle, label every row, and rerun the entire screen. Existing project cuBLAS source/behavior and old results remain unchanged. This is a reference-compatibility configuration correction, not a new custom kernel optimization or relaxed tolerance.

**Diagnostic implementation failure:** The first new reduction diagnostic mistakenly passed the host-order GPU pointer vector A, W, Q, S, W_hat to the bridge expecting A, Q, S, W, W_hat. cuBLAS accessed invalid memory and that process failed with CUDA error 700. Its script and log are preserved as `diagnose_cublas_attempt1.py.txt` and `cublas_reduction_diagnostic-attempt1.log`. The corrected explicit indices were independently reviewed, then the successful ablation ran in a new process. This failure is not evidence about cuBLAS arithmetic or Marlin.

**Gate 2 complete screen:** All 384 rows (3 shapes*8 batch sizes*8 implementations*2 policies) pass strict FP64 validation with every row checked, code/scale/W_hat round trips exact, and retained full-output controls equal. No requested shape needs padding or is unsupported. This is a short screen, 10 warmups/20 events, with explicit cuBLASmode 16 throughout. Preliminary patterns warrant independent repeats: Marlin no-flush is flat at M=1..8 then rises at16, and at square M=64 it is close to/reported slightly slower than reconstructed cuBLAS. Select M={1,8,16,64}, K=N=4096, all 8 comparators/bothpolicies for two independent 10/100 repeats. This is a diagnostic choice, not an optimization acceptance criterion. Separately compare eager and graph events at M={1,32}, square, Marlin/E002/reconstructed-cuBLAS, identical 10/100 conditions, to quantify wrapper-gap sensitivity. Do not splice these runs into the screen. Select at most 3 final profiling cases after inspecting repeat stability.

**Repeat and timing-method findings:** Both independent 64-row 10/100 series pass all numerical and retained-control checks. Square no-flush Marlin M=8 is13.888 us in both, M=16 is19.584/19.456 us, and M=64 is58.304 us in both. Same-run reconstructed cuBLAS M=64 is54.768/54.752 us, so its modest lead reproduces. M=16 store-flush varies 30.848→32.384 us, and screen/repeat differences must not be hidden. Matching 12-row eager/graph diagnostics show M=1 no-flush Marlin 18.560 us eager versus 13.536 us graph (p90 20.608vs13.856). M=32 and longer kernels show much smaller/mixed differences. This supports common graph events for kernel comparison rather than treating Python dispatch gaps as GPU work. The runs are independent, clocks unlocked, and the difference is not an isolated pure wrapper-cost subtraction.

**Gate 3 selection:** Exactly three square Marlin cases: M=8 (eight-row specialization), M=16 (boundary and resource comparison), M=64 (reproduced cuBLAS crossover). Each profiles one actual kernel after 10 untimed direct launches, explicit profiler start/stop, `--cache-control none --clock-control none` with kernel replay, no application flushing. Query supported target metrics and record the kernel, geometry, resources, tensor/scalar work, eligibility/stalls, L2/DRAM. These are reused-input profiler diagnostics, not the application's graph-event timings and not guaranteed identical cache state on every replay. No extra shape/configuration search or external kernel patch is authorized. A potential two-CTA resource ablation remains only a hypothesis until counters are assessed.


**Final versions and environment:** E004 predeclared gates `670b32d`, initial adapter/provenance `616c969`, passed integration gate `cffd4ca`, explicit cuBLAS reduction setting and measured benchmark source `209442c`, complete timings/repeats `640e9e5`, profiler/provenance evidence `528c430`. Final documentation/analysis commits may follow. The unchanged bridge binary SHA 256 is `85f332951a8c24b3eb06594977b3e3d62b7954fa01b986da41ffeff851bfbd3f`. Loaded official vLLM operator library SHA 256 is `e634b80e72078e964f8bd2ac541d66f2a900920773cf993544609612190bd9a9`, independently matched to the checksum-verified wheel. Exact source is v0.33.0 / `4ba787ec9959e566d3b587a796dff07673324da5`, Apache-2.0, with upstream Marlin copyright notices preserved in the source archive. The wheel requires Torch 2.14.0. It was installed with `--no-deps`, without serving/model packages, into an isolated Python 3.11 environment. `pip-freeze.txt` records all installed runtime packages, including NumPy 2.4.6, Triton 3.8.0 (a Torch dependency, not the Marlin comparator), cuBLAS 13.1.1.3 and CUDA 13.0 runtime. No GemLite/Triton kernel was substituted.

Actual execution is on c58, RTX 5060 Ti 16 GB, sm_120,36 SMs,32 MiB L2, driver 615.71.09, EPYC 9115,16 CPU cores, one prun rank, OpenMP 16 threads, no MPI binding. CUDA toolkit 13.4.92 and GCC 12.4.0 build the project/bridge. Bridge flags are `-O3 -DNDEBUG -std=c++17 -lineinfo -arch=sm_120 -shared -Xcompiler=-fPIC,-fopenmp`, linking cuBLAS, with no native CPU ISA option. Exact project CMake/compiler/link flags are in `build-settings.txt`, with tool and environment details in `metadata.txt`/`project-build.log`. Torch's build configuration and the actual loaded library paths are recorded. The release wheel contains sm_120 device objects, and actual dense Marlin execution is established by traces and profiles. Exact upstream wheel compiler invocation is not published in these artifacts and was not reconstructed. Artifact hashes and source pin, not an inferred build command, define that binary. Clocks were observed, not locked: screen starts at180/405 MHz SM/memory and ends 2745/14001 MHz. Repeats finish near 2745/14001 MHz. These observations are not per-launch clock measurements.

**Compact performance map:** Each cell below is store-flush / no-flush median microseconds from the complete 10/20 screening run. These are diagnostic measurements, not all independently repeated points. cuBLAS original uses original W and its own FP64 reference. Other GEMMs use identical reconstructed W_hat. Every cuBLAS entry uses the explicitly labeled FP32-reduction math setting. Best retained means the minimum among naive/E002/E003 two/four rows separately for that shape and policy. `performance_map.csv` also identifies that selector, while `screen.jsonl` preserves every individual comparator, p90 and raw sample. Unfused context is in `screen.jsonl`, not the compact CSV. Ratios compare same-run medians only.

| K×N | M | Marlin | cuBLAS original | cuBLAS W_hat | Best retained fused |
|---|---:|---:|---:|---:|---:|
| 4096×4096 | 1 | 29.632 / 13.872 | 94.208 / 85.040 | 94.176 / 83.520 | 51.200 / 29.760 |
| 4096×4096 | 2 | 30.624 / 13.888 | 86.016 / 52.800 | 86.016 / 52.800 | 53.232 / 36.384 |
| 4096×4096 | 4 | 30.688 / 13.888 | 86.032 / 52.800 | 87.760 / 53.568 | 65.536 / 58.944 |
| 4096×4096 | 8 | 30.720 / 13.888 | 86.016 / 52.800 | 86.016 / 52.800 | 114.688 / 110.144 |
| 4096×4096 | 16 | 30.784 / 19.616 | 87.296 / 54.320 | 88.032 / 54.816 | 204.800 / 200.416 |
| 4096×4096 | 32 | 36.928 / 34.368 | 88.064 / 54.400 | 88.064 / 53.536 | 391.168 / 386.624 |
| 4096×4096 | 64 | 61.440 / 56.896 | 90.112 / 54.848 | 90.112 / 54.848 | 755.744 / 751.168 |
| 4096×4096 | 128 | 106.464 / 98.416 | 118.784 / 106.176 | 117.600 / 106.048 | 1501.232 / 1495.616 |
| 4096×11008 | 1 | 67.568 / 28.224 | 249.840 / 238.816 | 244.432 / 234.592 | 94.224 / 63.040 |
| 4096×11008 | 2 | 67.552 / 28.224 | 307.200 / 265.696 | 305.120 / 265.552 | 102.368 / 81.472 |
| 4096×11008 | 4 | 67.584 / 28.224 | 312.672 / 271.472 | 309.248 / 271.728 | 147.488 / 136.288 |
| 4096×11008 | 8 | 67.584 / 28.224 | 258.048 / 247.344 | 254.512 / 241.680 | 276.480 / 267.952 |
| 4096×11008 | 16 | 71.648 / 43.392 | 262.144 / 251.424 | 258.048 / 245.312 | 522.240 / 516.672 |
| 4096×11008 | 32 | 83.936 / 75.328 | 258.048 / 247.072 | 251.904 / 241.056 | 1023.920 / 1013.152 |
| 4096×11008 | 64 | 139.216 / 131.392 | 262.224 / 247.360 | 254.032 / 242.624 | 2012.512 / 2001.472 |
| 4096×11008 | 128 | 258.048 / 255.200 | 290.784 / 263.328 | 290.816 / 264.288 | 4022.304 / 4013.632 |
| 11008×4096 | 1 | 67.648 / 26.176 | 239.616 / 218.688 | 238.448 / 218.688 | 104.480 / 76.096 |
| 11008×4096 | 2 | 67.616 / 26.176 | 223.232 / 215.840 | 223.232 / 216.368 | 116.736 / 95.312 |
| 11008×4096 | 4 | 67.584 / 26.176 | 223.584 / 216.640 | 223.232 / 216.640 | 170.016 / 152.816 |
| 11008×4096 | 8 | 69.600 / 26.176 | 225.248 / 216.976 | 224.256 / 216.640 | 301.056 / 296.224 |
| 11008×4096 | 16 | 71.680 / 40.512 | 226.784 / 218.096 | 225.280 / 216.640 | 544.768 / 541.136 |
| 11008×4096 | 32 | 83.952 / 75.328 | 227.424 / 220.688 | 229.296 / 220.736 | 1038.128 / 1028.816 |
| 11008×4096 | 64 | 137.216 / 132.608 | 231.424 / 224.832 | 231.424 / 224.832 | 2018.304 / 2010.384 |
| 11008×4096 | 128 | 253.952 / 247.360 | 335.872 / 296.128 | 333.824 / 298.560 | 4016.144 / 4009.216 |

Marlin beats every retained scalar fused variant in this screen: 1.395–15.815× under store flushing and 2.145–16.208× without flushing. This establishes a substantially stronger comparator, not a project contribution over maintained Marlin. Reconstructed cuBLAS/Marlin ratios range 1.105–4.576 under store flushing and 0.964–9.628 without it. The small no-flush square M=64 reversal is reproducible: repeat 1/2 Marlin 58.304/58.304 us versus cuBLAS W_hat 54.768/54.752 us, about 6.5% more latency for Marlin. At that same point under store flushing, Marlin 61.440 us beats cuBLAS 90.112 us. Cache policy changes the relative winner. The square M=8→M=16 no-flush increase is13.888→19.584 us and 13.888→19.456 us across repeats. Source selection changes the eight-row specialization at this boundary and doubles Tensor Core instruction count. That is not evidence of a correctness bug or an avoidable discontinuity by itself.

**Numerical conclusion:** All 384 screen rows, both 64-row repeated series and both 12-row timing diagnostics pass all-output FP64 checks, including all 128 rows at M=128. Maximum screen relative L2 is2.1282073e-4 and maximum absolute error 1.9701799e-3. Every shape's code/scale inverse conversion and all reconstructed FP16 elements match. E001/E002 and E003/E002 full-output obligations also pass. Implementation error is measured against each kernel's proper FP64 reference. Error against original weights is not substituted for reconstructed-weight implementation error. Quantization error remains a separate operand difference. Upstream's looser aggregate test tolerance was never adopted. The successful external validation is numerical and layout evidence, not an external-library sanitizer proof. Six project sanitizer runs passed separately.

**Targeted counter evidence:** Three reports only, square M=8/16/64, each one kernel after 10 direct warmups,13 kernel-replay passes, cache-control none and unlocked clocks. Normalized traffic below uses decimal MB. Source static shared is0, requested dynamic shared 101376 B, and allocated shared 102400 B including 1024driver bytes in every case. Every launch has 36 blocks of256 threads. These counters describe the recorded profiler setup, not graph-event steady state or physical input eviction. The replay DRAM reads remain about 2.4–2.5 MB despite warmups, so do not claim completely resident warm operands.

| M | Registers/thread (allocated) | Shared/register block limits per SM | Active warps % | Eligible warps/scheduler/cycle | Tensor instructions | Tensor-active % of sustained elapsed peak | L2 MB | DRAM read MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 122 (128) | 1 / 2 | 17.257 | 0.429 | 65536 | 44.613 | 11.239 | 2.478 |
| 16 | 128 (128) | 1 / 2 | 16.851 | 0.282 | 131072 | 61.819 | 13.695 | 2.413 |
| 64 | 255 (256) | 1 / 1 | 16.733 | 0.165 | 524288 | 77.793 | 22.176 | 2.531 |

Executed warp instructions are 1.438818/1.538644/2.600373million, while predicated integer thread instructions are 15.948036/15.631492/31.262512million. These are different counter scopes and must not be added or treated as FLOPs. Scalar FP32 FFMA count is0. Tensor instructions scale exactly with useful M=8/16/64, providing no evidence of extra M padding at those three points. Issue-active percentages are 35.619/24.614/13.451. Long/short-scoreboard stall metrics are 0.423/0.583,0.370/0.465 and 0.324/0.202%, respectively. Wait metrics are 3.141/3.450/2.363%, barrier 2.096/1.732/1.414%, and math-pipe throttle 3.114/5.554/8.691%. These use the exact metric normalizations saved in `profile_summary.csv` and do not form a percentage partition of runtime. No single dominant-stall cause was isolated. M=64's high tensor activity and register limit rule out treating it as the same shared-memory-only opportunity. Tensor-active percentage is not percentage of peak FLOP/s. No roofline classification is made.

**Observed resource opportunity and design alternatives:** The current dispatcher always returns one block per SM from its first valid ordered configuration and launches with the opt-in shared-memory maximum. For M<=8, its selected 128K×128N,256-thread, four-stage tile uses 8192 B A pipeline,32768 B overlaid B/reduction/bias and 1024 B scales:41984 B total, derived from template offsets. The host estimator assumes 16A rows and returns 50176 B even for the eight-row specialization. A two-block budget from the existing branch is101376/2−1024=49664 B,512 B below that conservative estimate but above the actual eight-row layout. Registers allocate 32768per block at M=8, fitting two within the observed 65536per SM. Requested shared memory currently limits residency to one block, and grid 36 supplies only one block per SM anyway. Merely lowering shared memory with grid 36 cannot test additional residency. A two-block launch must change grid and shared budget together, and changes striped K partitioning/global reduction work. This is a code-derived feasible candidate plus low eligible-warp evidence, not a measured bottleneck or promised latency gain.

Do not apply it universally: M=16 uses 50176 B of layout, exceeding 49664 B, and M=64 is register-limited to one block. Tensor Core implementation from scratch, wider tile sweeps, another framework, automatic cuBLAS/Marlin dispatch, and cache-policy redesign were considered outside this bounded E004 objective and were not implemented or measured failures. The retained scalar project controls remain intact. The integrated external implementation is the comparator, not a new C++/CUDA optimization contribution.

**Gate 4 decision — exactly one next experiment, not implemented:** Recommend E005 as a bounded **one-versus-two-CTA launch-resource ablation for M<=8**. This is a go for a diagnostic ablation, not an acceptance of an optimization. Keep an untouched pinned Marlin control. In a separately versioned candidate host dispatcher, account for the actual eight-row shared layout and test 72 CTAs with 49664 B requested dynamic shared against 36 CTAs/101376 B, preserving the 128×128 K/N tile, 256 threads, four stages, Tensor Core body, integer codes/scales/group 128, and FP32 reduction. Before launching, prove lock and C_tmp index bounds for 72 CTAs, including the source's mapping and allocation formulas. Capacity arithmetic alone is not that proof. No candidate was compiled or launched in E004.

1. Affected workload: M=1 and M=8, K=N=4096, group 128 on this 36 SM sm_120 device, under both existing policies.
2. Suspected mechanism: extra residency could improve latency hiding where requested shared memory currently admits one block and eligible warps average 0.429 at M=8. Occupancy is not itself the performance objective.
3. Proposed change: the host-only paired grid/shared-budget candidate specified above, with precise eight-row resource accounting, an untouched control and a proved lock/C_tmp mapping. Keep the Tensor Core kernel body unchanged.
4. Success criterion: in two independent 10 warmup/100 sample same-binary control/candidate series, at least 5% lower no-flush median at both M=1 and M=8, no more than 5% store-flush regression at either, all strict FP64 checks and selected memory/synchronization checks passing, and coherent resource/traffic evidence. Report p90 and variance rather than counting a timing-resolution tie as a win.
5. Expected tradeoffs/regressions: extra K slices, locks/global FP32 reductions and L2 traffic may erase the gain. Larger M may not fit or may lose register/tensor efficiency. Retain unchanged M=16/M=64 fallback and check both for <=5% regression.
6. Held-out validation: M=2/M=4 at square and both rectangular K/N pairs, seed 2, both policies. Hold these out of candidate tuning before reporting generalization. Their baseline screen is already known, while seed 2 is new.
7. Prior work: this is standard Marlin launch/resource tuning and reproduction of its existing blocks-per-SM mechanism, not an algorithmic novelty claim. The dispatcher already contains grid/shared-budget logic and ordered tiling. Upstream Marlin's pipelining and striped reduction design remain intact.

**Setup costs and limits:** Screening host generation/quantization/reconstruction takes 1.094 s for square and 2.831/2.940 s for the rectangular sets. First repack-plus-scale-permutation wall times are 29.748 ms,0.438 ms and 0.366 ms in shape order. The first is first-use setup and may include lazy initialization and host gaps, and corresponding setup event intervals 29.659 ms/381.280 us/310.592 us are not isolated repack-kernel throughput. First validation-call wall time and each graph capture cost are separately saved per row. No tuning occurs. Prebuilt loading, transfers and CPU references are excluded from GEMM event timing. Python Torch/ctypes integration changes host overhead and graph scheduling relative to historical C++ timings, so only fresh E004 denominators are used. One GPU/node/seed,20-sample screen except selected 100-sample repeats, unlocked clocks, no full serving run, no RTX PRO 6000 trial, no external-kernel sanitizer run, and no novel optimization implemented. The store flush still does not prove eviction. Physical traffic comes only from the recorded counter experiment, not byte-model bandwidth.

**Reproduction and preserved evidence:** Raw artifacts and analysis are in `results/e004_20261106/`. `python3 scripts/e004/analyze.py results/e004_20261106` checks 384-row completeness, raw timing aggregation, strict full-output checks, repeated operand provenance, all three profiles and units, six sanitizer summaries, source archive hashes and the same bridge binary across screen/repeats. It regenerates `analysis.json`, `performance_map.csv`, `repeat_comparison.csv`, and `profile_summary.csv`. Existing source-hash assertions require the recorded source, not edited numerical code. Exact launched arguments appear in raw prun logs, with stage commands in `prepare_arc.sh`, `resume_prerequisite.sh`, `run_arc.sh`, `repeat_arc.sh`, and `profile_arc.sh`. The original failed prerequisite script is recoverable from `616c969`, and `diagnose_cublas_attempt1.py.txt` preserves the uncommitted failed diagnostic. Final scripts fix those errors. Do not rerun preservation/resume scripts against existing raw records.

For a new run, first verify ARC policy, allocation ownership/state/time/node/GPU, then use a distinct output directory on the compute node. Export the absolute shared PROJECT_ROOT and actual SLURM_JOB_ID/NODELIST. Recreate the isolated Python 3.11 environment from the frozen packages with `pip install --no-deps -r .../pip-freeze.txt`, which intentionally avoids resolving vLLM's serving dependencies. Use the checksum-pinned wheel URL in `prepare_arc.sh` for byte-exact vLLM provenance. The following assumes the validated one-rank prun environment recorded in that script and a runtime compatible with the actual sm_120 target. Set E004_PYTHON to the isolated interpreter, never the default Python 3.6.

```bash
cd "$PROJECT_ROOT"
export RUN_DIR="$PROJECT_ROOT/results/e004_replay_UNIQUE"
mkdir "$RUN_DIR"                   # refuse reuse; preserve the old directory
mkdir -p external/vllm
tar -xzf results/e004_20261106/marlin-source.tar.gz -C external/vllm
prun bash scripts/build.sh
prun bash scripts/e004/build_bridge.sh
prun bash scripts/check_inblock.sh  # inspect correctness/sanitizer output first
# LIB is the exact isolated vllm/_C_stable_libtorch.abi3.so path, printed by check_runtime.py.
prun "$E004_PYTHON" scripts/e004/check_runtime.py
prun "$E004_PYTHON" scripts/e004/bench.py --source "$PROJECT_ROOT/external/vllm"   --library "$LIB" --bridge "$RUN_DIR/libe004.so" --out "$RUN_DIR/smoke.jsonl"   --shapes 256,128 --ms 1 5 32 --warmup 2 --iters 3 --trace
# Inspect, then repeat K=N4096,M1/32 smoke before the complete screen.
prun "$E004_PYTHON" scripts/e004/bench.py --source "$PROJECT_ROOT/external/vllm"   --library "$LIB" --bridge "$RUN_DIR/libe004.so" --out "$RUN_DIR/screen.jsonl" --iters 20
```

Use distinct output filenames for the repeated and eager/graph diagnostics, with the exact arguments in `repeat_arc.sh`. The profile helper records exactly three selected cases and refuses an existing ncu directory. Its historical fixed directory is part of the recorded command, so adapt its output root explicitly for a new run rather than overwriting E004. Future unattended jobs must use sbatch with a successful prerequisite dependency and an exported absolute project path. This session reused the existing interactive allocation through prun, submitted no new allocation, and leaves the user's job and all prior project directories intact. E004 is complete. Await authorization before implementing E005.


**Final metadata and prior-work check:** `runtime-affinity.log` verifies exactly one prun rank, Torch CPU thread setting 16 and worker masks 0 through 15 under OMP_PROC_BIND=true. The main thread is bound to CPU 0 after OpenMP initialization. `build-settings.txt` captures actual CMake/cache/compiler/link settings and ELF compiler comments. The user allocation remains RUNNING at final check, runtime 01:48:24 of 06:00:00, scheduled end 20:00:41 Eastern. A late metadata SSH command returned 255 without diagnostic output. Subsequent successful retrieval confirmed complete runtime/allocation records, and a separate affinity check completed successfully. No performance series was resumed from that inspection failure. Main-source inspection at the recorded initial 4d82b569 commit confirms the same one-block resource policy/16-row estimate remains there, despite other Marlin changes. Excerpts and source hash are preserved. That is a bounded upstream check, not an exhaustive novelty search.


## E005 — Specialization-aware reservation and one-versus-two-block residency

Status: complete, no-go at Gate 3. Performance hypothesis remains inconclusive. Authorized 2026-11-06. E004 checkpoint verified: `cf8a419`, clean tree.

Problem and hypothesis: E004's small-M dense uint4b8/G128 eight-row 128x128 tile appears to reserve substantially more shared memory than its layout needs. Reducing reservation may help independently, and two blocks per SM may improve latency enough to offset changed stripes and FP32 global reductions. These remain hypotheses.

Predeclared configurations: A retains the pinned dispatch and full opt-in allocation. B changes only the verified small-M reservation to the device-derived two-block budget, retaining SM-count grid. C uses the same reservation as B and twice the SM-count grid. On the previously measured device these should be 36/101376 B, 36/49664 B, and 72/49664 B. Query, do not assume, these limits. One compiled device kernel should serve all three. Unsupported selections, M=16 and M=64 keep original dispatch.

Predeclared acceptance: all correctness and selected external memcheck/racecheck/synccheck pass. In each of two independent balanced series, C no-flush median must be at least 5% below same-build A at both M=1 and M=8, square 4096, seed 1. C store-flush regression must be <=5% at both. M=16/M=64 fallback must show no reproducible >5% regression. Profile launch/resources must agree with intended change. Each point has 10 warmups and 100 samples. Report B independently even if C fails. Criteria will not be changed after measurement.

Gates and scope: prove complete shared/workspace/index/reset/progress properties and query compiled resources before modified launches. Preserve untouched source and explicit host-only patch. Build local A/B/C with identical flags/dependencies and common device code. Wheel comparison is provenance only. Preserve E004 operands, lossless packing, FP16 reconstruction and all-row FP64 tolerance. Validate small shapes M=1,2,4,5,8, main square and rectangles, zero/mixed-sign operands, repeated reused workspace and poisoned graph replay. Bound possible hangs. Keep required per-call maintenance within measured operation. Hold out seed 2, M=2/4, all three K/N pairs, both policies without retuning. Profile A/B/C at square M=8, add cases only to answer a specific unresolved question. No tile/body/stage/format changes, automatic performance dispatch or next optimization.

Baseline and candidate, preimplementation record: matched build was pending at criteria commit `bc94827`. The completed build and source hashes are recorded below. E004 wheel/source remains pinned at vLLM v0.33.0 / `4ba787ec9959e566d3b587a796dff07673324da5`. New scripts/evidence use `scripts/e005/` and `results/e005_20261106/`. Source safety analysis, commands, build/runtime provenance, raw results and final decision follow as gates complete.

Revisit only if: the final evidence identifies a changed prerequisite or a specific unresolved condition. Failure does not authorize a tuning search.


E005 Gate 1 source derivation and executable finite index model are preserved in `results/e005_20261106/GATE1.md` and `safety-model.json`. The complete enabled shared layout is 41984 B. Some reduction scratch accesses exceed the nominal `sh_red_size`, but stay within the B overlay. Explicit locks/C_tmp enumeration covers both grids and the rectangular branch. The common caller uses 4*SM locks and the original scratch capacity, with guards. Dependency edges point to larger CTA IDs. Query-based full-grid capacity and acyclic waits support bounded progress, without claiming a guaranteed CUDA scheduling order or achieved residency.

Gate 2 local build: original source archive plus explicit host-only patch, 15 generated FP16/uint4b8/G128 specializations, identical compiler flags and one library/kernel entrypoint for A/B/C. Unmodified dispatcher priorities/checks and Tensor Core body remain. The first compile failed because Torch CUDA declarations require `USE_CUDA`. Added that same declaration for the entire build, preserving failed log/script. Local dry attributes report 126 registers/thread, 32 local bytes, zero static shared for small M. Wheel E004 reported 122 registers. This is a provenance difference, not a candidate advantage. Queries report A (36,101376 B,1 maximum block/SM), B (36,49664 B,2), C (72,49664 B,2). M=16/64 A/B/C plans and entrypoints match. No modified kernel launched before the source/attribute gates.

Operational prerequisites: the first hostname assertion incorrectly included prun banners. The second assumed rg existed on ARC. Both stopped before compilation. Preserved scripts and launcher logs, then used a count of exact hostname lines with available grep. The runtime/one-rank prerequisite passed. No expensive timing series has begun.


### E005 outcome and design decision

**Stop at the predeclared sanitizer gate.** The external local Marlin body passes memcheck and synccheck but fails racecheck. The unchanged local A and isolated official E004 wheel also reproduce a racecheck report at M=8,K=N=4096. This is inherited reported behavior, not evidence that the smaller reservation or larger grid introduced the hazard. It still violates the predeclared requirement that all selected checks pass. No performance series, held-out seed-2 evaluation, or Nsight counter profile was released. No speedup/regression or achieved two-block residency is claimed. E005 is a completed bounded no-go, with its performance hypothesis inconclusive.

The selected design uses runtime A/B/C in one local library, keeping every device header/template byte-identical to the pin. A follows the original dispatcher/accounting and grid. B changes the final eligible eight-row accounting/reservation only. C adds twice the SM-count grid. Runtime eligibility requires sm_120, dense FP16 A/C/scales, uint4b8/G128, no optional paths, FP32 reduction, 256 threads, tile 128x128, four stages and M<=8. Device limits derive the budget. CUDA queries require two-block capacity before a candidate launch. Smaller N=128 selects the original 128-thread tile 64 fallback. A K256,N2048 smoke was therefore added to exercise the intended candidates without changing dispatch. M=16 and M=64 preserve original plans, entrypoints and checked output bits. No automatic timing-based dispatch is introduced.

### E005 spatial bounds, lifetime finding and compiled resources

The full derivation and executable finite model are `results/e005_20261106/GATE1.md`, `safety-model.json`, and `scripts/e005/safety.py`. Source offsets refer to the untouched archive. Layout is B/reduction overlay [0,32768), grouped scales [32768,33792), and four A stages [33792,41984), in bytes. Four stages use 8192 B A,32768 B packed B,1024 B scales. No zero-point, activation-scale, bias or global-scale path is enabled. The nominal 4352 B reduction-size expression is not the maximum block-reduction access: enabled indices reach int4 index 959, within the 32768 B overlay. Every base/stage is 16-byte aligned. The original host assumes 16 A rows and estimates 50176 B. The smaller 41984 B footprint is real as a spatial requirement. It does not by itself prove a safe overlay phase transition.

The common caller uses 4*SM int32 locks with guards. The original small-M scratch capacity 36*16*256 floats=589824 B already covers 72 slots of 8192 B. Explicit enabled accesses are lock*512 plus int4 windows[0,64),[128,192),[256,320),[384,448). The most demanding requested rectangle reaches lock 70 and exclusive scratch end 580608 B under C. The source model verifies actual slice indices, exact K coverage, lock bounds and acyclic wait edges for both grids. It also shows why C changes work/reduction partitioning: K4096,N11008 moves from 72 whole part1 tiles plus 14 part2 tiles in A to 86 part2 tiles and 68 split columns in C. This is not a pure occupancy ablation.

Locks are initialized once and reset by last reducers. First partial owners initialize scratch fragments before successors read them. Ordered same-stream calls/graphs reuse this storage without per-call clearing. Repeated checks assert every lock returns to zero, guards remain intact, and poisoned output is rewritten. No maintenance was omitted from a measured operation because no operation was performance-measured. Original C_tmp sizing is used for the maximum M in each validation invocation. M<=8-only invocations use the upstream 589824 B capacity. The combined main validation invocation allocates the original M64 capacity 2359296 B. This is validation storage, not a timing choice.

Actual dry CUDA attributes and occupancy queries are preserved in `attributes.log` and `resources.txt`. These are predicted maxima and launch plans, not achieved concurrent residency. A/B/C share one device entrypoint in-process. Source and SASS are common across these configurations.

| Configuration, square M<=8 | Grid | Threads | Requested dynamic B | Registers/thread | Static B | Local bytes/thread | Maximum blocks/SM |
|---|---:|---:|---:|---:|---:|---:|---:|
| A, same-build control |36|256|101376|126|0|32|1|
| B, reservation only |36|256|49664|126|0|32|2|
| C, reservation and grid |72|256|49664|126|0|32|2|

The local build's 126 registers differ from the E004 wheel's 122. This is why the primary denominator was specified as local A. No wheel-vs-local latency diagnostic was performed after the gate failure. A separate process confirms actual wheel `marlin::Marlin<...,256,1,8,8,true,4,8,false>` execution, whereas local traces select `e005_marlin::Marlin` with the same template parameters. The initial mixed-library profiler session assigned the local name to its wheel event. Its cause is unresolved, its loose substring assertion was tightened, and it is preserved as unreliable naming evidence. Subsequent wheel diagnostics never load libe005.so.

### E005 correctness and sanitizer evidence

The numerical contract and tolerance are unchanged. All reconstructed codes/scales/FP16 weights invert losslessly. Every checked output row/column passes the FP64 condition, including main M64. Main maximum relative L2 is 0.00020994083360038762 and maximum absolute error 0.0019259467462688917. Distinct JSONL rows count both policy labels even for untimed correctness. They are not timing samples. Each main/small case runs five direct launches and five poisoned graph replays with reused workspace.

| Evidence | Coverage | Result |
|---|---|---|
| `validation_small.jsonl` |180 rows: M1,2,4,5,8; K256,N128/2048; mixed-sign,zero-A,zero-W; A/B/C; both labels |All numerical/roundtrip/guard/lock checks pass |
| `validation_main.jsonl` |126 rows: M1,2,4,5,8,16,64; square and both E004 rectangles; A/B/C; both labels |All numerical/roundtrip/guard/lock checks pass. Fallback bits agree |
| `project-check.log` |Retained project CPU/GPU tails plus six memcheck/synccheck runs |Pass. These are separate from external checks |
| `sanitizer_memcheck.log` |External local A/B/C; M1,5,8; K256,N2048; two direct/two graph calls |0 errors |
| `sanitizer_synccheck.log` |Same external coverage |0 errors |
| `sanitizer_memcheck_rect.log` |External A/B/C; M5,K4096,N11008; two direct/two graph calls |0 errors, including largest C scratch/lock range |
| `sanitizer_racecheck.log` |External A/B/C; M1,5,8; K256,N2048; two direct/two graph calls |36 errors,0 warnings. Exit 99 stops gate |
| `race_diagnostic_A.log` |Unchanged local A; M8,K=N4096; one direct call |2 errors,exit 99 |
| `race_diagnostic_wheel.log` |Official wheel in isolation; same main shape |2 errors,exit 99 |

External tools filter local `e005_marlin` bodies. They do not sanitize repacking/fill/runtime kernels. The bounded control diagnosis uses `kns=marlin`, which also includes repack naming, and the reported error names explicitly identify the dense GEMM specialization. Racecheck error summaries count reported hazards, not distinct failing output elements or corrupted launches. Do not add them to numerical error counts. All commands use bounded timeouts. No hang occurred. The selected tools do not prove inter-block progress.

Local racecheck identifies a B-fragment shared read at `marlin_template.h:843` against overlay reduction writes at `:1237`. Source `:1659–1674` drains each thread's async groups and enters block reduction without a block-wide entry barrier before the first writes. This is consistent with a temporal overlap not covered by the spatial capacity analysis. The final prefetch can be unused at stripe termination, so a user-visible output error and the exact causal read have not been isolated. Neither a false positive nor harmlessness has been established. The wheel has no matching line metadata, so exact offset-to-source equivalence is unverified. Do not silence the report because control A also fails. A body barrier/fix, compiler workaround, tool suppression or new optimization is outside this experiment and was not attempted.

### E005 matched provenance, failures and reproduction

Criteria commit `bc94827`, host-policy/source-model commit `eb4f88c`, raw evidence/analyzer commit `1fb86e1`. Final raw evidence is `results/e005_20261106/`. Local binary SHA256 `733c7d45777266a758d6426027ef433cc9b4d76f0c7a5e292bbb4871ab095a21`, preserved remotely at `/home/hpuvvad/csc548-w4a16-e005-20261106/results/e005_20261106/libe005.so`. Untouched source archive SHA256 `52cd2f7db842b9a72a752100118e0a701a2fc72f5959706f521fc4c9779f627c`. Explicit patch is `e005-host.patch`. The common build instantiates only 15 FP16/uint4b8/G128 configurations needed by the original selection priorities. It does not support unrelated vLLM formats. The Torch tensor wrapper is compiled but the local C ABI calls upstream marlin_mm directly with preallocated typed operands/workspace.

The official E004 operator remains SHA256 `e634b80e72078e964f8bd2ac541d66f2a900920773cf993544609612190bd9a9`. The existing E004 generation/reference bridge is reused read-only from its isolated root. Local library loading paths and Torch/CUDA mappings are in each log. Pinned core/scalar_type.hpp and torch_utils.h were added for compilation, with exact source hashes and upstream license retained. Device template matches E004's archive byte-for-byte. `build.log` retains the compiler invocation, `build-source.sha256` its inputs, `sass.txt.gz` machine code, `resources.txt` resources, and `ldd.txt` linkage.

Build uses CUDA13.4.92/GCC12.4.0, `-DUSE_CUDA -O3 -DNDEBUG -std=c++17 -lineinfo -arch=sm_120 --expt-relaxed-constexpr -shared -Xcompiler=-fPIC -Xptxas=-v`, same for every configuration, linked against the existing Python3.11/Torch2.14.0+cu130 environment. No march=native. ARC documentation/allocation were revalidated, one prun rank confirmed, and runtime thread/affinity plus full CPU/GPU/tool metadata preserved. User job 328410 remained RUNNING on c58, scheduled end 20:00:41 Eastern. Actual allocation is historical at continuation. Local host build/CTest pass is distinct from ARC CUDA/external checks.

Additional diagnostic corrections are preserved: `build-attempt1.log`/`build_attempt1.sh` show missing USE_CUDA declaration guard before the successful common build. The first finite A-address model erroneously treated all 32 ldmatrix.x2 lanes as address suppliers. Corrected to the 16 supplying lanes before modified launches. The later N2048 candidate smoke was added to the local finite model before launch. Evidence retrieval then restored the older remote model, and the analyzer correctly rejected it. Retained that snapshot as `safety-model-initial.json` and regenerated the extended model. None of these changes alter the device body or excuse racecheck. No completed measurement was discarded or overwritten. The executed preparation helper also made a redundant direct runtime probe on the allocated compute node before its prun runtime check. The final helper removes that direct probe. The executed version is preserved as `prepare_arc_build.sh`. All E005 validation/diagnostic kernel invocations use prun. The final helper was syntax-checked but not rerun against the protected result root.

Reproduction on a newly verified allocation requires the preserved isolated dependencies or a separately recorded equivalent environment. Use a fresh output root and preserve all old records. `prepare_source.py` preserves the original and emits the exact patch/selector. `prepare_arc.sh` runs actual launcher/runtime prerequisites and build. `attributes.py` performs dry resource gates. `run_arc.sh` supplies fixed direct-operator paths and stream/event harness. `validate_arc.sh` stops at the failing racecheck and never creates VALIDATED. `diagnose_gate_arc.sh` explicitly records expected failure codes for the bounded A/wheel diagnosis and runs the remaining independent checks. These helpers have historical absolute E004 bridge/result paths and must be reviewed for a new run. Never rerun against this result directory.

```bash
python3 scripts/e005/analyze.py results/e005_20261106
python3 scripts/e005/safety.py
# Historical validated-node invocation, not a new allocation request:
# PROJECT_ROOT=/home/hpuvvad/csc548-w4a16-e005-20261106
# E005_PYTHON=/tmp/e004-hpuvvad-20261106/bin/python
# bash "$PROJECT_ROOT/scripts/e005/validate_arc.sh"
```

`measure_arc.sh` is preserved as an **unexecuted gated plan**, not a tested performance script or result. It requires the absent VALIDATED sentinel and matching binary hash. Its predeclared A/B/C order reverses/rotates between the two 10-warmup/100-sample series, includes separate fallback series, and retains E004 common graph/event/cache behavior through the unchanged helper. It would keep allocation, repack, validation, capture and optional store flush outside event intervals. Device-required lock/scratch maintenance stays inside the kernel. The graph stream is explicit. No sample, latency table, clock-variability series, held-out result, achieved-occupancy counter, Tensor/L2/DRAM counter or roofline value exists for E005. Metadata clocks are observations only.

### E005 blocked measurement record

| Requested evaluation | A | B | C | Reason |
|---|---|---|---|---|
| M1/M8, both cache policies, two primary latency series |Not run|Not run|Not run|Racecheck gate failed |
| M16/M64 fallback latency series |Not run|Not run|Not run|Numerical/plan fallback checks passed, timing gate did not |
| Held-out seed2,M2/M4,all three shapes |Not run|Not run|Not run|Candidate was not released beyond safety gate |
| M8 Nsight counters/achieved residency |Not run|Not run|Not run|No profiling past the unresolved safety failure |

These cells are unavailable results, not zero latency or evidence of a regression. The actual A/B/C resource-query table and sanitizer evidence above are the completed measurements.

### E005 decision and revisit condition

Do not accept B or C. Excess spatial reservation is established. Whether B helps, whether C adds benefit, and whether either generalizes remain unmeasured. The 5% latency criteria are untested, not failed performance comparisons. Preserve all three selectable configurations only as experimental controls. Existing project dispatch and E004 wheel remain unchanged. Stop after E005.

Revisit only after a separately authorized safety investigation resolves the inherited racecheck report through an applicable documented tool explanation or an independently validated upstream/body/build correction. Such a correction requires a new matched control/candidate build, the full strict correctness/sanitizer prerequisite and every point of the predeclared timing series again. Do not repeat E005 unchanged or start another optimization to avoid the failed gate.

## E006 — bounded inherited Marlin racecheck investigation

Predeclared status: proposed in `cfcfab5`. Starting repository verified clean at `182d200` on 2026-11-06. E005 and all earlier records are immutable controls. Source analysis/reproducer/one-patch generator checkpoint: `46d3e8d`. Final outcome follows below.

Problem: E005 is blocked by shared B-fragment reads conflicting with reduction-overlay writes. Passing output checks and reproduction in the official wheel do not establish harmlessness. No incorrect output has been demonstrated.

Hypothesis: a block-wide join after every thread drains its outstanding async groups, before reduction first overwrites the B overlay, supplies the missing cross-warp ordering. Inspect terminal prefetch dataflow and uniform control flow before implementing one barrier. A per-thread async wait alone orders neither other threads nor ordinary shared loads.

Design: reuse the E005 packing/reference/workspace harness and unchanged local A binary for one direct launch. Preserve its hashes, exact specialization, full hazard-mode report, sanitizer version and exit code. Reuse isolated E005 official-wheel evidence unless a specific uncertainty requires another run. Only minimize once, to the existing 256x2048 candidate shape, if the specialization and hazard survive. Preserve original dimensions too. Rebuild unmodified and patched local sources with identical E005 settings in separate directories. Preserve the complete source diff and SASS/resources. Keep all three E005 host configurations and numerical/layout settings.

Success criterion: a source-supported ordering explanation and uniform barrier point, followed by disappearance of the original report on the first justified patch. First run the original failing square M8 case under racecheck. If it passes, check A/B/C at M1/5/8, square and both rectangles, M16/64 fallback, zero/mixed operands, reused guarded workspace, repeated direct calls and poisoned graph replay. Use unchanged all-output FP64 tolerances and lossless code/scale/reconstruction checks. Run representative memcheck/racecheck/synccheck with bounded timeouts. Any unexplained failure stops dependent work. If the first patch fails, inspect whether the signature is unchanged/new, record and stop. Never add another barrier to chase reports.

Scope: no E005 timings, residency profiles, dependency upgrades, suppression, external publication, or further optimization. Classification will distinguish source-supported correction, exact documented tool limitation, analyzed unused access, or unresolved. Even a sanitizer-clean correction does not demonstrate that the original produced incorrect outputs. Any future E005 timing must use a new matched corrected A/B/C body and a separately authorized continuation.

Evidence root: `results/e006_20261106/`. Tooling: `scripts/e006/`. Current ARC documentation checked and user allocation 328410 verified RUNNING, owned by hpuvvad, c58 RTX 5060 Ti, scheduled end 20:00:41 Eastern. Leave it running. Capture fresh metadata before tests.

### E006 reproducer, source analysis and selected design

The practical driver `scripts/e006/repro.py` reuses E005's `Case` instead of duplicating the packing/reference implementation. It initializes guarded locks and scratch, losslessly converts all codes/scales/reconstructed FP16 weights, launches A once directly and validates every output against FP64. The host-only dry query does not launch a device kernel. No warmup, graph, event timing or second GEMM occurs in this driver. E005's common allocator, including its unused flush buffer, is retained. Original dimensions are M8/K4096/N4096. One reduction to K256/N2048 preserves the exact threads256/tm1/tn8/tk8/m8=true/stages4/group_blocks8 specialization and the relevant hazard. No further minimization was attempted. K256 preserves two G128 scale groups. N128 was not selected because it invokes the known upstream fallback.

Original library `733c7d45777266a758d6426027ef433cc9b4d76f0c7a5e292bbb4871ab095a21` reproduces the square failure, passes all 32768 numerical outputs and exits99 under racecheck. The detailed report displays100 byte hazards at the default print limit and summarizes29698 errors. Grouped analysis of the rebuilt control instead reports two errors. These are different report modes and counts, not a numerical increase in independent defects. The minimized concise-analysis run passes all 16384 outputs, reports one grouped error and exits99. Its reader PC 0x2bc0 and overlay writer PCs0x4a00/10/20 match the original signature. Full command lines and exact return codes are in `original_*.log/.exit`.

The [access analysis](../results/e006_20261106/ACCESS_ANALYSIS.md) derives the shared ranges, thread/warp mapping, pipeline dataflow, pending-copy lifetime and uniformity argument. The first displayed pair is block 11, read thread 32/warp 1 at shared 0x2600 versus write thread 224/warp 7. Subtract the compiled0x400 base to obtain8704 bytes into the common B/reduction storage. Source and SASS identify an ordinary `LDS.128` next-stage B load versus a first-round `STS.128` reduction write. The first reduction barrier occurs after these writes. The prior async wait completes only the issuing thread's copy groups and does not join other warps' ordinary shared loads. Source double buffering and the terminal SASS branch show that the conflicting next-stage fragment is unused after this stripe ends in the analyzed specialization. The discarded lookahead explains why an unordered read can coexist with passing outputs. It does not make the access ordered, establish a general CUDA exemption, or demonstrate harmlessness of every Marlin report.

Selected change: one `__syncthreads()` immediately after the existing stripe-final `cp_async_wait<0>()`, before overlay reuse. Every thread first drains its own copies and completes its shared reads. The block join then prevents any thread from overwriting the overlay until its peers finish those accesses. No new B copy is issued between that join and reduction in G128. All control predicates governing this point derive uniformly from dimensions, blockIdx, gridDim and common stripe counters. Empty CTAs skip it uniformly. Stripe transitions, row-tail predicates and M16/64 fallback paths do not introduce a per-thread exit before the barrier. The full argument is in the access record. Spatial capacity alone was insufficient in E005.

Alternatives considered: a warp-only join cannot order the observed different warps. An additional async wait without a block join cannot order ordinary shared reads. Separating the reduction allocation would change shared footprint and the E005 resource question. Removing terminal prefetches would change the pipeline/control flow and needs separate analysis. A tool suppression would discard evidence without resolving the dependency. No alternate patch was implemented. The one-barrier correction preserves tiles, stages, math, format, flags and all A/B/C host policies.

Official NVIDIA PTX documents the exact async-group and CTA-barrier semantics, and Compute Sanitizer documents byte hazards, grouped reports and cp.async support. No applicable documented limitation was established for this ordinary cross-warp LDS/STS pair. Targeted upstream history finds vLLM issue 11205/PR 11493, merge `5dba2575065f5e27d468f2776e3d460a21d916e6` (2025-01-02), which separated reduction storage and fixed other older scale accesses. The later template introduction `1d0c9d6b2d74e6884c6c63f4856ac0a2cddd0508`, PR 16850 (2025-05-05), again aliases B/reduction and retains the drain/reduce transition. That history is context, not proof that the older output bug or its patch applies unchanged. Original Marlin issue 42 uses the disabled per-column scale branch on A100, and issue 45 uses different sm87 hardware/source. Neither resolves this exact report. Links, dates, applicability, patches and blame are retained in the access record and references. No dependency upgrade or broad upstream patch was imported.

The isolated official-wheel E005 records already establish the same dense specialization and a separate process with no local library mapped. Those complete records were reused without repeating a wheel run. Exact wheel PC-to-source correspondence remains unestablished and is not required to explain the line-informed local body. No upstream issue, message, publication or push was made.

### E006 matched build and preserved evidence

Unmodified and patched sources live under the isolated ARC root `/home/hpuvvad/csc548-w4a16-e006-20261106/external/{control,patched}`. `matched-source.tar.gz` preserves both complete generated trees. Its only inter-version source difference is the two added lines comprising the why comment and barrier. The E005 host dispatcher, typed C ABI and all other files match. The six original device headers in the control match the pinned E005 source archive byte-for-byte. The generated `e006-barrier.patch` and exact preparer assertions make this separately versioned body reviewable without altering historical sources.

Both builds use CUDA 13.4.92, GCC12.4, isolated Python 3.11/Torch 2.14.0+cu130, `-DUSE_CUDA -O3 -DNDEBUG -std=c++17 -lineinfo -arch=sm_120 --expt-relaxed-constexpr -shared -Xcompiler=-fPIC -Xptxas=-v`, identical include/link dependencies and original `scripts/e005/local.cu`. Only the source include root and output library path differ. Unpatched `.so` SHA256 is `51cfcf3a8c3758631e2824d0be8f0da4c6a4cf29021a769ccf864afb4d1d652a`. Patched is `e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7`. Both binaries remain intact remotely in `results/e006_20261106/{control,patched}/libe005.so`. Source/hash manifests, linked-library inventories, build logs and compressed SASS/resources are committed locally. Historical E005 binary and wheel remain untouched.

All 16 unpatched function SASS bodies match the E005 control exactly, including instruction/control encodings. The patched target specialization has 20 static block-barrier sites versus 19, and 35 padding NOPs versus 36. Total padded static instructions remain1960. Branch encodings and scheduling also change, so disappearance of a report alone would not isolate why. Source-level ordering plus the compiled barrier before overlay stores supplies the stronger explanation. Runtime resources remain126 registers/thread,32 local bytes and 0 static shared on the small target. A/B/C continue to use 36/101376 B,36/49664 B and 72/49664 B respectively, sharing one compiled entrypoint in each process. CUDA predicts1/2/2 blocks per SM. No achieved-residency observation is claimed. Fallback M16/64 launch plans stay original, and their checked A/B/C output bits match. Runtime zero static shared and cuobjdump's1024-byte shared-base convention are recorded separately, with no inferred footprint change.

### E006 failures and bounded operational corrections

1. The first detailed diagnostic filtered block 0 and displayed zero hazards. It was preserved as `original_square_filter0.*` with exit 0. Filtering one block did not reproduce the full-grid report. Removing the filter reproduced it. No acceptance evidence uses the filtered result.
2. Full detailed small-case reporting exceeded its 300-second timeout, leaving `original_small.log` and exit 124 with displayed WAR evidence but no completed numerical result. It is inconclusive as a complete driver run and is never counted as a pass. The same shape in a fresh process with concise analysis reporting completed, passed numerical checks and exited99 with the original hazard. No dimensions or checks changed in that reporting-mode correction.
3. A local upstream-history lookup initially attempted lazy Git object fetching inside the restricted network sandbox and failed DNS resolution. The read-only targeted check was repeated with network permission. The retained patches/blame are from the successful lookup. This did not affect kernel builds or cluster jobs.
4. Two generated Python bytecode cache files were accidentally included in the initial tooling checkpoint and removed in the following checkpoint. They were not source/build inputs. Subsequent staging excludes bytecode caches.
5. The evidence analyzer was first run before the remote matrix finished and correctly refused to produce a final result when `racecheck_main.exit` was absent. Completion is required. No missing or partial rows were accepted.

The final `reproduce_arc.sh` incorporates the tested concise small-report invocation. Its individual commands ran on ARC, but the reordered final helper was syntax-checked rather than rerun over preserved records. The executed initial/resume files remain in the result directory. No script is claimed guaranteed on a future allocation. Sanitizer/main validation uses 900-second timeouts, and every unexpected return code stops the script. Source and flags were held fixed throughout the validation matrix.

### E006 validation and decision

Final status: **accepted source-supported synchronization correction for the bounded tested implementation**. Complete raw evidence/tooling checkpoint: `00c665a`. Documentation/handoff commits follow. The first justified patch passed the original square A racecheck case with zero hazards and exit 0. The matched unpatched control still reports two grouped errors and exit 99. No second barrier patch was tried.

All validation uses c58's RTX 5060 Ti 16 GB/sm120,36SM, driver 615.71.09, CUDA 13.4.92 and Compute Sanitizer 2026.3.0.0 build38855100. User allocation 328410, one MPI rank and 16 OpenMP/Torch threads were verified with prun and runtime/affinity metadata. Some independent diagnostics/build work overlapped in wall time. No performance measurement or timing comparison used those runs. Final allocation check remains RUNNING and the allocation was left intact. No gateway compute, new allocation or costly submission was needed.

The complete patched-body matrix below checks every output, exact operand conversion, reusable lock reset and workspace guards. Counts are unique configuration/shape/pattern rows, without duplicate cache-policy labels. Every row uses iters=0 and warmup=0. The no_flush label identifies absence of flushing only, not a measured cache condition.

| Validation | Rows | Coverage | Direct calls and poisoned graph replays per row | Result |
|---|---:|---|---|---|
| Numerical small |90|A/B/C, M1/5/8/16/64, K256 with N128/2048, mixed/zero-A/zero-W|5 direct +5 graph|All pass|
| Numerical main |45|A/B/C, M1/5/8/16/64, 4096² and both4096/11008 rectangles, mixed|5+5|All pass|
| Each of memcheck/racecheck/synccheck small |27 per tool|A/B/C, M1/5/8, K256/N2048, all three operand patterns|2+2|Each exit 0, zero errors. Racecheck zero warnings|
| Each of memcheck/racecheck/synccheck main |27 per tool|A/B/C, M8/16/64, square and both rectangles, mixed|1+1|Each exit 0, zero errors. Racecheck zero warnings|

This is 135 numerical rows and 162 sanitizer rows, plus the separately recorded one-launch reproductions. The135-row numerical matrix retains `abs(error)<=1e-3*abs(ref)+1e-3*rms(ref)` with full FP64 output coverage. Maximum small/main relative L2 is 0.00023472576913646825/0.00020994083360038762, and maximum absolute error0.0019093224545940757/0.0019259467462688917. M5 exercises an incomplete eight-row tile. N128 exercises upstream small-N fallback. M16/64 cover fallback body paths and checked bit agreement among patched A/B/C. Rectangles exercise changed C stripe partitioning and the large scratch bound. Guards, locks and poisoned output prevent old workspace/output values from satisfying checks accidentally.

External sanitizer filtering is exactly `--kernel-name kns=e005_marlin`. It instruments the local GEMM, not the upstream repack/fill kernels. Those conversion results are independently bit-checked. Local host build/CTest pass, and the unchanged E005 analyzer still reproduces its historical stopped-gate evidence. No local CUDA compiler is present, so local CPU success is not the target CUDA validation. The target matched builds, original failure, first patched pass and full matrix provide that evidence.

Reproduction commands for a fresh verified root are documented in `scripts/e006/README.md`. The exact executed prun/tool/driver arguments are retained at the start of every raw log, with separate exit files. Main stages are:

```bash
# Export a fresh absolute PROJECT_ROOT and currently verified allocation first.
bash "$PROJECT_ROOT/scripts/e006/reproduce_arc.sh"
bash "$PROJECT_ROOT/scripts/e006/build_arc.sh"
bash "$PROJECT_ROOT/scripts/e006/test_patch_arc.sh"
bash "$PROJECT_ROOT/scripts/e006/validate_arc.sh"
python3 scripts/e006/analyze.py results/e006_20261106
```

The analyzer validates complete row sets, strict checks, source/patch identity, all control SASS against E005, the extra barrier in every compiled specialization, byte-address/thread mapping of displayed hazards, finite buffer-tag dataflow consistency and matched per-process A/B/C entrypoints. It also requires the original failures, preserved timeout, clean patched checks and absence of any timing/VALIDATED artifact. Final analyzer success means consistent E006 evidence, not that E005's performance criteria passed.

The analyzer passes locally and under ARC prun. The isolated ARC root keeps E005 results in their original directory, so that invocation passes `--e005-results /home/hpuvvad/csc548-w4a16-e005-20261106/results/e005_20261106`. The local default remains the repository's E005 evidence directory. `analyzer-arc.log` preserves the target invocation/result. This explicit path avoids duplicating historical evidence and changes no validation criterion or kernel.

Decision: classify as outcome1, a source-supported synchronization correction with the reported hazard absent in the tested cases. Outcome3 supplies an additional bounded observation: the terminal lookahead value is unused in the analyzed small-M specialization. There is no applicable documented tool limitation and no demonstrated incorrect output in the original reproducer. Report disappearance plus unchanged numerical outputs is not a proof of general race freedom, a general upstream output bug, or harmlessness on all hardware/formats. Only the fixed dense FP16/uint4b8/G128 implementation and recorded GPU/seed/paths were validated. Exact official-wheel source-PC mapping remains unknown.

**E005 may now be reconsidered using the common corrected A/B/C body**, preserved at patched SHA256 `e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7`. Historical E005 remains a failed original-body record. Its library, source, evidence and absent VALIDATED sentinel are unchanged. No latency, achieved residency, held-out seed2 evaluation or Nsight counters were collected in E006. The single next decision is whether to authorize a separately recorded E005 continuation: corrected same-body A/B/C denominators, validation prerequisite, two balanced primary/fallback series, held-out cases and targeted profiles under the original criteria. Never compare corrected candidates with original-body or wheel timings as denominators. Do not resume timing automatically or start another optimization.

Revisit only if a future authorized continuation uses the corrected matched body, or a specific new source/tool/hardware observation challenges this bounded explanation. Do not repeat the old unpatched optimization hoping that the report disappears. Preserve the unpatched body as the diagnostic control. Stop after E006.
## E005-R1 — Corrected-body reservation and grid performance ablation

Status: complete, rejected under the predeclared performance criterion. Authorized 2026-11-06 from verified clean `b3abd3d`. Criteria `2c0c4cd`, initial harness `f36a1bd`, targeted profile harness `5379eb3`, complete raw evidence and checked analyzer `607a7f3`. E005's original-body stopped run and E006's bounded synchronization evidence remain preserved. The user explicitly authorized performance with the corrected body and reuse of unchanged validation.

Problem and hypothesis: reducing unused reservation alone may improve latency, and adding CTAs may improve latency further despite additional K partitioning and FP32 reduction work. A versus B isolates reservation. B versus C changes grid, partitioning and reduction work together. Occupancy capacity is not achieved residency or proof of a bottleneck.

Fixed design: reuse E006 patched library SHA256 `e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7` without rebuilding. A uses 36 CTAs/101376 B, B36/49664 B, C72/49664 B on the verified c58 GPU. All share the corrected body, numerical format, G128, tile/stages and flags. M16/64 retain original fallback. Use unchanged E005 `Case` and E004 graph-event timing helpers. New orchestration/evidence live in `scripts/e005_r1/` and `results/e005_r1_20261106/`. Do not create historical VALIDATED sentinels. Do not use wheel or historical Marlin timings as denominators. A wheel timing diagnostic supplies no needed evidence for this request and is omitted.

Validation prerequisite: verify E006 handoff/raw-evidence hashes, exact generated source/build inputs and binary, E004 bridge and unchanged benchmark helpers. Capture fresh GPU UUID, driver/tool/runtime/CPU/topology/affinity/rank/thread/dependency metadata. Reuse E006's 135 numerical and 162 sanitizer rows only if identity holds. Run the actual launcher, Python runtime and a cheap K256/N2048 A/B/C smoke with M1/5/8, repeated direct launches, poisoned graph replay and unchanged all-output FP64/roundtrip/guard/lock checks. Every timed/held-out/profile invocation also performs numerical checks. If identity changes or validation fails, stop dependent work and reassess relevant validation. No repeat of the entire E006 investigation is planned.

Unchanged acceptance: two independent balanced series, each M1/8 square 4096 seed 1, A/B/C, both store_flush and no_flush, 10 warmups and 100 samples. C must have >=5% lower no-flush median than corrected A at both M points in each series, with <=5% store-flush regression at either. Two matched M16/64 fallback series must show no reproducible >5% regression. Report B independently even if C fails. Report raw samples, median/min/max/p90, sample variance and between-series variation. Interpret timing-resolution ties and noisy observations explicitly. Keep method and executable fixed across all points.

Held-out: after fixed primary/fallback runs, M2/4 seed 2 at square 4096, K4096/N11008 and K11008/N4096, both policies, A/B/C, 10 warmups/100 samples. No retuning or candidate changes after viewing these cases. A/B/C profiles at square M8 after 10 warmups use queried target-supported launch/shared/register/occupancy/active-warps/eligible-warps/Tensor/instruction/stall/L2/DRAM counters, cache-control none, clock-control none and one kernel-replayed target launch. Profiler timing remains distinct from application events. Extra profile cases require a specific unresolved question. No roofline or another optimization is included.

Decision: reject B/C as a replacement under the fixed square-case acceptance criterion. C increases concurrent CTA activity and gives a repeatable small M8 gain, but does not reach 5% at either required M point. Held-out rectangular gains do not change the criterion. Preserve A/B/C as experimental selectors without automatic dispatch. Stop after E005-R1. Revisit only if separately requested work brings new bottleneck evidence or changed hardware/compiler/workload/cache conditions, with a freshly matched corrected A series. Do not repeat the same trial or tune the held-outs merely to cross the threshold.

### E005-R1 identity, validation reuse and execution

E006's 113-file handoff and four library hashes match before and after the experiment. All generated corrected source/build inputs match their E006 manifest. The corrected library, unchanged E005 host ABI, E004 bridge, benchmark helpers and official wheel used only for lossless repacking match their recorded hashes. The bridge is `85f332951a8c24b3eb06594977b3e3d62b7954fa01b986da41ffeff851bfbd3f`. The wheel is `e634b80e72078e964f8bd2ac541d66f2a900920773cf993544609612190bd9a9`. No compilation, dependency install, numerical change or device-body change occurs. Historical E005 and E006 VALIDATED sentinels remain absent. E005-R1 has its own identity/smoke prerequisite marker.

The actual environment remains c58, GPU UUID `GPU-51745a0b-32d9-31c9-1ecd-359ac0ac89cc`, RTX 5060 Ti 16 GB / sm_120 / 36 SM / 32 MiB L2, driver 615.71.09, CUDA 13.4.92, Compute Sanitizer 2026.3.0.0 build 38855100, Nsight Compute 2026.3.1.0 and isolated Python 3.11/Torch 2.14.0+cu130. The complete pip freeze is byte-identical to E004. CPU remains AMD EPYC 9115, 16 physical cores, one NUMA node, GCC 12.4/OpenMPI 4.1.6. One prun rank, 16 OpenMP/Torch threads and worker affinity are checked in `runtime.log`. Metadata, source/binary hashes, actual commands and exit codes are preserved. ARC documentation and user allocation 328410 were verified before script design. All GPU work uses prun inside that interactive allocation. No new allocation, unattended job, gateway compute or srun is used. Leave the user's allocation running.

The original E006 evidence is reused: 135 numerical and 162 representative sanitizer rows, including all A/B/C, M tails, fallback, zeros, rectangles, reused locks/scratch, poisoned replay and all three tools. This reuse is justified by unchanged binary/source/runtime/environment identity, not by passing outputs alone. Fresh R1 checks add 27 small A/B/C rows at M1/5/8, K256, N2048, mixed/zero-A/zero-W, three direct calls and three poisoned graph replays each. Retained-project GPU tails and six memcheck/synccheck checks pass. Every one of 84 timed rows also passes strict all-output FP64, exact codes/scales/reconstruction, workspace guards and lock reset, with three direct and three poisoned graph checks before timing. All three profile rows pass the same checks. Maximum error across timed rows is 0.0018681105011637555 and maximum relative L2 is 0.0002088685391376363. No tolerance was relaxed. E006's narrow conclusion remains: the reported hazard is removed in this tested body, and incorrect original output was not demonstrated.

One operational failure is preserved. The first retained-project check used an absent `scripts/check_inblock.sh` under the E005 root and exited 127. The actual E004 root and executable were verified, and `project-retry.log/.exit` record all checks passing with six zero-error sanitizer summaries. No timing began before that retry passed. The final prerequisite helper points to E004. It was validated component by component, while the original fresh-directory helper ran before that path correction. The timing/profile scripts themselves completed on ARC. A provisional local analyzer invocation before profile retrieval stopped on a missing profile file and emitted no final acceptance result. After full retrieval, a parser assertion was adjusted to the observed namespace-stripping raw exporter and byte/block units. Additional mangled-name exports from the same saved reports verify the exact `e005_marlin` symbol for A/B/C. These are analysis corrections, with no new profile, changed counters or altered acceptance criterion. The completed analyzer passes and rejects an intentionally corrupted sample in a temporary evidence copy.

### E005-R1 primary latency and variability

All values below are common graph-event GEMM medians in microseconds. Each cell is series1/series2, 10 warmups and 100 samples per row. A, B and C use the same corrected library. Series1 order is ABC at M1 and BCA at M8. Series2 reverses those orders to CBA and ACB. Store flushing writes exactly 2*runtimeL2 outside the measured event interval. Setup, repacking, CPU checks and capture are excluded. Device-required reductions remain inside the operation. Profiling was run afterward, with no overlap with these timing series.

| M | Application policy | Corrected A us | B us | C us | C latency change versus A |
|---|---|---:|---:|---:|---:|
|1|no_flush|13.504 /13.504|13.888 /13.888|13.472 /13.504|−0.237% /0.000%|
|8|no_flush|13.872 /13.888|13.856 /13.888|13.504 /13.504|−2.653% /−2.765%|
|1|store_flush|28.640 /28.672|28.672 /28.640|28.640 /28.640|0.000% /−0.112%|
|8|store_flush|28.704 /28.672|28.672 /28.656|28.672 /28.672|−0.111% /0.000%|

A versus B does not support a reservation-only latency gain. B is 2.844% slower at M1 no-flush in both series, and effectively tied at M8 and under flushing. B versus C improves no-flush by 2.54–3.00% at the required points. Relative to corrected A, however, C gives only 0–0.237% at M1 and 2.653–2.765% at M8. Both fail the required 5% improvement in each series. Neither candidate violates the store-flush constraint.

Raw per-sample values, min/max/p90 and unbiased sample variance are in `series*.jsonl` and `sample_statistics.csv`. Primary no-flush p90 spans 13.856–13.952 us, while individual maxima reach 18.176 us. Sample variance spans 0.0674–0.4052 us². Store-flush p90 spans 28.704–30.720 us and variance 0.0635–0.9496 us², with maxima 31.168 us. The largest absolute matched primary median change between series is 0.238%. Stable medians do not eliminate within-run tails or establish population confidence intervals. Tiny store-flush differences and exact ties are not wins.

All 24 fallback rows pass. Across both M16/64 series, B/C latency changes versus A range from −2.320% to +0.110%, so no reproducible >5% regression is observed. M16 no-flush B appears 1.76–2.32% faster even though the candidate is disabled and device launch/body are identical. This is a useful control showing that small timing differences need not be an optimization effect. Full fallback medians, p90 and variance remain in the compact tables. No baseline is taken from historical wheel or original-body Marlin measurements.

### E005-R1 held-out cases

These 36 rows were run once, after fixing all code and primary/fallback series, at seed 2,M2/4 and both policies. No tuning followed. Each row retains100 samples. The table reports C latency change versus corrected A, calculated as100*(C/A−1). Negative is faster.

| K,N | M2 no_flush | M4 no_flush | M2 store_flush | M4 store_flush |
|---|---:|---:|---:|---:|
|4096, 4096|0.000%|−2.765%|0.000%|−0.056%|
|4096, 11008|−7.878%|−9.081%|+3.079%|+3.054%|
|11008, 4096|−7.824%|−1.834%|+0.024%|+0.047%|

Some rectangular no-flush cases benefit more, consistent with workload-dependent partitioning tradeoffs. The K4096/N11008 store-flush cases instead regress about 3.1%. This is one held-out series and does not establish a universal configuration or authorize dispatch. B's held-out changes and all raw latencies are in `comparisons.csv`. These findings cannot rescue a failed primary criterion.

### E005-R1 targeted profile and design decision

Three square M8 profiles use the corrected A/B/C body after 10 direct warmups, one selected target launch, Nsight kernel replay with 13 passes, cache-control none and clock-control none. The complete target-supported metric query, commands, reports, raw CSV, units and clock observations are preserved. A second export from each same report retains the mangled kernel name and verifies identical `e005_marlin::Marlin` specialization. The default raw demangler omits its namespace. Per-row resource queries also establish the same A/B/C function address within each timing process.

The table separates launch/resource limits from observed activity. L2 and DRAM are measured counter traffic for replay, not logical byte models or application cache guarantees.

| M8 metric | Corrected A | B | C |
|---|---:|---:|---:|
|Grid blocks|36|36|72|
|Dynamic shared requested B|101376|49664|49664|
|Shared allocated B|102400|50688|50688|
|Registers/thread|126|126|126|
|Shared/register block limits|1/2|2/2|2/2|
|Active CTAs per SM active cycle, average|1.00014|1.00014|1.75737|
|Active CTAs per SM elapsed cycle, average|0.82695|0.83661|1.36906|
|Active warps, % peak active|16.6248|16.6268|28.9537|
|Eligible warps per scheduler active cycle|0.42657|0.42605|0.64638|
|Tensor-pipe active, % elapsed|45.0536|45.4797|47.2285|
|Executed warp instructions|1442222|1442222|1575593|
|Tensor instructions|65536|65536|65536|
|L2 traffic B|11231840|11261344|11519552|
|DRAM read B|2122240|2001920|2068736|
|DRAM write B|0|512|0|
|Profiler duration us|13.344|13.920|12.992|

B raises capacity but retains36 grid blocks, and observed active CTA/warp metrics remain close to A. C's average active CTA count exceeds one and active-warps/eligible-warps rise, supporting actual concurrent CTA activity rather than only a theoretical capacity claim. This does not mean every SM maintains exactly two blocks for the whole kernel. Tensor instruction count stays fixed. Warp instructions rise 9.248%, integer thread instructions14.473%, and L2 traffic2.562% versus A. Barrier-stall percentage rises 2.096→4.286 and long-scoreboard0.417→1.265 in these replay profiles. The larger grid improves available work while adding partition/reduction/control and traffic costs. The evidence is consistent with those costs limiting the latency payoff, but does not uniquely attribute the complete timing difference to any one counter.

Profiler durations are not application event medians. Kernel replay can alter cache state and averages, and nonzero DRAM reads remain despite cache-control none. Zero observed DRAM write bytes do not imply no output/scratch stores. The store flush does not prove eviction, and no-flush does not prove residency. Clocks were not locked. The first primary pre-run metadata snapshot reports idle P8/180MHz, while subsequent active snapshots report P0 near2752MHz. Warmups precede timing, but point-level clock traces and a clock-isolated causal experiment were not collected. No roofline was constructed.

The design considered here was intentionally limited to the existing A/B/C host policies with a shared corrected body. Reservation-only B gives no measured primary benefit. C produces actual higher concurrent activity and small square latency gains but fails the acceptance threshold. Rectangular gains remain useful evidence for a future separately requested workload decision, not grounds to retune this experiment or introduce a new optimization. Keep corrected A as the performance control, retain all three selectors and stop. Original E005's stopped record and E006's narrow synchronization conclusion remain unchanged.

Reproduce analysis with `python3 scripts/e005_r1/analyze.py results/e005_r1_20261106`. It asserts 84 complete timing rows/8400 raw samples, balanced ordering, unchanged operands/body/launch plans, all-output checks, fallback behavior, held-out coverage, three exact-symbol profiles and validation-reuse manifests, then regenerates comparisons, sample statistics, repeat variability, profile summary and acceptance JSON. `scripts/e005_r1/README.md` records target commands and the prerequisite-path retry. ARC root is `/home/hpuvvad/csc548-w4a16-e005-r1-20261106`. No push or new optimization is performed.

Final handoff verification found15 last-bit differences in derived stdev values between local Python 3.10 and ARC Python 3.11. Both original exports are preserved as `sample_statistics-python310.csv` and `sample_statistics-python311.csv`, with the failed hash check in `handoff-post-analyzer-failure.log`. All raw samples, medians, variances, profile counters and acceptance results agree. The analyzer now serializes only variance/stdev to12 significant digits, retaining substantially more precision than the timing evidence warrants. This is a deterministic display correction, not a numerical-tolerance, timing-method or criterion change. The initial post-ARC hash failure is resolved by regenerating this derived file and verifying the complete handoff again, without new GPU measurements.

## E007 — Elide unused terminal shared-to-register lookahead

Status: complete, rejected under the predeclared performance criterion. The user authorized continued project work while away on 2026-11-06. Starting commit `fdd0226` was verified clean. Criteria `34a2cc1`, source/harness `f3ea292`, safety gates `b351d82`, timing/profile plan `d00053f` and provenance `c813edd`, initial evidence `e3dcd69`, complete measurement and corrected accounting `520c4e5`. Documentation commits follow. This is one bounded experiment and does not repeat rejected E005-R1 launch tuning.

Evidence and hypothesis: E006 source/dataflow/SASS establishes that the terminal next-stage B fragment is unused in the analyzed eight-row specialization. Source inspection shows its A fragment uses the same double-buffer index and has the same lifetime. E005-R1's resource ablation increases concurrency but adds executed instructions and misses the required latency gain. Skipping the terminal A/B register fetch may reduce shared-load work enough to offset a new uniform predicate and changed scheduling. Shared-load dominance has not been measured, so a gain is uncertain.

Selected change: keep the corrected E006 body as control. Only for dense FP16/uint4b8/G128, eight-row, 256-thread, 128Kx128N,four-stage specialization, skip `fetch_to_registers(k+1,pipe)` when `k==b_sh_wr_iters-1` and `slice_iters==1`. The counter decrements after the k loop, so the terminal test is one, not zero. Its final A/B buffer0 fragment has no consumer: final matmul1 uses buffer1, reduction uses frag_c, and a new stripe unconditionally initializes buffer0 before use. Leave scale/zero-point loading, global async copies/commit groups/waits, all block barriers, arithmetic/reduction order, shared reservation, grids and dispatch unchanged. The condition is CTA-uniform, including ldmatrix participants. Scope the source change at compile time so all other specializations retain their original SASS.

Alternatives: removing the E006 barrier would reopen a separate lifetime question and is excluded. Removing scale loads, changing async fetches, changing tiles/stages, adding dispatch or combining the candidate with C would confound this question. Removing terminal A/B loads is selected as one small, directly supported opportunity. No other optimization will be stacked onto a failed candidate.

Preimplementation gates: preserve source/explicit patch and finite double-buffer tag model for stripe lengths 1–86, verifying exactly one discarded fetch per stripe and identical consumed A/B tags. Generate matched control/candidate from E006's preserved corrected source, same compiler/flags/dependencies, separate libraries and isolated processes. Verify all rebuilt control SASS matches E006, and all candidate fallback SASS is unchanged. Compare full-output FP16 hashes for every matched control/candidate configuration/shape. Use original host A as the primary policy to isolate the body change. A/B/C remain available for safety coverage but no grid/reservation tuning occurs.

Validation: launcher/runtime/paths/ranks/threads and cheap K256/N2048 M1/5/8 A/B/C direct/poisoned-graph smoke before expensive checks. Candidate validation covers M1/5/8/16/64, K256/N128 and 2048, mixed/zero-A/zero-W, square 4096 and both 4096/11008 rectangles, full-output FP64, lossless packing/reconstruction, repeated guarded workspace and graph replays. Representative memcheck/racecheck/synccheck cover small tails and main M8/16/64. A/B/C share each body. All selected tools must pass without suppression, and unchanged tolerance applies. A mismatch or unexplained hazard stops dependent performance. New binaries require fresh validation, with relevant corrected-control checks and preserved E006 evidence as supporting context.

Performance and acceptance: fixed host A only, two independent process-paired series, control then candidate in series1 and reverse in series2, square M1/8, seed 1, 10 warmups/100 common graph-event samples, both policies. Require >=5% lower no-flush median at both M points in each series, <=5% store-flush regression and no reproducible >5% M16/64 fallback regression. Report p90, variance, ranges and inter-series variation. Keep all fresh corrected control denominators, without historical or wheel times. Hold out M2/4 seed 2 at square and both rectangles, both policies, after code/primary series are fixed. Profile control/candidate at square M8, one selected kernel after 10 warmups, queried executed instructions/shared-load activity/resources/Tensor/L2/DRAM/stalls, cache-control none and unlocked clocks. Profiler timing is distinct from application events.

Stopping rule: one candidate only. Stop on failed safety/code-identity gates, or after the complete bounded measurement establishes acceptance/rejection/inconclusive. Do not chase a threshold with additional changes, remove the barrier or retune held-outs. If allocation time becomes insufficient, preserve incomplete evidence as inconclusive and hand off the exact remaining task. No new GPU allocation is required for this window, and the user's existing allocation must be left intact.

Evidence root: `results/e007_20261106/`, tooling `scripts/e007/`, separate ARC root `/home/hpuvvad/csc548-w4a16-e007-20261106`. Current ARC policy and allocation 328410/c58 were verified, with about two hours remaining at start and no GPU process active. Ponytail and the technical-report skill remain in use. Raw build/source/commands/results and the final design decision follow.

### E007 source and compiled design

The candidate adds one compile-time-scoped runtime guard at the main-loop register-fetch call. The last subfragment is skipped only when `slice_iters==1`, because decrement occurs after the subfragment loop. A/B buffer0 would otherwise receive the next tile's subfragment0. Final matmul1 consumes buffer1, reduction consumes frag_c, and a new stripe initializes buffer0 in `start_pipes` before use. Scale loading remains independent and unchanged. The finite tag model verifies identical consumed A/B fragments for lengths 1–86 and one elided source fetch per stripe. The runtime predicate is CTA-uniform, so ldmatrix warp participation is uniform. All async copies, commits, waits and barriers, including E006's phase-entry join, are retained. No barrier-removal claim follows from this experiment.

All 16 rebuilt control SASS functions match the preserved E006 corrected body exactly. Only the intended small-M specialization changes in the candidate. Its static instruction total, including compiler padding, rises 1960→1976. The original static summary mistakenly labeled only unpredicated shared-load sites as all shared loads, reporting 51→43. Full accounting finds121 LDS/LDSM opcode sites in each body:70 are permanently predicated off in both, leaving 51 potentially active sites. Among those 51, unpredicated sites fall 51→43 and conditional sites rise 0→8. There are four conditional LDS.128 sites and four conditional LDSM sites. Thus the compiler predicates the terminal fetch rather than removing its instruction slots. `predicated-loads.txt` preserves relevant exact disassembly. The complete SASS remains archived.

The initial prefix parser missed predicated instructions. A first regex refinement also missed scalar `LDS` without a dot modifier, which the self-check rejected. The final parser handles plain/dotted LDS, LDSM and predicates. Independent line-level counting confirms121 total sites, with total predicated sites70/78 and always-false sites70/70. `check.py.measured.txt`, `analyze.py.measured.txt`, `build-check-initial.json` and `validation-check-initial.json` preserve the original checker and derived summaries. Raw gate logs, source, samples and profiles remain unchanged. Execution manifests resolve the preserved checker/analyzer snapshots, while the final handoff hashes the corrected current scripts. This correction changes reporting only. Source/SASS identity gates, bitwise checks and acceptance criteria are unchanged. A temporary-copy negative timing sample is rejected by the completed analyzer.

Both libraries use identical E006 flags, CUDA 13.4.92/GCC 12.4, Torch includes/links and portable host settings. Control SHA256 is `97cbee4b0dea3d96239c94a49aa05809b5dc48f2a13a64b6057bf231f8faff35`. Candidate is `4e387599c86a4cca23189f81d8bc13bbe6dce0434d12b8a97f8700a59a036b2b`. The control ELF differs from E006 because of a separate build/source path, while all device SASS matches exactly. Both libraries remain preserved on ARC. Complete generated sources and the explicit patch are archived locally. The E005 host policy, numerical format and group size remain unchanged. Fresh processes load only one local body at a time, with the official wheel used only for lossless repacking.

### E007 safety and numerical result

All 54 smoke rows pass, 27 per body. All 270 regular numerical rows pass, 135 per body, with exact full-output FP16 hash agreement for all 135 logical configuration/shape/pattern pairs. Small coverage is M1/5/8/16/64,K256,N128/2048,mixed/zero-A/zero-W,A/B/C. Main coverage is those five M points across square 4096 and both 4096/11008 rectangles,A/B/C. Each regular row checks five direct launches and five poisoned graph replays using reused guarded locks/scratch. Every output uses the unchanged FP64 threshold, and codes/scales/reconstructed FP16 weights invert exactly. Candidate fallback bits agree among A/B/C and with the matched control.

All 162 candidate sanitizer rows pass: each memcheck/racecheck/synccheck has 27 small and 27 main rows, including M1/5/8 tails, zeros, M16/64 fallback, square and rectangles. All six runs exit0 with zero errors, and both racechecks report zero warnings. Retained project GPU tails and six memcheck/synccheck checks also pass. The main matrix's maximum absolute error is 0.0019259467462688917. Every timed/profile row also passes strict numerical/workspace checks and matching control/candidate output hashes. The maximum timed error is 0.0018681105011637555, relative L2 is 0.0002088685391376363. This validates the tested candidate, not all formats or hardware. E006's narrow synchronization conclusion and its barrier remain intact.

The actual target remains allocation 328410,c58 RTX 5060 Ti 16 GB / sm_120 / 36 SM, GPU UUID `GPU-51745a0b-32d9-31c9-1ecd-359ac0ac89cc`, driver 615.71.09, Compute Sanitizer 2026.3.0.0 build 38855100, Nsight Compute 2026.3.1.0 and Python 3.11/Torch 2.14.0+cu130. Dependency freeze is byte-identical to E004. CPU/ISA/topology/NUMA/affinity/compiler/MPI metadata and one prun rank with 16 OpenMP/Torch threads are recorded. No new allocation, gateway compute, srun, dependency upgrade, external message or push occurs. Local host-only build/CTest passes separately. Every target stage was run on ARC, with cheap source/runtime/smoke gates inspected before broader validation or timing.

### E007 primary latency, fallback and held-outs

The table reports common graph-event medians in microseconds. Each cell is series1/series2, 10 warmups and 100 samples per body/point/policy. Both bodies use host A:36 blocks, 256 threads, 101376 B requested dynamic shared. Series1 runs control primary/fallback then candidate primary/fallback. Series2 reverses body order. Setup, allocation, references, full-output hashing, graph capture and store flushing remain outside GEMM event intervals. Device-required reduction/lock work remains inside. Profiles were collected afterward, with no other GPU workload observed. Raw file retrieval uses CPU/storage and is not a competing GPU workload.

| M | Policy | Fresh corrected control us | Candidate us | Candidate latency change |
|---|---|---:|---:|---:|
|1|no_flush|13.520 /13.504|13.472 /13.504|−0.355% /0.000%|
|8|no_flush|13.888 /13.888|13.888 /13.888|0.000% /0.000%|
|1|store_flush|28.672 /28.672|28.672 /28.640|0.000% /−0.112%|
|8|store_flush|28.672 /28.672|28.672 /28.672|0.000% /0.000%|

The candidate fails the required 5% no-flush gain at both primary points in both series. Exact ties and a one-series0.355% difference are not accepted speedups. All store-flush constraints pass. Primary no-flush p90 spans 13.856–13.952 us, sample variance 0.0872–0.1582 us² and maximum 17.440 us. Store-flush p90 spans 28.736–30.720 us, variance 0.1767–0.8438 us² and maximum 31.168 us. Matched primary median drift between series is at most 0.238%. Raw samples, p90, unbiased sample variance and repeat variation are retained. Two series do not establish a population confidence interval.

M16/64 fallback uses identical control/candidate SASS. Across both policies and series its candidate changes range −0.027% to+0.160%, with no reproducible >5% regression. Held-out seed 2 M2/4 at square and both rectangles is also effectively tied. The largest no-flush held-out change is +0.114% at M4,K4096,N11008, while the only store-flush difference is −0.112% at square M2. All other held-out medians tie. Held-outs were not used to retune the candidate. Across primary, fallback and held-out comparisons there are56 timing rows and 5600 samples.

### E007 profile interpretation and decision

Two square M8 profiles select one control/candidate GEMM after 10 direct warmups, kernel replay with 7 passes, cache-control none and clock-control none. Both exported mangled symbols match the intended local specialization. The table separates register count from allocation and predicate-inclusive load issue from participating load instructions.

| M8 metric | Corrected control | Candidate |
|---|---:|---:|
|Registers/thread|126|124|
|Registers allocated/thread|128|128|
|Requested/allocated shared B|101376 /102400|101376 /102400|
|Active CTAs per SM active cycle|1.000136|1.000138|
|Active warps, % peak active|16.6273|16.6306|
|Eligible warps/scheduler active cycle|0.42510|0.43364|
|Tensor active, % elapsed|45.2499|45.5188|
|Executed warp instructions|1442222|1467566|
|LDSM warp instructions|16912|16912|
|LDS warp instructions, all predicates|55128|55128|
|LDS warp instructions, pred_on_any|27344|26816|
|Predicated-on integer thread instructions|16008452|16097028|
|Tensor instructions|65536|65536|
|L2 traffic B|11238688|11238400|
|DRAM read B|1932800|1976832|
|DRAM write B|0|0|
|Profiler duration us|13.344|13.248|

The participating LDS count falls 528 warp instructions, or1.931%. The fixed square/grid partition has 66 stripes, consistent with 66*8 warps skipping one B fetch. This is a finite source/model consistency observation. Predicate-inclusive LDS and LDSM issue counts remain unchanged. SASS shows explicit predicates on the four terminal LDS/LDSM pairs. The measured LDSM counter alone therefore does not establish reduced matrix-load participation or byte traffic. Total executed warp instructions rise 1.757% and predicated-on integer thread instructions0.553%, consistent with added predicate/control work. Register count falls but the allocation remains128, so this change supplies no new register-residency capacity, and shared reservation remains the one-block constraint under host A. Tensor work is unchanged and measured L2 traffic is effectively equal.

The candidate profile duration is 0.096 us lower, but application medians tie at M8. Profiler replay is not the application timing method and does not override it. DRAM reads remain nonzero, so no-flush/cache-control-none does not demonstrate complete residency. Zero DRAM write bytes do not imply no output/scratch stores. Store flushing does not prove eviction. Clocks were unlocked. The first primary metadata snapshot shows idle P8/180MHz and subsequent active snapshots show P0, with warmups before samples. No per-point clock-isolated causal measurement, other GPU, alternate compiler, LDSM pred_on metric or full LLM serving run is included. No roofline is constructed.

Decision: reject this guard as a latency optimization under the predeclared criterion. Keep the corrected E006 body as the performance control and preserve the candidate as a measured negative result. It is safe in the tested matrix, but fewer participating shared loads and nominal registers do not produce the required benefit, and issued instruction work rises. Do not remove the barrier, change stages/tiles, combine with C or try a second guard in this experiment. Revisit only with evidence that a different compiler/architecture lowers the terminal omission to fewer issued instructions, or a different workload has a demonstrated relevant load bottleneck, then use fresh matched controls under a separately recorded experiment.

Reproduce completed evidence checks with `python3 scripts/e007/analyze.py results/e007_20261106`. It verifies source/SASS identity, 270 regular/162 sanitizer rows, matching full-output hashes, all 56 timing rows/5600 samples, input/measurement source hashes via preserved snapshots, two exact-symbol profiles, fixed criteria and normalized units. Current analyzer and negative-data self-check pass locally. ARC reproduction and final synchronization are recorded in the final handoff. No previous source/results/binary is overwritten. Stop after the bounded E007 experiment and leave the allocation intact.
