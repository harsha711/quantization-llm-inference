# Experiment and design decision log

Read the numerical/measurement [contract](CONTRACT.md) before interpreting these records. Preserve raw evidence, failed attempts, and inconclusive outcomes. An alternative that has not been implemented or measured is not a failed optimization. Historical entries below reconstruct only what the repository supports. Missing prior failure numbers or commit IDs remain unknown.

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

**Change:** Block=(32,8), grid=(ceil(N/32),M). Each warp owns 32 columns and one of eight contiguous packed-word K intervals. For W=ceil(K/8), slice y consumes `[floor(W*y/8),floor(W*(y+1)/8))`. These intervals cover all words exactly once, even W<8. Each thread preserves FP16 reconstruction and FP32 FMA, writes its partial to `float partial[8][32]`, and reaches the barrier. Invalid N lanes/empty slices contribute zero. Warp zero adds eight partials in slice order and writes FP16 once. Shared storage is 1024 bytes. There are no atomics or global partials. The FP32 reduction order changes.

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

**Archive guard during restart:** Sending the saved original profiler script created the intended attempt-archive directory before the restart script's nonexistence guard, so the first restart stopped before measurements. Moved that single script into the original result directory and removed the now-empty archive directory with `rmdir`, then reran the guarded move. No evidence was overwritten or deleted. The original full timing/partial profile attempt is preserved separately. The final prerequisite reuses the unchanged executable only after checking its saved SHA256 gate.

**Versions, execution, and reproduction:** Final evidence is `results/e003_c58_20261106/`, evidence checkpoint `a16cbc5`. The remote source is `/home/hpuvvad/csc548-w4a16-e003-20261106`. Implementation checkpoint `5a7bcaf` and final measurement-script checkpoint `25b2291` use the same executable, SHA256 `5ee05fc2046d6e821cdd6f7c9b944541eb7f9e00e9f5e7a50b0e61a7887a232e`. Metadata's `baseline_commit=8a68227` identifies the historical naive origin, not the E003 denominator. The actual controls are same-executable naive and E002. Exact commands are in `prepare_commands.sh`, `restart_commands.sh`, and `measure_commands.sh`. `resource_usage.txt` records output of the separate post-measurement cuobjdump command retained in `measure_commands.sh`. The first timing series and interrupted profiles are retained in `results/e003_c58_20261106_attempt1/`, including its original profiler script. Attempt-local analysis scripts are frozen snapshots, not acceptance tools for an incomplete profile set. Use the final directory's analysis scripts.

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

Status: proposed. E003 checkpoint `c1c6423` verified clean before work. No external kernel or custom optimization will be changed.

**Objective and hypothesis:** Determine whether maintained dense INT4 W4A16 Marlin exposes a reproducible, tractable limitation on the actual RTX 5060 Ti. A clear no-go is acceptable. Faster external execution alone is not an optimization opportunity or novelty claim.

**Design and preservation:** Inspect and pin vLLM source, start from its kernel tests/benchmarks, directly invoke the GEMM operator, and losslessly repack existing codes/scales. Prefer an isolated environment and adapter over changing this project's build dependencies. Preserve all E001–E003 sources, selectors, evidence, and user allocations. No serving engine, model download, external kernel patch, or E005 implementation.

**Gate 1 — prerequisites and compatibility:** Verify live allocation/GPU, Python/CUDA/runtime and build requirements. Pin source commit, dependencies, license, build flags and hashes. Inspect integer encoding, tile packing, scale permutation/indexing, matrix orientation, reconstruction rounding, intermediate reductions, padding and workspace. Validate conversion and small valid shapes before M=1/32, K=N=4096. Existing tolerance remains unchanged. Distinct numerical operations require separate labels/references. Confirm executed kernel, reject fallback. A concrete integration blocker ends dependent gates after bounded diagnosis and independent documentation/tooling.

**Gate 2 — performance screening:** M={1,2,4,8,16,32,64,128}, (K,N)={(4096,4096),(4096,11008),(11008,4096)}, group=128, seed=1, both existing store-flush/no-flush policies. Unsupported shapes are explicit. Compare Marlin, original/reconstructed-weight cuBLAS, retained naive/E002/E003, and inexpensive unfused context using identical operands. CUDA events exclude setup/repacking/tuning, which are recorded separately. Start with a short complete screen, repeat promising/surprising cases independently. Capture per-launch samples, warmups/iterations, medians/p90, clocks, exact environment, launch/stream method and hashes. Diagnose wrapper launch gaps before claims. Never combine profiler-replay and event timing.

**Gate 3 — evidence:** Select at most three representative cases for supported-metric profiling. Inspect actual kernel/geometry, registers/shared memory/occupancy, tensor/scalar activity, eligibility/stalls, L2/DRAM and extra work. Separate observations from hypotheses and record what explanations are excluded.

**Gate 4 — decision criterion:** Recommend exactly one next experiment only with an affected workload, suspected mechanism, concrete proposed change, measurable criterion, tradeoffs/regressions, held-out cases, and prior-work classification. Otherwise recommend one bounded ablation or document the integration no-go. Do not search or patch indefinitely.

**Validation, commands, findings and failures:** Pending. New records live under `results/e004_20261106/`. ARC job 328410 is RUNNING, owned by hpuvvad on c58, verified at runtime 01:16:34 of 06:00:00. This does not authorize reuse without checking remaining time again as needed. Current ARC documentation still specifies prun inside allocations.

**Revisit only if:** A documented blocker changes, or a separately requested experiment tests the final evidence-based recommendation.

**Pre-measurement integration decisions:** Selected released vLLM v0.33.0, peeled source `4ba787ec9959e566d3b587a796dff07673324da5`, with official wheel SHA256 `4b6bdec4ab269990bf70d45a3c1e94c3760a27064513760193b26d6f0da883aa` and embedded `g4ba787ec9`. Initial moving main was inspected and recorded but not benchmarked. Release changes relative to main are explicit, not silently mixed. Source subset/license archive SHA256 `79e67e69caff6099a6346a90406b7168dee18704c1441e28b21e4b948eb7b33f`. Python3.11 isolated at `/tmp/e004-hpuvvad-20261106`, shared project `/home/hpuvvad/csc548-w4a16-e004-20261106`. No original dependencies changed. Initial Python3.6 default was unsuitable, but installed Python3.11 and glibc2.28 satisfy the selected wheel tags. Torch2.14.0 installed successfully. Retained-project CUDA build and prerequisite are pending.

**Pre-measurement review corrections (not performance failures):** Static review found that PyTorch's implicit graph capture stream differs from an outer stream hardcoded in ctypes closures. Explicit capture on `torch.cuda.current_stream()` now makes all launches/events/flushes use the same stream. Added NaN output poisoning before each candidate check and graph replay to reject stale results or empty captures. Removed dequantization from public timed selector choices because it produces W_hat rather than C, while retaining internal bit-check materialization. Hash only source files, excluding generated Python cache directories. These issues were corrected before adapter measurement. The graph capture/setup cost is now recorded separately. Small grouped smoke uses K=256,N=128 to avoid the different per-column path inferred from a single scale row at K=128.

**Prerequisite attempt 1:** The initial target Release project and bridge builds succeeded, but the bridge's post-build source hash glob encountered `scripts/e004/__pycache__` and exited 1. The file-only correction arrived after that shell had started. No timing followed. `bridge-build-attempt1.log` and `bridge-source-attempt1.sha256` preserve the failed gate. `resume_prerequisite.sh` records preservation and rehashing of the unchanged binary, then runs the required project correctness/sanitizer gate. This is a concrete failed prerequisite, not a failed Marlin kernel integration or performance result.

**Prerequisite attempt 2:** The retained project correctness gate completed with six zero-error sanitizer runs, and checksum-pinned vLLM0.33.0 installed successfully. ARC `prun` flattened the quoted `python -c` argument, producing `SyntaxError: invalid syntax` on the token `import`. The prerequisite failed immediately. The correction uses `scripts/e004/check_runtime.py` as a real file, avoiding compound interpreter arguments. Preserve `torch-smoke-attempt1.log`. This launcher quoting failure occurred before any Marlin measurement, and does not indicate a CUDA or numerical failure.

**Gate 1 passed:** Target runtime loads official vLLM0.33.0 with Torch2.14.0+cu130, CUDA13.0 package runtime and native Torch sm_120 support on RTX5060Ti. Project build/CPU tests/small correctness and six Compute Sanitizer runs pass. Small K=256,N=128,M={1,5,32} and main K=N=4096,M={1,32} pass lossless inverse packing/scale-bit checks, all-element FP16 reconstruction checks, full E001/E002/E003 control comparisons, strict all-row FP64 checks for all eight timed comparators, and poisoned-output graph replay checks under both policies. Torch GPU traces show one actual dense `marlin::Marlin<...>` kernel for each direct operator call, not a fallback. Raw smoke logs/jsonl/traces preserve selection and numerical errors. Smoke timing uses only 2 warmups/3 events and is not acceptance performance evidence. First repack calls include lazy initialization/host gaps in their setup event interval, so wall/event setup costs are labeled, not claimed as isolated GPU repack throughput. The next complete screen uses 10 warmups/20 events; targeted repeats use 10/100. The same bridge, operands, graph events, node and cache policies apply to every comparator in each stage. Per-selector first validation-call wall time and capture costs are saved separately. The isolated wheel cuBLAS13.1.1.3 is loaded in-process, as shown in process maps, so new cuBLAS comparisons must not use historical system-library medians.

**Screen attempt 1 stopped on correctness:** The first screen produced 122 passing rows then original-weight cuBLAS at M=128,K=N=4096 failed the unchanged all-row threshold on 2/524288 outputs. Failed row/column pairs are (5,789) and (91,677). Relative L2 is 2.950522e-4, and maximum absolute error is 2.415464e-3. No failing timing row was emitted and the process stopped. Preserve the partial screen as `screen_attempt1*`; do not treat it as a complete performance map or combine it with rerun points. Historical M128 checks sampled64rows, excluding both failed rows, and used a different cuBLAS library. They were not retroactively invalidated by claiming identical coverage/environment.

**Bounded numerical diagnosis:** `diagnose_cublas.py` compares default math mode0 and `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` mode16 with identical operands, FP32 compute type, and original/reconstructed-weight FP64 references, two repetitions each. Default original-weight failures reproduce exactly. Mode16 has zero bad outputs for both paths, relative L2 2.070527e-4 original and 2.076626e-4 reconstructed. The flag forces reductions to use compute precision, according to NVIDIA's cuBLAS13.0 documentation. It may also change internal algorithm selection, so the ablation does not isolate every internal cause. E004 will explicitly set mode16 on its own handle, label every row, and rerun the entire screen. Existing project cuBLAS source/behavior and old results remain unchanged. This is a reference-compatibility configuration correction, not a new custom kernel optimization or relaxed tolerance.

**Diagnostic implementation failure:** The first new reduction diagnostic mistakenly passed the host-order GPU pointer vector A,W,Q,S,W_hat to the bridge expecting A,Q,S,W,W_hat. cuBLAS accessed invalid memory and that process failed with CUDA error700. Its script and log are preserved as `diagnose_cublas_attempt1.py.txt` and `cublas_reduction_diagnostic-attempt1.log`. The corrected explicit indices were independently reviewed, then the successful ablation ran in a new process. This failure is not evidence about cuBLAS arithmetic or Marlin.

**Gate 2 complete screen:** All384 rows (3 shapes*8 batch sizes*8 implementations*2 policies) pass strict FP64 validation with every row checked, code/scale/W_hat round trips exact, and retained full-output controls equal. No requested shape needs padding or is unsupported. This is a short screen, 10warmups/20events, with explicit cuBLASmode16 throughout. Preliminary patterns warrant independent repeats: Marlin no-flush is flat at M=1..8 then rises at16, and at square M=64 it is close to/reported slightly slower than reconstructed cuBLAS. Select M={1,8,16,64}, K=N=4096, all8comparators/bothpolicies for two independent10/100 repeats. This is a diagnostic choice, not an optimization acceptance criterion. Separately compare eager and graph events at M={1,32}, square, Marlin/E002/reconstructed-cuBLAS, identical10/100 conditions, to quantify wrapper-gap sensitivity. Do not splice these runs into the screen. Select at most3 final profiling cases after inspecting repeat stability.

**Repeat and timing-method findings:** Both independent64-row10/100 series pass all numerical and retained-control checks. Square no-flush Marlin M8 is13.888us in both, M16 is19.584/19.456us, and M64 is58.304us in both. Same-run reconstructed cuBLAS M64 is54.768/54.752us, so its modest lead reproduces. M16 store-flush varies30.848→32.384us, and screen/repeat differences must not be hidden. Matching12-row eager/graph diagnostics show M1 no-flush Marlin18.560us eager versus13.536us graph (p90 20.608vs13.856). M32 and longer kernels show much smaller/mixed differences. This supports common graph events for kernel comparison rather than treating Python dispatch gaps as GPU work. The runs are independent, clocks unlocked, and the difference is not an isolated pure wrapper-cost subtraction.

**Gate 3 selection:** Exactly three square Marlin cases: M8 (eight-row specialization), M16 (boundary and resource comparison), M64 (reproduced cuBLAS crossover). Each profiles one actual kernel after10 untimed direct launches, explicit profiler start/stop, `--cache-control none --clock-control none` with kernel replay, no application flushing. Query supported target metrics and record the kernel, geometry, resources, tensor/scalar work, eligibility/stalls, L2/DRAM. These are reused-input profiler diagnostics, not the application's graph-event timings and not guaranteed identical cache state on every replay. No extra shape/configuration search or external kernel patch is authorized. A potential two-CTA resource ablation remains only a hypothesis until counters are assessed.
