# E011 endpoint profiling

Status: complete. All eight complete-operation profiles and both endpoint timing series pass, with complete/negative-data analyzers on ARC and locally. The preserved timeout is resolved by resume 329113. Corrected E006 A and the E004 FP16 bridge are unchanged. No kernel change is justified yet; no dispatch is implemented. See [the final table/findings](../../docs/EXPERIMENTS.md#e011-final-measurements-and-decision).

## Scope and method

Eight profiles cover `M=4,128`, `K/N=4096/11008,11008/4096`, corrected A and pre-materialized W_hat cuBLAS. Reuse the frozen E010 preparation and complete-cycle timer. Four independently allocated weight pairs, seeds 3–6, capacity 128 and nested common activations stay fixed. Mode 16, FP16 reconstruction, FP32 reduction and the numerical tolerance are unchanged.

`profile.py --trace` validates all outputs through three direct and three poisoned complete graph cycles. It traces every individual operation and the complete four-operation graph, requiring identical per-matrix kernel sequences. It exports the fourth operation and complete-cycle traces. `plan.target` selects every kernel of the fourth GEMM in cycle 100, never a subset of a multi-kernel cuBLAS operation.

Nsight application replay reruns setup and validation, ten direct warmup cycles, and the complete 100-cycle graph for every collection pass. The graph includes the same external event nodes and first-output poisoning as E010. A CUDA profiler start/stop window encloses its replay. Skip 399 times the traced kernels-per-operation, then collect the entire final operation. Use graph-node mode, strict grid/order matching, cache-control none, clock-control none and dynamic pipeline boost. A tiny calibration has 399 fill kernels followed by a unique add kernel, verifying that disabled-profile warmups do not shift target selection. No alternative replay campaign is included.

Profiler instrumentation and node serialization can affect cache service, scheduling and duration. The selected seed 6 operation is distinct from an amortized four-matrix application median. No identical cache-line state, traffic attribution to weights alone, or equivalence of profiler and application timing is claimed. Required Marlin scratch/lock handling and cuBLAS private workspace behavior remain unchanged.

## Gates and commands

Use a new shared directory/output root for a fresh experiment. The current isolated root is `/home/hpuvvad/csc548-w4a16-e011-c4-20261107`, with results under `results/e011_c4_20261107`. Verify ARC policy and the actual allocation every session. Every GPU command uses one prun rank/16 threads inside sbatch. Export absolute PROJECT_ROOT at submission. Never infer paths from a spooled script.

1. `prerequisite_arc.sh` verifies c4 ownership/GPU/driver, full E010 handoff, exact dependencies/binaries and actual mapped libraries. It records installed Nsight help and supported metrics, and runs four small composed correctness rows. Inspect its output.
2. `release_arc.sh` repeats a cheap identity/library and small correctness check if the inspected completed job is no longer a valid dependency. Submit its dependent `validate_arc.sh` immediately, before Slurm purges its dependency record. Preserve rejected submissions.
3. `validate_arc.sh` verifies the profiler-window calibration, all eight complete-operation traces and selected outputs against E010. Its analyzer writes `TARGETS_VALIDATED` only after passing. Reuse matching E010 sanitizer evidence without repeating historical investigations.
4. `measure_arc.sh` requires target validation. The preserved first run times out after five profiles. `resume_profiles_arc.sh` applies only to that checkpoint, rechecks identical frozen inputs and collects only the remaining three cases without repeating timings. For a separately authorized fresh run, override the frozen 30-minute batch header with `sbatch --time=01:00:00 --export=ALL,PROJECT_ROOT=/absolute/new/project --dependency=afterok:<validation-job> /absolute/new/project/scripts/e011/measure_arc.sh`, after adapting all fixed evidence paths to a distinct root. Do not run against the completed record. The 33 counters require 13 application passes per kernel on the installed tool. The three-case resume actually completes within its reduced 30-minute allocation.

`measure_arc.sh` runs exactly 16 fresh endpoint timing rows, two independent reversed series with ten warmups/100 complete-cycle samples. Timing is outside Nsight. Then collect exactly eight profiles using the 33 queried counters in `metrics.txt`, preserving binary reports, raw CSV exports, logs, clock snapshots and before/after identities. Stop on any error.

Local checks:

```sh
export PYTHONDONTWRITEBYTECODE=1
python3 scripts/e011/check_local.py
python3 scripts/e011/analyze.py results/e011_c4_20261107 --stage validation
python3 scripts/e011/analyze.py results/e011_c4_20261107 --stage timing
python3 scripts/e011/analyze.py results/e011_c4_20261107
python3 scripts/e011/check_analysis.py results/e011_c4_20261107
```

Validation, timing and complete-analysis commands require the corresponding real GPU evidence. Never manufacture sentinels. A passing CPU-only build or self-check does not validate CUDA. Recheck all selected identities after queueing, because a successful earlier job does not establish a later environment.

## Metrics and normalization

`metrics.txt` is the exact 33-counter selection. Hardware names are verified against c4's saved query, launch attributes against its separate launch query. No full metric set or SASS patching is requested. Save actual pass counts and unavailable-field failures, rather than silently changing the selection.

| Group | Units and interpretation |
|---|---|
| Duration | Exported ns, converted to us. Sum sequential kernel durations for a complete operation, excluding inter-kernel gaps. Keep separate from graph-event cycle/amortized timing. |
| DRAM/L2 | Exported bytes and sectors, throughput as percent of sustained peak, L2 hit ratio as percent of sector lookups. Aggregate traffic includes activation/output/workspace and other serviced accesses. |
| Instructions/compute | Executed warp and tensor instructions, ALU/FMA instruction counts and active cycles as percent of sustained elapsed peak. Tensor activity is not useful FLOP utilization. |
| Ready work | Active CTAs per elapsed cycle, active warps as percent of active-cycle capacity, eligible warps per active scheduler cycle, issue activity as percent of active-cycle peak. Resource limits do not establish achieved residency. |
| Resources/local traffic | Grid/block sizes, registers/thread, static/dynamic shared bytes, local load/store sectors. Local requests are not necessarily DRAM misses or evidence of spill cost alone. |
| Waits | Long/short scoreboard, barrier, math throttle and wait as percent of peak sustained elapsed warp capacity. These are neither mutually exclusive wall-time fractions nor isolated causes. |

`analyze.py` preserves exact raw units and per-kernel values in `profile-analysis.json`. It sums only additive quantities for a multi-kernel operation. Utilization/resources remain arrays by kernel in `endpoint-summary.csv`, avoiding summed percentages or unjustified averages. `comparisons.csv` contains fresh matched timing denominators. `check_analysis.py` rejects incomplete/corrupt evidence and wrong target positions. A source-PC campaign or kernel change requires a separate request.

`source_analysis.py RESULTS` runs locally in the full evidence checkout. It verifies the archived E006 source hash and models the exact finite stripe topology/FP32 scratch requests from actual target resources. Its results are not measured memory traffic or an isolated reduction-time attribution.

## Completed evidence and limits

Evidence checkpoint `ca574cc` contains all measurements. `endpoint-table.md`/`endpoint-summary.csv` provide the eight cases; `comparisons.csv` and `sample-statistics.csv` preserve fresh denominators/variability. `profile-analysis.json` retains per-kernel counter values/units. Chrome cycle traces confirm one device and one kernel stream in all eight cases. All 104 completed application executions validate their outputs; incomplete attempt 1 contributes no counters. ARC's derived statistics export is retained as `sample-statistics-arc.csv`; local standard deviations differ only by at most 8.89e-16 us due to Python versions.

Profiles show nearly constant Marlin off-chip reads with increased M128 instruction/Tensor work. Different cuBLAS kernel selection, traffic and observed ready work accompany the M128 orientation difference. These are supported explanations, not isolated causality. Profile durations are 1.13–6.69% above amortized application timing. Unlocked clocks, exact cache lines, one-case allocator/library history and private cuBLAS workspace remain uncontrolled. The 13 passes merge counters rather than measure independent latency replicates. No inference benchmark or source-PC attribution is claimed. Stop after E011.

Final handoff verification from the repository root:

```sh
sha256sum -c results/e011_c4_20261107/handoff.sha256
```
