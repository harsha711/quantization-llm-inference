# E013: two existing M128 Marlin tiles

Read the predeclared E013 entry in `docs/EXPERIMENTS.md`. A is the unchanged corrected automatic selection, expected 64K×256N/256 threads. B explicitly selects the already compiled 64K×128N/128-thread specialization. The entire corrected E006 device source and all 16 SASS functions must match the retained build exactly. This experiment changes host selection only. It does not lower shared reservation, increase the grid, alter FP32 reduction, add a dispatch rule or implement batching.

The generator accepts only the SHA256-verified E006 archive and refuses an existing output directory. It extracts the corrected tree without rewriting historical source. Both choices are in one fresh library using unchanged `scripts/e005/local.cu`. Its mode0/1 calls are A/B here. The new host branch rejects unsupported experimental formats or M. Existing E005/E006/E010 files are imports or read-only controls.

The four-matrix preparation preserves E010's capacity128, separate live FP16 W_hat buffers, common activation, guarded output/locks/scratch, complete cycles, three direct/three poisoned graph checks, ten warmups and 100 timed cycle samples. The controls use seeds3–6, and conditional held-outs use7–10. No artificial flush or per-matrix timing is inserted. Cycle median divided by four is derived amortized latency. There is no cuBLAS timing or new profiler campaign.

Primary shape-specific success requires at least5% lower B median in each of two reversed series. Exactly eight rows/800 cycle samples are collected. If neither shape passes, stop without held-out timing. Otherwise validate both shapes with fresh seeds and collect eight held-out rows with the same ordering. A restricted improvement requires both primary and held-out gates for that shape. Report losses in the other shape. Keep corrected A default. No inference-throughput or unique-bottleneck claim follows.

The current isolated ARC root is `/home/hpuvvad/csc548-w4a16-e013-c4-20261108`, results `results/e013_20261108`. Scripts target the inspected c4 allocation and require absolute exported PROJECT_ROOT. All workloads use one prun rank/16 threads inside sbatch, with portable host compiler settings. Every stage checks current ownership/RUNNING state/node/GPU. Preserve CPU/ISA/topology/NUMA/affinity, compiler/MPI/GPU/tool versions, library/source/binary hashes, exact commands and logs. Verify live availability before submission, rather than assuming these historical IDs remain valid. A fresh rerun needs a distinct shared/result root and corresponding environment paths.

Run `capability_arc.sh` first. It records the actual GPU/runtime, checks unchanged historical evidence and runs the small runtime test. Inspect this result. Submit `prerequisite_arc.sh` with an afterok dependency. It generates source, compiles once, checks exact SASS and traces/validates both actual executable choices on mixed/zero cases. Inspect the six completed rows and successful process exit, then submit `validate_arc.sh` with afterok. This performs retained project build/CTest/GPU checks, all four matrices on both target shapes, complete-cycle memcheck/synccheck and representative one-matrix racecheck. Passing numerical output never overrides a sanitizer failure.

Only after inspecting successful validation release dependent `measure_arc.sh`. A user-held job can be submitted earlier with afterok, then released after inspection to avoid dependency-record expiry. This produces the primary series only. Inspect `primary-analysis.json`. If its heldout_required flag is true, submit `heldout_arc.sh` with afterok after checking the primary job. Otherwise run the complete CPU analyzer against the primary record and stop. Never manufacture sentinels or overwrite failed outputs.

Local checks, with Python bytecode writing disabled:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/e013/check_local.py
bash scripts/build.sh
ctest --test-dir build --output-on-failure
```

After real evidence is retrieved:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/e013/analyze.py results/e013_20261108
PYTHONDONTWRITEBYTECODE=1 python3 scripts/e013/check_analysis.py results/e013_20261108
```

Offline tests exercise exact extraction, overwrite/corruption rejection, two-series traversal, the5% criterion and invalid evidence. They do not validate CUDA. The real analyzer requires compiled-source/SASS identity, actual trace/template agreement, complete numerical/sanitizer records, matching operands/environments and all raw samples. Its output records negative results as valid outcomes. The negative-data check modifies temporary copies only.

Operational history: initial ARC access failed routing/DNS and resumed after the user restored access. Capability330208 passes. Build330209 matches all16 device functions and the first mixed A/B case passes, then smoke exits1 on a Python `del case` cleanup error. The failing code snapshot and raw outputs are preserved. The repair deletes only the actual cases list. `resume_prerequisite_arc.sh` is deliberately specific to that checkpoint, rechecks runtime/build identity, preserves the failed smoke files in `smoke-attempt1/` and reruns all six small rows. Resume330211 passes. No numerical tolerance, kernel binary or acceptance criterion changes, and no failed attempt contributes timing.
