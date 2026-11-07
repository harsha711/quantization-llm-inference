# E012 source-PC attribution

Complete, evidence checkpoint `181f6fe`. All four instruction and eight sampling profiles pass structural/numerical checks. All eight sampling captures fail the predeclared quality gate. Instruction caller ambiguity also prevents a firm majority attribution. Keep corrected A unchanged. Read the [tables, decisions and failures](../../docs/EXPERIMENTS.md#e012-final-findings-and-decision) before another experiment.

## Fixed workload and method

Corrected E006 A only, M4/M128, K/N=4096/11008 and 11008/4096. Reuse frozen `scripts/e011/profile.py`, four distinct seed-3–6 matrices, common capacity-128 activation prefixes, G128, mode 16 and the exact numerical contract. The profiling harness repeats all setup/full-output/guard checks, ten direct warmup cycles and the complete 100-cycle event graph on every pass. Application replay selects the fourth operation of the final cycle: skip 399/count 1. Use strict grid/order matching, graph-node mode, cache-control none, clock-control none and dynamic pipeline boost. No new application timing, kernel change or cuBLAS comparison.

`instruction-metrics.txt` requests one software-patched counter at four cases. `sampling-metrics.txt` requests total samples, five not-issued indicators and six diagnostics in two reversed case orders. Sampling is fixed at option 0/32 cycles, one pass and 33554432 B buffer. Patched instruction counts are diagnostic. Sampling locates consumers and has no phase-time resolution. Neither yields production phase latency or a uniquely identified producer bottleneck.

`export_pc.py` combines exact report API metric instances with all CLI-exported inline source contexts. The installed CUDA `ncu` wrapper selects `/opt/nvidia/nsight-compute/2026.3.1/extras/python`. The initial parent-directory locator failed and remains preserved. `regions.py` verifies the E006 source archive/hash and representative boundaries. It counts each PC once and keeps conflicting phases or helper-only callers ambiguous. Exact all-ones PC samples remain unmapped and in the denominator.

## Evidence and gates

- `capability_arc.sh`: c4 hardware, source/warp queries and preserved E011 handoff. Original job 329192 fails only at the API locator after successful earlier steps.
- `source_inspection_arc.sh`, `inspect_install.py`, `resume_source_inspection_arc.sh`, `query_chip_arc.sh`: CPU-only c54 report/installation queries. The explicit gb206 query is not a query of c54's counters. No sm120 kernel executes on c54. The initial c52 job 329188 is cancelled unstarted after its node becomes occupied.
- `validate_arc.sh`: corrected API, actual runtime/small correctness, exact libraries, retained GPU checks and all four new full-output traces matching E011. Reuse the unchanged same-environment external-kernel sanitizer evidence. Job 329217 completes in 2m01s.
- `measure_arc.sh`: fixed twelve captures. Job 329220 has afterok:329217 and is submitted on hold, then released only after inspecting the gate. It completes in 5m56s. No retry/tuning follows failed sampling quality.
- `analyze.py`, `check_analysis.py`: verify every replay/output/source, exact counter units and aggregate/PC sums, source contexts, full template arguments and quality criteria. Corrupt/missing evidence is rejected. Invalid-quality captures remain inconclusive. `analyze_arc.sh` runs only CPU checks on c54 and hashes raw profiles before/after.

ARC jobs use sbatch/prun, one task and absolute PROJECT_ROOT. GPU jobs use 16 threads on c4. CPU-only c54 jobs request one CPU/task, while Slurm allocates the whole 16-CPU node. Saved job IDs are historical. Use distinct roots for any separately authorized rerun, never the completed output paths. The initial API failures, stale queue-status correction and premature retrieval extraction are documented. Successful raw records are unchanged. The early partial identity log is preserved under `checkpoints/gate-retrieval/`.

## Reproduce the analysis

From the full repository checkout:

```sh
export PYTHONDONTWRITEBYTECODE=1
python3 scripts/e012/check_local.py
python3 scripts/e012/check_validation.py results/e012_c4_20261107 results/e011_c4_20261107
python3 scripts/e012/analyze.py results/e012_c4_20261107 results/e011_c4_20261107 results/e006_20261106
python3 scripts/e012/check_analysis.py results/e012_c4_20261107 results/e011_c4_20261107 results/e006_20261106
sha256sum -c results/e012_c4_20261107/handoff.sha256
```

The last three positional paths are this experiment, retained E011 references and archived E006 source evidence. `attribution.json` retains counts, quality failures, series shifts and conservative ownership bounds. `attribution-summary.csv` and the two Markdown tables are generated, with deterministic phase-key ordering. Exact PC instances remain in compressed JSON alongside every binary report and raw/source CSV. No phase-time claim is made. The analysis marker records structural completion, not acceptance of sampling quality. Reanalysis can regenerate derived files, so verify the handoff before regenerating them.

## Result and stopping rule

Known main-loop instruction increments are about 70% of the total increments, but caller-ambiguity bounds allow less than 50%. All eight sampling captures report dropped bytes despite zero overflow. M128 also falls below 95% source coverage. Do not use the saved failed-quality shares to choose a kernel optimization, discard all-ones PC samples, reinterpret zero overflow as losslessness or repeat unchanged sampling. A separately requested quality-focused pilot and better caller evidence are prerequisites for further attribution. No dispatch, partition rewrite, inference benchmark or next experiment is implemented.
