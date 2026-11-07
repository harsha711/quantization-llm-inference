# E010 batch crossover

Read the E010 entry in `docs/EXPERIMENTS.md` before running. This harness reuses the unchanged E005 Case and E004 bridge/timer. It fixes four independently allocated matrices and compares corrected A with mode-16 pre-materialized W_hat cuBLAS at M4/16/64/128. Fixed capacity128 preserves identical weight and activation prefixes across M. Both representations remain live. Project Q/S and an unfused temporary are unnecessary and omitted. Guarded scratch capacity uses64 rows as in the upstream allocation formula. Full-output FP64 and row-tail checks are outside timing.

Verify current ARC policy and a live user-owned allocation. Export absolute `PROJECT_ROOT`, verified `SLURM_JOB_ID`, `SLURM_JOB_NODELIST` and `SLURM_NODELIST`. The scripts use ARC `prun` inside the existing interactive allocation, one rank/16 threads. On fresh isolated shared storage run `prepare_arc.sh`, inspect the prerequisite, then `validate_arc.sh`, inspect validation, then `measure_arc.sh`. All helpers refuse existing raw outputs. They do not request/cancel allocations or rebuild the corrected library.

```bash
python3 scripts/e010/analyze.py --self-test
python3 scripts/e010/analyze.py results/e010_c4_20261107 --stage prerequisite
python3 scripts/e010/analyze.py results/e010_c4_20261107 --stage validation
python3 scripts/e010/analyze.py results/e010_c4_20261107
```

The initial series contains32 rows/3200 complete-cycle samples/12800 GEMMs. Each sample surrounds four sequential calls with required workspace handling. Ten warmup cycles precede100 samples. Series2 reverses shape/M/implementation traversal. Trace one untimed call at every shape/M and compare the actual demangled symbol with the unchanged dispatcher query. No counters or profiler timing are used.

`analysis.json` reports agreed winners and any adjacent initial crossover brackets. Only such brackets authorize a single integer arithmetic midpoint, measured for both shapes with matching validation and two reversed series. Preserve refinement in separate files. Do not expand the M range, tune, dispatch, rerun for better percentages or begin another optimization. Exact commands, raw samples, failures and decisions belong in the experiment log.
