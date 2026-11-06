# E005-R1 corrected A/B/C performance ablation

Read `AGENTS.md`, `docs/STATUS.md`, the E005/E006/E005-R1 records and E006's access analysis first. R1 reuses the exact E006 corrected binary. It changes no kernel, format, G128 group size, arithmetic, tile or stage. Historical results, libraries and absent VALIDATED sentinels are preserved.

The completed result rejects B/C under the original primary latency criterion. Both balanced timing series, fallback checks, held-out seed 2 cases and three M8 profiles are complete in `results/e005_r1_20261106/`. Run the stdlib-only analyzer locally:

```bash
python3 scripts/e005_r1/analyze.py results/e005_r1_20261106
```

The analyzer validates the recorded run, including the preserved path failure/retry. It regenerates sample statistics, corrected-A comparisons, between-series variation, normalized profile metrics and the fixed-criterion decision. Variance/stdev display uses12 significant digits for stable exports across Python3.10/3.11. The original differing last-bit exports are preserved separately, with identical samples/medians/decision. A temporary-copy corruption test verifies that a negative timing sample is rejected. The default Nsight raw demangler omits the namespace. Additional same-report mangled exports verify the exact local symbol. They are exports of the existing reports, not additional GPU measurements.

The tested ARC root is `/home/hpuvvad/csc548-w4a16-e005-r1-20261106`. `env_arc.sh` uses fixed historical E004/E005/E006 control roots and the existing isolated Python environment. Verify those paths and current ARC policy, GPU identity and allocation ownership/state/node/remaining time before any future execution. Historical allocation 328410 is not evidence of future availability. Do not rerun against existing result filenames or retune held-outs. A new experiment/repetition requires user authorization and a fresh absolute root.

The tested stage sequence exports PROJECT_ROOT and the verified allocation environment, then uses prun through the helpers. These are interactive-allocation commands. For an unattended run use sbatch with a validated absolute project path and successful prerequisite dependency. Never derive a spooled batch script's project root from BASH_SOURCE, and do not use srun.

```bash
export PROJECT_ROOT=/absolute/new/shared/project/root
export SLURM_JOB_ID=VERIFIED_JOB SLURM_JOB_NODELIST=VERIFIED_NODE
# Stage scripts/e004, scripts/e005, scripts/e006 and scripts/e005_r1 first.
mkdir -p "$PROJECT_ROOT/results"
bash "$PROJECT_ROOT/scripts/e005_r1/prerequisite_arc.sh"
# Inspect identity, runtime, smoke and retained-project check logs before continuing.
bash "$PROJECT_ROOT/scripts/e005_r1/measure_arc.sh"
bash "$PROJECT_ROOT/scripts/e005_r1/profile_arc.sh"
```

The original fresh prerequisite helper completed identity/runtime/smoke. Its extra retained-project GPU check was first invoked separately with an absent E005 path, preserving `project.log` and exit127. The correct E004 script/executable were verified and `project-retry` passed before timing. The final helper includes that correct path. Its components were target-tested, while the final assembled fresh-root helper was syntax-checked only. `measure_arc.sh` accepts the successful recorded retry and completed on ARC. `profile_arc.sh` completed all three cases on ARC. No script is guaranteed for an unverified future environment.

The actual retry used the shared `run` function from `env_arc.sh`:

```bash
source "$PROJECT_ROOT/scripts/e005_r1/env_arc.sh"
run project-retry bash /home/hpuvvad/csc548-w4a16-e004-20261106/scripts/check_inblock.sh
```

The supplementary exact-symbol exports use no new GPU work:

```bash
for config in A B C; do
  ncu --import "$R/ncu/${config}_M8_reused_kernel_replay.ncu-repz" \
    --page raw --csv --print-kernel-base mangled > "$R/ncu/${config}_mangled.csv"
done
```

`identity.py` verifies the E006 source/build manifest, library/bridge/helper hashes, E006 evidence counts and current Torch/GPU/driver/tool/rank/thread identity. The shell also verifies all 113 original handoff files, four binaries and identical pip freeze. R1 reuses 135 numerical/162 sanitizer rows and adds 27 small numerical/graph/workspace cases. Final identity checks pass after measurement. The historical binary is never rebuilt or copied over. Wheel code is used for repacking only, and only corrected A is a timing denominator.

Primary/fallback series use unchanged E005 `Case` and E004 graph-event timing. Each point has 10 warmups/100 samples and both operational cache policies. The complete count is 84 rows and 8400 samples. Setup, references, allocation, repacking, capture and store flushing stay outside GEMM event intervals. Required kernel maintenance remains inside. Profile replay has 13 passes with cache-control none and unlocked clocks, and is separate from application timing. No automatic dispatch, further optimization, wheel timing comparison or roofline is included.
