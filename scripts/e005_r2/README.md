# E005-R2: rectangular confirmation

One unchanged E006 corrected library, preserved A/C. No E008 or dispatch. Read the predeclared entry in `docs/EXPERIMENTS.md`. Fresh c10 validation is required because its driver/runtime path differs from c58. The exact dependency freeze is restored with dependency resolution disabled, then all three binary hashes are verified.

Run these scripts over SSH on the allocated c10 node after verifying ownership, RUNNING state, remaining time, GPU and current ARC policy. Set absolute `PROJECT_ROOT`, verified `SLURM_JOB_ID`, `SLURM_JOB_NODELIST=c10` and `SLURM_NODELIST=c10`. `prepare_arc.sh` runs a cheap one-rank runtime/identity/small correctness prerequisite. Inspect its logs before invoking `validate_arc.sh`, which builds only the retained project checks and validates corrected A/C at the requested rectangles using FP64, graph/workspace checks and three sanitizers. Never rebuild the corrected library.

Only after inspecting successful validation, run `measure_arc.sh`: two independent processes, reversed configuration/rectangle/cache-policy order, M4/seed2, 10 warmups, 100 samples, exactly16 timing rows. `analyze.py RESULTS` checks complete evidence and writes the per-rectangle 5% continuation gate. Store-flush changes are separate. The profile stage is conditional on that gate. A passing gate warrants investigation only, never automatic dispatch or a new partitioner.

All scripts fail on nonzero exit, refuse to overwrite raw logs, and preserve prior experiment directories. Shared ARC root is `/home/hpuvvad/csc548-w4a16-e005-r2-20261106`. The c10-local runtime is `/tmp/e005-r2-hpuvvad-20261106`. Historical helper scripts with fixed evidence paths are not invoked.
