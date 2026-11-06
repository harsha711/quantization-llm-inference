# E006 bounded overlay-synchronization investigation

Read `AGENTS.md`, the E005/E006 experiment entries and `results/e006_20261106/ACCESS_ANALYSIS.md` first. This harness preserves all earlier source/results. It adds exactly one barrier after the existing stripe-final async drain, and supports the same E005 A/B/C host policies. It has no timing command.

The tested root is `/home/hpuvvad/csc548-w4a16-e006-20261106` on c58. Original E005 library/source and E004 reference bridge remain in their historical roots. The existing Python3.11/Torch2.14 environment is reused without installation. Reverify current ARC documentation, allocation ownership/state/node/remaining time/GPU before any remote execution. These historical paths and allocation328410 do not establish future availability. Leave user allocations intact.

Use a fresh absolute project root for reproduction. Stage `scripts/e004`, `scripts/e005`, `scripts/e006` and create `results/e006_20261106` there. Export PROJECT_ROOT and the verified SLURM_JOB_ID/SLURM_JOB_NODELIST. `env_arc.sh` sets the recorded one-rank/16-thread prun environment. It uses fixed historical control/environment paths, so verify those first. No `srun` or batch-file-relative project discovery is used.

```bash
bash "$PROJECT_ROOT/scripts/e006/reproduce_arc.sh"
bash "$PROJECT_ROOT/scripts/e006/build_arc.sh"
bash "$PROJECT_ROOT/scripts/e006/test_patch_arc.sh"
bash "$PROJECT_ROOT/scripts/e006/validate_arc.sh"
```

The original/control processes intentionally return sanitizer99, captured explicitly. Patched validation requires zero. Each stage refuses to overwrite records and stops on unexpected errors. The final reproducer helper uses concise analysis reporting for the small shape after a detailed-report timeout was observed. Its component commands were ARC-tested, but this reordered final helper was only syntax-checked as a whole. Executed historical command files and full logs are preserved in the evidence root. `reproduce_small_analysis_arc.sh` is the historical timeout-resume command, not a second minimization search.

The practical minimal driver is `repro.py`. It reuses the existing E005 `Case` for exact operand generation, lossless codes/scales/FP16 reconstruction, original scratch/lock initialization and guards, one direct A launch and strict all-output FP64 validation. A dry host query is not a device launch. Defaults are M8/K4096/N4096. `--k 256 --n 2048` retains the failing specialization. It deliberately retains E005's common allocation support, including an unused flush buffer, instead of duplicating packing/reference infrastructure. There are no warmups, graph calls or timing events in this driver.

Example sanitizer invocation inside the verified allocation, with paths resolved from `env_arc.sh`:

```bash
prun timeout 300 compute-sanitizer --tool racecheck --racecheck-report analysis \
  --error-exitcode 99 --kernel-name kns=e005_marlin \
  "$E005_PYTHON" scripts/e006/repro.py \
  --source "$E005_CONTROL/external/pinned" \
  --local "$E006_RESULTS/patched/libe005.so" --wheel "$E006_WHEEL" \
  --bridge "$E004_CONTROL/results/e004_20261106/libe004.so" \
  --out "$E006_RESULTS/new-case.json"
```

The unpatched library is `$E006_RESULTS/control/libe005.so`, or the preserved original `$E005_CONTROL/results/e005_20261106/libe005.so`. Never load two local versions in the same process. Wheel code is loaded only for repacking in local runs. The independent official-wheel GEMM evidence remains in E005's isolated process records, with no local library mapped. E006 does not repeat it or assert exact wheel PC/source equivalence.

Source preparation and matching have assert-based checks. Preserve the generated complete source trees, patch, build flags/hashes, SASS/resources, sanitizer versions/exits/logs and strict-check JSONL. The local repository stores those sources as `matched-source.tar.gz`. Both shared libraries remain preserved on ARC with hashes. `analyze.py results/e006_20261106` checks completed recorded evidence. Its success does not authorize performance work or create E005's absent VALIDATED sentinel.
