# E053 command and source checkpoints

All submissions used ARC home as the submission directory, partition rtx5060ti16g, node c8, one task/16 CPUs/exclusive, 45-minute upper limits. The generated scripts export an absolute BeeGFS PROJECT_ROOT and invoke prun. Actual source payloads, compiler commands, arguments and checks are retained in each source.tar.gz, source-commit.txt, build/driver/inner logs and per-step logs.

- Integration 1418165: merge retained optimize/inblock-k and samarth/bandwidth-roofline, preserving device sources. Local bash scripts/build.sh and ctest --test-dir build --output-on-failure (host-only, 2/2).
- 7eec31e, job 343808: `bash scripts/e053/submit.sh baseline | ssh arc-off-network 'cd /home/hpuvvad && sbatch --parsable --nodelist=c8'`. Stopped at first-use workspace memset in trace; no candidate.
- d7f02b0, job 343816: same command, fixed trace setup. Stopped on Python inspect-module shadowing before smoke; no candidate.
- 628f389, job 343819: `bash scripts/e053/submit.sh baseline baseline-c8-343816 | ssh arc-off-network 'cd /home/hpuvvad && sbatch --parsable --nodelist=c8'`. Hash-checked build/gate reuse after recorded inspection, fresh smoke/calibration/profile pass.
- 13b29bd, job 343823: `bash scripts/e053/submit.sh validation | ssh arc-off-network 'cd /home/hpuvvad && sbatch --parsable --nodelist=c8 --dependency=afterok:343819'`. CUDA build, numerical/parent bits/SASS and representative sanitizers pass.
- 2aec26a, job 343828: `bash scripts/e053/submit.sh full validation-c8-343823 | ssh arc-off-network 'cd /home/hpuvvad && sbatch --parsable --nodelist=c8 --dependency=afterok:343823'`. Uses exact inspected validation library/executable and checks frozen measured source hashes before timing.

Retrieval used `ssh arc-off-network 'ssh c8 "tar -C /mnt/beegfs/hpuvvad/csc548-w4a16-e053-20261119 -cf - STAGE-c8-JOB"' | tar -xf - -C results/e053_20261119`. Successful-stage all.sha256 manifests were checked before derived analysis. Failed stages lack final all.sha256 because their driver stopped before that step. All attempts remain preserved. Obsolete retrieval acknowledgements after a node released its allocation failed harmlessly and caused no compute or overwritten evidence.

Offline checks:

```bash
python3 -B scripts/e053/check_gate.py results/e053_20261119/baseline-c8-343819 --stage baseline
python3 -B scripts/e053/check_gate.py results/e053_20261119/validation-c8-343823 --stage validation --baseline results/e053_20261119/baseline-c8-343819
python3 -B scripts/e053/analyze.py --self-test
python3 -B scripts/e053/analyze.py results/e053_20261119/full-c8-343828
python3 -B scripts/e053/check_analysis.py results/e053_20261119/full-c8-343828
```

No push, default change, model run or further optimization. These job IDs are historical, not allocations to reuse.
