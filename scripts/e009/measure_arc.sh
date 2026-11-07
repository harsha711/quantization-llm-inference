#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e009/env_arc.sh"
[[ -e "$R/VALIDATION_PASSED" ]]
cycle_base
sha256sum -c "$R/verified-inputs.sha256" > "$R/measurement-input-check.log"
sha256sum scripts/e009/* scripts/e005/bench.py scripts/e004/bench.py scripts/e005_r2/{env_arc.sh,identity.py} > "$R/measurement-tooling.sha256"
clock > "$R/series1-clocks-before.csv"
run series1 "${base[@]}" --out "$R/series1.jsonl"
clock > "$R/series1-clocks-after.csv"
clock > "$R/series2-clocks-before.csv"
run series2 "${base[@]}" --out "$R/series2.jsonl" --shapes 11008,4096 4096,11008 --counts 4 2 1 --series 2
clock > "$R/series2-clocks-after.csv"
run identity-final "$E005_PYTHON" scripts/e005_r2/identity.py
sha256sum -c "$R/verified-inputs.sha256" > "$R/final-input-check.log"
(cd "$E006_CONTROL"; sha256sum -c results/e006_20261106/handoff.sha256) > "$R/final-e006-check.log"
(cd /home/hpuvvad/csc548-w4a16-e005-r2-20261106; sha256sum -c results/e005_r2_c10_20261106/handoff.sha256) > "$R/final-e005-r2-check.log"
cp build/CMakeCache.txt "$R/project-CMakeCache.txt"
cp build/CMakeFiles/w4a16_bench.dir/flags.make "$R/project-flags.make"
cp build/CMakeFiles/w4a16_host.dir/flags.make "$R/project-host-flags.make"
sha256sum CMakeLists.txt scripts/build.sh build/w4a16_bench build/test_cpu > "$R/project-build.sha256"
find src include tests -type f -print0 | sort -z | xargs -0 sha256sum > "$R/project-source.sha256"
(cd /home/hpuvvad/csc548-w4a16-e008-20261106; sha256sum -c results/e008_c10_20261106/handoff.sha256) > "$R/final-e008-check.log"
scontrol show job "$SLURM_JOB_ID" > "$R/allocation-final.txt"
