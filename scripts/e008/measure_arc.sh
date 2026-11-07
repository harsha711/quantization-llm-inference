#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e008/env_arc.sh"
[[ -e "$R/VALIDATION_PASSED" ]]
cycle_base
sha256sum -c "$R/verified-inputs.sha256" > "$R/measurement-input-check.log"
sha256sum scripts/e008/* scripts/e005/bench.py scripts/e004/bench.py scripts/e005_r2/{env_arc.sh,identity.py} > "$R/measurement-tooling.sha256"
clock > "$R/series1-clocks-before.csv"
run series1 "${base[@]}" --out "$R/series1.jsonl"
clock > "$R/series1-clocks-after.csv"
clock > "$R/series2-clocks-before.csv"
run series2 "${base[@]}" --out "$R/series2.jsonl" --shapes 11008,4096 4096,11008 --counts 4 2 1 --configs C A
clock > "$R/series2-clocks-after.csv"
run identity-final "$E005_PYTHON" scripts/e005_r2/identity.py
sha256sum -c "$R/verified-inputs.sha256" > "$R/final-input-check.log"
