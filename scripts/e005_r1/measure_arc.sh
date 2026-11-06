#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e005_r1/env_arc.sh"
[[ -e "$R/PREREQUISITE_PASSED" ]]
sha256sum -c "$R/verified-inputs.sha256" > "$R/measurement-input-check.log"
sha256sum scripts/e005_r1/* scripts/e005/bench.py scripts/e004/bench.py > "$R/measurement-tooling.sha256"
for series in 1 2; do
 clock > "$R/series${series}-clocks-before.csv"
 run "series${series}" "${base[@]}" --out "$R/series${series}.jsonl" --ms 1 8 --shapes 4096,4096 \
  --warmup 10 --iters 100 --balance "$series"
 clock > "$R/series${series}-clocks-after.csv"
 run "fallback${series}" "${base[@]}" --out "$R/fallback${series}.jsonl" --ms 16 64 --shapes 4096,4096 \
  --warmup 10 --iters 100 --balance "$series"
 clock > "$R/fallback${series}-clocks-after.csv"
done
run heldout "${base[@]}" --out "$R/heldout.jsonl" --ms 2 4 --seed 2 \
 --shapes 4096,4096 4096,11008 11008,4096 --warmup 10 --iters 100 --balance 1
clock > "$R/heldout-clocks-after.csv"
