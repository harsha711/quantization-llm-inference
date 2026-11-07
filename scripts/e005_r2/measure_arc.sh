#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
[[ -e "$R/VALIDATION_PASSED" ]]
bench_base
sha256sum -c "$R/verified-inputs.sha256" > "$R/measurement-input-check.log"
sha256sum scripts/e005_r2/* scripts/e005/bench.py scripts/e004/bench.py > "$R/measurement-tooling.sha256"
clock > "$R/series1-clocks-before.csv"
run series1 "${base[@]}" --out "$R/series1.jsonl" --shapes 4096,11008 11008,4096 \
 --warmup 10 --iters 100 --balance 1 --policies store_flush no_flush
clock > "$R/series1-clocks-after.csv"
# Separate process with all traversal orders reversed, no extra timing rows.
clock > "$R/series2-clocks-before.csv"
run series2 "${base[@]}" --out "$R/series2.jsonl" --shapes 11008,4096 4096,11008 \
 --warmup 10 --iters 100 --balance 2 --policies no_flush store_flush
clock > "$R/series2-clocks-after.csv"
run identity-final "$E005_PYTHON" scripts/e005_r2/identity.py
sha256sum -c "$R/verified-inputs.sha256" > "$R/final-input-check.log"
