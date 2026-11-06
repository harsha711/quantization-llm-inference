#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e007/env_arc.sh"
[[ $(cat "$R/validation-check.exit") == 0 && ! -e "$R/measurement-input-check.log" ]]
sha256sum -c "$R/validated-inputs.sha256" > "$R/measurement-input-check.log"
sha256sum scripts/e007/* scripts/e005/bench.py scripts/e004/bench.py > "$R/measurement-tooling.sha256"
for series in 1 2; do
 variants=(control candidate)
 [[ $series == 1 ]] || variants=(candidate control)
 for variant in "${variants[@]}"; do
  clock > "$R/${variant}_series${series}-clocks-before.csv"
  run "${variant}_series${series}" "${base[@]}" --variant "$variant" --out "$R/${variant}_series${series}.jsonl" \
   --ms 1 8 --shapes 4096,4096 --configs A --warmup 10 --iters 100
  clock > "$R/${variant}_series${series}-clocks-after.csv"
  run "${variant}_fallback${series}" "${base[@]}" --variant "$variant" --out "$R/${variant}_fallback${series}.jsonl" \
   --ms 16 64 --shapes 4096,4096 --configs A --warmup 10 --iters 100
  clock > "$R/${variant}_fallback${series}-clocks-after.csv"
 done
done
for variant in control candidate; do
 run "${variant}_heldout" "${base[@]}" --variant "$variant" --out "$R/${variant}_heldout.jsonl" \
  --ms 2 4 --seed 2 --shapes 4096,4096 4096,11008 11008,4096 --configs A --warmup 10 --iters 100
done
clock > "$R/heldout-clocks-after.csv"
