#!/usr/bin/env bash
# Correctness gate for the roofline-series kernels (docs/OURS.md). Run on a GPU node
# after `bash scripts/build.sh`. Any failure stops the script.
#
#   KERNELS=v0 bash scripts/ours/check.sh
#   KERNELS=v0,v1 SANITIZE=0 bash scripts/ours/check.sh   # skip compute-sanitizer
#
# Every call also runs `naive` (Harsha's retained baseline), so a shape that breaks
# the harness rather than your kernel shows up there first. Each kernel is checked
# against the FP64 CPU reference with |err| <= 1e-3*|ref| + 1e-3*rms(ref).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
B=./build/w4a16_bench
KERNELS="${KERNELS:-v0}"
SANITIZE="${SANITIZE:-1}"
run() { "$B" --warmup 1 --iters 2 --no-flush-l2 --kernels "naive,$KERNELS" "$@" >/dev/null; }

echo "== tails in K (partial final word / partial final group)"
for K in 1 3 7 8 9 40 65 128 129 136 137 257; do
  run --M 3 --N 33 --K "$K" --group 128 --seed 1
done
echo "== tails in N (fewer columns than a warp / block)"
for N in 1 31 32 33 63 65 257; do
  run --M 2 --N "$N" --K 137 --group 128 --seed 7
done
echo "== several rows"
for M in 1 2 3 5 16; do
  run --M "$M" --N 300 --K 1000 --group 128 --seed 3
done
echo "== zero weights (scale 0 path)"
"$B" --M 3 --N 33 --K 137 --group 128 --zero-weights --warmup 1 --iters 2 --no-flush-l2 \
  --kernels "naive,$KERNELS" >/dev/null
echo "== 128-bit fast path (v2/v3): N % 4 == 0 but not a multiple of 128, K % 8 == 0"
for NK in "4 1000" "36 136" "132 1000" "260 1032" "388 8"; do
  read -r N K <<<"$NK"
  run --M 3 --N "$N" --K "$K" --group 128 --seed 5
done
echo "== forced K-splits (v1 W4A16_V1_SPLITS / v2,v3 W4A16_V2_SPLITS), M=2"
for S in 1 2 3 7 16; do
  W4A16_V1_SPLITS=$S W4A16_V2_SPLITS=$S run --M 2 --N 260 --K 1032 --group 128 --seed 9
done
echo "== real shapes at M=1"
for KN in "4096 4096" "1536 1536" "1536 8960" "8960 1536"; do
  read -r K N <<<"$KN"
  run --M 1 --N "$N" --K "$K" --group 128 --seed 1
done

if [[ "$SANITIZE" == 1 ]]; then
  if command -v compute-sanitizer >/dev/null 2>&1; then
    for tool in memcheck racecheck synccheck; do
      echo "== compute-sanitizer $tool"
      for K in 3 137; do
        compute-sanitizer --tool "$tool" --error-exitcode 99 \
          "$B" --M 5 --N 33 --K "$K" --group 128 --seed 1 --warmup 0 --iters 1 --no-flush-l2 \
          --kernels "$KERNELS" >/dev/null
      done
      # v2/v3 fast path, with and without the cross-block reduction
      for S in 1 3; do
        W4A16_V2_SPLITS=$S compute-sanitizer --tool "$tool" --error-exitcode 99 \
          "$B" --M 3 --N 132 --K 1000 --group 128 --seed 1 --warmup 0 --iters 1 --no-flush-l2 \
          --kernels "$KERNELS" >/dev/null
      done
    done
  else
    echo "compute-sanitizer not found: memory/race checks NOT run" >&2
  fi
fi
echo "PASS: $KERNELS"
