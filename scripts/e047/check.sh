#!/usr/bin/env bash
# E047 checks on the tc_splitk path (run after scripts/check_inblock.sh). check.sh BENCH
set -euo pipefail
B=${1:?bench}
# Split path (K >= 2048, small grids): M tails, two M tiles, uneven split (33 chunks), group 64, K 4096.
for M in 1 5 16 17 33; do
  "$B" --M "$M" --N 64 --K 2048 --group 128 --seed 3 --warmup 1 --iters 2 --check-rows 4096 --kernels tc_splitk,tc
done
"$B" --M 9 --N 96 --K 2112 --group 64 --seed 4 --warmup 1 --iters 2 --check-rows 4096 --kernels tc_splitk,tc
"$B" --M 20 --N 32 --K 4096 --group 128 --seed 5 --warmup 1 --iters 2 --check-rows 4096 --kernels tc_splitk,tc
# Fallbacks: too few chunks (tc), N % 32 != 0 (vec), large grid (tc).
"$B" --M 5 --N 64 --K 256 --group 128 --seed 3 --warmup 1 --iters 2 --check-rows 4096 --kernels tc_splitk
"$B" --M 5 --N 33 --K 2048 --group 128 --seed 3 --warmup 1 --iters 2 --check-rows 4096 --kernels tc_splitk
"$B" --M 128 --N 4096 --K 4096 --group 128 --seed 1 --warmup 1 --iters 2 --check-rows 4096 --kernels tc_splitk
for tool in memcheck synccheck racecheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 99 \
    "$B" --M 17 --N 64 --K 2112 --group 64 --seed 3 --warmup 0 --iters 1 --kernels tc_splitk --no-flush-l2
done
compute-sanitizer --tool memcheck --error-exitcode 99 \
  "$B" --M 16 --N 4096 --K 4096 --group 128 --seed 1 --warmup 0 --iters 1 --kernels tc_splitk --no-flush-l2
echo CHECK-OK
