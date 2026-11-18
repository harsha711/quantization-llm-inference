#!/usr/bin/env bash
# E048 checks on tc_m32 (run after scripts/check_inblock.sh). check.sh BENCH
# The bench compares every tc_m32 output bitwise with tc before timing and checks all rows against FP64.
set -euo pipefail
B=${1:?bench}
# 32-row path: exact tile, M tails inside the first/second tile, several tiles, group 64, more chunks than warps.
for M in 32 33 48 63 64 65 96 128; do
  "$B" --M "$M" --N 64 --K 256 --group 128 --seed 3 --warmup 1 --iters 2 --check-rows 4096 --kernels tc_m32
done
"$B" --M 40 --N 96 --K 192 --group 64 --seed 4 --warmup 1 --iters 2 --check-rows 4096 --kernels tc_m32
"$B" --M 36 --N 32 --K 1088 --group 128 --seed 5 --warmup 1 --iters 2 --check-rows 4096 --kernels tc_m32
# Dispatch to tc below 32 rows, vec fallback for N % 32 != 0.
"$B" --M 17 --N 64 --K 256 --group 128 --seed 3 --warmup 1 --iters 2 --check-rows 4096 --kernels tc_m32
"$B" --M 33 --N 33 --K 256 --group 128 --seed 3 --warmup 1 --iters 2 --check-rows 4096 --kernels tc_m32
for tool in memcheck synccheck racecheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 99 \
    "$B" --M 33 --N 64 --K 256 --group 128 --seed 3 --warmup 0 --iters 1 --kernels tc_m32 --no-flush-l2
done
compute-sanitizer --tool memcheck --error-exitcode 99 \
  "$B" --M 128 --N 4096 --K 4096 --group 128 --seed 1 --warmup 0 --iters 1 --kernels tc_m32 --no-flush-l2
echo CHECK-OK
