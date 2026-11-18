#!/usr/bin/env bash
# Run through prun on ARC, or directly on a local GPU. All small rows are checked.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
B=./build/w4a16_bench
./build/test_cpu
"$B" --unpack-selftest
# E044: shapes on the Tensor Core path (K%64, group%64, N%32): M tails, two M tiles, group 64, more K chunks than warps
for M in 1 5 8 16 17 33; do
  "$B" --M "$M" --N 64 --K 256 --group 128 --seed 3 --warmup 1 --iters 2 --kernels tc,tc_cvt,inblock_m4_vec
done
"$B" --M 9 --N 96 --K 192 --group 64 --seed 4 --warmup 1 --iters 2 --kernels tc,tc_cvt
"$B" --M 20 --N 32 --K 1088 --group 128 --seed 5 --warmup 1 --iters 2 --kernels tc,tc_cvt
for K in 1 3 7 8 9 40 65 128 129 136 137 257; do
  "$B" --M 3 --N 33 --K "$K" --group 128 --seed 1 --warmup 1 --iters 2 --check-rows 3
done
for N in 1 31 32 63; do
  "$B" --M 2 --N "$N" --K 137 --group 128 --seed 7 --warmup 1 --iters 2
done
"$B" --M 3 --N 33 --K 137 --group 128 --zero-weights --warmup 1 --iters 2

for M in 1 2 3 4 5 7; do
  for K in 136 137; do
    "$B" --M "$M" --N 33 --K "$K" --group 128 --seed 7 --warmup 1 --iters 2 \
      --kernels naive,inblock_specialized,inblock_m2,inblock_m4,inblock_vec,inblock_m4_vec,tc,tc_cvt,dequant
  done
done

if command -v compute-sanitizer >/dev/null 2>&1; then
  for tool in memcheck synccheck; do
    for K in 3 136 137; do
      compute-sanitizer --tool "$tool" --error-exitcode 99 \
        "$B" --M 5 --N 33 --K "$K" --group 128 --seed 1 --warmup 0 --iters 1 \
        --kernels naive,inblock,inblock_specialized,inblock_m2,inblock_m4,inblock_vec,inblock_m4_vec,dequant --no-flush-l2
    done
  done
  # E044: Tensor Core path under all three tools
  for tool in memcheck synccheck racecheck; do
    compute-sanitizer --tool "$tool" --error-exitcode 99 \
      "$B" --M 17 --N 64 --K 256 --group 128 --seed 3 --warmup 0 --iters 1 --kernels tc,tc_cvt --no-flush-l2
  done
  # E043: racecheck on the vector kernels (aligned and K-tail fallback)
  for K in 3 136; do
    compute-sanitizer --tool racecheck --error-exitcode 99 \
      "$B" --M 5 --N 33 --K "$K" --group 128 --seed 1 --warmup 0 --iters 1 \
      --kernels inblock_vec,inblock_m4_vec --no-flush-l2
  done
else
  echo 'Compute Sanitizer unavailable; memory/synchronization checks not run.' >&2
fi
if [[ -n "${RUN_DIR:-}" ]]; then
  sha256sum build/w4a16_bench > "$RUN_DIR/check.sha256"
fi
