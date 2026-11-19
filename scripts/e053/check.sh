#!/usr/bin/env bash
# E053 candidate/parent checks. All sanitizer stdout is retained by inner.sh.
set -euo pipefail
B=./build/w4a16_bench
run() { "$B" --warmup 1 --iters 2 --no-flush-l2 --check-rows 4096 --kernels v3,v3_m2 "$@"; }
for m in 1 2 3 4 5 7 16; do
  for nk in '132 1000' '260 1032' '4 8' '36 136'; do
    read -r n k <<< "$nk"
    run --M "$m" --N "$n" --K "$k" --group 128 --seed 7
  done
done
for split in 1 2 3 7 16; do
  for m in 2 5; do
    W4A16_V2_SPLITS=$split run --M "$m" --N 260 --K 1032 --group 128 --seed 9
  done
done
for group in 8 64 128; do run --M 5 --N 132 --K 264 --group "$group" --seed 4; done
for k in 3 137; do run --M 5 --N 33 --K "$k" --group 128 --seed 3; done
run --M 5 --N 132 --K 1000 --group 128 --zero-weights
command -v compute-sanitizer >/dev/null
for tool in memcheck racecheck synccheck; do
  for split in 1 3; do
    W4A16_V2_SPLITS=$split compute-sanitizer --tool "$tool" --error-exitcode 99 \
      "$B" --M 5 --N 132 --K 1000 --group 128 --seed 7 --warmup 0 --iters 2 \
      --no-flush-l2 --check-rows 4096 --kernels v3,v3_m2
  done
done
echo 'PASS: E053 all-row FP64, parent bits, tails, split paths, zero weights and three sanitizers'
