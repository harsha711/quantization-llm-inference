#!/usr/bin/env bash
# Timing sweep for the roofline series (docs/OURS.md). Run on the GPU node after
# check.sh passes. Writes a fresh results/ours_<timestamp>/ directory.
#
#   bash scripts/ours/sweep.sh                       # runs the bandwidth probe first
#   PEAK=410.2 KERNELS=naive,v0,v1,cublas bash scripts/ours/sweep.sh
#
# Env: PEAK (GB/s ceiling; measured with w4a16_bwprobe if unset), KERNELS, MS,
#      ITERS, FLUSH (read = read-only cold-L2 flush, the default; store = the original
#      write flush, which inflates fast kernels, see docs/OURS.md; none = warm L2).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
KERNELS="${KERNELS:-naive,inblock_specialized,v0,cublas,unfused}"
MS="${MS:-1 2 4 8 16}"
ITERS="${ITERS:-200}"
FLUSH="${FLUSH:-read}"
SHAPES=("4096 4096" "1536 1536" "1536 8960" "8960 1536")   # K N; Qwen2.5-1.5B layers + square

OUT="results/ours_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
{
  echo "date=$(date -Iseconds)"
  echo "host=$(hostname)"
  echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo none) dirty=$(git status --porcelain -- src scripts CMakeLists.txt 2>/dev/null | wc -l)"
  echo "kernels=$KERNELS"; echo "Ms=$MS"; echo "iters=$ITERS"; echo "flush=$FLUSH"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm,clocks.max.memory --format=csv 2>/dev/null || true
  nvcc --version 2>/dev/null | tail -1 || true
} >"$OUT/meta.txt"

if [[ -z "${PEAK:-}" ]]; then
  ./build/w4a16_bwprobe | tee "$OUT/bwprobe.txt"
  PEAK="$(sed -n 's/^ceiling_gbps=//p' "$OUT/bwprobe.txt")"
fi
echo "peak_gbps=$PEAK" >>"$OUT/meta.txt"

case "$FLUSH" in
  read) flush_flag=(--flush-read) ;;
  store) flush_flag=() ;;
  none) flush_flag=(--no-flush-l2) ;;
  *) echo "FLUSH must be read|store|none" >&2; exit 1 ;;
esac

for KN in "${SHAPES[@]}"; do
  read -r K N <<<"$KN"
  for M in $MS; do
    echo "== M=$M K=$K N=$N"
    ./build/w4a16_bench --M "$M" --N "$N" --K "$K" --group 128 --seed 1 \
      --warmup 10 --iters "$ITERS" ${flush_flag[@]+"${flush_flag[@]}"} --kernels "$KERNELS" \
      --peak-gbps "$PEAK" --csv "$OUT/sweep.csv" | tee -a "$OUT/sweep.log"
  done
done
echo "wrote $OUT"
