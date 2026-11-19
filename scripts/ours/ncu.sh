#!/usr/bin/env bash
# Nsight Compute profile of fused kernels at M=1 (docs/OURS.md). GPU node only.
#
#   KERNELS=naive,inblock_specialized bash scripts/ours/ncu.sh
#   KERNELS=v0 SHAPES="4096x4096" OUT=results/ours_x/ncu_v0.txt bash scripts/ours/ncu.sh
#
# One launch per (kernel, shape): --nocheck and --warmup 0 --iters 1 so the timed
# launch is the first and only matching one. ncu flushes caches before each profiled
# launch (--cache-control all) and locks clocks to base, so absolute times differ
# from the event-timed sweep; use this for *why*, the sweep for *how fast*.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
KERNELS="${KERNELS:-naive,inblock_specialized}"
SHAPES="${SHAPES:-4096x4096 8960x1536}"   # KxN
OUT="${OUT:-results/ours_ncu_$(date +%Y%m%d_%H%M%S).txt}"
NCU="${NCU:-$(command -v ncu || echo /usr/local/cuda/bin/ncu)}"
mkdir -p "$(dirname "$OUT")"
: >"$OUT"
IFS=, read -r -a ks <<<"$KERNELS"
for shape in $SHAPES; do
  K="${shape%x*}"; N="${shape#*x}"
  for k in "${ks[@]}"; do
    echo "==== kernel=$k M=1 K=$K N=$N" | tee -a "$OUT"
    "$NCU" --cache-control all --clock-control base \
      --kernel-name regex:'w4a16_(naive|inblock|v[0-9])' \
      --section SpeedOfLight --section Occupancy --section LaunchStats \
      --section WarpStateStats --section MemoryWorkloadAnalysis \
      ./build/w4a16_bench --M 1 --N "$N" --K "$K" --group 128 --nocheck \
      --warmup 0 --iters 1 --no-flush-l2 --kernels "$k" >>"$OUT" 2>&1
  done
done
echo "wrote $OUT"
