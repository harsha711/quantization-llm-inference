#!/usr/bin/env bash
# Marlin vs our kernels under one timer and the read flush (docs/OURS.md). GPU node.
# One-time setup on the LOGIN node (compute nodes have no internet): the pinned vLLM
# 0.33.0 wheel's _C_stable_libtorch.abi3.so in external/vllm-lib/ and
# results/e004_20261106/marlin-source.tar.gz extracted to external/marlin-src/.
#
#   PY=~/miniforge3/envs/accelprof/bin/python bash scripts/ours/marlin_compare.sh [args...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PY:?python with torch (cu130)}"
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p build
nvcc -O3 -DNDEBUG -std=c++17 -lineinfo -arch=sm_120 -shared -Xcompiler=-fPIC,-fopenmp \
  -Iinclude -Isrc/cuda scripts/ours/marlin_bridge.cu src/quant.cpp src/reference.cpp src/synthetic.cpp \
  src/cuda/ours_v1.cu src/cuda/ours_v2.cu src/cuda/ours_v3.cu src/cuda/w4a16_inblock_specialized.cu \
  src/cuda/flush_read.cu src/cuda/dequant.cu src/cuda/cublas_gemm.cu -lcublas -o build/libours_bridge.so
OUT="${OUT:-results/ours_marlin_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
{ date -Iseconds; hostname; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.memory --format=csv
  "$PY" -c 'import torch;print("torch", torch.__version__, torch.version.cuda)'
  sha256sum external/vllm-lib/_C_stable_libtorch.abi3.so; } >"$OUT/meta.txt" 2>&1
"$PY" scripts/ours/marlin_compare.py --source external/marlin-src \
  --library external/vllm-lib/_C_stable_libtorch.abi3.so --bridge build/libours_bridge.so \
  --out "$OUT/results.jsonl" "$@" 2>&1 | tee "$OUT/run.log"
echo "wrote $OUT"
