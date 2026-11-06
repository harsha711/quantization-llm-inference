#!/usr/bin/env bash
# Execute from the real shared project path inside the verified allocation.
set -euo pipefail
: "${PROJECT_ROOT:?Export an absolute project root}"
: "${RUN_DIR:?Set an E004 evidence directory}"
cd "$PROJECT_ROOT"
mkdir -p "$RUN_DIR"
[[ ! -e "$RUN_DIR/libe004.so" ]] || { echo 'Refusing to overwrite bridge' >&2; exit 1; }
export PATH=/usr/local/cuda/bin:$PATH
nvcc -O3 -DNDEBUG -std=c++17 -lineinfo -arch=sm_120 -shared -Xcompiler=-fPIC,-fopenmp \
  -Iinclude -Isrc/cuda scripts/e004/bridge.cu src/quant.cpp src/reference.cpp src/synthetic.cpp \
  src/cuda/w4a16_naive.cu src/cuda/w4a16_inblock.cu \
  src/cuda/w4a16_inblock_specialized.cu src/cuda/w4a16_inblock_rows.cu \
  src/cuda/dequant.cu src/cuda/cublas_gemm.cu -lcublas -o "$RUN_DIR/libe004.so"
sha256sum scripts/e004/*.py scripts/e004/*.cu scripts/e004/*.sh src/*.cpp src/cuda/* include/w4a16/* "$RUN_DIR/libe004.so" \
  > "$RUN_DIR/bridge-source.sha256"
