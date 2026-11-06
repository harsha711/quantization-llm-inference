#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?Absolute project root}" "${E005_PYTHON:?Existing isolated Python}" "${RUN_DIR:?New result directory}"
cd "$PROJECT_ROOT"
[[ ! -e "$RUN_DIR/libe005.so" ]]
export PATH=/usr/local/cuda/bin:$PATH
TORCH_ROOT=$("$E005_PYTHON" -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
flags=(-DUSE_CUDA -O3 -DNDEBUG -std=c++17 -lineinfo -arch=sm_120 --expt-relaxed-constexpr -shared -Xcompiler=-fPIC -Xptxas=-v)
set -x
nvcc "${flags[@]}" -Iexternal/local/csrc/quantization/marlin -Iexternal/local/csrc \
 -I"$TORCH_ROOT/include" -I"$TORCH_ROOT/include/torch/csrc/api/include" \
 scripts/e005/local.cu -L"$TORCH_ROOT/lib" -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
 -Xlinker=-rpath -Xlinker="$TORCH_ROOT/lib" -o "$RUN_DIR/libe005.so"
set +x
sha256sum "$RUN_DIR/libe005.so" scripts/e005/*.* external/local/csrc/quantization/marlin/* external/pinned/csrc/core/scalar_type.hpp > "$RUN_DIR/build-source.sha256"
ldd "$RUN_DIR/libe005.so" > "$RUN_DIR/ldd.txt"
cuobjdump --dump-resource-usage "$RUN_DIR/libe005.so" > "$RUN_DIR/resources.txt"
cuobjdump --dump-sass "$RUN_DIR/libe005.so" | gzip > "$RUN_DIR/sass.txt.gz"
