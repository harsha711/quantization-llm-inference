#!/usr/bin/env bash
# Compilation only. Run on an allocated compute node after inspecting its runtime.
set -euo pipefail
: "${PROJECT_ROOT:?Absolute project root}" "${E013_PYTHON:?Isolated Python 3.11}"
: "${E013_SOURCE:?Generated source directory}" "${E013_RESULTS:?New result directory}"
[[ "$PROJECT_ROOT" == /* && "$E013_SOURCE" == /* && "$E013_RESULTS" == /* ]]
cd "$PROJECT_ROOT"
[[ -f "$E013_SOURCE/e013-source.json" && ! -e "$E013_RESULTS/libe005.so" ]]
mkdir -p "$E013_RESULTS"
export PATH=/usr/local/cuda/bin:$PATH
TORCH_ROOT=$("$E013_PYTHON" -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
flags=(-DUSE_CUDA -O3 -DNDEBUG -std=c++17 -lineinfo -arch=sm_120 --expt-relaxed-constexpr -shared -Xcompiler=-fPIC -Xptxas=-v)
set -x
nvcc "${flags[@]}" -I"$E013_SOURCE/csrc/quantization/marlin" -I"$E013_SOURCE/csrc" \
 -I"$TORCH_ROOT/include" -I"$TORCH_ROOT/include/torch/csrc/api/include" \
 scripts/e005/local.cu -L"$TORCH_ROOT/lib" -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
 -Xlinker=-rpath -Xlinker="$TORCH_ROOT/lib" -o "$E013_RESULTS/libe005.so"
set +x
sha256sum "$E013_RESULTS/libe005.so" scripts/e005/local.cu > "$E013_RESULTS/build-binaries.sha256"
find "$E013_SOURCE" -type f -print0 | sort -z | xargs -0 sha256sum > "$E013_RESULTS/build-source.sha256"
ldd "$E013_RESULTS/libe005.so" > "$E013_RESULTS/ldd.txt"
cuobjdump --dump-resource-usage "$E013_RESULTS/libe005.so" > "$E013_RESULTS/resources.txt"
cuobjdump --dump-sass "$E013_RESULTS/libe005.so" | gzip > "$E013_RESULTS/sass.txt.gz"
cp "$E013_SOURCE/e013-source.json" "$E013_SOURCE/e013-host-tile.patch" "$E013_RESULTS/"
