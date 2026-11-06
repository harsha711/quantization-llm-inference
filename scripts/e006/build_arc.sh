#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e006/env_arc.sh"
[[ -s "$E006_RESULTS/original_square.exit" ]]
[[ $(cat "$E006_RESULTS/original_square.exit") == 99 ]]
"$E005_PYTHON" scripts/e006/prepare_source.py "$E005_CONTROL/external/pinned" external
cp external/e006-barrier.patch "$E006_RESULTS/"
for variant in control patched; do
 R="$E006_RESULTS/$variant"
 mkdir "$R"
 TORCH_ROOT=$("$E005_PYTHON" -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
 flags=(-DUSE_CUDA -O3 -DNDEBUG -std=c++17 -lineinfo -arch=sm_120 --expt-relaxed-constexpr -shared -Xcompiler=-fPIC -Xptxas=-v)
 (
 set -x
 prun timeout 600 nvcc "${flags[@]}" -I"external/$variant/csrc/quantization/marlin" -I"external/$variant/csrc" \
  -I"$TORCH_ROOT/include" -I"$TORCH_ROOT/include/torch/csrc/api/include" \
  scripts/e005/local.cu -L"$TORCH_ROOT/lib" -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
  -Xlinker=-rpath -Xlinker="$TORCH_ROOT/lib" -o "$R/libe005.so"
 ) > "$R/build.log" 2>&1
 sha256sum "$R/libe005.so" scripts/e005/local.cu scripts/e005/prepare_source.py scripts/e006/*.* \
  external/"$variant"/csrc/quantization/marlin/* external/"$variant"/csrc/core/scalar_type.hpp > "$R/build-source.sha256"
 ldd "$R/libe005.so" > "$R/ldd.txt"
 cuobjdump --dump-resource-usage "$R/libe005.so" > "$R/resources.txt"
 cuobjdump --dump-sass "$R/libe005.so" | gzip > "$R/sass.txt.gz"
done
