#!/usr/bin/env bash
# Configure and build. The CUDA architecture comes from $CUDA_ARCH if set,
# otherwise from the compute capability of the GPU visible to nvidia-smi.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"
args=(-DCMAKE_BUILD_TYPE=Release)

if command -v nvcc >/dev/null 2>&1 || [[ -n "${CUDACXX:-}" ]]; then
  arch="${CUDA_ARCH:-}"
  if [[ -z "$arch" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u)"
    if [[ "$(wc -l <<<"$caps")" -gt 1 ]]; then
      echo "Multiple GPU types visible ($caps); set CUDA_ARCH explicitly." >&2
      exit 1
    fi
    arch="${caps//./}"
  fi
  if [[ -z "$arch" ]]; then
    echo "nvcc found but no GPU visible; set CUDA_ARCH (e.g. CUDA_ARCH=120)." >&2
    exit 1
  fi
  echo "Building CUDA targets for sm_$arch"
  args+=(-DCMAKE_CUDA_ARCHITECTURES="$arch")
else
  echo "nvcc not found: building host-only targets"
fi

cmake -S "$ROOT" -B "$BUILD" "${args[@]}"
cmake --build "$BUILD" -j"$(nproc)"
