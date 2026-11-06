#!/usr/bin/env bash
# Run inside the verified allocation, after inspecting check_inblock.sh output.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${RUN_DIR:?Set RUN_DIR to a distinct results directory}"
sha256sum -c "$RUN_DIR/check.sha256"
[[ ! -e "$RUN_DIR/timings.csv" ]] || { echo 'Refusing to overwrite timings.csv' >&2; exit 1; }

{
  date -Is
  hostname
  printf 'baseline_commit=8a68227\nsource_commit=%s\n' "${SOURCE_COMMIT:-see source-build.sha256}"
  env | sort | grep -E '^(SLURM|OMP|OMPI|CUDA|LMOD_FAMILY)' || true
  scontrol show job "${SLURM_JOB_ID:?Run inside the verified allocation}"
  lscpu
  lscpu -e
  cat /proc/cpuinfo
  numactl --hardware
  taskset -pc $$
  findmnt -T "$ROOT"
  nvidia-smi -q
  nvidia-smi topo -m
  nvcc --version
  c++ --version
  mpirun --version
  cmake --version
  ncu --version
  compute-sanitizer --version
  cat build/CMakeFiles/w4a16_host.dir/flags.make
  cat build/CMakeFiles/w4a16_bench.dir/flags.make
  cat build/CMakeFiles/w4a16_bench.dir/link.txt
} > "$RUN_DIR/metadata.txt" 2>&1
sha256sum CMakeLists.txt src/*.cpp src/cuda/* include/w4a16/* tests/* scripts/*.sh \
  build/w4a16_bench build/test_cpu > "$RUN_DIR/source-build.sha256"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate \
  --format=csv > "$RUN_DIR/gpu_before.csv"

set -x
for policy in cold warm; do
  cache_args=()
  if [[ "$policy" == warm ]]; then cache_args+=(--no-flush-l2); fi
  for M in 1 8 32 128; do
    ./build/w4a16_bench --M "$M" --N 4096 --K 4096 --group 128 --seed 1 \
      --warmup 10 --iters 100 --kernels naive,inblock,inblock_specialized,cublas,cublas_dq,dequant,unfused \
      "${cache_args[@]}" --csv "$RUN_DIR/timings.csv"
  done
done
set +x
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate \
  --format=csv > "$RUN_DIR/gpu_after.csv"
