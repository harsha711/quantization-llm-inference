#!/usr/bin/env bash
# E046 inner run (one prun process on the node). inner.sh MODE SRC S
set -euo pipefail
MODE=$1 SRC=$2 S=$3
cd "$SRC"
export PATH=/usr/local/cuda/bin:$PATH
{ date -Is; hostname; nproc; lscpu; nvidia-smi -q; nvcc --version; command -v ncu; ncu --version; c++ --version; cmake --version; env | sort | grep -E '^SLURM'; } > "$S/metadata.txt" 2>&1
BUILD_DIR=$SRC/build bash scripts/build.sh > "$S/build.log" 2>&1
cp build/CMakeFiles/w4a16_bench.dir/flags.make "$S/flags.make" 2>/dev/null || true
ctest --test-dir build --output-on-failure > "$S/ctest.log" 2>&1
build/w4a16_bench --unpack-selftest > "$S/selftest.log" 2>&1
bash scripts/check_inblock.sh > "$S/check.log" 2>&1
sha256sum build/w4a16_bench src/cuda/*.cu > "$S/binaries.sha256"
cuobjdump -sass build/w4a16_bench > "$S/sass.txt"
cuobjdump -res-usage build/w4a16_bench > "$S/res-usage.txt" 2>&1
# Installed-capability query (recorded), then keep only metrics whose base name the node knows.
ncu --list-sets > "$S/ncu-sets.txt" 2>&1
ncu --list-sections > "$S/ncu-sections.txt" 2>&1
ncu --query-metrics > "$S/ncu-query-metrics.txt" 2>&1
: > "$S/metrics-dropped.txt"; keep=()
while read -r m; do [[ -z $m ]] && continue
  if grep -q "^${m%%.*}[[:space:]]" "$S/ncu-query-metrics.txt"; then keep+=("$m"); else echo "$m" >> "$S/metrics-dropped.txt"; fi
done < scripts/e046/metrics.txt
MET=$(IFS=,; echo "${keep[*]}"); echo "$MET" > "$S/metrics-used.txt"
mkdir "$S/ncu"
# label kernel M K N seed
if [[ $MODE == prereq ]]; then CASES="tc_M1_4096x4096 tc 1 4096 4096 1
cublas_dq_M1_4096x4096 cublas_dq 1 4096 4096 1"
else CASES="tc_M1_4096x4096 tc 1 4096 4096 1
tc_M16_4096x4096 tc 16 4096 4096 1
tc_M128_4096x4096 tc 128 4096 4096 1
tc_M4_K4096xN11008 tc 4 4096 11008 2
tc_M128_K4096xN11008 tc 128 4096 11008 2
tc_M4_K11008xN4096 tc 4 11008 4096 2
tc_M128_K11008xN4096 tc 128 11008 4096 2
inblock_m4_vec_M1_4096x4096 inblock_m4_vec 1 4096 4096 1
inblock_m4_vec_M128_4096x4096 inblock_m4_vec 128 4096 4096 1
cublas_dq_M1_4096x4096 cublas_dq 1 4096 4096 1
cublas_dq_M128_4096x4096 cublas_dq 128 4096 4096 1"; fi
while read -r L KER M K N SEED; do
  R="$S/ncu/$L"
  echo "ncu --set full --metrics $MET --import-source yes --cache-control all --clock-control none --replay-mode kernel --export $R build/w4a16_bench --M $M --K $K --N $N --group 128 --seed $SEED --warmup 0 --iters 1 --no-flush-l2 --kernels $KER --csv $R.bench.csv" >> "$S/commands.txt"
  ncu --set full --metrics "$MET" --import-source yes --cache-control all --clock-control none --replay-mode kernel \
    --export "$R" build/w4a16_bench --M $M --K $K --N $N --group 128 --seed $SEED --warmup 0 --iters 1 \
    --no-flush-l2 --kernels $KER --csv "$R.bench.csv" > "$R.log" 2>&1
  ncu --import "$R.ncu-rep" --page raw --csv --print-units base > "$R.raw.csv"
  ncu --import "$R.ncu-rep" --page details --csv --print-units base > "$R.details.csv"
  ncu --import "$R.ncu-rep" --page source --print-source sass --csv --print-units base > "$R.source.csv" 2> "$R.source.err" || echo "source export failed rc=$?" >> "$R.source.err"
done <<< "$CASES"
echo INNER-OK
