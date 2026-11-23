#!/usr/bin/env bash
# Sweep the K-split count S by hand (docs/OURS.md, v1 and v2 split rules). GPU node.
#
#   PEAK=427.4 KERNEL=v2 bash scripts/ours/splits.sh
#   PEAK=427.4 KERNEL=v1 SPLITS="1 2 3 4 5 7 8" MS="1 4" bash scripts/ours/splits.sh
#
# Forces S through W4A16_V1_SPLITS (v1) or W4A16_V2_SPLITS (v2, v3) and writes one
# CSV row per (shape, M, S) to results/ours_<ts>/splits.csv; the `splits` column is
# appended by this script. Read flush, same timing as sweep.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
KERNEL="${KERNEL:-v2}"
SPLITS="${SPLITS:-1 2 3 4 5 6 8 10 12 16 20 24 32}"
MS="${MS:-1}"
ITERS="${ITERS:-200}"
PEAK="${PEAK:?set PEAK (GB/s) from w4a16_bwprobe}"
SHAPES=("4096 4096" "1536 1536" "1536 8960" "8960 1536")
case "$KERNEL" in
  v1) VAR=W4A16_V1_SPLITS ;;
  v2|v3) VAR=W4A16_V2_SPLITS ;;
  *) echo "KERNEL must be v1|v2|v3" >&2; exit 1 ;;
esac

OUT="${OUT:-results/ours_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
CSV="$OUT/splits_$KERNEL.csv"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
for KN in "${SHAPES[@]}"; do
  read -r K N <<<"$KN"
  for M in $MS; do
    for S in $SPLITS; do
      rm -f "$tmp"
      env "$VAR=$S" ./build/w4a16_bench --M "$M" --N "$N" --K "$K" --group 128 --seed 1 \
        --warmup 10 --iters "$ITERS" --flush-read --kernels "$KERNEL" --peak-gbps "$PEAK" \
        --csv "$tmp" >>"$OUT/splits_$KERNEL.log"
      if [[ ! -f "$CSV" ]]; then head -1 "$tmp" | sed 's/$/,splits/' >"$CSV"; fi
      tail -n +2 "$tmp" | sed "s/\$/,$S/" >>"$CSV"
      printf '%s K=%s N=%s M=%s S=%-3s ' "$KERNEL" "$K" "$N" "$M" "$S"
      python3 - "$CSV" <<'PY'
import csv, sys
r = list(csv.DictReader(open(sys.argv[1])))[-1]
print(f"median={float(r['median_us']):7.2f}  mean={float(r['mean_us']):7.2f}  pct_floor={float(r['pct_floor']):5.1f}  pass={r['pass']}")
PY
    done
  done
done
echo "wrote $CSV"
