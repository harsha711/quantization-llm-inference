#!/usr/bin/env bash
# Local helper: emits one E053 batch script (tooling embedded, as E052). MODE: build | probe | full.
#   scripts/e053/submit.sh build | ssh arc-off-network sbatch --parsable --partition=rtx5060ti16g --nodelist=c8
#   scripts/e053/submit.sh probe BUILD_DIR | ssh arc-off-network sbatch --parsable --partition=P --nodelist=cN
#   scripts/e053/submit.sh full BUILD_DIR PROBE_DIR "inspection note" | ssh ... --dependency=afterok:PROBE_JOB
# Ours (tc_ladder) runs through t.py; Marlin and FP16 arms through t.py --marlin. Both operators are E053's multi-arch rebuilds.
set -euo pipefail
mode=$1 build=${2:-} pre=${3:-} note=${4:-}
case $mode in build) ;; probe) [[ -n $build ]];; full) [[ -n $build && -n $pre && -n $note ]];; *) exit 2;; esac
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); root=$(cd "$here/../.." && pwd)
embed() { printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$(basename "$1")"; cat "$1"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(basename "$1")"; }
cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e053-$mode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=$(case $mode in build) echo 03:00:00;; probe) echo 05:00:00;; full) echo 12:00:00;; esac)
#SBATCH --output=/tmp/e053-$mode-%j.log
MODE=$mode BUILD_NAME=$build PRE_NAME=$pre
NOTE="$note"
EOF
cat <<'EOF'
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e053-20261119
VENV=/tmp/e017-hpuvvad-20261109
OLD_TC=/mnt/beegfs/hpuvvad/csc548-w4a16-e050-20261118/prereq-c8-343302/libe050_tc.so
OLD_TC_SHA=a24a78fad127230f04a867e4a78e7a2a0453a6d4f0db95ea35900c7adaa971fe
E006=/home/hpuvvad/csc548-w4a16-e006-20261106
OLD_M=$E006/results/e006_20261106/patched/libe005.so
OLD_M_SHA=e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7
mkdir -p "$OUT"; S="$OUT/$MODE-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
trap 'rc=$?;echo "$rc" > "$S/driver.exit";touch "$S/FINISHED";exit "$rc"' EXIT
step() { echo "STEP $* $(date -Is)"; }
step metadata
{ date --iso-8601=seconds; hostname; lscpu; free -g; nvidia-smi -q; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
  ls -l /usr/local/cuda; /usr/local/cuda/bin/nvcc --version; /usr/local/cuda/bin/compute-sanitizer --version; which python3.11; } > "$S/metadata.txt" 2>&1 || true
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader > "$S/gpu.txt"
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader); [[ ${cap%%.*} -ge 8 ]]
scontrol show job "$SLURM_JOB_ID" > "$S/allocation.txt"
grep -q 'UserId=hpuvvad' "$S/allocation.txt"; grep -q 'JobState=RUNNING' "$S/allocation.txt"; [[ $(hostname) == "$SLURM_JOB_NODELIST" ]]
# Same runtime bytes as every earlier model job, unpacked to the identical node-local path (the venv is not relocatable).
T=/mnt/beegfs/hpuvvad/csc548-w4a16-e024-20261110/runtime.tar
if [[ ! -e "$VENV/bin/python" ]]; then echo restored > "$S/runtime-restored.txt"; sha256sum -c "$T.sha256"; tar xf "$T" -C /tmp; fi
EOF
for f in ../e018/diagnose.py ../e021/validate.py ../e023/run.py ../e026/att.py ../e030/gprofile.py t.py opcheck.py eqcheck.py analyze.py; do embed "$here/$f"; done
cat <<'EOF'
export PROJECT_ROOT=$E017_ROOT
set +u; source "$E017_ROOT/scripts/e017/env_arc.sh"; set -u
prun /bin/hostname > "$S/launcher.log" 2>&1
[[ $(grep -c "^$SLURM_JOB_NODELIST$" "$S/launcher.log") == 1 ]]
"$E017_PYTHON" -m pip freeze > "$S/pip-freeze.txt"
diff -u "$R/pip-freeze.txt" "$S/pip-freeze.txt"
(cd "$E017_ROOT" && sha256sum -c "$R/source-validation.sha256") > "$S/source-check.log"
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv; }
L="$OUT/$BUILD_NAME"
if [[ $MODE == build ]]; then
 step build
 [[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]
 mkdir "$S/src" "$S/m"
 base64 -d > "$S/tc-src.tar.gz" <<'B64'
EOF
(cd "$root" && git archive HEAD scripts/e050/tc_bridge.cu src/cuda | gzip -n | base64)
cat <<'EOF'
B64
 tar xzf "$S/tc-src.tar.gz" -C "$S/src"
 ARCHS=(-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90)
 # tc: E050's exact command (scripts/e050/inner.sh) plus SASS for sm_80/86/89/90
 (cd "$S/src" && set -x && prun timeout 3600 nvcc -O3 -DNDEBUG -std=c++17 -lineinfo "${ARCHS[@]}" "--generate-code=arch=compute_120,code=[compute_120,sm_120]" \
   -shared -Xcompiler=-fPIC -Isrc/cuda scripts/e050/tc_bridge.cu src/cuda/w4a16_tc.cu src/cuda/w4a16_tc_m32.cu \
   src/cuda/w4a16_inblock_vec.cu src/cuda/w4a16_inblock_rows.cu src/cuda/w4a16_inblock_specialized.cu -o "$S/libe053_tc.so") > "$S/tc-build.log" 2>&1
 # Marlin: E006's exact patched source and flags (scripts/e006/build_arc.sh) plus SASS for sm_80/86/89/90; E017 runtime's torch headers
 tar xzf "$E006/results/e006_20261106/matched-source.tar.gz" -C "$S/m"
 grep ' external/patched/' "$E006/results/e006_20261106/patched/build-source.sha256" > "$S/marlin-source.sha256"
 (cd "$S/m" && sha256sum -c "$S/marlin-source.sha256") > "$S/marlin-source-check.log"
 echo "122e8680e98ab66118b5d8ce5f3fc1d43245d44be4c0d8cbe4635600485183b7  $E006/scripts/e005/local.cu" | sha256sum -c
 mkdir -p "$S/m/scripts/e005"; cp "$E006/scripts/e005/local.cu" "$S/m/scripts/e005/"
 TORCH_ROOT=$("$E017_PYTHON" -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
 (cd "$S/m" && set -x && prun timeout 7200 nvcc -DUSE_CUDA -O3 -DNDEBUG -std=c++17 -lineinfo "${ARCHS[@]}" -gencode "arch=compute_120,code=[sm_120,compute_120]" \
   --expt-relaxed-constexpr -shared -Xcompiler=-fPIC -Xptxas=-v -Iexternal/patched/csrc/quantization/marlin -Iexternal/patched/csrc \
   -I"$TORCH_ROOT/include" -I"$TORCH_ROOT/include/torch/csrc/api/include" scripts/e005/local.cu -L"$TORCH_ROOT/lib" \
   -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -Xlinker=-rpath -Xlinker="$TORCH_ROOT/lib" -o "$S/libe053_marlin.so") > "$S/marlin-build.log" 2>&1
 (cd "$S" && sha256sum libe053_tc.so libe053_marlin.so > lib.sha256)
 for x in "$S/libe053_tc.so" "$S/libe053_marlin.so" "$OLD_TC" "$OLD_M"; do echo "== $x"; cuobjdump -lelf "$x"; done > "$S/elf-list.txt" 2>&1
 cuobjdump -res-usage "$S/libe053_tc.so" > "$S/res-usage-tc.txt" 2>&1; cuobjdump -res-usage "$S/libe053_marlin.so" > "$S/res-usage-marlin.txt" 2>&1
 # sm_120 SASS of the rebuilds must equal the originals, function by function
 cuobjdump -sass -arch sm_120 "$S/libe053_tc.so" > "$S/sass120-tc-new.txt"; cuobjdump -sass -arch sm_120 "$OLD_TC" > "$S/sass120-tc-old.txt"
 cuobjdump -sass -arch sm_120 "$S/libe053_marlin.so" > "$S/sass120-marlin-new.txt"; cuobjdump -sass -arch sm_120 "$OLD_M" > "$S/sass120-marlin-old.txt"
 python3 - "$S" > "$S/sass-compare.json" <<'PY'
import json, re, sys
from pathlib import Path
S = Path(sys.argv[1])
def funcs(p):
    out, cur = {}, None
    for line in Path(p).read_text().splitlines():
        m = re.match(r'\s*Function : (\S+)', line)
        if m: cur = m.group(1); out[cur] = []; continue
        if cur and re.match(r'\s*/\*[0-9a-f]{4,}\*/|\s*/\* 0x', line): out[cur].append(line.strip())
    return out
res = {}
for k in ('tc', 'marlin'):
    new, old = funcs(S / f'sass120-{k}-new.txt'), funcs(S / f'sass120-{k}-old.txt')
    res[k] = dict(functions_new=len(new), functions_old=len(old), same_names=sorted(new) == sorted(old),
                  identical=new == old and len(new) > 0, differing=[f for f in old if new.get(f) != old[f]][:20])
print(json.dumps(res, indent=1))
assert all(v['identical'] for v in res.values()), res
PY
 export E052_LIB="$S/libe053_tc.so" E053_TC_SHA=$(sha256sum < "$S/libe053_tc.so" | cut -d' ' -f1)
 export E017_LOCAL="$S/libe053_marlin.so" E053_MARLIN_SHA=$(sha256sum < "$S/libe053_marlin.so" | cut -d' ' -f1)
 step eqcheck
 (set -x; prun timeout 3600 "$E017_PYTHON" "$S/eqcheck.py" "$S/eq" "$OLD_TC" "$OLD_TC_SHA" "$OLD_M" "$OLD_M_SHA") > "$S/eqcheck.log" 2>&1
else
 (cd "$L" && sha256sum -c lib.sha256) > "$S/lib-check.log"
 export E052_LIB="$L/libe053_tc.so" E053_TC_SHA=$(grep libe053_tc.so "$L/lib.sha256" | cut -d' ' -f1)
 export E017_LOCAL="$L/libe053_marlin.so" E053_MARLIN_SHA=$(grep libe053_marlin.so "$L/lib.sha256" | cut -d' ' -f1)
 { ldd "$E052_LIB"; ldd "$E017_LOCAL"; } > "$S/lib-info.txt" 2>&1
fi
# t KERNEL LABEL SCRIPT ARGS: KERNEL T = tc_ladder, M = corrected Marlin
t() { local k=$1 label=$2 flag=; shift 2; [[ $k == M ]] && flag=--marlin
 (set -x; prun timeout 9000 "$E017_PYTHON" "$S/t.py" $flag "$@") > "$S/$label.log" 2>&1; }
# Sanitizer: exit 99 = reported error/hazard (stops the node); any other failure = tool unavailable on this node (recorded).
san() { local label=$1 rc=0; shift
 (set -x; prun timeout 9000 compute-sanitizer --error-exitcode 99 "$@") > "$S/$label.log" 2>&1 || rc=$?
 echo "$rc" > "$S/$label.exit"
 if [[ $rc == 99 ]]; then echo "SANITIZER REPORT $label"; exit 99; fi
 [[ $rc == 0 ]] || echo "SANITIZER UNAVAILABLE $label rc=$rc" >> "$S/unavailable.txt"; }
case $MODE in
build) ;;
probe)
 step opcheck
 (set -x; prun timeout 600 "$E017_PYTHON" -c 'import torch; a=torch.randn(256,256,device="cuda").half(); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.version.cuda, float((a@a).float().abs().sum()))') > "$S/torch-check.log" 2>&1
 t T opcheck-tc opcheck.py "$S/opcheck-tc"
 t M opcheck-marlin opcheck.py "$S/opcheck-marlin" --marlin   # --marlin in t.py: bridge left unpatched
 step sanitizers
 san memcheck-tc --tool memcheck "$E017_PYTHON" "$S/opcheck.py" "$S/memcheck-tc" --quick
 san racecheck-tc --tool racecheck --kernel-name kns=w4a16_tc "$E017_PYTHON" "$S/opcheck.py" "$S/racecheck-tc" --quick
 san memcheck-marlin --tool memcheck "$E017_PYTHON" "$S/opcheck.py" "$S/memcheck-marlin" --quick --marlin
 san racecheck-marlin --tool racecheck --kernel-name kns=Marlin "$E017_PYTHON" "$S/opcheck.py" "$S/racecheck-marlin" --quick --marlin
 step gate
 t T gate-eager-128 validate.py "$S/gateE-128" 64 128
 t T gate-eager-2048 validate.py "$S/gateE-2048" 64 2048
 t T gate-graph att.py validate "$S/refT" 128 512 2048
 t M control-graph att.py validate "$S/ref" 128 512 2048
 grep -h "ACCEPTED\|REJECTED" "$S"/gate-*.log "$S"/control-*.log > "$S/gate-summary.txt"
 ! grep -q REJECTED "$S/gate-summary.txt"
 [[ $(grep -c ACCEPTED "$S/gate-summary.txt") -ge 16 ]]
 ;;
full)
 PRE="$OUT/$PRE_NAME"
 [[ $(cat "$PRE/driver.exit") == 0 ]]
 [[ -e "$PRE/INSPECTED" ]] || echo "$NOTE" > "$PRE/INSPECTED"
 note() { printf '{"arm":"%s","context":%s,"prompt":"%s","series":%s,"file":"%s"}\n' "$1" "$2" "$3" "$4" "$5" >> "$S/cells.jsonl"; }
 cell() { local arm=$1 c=$2 p=$3 s=$4 o f; local label="${p}_${arm}_${c}_s${s}"
  clock > "$S/clk-$label.csv"
  case $arm in
  GRA|GQ|GQA) o=out; t M "$label" att.py time "$S/$o" "$arm" "$c" "$p" "$s" "$PRE/ref"; f="$S/$o/${p}_${arm}_${c}_s${s}.json";;
  GT|GTA) o=outT; v=${arm/T/Q}; t T "$label" att.py time "$S/$o" "$v" "$c" "$p" "$s" "$PRE/refT"; f="$S/$o/${p}_${v}_${c}_s${s}.json";;
  esac
  note "$arm" "$c" "$p" "$s" "$f"; ls "$f" > /dev/null; }
 mkdir -p "$S/out" "$S/outT"
 step timing
 CONTEXTS=(128 512 2048) ARMS=(GRA GQ GQA GT GTA)
 for s in 1 2 3 4; do
  cs=("${CONTEXTS[@]}") ps=(primary heldout) vs=("${ARMS[@]}")
  if (( s % 2 == 0 )); then cs=($(printf '%s\n' "${CONTEXTS[@]}" | tac)); ps=(heldout primary); vs=($(printf '%s\n' "${ARMS[@]}" | tac)); fi
  for c in "${cs[@]}"; do for p in "${ps[@]}"; do for a in "${vs[@]}"; do cell "$a" "$c" "$p" "$s"; done; done; done
 done
 step profile
 t T prof-T gprofile.py "$S/prof-T" 128 2048
 t M prof-M gprofile.py "$S/prof-M" 128 2048
 clock > "$S/clk-final.csv"
 "$E017_PYTHON" "$S/analyze.py" "$S" > "$S/analysis.json"
 ;;
esac
(cd "$S" && find . -type f ! -name FINISHED ! -name 'driver.*' ! -name all.sha256 ! -name '*.npz' | sort | xargs sha256sum > all.sha256)
echo STAGE-OK
EOF
