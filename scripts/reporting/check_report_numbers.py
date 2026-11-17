#!/usr/bin/env python3
"""Assert that report/main.tex quotes the generated numbers (report/figs/numbers.json) exactly as rounded in the text."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
tex, n = (root / 'report/main.tex').read_text(), json.loads((root / 'report/figs/numbers.json').read_text())
e = n['e041']
need = []
for M in ('1', '4', '128'):
    need += [f"{M} & {e[M]['T1_ms']:.2f} & {e[M]['speedup']['16']:.2f} & {e[M]['efficiency']['16']:.2f}"]
best = {'1': 'inblock', '4': 'inblock_m4', '128': 'inblock_m4'}
need += [f"{e['gpu_us'][M][k]['noflush']:.1f} ({k.replace('_', chr(92) + '_')}) & {e['gpu_speedup_vs_T1'][M][k]:.0f}" for M, k in best.items()]
need += [f"weak-scaling efficiency falls to {e['weak']['efficiency']['16']:.2f}", f"ridge {n['roofline']['ridge_flop_per_byte']:.0f} FLOP/byte",
         f"{n['roofline']['peak_TFLOPs']:.1f} TFLOP/s"]
q = n['quality_rise_pct']
need += [f"+{q['round-to-nearest']:.1f}\\%" .replace('+', '') , f"+{q['our GPTQ (E037)']:.2f}\\%", f"+{q['upstream GPTQ']:.2f}\\%", f"+{q['AWQ asym.']:.2f}\\%"]
pts = n['roofline']['points']
ai = lambda k, M: next(p['ai'] for p in pts if p['kernel'].startswith(k) and p['M'] == M)
need += [f"cuBLAS sits at {ai('cuBLAS', 4):.1f} FLOP/byte and Marlin at {ai('Marlin', 4):.1f}"]
missing = [s for s in need if s not in tex]
assert not missing, missing
print('report numbers ok:', len(need))
