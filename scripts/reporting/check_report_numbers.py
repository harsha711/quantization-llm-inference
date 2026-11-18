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
g = n['e042']
ty, bl, sm = g['type_rise_pct'], g['block_rise_pct'], g['summary']['test']
need += [f"(+{ty['down_proj']:.2f}\\%)", f"cost +{ty['v_proj']:.2f}\\%", f"hold {100 * g['type_param_fraction']['v_proj']:.1f}\\%",
         f"last block (+{bl[-1]:.2f}\\%)", f"block 1 (+{bl[1]:.2f}\\%)",
         f"{100 * sm['sum_single_block_rise_pct'] / sm['rtn_rise_pct']:.0f}--{100 * sm['sum_single_type_rise_pct'] / sm['rtn_rise_pct']:.0f}\\%",
         f"recovers {100 * sm['KB3_recovered_fraction']:.0f}\\%", f"recovers {100 * sm['KT1_recovered_fraction']:.0f}\\%",
         f"({100 * g['type_param_fraction']['down_proj']:.0f}\\% of the weights)"]
assert g['selection']['worst_type'] == 'down_proj' and sorted(g['selection']['worst_blocks']) == [1, 26, 27]
assert sorted(bl)[0] >= 0.1 and sorted(bl)[-4] <= 0.5 + 0.02  # "most cost 0.1--0.5%"
missing = [s for s in need if s not in tex]
assert not missing, missing
print('report numbers ok:', len(need))
