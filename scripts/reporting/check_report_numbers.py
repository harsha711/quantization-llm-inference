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
v = json.loads((root / 'results/e043_20261117/summary.json').read_text())
assert v['decision'] == 'accepted' and len(v['nodes']) == 5
cs = [c for nd in v['nodes'].values() for c in nd['cells']]
rng = lambda f: (min(f), max(f))
def gain(k, M, pol):
    lo, hi = rng([c['gain_pct'] for c in cs if c['split'] == 'main' and c['kernel'] == k and c['M'] == M and c['policy'] == pol])
    return f"{lo:.0f}--{hi:.0f}\\%"
need += [f"this is {gain('inblock_vec', 1, 'flush')} faster at $M=1$ ({gain('inblock_vec', 1, 'noflush')} without flush)",
         f"{gain('inblock_m4_vec', 4, 'flush')} faster for the four-row kernel at $M=4$ ({gain('inblock_m4_vec', 4, 'noflush')})"]
small = [c['vs_cublas_dq'] for c in cs if c['split'] == 'main' and (c['kernel'], c['M']) in {('inblock_vec', 1), ('inblock_m4_vec', 4)}]
big = rng([c['vs_cublas_dq'] for c in cs if c['split'] == 'main' and c['kernel'] == 'inblock_m4_vec' and c['M'] == 128])
roof = [c for c in cs if c['primary'] and c['M'] == 1 and c['policy'] == 'flush']
need += [f"{1 / max(small):.1f}--{1 / min(small):.1f}$\\times$ faster than cuBLAS", f"still {big[0]:.1f}--{big[1]:.1f}$\\times$ slower",
         f"from {min(c['parent_roof_frac'] for c in roof):.2f} to {max(c['roof_frac'] for c in roof):.2f} of the modelled memory roof"]
assert min(c['gain_pct'] for c in cs) > 0  # "no regression"
missing = [s for s in need if s not in tex]
assert not missing, missing
print('report numbers ok:', len(need))
