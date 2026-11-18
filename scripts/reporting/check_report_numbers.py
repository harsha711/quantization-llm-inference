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
w = json.loads((root / 'results/e044_20261117/summary.json').read_text())
assert w['decision'] == 'accepted' and len(w['nodes']) == 5
ws = [c for nd in w['nodes'].values() for c in nd['cells'] if c['split'] == 'main']
r0 = lambda x: int(x + 0.5)
def tg(M, pol, k='tc'):
    g = [c['gain_pct'] for c in ws if c['kernel'] == k and c['M'] == M and c['policy'] in pol]
    lo, hi = r0(min(g)), r0(max(g))
    return f"{lo}\\%" if lo == hi else f"{lo}--{hi}\\%"
c128 = rng([c['vs_cublas_dq'] for c in ws if c['kernel'] == 'tc' and c['M'] == 128])
big8 = [c['gain_pct'] for c in ws + [c for nd in w['nodes'].values() for c in nd['cells'] if c['split'] == 'held'] if c['kernel'] == 'tc' and c['M'] >= 8]
need += [f"it is {tg(8, ['flush'])} faster at $M=8$, {tg(32, ['flush'])} at $M=32$ and {tg(128, ['flush'])} at $M=128$ with L2 flushed",
         f"{r0(min(c['gain_pct'] for c in ws if c['kernel']=='tc' and c['M']>=8 and c['policy']=='noflush'))}--{r0(max(big8))}\\% without",
         f"{r0(min(big8))}--{r0(max(big8))}\\% faster at $M \\ge 8$ on Tensor Cores",
         f"to {c128[0]:.2f}--{c128[1]:.2f}$\\times$"]
assert all(c['vs_cublas_dq'] < 1 for c in ws if c['kernel'] == 'tc' and c['M'] <= 64 and c['policy'] == 'flush')  # "faster than cuBLAS up to M=64 flushed"
assert min(c['gain_pct'] for nd in w['nodes'].values() for c in nd['cells'] if c['kernel'] == 'tc') > 0  # "faster at every tested M"
t45 = json.loads((root / 'results/e045_20261117/summary.json').read_text())['table']
cell = lambda M, k: [r[x][k] for r in t45 if r['M'] == M for x in ('s1', 's2')]
pc = lambda v: f"{r0(100 * min(v))}--{r0(100 * max(v))}\\%"
need += [f"{pc(cell(4, 'fraction_of_marlin_speed'))} of corrected Marlin's speed at $M=4$, {pc(cell(16, 'fraction_of_marlin_speed'))} at $M=16$, "
         f"{pc(cell(64, 'fraction_of_marlin_speed'))} at $M=64$ and {pc(cell(128, 'fraction_of_marlin_speed'))} at $M=128$",
         f"{1 / max(cell(4, 'tc_over_cublas') + cell(16, 'tc_over_cublas')):.1f}--{1 / min(cell(4, 'tc_over_cublas') + cell(16, 'tc_over_cublas')):.1f}$\\times$ faster than cuBLAS",
         f"cuBLAS is {min(cell(128, 'tc_over_cublas')):.2f}--{max(cell(128, 'tc_over_cublas')):.2f}$\\times$ faster than ours"]
assert all(f < 1 for M in (4, 16, 64, 128) for f in cell(M, 'fraction_of_marlin_speed'))  # "slower than Marlin in every cell"
missing = [s for s in need if s not in tex]
assert not missing, missing
print('report numbers ok:', len(need))
