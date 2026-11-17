#!/usr/bin/env python3
"""Figures and numbers for the ACM report (report/figs/). report_figures.py E041_RUN_DIR

Inputs are read-only: E041 CSVs, the project-synthesis matched-comparison table (E010 rows), E027 analysis,
E032/E039/E040 quality JSON. Writes report/figs/*.pdf and report/figs/numbers.json (every number the report quotes
from these figures). Roofline ceilings: 448 GB/s GDDR7 (NVIDIA RTX 5060 Ti specification) and 47.4 TFLOP/s dense
FP16 Tensor with FP32 accumulation (36 SM x 2.572 GHz boost x 512 FLOP/clk/SM, third-party listing); traffic is the
CONTRACT byte model (each operand once), not measured DRAM.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / 'report/figs'
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'figure.dpi': 150, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
W1 = 3.33  # one ACM column, inches
C = ['#1f5fa8', '#d1612a', '#3a8f4b', '#8a4fa3', '#7a7a7a']
num = {}


def save(fig, name):
    fig.savefig(FIGS / f'{name}.pdf', metadata={'CreationDate': None})
    plt.close(fig)


# ---- E041: serial baseline, strong and weak scaling ----
run = Path(sys.argv[1])
rows = lambda f: list(csv.DictReader(open(run / f)))
strong, weak, gpu = rows('strong.csv'), rows('weak.csv'), rows('gpu.csv')
assert all(r['fp64_pass'] == '1' and r['num_bad'] == '0' for r in strong + weak), 'FP64 check failed'
assert all(r['pass'] == '1' for r in gpu), 'GPU check failed'
Ms = sorted({int(r['M']) for r in strong})
ps = sorted({int(r['threads']) for r in strong})
e041 = {}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1 * 2, 2.1))
for i, M in enumerate(Ms):
    rs = {int(r['threads']): r for r in strong if int(r['M']) == M}
    assert sorted(rs) == ps and len({r['fnv1a'] for r in rs.values()}) == 1, ('incomplete or hash differs', M)
    t1 = float(rs[1]['median_ms'])
    sp = {p: t1 / float(rs[p]['median_ms']) for p in ps}
    e041[M] = dict(T1_ms=t1, Tp_ms={p: float(rs[p]['median_ms']) for p in ps}, speedup=sp, efficiency={p: sp[p] / p for p in ps})
    a1.plot(ps, [sp[p] for p in ps], 'o-', color=C[i], ms=3, label=f'M={M}')
a1.plot(ps, ps, '--', color=C[4], lw=1, label='ideal')
a1.set(xlabel='OpenMP threads p', ylabel='speedup $T_1/T_p$', xticks=ps, title='(a) strong scaling, K=N=4096')
a1.legend(frameon=False)
if weak:
    wr = {int(r['threads']): r for r in weak}
    assert sorted(wr) == ps
    w1 = float(wr[1]['median_ms'])
    eff = {p: w1 / float(wr[p]['median_ms']) for p in ps}
    e041['weak'] = dict(T_ms={p: float(wr[p]['median_ms']) for p in ps}, efficiency=eff)
    a2.plot(ps, [eff[p] for p in ps], 'o-', color=C[1], ms=3, label='measured')
    a2.axhline(1, ls='--', color=C[4], lw=1, label='ideal')
    a2.set(xlabel='OpenMP threads p (N = 512p)', ylabel='weak efficiency $T_1/T_p$', xticks=ps, ylim=(0, 1.1),
           title='(b) weak scaling, M=4, K=4096')
    a2.legend(frameon=False)
save(fig, 'e041_scaling')
# GPU kernels on the same node, no-flush medians, against the CPU at p=1 and p=max
g = {}
for r in gpu:
    g.setdefault(int(r['M']), {}).setdefault(r['kernel'], {})['noflush' if r['flush_l2'] == '0' else 'flush'] = float(r['median_us'])
e041['gpu_us'] = g
e041['gpu_speedup_vs_T1'] = {M: {k: e041[M]['T1_ms'] * 1e3 / v['noflush'] for k, v in g[M].items()} for M in g if M in e041}
e041['gpu_speedup_vs_Tmax'] = {M: {k: e041[M]['Tp_ms'][ps[-1]] * 1e3 / v['noflush'] for k, v in g[M].items()} for M in g if M in e041}
num['e041'] = e041

# ---- Roofline from E010 (four cycled matrices, amortized per GEMM, series 1 and 2) ----
BW, PEAK = 448e9, 36 * 2.572e9 * 512
num['roofline'] = dict(bandwidth_GBps=BW / 1e9, peak_TFLOPs=PEAK / 1e12, ridge_flop_per_byte=PEAK / BW)
pts = []
for r in csv.DictReader(open(ROOT / 'results/reporting/project_synthesis/matched-comparisons.csv')):
    if r['experiment'] != 'E010':
        continue
    M, N, K = int(r['M']), int(r['N']), int(r['K'])
    fl = 2 * M * N * K
    by = {'cuBLAS FP16 $\\hat W$': K * N * 2 + M * K * 2 + M * N * 2,
          'Marlin W4A16': K * N // 2 + (K // 128) * N * 2 + M * K * 2 + M * N * 2}
    for (name, b), t in zip(by.items(), (float(r['baseline_amortized_us']), float(r['candidate_amortized_us']))):
        ai, perf = fl / b, fl / (t * 1e-6)
        roof = min(PEAK, ai * BW)
        assert perf <= roof * 1.0001, ('above roof', r)
        pts.append(dict(kernel=name, M=M, K=K, N=N, series=int(r['series']), ai=ai, tflops=perf / 1e12, frac_roof=perf / roof))
num['roofline']['points'] = pts
fig, ax = plt.subplots(figsize=(W1, 2.5))
xs = [v / 10 for v in range(5, 10001)]
ax.plot(xs, [min(PEAK, v * BW) / 1e12 for v in xs], color='k', lw=1)
ax.axvline(PEAK / BW, ls=':', color=C[4], lw=0.8)
ax.text(PEAK / BW * 1.08, 0.08, f'ridge {PEAK / BW:.0f}', color=C[4], fontsize=6.5)
for i, name in enumerate(dict.fromkeys(p['kernel'] for p in pts)):
    for mk, (K, N) in zip('os', ((4096, 11008), (11008, 4096))):
        q = [p for p in pts if p['kernel'] == name and p['K'] == K and p['series'] == 1]
        ax.scatter([p['ai'] for p in q], [p['tflops'] for p in q], marker=mk, s=14, color=C[i], zorder=3,
                   label=f'{name}, K{K}×N{N}')
        for p in q:
            ax.annotate(f"{p['M']}", (p['ai'], p['tflops']), fontsize=5.5, xytext=(3, -6), textcoords='offset points')
ax.set(xscale='log', yscale='log', xlabel='arithmetic intensity (FLOP/byte, modelled)', ylabel='TFLOP/s', xlim=(0.5, 1000), ylim=(0.05, 80))
ax.legend(frameon=False, fontsize=5.5, loc='upper left')
save(fig, 'roofline')

# ---- Full model decode (E027, c9, 63 decode steps, batch 1) ----
cells = json.load(open(ROOT / 'results/e027_20261110/full-c9-334243-full/analysis.json'))['cells']
arms = ['GR', 'GQ', 'GQA', 'V']
dec = {ctx: {a: statistics.median(c['decode_ms'][a] for c in cells if c['context'] == ctx) for a in arms} for ctx in (128, 512, 2048)}
num['e027_decode_ms_median_of_cells'] = dec
fig, ax = plt.subplots(figsize=(W1, 1.9))
lab = {'GR': 'FP16 (graph)', 'GQ': 'ours: Marlin (graph)', 'GQA': 'ours: + decode attention', 'V': 'vLLM, same 4-bit weights'}
for i, a in enumerate(arms):
    ax.bar([j + (i - 1.5) * 0.2 for j in range(3)], [dec[c][a] for c in dec], 0.2, color=C[i], label=lab[a])
ax.set(xticks=range(3), xticklabels=[f'context {c}' for c in dec], ylabel='decode, 63 tokens (ms)')
ax.legend(frameon=False, fontsize=6, ncol=2, loc='upper left')
ax.set_ylim(0, 1150)
save(fig, 'model_decode')

# ---- Quality (WikiText-2 test perplexity, FP16 head) ----
e039 = json.load(open(ROOT / 'results/e039_20261116/full-342265/quality.json'))['arms']
e040 = json.load(open(ROOT / 'results/e040_20261116/full-342317/quality.json'))['arms']
ppl = lambda d, a: d[a]['test']['perplexity']
q = {'FP16 original': ppl(e039, 'O'), 'round-to-nearest': ppl(e039, 'RTN'), 'our GPTQ (E037)': ppl(e039, 'AT15'),
     'upstream GPTQ': ppl(e039, 'L'), 'AWQ+GPTQ': ppl(e040, 'AWQG'), 'AWQ asym.': ppl(e040, 'AWQa')}
num['quality_test_ppl'] = q
num['quality_rise_pct'] = {k: 100 * (v / q['FP16 original'] - 1) for k, v in q.items()}
fig, ax = plt.subplots(figsize=(W1, 1.9))
ax.barh(range(len(q))[::-1], list(q.values()), color=[C[4]] + [C[1]] + [C[0]] * 2 + [C[2]] * 2)
for i, (k, v) in enumerate(q.items()):
    ax.text(v + 0.03, len(q) - 1 - i, f'{v:.2f}' + ('' if i == 0 else f'  (+{num["quality_rise_pct"][k]:.1f}%)'), va='center', fontsize=6)
ax.axvline(q['FP16 original'] * 1.05, ls='--', color='k', lw=0.8)
ax.text(q['FP16 original'] * 1.05 + 0.02, len(q) - 0.6, '+5% bar', fontsize=6)
ax.set(yticks=range(len(q))[::-1], yticklabels=list(q), xlabel='WikiText-2 test perplexity (lower is better)', xlim=(8.3, 10.6))
save(fig, 'quality')

(FIGS / 'numbers.json').write_text(json.dumps(num, indent=1, default=str))
print('ok', sorted(p.name for p in FIGS.iterdir()))
