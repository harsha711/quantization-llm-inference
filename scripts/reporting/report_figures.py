#!/usr/bin/env python3
"""Figures and numbers for the ACM report (report/figs/). report_figures.py E041_RUN_DIR

Inputs are read-only: E041 CSVs, E045 summary.json (roofline: tc, corrected Marlin, cuBLAS), E027 analysis,
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

# ---- Roofline from E045 (four cycled matrices, amortized per GEMM, series 1 and 2, one job/node c7):
# our Tensor Core kernel tc, corrected Marlin A and cuBLAS on pre-materialized FP16 W_hat (math mode 16) ----
BW, PEAK = 448e9, 36 * 2.572e9 * 512
num['roofline'] = dict(bandwidth_GBps=BW / 1e9, peak_TFLOPs=PEAK / 1e12, ridge_flop_per_byte=PEAK / BW, source='E045')
pts = []
for r in json.load(open(ROOT / 'results/e045_20261117/summary.json'))['table']:
    M, N, K = r['M'], r['N'], r['K']
    fl = 2 * M * N * K
    w4 = K * N // 2 + (K // 128) * N * 2 + M * K * 2 + M * N * 2  # codes + FP16 scales + A + C, each once
    by = {'cuBLAS FP16 $\\hat W$': (K * N * 2 + M * K * 2 + M * N * 2, 'cublas_us'),
          'Marlin W4A16 (corrected)': (w4, 'marlin_us'), 'ours: tc (E044)': (w4, 'tc_us')}
    for s in (1, 2):
        for name, (b, key) in by.items():
            t = r[f's{s}'][key]
            ai, perf = fl / b, fl / (t * 1e-6)
            roof = min(PEAK, ai * BW)
            assert perf <= roof * 1.0001, ('above roof', r)
            pts.append(dict(kernel=name, M=M, K=K, N=N, series=s, ai=ai, tflops=perf / 1e12, frac_roof=perf / roof))
num['roofline']['points'] = pts
fig, ax = plt.subplots(figsize=(W1, 2.6))
xs = [v / 10 for v in range(5, 10001)]
ax.plot(xs, [min(PEAK, v * BW) / 1e12 for v in xs], color='k', lw=1)
ax.axvline(PEAK / BW, ls=':', color=C[4], lw=0.8)
ax.text(PEAK / BW * 1.08, 4, f'ridge {PEAK / BW:.0f}', color=C[4], fontsize=6.5)
for i, name in enumerate(dict.fromkeys(p['kernel'] for p in pts)):
    for mk, (K, N) in zip('os', ((4096, 11008), (11008, 4096))):
        q = [p for p in pts if p['kernel'] == name and p['K'] == K and p['series'] == 1]
        ax.scatter([p['ai'] for p in q], [p['tflops'] for p in q], marker=mk, s=14, color=C[i], zorder=3,
                   facecolors='none' if mk == 's' else C[i], label=f'{name}, K{K}×N{N}')
        if mk == 'o' and i < 2:  # label M once per intensity: cuBLAS left of its point, Marlin above (tc shares Marlin's x)
            for p in q:
                ax.annotate(f"M{p['M']}", (p['ai'], p['tflops']), fontsize=5.5, color=C[i],
                            xytext=(5, -7) if i == 0 else (-14, 4), textcoords='offset points')
ax.set(xscale='log', yscale='log', xlabel='arithmetic intensity (FLOP/byte, modelled)', ylabel='TFLOP/s', xlim=(0.5, 1000), ylim=(0.05, 80))
ax.legend(frameon=False, fontsize=5, loc='lower right')
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
E032_RTN = q['round-to-nearest']
num['quality_rise_pct'] = {k: 100 * (v / q['FP16 original'] - 1) for k, v in q.items()}
fig, ax = plt.subplots(figsize=(W1, 1.9))
ax.barh(range(len(q))[::-1], list(q.values()), color=[C[4]] + [C[1]] + [C[0]] * 2 + [C[2]] * 2)
for i, (k, v) in enumerate(q.items()):
    ax.text(v + 0.03, len(q) - 1 - i, f'{v:.2f}' + ('' if i == 0 else f'  (+{num["quality_rise_pct"][k]:.1f}%)'), va='center', fontsize=6)
ax.axvline(q['FP16 original'] * 1.05, ls='--', color='k', lw=0.8)
ax.text(q['FP16 original'] * 1.05 + 0.02, len(q) - 0.6, '+5% bar', fontsize=6)
ax.set(yticks=range(len(q))[::-1], yticklabels=list(q), xlabel='WikiText-2 test perplexity (lower is better)', xlim=(8.3, 10.6))
save(fig, 'quality')

# ---- E042: where the plain-rounding block cost sits (test split; selection was on validation) ----
s = json.load(open(ROOT / 'results/e042_20261117/full-342883/quality.json'))
rp, fr = s['rise_pct'], {a: v['quantized_param_fraction'] for a, v in s['arms'].items()}
assert abs(s['arms']['RTN']['test']['perplexity'] / E032_RTN - 1) < 1e-5
types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
nblk = sum(a.startswith('B:') for a in rp)
num['e042'] = dict(selection=s['selection'], summary=s['summary'], type_rise_pct={t: rp[f'T:{t}']['test'] for t in types},
                   type_param_fraction={t: fr[f'T:{t}'] for t in types}, block_rise_pct=[rp[f'B:{b}']['test'] for b in range(nblk)])
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 3.2), gridspec_kw=dict(height_ratios=(1, 1.1)))
a1.bar(range(7), [rp[f'T:{t}']['test'] for t in types], color=C[0])
for i, t in enumerate(types):
    a1.text(i, rp[f'T:{t}']['test'] + 0.1, f"{100 * fr[f'T:{t}']:.1f}%", ha='center', fontsize=5.5, color=C[4])
a1.set(xticks=range(7), xticklabels=[t[:-5] for t in types], ylabel='perplexity rise (%)', ylim=(0, 4.4))
a1.set_title('(a) one projection type, all blocks (label: share of weights)', fontsize=7)
a2.bar(range(nblk), [rp[f'B:{b}']['test'] for b in range(nblk)], color=[C[1] if b in s['selection']['worst_blocks'] else C[0] for b in range(nblk)])
a2.set(xlabel='block', ylabel='perplexity rise (%)', xticks=range(0, nblk, 3), xlim=(-0.8, nblk - 0.2))
a2.set_title('(b) one block, all types (orange: kept FP16 in KB3)', fontsize=7)
fig.subplots_adjust(hspace=0.55)
save(fig, 'e042_sensitivity')

(FIGS / 'numbers.json').write_text(json.dumps(num, indent=1, default=str))
print('ok', sorted(p.name for p in FIGS.iterdir()))
