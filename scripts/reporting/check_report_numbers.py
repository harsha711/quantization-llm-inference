#!/usr/bin/env python3
"""Assert that report/main.tex quotes the numbers in the results files exactly as rounded in the text.

Sources (read-only): report/figs/numbers.json (E041, roofline from E045), results/*/summary.json (E043-E048), E047/E048 SASS counts and resources,
results/reporting/project_synthesis/matched-comparisons.csv (E001-E003, E008-E010), E001-E003 profile/SASS records,
E046 raw Nsight CSV, model-level analysis/quality JSON (E017, E023, E026, E027, E030, E032, E039, E040)."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
R = root / 'results'
tex, n = (root / 'report/main.tex').read_text(), json.loads((root / 'report/figs/numbers.json').read_text())
J = lambda p: json.loads((R / p).read_text())
r0 = lambda x: int(x + 0.5)
rng = lambda f: (min(f), max(f))
f2 = lambda lo, hi, d=2: f"{lo:.{d}f}" if f"{lo:.{d}f}" == f"{hi:.{d}f}" else f"{lo:.{d}f}--{hi:.{d}f}"
pc = lambda v: f"{r0(min(v))}\\%" if r0(min(v)) == r0(max(v)) else f"{r0(min(v))}--{r0(max(v))}\\%"
need = []

# ---- E041 (numbers.json) ----
e = n['e041']
kern = {'1': ('inblock', 'E001'), '4': ('inblock_m4', 'E003'), '128': ('inblock_m4', 'E003')}
for M, (k, lab) in kern.items():
    need += [f"{M} & {e[M]['T1_ms']:.2f} & {e[M]['speedup']['16']:.2f} & {e[M]['efficiency']['16']:.2f} & "
             f"{e['gpu_us'][M][k]['noflush']:.1f} & {lab} & {e['gpu_speedup_vs_T1'][M][k]:.0f} \\\\"]
s16 = [e[M]['speedup']['16'] for M in kern]
need += [f"weak-scaling efficiency falls to {e['weak']['efficiency']['16']:.2f}",
         f"scales {min(s16):.1f}--{max(s16):.1f}$\\times$ on 16 cores", f"At 16 threads speedup is {min(s16):.1f}--{max(s16):.1f}$\\times$",
         f"near-linearly to 8 threads ({e['1']['speedup']['8']:.2f}$\\times$ and {e['4']['speedup']['8']:.2f}$\\times$); $M=128$ reaches {e['128']['speedup']['8']:.2f}$\\times$",
         f"the measured {min(e[M]['speedup']['12'] for M in kern):.1f}--{max(e[M]['speedup']['12'] for M in kern):.1f}$\\times$ stay below",
         f"bounds $S_{{12}}$ by {64 / 6:.1f}"]
best = [e['gpu_speedup_vs_T1'][M][k] for M, (k, _) in kern.items()]
tmax = [e['gpu_speedup_vs_Tmax'][M][k] for M, (k, _) in kern.items()]
need += [f"{min(best):.0f}--{max(best):.0f}$\\times$ faster than $T_1$ and {min(tmax):.0f}--{max(tmax):.0f}$\\times$ faster than all 16 cores"]
gpu = list(csv.DictReader(open(R / 'e041_20261117/full-342501/gpu.csv')))
lb = int(next(r['logical_bytes'] for r in gpu if r['M'] == '1'))
need += [f"against {lb / 448e9 * 1e6:.1f}\\,$\\mu$s to stream the {lb / 1e6:.2f}\\,MB", f"best kernels at {e['gpu_us']['1']['inblock']['noflush']:.1f}\\,$\\mu$s ($M=1$) and {e['gpu_us']['4']['inblock_m4']['noflush']:.1f}\\,$\\mu$s ($M=4$"]
strong = list(csv.DictReader(open(R / 'e041_20261117/full-342501/strong.csv')))
weak = list(csv.DictReader(open(R / 'e041_20261117/full-342501/weak.csv')))
wb = 4096 * 4096 // 2 + 4096 // 128 * 4096 * 2  # codes + scales
need += [f"about {wb / (e['1']['Tp_ms']['16'] * 1e-3) / 1e9:.1f}\\,GB/s of modelled weight bytes at 16 threads, and its {wb / 1e6:.2f}\\,MB of codes and scales"]
need += [f"all {len(strong)} strong- and {len(weak)} weak-scaling points pass"]
assert all(r['fp64_pass'] == '1' for r in strong + weak)

# ---- E001-E003 and E008-E010 (matched-comparisons.csv) ----
mc = list(csv.DictReader(open(R / 'reporting/project_synthesis/matched-comparisons.csv')))
sel = lambda ex, **kw: [r for r in mc if r['experiment'] == ex and all(r[k] == str(v) for k, v in kw.items())]
sp = lambda rows: [float(r['speedup']) for r in rows]
def one(ex, **kw):
    rs = sel(ex, **kw)
    assert len(rs) == 1, (ex, kw, rs)
    return rs[0]
us = lambda x: f"{float(x):.1f}"
def lrow(ex, M, **kw):
    a, b = one(ex, M=M, policy='store_flush', **kw), one(ex, M=M, policy='no_flush', **kw)
    return (f"& {M} & {us(a['baseline_us'])} $\\to$ {us(a['candidate_us'])} & {us(b['baseline_us'])} $\\to$ {us(b['candidate_us'])} & "
            f"{float(a['speedup']):.2f} / {float(b['speedup']):.2f} \\\\")
need += [lrow('E001', 1), lrow('E002', 128), lrow('E003', 128, candidate='inblock_m4', series=2)]
e1 = one('E001', M=1, policy='store_flush')
need += [f"is {float(e1['speedup']):.2f}$\\times$ faster than our naive kernel at $M=1$", f"{float(e1['speedup']):.2f}$\\times$ with store flush and {float(one('E001', M=1, policy='no_flush')['speedup']):.2f}$\\times$ without"]
s8 = sp(sel('E001', M=8))
need += [f"(0.{r0(100 * float(one('E001', M=8, policy='store_flush')['speedup'])):02d}/0.{r0(100 * float(one('E001', M=8, policy='no_flush')['speedup'])):02d} at $M=8$, "
         f"{min(sp(sel('E001', M=32) + sel('E001', M=128))):.2f}--{max(sp(sel('E001', M=32) + sel('E001', M=128))):.2f} at $M=32$ and 128)"]
assert max(s8 + sp(sel('E001', M=32) + sel('E001', M=128))) < 1
e2 = sp(sel('E002', M=32) + sel('E002', M=128))
need += [f"shape specialization adds {min(e2):.2f}--{max(e2):.2f}$\\times$", f"\\emph{{Result:}} {min(e2):.2f}--{max(e2):.2f}$\\times$ at $M=32$--128; at $M=1$ "
         f"{float(one('E002', M=1, policy='store_flush')['speedup']):.2f}$\\times$ with store flush and {float(one('E002', M=1, policy='no_flush')['speedup']):.2f}$\\times$ without"]
big = lambda c: sp(sel('E003', M=32, candidate=c) + sel('E003', M=128, candidate=c))
need += [f"row reuse {min(big('inblock_m4')):.2f}--{max(big('inblock_m4')):.2f}$\\times$ at $M=32$--128",
         f"two rows {min(big('inblock_m2')):.2f}--{max(big('inblock_m2')):.2f}$\\times$ and four rows {min(big('inblock_m4')):.2f}--{max(big('inblock_m4')):.2f}$\\times$ at $M=32$--128"]
m1 = lambda c, p: one('E003', M=1, policy=p, candidate=c, series=2)  # final series
need += [f"(no flush: {float(m1('inblock_m2', 'no_flush')['baseline_us']):.1f}\\,$\\mu$s rises to {float(m1('inblock_m2', 'no_flush')['candidate_us']):.1f} and "
         f"{float(m1('inblock_m4', 'no_flush')['candidate_us']):.1f}\\,$\\mu$s; four rows with store flush {float(m1('inblock_m4', 'store_flush')['candidate_us']):.1f}\\,$\\mu$s)"]
m5 = lambda c, p: float(one('E003', M=5, policy=p, candidate=c, series=2)['speedup'])
need += [f"at $M=5$ two rows gain {m5('inblock_m2', 'store_flush'):.2f}$\\times$/{m5('inblock_m2', 'no_flush'):.2f}$\\times$, four rows gain {m5('inblock_m4', 'store_flush'):.2f}$\\times$ with store flush"]
assert m5('inblock_m4', 'no_flush') < 1
# E001-E003 profiles (cache-invalidated Nsight)
p1 = {(r['profile'], r['metric']): r['value'] for r in csv.DictReader(open(R / 'inblock_c58_20261106_final/profile_summary.csv'))}
gv = lambda k, m: float(p1[(f'{k}_M1_ncu_invalidated_metrics', m)])
need += [f"active warps {gv('naive', 'sm__warps_active.avg.pct_of_peak_sustained_active'):.1f}\\% of peak and SM throughput {gv('naive', 'sm__throughput.avg.pct_of_peak_sustained_elapsed'):.1f}\\%",
         f"shows {gv('inblock', 'launch__grid_size'):.0f} blocks and active warps rising from {gv('naive', 'sm__warps_active.avg.pct_of_peak_sustained_active'):.1f}\\% to {gv('inblock', 'sm__warps_active.avg.pct_of_peak_sustained_active'):.1f}\\%",
         f"only {gv('naive', 'launch__grid_size'):.0f} blocks for 36 SMs"]
p2 = {(r['profile'], r['metric']): r['value'] for r in csv.DictReader(open(R / 'e002_c58_20261106/profile_summary.csv'))}
ins = lambda k, M: float(p2[(f'{k}_M{M}_ncu_invalidated_metrics', 'smsp__inst_executed.sum')])
red = [100 * (1 - ins('inblock_specialized', M) / ins('inblock', M)) for M in (1, 128)]
assert f"{red[0]:.1f}" == f"{red[1]:.1f}"
ss = J('e002_c58_20261106/sass_summary.json')
need += [f"{red[0]:.1f}\\% fewer executed warp instructions at both $M=1$ and $M=128$",
         f"shrinks from {ss['E001']['static_instructions_including_padding']} to {ss['E002_aligned']['static_instructions_including_padding']} instructions and branches from "
         f"{ss['E001']['opcode_histogram']['BRA']} to {ss['E002_aligned']['opcode_histogram']['BRA']}"]
p3 = {(r['kernel'], r['M']): r for r in csv.DictReader(open(R / 'e003_c58_20261106/profile_summary.csv'))}
g3 = lambda k, f: float(p3[(k, '128')][f])
need += [f"records {g3('inblock_specialized', 'l2_bytes') / 1e6:.0f}\\,MB of L2 traffic against {g3('inblock_specialized', 'dram_read_bytes') / 1e6:.1f}\\,MB read from DRAM",
         f"L2 traffic at $M=128$ is {g3('inblock_m2', 'l2_bytes') / 1e6:.0f}\\,MB (two rows) and {g3('inblock_m4', 'l2_bytes') / 1e6:.0f}\\,MB (four rows), "
         f"{100 * (1 - g3('inblock_m2', 'l2_bytes') / g3('inblock_specialized', 'l2_bytes')):.1f}\\% and {100 * (1 - g3('inblock_m4', 'l2_bytes') / g3('inblock_specialized', 'l2_bytes')):.1f}\\% less, "
         f"executed warp instructions fall {100 * (1 - g3('inblock_m2', 'executed_warp_instructions') / g3('inblock_specialized', 'executed_warp_instructions')):.1f}\\% and "
         f"{100 * (1 - g3('inblock_m4', 'executed_warp_instructions') / g3('inblock_specialized', 'executed_warp_instructions')):.1f}\\%, and DRAM reads stay at {g3('inblock_m4', 'dram_read_bytes') / 1e6:.1f}\\,MB"]
# E008-E010
e9 = lambda base: sp(sel('E009', matrices=4, baseline=base))
need += [f"corrected Marlin is {min(e9('cublas_dq')):.2f}--{max(e9('cublas_dq')):.2f}$\\times$ faster than cuBLAS on pre-materialized FP16 $\\hat W$, and {min(e9('unfused')):.2f}--{max(e9('unfused')):.2f}$\\times$ faster than reconstructing"]
g10 = lambda K: [float(r['gain_pct']) for r in sel('E010', M=128, K=K)]
need += [f"{min(g10(4096)):.1f}--{max(g10(4096)):.1f}\\% for $K=4096,N=11008$", f"{min(g10(11008)):.1f}--{max(g10(11008)):.1f}\\% for the transpose"]
assert min(float(r['speedup']) for r in sel('E010')) > 1  # "without reversing up to 128"
g8 = lambda K, m: [float(r['gain_pct']) for r in sel('E008', K=K, matrices=m)]
need += [f"one repeated matrix ({min(g8(4096, 1)):.1f}\\% for $4096\\times11008$) did not survive cycling four matrices ({min(g8(4096, 4)):.1f}--{max(g8(4096, 4)):.1f}\\%, "
         f"and a {-max(g8(11008, 4)):.1f}--{-min(g8(11008, 4)):.1f}\\% loss on the transpose"]
assert r0(min(g8(4096, 1))) == 13 and max(g8(11008, 4)) < 0

# ---- E043 ----
v = J('e043_20261117/summary.json')
assert v['decision'] == 'accepted' and len(v['nodes']) == 5
cs = [c for nd in v['nodes'].values() for c in nd['cells']]
def gain(k, M, pol):
    lo, hi = rng([c['gain_pct'] for c in cs if c['split'] == 'main' and c['kernel'] == k and c['M'] == M and c['policy'] == pol])
    return f"{lo:.0f}--{hi:.0f}\\%"
need += [f"this is {gain('inblock_vec', 1, 'flush')} faster at $M=1$ ({gain('inblock_vec', 1, 'noflush')} without flush)",
         f"{gain('inblock_m4_vec', 4, 'flush')} faster for the four-row kernel at $M=4$ ({gain('inblock_m4_vec', 4, 'noflush')})",
         f"cut latency by {gain('inblock_vec', 1, 'flush')} at $M=1$ and {gain('inblock_m4_vec', 4, 'flush')} at $M=4$"]
small = [c['vs_cublas_dq'] for c in cs if c['split'] == 'main' and (c['kernel'], c['M']) in {('inblock_vec', 1), ('inblock_m4_vec', 4)}]
big32 = rng([c['vs_cublas_dq'] for c in cs if c['split'] == 'main' and c['kernel'] == 'inblock_m4_vec' and c['M'] >= 32])
big128 = rng([c['vs_cublas_dq'] for c in cs if c['split'] == 'main' and c['kernel'] == 'inblock_m4_vec' and c['M'] == 128])
roof = [c for c in cs if c['primary'] and c['M'] == 1 and c['policy'] == 'flush']
need += [f"{1 / max(small):.1f}--{1 / min(small):.1f}$\\times$ faster than cuBLAS", f"cuBLAS with Tensor Cores is {big32[0]:.1f}--{big32[1]:.1f}$\\times$ faster",
         f"where E043's kernel was {big128[0]:.1f}--{big128[1]:.1f}$\\times$ slower",
         f"from {min(c['parent_roof_frac'] for c in roof):.2f} to {max(c['roof_frac'] for c in roof):.2f} of the modelled memory roof"]
assert min(c['gain_pct'] for c in cs) > 0  # "no regression"
def lrange(cells, M):
    out = []
    for pol in ('flush', 'noflush'):
        q = [c for c in cells if c['M'] == M and c['policy'] == pol]
        out.append((rng([c['parent_us'] for c in q]), rng([c['cand_us'] for c in q]), rng([c['parent_us'] / c['cand_us'] for c in q])))
    (pa, ca, ra), (pb, cb, rb) = out
    return (f"& {M} & {f2(*pa, d=1)} $\\to$ {f2(*ca, d=1)} & {f2(*pb, d=1)} $\\to$ {f2(*cb, d=1)} & "
            f"{f2(*ra)} / {f2(*rb)} \\\\")
mainv = [c for c in cs if c['split'] == 'main']
need += [lrange([c for c in mainv if c['kernel'] == 'inblock_vec'], 1), lrange([c for c in mainv if c['kernel'] == 'inblock_m4_vec'], 4)]
# static LDG sites (SASS, prerequisite record)
need += ["fall from 10 to 7 (one row) and from 34 to 6 (four rows)"]

# ---- E044 ----
w = J('e044_20261117/summary.json')
assert w['decision'] == 'accepted' and len(w['nodes']) == 5
allw = [c for nd in w['nodes'].values() for c in nd['cells']]
ws = [c for c in allw if c['split'] == 'main']
def tg(M, pol, k='tc'):
    g = [c['gain_pct'] for c in ws if c['kernel'] == k and c['M'] == M and c['policy'] in pol]
    return pc(g)
c128 = rng([c['vs_cublas_dq'] for c in ws if c['kernel'] == 'tc' and c['M'] == 128])
prim = [c['gain_pct'] for c in ws if c['kernel'] == 'tc' and c['M'] in (8, 32, 128)]
need += [f"it is {tg(8, ['flush'])} faster at $M=8$, {tg(32, ['flush'])} at $M=32$ and {tg(128, ['flush'])} at $M=128$ with store flush",
         f"({pc([c['gain_pct'] for c in ws if c['kernel'] == 'tc' and c['M'] in (8, 32, 128) and c['policy'] == 'noflush'])} without)",
         f"{pc(prim)} faster than the best scalar kernel at $M=8$--128",
         f"and {c128[0]:.2f}--{c128[1]:.2f}$\\times$ slower at $M=128$"]
assert all(c['vs_cublas_dq'] < 1 for c in ws if c['kernel'] == 'tc' and c['M'] <= 64 and c['policy'] == 'flush')  # "faster up to M=64 with store flush"
assert min(c['gain_pct'] for c in allw if c['kernel'] == 'tc') > 0  # "faster at every tested M and held-out shape"
need += [lrange([c for c in ws if c['kernel'] == 'tc'], 8), lrange([c for c in ws if c['kernel'] == 'tc'], 128)]
unp = {}
for c in ws:
    if c['kernel'] == 'tc_cvt':
        t = next(d for d in ws if d['kernel'] == 'tc' and (d['M'], d['policy'], d['series'], d['parent_us']) == (c['M'], c['policy'], c['series'], c['parent_us']))
        unp.setdefault(c['M'] <= 8, []).append(c['cand_us'] / t['cand_us'])
need += [f"worth {min(unp[True]):.2f}--{max(unp[True]):.2f}$\\times$ at $M \\le 8$ and {min(unp[False]):.2f}--{max(unp[False]):.2f}$\\times$ above"]

# ---- E047, E048 (summary.json, SASS counts and resources of the c58 full job) ----
def rung(e, k):
    s = J(f'{e}_20261117/summary.json')
    assert s['decision'] == 'rejected' and len(s['nodes']) == 5
    cs = [c for nd in s['nodes'].values() for c in nd['cells']]
    pr = [c for c in cs if c['primary']]
    assert len(pr) == 60
    return cs, pr, sum(c['gain_pct'] >= 5 for c in pr)
pcu = lambda v: f"{r0(min(v))}--{r0(max(v))}\\%"
sel4 = lambda cs, M, pol, sp='main', K=4096, N=4096: [c for c in cs if c['split'] == sp and c['K'] == K and c['N'] == N and c['M'] in M and c['policy'] in pol]
c47, p47, n47 = rung('e047', 'tc_splitk')
assert n47 == 0 and all(c['gain_pct'] < 0 for c in p47)  # "slower than its parent in every primary cell"
need += [f"rejected, {n47} of 60 primary cells", f"splitting $K$ across blocks is slower in all {len(p47)} primary cells",
         f"{pcu([-c['gain_pct'] for c in p47 if c['policy'] == 'flush'])} with store flush and {pcu([-c['gain_pct'] for c in p47 if c['policy'] == 'noflush'])} without at $M=1$--16",
         f"$M=32$ with store flush ({pcu([c['gain_pct'] for c in sel4(c47, (32,), ('flush',), 'held', 11008, 4096)])})",
         f"Rejected, {n47}/60 primary cells; slower in every one."]
assert all(c['gain_pct'] > 0 for c in sel4(c47, (32,), ('flush',), 'held', 11008, 4096))
assert all(c['gain_pct'] < 5 for c in c47 if not (c['split'] == 'held' and c['K'] == 11008 and c['M'] == 32 and c['policy'] == 'flush'))  # "the only gain"
need += ["E047 & E044 $\\to$ split-$K$ " + lrange(sel4(c47, (M,), ('flush', 'noflush')), M) for M in (1, 16)]
cnt = lambda e, j, name: [tuple(map(int, l.split()[:4])) for l in open(R / f'{e}_20261117/{j}/ldg_hmma_stg_atom_counts.txt') if name in l]
assert {x[1] for x in cnt('e047', 'full-c58-343164', 'w4a16_tc_splitk_kernel')} == {16} and {x[1] for x in cnt('e047', 'full-c58-343164', 'w4a16_tc_kernel')} == {32}
need += ["holds 16 MMA instructions against the parent's 32"]
c48, p48, n48 = rung('e048', 'tc_m32')
g48 = lambda M, pol, sp='main', K=4096, N=4096: [c['gain_pct'] for c in sel4(c48, M, pol, sp, K, N)]
ho = [g for K, N in ((4096, 11008), (11008, 4096)) for g in g48((32, 64, 128), ('flush', 'noflush'), 'held', K, N)]
m128 = sel4(c48, (128,), ('flush', 'noflush'))
f128 = sel4(c48, (128,), ('flush',))
need += [f"rejected, {n48} of 60 primary cells", f"Rejected, {n48}/60 primary cells",
         f"is {f2(*rng(g48((128,), ('flush', 'noflush'))), d=1)}\\% faster at $M=128$ but up to {r0(-min(g48((32,), ('flush',))))}\\% slower at $M=32$",
         f"At $M=128$ it is {f2(*rng(g48((128,), ('flush', 'noflush'))), d=1)}\\% faster ({sum(c['gain_pct'] >= 5 for c in m128)} of {len(m128)} cells pass), "
         f"and {f2(*rng(ho), d=1)}\\% faster on the held-out rectangles at $M \\ge 32$; but with store flush it is {r0(-max(g48((32,), ('flush',))))}--{r0(-min(g48((32,), ('flush',))))}\\% slower at $M=32$ "
         f"and ties at $M=64$ (${min(g48((64,), ('flush',))):.1f}$ to ${max(g48((64,), ('flush',))):+.1f}\\%$)",
         f"from {f2(*rng([c['parent_vs_cublas_dq'] for c in f128]))}$\\times$ to {f2(*rng([c['cand_vs_cublas_dq'] for c in f128]))}$\\times$"]
assert max(g48((32,), ('flush',))) < 0 and abs(min(g48((64,), ('flush',)))) < 5 and max(g48((64,), ('flush',))) < 5
need += ["E048 & E044 $\\to$ 32-row tile " + lrange(sel4(c48, (M,), ('flush', 'noflush')), M) for M in (32, 128)]
ru = [l for l in open(R / 'e048_20261117/full-c58-343175/res-usage.txt')]
regs = lambda name: [int(ru[i + 1].split('REG:')[1].split()[0]) for i, l in enumerate(ru) if 'Function' in l and name in l]
rm, rt = regs('w4a16_tc_m32_kernel'), regs('w4a16_tc_kernel')
assert len(set(rm)) == 1 and len(rt) == 4
need += [f"needs {rm[0]} registers per thread against the parent's {min(rt)}--{max(rt)}, so {65536 // (rm[0] * 256)} instead of {65536 // (max(rt) * 256)} blocks fit per SM",
         f"$M=32$ the grid has only {4096 // 32} blocks"]

# ---- E045 (summary.json) ----
t45 = J('e045_20261117/summary.json')['table']
cell = lambda M, k: [r[x][k] for r in t45 if r['M'] == M for x in ('s1', 's2')]
pcs = lambda v: f"{r0(100 * min(v))}--{r0(100 * max(v))}\\%"
need += [f"{pcs(cell(4, 'fraction_of_marlin_speed'))} of corrected Marlin's speed at $M=4$, {pcs(cell(16, 'fraction_of_marlin_speed'))} at $M=16$, "
         f"{pcs(cell(64, 'fraction_of_marlin_speed'))} at $M=64$ and {pcs(cell(128, 'fraction_of_marlin_speed'))} at $M=128$",
         f"{1 / max(cell(4, 'tc_over_cublas') + cell(16, 'tc_over_cublas')):.1f}--{1 / min(cell(4, 'tc_over_cublas') + cell(16, 'tc_over_cublas')):.1f}$\\times$ faster than cuBLAS",
         f"cuBLAS is {min(cell(128, 'tc_over_cublas')):.2f}--{max(cell(128, 'tc_over_cublas')):.2f}$\\times$ faster than ours"]
hi = cell(64, 'fraction_of_marlin_speed') + cell(128, 'fraction_of_marlin_speed')
need += [f"{pcs(cell(4, 'fraction_of_marlin_speed'))} of corrected Marlin's speed at $M=4$ but only {pcs(hi)} at $M \\ge 64$",
         f"{pcs(cell(4, 'fraction_of_marlin_speed'))} of corrected Marlin's speed at $M=4$ and {pcs(hi)} at $M \\ge 64$, and it is slower",
         f"Descriptive: {pcs([f for M in (4, 16, 64, 128) for f in cell(M, 'fraction_of_marlin_speed')])} of Marlin's speed"]
assert all(f < 1 for M in (4, 16, 64, 128) for f in cell(M, 'fraction_of_marlin_speed'))  # "slower than Marlin in all 16 cell-series"
assert len(t45) * 2 == 16
for r in t45:
    s1, s2 = r['s1'], r['s2']
    need += [f"{r['K']}$\\times${r['N']} & {r['M']} & {s1['tc_us']:.1f} / {s2['tc_us']:.1f} & {s1['marlin_us']:.1f} / {s2['marlin_us']:.1f} & "
             f"{s1['cublas_us']:.1f} / {s2['cublas_us']:.1f} & {s1['fraction_of_marlin_speed']:.2f} / {s2['fraction_of_marlin_speed']:.2f} \\\\"]
clk = [int(list(csv.reader(open(R / f'e045_20261117/full-c7-343147/series{s}-clocks-after.csv')))[1][5].split()[0]) for s in (1, 2)]
need += [f"the GPU ran at {min(clk)}--{max(clk)}\\,MHz after each E045 series"]

# ---- E046 (summary.json, raw Nsight CSV) ----
p46 = J('e046_20261117/summary.json')
SLOTS = 36 * 3
top = lambda d: max(d['stalls_pct'].items(), key=lambda kv: kv[1])
sname = {'long_sb': 'long scoreboard', 'math': 'math pipe'}
for lab, key, waves in (('ours (tc), 1', 'tc_M1_4096x4096', True), ('ours (tc), 16', 'tc_M16_4096x4096', True), ('ours (tc), 128', 'tc_M128_4096x4096', True),
                        ('E043 four-row, 1', 'inblock_m4_vec_M1_4096x4096', False), ('cuBLAS, 1', 'cublas_dq_M1_4096x4096', False),
                        ('cuBLAS, 128', 'cublas_dq_M128_4096x4096', False)):
    d = p46[key]
    st, sv = top(d)
    need += [f"{lab} & {d['dur_us']:.1f} & {d['dram_pct']:.1f} & {d['hmma_pct']:.1f} & {d['grid'] / SLOTS:.2f} & {sname.get(st, st)} {r0(sv)}\\% \\\\" if waves else
             f"{lab} & {d['dur_us']:.1f} & {d['dram_pct']:.1f} & {d['hmma_pct']:.1f} & -- & {sname.get(st, st)} {r0(sv)}\\% \\\\"]
t1, t128, c128p = p46['tc_M1_4096x4096'], p46['tc_M128_4096x4096'], p46['cublas_dq_M128_4096x4096']
assert t1['dram_pct'] >= 60 and t128['hmma_pct'] >= 30  # both hypotheses rejected by their own thresholds
assert t128['ops']['HMMA'] == c128p['ops']['HMMA'] == 1048576
busy = (t128['hmma_pct'] / 100 * t128['dur_us'], c128p['hmma_pct'] / 100 * c128p['dur_us'])
assert r0(busy[0]) == r0(busy[1])
qw = t128['ldg']['LDG.E.CONSTANT']['l2_sec'] * 32 / (4096 * 4096 / 2)
assert r0(qw) == 8 and t1['ldg']['LDG.E.CONSTANT']['l2_excess'] == 0
need += [f"DRAM is already at {r0(t1['dram_pct'])}\\% of peak", f"the tensor pipe is {r0(t128['hmma_pct'])}\\% busy and executes exactly cuBLAS's 1,048,576 MMA instructions in the same tensor-pipe busy time ({r0(busy[0])}\\,$\\mu$s)",
         f"reads DRAM at {r0(t1['dram_pct'])}\\% of peak but fills only {t1['grid'] / SLOTS:.2f} waves", f"its tensor pipe is {r0(t128['hmma_pct'])}\\% busy",
         f"128 blocks fill only {t1['grid'] / SLOTS:.2f} waves", f"weight traffic through L2 is indeed eight times the code bytes",
         f"DRAM at {p46['inblock_m4_vec_M1_4096x4096']['dram_pct']:.1f}\\% of peak with long-scoreboard the top stall ({r0(p46['inblock_m4_vec_M1_4096x4096']['stalls_pct']['long_sb'])}\\% of samples)"]
assert t1['grid'] == 128
rect = p46['tc_M4_K4096xN11008']
need += [f"at {rect['grid'] / SLOTS:.2f} waves, reach {r0(rect['dram_pct'])}\\% of DRAM peak"]
need += [f"our kernel's {t1['grid']:.0f} blocks fill {t1['grid'] / SLOTS:.2f} waves, and the same weight bytes at {rect['grid'] / SLOTS:.2f} waves",
         f"the tensor pipe is already {r0(t128['hmma_pct'])}\\% busy"]  # E047/E048 evidence
assert 4096 // 64 // 4 // 8 == 2  # E047: "each warp keeps only two 64-wide chunks"
raw = list(csv.reader(open(R / 'e046_20261117/full-c58-343125/ncu/tc_M1_4096x4096.raw.csv')))
col = lambda name: float(raw[-1][raw[0].index(name)].replace(',', ''))
el = col('sm__cycles_elapsed.avg')
need += [f"active on average {r0(100 * col('sm__cycles_active.avg') / el)}\\% of the elapsed cycles (the least busy {r0(100 * col('sm__cycles_active.min') / el)}\\%)"]
assert raw[-1][raw[0].index('launch__waves_per_multiprocessor')] == f"{t1['grid'] / SLOTS:.2f}"

# ---- E049 (summary.json) ----
s49 = J('e049_20261117/summary.json')
p49, v49, t49 = s49['profiles'], list(s49['verdicts'].values()), s49['timing']
for shp in ('K4096_N11008', 'K11008_N4096'):
    for M in (4, 64):
        for impl, lab in (('tc', 'ours (tc)'), ('marlin', 'Marlin')):
            d = p49[f'{impl}_M{M}_{shp}']
            need += [f"{lab}, {M} & {d['dur_us']:.1f} & {d['dram_rd_over_unique']:.2f} & {d['dram_pct']:.1f} & {d['hmma_pct']:.1f} & "
                     f"{d['inst'] / 1e6:.2f} & {r0(d['stall_share']['long_scoreboard'])} \\\\"]
P49 = lambda impl, M: [d for k, d in p49.items() if k.startswith(f'{impl}_M{M}_')]
t64, m64 = P49('tc', 64), P49('marlin', 64)
assert [v[x]['verdict'] for v in v49 for x in 'abc'] == ['REJECTED', 'PARTIAL', 'CONFIRMED'] * 2
assert {d['hmma'] for d in t64 + m64} == {1409024} and all(d['occ'] > 0.4 * 100 for d in t64)
ir = [v['c']['inst_ratio'] for v in v49]
assert {f"{x:.1f}" for x in ir} == {'5.2'}
oth = lambda ds: [(d['inst'] - d['hmma']) / d['hmma'] for d in ds]
need += [f"we read {f2(*rng([d['dram_rd_over_unique'] for d in t64]))}\\,$U$ from DRAM (Marlin {f2(*rng([d['dram_rd_over_unique'] for d in m64]))}\\,$U$)",
         f"hit L2 ({f2(*rng([d['l2_tex_hit_pct'] for d in t64]), d=1)}\\% of SM reads), and the extra DRAM bytes explain {f2(*rng([100 * v['a']['share'] for v in v49]), d=1)}\\% of the profiled gap",
         f"our long-scoreboard share of {pc([d['stall_share']['long_scoreboard'] for d in t64])} is more than ten times Marlin's {pc([d['stall_share']['long_scoreboard'] for d in m64])}",
         f"same 1,409,024 MMA instructions, but we issue 5.2$\\times$ Marlin's warp instructions ({f2(*rng(oth(t64)), d=1)} against {f2(*rng(oth(m64)), d=1)} other instructions per MMA) at {pc([d['issue_pct'] for d in t64])} issue active",
         f"we keep {pc([d['occ'] for d in t64])} of warp slots resident against Marlin's {pc([d['occ'] for d in m64])}",
         f"Marlin is tensor-pipe bound ({pc([d['hmma_pct'] for d in m64])} busy, math-pipe throttle {pc([d['stall_share']['math_pipe_throttle'] for d in m64])} of stalls); ours is {pc([d['hmma_pct'] for d in t64])} busy",
         f"DRAM-bound (ours at {pcu([d['dram_pct'] for d in P49('tc', 4)])}, Marlin at {pcu([d['dram_pct'] for d in P49('marlin', 4)])} of peak), and on $11008\\times4096$ our {p49['tc_M4_K11008_N4096']['grid']:.0f} blocks fill only {p49['tc_M4_K11008_N4096']['waves']:.2f} waves",
         "5.2$\\times$ Marlin's instructions"]
assert all(v['b']['tc_long_sb_share'] < 30 and v['b']['tc_long_sb_share'] > 10 * v['b']['marlin_long_sb_share'] for v in v49)
assert all(d['dram_rd_over_unique'] < 1.2 for d in t64) and all(d['dram_rd_over_unique'] > 0.99 for d in P49('tc', 4) + P49('marlin', 4))
T49 = lambda M, k: [r[s][k] for r in t49 if r['M'] in M for s in ('s1', 's2')]
need += [f"reaches {f2(*rng(T49((64,), 'tc_m32_fraction_of_marlin_speed')))} of Marlin's speed at $M=64$ and {f2(*rng(T49((128,), 'tc_m32_fraction_of_marlin_speed')))} at $M=128$ "
         f"(ours: {f2(*rng(T49((64, 128), 'tc_fraction_of_marlin_speed')))}), {f2(*rng([100 * (1 - x) for x in T49((64, 128), 'tc_m32_over_tc')]), d=1)}\\% faster than our kernel there"]
assert all(f < 1 for M in (4, 16, 64, 128) for k in ('tc', 'tc_m32') for f in T49((M,), f'{k}_fraction_of_marlin_speed'))
drift = max(abs(x[s][k] / y[s][k2] - 1) for x, y in zip(t49, t45) for s in ('s1', 's2')
            for k, k2 in (('tc_us', 'tc_us'), ('marlin_us', 'marlin_us'), ('cublas_dq_us', 'cublas_us')) if (x['K'], x['N'], x['M']) == (y['K'], y['N'], y['M']))
assert drift < 0.01  # "reproduce E045's medians within 1%"
need += ["reproduce E045's medians within 1\\%"]

# ---- Roofline (numbers.json, from E045) ----
ro = n['roofline']
assert ro['source'] == 'E045'
need += [f"ridge {ro['ridge_flop_per_byte']:.0f} FLOP/byte", f"{ro['peak_TFLOPs']:.1f}\\,TFLOP/s", f"{ro['bandwidth_GBps']:.0f}\\,GB/s"]
pts = ro['points']
P = lambda k, M: [p for p in pts if p['kernel'].startswith(k) and p['M'] in M]
ai = lambda k, M: P(k, (M,))[0]['ai']
fr = lambda k, M: pc([100 * p['frac_roof'] for p in P(k, M)])
need += [f"cuBLAS sits at {ai('cuBLAS', 4):.1f} FLOP/byte and Marlin at {ai('Marlin', 4):.1f}",
         f"at {fr('cuBLAS', (4,))} and {fr('Marlin', (4,))} of the memory roof respectively; our kernel shares Marlin's intensity and reaches {fr('ours', (4,))}",
         f"the intensity ratio is {ai('Marlin', 4) / ai('cuBLAS', 4):.1f}",
         f"Marlin at $M=4$ ({f2(*rng([p['tflops'] for p in P('Marlin', (4,))]))}\\,TFLOP/s) matches cuBLAS at $M=16$ ({f2(*rng([p['tflops'] for p in P('cuBLAS', (16,))]))}\\,TFLOP/s)",
         f"Marlin ({ai('Marlin', 64):.0f}--{ai('Marlin', 128):.0f} FLOP/byte) is right of the ridge at {fr('Marlin', (64, 128))} of the nominal compute roof, "
         f"cuBLAS at $M=128$ ({ai('cuBLAS', 128):.0f} FLOP/byte) at {fr('cuBLAS', (128,))}, and our kernel at {fr('ours', (64, 128))}"]
assert all(p['frac_roof'] <= 1.0001 for p in pts) and len(pts) == 48  # "No point exceeds either roof"
assert ai('cuBLAS', 4) < ro['ridge_flop_per_byte'] and ai('Marlin', 4) < ro['ridge_flop_per_byte'] < ai('Marlin', 64)

# ---- Model-level background ----
ef = J('e017_20261109/independent-model-failure.json')
need += [f"E017 ({ef['bad_logits']} of {ef['total_logits']:,} logits outside an elementwise bound, all 64 tokens identical)"]
assert ef['greedy_tokens_equal']
q32 = J('e032_20261110/analysis.json')['perplexity_rise_pct']
need += [f"cost {q32['blocks']:.1f}\\% WikiText-2~\\cite{{wikitext}} perplexity over the FP16 original", f"costs a further {q32['head_on_quantized_blocks']:.1f}\\% perplexity"]
e23 = J('e023_20261110/time-334106/analysis.json')
g23 = lambda ctx: [s['decode_gain_pct']['GQ_vs_Q'] for s in e23[ctx]['series']]
need += [f"{pc(g23('128'))} faster than it at context 128 (passes) but {pc(g23('2048'))} at 2048 (misses)"]
c26 = J('e026_20261110/full-c7-334151/analysis.json')['cells'] + J('e026_20261110/full-c9-334177/analysis.json')['cells']
c7 = J('e026_20261110/full-c7-334151/analysis.json')['cells']
gq = lambda ctx: [c['decode_ms']['GQA'] for c in c7 if c['context'] == ctx]
ga = lambda ctx: [c['gain_pct']['GQA_vs_GRA'] for c in c7 if c['context'] == ctx]
need += [f"gives about {r0(min(gq(128)))} and {r0(min(gq(2048)))}\\,ms per 63 tokens at contexts 128 and 2048, {f2(*rng(ga(128)), d=1)}\\%/{f2(*rng(ga(2048)), d=1)}\\% below"]
assert max(gq(128)) - min(gq(128)) < 1 and max(gq(2048)) - min(gq(2048)) < 1  # "about"
gr = lambda ctx: [c['gain_pct']['GQA_vs_GR'] for c in c7 if c['context'] == ctx]
need += [f"and {f2(*rng(gr(128)), d=1)}\\%/{f2(*rng(gr(2048)), d=1)}\\% below graph-replayed FP16 with stock attention"]
a128 = [c['gain_pct']['GQA_vs_GQ'] for c in c26 if c['context'] == 128]
assert len(a128) == 8 and sum(g < 5 for g in a128) == 2
need += ["about 5\\% at context 128, below the bar in two of eight cells"]
c27 = J('e027_20261110/full-c9-334243-full/analysis.json')['cells']
dv = lambda a: rng([c['decode_ms'][a] for c in c27])
need += [f"decodes in {dv('V')[0]:.1f}--{dv('V')[1]:.1f}\\,ms against our {dv('GQA')[0]:.1f}--{dv('GQA')[1]:.1f}\\,ms, "
         f"{r0(-max(c['gain_pct']['GQA_vs_V'] for c in c27))}--{r0(-min(c['gain_pct']['GQA_vs_V'] for c in c27))}\\% slower at contexts 128--2048"]
ours, vl = J('e030_20261110/profile-c51-334252-p2/prof/ours.json'), J('e030_20261110/profile-c51-334252-p2/prof/vllm.json')
gap = [(ours[c]['device_us_per_step'] - vl[c]['device_us_per_step']) / 1e3 for c in ours]
mar = lambda d, c: [v for k, v in d[c]['kernel_us_and_calls_per_step'].items() if 'arlin' in k]
oth = lambda d, c: sum(v[1] for k, v in d[c]['kernel_us_and_calls_per_step'].items() if 'arlin' not in k)
mgap = [sum(v[0] for v in mar(ours, c)) - sum(v[0] for v in mar(vl, c)) for c in ours]
assert all(f"{x / 1e3:.1f}" == "0.4" for x in mgap)
assert all(sum(v[1] for v in mar(vl, c)) == 112 and sum(v[1] for v in mar(ours, c)) == 196 and oth(ours, c) / oth(vl, c) > 4 for c in ours)
need += [f"gap of {min(gap):.1f}--{max(gap):.1f}\\,ms, about 0.4\\,ms of it inside Marlin projections; vLLM issues 112 Marlin calls per step against our 196 and over four times fewer"]
q39 = {a: v['test']['perplexity'] for a, v in J('e039_20261116/full-342265/quality.json')['arms'].items()}
q40 = {a: v['test']['perplexity'] for a, v in J('e040_20261116/full-342317/quality.json')['arms'].items()}
rise = lambda q, a: 100 * (q[a] / q['O'] - 1)
aw = [rise(q40, a) for a in ('AWQs', 'AWQa', 'AWQG')]
need += [f"lowers the block cost to {rise(q39, 'AT15'):.2f}\\% (E037)", f"GPTQ costs {rise(q39, 'L'):.2f}\\% and our implementation on the standard 16-level grid {rise(q39, 'AT16'):.2f}\\%",
         f"costs {min(aw):.2f}--{max(aw):.2f}\\% across three tested recipes"]

missing = [s for s in need if s not in tex]
assert not missing, '\n'.join(missing)
print('report numbers ok:', len(need))
