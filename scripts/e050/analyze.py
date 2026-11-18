#!/usr/bin/env python3
"""E050 analyzer. analyze.py DIR --stage prereq|full [--out OUTDIR]; analyze.py --self-test.

prereq: build/SASS/resource identity (same-job bench and E049's library), agreement, traced validation,
sanitizers. full: both series, the table (per-GEMM us, fraction of Marlin speed) and the predeclared
decision: tc_ladder median <= 0.95 x tc median at M 64 and 128, all three shapes, both series (12 cells)."""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

SHAPES = [(4096,11008),(11008,4096),(4096,4096)]
MS = [4,16,64,128]
PRIMARY_MS = [64,128]
NAMES = ['marlin','tc','tc_ladder','cublas_dq']


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def ok(d,label):
    assert (d/f'{label}.exit').read_text().strip() == '0', label


def check_validation(row,key='validation'):
    checks = row[key]
    assert len(checks) == 4, row['config']
    for c in checks:
        assert c['num_bad'] == 0 and c['checked'] == row['M']*row['N'], c
    assert row['cublas_math_mode'] == 16 and row['output_tail_untouched'] and row['workspace_guards']
    assert row['seeds'] == [7,8,9,10] and row['activation_seed'] == 7


def same_bits(rs):
    """tc_ladder must reproduce tc's output bits in every cell present in rs (same device code)."""
    by = {(r['K'],r['N'],r['M'],r['config']): [c['output_sha256'] for c in r['validation']] for r in rs}
    n = 0
    for (k,nn,m,c),h in by.items():
        if c == 'tc_ladder' and (k,nn,m,'tc') in by:
            assert h == by[(k,nn,m,'tc')], (k,nn,m); n += 1
    return n


def check_untimed(path,shapes,configs,ms=MS):
    rs = rows(path)
    assert sorted((r['K'],r['N'],r['M'],r['config']) for r in rs) == sorted(
        (k,n,m,c) for k,n in shapes for m in ms for c in configs), path
    for r in rs:
        check_validation(r); assert r['iters'] == 0
    same_bits(rs)
    return rs


def sass_blocks(path):
    blocks, name = {}, None
    for line in path.read_text().splitlines():
        if 'Function :' in line:
            name = re.sub(r'_GLOBAL__N__[0-9a-f]+_','_GLOBAL__N__',line.split()[-1]); blocks[name] = []
        elif name and re.match(r'\s*/\*([0-9a-f]{4,}\*/| 0x)',line):
            blocks[name].append(line.strip())
    return blocks


def prereq(d):
    for label in ['project-build','project-ctest','project-gpu','e048-check','tc-build','agree-4096x11008',
                  'agree-11008x4096','agree-4096x4096','smoke','validation','memcheck','synccheck','racecheck']:
        ok(d,label)
    assert (d/'dependency-diff.txt').read_text() == ''
    assert 'FAILED' not in (d/'e010-inputs-check.log').read_text()
    bench, lib, e049 = (sass_blocks(d/f'sass-tc-{x}.txt') for x in ('bench','lib','e049'))
    assert len(lib) == 6 and lib.keys() <= bench.keys() and all(bench[k] == lib[k] and lib[k] for k in lib), sorted(lib)
    assert lib == e049, 'SASS differs from the E049 library'
    counts = {k: dict(HMMA=sum('HMMA' in x for x in v),LDG=sum('LDG' in x for x in v)) for k,v in lib.items()}
    for k,v in counts.items():  # E044: 32/32 per tc kernel; E048: 64/40 per tc_m32 kernel
        assert v == (dict(HMMA=64,LDG=40) if 'm32' in k else dict(HMMA=32,LDG=32)), counts
    res = {re.sub(r'_GLOBAL__N__[0-9a-f]+_','_GLOBAL__N__',m.group(1)): m.group(2) for m in re.finditer(
        r'Function (\S*w4a16_tc_(?:m32_)?kernel\S*):\s*\n\s*(REG:\d+ STACK:\d+ SHARED:\d+ LOCAL:\d+)',(d/'res-usage-lib.txt').read_text())}
    expect = {'ILb0ELb1E':64,'ILb0ELb0E':76,'ILb1ELb1E':68,'ILb1ELb0E':73}
    for k,v in res.items():
        if 'm32' in k:
            assert v == 'REG:114 STACK:0 SHARED:33792 LOCAL:0', (k,v)
        else:
            assert any(f'REG:{expect[t]} STACK:0 SHARED:17408 LOCAL:0' == v for t in expect if t in k), (k,v)
    assert len(res) == 6, res
    check_untimed(d/'smoke.jsonl',[(256,2048)],NAMES)
    val = check_untimed(d/'validation.jsonl',SHAPES,NAMES)
    bits = same_bits(val)
    assert bits == 12
    traced = {}
    for r in val:
        name, m, ks = r['config'], r['M'], r['traced_kernels']
        assert len(ks) == 1 or name == 'cublas_dq', ks  # cuBLAS may add splitKreduce (E011)
        if name == 'marlin': assert ks[0].startswith('void e005_marlin::Marlin<')
        elif name == 'tc' or (name == 'tc_ladder' and m < 64): assert 'w4a16_tc_kernel<true' in ks[0], ks
        elif name == 'tc_ladder': assert 'w4a16_tc_m32_kernel<false>' in ks[0], ks
        if name == 'tc_ladder': traced[f"{r['K']}x{r['N']} M{m}"] = ks[0]
    for tool in ['memcheck','synccheck']:
        check_untimed(d/f'{tool}.jsonl',SHAPES,NAMES)
        assert 'ERROR SUMMARY: 0 errors' in (d/f'{tool}.log').read_text(), tool
    check_untimed(d/'racecheck.jsonl',[(4096,4096)],['tc','tc_ladder'])
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (d/'racecheck.log').read_text()
    agree = list(csv.DictReader((d/'agree.csv').open()))
    assert len(agree) == 6 and all(a['pass'] == '1' and a['M'] == '128' and a['seed'] == '7' for a in agree)
    out = []
    for a in agree:  # bench tc_m32 at M128 is what tc_ladder runs at M128
        k,n = int(a['K']),int(a['N'])
        cfg = {'tc':'tc','tc_m32':'tc_ladder'}[a['kernel']]
        v = next(r for r in val if (r['K'],r['N'],r['M'],r['config']) == (k,n,128,cfg))['validation'][0]
        assert float(a['max_abs_err']) == float(f"{v['max_abs']:.4e}") and float(a['rel_l2_err']) == float(f"{v['rel_l2']:.4e}"), (a,v)
        out.append(dict(kernel=a['kernel'],K=k,N=n,max_abs=v['max_abs'],rel_l2=v['rel_l2']))
    print('PREREQ PASS',json.dumps(dict(sass_counts=counts,resources=res,agreement=out,
        tc_ladder_bitwise_cells=bits,tc_ladder_trace=traced)))


def order(k,n,m,series):  # mirror of bench.py
    shift = ((SHAPES.index((k,n)) if (k,n) in SHAPES else 0) + (MS.index(m) if m in MS else 0)) % 4
    names = NAMES[shift:]+NAMES[:shift]
    return names if series == 1 else names[::-1]


def cells(series_rows):
    out = {}
    for s,rs in series_rows.items():
        assert len(rs) == 48 and sum(len(r['samples_us']) for r in rs) == 4800
        assert sorted((r['K'],r['N'],r['M'],r['config']) for r in rs) == sorted(
            (k,n,m,c) for k,n in SHAPES for m in MS for c in NAMES)
        for r in rs:
            assert r['series'] == s and r['iters'] == 100 and r['warmup'] == 10 and len(r['samples_us']) == 100
            assert abs(statistics.median(r['samples_us'])-r['median_us']) < 1e-9
            assert r['order'] == order(r['K'],r['N'],r['M'],s).index(r['config'])
            check_validation(r); check_validation(r,'post_timing_validation')
            assert r['post_timing_validation'] == r['validation']
            out[(r['K'],r['N'],r['M'],r['config'],s)] = r['median_us']/4
        assert same_bits(rs) == 12
    return out


def decide(med):
    """Predeclared: tc_ladder/tc <= 0.95 at M 64/128, all shapes, both series."""
    ratios = {f'{k}x{n} M{m} s{s}': med[(k,n,m,'tc_ladder',s)]/med[(k,n,m,'tc',s)]
              for k,n in SHAPES for m in PRIMARY_MS for s in (1,2)}
    passed = sum(v <= 0.95 for v in ratios.values())
    return dict(primary_ratios=ratios,cells_passed=passed,cells=len(ratios),
                decision='ACCEPTED' if passed == len(ratios) else 'REJECTED')


def timing_table(med):
    table = []
    for k,n in SHAPES:
        for m in MS:
            row = dict(K=k,N=n,M=m)
            for s in (1,2):
                mr = med[(k,n,m,'marlin',s)]
                row[f's{s}'] = {c+'_us': med[(k,n,m,c,s)] for c in NAMES}
                row[f's{s}'].update({f'{c}_fraction_of_marlin_speed': mr/med[(k,n,m,c,s)] for c in NAMES if c != 'marlin'})
                row[f's{s}']['tc_ladder_over_tc'] = med[(k,n,m,'tc_ladder',s)]/med[(k,n,m,'tc',s)]
            table.append(row)
    return table


def text(summary):
    L = ['E050 per-GEMM cycle medians (us; four-matrix cycle median / 4), series 1 | series 2; fraction = Marlin/path',
         'K     N     M    tc               tc_ladder        Marlin           cuBLAS           frac tc        frac ladder    frac cuBLAS    ladder/tc']
    for r in summary['timing']:
        f = lambda key,fmt: ' | '.join(fmt.format(r[s][key]) for s in ('s1','s2'))
        L.append(f"{r['K']:<5} {r['N']:<5} {r['M']:<4} {f('tc_us','{:6.1f}')}  {f('tc_ladder_us','{:6.1f}')}  {f('marlin_us','{:6.1f}')}  "
                 f"{f('cublas_dq_us','{:6.1f}')}  {f('tc_fraction_of_marlin_speed','{:5.3f}')}  "
                 f"{f('tc_ladder_fraction_of_marlin_speed','{:5.3f}')}  {f('cublas_dq_fraction_of_marlin_speed','{:5.3f}')}  "
                 f"{f('tc_ladder_over_tc','{:5.3f}')}")
    d = summary['decision']
    L += ['',f"Primary cells (tc_ladder/tc <= 0.95): {d['cells_passed']}/{d['cells']} -> {d['decision']}",
          json.dumps(d['primary_ratios'])]
    return '\n'.join(L)


def full(d,out):
    med = cells({s: rows(d/f'series{s}.jsonl') for s in (1,2)})
    summary = dict(timing=timing_table(med),decision=decide(med))
    (out/'summary.json').write_text(json.dumps(summary,indent=1)+'\n')
    print(text(summary))


def self_test():
    for k,n in SHAPES:
        for m in MS:
            assert order(k,n,m,2) == order(k,n,m,1)[::-1] and sorted(order(k,n,m,1)) == sorted(NAMES)
    offsets = [((SHAPES.index(s)+MS.index(m)) % 4) for s in SHAPES for m in MS]
    assert sorted(offsets) == [0]*3+[1]*3+[2]*3+[3]*3
    med = {(k,n,m,c,s): 100.0 for k,n in SHAPES for m in MS for c in NAMES for s in (1,2)}
    for k,n in SHAPES:
        for m in PRIMARY_MS:
            for s in (1,2): med[(k,n,m,'tc_ladder',s)] = 94.0
    assert decide(med)['decision'] == 'ACCEPTED'
    med[(4096,4096,64,'tc_ladder',2)] = 95.5
    assert decide(med)['decision'] == 'REJECTED' and decide(med)['cells_passed'] == 11
    print('SELF-TEST PASS')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dir',type=Path,nargs='?')
    ap.add_argument('--stage',choices=['prereq','full'])
    ap.add_argument('--out',type=Path)
    ap.add_argument('--self-test',action='store_true')
    a = ap.parse_args()
    if a.self_test: self_test()
    elif a.stage == 'prereq': prereq(a.dir)
    else: full(a.dir,a.out or a.dir)
