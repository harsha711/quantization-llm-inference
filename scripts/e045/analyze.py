#!/usr/bin/env python3
"""E045 analyzer. analyze.py DIR --stage prereq|full [--out OUTDIR]; analyze.py --self-test.

full: checks both series, then writes OUTDIR/summary.json and prints the analysis text."""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

SHAPES = [(4096,11008),(11008,4096)]
MS = [4,16,64,128]
NAMES = ['marlin','tc','cublas_dq']


def order(k,n,m,series):  # mirror of bench.py, kept here so the analyzer runs without torch
    shift = ((SHAPES.index((k,n)) if (k,n) in SHAPES else 0) + (MS.index(m) if m in MS else 0)) % 3
    names = NAMES[shift:]+NAMES[:shift]
    return names if series == 1 else names[::-1]


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def check_validation(row,key='validation'):
    checks = row[key]
    assert len(checks) == 4, row['config']
    for c in checks:
        assert c['num_bad'] == 0 and c['checked'] == row['M']*row['N'], c
    assert row['cublas_math_mode'] == 16 and row['output_tail_untouched'] and row['workspace_guards']


def check_untimed(path,shapes,configs):
    rs = rows(path)
    assert sorted((r['K'],r['N'],r['M'],r['config']) for r in rs) == sorted(
        (k,n,m,c) for k,n in shapes for m in MS for c in configs), path
    for r in rs:
        check_validation(r)
        assert r['iters'] == 0
    return rs


def sass_blocks(path):
    blocks, name = {}, None
    for line in path.read_text().splitlines():
        if 'Function :' in line:
            name = line.split()[-1]; blocks[name] = []
        elif name and re.match(r'\s*/\*([0-9a-f]{4,}\*/| 0x)',line):
            blocks[name].append(line.strip())
    return blocks


def prereq(d):
    for label in ['project-build','project-ctest','project-gpu','tc-build','agree-4096x11008','agree-11008x4096',
                  'smoke','validation','memcheck','synccheck','racecheck']:
        assert (d/f'{label}.exit').read_text().strip() == '0', label
    assert (d/'dependency-diff.txt').read_text() == ''
    assert 'FAILED' not in (d/'e010-inputs-check.log').read_text()
    bench, lib = sass_blocks(d/'sass-tc-bench.txt'), sass_blocks(d/'sass-tc-lib.txt')
    assert len(bench) == 4 and bench.keys() == lib.keys() and all(bench[k] == lib[k] and bench[k] for k in bench)
    check_untimed(d/'smoke.jsonl',[(256,2048)],NAMES)
    val = check_untimed(d/'validation.jsonl',SHAPES,NAMES)
    for r in val:
        if r['config'] == 'tc':
            assert len(r['traced_kernels']) == 1 and 'w4a16_tc_kernel<true' in r['traced_kernels'][0]
        if r['config'] == 'marlin':
            assert r['traced_kernels'][0].startswith('void e005_marlin::Marlin<')
    for tool in ['memcheck','synccheck']:
        check_untimed(d/f'{tool}.jsonl',SHAPES,NAMES)
        assert 'ERROR SUMMARY: 0 errors' in (d/f'{tool}.log').read_text(), tool
    check_untimed(d/'racecheck.jsonl',SHAPES,['tc'])
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (d/'racecheck.log').read_text()
    agree = list(csv.DictReader((d/'agree.csv').open()))
    assert len(agree) == 2
    out = []
    for a in agree:
        k,n = int(a['K']),int(a['N'])
        assert a['kernel'] == 'tc' and a['pass'] == '1' and a['M'] == '128' and a['seed'] == '3' and a['check_rows'] == '128'
        v = next(r for r in val if (r['K'],r['N'],r['M'],r['config']) == (k,n,128,'tc'))['validation'][0]
        # Same seed-3 operands, all 128 rows: stats printed by the E044 build equal the bridge's.
        assert float(a['max_abs_err']) == float(f"{v['max_abs']:.4e}") and float(a['rel_l2_err']) == float(f"{v['rel_l2']:.4e}"), (a,v)
        out.append(dict(K=k,N=n,max_abs=v['max_abs'],rel_l2=v['rel_l2']))
    print('PREREQ PASS',json.dumps(dict(sass_functions=sorted(bench),agreement=out)))


def cells(series_rows):
    out = {}
    for s,rs in series_rows.items():
        assert len(rs) == 24
        for r in rs:
            assert r['series'] == s and r['iters'] == 100 and r['warmup'] == 10 and len(r['samples_us']) == 100
            assert abs(statistics.median(r['samples_us'])-r['median_us']) < 1e-9
            assert r['order'] == order(r['K'],r['N'],r['M'],s).index(r['config'])
            check_validation(r); check_validation(r,'post_timing_validation')
            out[(r['K'],r['N'],r['M'],r['config'],s)] = r['median_us']/4
    return out


def summarize(med):
    table, confirmed, notes = [], True, []
    for k,n in SHAPES:
        for s in (1,2):
            ratio = {m: med[(k,n,m,'tc',s)]/med[(k,n,m,'marlin',s)] for m in MS}
            if min(ratio,key=ratio.get) != 4 or max(ratio,key=ratio.get) != 128:
                confirmed = False; notes.append(f'K{k}/N{n} series {s}: tc/Marlin ratio not smallest at M4 and largest at M128')
        for m in MS:
            row = dict(K=k,N=n,M=m)
            for s in (1,2):
                t,mr,cb = (med[(k,n,m,c,s)] for c in ('tc','marlin','cublas_dq'))
                row[f's{s}'] = dict(tc_us=t,marlin_us=mr,cublas_us=cb,tc_over_marlin=t/mr,tc_over_cublas=t/cb,
                                    fraction_of_marlin_speed=mr/t)
                if t <= mr:
                    confirmed = False; notes.append(f'K{k}/N{n} M{m} series {s}: tc not slower than Marlin')
            table.append(row)
    return dict(table=table,hypothesis='confirmed' if confirmed else 'not confirmed',failures=notes)


def text(summary):
    lines = ['E045 per-GEMM cycle medians (us; four-matrix cycle median / 4), series 1 | series 2',
             'K     N     M    tc            Marlin        cuBLAS        tc/Marlin     tc/cuBLAS     Marlin/tc']
    for r in summary['table']:
        f = lambda key,fmt: ' | '.join(fmt.format(r[s][key]) for s in ('s1','s2'))
        lines.append(f"{r['K']:<5} {r['N']:<5} {r['M']:<4} {f('tc_us','{:6.1f}')}  {f('marlin_us','{:6.1f}')}  "
                     f"{f('cublas_us','{:6.1f}')}  {f('tc_over_marlin','{:5.2f}')}  {f('tc_over_cublas','{:5.2f}')}  "
                     f"{f('fraction_of_marlin_speed','{:5.3f}')}")
    lines.append(f"Hypothesis (tc slower than Marlin in all 16 cell-series; tc/Marlin smallest at M4, largest at M128 "
                 f"per shape and series): {summary['hypothesis']}")
    lines += ['  '+x for x in summary['failures']]
    return '\n'.join(lines)


def full(d,out):
    med = cells({s: rows(d/f'series{s}.jsonl') for s in (1,2)})
    summary = summarize(med)
    (out/'summary.json').write_text(json.dumps(summary,indent=1)+'\n')
    print(text(summary))


def self_test():
    for k,n in SHAPES:
        for m in MS:
            assert order(k,n,m,2) == order(k,n,m,1)[::-1] and sorted(order(k,n,m,1)) == sorted(NAMES)
    med = {(k,n,m,c,s): (1.0 if c != 'tc' else 1.0+m/100) for k,n in SHAPES for m in MS for c in NAMES for s in (1,2)}
    assert summarize(med)['hypothesis'] == 'confirmed'
    med[(4096,11008,64,'tc',2)] = 0.9
    r = summarize(med)
    assert r['hypothesis'] == 'not confirmed' and len(r['failures']) == 2  # slower rule and ordering rule
    print('SELF-TEST PASS')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dir',type=Path,nargs='?')
    p.add_argument('--stage',choices=['prereq','full'])
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a = p.parse_args()
    if a.self_test:
        self_test()
    elif a.stage == 'prereq':
        prereq(a.dir)
    else:
        full(a.dir,a.out or a.dir)
