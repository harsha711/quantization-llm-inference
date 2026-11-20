#!/usr/bin/env python3
"""E059 analyzer. analyze.py DIR --stage prereq|full [--out OUTDIR]; analyze.py --self-test.

prereq: build/SASS identity (tc/tc_m32 == E050's library and the same-job bench; tc_splitk == same-job bench),
teammate source hashes, contract (FP64, every row) of every path incl. v1/v2 on every shape, traces, bitwise
splitk_s1 == tc, sanitizers, smoke timing at M = 1 against the DRAM floor (inspection only, not a result).
full: both series; per-GEMM tables (cycle median / matrices), fraction of Marlin speed, the predeclared DRAM-floor
validity check per cell (cyclic and read flush), the read-flush side-by-side, and the predeclared C1/C2 readings."""
import argparse
import json
import re
from pathlib import Path

MS = [1,4,16]
SHAPES = [(1536,8960),(8960,1536),(1536,1536),(4096,11008),(11008,4096)]
SPLIT_SHAPES = [(8960,1536),(11008,4096),(4096,11008)]
TALL = [(11008,4096),(8960,1536)]
CONTROL = (4096,11008)
COUNTS = {(1536,8960):10,(8960,1536):10,(1536,1536):56,(4096,11008):4,(11008,4096):4}
BALANCE = {(8960,1536):3,(11008,4096):7,(4096,11008):1}
BASE = ['marlin','tc_ladder','v3','cublas_dq']
READ_GBPS = 427.2          # E057 measured read-only bandwidth, c8 (job 344144)
SEED0 = 101
TEAMMATE = {'ours_v1.cu':'805fdf4b4aa8c10469704a903915536d3a0af1bfb21d96764242976fae563136',
            'ours_v2.cu':'9d803287ed35d75e2adb3de68d18a52b442160837734eca689c074f55723e4df',
            'ours_v3.cu':'481f2eac3d3b5264a8b38c4cd19fa305fc6e4226da35b2cc4b5fdcc54049a95a',
            'kernels.cuh':'7fd1ea543503349d55f67a60c0c113934f1e591c7174bee9248afeab0bb9ee59',
            'flush_read.cu':'f6839509fbfb43b965bf440994ad873bb0c8c93474f94c734d67fa71e5452286'}


def splits(k,n):
    return sorted({1,2,4,8,BALANCE[(k,n)]}) if (k,n) in SPLIT_SHAPES else []


def paths(k,n):
    return BASE+[f'splitk_s{s}' for s in splits(k,n)]


def order(k,n,m,series,extra=()):
    names = paths(k,n)+list(extra)
    shift = (SHAPES.index((k,n))+MS.index(m)) % len(names)
    names = names[shift:]+names[:shift]
    return names if series == 1 else names[::-1]


def floor_bytes(name,m,k,n):
    """Predeclared byte model per GEMM (LATEST_RESULTS rule, as E058): 4-bit codes + FP16 scales (4-bit paths) or
    FP16 W_hat (cuBLAS), plus FP16 activation rows read and FP16 output rows written. Workspaces not counted."""
    weights = 2*k*n if name == 'cublas_dq' else k*n//2+(k//128)*n*2
    return weights+2*m*k+2*m*n


def floor_us(name,m,k,n):
    return floor_bytes(name,m,k,n)/(READ_GBPS*1e3)


def classify(a,b):
    """E027/E054 R1 rule over the series: 'faster' if a <= 0.95 b in every series, 'slower' if b <= 0.95 a in
    every series, 'comparable' if neither side is >= 5% below the other in any series, else 'mixed'."""
    if all(x <= 0.95*y for x,y in zip(a,b)): return 'faster'
    if all(y <= 0.95*x for x,y in zip(a,b)): return 'slower'
    if all(x > 0.95*y and y > 0.95*x for x,y in zip(a,b)): return 'comparable'
    return 'mixed'


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def ok(d,label):
    assert (d/f'{label}.exit').read_text().strip() == '0', label


def check_row(r,count=None,timed=False):
    assert 'contract_failure' not in r, (r['K'],r['N'],r['M'],r['config'],r.get('contract_failure'))
    c = count or COUNTS[(r['K'],r['N'])]
    assert r['matrices'] == c and r['seeds'] == list(range(SEED0,SEED0+c)) and r['activation_seed'] == SEED0
    keys = ['validation']+(['post_timing_validation'] if timed else [])
    for key in keys:
        assert len(r[key]) == c, (key,r['config'])
        for v in r[key]:
            assert v['num_bad'] == 0 and v['checked'] == r['M']*r['N'], v
    if timed:
        assert r['post_timing_validation'] == r['validation']
        assert len(r['samples_us']) == r['iters'] and abs(r['median_per_matrix_us']*c-r['median_us']) < 1e-9
    assert r['cublas_math_mode'] == 16 and r['output_tail_untouched'] and r['workspace_guards']
    if r['config'] == 'v3':
        assert r['info']['fast_path'] is True
    if r['config'].startswith('splitk_s'):
        assert r['info']['S'] == int(r['config'][8:]) and r['info']['balance_rule_S'] == BALANCE[(r['K'],r['N'])]


def expect_kernel(r):
    name, ks = r['config'], r['traced_kernels']
    one = lambda s: len(ks) == 1 and s in ks[0]
    if name == 'marlin': assert len(ks) == 1 and ks[0].startswith('void e005_marlin::Marlin<'), ks
    elif name == 'tc_ladder': assert one('w4a16_tc_kernel<true'), ks
    elif name in ('v1','v2','v3'): assert one(f'w4a16_{name}_kernel'), ks
    elif name.startswith('splitk_s'): assert one('w4a16_tc_splitk_kernel'), ks
    else: assert ks, ks          # cuBLAS: one GEMM, possibly plus a split-K reduction (E011)


def sass_blocks(path):
    blocks, name = {}, None
    for line in path.read_text().splitlines():
        if 'Function :' in line:
            name = re.sub(r'_GLOBAL__N__[0-9a-f]+_','_GLOBAL__N__',line.split()[-1]); blocks[name] = []
        elif name and re.match(r'\s*/\*([0-9a-f]{4,}\*/| 0x)',line):
            blocks[name].append(line.strip())
    return blocks


def log_lines(d,label,prefix):
    return [x for x in (d/f'{label}.log').read_text().splitlines() if x.startswith(prefix)]


def prereq(d):
    for label in ['project-build','project-ctest','project-gpu','e047-check','lib-build','validation','smoke',
                  'memcheck','synccheck','racecheck-v3','racecheck-splitk']:
        ok(d,label)
    assert (d/'dependency-diff.txt').read_text() == ''
    assert 'FAILED' not in (d/'e010-inputs-check.log').read_text()
    src = (d/'teammate-src-check.log').read_text()
    assert all(f'teammate_src/{f}: OK' in src for f in TEAMMATE), src
    binaries = (d/'binaries.sha256').read_text()
    for f,h in TEAMMATE.items():
        assert f'{h}  scripts/e059/teammate_src/{f}' in binaries, f
    bench, lib, e050 = (sass_blocks(d/f'sass-tc-{x}.txt') for x in ('bench','lib','e050'))
    assert len(lib) == 6 and lib == e050 and all(bench.get(k) == v and v for k,v in lib.items()), sorted(lib)
    sb, sl = sass_blocks(d/'sass-splitk-bench.txt'), sass_blocks(d/'sass-splitk-lib.txt')
    assert len(sl) == 2 and sl == sb and all(sl.values()), sorted(sl)
    counts = {k: dict(HMMA=sum('HMMA' in x for x in v),LDG=sum('LDG' in x for x in v)) for k,v in sl.items()}
    ours = sass_blocks(d/'sass-ours-lib.txt')
    print(f'SASS: 6 tc/tc_m32 kernels == E050 library == same-job bench; 2 tc_splitk kernels == same-job bench {counts}')
    print('teammate kernels in library:',{k[:60]: dict(HMMA=sum('HMMA' in x for x in v),LDG=sum('LDG' in x for x in v),
          n=len(v)) for k,v in ours.items()})
    res = (d/'res-usage-lib.txt').read_text()
    for m in re.finditer(r'Function (\S*(?:w4a16_v[123]_kernel|w4a16_tc_splitk_kernel)\S*):\s*\n\s*(REG:\d+ STACK:\d+ SHARED:\d+ LOCAL:\d+)',res):
        print('RES',m.group(1)[:70],m.group(2))
    # Contract of every path (incl. v1/v2) on every shape at M 1/4/16, 2 matrices, traced.
    val = rows(d/'validation.jsonl')
    want = sorted((k,n,m,x) for k,n in SHAPES for m in MS for x in paths(k,n)+['v1','v2'])
    assert sorted((r['K'],r['N'],r['M'],r['config']) for r in val) == want, 'validation rows'
    for r in val:
        check_row(r,count=2); expect_kernel(r); assert r['iters'] == 0
    bits = log_lines(d,'validation','BITWISE splitk_s1')
    assert len(bits) == 9 and all(x.endswith('True') for x in bits), bits
    v32 = log_lines(d,'validation','BITWISE v3 == v2')
    print(f'validation: {len(val)} rows, every row of every matrix within the FP64 bound; splitk_s1 == tc bitwise in 9/9 cells;'
          f' v3 == v2 bitwise in {sum(x.endswith("True") for x in v32)}/{len(v32)} cells (teammate claim)')
    for r in val:
        if r['config'] in ('v1','v2','v3'):
            print(f"  {r['config']} K{r['K']}xN{r['N']} M{r['M']}: v2/v3 S={r['info']['v2_splits']} v1 S={r['info']['v1_splits']} "
                  f"fast={r['info']['fast_path']} max_abs={max(v['max_abs'] for v in r['validation']):.3e} kernel={r['traced_kernels'][0][:40]}")
    for tool in ('memcheck','synccheck'):
        log = (d/f'{tool}.log').read_text()
        assert 'ERROR SUMMARY: 0 errors' in log, tool
        rs = rows(d/f'{tool}.jsonl')
        assert len(rs) == 3*(len(paths(8960,1536))+len(paths(1536,1536))), tool
        for r in rs: check_row(r,count=2,timed=True)
    for label,n in (('racecheck-v3',2),('racecheck-splitk',10)):
        log = (d/f'{label}.log').read_text()
        assert re.search(r'RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)',log), label
        rs = rows(d/f'{label}.jsonl'); assert len(rs) == n, label
        for r in rs: check_row(r,count=1)
    print('sanitizers: memcheck/synccheck 0 errors (one timed cycle + one read-flush sample of every path, 2 matrices); '
          'racecheck v3 and tc_splitk S 1/2/3/4/8: 0 hazards')
    smoke = rows(d/'smoke.jsonl')
    assert sorted((r['K'],r['N'],r['config']) for r in smoke) == sorted(
        (k,n,x) for k,n in [(11008,4096),(8960,1536),(1536,1536)] for x in paths(k,n))
    print('smoke (M = 1, planned counts, 20 samples; inspection only, NOT a result): per-GEMM cyclic / read-flush median us, floor us')
    for r in smoke:
        check_row(r,timed=True)
        f = floor_us(r['config'],1,r['K'],r['N'])
        cyc, rf = r['median_per_matrix_us'], r['readflush']['median_us']
        print(f"  K{r['K']}xN{r['N']} x{r['matrices']} {r['config']:10s} cyclic {cyc:8.2f} ({'ok' if cyc >= f else 'BELOW FLOOR'})"
              f"  read-flush {rf:8.2f} ({'ok' if rf >= f else 'BELOW FLOOR'})  floor {f:6.2f}")
    print('PREREQ PASS')


def series_rows(d):
    out = {}
    for s in (1,2):
        rs = rows(d/f'series{s}.jsonl')
        assert len(rs) == sum(len(paths(k,n)) for k,n in SHAPES)*len(MS), (s,len(rs))
        cells = {}
        for r in rs:
            assert r['series'] == s
            check_row(r,timed=True)
            assert r['iters'] == 100 and r['warmup'] == 10 and r['readflush']['iters'] == 200 and r['readflush']['warmup'] == 10
            assert r['readflush']['validation'] == r['validation'][:1]
            cells.setdefault((r['K'],r['N'],r['M']),[]).append(r)
            out[(s,r['K'],r['N'],r['M'],r['config'])] = r
        for (k,n,m),cr in cells.items():
            assert [r['config'] for r in sorted(cr,key=lambda r: r['order'])] == order(k,n,m,s), (s,k,n,m)
            h = {r['config']: [v['output_sha256'] for v in r['validation']] for r in cr}
            if 'splitk_s1' in h: assert h['splitk_s1'] == h['tc_ladder'], ('S1 bits',s,k,n,m)
        assert len(out) == len(rs)*s
    log = '\n'.join((d/f'series{s}.log').read_text() for s in (1,2))
    assert log.count('BITWISE splitk_s1 == tc_ladder') == 18 and 'False' not in ''.join(
        x for x in log.splitlines() if x.startswith('BITWISE'))
    return out


def summarize(R):
    cell = {}
    for (s,k,n,m,x),r in R.items():
        c = cell.setdefault((k,n,m,x),dict(cyc=[None,None],rf=[None,None],rf_mean=[None,None]))
        c['cyc'][s-1] = r['median_per_matrix_us']; c['rf'][s-1] = r['readflush']['median_us']
        c['rf_mean'][s-1] = r['readflush']['mean_us']
        c['floor'] = floor_us(x,m,k,n)
    for c in cell.values():
        c['cyc_valid'] = all(t >= c['floor'] for t in c['cyc'])
        c['rf_valid'] = all(t >= c['floor'] for t in c['rf'])
    return cell


def fr(cell,k,n,m,x,key='cyc'):
    return [a/b for a,b in zip(cell[(k,n,m,'marlin')][key],cell[(k,n,m,x)][key])]


def readings(cell,key='cyc'):
    """Predeclared C1/C2 readings on one method's medians ('cyc' = cyclic, the result; 'rf' = their read flush)."""
    valid = lambda k,n,m,*xs: all(cell[(k,n,m,x)][key+'_valid'] for x in xs)
    t = lambda k,n,m,x: cell[(k,n,m,x)][key]
    out = dict(C1={},C2_tall={},C2_split={})
    for k,n in SHAPES:
        if not valid(k,n,1,'v3','marlin','tc_ladder'):
            out['C1'][f'{k}x{n}'] = 'open (cache-affected cell)'; continue
        vm = classify(t(k,n,1,'v3'),t(k,n,1,'marlin'))
        best = all(a <= b for a,b in zip(t(k,n,1,'v3'),t(k,n,1,'tc_ladder')))
        out['C1'][f'{k}x{n}'] = dict(v3_vs_marlin=vm,v3_vs_tc_ladder=classify(t(k,n,1,'v3'),t(k,n,1,'tc_ladder')),
            v3_le_tc_ladder_both_series=best,holds=vm == 'comparable' and best)
    h = [v['holds'] for v in out['C1'].values() if isinstance(v,dict)]
    opened = [s for s,v in out['C1'].items() if not isinstance(v,dict)]
    out['C1_verdict'] = ('open' if not h else 'supported' if all(h) else 'contradicted' if not any(h) else
                         'partly supported (shape-dependent)')+(f' on the valid shapes; open on {opened}' if opened and h else '')
    for m in MS:
        shapes = [s for s in SHAPES if valid(*s,m,'tc_ladder','marlin')]
        fracs = {s: fr(cell,*s,m,'tc_ladder',key) for s in shapes}
        if not all(s in shapes for s in TALL) or len(shapes) < 3:
            out['C2_tall'][m] = dict(verdict='open (cache-affected cells)'); continue
        lowest = [sorted(shapes,key=lambda s: fracs[s][i])[:2] for i in (0,1)]
        out['C2_tall'][m] = dict(fractions={f'{k}x{n}': v for (k,n),v in fracs.items()},
            verdict='supported' if all(set(x) == set(TALL) for x in lowest) else 'not supported')
        for k,n in TALL:
            sb = f'splitk_s{BALANCE[(k,n)]}'
            if not valid(k,n,m,sb,'tc_ladder','marlin') or not valid(*CONTROL,m,'tc_ladder','marlin'):
                out['C2_split'][f'{k}x{n} M{m}'] = 'open (cache-affected cell)'; continue
            helps = classify(t(k,n,m,sb),t(k,n,m,'tc_ladder'))
            closes = all(a >= b for a,b in zip(fr(cell,k,n,m,sb,key),fr(cell,*CONTROL,m,'tc_ladder',key)))
            sweep = {f'S{s}': classify(t(k,n,m,f'splitk_s{s}'),t(k,n,m,'tc_ladder')) for s in splits(k,n)
                     if valid(k,n,m,f'splitk_s{s}')}
            best = min((s for s in splits(k,n) if valid(k,n,m,f'splitk_s{s}')),key=lambda s: sum(t(k,n,m,f'splitk_s{s}')))
            out['C2_split'][f'{k}x{n} M{m}'] = dict(balance_S=BALANCE[(k,n)],balance_vs_tc=helps,
                balance_fraction_of_marlin=fr(cell,k,n,m,sb,key),control_tc_fraction=fr(cell,*CONTROL,m,'tc_ladder',key),
                balance_reaches_control=closes,
                verdict='fixes' if helps == 'faster' and closes else 'helps, does not close' if helps == 'faster' else 'does not help',
                sweep_vs_tc=sweep,best_S_POST_HOC=best,best_fraction_POST_HOC=fr(cell,k,n,m,f'splitk_s{best}',key))
    return out


def full(d,out_dir):
    R = series_rows(d)
    cell = summarize(R)
    lines = ['E059 full analysis: per-GEMM us = cycle median / matrices (cyclic, no artificial flush), series 1 / series 2;',
             'frac = Marlin median / path median (fraction of Marlin speed); floor = byte model / 427.2 GB/s;',
             'rf = the teammate\'s read-flush single-matrix median (their method, side by side). * = below floor (cache-affected).','']
    star = lambda ts,f: '/'.join(f'{t:.1f}'+('*' if t < f else '') for t in ts)
    for k,n in SHAPES:
        for m in MS:
            lines.append(f'K{k}xN{n} M{m} ({COUNTS[(k,n)]} matrices)  floor 4-bit {floor_us("v3",m,k,n):.2f} us, FP16 {floor_us("cublas_dq",m,k,n):.2f} us')
            for x in paths(k,n):
                c = cell[(k,n,m,x)]
                lines.append(f'  {x:10s} cyc {star(c["cyc"],c["floor"]):>15s}  frac {"/".join(f"{v:.3f}" for v in fr(cell,k,n,m,x))}'
                             f'   rf {star(c["rf"],c["floor"]):>15s}  frac {"/".join(f"{v:.3f}" for v in fr(cell,k,n,m,x,"rf"))}'
                             f'   cyc/rf {"/".join(f"{a/b:.3f}" for a,b in zip(c["cyc"],c["rf"]))}')
    invalid = sorted(f'{k}x{n} M{m} {x}' for (k,n,m,x),c in cell.items() if not c['cyc_valid'])
    rinvalid = sorted(f'{k}x{n} M{m} {x}' for (k,n,m,x),c in cell.items() if not c['rf_valid'])
    lines += ['',f'cyclic cells below the DRAM floor (cache-affected, not results): {invalid or "none"}',
              f'read-flush cells below the floor: {rinvalid or "none"}','']
    res = dict(cyclic=readings(cell,'cyc'),readflush_their_method=readings(cell,'rf'))
    lines.append(json.dumps(res,indent=1))
    text = '\n'.join(lines)
    print(text)
    if out_dir:
        (out_dir/'analysis.txt').write_text(text+'\n')
        (out_dir/'summary.json').write_text(json.dumps(dict(cells={f'{k}x{n}|{m}|{x}': v for (k,n,m,x),v in cell.items()},
            readings=res,invalid_cyclic=invalid,invalid_readflush=rinvalid),indent=1)+'\n')


def self_test():
    assert classify([90,91],[100,100]) == 'faster' and classify([110,111],[100,100]) == 'slower'
    assert classify([100,103],[100,100]) == 'comparable' and classify([94,100],[100,100]) == 'mixed'
    assert classify([95.0],[100.0]) == 'faster' and classify([95.1],[100.0]) == 'comparable'
    assert round(floor_us('v3',1,8960,1536),2) == 16.66 and round(floor_us('cublas_dq',1,8960,1536),2) == 64.48
    assert [len(paths(k,n)) for k,n in SHAPES] == [4,9,4,8,9]
    assert sum(len(paths(k,n)) for k,n in SHAPES)*3 == 102
    for k,n in SHAPES:
        for m in MS:
            assert sorted(order(k,n,m,1)) == sorted(paths(k,n)) and order(k,n,m,2) == order(k,n,m,1)[::-1]
    # Synthetic cell: v3 ties Marlin and beats tc everywhere -> C1 supported; split S_bal helps on tall shapes.
    cell = {}
    for k,n in SHAPES:
        for m in MS:
            for x in paths(k,n):
                base = 100.0 if x in ('marlin','v3') else 150.0 if x == 'cublas_dq' else 108.0
                if x.startswith('splitk') and (k,n) in TALL and x == f'splitk_s{BALANCE[(k,n)]}': base = 100.5
                if x == 'tc_ladder' and (k,n) in TALL: base = 115.0
                cell[(k,n,m,x)] = dict(cyc=[base,base],rf=[base,base],floor=1.0,cyc_valid=True,rf_valid=True)
    r = readings(cell)
    assert r['C1_verdict'] == 'supported', r['C1']
    assert all(v['verdict'] == 'supported' for v in r['C2_tall'].values()), r['C2_tall']
    assert all(v['verdict'] == 'fixes' for v in r['C2_split'].values()), r['C2_split']
    cell[(1536,1536,1,'v3')]['cyc_valid'] = False
    assert readings(cell)['C1']['1536x1536'].startswith('open')
    print('SELF-TEST OK')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dir',type=Path,nargs='?')
    p.add_argument('--stage',choices=['prereq','full'])
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a = p.parse_args()
    if a.self_test: self_test()
    elif a.stage == 'prereq': prereq(a.dir)
    else: full(a.dir,a.out)
