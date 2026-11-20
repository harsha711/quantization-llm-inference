#!/usr/bin/env python3
"""E059 analyzer. analyze.py DIR --stage prereq|full [--out OUTDIR]. Unit tests: scripts/e059/test_e059.py.

prereq: build/SASS identity (tc/tc_m32 == E050's library and the same-job bench; tc_splitk == same-job bench),
teammate source hashes, contract (FP64, every row) of every path incl. v1/v2 on every shape, traces, bitwise
splitk_s1 == tc, sanitizers, smoke timing at M = 1 against the DRAM floor (inspection only, not a result).
full: both series; per-GEMM tables (cycle median / matrices), fraction of Marlin speed, the predeclared DRAM-floor
validity check per cell (cyclic and read flush), the read-flush side-by-side, and the predeclared C1/C2 readings."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import e059lib as L  # noqa: E402  (tested helpers; floors/byte model/working set from scripts/checks/floorcheck.py)
MS, SHAPES, SPLIT_SHAPES, TALL, CONTROL = L.MS, L.SHAPES, L.SPLIT_SHAPES, L.TALL, L.CONTROL
COUNTS, BALANCE, paths, order, splits, readings = L.COUNTS, L.BALANCE, L.paths, L.order, L.splits, L.readings
TEAMMATE = {'ours_v1.cu':'805fdf4b4aa8c10469704a903915536d3a0af1bfb21d96764242976fae563136',
            'ours_v2.cu':'9d803287ed35d75e2adb3de68d18a52b442160837734eca689c074f55723e4df',
            'ours_v3.cu':'481f2eac3d3b5264a8b38c4cd19fa305fc6e4226da35b2cc4b5fdcc54049a95a',
            'kernels.cuh':'7fd1ea543503349d55f67a60c0c113934f1e591c7174bee9248afeab0bb9ee59',
            'flush_read.cu':'f6839509fbfb43b965bf440994ad873bb0c8c93474f94c734d67fa71e5452286'}


def check_row(r,count=None,timed=False):
    L.check_row(r,count=count,timed=timed)
    assert r['info'].get('independent_violations') == 0, r['config']


def floor_us(name,m,k,n):
    return L.floor_for(name,k,n,m)


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def ok(d,label):
    assert (d/f'{label}.exit').read_text().strip() == '0', label


def expect_kernel(r):
    name, ks = r['config'], r['traced_kernels']
    one = lambda s: len(ks) == 1 and s in ks[0]
    if name == 'marlin': assert len(ks) == 1 and ks[0].startswith('void e005_marlin::Marlin<'), ks
    elif name == 'tc_ladder': assert one('w4a16_tc_kernel<true'), ks
    elif name in ('v1','v2','v3'): assert one(f'w4a16_{name}_kernel'), ks
    elif name.startswith('splitk_s'): assert one('w4a16_tc_splitk_kernel'), ks
    else: assert ks, ks          # cuBLAS: one GEMM, possibly plus a split-K reduction (E011)


def sass_name(name):
    """Kernel name without its anonymous-namespace prefix, which embeds the compiling file's name and a hash
    (the E059 launcher compiles E047's kernel inside splitk_forced.cu; prerequisite 344190)."""
    return re.sub(r'_ZN\d+_GLOBAL__N__[0-9a-f]{8}_\d+_[A-Za-z0-9_]+?_cu_[0-9a-f]{8}','_ZN_ANON_',name)


def sass_blocks(path):
    blocks, name = {}, None
    for line in path.read_text().splitlines():
        if 'Function :' in line:
            name = sass_name(line.split()[-1]); blocks[name] = []
        elif name and re.match(r'\s*/\*([0-9a-f]{4,}\*/| 0x)',line):
            blocks[name].append(line.strip())
    return blocks


def log_lines(d,label,prefix):
    return [x for x in (d/f'{label}.log').read_text().splitlines() if x.startswith(prefix)]


def prereq(d):
    for label in ['unit-tests','mutations','project-build','project-ctest','project-gpu','e047-check','lib-build','validation','smoke',
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
    # Forced S changes the FP32 addition order: in every split cell some S >= 2 must give other bits than S = 1.
    vb = {(r['K'],r['N'],r['M'],r['config']): [v['output_sha256'] for v in r['validation']] for r in val}
    for k,n in SPLIT_SHAPES:
        for m in MS:
            differ = [s for s in splits(k,n) if s > 1 and vb[(k,n,m,f'splitk_s{s}')] != vb[(k,n,m,'splitk_s1')]]
            assert differ, ('every S gave S = 1 bits',k,n,m)
            print(f'  split K{k}xN{n} M{m}: S with bits different from S = 1: {differ}')
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
        sc = L.status(r['config'],r['K'],r['N'],1,cyc,matrices=r['matrices'])
        sr = L.status(r['config'],r['K'],r['N'],1,rf,flushed=True)
        print(f"  K{r['K']}xN{r['N']} x{r['matrices']} {r['config']:10s} cyclic {cyc:8.2f} ({sc})"
              f"  read-flush {rf:8.2f} ({sr})  floor {f:6.2f}")
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
    for (sr,k,n,m,x),r in R.items():
        c = cell.setdefault((k,n,m,x),dict(cyc=[None,None],rf=[None,None],rf_mean=[None,None],cyc_status=[None,None],rf_status=[None,None]))
        c['cyc'][sr-1] = r['median_per_matrix_us']; c['rf'][sr-1] = r['readflush']['median_us']
        c['rf_mean'][sr-1] = r['readflush']['mean_us']
        c['cyc_status'][sr-1] = L.status(x,k,n,m,r['median_per_matrix_us'],matrices=r['matrices'])
        c['rf_status'][sr-1] = L.status(x,k,n,m,r['readflush']['median_us'],flushed=True)
        c['floor'] = floor_us(x,m,k,n)
    for c in cell.values():
        c['cyc_valid'] = all(L.valid(v) for v in c['cyc_status'])
        c['rf_valid'] = all(L.valid(v) for v in c['rf_status'])
    return cell


def fr(cell,k,n,m,x,key='cyc'):
    return [L.fraction(a,b) for a,b in zip(cell[(k,n,m,'marlin')][key],cell[(k,n,m,x)][key])]


def full(d,out_dir):
    R = series_rows(d)
    cell = summarize(R)
    lines = ['E059 full analysis: per-GEMM us = cycle median / matrices (cyclic, no artificial flush), series 1 / series 2;',
             'frac = Marlin median / path median (fraction of Marlin speed); floor = byte model / 427.2 GB/s;',
             'rf = the teammate\'s read-flush single-matrix median (their method, side by side). * = floorcheck status not OK (below a floor or working set < 2 x L2: cache-affected).','']
    star = lambda ts,st: '/'.join(f'{t:.1f}'+('' if L.valid(v) else '*') for t,v in zip(ts,st))
    for k,n in SHAPES:
        for m in MS:
            lines.append(f'K{k}xN{n} M{m} ({COUNTS[(k,n)]} matrices)  floor 4-bit {floor_us("v3",m,k,n):.2f} us, FP16 {floor_us("cublas_dq",m,k,n):.2f} us')
            for x in paths(k,n):
                c = cell[(k,n,m,x)]
                lines.append(f'  {x:10s} cyc {star(c["cyc"],c["cyc_status"]):>15s}  frac {"/".join(f"{v:.3f}" for v in fr(cell,k,n,m,x))}'
                             f'   rf {star(c["rf"],c["rf_status"]):>15s}  frac {"/".join(f"{v:.3f}" for v in fr(cell,k,n,m,x,"rf"))}'
                             f'   cyc/rf {"/".join(f"{a/b:.3f}" for a,b in zip(c["cyc"],c["rf"]))}')
    invalid = sorted(f'{k}x{n} M{m} {x}' for (k,n,m,x),c in cell.items() if not c['cyc_valid'])
    rinvalid = sorted(f'{k}x{n} M{m} {x}' for (k,n,m,x),c in cell.items() if not c['rf_valid'])
    lines += ['',f'cyclic cells not OK by floorcheck (cache-affected, not results): {invalid or "none"}',
              f'read-flush cells not OK by floorcheck: {rinvalid or "none"}','']
    res = dict(cyclic=readings(cell,'cyc'),readflush_their_method=readings(cell,'rf'))
    lines.append(json.dumps(res,indent=1))
    text = '\n'.join(lines)
    print(text)
    if out_dir:
        (out_dir/'analysis.txt').write_text(text+'\n')
        (out_dir/'summary.json').write_text(json.dumps(dict(floorcheck=L.fc.__file__,cells={f'{k}x{n}|{m}|{x}': v for (k,n,m,x),v in cell.items()},
            readings=res,invalid_cyclic=invalid,invalid_readflush=rinvalid),indent=1)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dir',type=Path,nargs='?')
    p.add_argument('--stage',choices=['prereq','full'])
    p.add_argument('--out',type=Path)
    a = p.parse_args()
    if a.stage == 'prereq': prereq(a.dir)
    else: full(a.dir,a.out)
