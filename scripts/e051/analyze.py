#!/usr/bin/env python3
"""E051 analyzer. analyze.py DIR --stage prereq|full [--out OUTDIR]; analyze.py --self-test.

Asserts complete rows, FP64 passes and tc_ladder == tc bitwise in every part; prints descriptive tables
(no gate) and writes OUTDIR/summary.json in the full stage. Profile helpers are E049's (imported unchanged)."""
import argparse
import csv
import importlib.util
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('e049a',ROOT/'scripts/e049/analyze.py')
e049 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e049)
METRIC, STALLS, raw, num, ok, rows, sass_blocks = e049.METRIC, e049.STALLS, e049.raw, e049.num, e049.ok, e049.rows, e049.sass_blocks

KMAIN = 'naive,inblock,inblock_specialized,inblock_m2,inblock_m4,inblock_vec,inblock_m4_vec,tc,tc_m32,tc_ladder,cublas,cublas_dq,dequant,unfused'.split(',')
KHELD = 'inblock_m4_vec,tc,tc_m32,tc_ladder,cublas_dq'.split(',')
MMAIN, MHELD, MC = [1,5,8,32,64,128], [8,32,64,128], [1,4,128]
SHAPES = [(4096,11008),(11008,4096)]
CNAMES, COUNTS = ['marlin','tc_ladder','cublas_dq','unfused'], [1,2,4]
PROFILES = [f'{i}_M{m}_K{k}_N{n}' for k,n in SHAPES for i,m in (('tc_ladder',64),('tc_ladder',128),('marlin',128))]


def bench_csv(path,kernels,ms,shapes,seed,check_rows=None):
    """Complete single-matrix rows: every kernel x M x shape x policy once, all checked and passing."""
    rs = list(csv.DictReader(path.open()))
    key = lambda r: (int(r['K']),int(r['N']),int(r['M']),int(r['flush_l2']),r['kernel'])
    want = sorted((k,n,m,f,c) for k,n in shapes for m in ms for f in (1,0) for c in kernels)
    assert sorted(map(key,rs)) == want, (path,len(rs),len(want))
    for r in rs:
        assert r['pass'] == '1' and int(r['seed']) == seed and int(r['group']) == 128, r
        assert int(r['warmup']) == 10 and int(r['iters']) == 100, r
        if check_rows: assert int(r['check_rows']) == check_rows, r  # E041 GPU command: default 64 sampled rows
        elif r['kernel'] != 'dequant': assert int(r['check_rows']) >= int(r['M']), r
    return {key(r): float(r['median_us']) for r in rs}


def bitwise_logs(d,pattern,expect):
    """Each bench process prints 'tc_ladder vs tc: 0 / N FP16 outputs differ' (bench aborts otherwise)."""
    n = 0
    for log in sorted(d.glob(pattern)):
        hits = re.findall(r'tc_ladder vs tc: (\d+) / (\d+) FP16 outputs differ',log.read_text())
        assert len(hits) == 1 and hits[0][0] == '0', (log,hits); n += 1
    assert n == expect, (pattern,n,expect)
    return n


def check_cycle_row(r,timed):
    assert r['cublas_math_mode'] == 16 and r['workspace_guards'] and r['temporary_guards'], r['config']
    keys = ['validation']+(['post_timing_validation'] if timed else [])
    for key in keys:
        assert len(r[key]) == r['matrices']
        for c in r[key]:
            assert c['num_bad'] == 0 and c['checked'] == 4*r['N'], c
    if r['config'] == 'tc_ladder': assert r['bitwise_equal_tc'], r['config']
    if timed:
        assert r['iters'] == 100 and r['warmup'] == 10 and len(r['samples_us']) == 100
        assert abs(statistics.median(r['samples_us'])-r['median_us']) < 1e-9
        assert [c['output_sha256'] for c in r['post_timing_validation']] == [c['output_sha256'] for c in r['validation']]


def cycle(path,shapes,timed):
    rs = rows(path)
    assert sorted((r['K'],r['N'],r['matrices'],r['config']) for r in rs) == sorted(
        (k,n,c,x) for k,n in shapes for c in COUNTS for x in CNAMES), path
    for r in rs: check_cycle_row(r,timed)
    return rs


def profile(d,tag,impl_cfg):
    """E049's profile check with the tc_ladder tag: tc_ladder at M >= 64 profiles the unchanged tc_m32 launcher."""
    ok(d,f'profile_{tag}')
    impl,m,k,n = re.match(r'(tc_ladder|marlin)_M(\d+)_K(\d+)_N(\d+)$',tag).groups(); m,k,n = int(m),int(k),int(n)
    passes = rows(d/'ncu'/f'{tag}.jsonl')
    assert passes and all(p['post_profile_validation'] == p['validation'] == passes[0]['validation'] for p in passes)
    for p in passes:
        e049.check_validation(p); e049.check_validation(p,'post_profile_validation')
        assert (p['config'],p['M'],p['K'],p['N'],p['launch_skip'],p['launch_count']) == (impl_cfg[impl],m,k,n,399,1)
        assert p['profile_cycles'] == 100 and p['profile_warmup'] == 10
    rs = raw(d/'ncu'/f'{tag}.raw.csv')
    assert len(rs) == 1, (tag,len(rs))
    name = rs[0]['Kernel Name']
    assert ('Marlin' in name) if impl == 'marlin' else 'w4a16_tc_m32_kernel' in name, name
    c = {key: num(rs[0].get(col)) for key,col in METRIC.items()}
    c.update(case=tag,impl=impl,M=m,K=k,N=n,kernel=name,passes=len(passes),missing=[key for key,v in c.items() if v is None])
    c['dur_us'] = c['dur']/1e3
    c['unique_bytes'] = k*n//2+2*n*k//128+2*m*k+2*m*n
    c['dram_rd_over_unique'] = c['dram_rd']/c['unique_bytes']
    tot = sum(c['st_'+s] or 0 for s in STALLS) or 1
    c['stall_share'] = {s: 100*(c['st_'+s] or 0)/tot for s in STALLS}
    c['sm_active_pct'] = 100*c['sm_active']/c['sm_elapsed']
    c['l2_tex_hit_pct'] = 100*c['l2_tex_hit']/(c['l2_tex_hit']+c['l2_tex_miss']) if c['l2_tex_hit'] is not None else None
    return c


CFG = dict(tc_ladder='tc_m32',marlin='marlin')


def prereq(d):
    for label in ['project-build','project-ctest','unpack-selftest','project-gpu','e048-check','ladder-memcheck','lib-e050','lib-e049',
                  'b-smoke','b-validation','b-memcheck','b-synccheck','b-racecheck','calibration']+[f'ladder-check-M{m}' for m in (1,5,63,64,65,128)]:
        ok(d,label)
    assert (d/'dependency-diff.txt').read_text() == ''
    assert 'FAILED' not in (d/'e010-inputs-check.log').read_text()
    bench = sass_blocks(d/'sass-tc-bench.txt')
    for lib in ('sass-tc-lib50.txt','sass-tc-lib49.txt'):
        b = sass_blocks(d/lib)
        assert len(b) == 6 and b.keys() <= bench.keys() and all(bench[k] == b[k] and b[k] for k in b), (lib,sorted(b))
    bitwise_logs(d,'ladder-check-M*.log',6)
    assert 'ERROR SUMMARY: 0 errors' in (d/'ladder-memcheck.log').read_text()
    for M in (5,128):
        bitwise_logs(d,f'a-smoke-M{M}-*.log',2)
    for f in [x for x in csv.DictReader((d/'a-smoke.csv').open())]:
        assert f['pass'] == '1', f
    for line in (d/'c-smoke.csv').read_text().split():
        assert line.split(',')[10] == '1', line
    cycle(d/'b-smoke.jsonl',[(256,2048)],False)
    cycle(d/'b-validation.jsonl',SHAPES,False)
    for tool in ('memcheck','synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (d/f'b-{tool}.log').read_text(), tool
        cycle(d/f'b-{tool}.jsonl',SHAPES,False)
    rr = rows(d/'b-racecheck.jsonl')
    assert len(rr) == 6 and all(r['config'] == 'tc_ladder' and r['bitwise_equal_tc'] for r in rr)
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (d/'b-racecheck.log').read_text()
    assert (d/'metrics-dropped.txt').read_text() == ''
    cal = raw(d/'ncu'/'calibration.raw.csv')
    assert len(cal) == 1 and 'add' in cal[0]['Kernel Name'].lower(), cal
    first = profile(d,'tc_ladder_M64_K4096_N11008',CFG)
    print('PREREQ PASS',json.dumps({k: first[k] for k in ('kernel','passes','dur_us','dram_rd','inst','hmma','missing')}))


def ratio_table(med,kernels,ms,shapes,pairs):
    out = []
    for k,n in shapes:
        for m in ms:
            for f in (1,0):
                row = dict(K=k,N=n,M=m,flush=f)
                for s in (1,2):
                    row[f's{s}'] = {c: med[s][(k,n,m,f,c)] for c in kernels}
                    for a,b in pairs:
                        row[f's{s}'][f'{a}/{b}'] = med[s][(k,n,m,f,a)]/med[s][(k,n,m,f,b)]
                out.append(row)
    return out


def full(d,out):
    # (a) single-matrix sweep
    main = {s: bench_csv(d/f'a-main-s{s}.csv',KMAIN,MMAIN,[(4096,4096)],1) for s in (1,2)}
    held = {s: bench_csv(d/f'a-held-s{s}.csv',KHELD,MHELD,SHAPES,2) for s in (1,2)}
    bitwise_logs(d,'a-main-s*-M*-*.log',2*len(MMAIN)*2)
    bitwise_logs(d,'a-held-s*-K*-M*-*.log',2*2*len(MHELD)*2)
    pairs = [('tc_ladder','tc'),('tc_ladder','tc_m32'),('tc_ladder','inblock_m4_vec'),('tc_ladder','cublas_dq'),('tc_ladder','unfused')]
    a_main = ratio_table(main,KMAIN,MMAIN,[(4096,4096)],pairs+[('tc_ladder','naive'),('tc_ladder','inblock_specialized'),('tc_ladder','cublas')])
    a_held = ratio_table(held,KHELD,MHELD,SHAPES,pairs[:4])
    # (c) E041 row
    cpu = list(csv.DictReader((d/'c-cpu.csv').open()))
    assert sorted((int(r['M']),int(r['threads'])) for r in cpu) == sorted((m,p) for m in MC for p in (1,16))
    for r in cpu: assert r['fp64_pass'] == '1' and int(r['reps']) == 10, r
    for m in MC: assert len({r['fnv1a'] for r in cpu if int(r['M']) == m}) == 1, m
    g = bench_csv(d/'c-gpu.csv',['naive','inblock','inblock_m4','cublas','tc_ladder'],MC,[(4096,4096)],1,check_rows=64)
    bitwise_logs(d,'c-gpu-M*-*.log',6)
    c_tab = []
    for m in MC:
        t1 = next(float(r['median_ms']) for r in cpu if int(r['M']) == m and r['threads'] == '1')
        t16 = next(float(r['median_ms']) for r in cpu if int(r['M']) == m and r['threads'] == '16')
        row = dict(M=m,cpu_T1_ms=t1,cpu_T16_ms=t16,cpu_Sp16=t1/t16)
        for f in (1,0):
            for c in ('naive','inblock','inblock_m4','cublas','tc_ladder'):
                us = g[(4096,4096,m,f,c)]
                row[f'{c}_{"flush" if f else "noflush"}_us'] = us
                row[f'{c}_{"flush" if f else "noflush"}_T1_over_gpu'] = t1*1e3/us
        c_tab.append(row)
    # (b) E009 cycle
    b = {s: cycle(d/f'b-series{s}.jsonl',SHAPES,True) for s in (1,2)}
    for s,rs in b.items():
        for r in rs:
            assert r['series'] == s
            assert r['order'] == order(r['K'],r['N'],r['matrices'],s).index(r['config'])
    bmed = {(r['K'],r['N'],r['matrices'],r['config'],s): r['median_per_matrix_us'] for s,rs in b.items() for r in rs}
    b_tab = []
    for k,n in SHAPES:
        for c in COUNTS:
            row = dict(K=k,N=n,matrices=c)
            for s in (1,2):
                v = {x: bmed[(k,n,c,x,s)] for x in CNAMES}
                v['tc_ladder_fraction_of_marlin_speed'] = v['marlin']/v['tc_ladder']
                v['cublas_dq_over_tc_ladder'] = v['cublas_dq']/v['tc_ladder']
                v['unfused_over_tc_ladder'] = v['unfused']/v['tc_ladder']
                v['cublas_dq_over_marlin'] = v['cublas_dq']/v['marlin']
                v['unfused_over_marlin'] = v['unfused']/v['marlin']
                row[f's{s}'] = v
            b_tab.append(row)
    # (d) profiles, with E049's tc/Marlin M64 profiles (same node, same harness) as references
    prof = {t: profile(d,t,CFG) for t in PROFILES}
    ref49 = json.loads((ROOT/'results/e049_20261117/summary.json').read_text())['profiles']
    d_tab = {}
    for k,n in SHAPES:
        l64,l128,m128 = (prof[f'{i}_M{m}_K{k}_N{n}'] for i,m in (('tc_ladder',64),('tc_ladder',128),('marlin',128)))
        t64,m64 = ref49[f'tc_M64_K{k}_N{n}'],ref49[f'marlin_M64_K{k}_N{n}']
        d_tab[f'K{k}xN{n}'] = dict(
            inst_ratio_vs_marlin_M64=l64['inst']/m64['inst'],inst_ratio_vs_marlin_M128=l128['inst']/m128['inst'],
            inst_ratio_vs_tc_M64=l64['inst']/t64['inst'],hmma_equal_marlin_M64=l64['hmma'] == m64['hmma'],
            hmma_equal_marlin_M128=l128['hmma'] == m128['hmma'],
            replay_us=dict(tc_M64_e049=t64['dur_us'],tc_ladder_M64=l64['dur_us'],marlin_M64_e049=m64['dur_us'],
                           tc_ladder_M128=l128['dur_us'],marlin_M128=m128['dur_us']),
            long_scoreboard_pct=dict(tc_M64_e049=t64['stall_share']['long_scoreboard'],tc_ladder_M64=l64['stall_share']['long_scoreboard'],
                                     tc_ladder_M128=l128['stall_share']['long_scoreboard'],marlin_M64_e049=m64['stall_share']['long_scoreboard'],
                                     marlin_M128=m128['stall_share']['long_scoreboard']),
            hmma_pipe_pct=dict(tc_M64_e049=t64['hmma_pct'],tc_ladder_M64=l64['hmma_pct'],tc_ladder_M128=l128['hmma_pct'],
                               marlin_M64_e049=m64['hmma_pct'],marlin_M128=m128['hmma_pct']),
            dram_rd_over_unique=dict(tc_ladder_M64=l64['dram_rd_over_unique'],tc_ladder_M128=l128['dram_rd_over_unique'],
                                     marlin_M128=m128['dram_rd_over_unique']))
    summary = dict(a_main=a_main,a_held=a_held,c=c_tab,b=b_tab,profiles=prof,profile_comparison=d_tab)
    (out/'summary.json').write_text(json.dumps(summary,indent=1)+'\n')
    print(text(summary))


def order(k,n,count,series):  # mirror of cyclic.py
    cell = [(a,b,c) for a,b in SHAPES for c in COUNTS]
    shift = cell.index((k,n,count)) % 4 if (k,n,count) in cell else COUNTS.index(count)
    names = CNAMES[shift:]+CNAMES[:shift]
    return names if series == 1 else names[::-1]


def text(s):
    f = lambda v: f'{v:.1f}' if v >= 10 else f'{v:.2f}'
    L = ['(a) single matrix, K=N=4096 seed 1, median us, series 1 | series 2']
    show = ['inblock_specialized','inblock_m4_vec','tc','tc_m32','tc_ladder','cublas','cublas_dq','unfused']
    L.append('M    flush  '+'  '.join(f'{c:>15}' for c in show)+'   ladder/tc      ladder/m4vec   ladder/cublas_dq')
    for r in s['a_main']:
        L.append(f"{r['M']:<4} {r['flush']:<6} "+'  '.join(f"{f(r['s1'][c]):>7}|{f(r['s2'][c]):<7}" for c in show)+'  '+
                 '  '.join(f"{r['s1'][x]:.3f}|{r['s2'][x]:.3f}" for x in ('tc_ladder/tc','tc_ladder/inblock_m4_vec','tc_ladder/cublas_dq')))
    L.append('(a) held-out rectangles, seed 2')
    for r in s['a_held']:
        L.append(f"{r['K']}x{r['N']} M{r['M']} flush{r['flush']} "+'  '.join(f"{c} {f(r['s1'][c])}|{f(r['s2'][c])}" for c in KHELD)+
                 '  '+'  '.join(f"{x} {r['s1'][x]:.3f}|{r['s2'][x]:.3f}" for x in ('tc_ladder/tc','tc_ladder/cublas_dq')))
    L.append('(c) E041 row on this node: CPU ms, GPU us, T1/GPU processor ratios (not parallel speedups)')
    for r in s['c']:
        L.append(json.dumps({k: round(v,3) if isinstance(v,float) else v for k,v in r.items()}))
    L.append('(b) E009 cycle M=4, per-matrix median us, series 1 | series 2')
    for r in s['b']:
        L.append(f"{r['K']}x{r['N']} x{r['matrices']} "+'  '.join(f"{c} {f(r['s1'][c])}|{f(r['s2'][c])}" for c in CNAMES)+'  '+
                 '  '.join(f"{x} {r['s1'][x]:.3f}|{r['s2'][x]:.3f}" for x in ('tc_ladder_fraction_of_marlin_speed','cublas_dq_over_tc_ladder','unfused_over_tc_ladder')))
    L.append('(d) profiles (replay values, not production latency)')
    for c in s['profiles'].values():
        top = ', '.join(f'{k} {v:.0f}' for k,v in sorted(c['stall_share'].items(),key=lambda x:-x[1])[:4])
        L.append(f"{c['case']} {c['kernel'][:30]} passes {c['passes']} dur {c['dur_us']:.1f} dram_rd/unique {c['dram_rd_over_unique']:.2f} "
                 f"L2 tex hit {c['l2_tex_hit_pct']:.1f} inst {c['inst']/1e6:.3f}M hmma {c['hmma']:.0f} hmma_pipe {c['hmma_pct']:.1f} "
                 f"issue {c['issue_pct']:.1f} occ {c['occ']:.1f}/{c['occ_theo']:.1f} regs {c['regs']:.0f} stalls: {top}"+
                 (f" MISSING {c['missing']}" if c['missing'] else ''))
    for k,v in s['profile_comparison'].items():
        L.append(f'{k}: '+json.dumps(v))
    return '\n'.join(L)


def self_test():
    for k,n in SHAPES:
        for c in COUNTS:
            assert order(k,n,c,2) == order(k,n,c,1)[::-1] and sorted(order(k,n,c,1)) == sorted(CNAMES)
    src = (ROOT/'scripts/e051/cyclic.py').read_text()
    assert "NAMES = ['marlin', 'tc_ladder', 'cublas_dq', 'unfused']" in src and '% 4 if' in src
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
