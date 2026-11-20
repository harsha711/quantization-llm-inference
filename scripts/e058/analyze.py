#!/usr/bin/env python3
"""E058 analyzer. analyze.py DIR --stage prereq|full [--out OUTDIR]; analyze.py --self-test.

prereq: library identity (E050 hash), project checks, traced FP64 validation of every path at every matrix count,
sanitizers, the smoke timing at M = 4 (4/8/16 matrices) against the DRAM floor (inspection only, not a result).
full: both series; per-GEMM tables at 4/8/16 matrices; the predeclared validity check per cell
(path x M x series): (1) per-GEMM medians at 8 AND 16 matrices >= the DRAM floor (byte model / 427.2 GB/s);
(2) |t8/t16 - 1| <= 3%. A cell failing either is reported as still cache-affected, not as a result.
Descriptive quantities (fraction of Marlin speed, tc_ladder/tc, cuBLAS fraction) use the 16-matrix medians;
comparison with E050's 4096^2 numbers (results/e050_20261118/summary.json)."""
import argparse
import json
import statistics
from pathlib import Path

K = N = 4096
MS = [4,16,64,128]
COUNTS = [4,8,16]
NAMES = ['marlin','tc','tc_ladder','cublas_dq']
SEEDS = list(range(11,27))
E050_TC_SHA = 'a24a78fad127230f04a867e4a78e7a2a0453a6d4f0db95ea35900c7adaa971fe'
READ_GBPS = 427.2         # E057 measured read-only bandwidth, c8 (job 344144)
CONVERGENCE_TOL = 0.03    # predeclared |t8/t16 - 1| bound
E050_SUMMARY = Path(__file__).resolve().parents[2]/'results/e050_20261118/summary.json'


def floor_bytes(name,m,k=K,n=N):
    """Predeclared byte model per GEMM (LATEST_RESULTS rule): weights + FP16 scales (4-bit paths) or FP16 W_hat
    (cuBLAS), plus FP16 activation rows read and FP16 output rows written."""
    weights = 2*k*n if name == 'cublas_dq' else k*n//2 + (k//128)*n*2
    return weights + 2*m*k + 2*m*n


def floor_us(name,m):
    return floor_bytes(name,m)/(READ_GBPS*1e3)


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def ok(d,label):
    assert (d/f'{label}.exit').read_text().strip() == '0', label


def check_validation(row,key='validation'):
    checks = row[key]
    c = row['matrices']
    assert len(checks) == c and row['seeds'] == SEEDS[:c] and row['activation_seed'] == 11, row['config']
    for v in checks:
        assert v['num_bad'] == 0 and v['checked'] == row['M']*row['N'], v
    assert row['cublas_math_mode'] == 16 and row['output_tail_untouched'] and row['workspace_guards']
    assert row['tc_sha256'] == E050_TC_SHA


def same_bits(rs):
    """tc_ladder reproduces tc's output bits per matrix in every (count, M) cell present (same device code)."""
    by = {(r['matrices'],r['M'],r['config']): [v['output_sha256'] for v in r['validation']] for r in rs}
    n = 0
    for (c,m,cfg),h in by.items():
        if cfg == 'tc_ladder' and (c,m,'tc') in by:
            assert h == by[(c,m,'tc')], (c,m); n += 1
    return n


def check_untimed(path,counts,configs,ms=MS,shape=(K,N)):
    rs = rows(path)
    assert sorted((r['matrices'],r['M'],r['config']) for r in rs) == sorted(
        (c,m,x) for c in counts for m in ms for x in configs), path
    for r in rs:
        assert (r['K'],r['N']) == shape
        check_validation(r); assert r['iters'] == 0
    same_bits(rs)
    return rs


def expect_kernel(r):
    name, m, ks = r['config'], r['M'], r['traced_kernels']
    assert len(ks) == 1 or name == 'cublas_dq', ks  # cuBLAS may add splitKreduce (E011)
    if name == 'marlin': assert ks[0].startswith('void e005_marlin::Marlin<'), ks
    elif name == 'tc' or (name == 'tc_ladder' and m < 64): assert 'w4a16_tc_kernel<true' in ks[0], ks
    elif name == 'tc_ladder': assert 'w4a16_tc_m32_kernel<false>' in ks[0], ks


def smoke_table(rs):
    L = ['Smoke timing (prerequisite, series 1, inspection only, NOT a result): per-GEMM us at M=4 vs DRAM floor',
         'path        count  per-GEMM  floor   time/floor']
    out = {}
    for r in sorted(rs,key=lambda r: (NAMES.index(r['config']),r['matrices'])):
        t = r['median_us']/r['matrices']; f = floor_us(r['config'],r['M'])
        out[f"{r['config']} c{r['matrices']}"] = dict(per_gemm_us=t,floor_us=f,ratio=t/f)
        L.append(f"{r['config']:<11} {r['matrices']:>5}  {t:8.2f}  {f:6.2f}  {t/f:6.3f}")
    return out,'\n'.join(L)


def prereq(d):
    for label in ['project-build','project-ctest','project-gpu','smoke','validation','smoke-timing','memcheck','synccheck']:
        ok(d,label)
    assert (d/'dependency-diff.txt').read_text() == ''
    assert 'FAILED' not in (d/'e010-inputs-check.log').read_text()
    assert (d/'tc-lib.sha256').read_text().split()[0] == E050_TC_SHA
    check_untimed(d/'smoke.jsonl',COUNTS,NAMES,shape=(256,2048))
    val = check_untimed(d/'validation.jsonl',COUNTS,NAMES)
    assert same_bits(val) == 12
    for r in val: expect_kernel(r)
    for tool in ['memcheck','synccheck']:
        check_untimed(d/f'{tool}.jsonl',[16],NAMES)
        assert 'ERROR SUMMARY: 0 errors' in (d/f'{tool}.log').read_text(), tool
    mem = val[0]['memory_after_prepare']
    assert mem['allocated'] < mem['total']
    st = rows(d/'smoke-timing.jsonl')
    assert sorted((r['matrices'],r['M'],r['config']) for r in st) == sorted((c,4,x) for c in COUNTS for x in NAMES)
    for r in st:
        check_validation(r); check_validation(r,'post_timing_validation')
        assert r['post_timing_validation'] == r['validation'] and r['iters'] == 100 and len(r['samples_us']) == 100
    assert same_bits(st) == 3
    smoke,tab = smoke_table(st)
    print(tab)
    print('PREREQ PASS',json.dumps(dict(memory_after_prepare_16=mem,tc_ladder_bitwise_cells=12,
        floors_us={x: {m: floor_us(x,m) for m in MS} for x in ('tc','cublas_dq')},smoke=smoke)))


def order(count,m,series):  # mirror of bench.py
    shift = ((COUNTS.index(count) if count in COUNTS else 0) + (MS.index(m) if m in MS else 0)) % 4
    names = NAMES[shift:]+NAMES[:shift]
    return names if series == 1 else names[::-1]


def cells(series_rows):
    out = {}
    for s,rs in series_rows.items():
        assert len(rs) == 48 and sum(len(r['samples_us']) for r in rs) == 4800
        assert sorted((r['matrices'],r['M'],r['config']) for r in rs) == sorted(
            (c,m,x) for c in COUNTS for m in MS for x in NAMES)
        for r in rs:
            assert (r['K'],r['N']) == (K,N)
            assert r['series'] == s and r['iters'] == 100 and r['warmup'] == 10 and len(r['samples_us']) == 100
            assert abs(statistics.median(r['samples_us'])-r['median_us']) < 1e-9
            assert r['order'] == order(r['matrices'],r['M'],s).index(r['config'])
            check_validation(r); check_validation(r,'post_timing_validation')
            assert r['post_timing_validation'] == r['validation']
            out[(r['matrices'],r['M'],r['config'],s)] = r['median_us']/r['matrices']
        assert same_bits(rs) == 12
    return out


def validity(med):
    """Predeclared per (path, M, series): floor at 8 and 16 matrices, and 8-vs-16 convergence within 3%."""
    out = {}
    for x in NAMES:
        for m in MS:
            for s in (1,2):
                f = floor_us(x,m); t8, t16 = med[(8,m,x,s)], med[(16,m,x,s)]
                conv = t8/t16 - 1
                v = dict(floor_us=f,t4=med[(4,m,x,s)],t8=t8,t16=t16,above_floor_8=t8 >= f,above_floor_16=t16 >= f,
                         t4_below_floor=med[(4,m,x,s)] < f,t8_over_t16_minus_1=conv,converged=abs(conv) <= CONVERGENCE_TOL)
                v['valid'] = v['above_floor_8'] and v['above_floor_16'] and v['converged']
                out[(x,m,s)] = v
    return out


def e050_cells():
    if not E050_SUMMARY.exists():
        return None
    t = json.loads(E050_SUMMARY.read_text())['timing']
    return {(r['M'],s): r[f's{s}'] for r in t if (r['K'],r['N']) == (K,N) for s in (1,2)}


def summarize(med):
    val = validity(med)
    e050 = e050_cells()
    table = []
    for m in MS:
        for s in (1,2):
            row = dict(M=m,series=s)
            for c in COUNTS:
                row[f'c{c}'] = {x+'_us': med[(c,m,x,s)] for x in NAMES}
                mr = med[(c,m,'marlin',s)]
                row[f'c{c}'].update({f'{x}_fraction_of_marlin_speed': mr/med[(c,m,x,s)] for x in NAMES if x != 'marlin'})
                row[f'c{c}']['tc_ladder_over_tc'] = med[(c,m,'tc_ladder',s)]/med[(c,m,'tc',s)]
            row['valid'] = {x: val[(x,m,s)]['valid'] for x in NAMES}
            # A ratio is reported as a DRAM-regime result only if both of its cells are valid.
            row['c16_ratio_valid'] = {'tc_fraction': row['valid']['tc'] and row['valid']['marlin'],
                'tc_ladder_fraction': row['valid']['tc_ladder'] and row['valid']['marlin'],
                'cublas_fraction': row['valid']['cublas_dq'] and row['valid']['marlin'],
                'tc_ladder_over_tc': row['valid']['tc_ladder'] and row['valid']['tc']}
            if e050:
                ref = e050[(m,s)]
                row['e050'] = {x+'_us': ref[x+'_us'] for x in NAMES}
                row['c4_over_e050'] = {x: med[(4,m,x,s)]/ref[x+'_us'] for x in NAMES}
                row['c16_over_e050'] = {x: med[(16,m,x,s)]/ref[x+'_us'] for x in NAMES}
            table.append(row)
    nvalid = sum(v['valid'] for v in val.values())
    return dict(timing=table,validity={f'{x} M{m} s{s}': v for (x,m,s),v in val.items()},
                valid_cells=nvalid,cells=len(val),read_gbps=READ_GBPS,convergence_tol=CONVERGENCE_TOL,
                e050_summary=str(E050_SUMMARY) if e050 else None)


def text(summary):
    L = [f"E058 per-GEMM cycle medians (us; cycle median / matrices), K=N=4096; DRAM floor at {READ_GBPS} GB/s",
         '', 'Validity (per path, M, series): t8 and t16 >= floor, |t8/t16-1| <= 3%',
         'path        M    s  floor    t4      t8      t16     t8/t16-1  valid']
    for key,v in summary['validity'].items():
        x,m,s = key.split()
        L.append(f"{x:<11} {m[1:]:<4} {s[1:]}  {v['floor_us']:6.2f}  {v['t4']:6.2f}  {v['t8']:6.2f}  {v['t16']:6.2f}  "
                 f"{v['t8_over_t16_minus_1']:+7.3%}  {'VALID' if v['valid'] else 'CACHE-AFFECTED'}")
    L += ['',f"Valid cells: {summary['valid_cells']}/{summary['cells']}",'',
          'Ratios at 16 matrices (fraction = Marlin/path; * = not a DRAM-regime result, a cell failed validity)',
          'M    s  tc frac   ladder frac  cuBLAS frac  ladder/tc']
    for r in summary['timing']:
        c, ok_ = r['c16'], r['c16_ratio_valid']
        mark = lambda k: '' if ok_[k] else '*'
        L.append(f"{r['M']:<4} {r['series']}  {c['tc_fraction_of_marlin_speed']:6.3f}{mark('tc_fraction'):1}  "
                 f"{c['tc_ladder_fraction_of_marlin_speed']:9.3f}{mark('tc_ladder_fraction'):1}  "
                 f"{c['cublas_dq_fraction_of_marlin_speed']:9.3f}{mark('cublas_fraction'):1}  "
                 f"{c['tc_ladder_over_tc']:7.3f}{mark('tc_ladder_over_tc'):1}")
    if summary['e050_summary']:
        L += ['','Comparison with E050 4096^2 (4 matrices, seeds 7-10): per-GEMM us E050 | E058 c4 | E058 c16',
              'M    s  path        E050    c4      c16     c4/E050  c16/E050']
        for r in summary['timing']:
            for x in NAMES:
                L.append(f"{r['M']:<4} {r['series']}  {x:<11} {r['e050'][x+'_us']:6.2f}  {r['c4'][x+'_us']:6.2f}  "
                         f"{r['c16'][x+'_us']:6.2f}  {r['c4_over_e050'][x]:7.3f}  {r['c16_over_e050'][x]:7.3f}")
    return '\n'.join(L)


def full(d,out):
    med = cells({s: rows(d/f'series{s}.jsonl') for s in (1,2)})
    summary = summarize(med)
    (out/'summary.json').write_text(json.dumps(summary,indent=1)+'\n')
    print(text(summary))


def self_test():
    for c in COUNTS:
        for m in MS:
            assert order(c,m,2) == order(c,m,1)[::-1] and sorted(order(c,m,1)) == sorted(NAMES)
    offsets = [((COUNTS.index(c)+MS.index(m)) % 4) for c in COUNTS for m in MS]
    assert sorted(offsets) == [0]*3+[1]*3+[2]*3+[3]*3
    assert floor_bytes('tc',4) == 8716288 and abs(floor_us('tc',4)-20.403) < 1e-3   # LATEST_RESULTS: 20.4 us
    assert abs(floor_us('tc',16)-20.863) < 1e-3                                      # 20.9 us
    assert floor_bytes('cublas_dq',4) == 33619968
    med = {(c,m,x,s): 2*floor_us(x,m) for c in COUNTS for m in MS for x in NAMES for s in (1,2)}
    assert validity(med)[('tc',4,1)]['valid'] and sum(v['valid'] for v in validity(med).values()) == 32
    med[(8,4,'tc',1)] = 2*floor_us('tc',4)*1.031          # convergence fails
    med[(16,4,'marlin',2)] = 0.99*floor_us('marlin',4)    # floor fails (and convergence)
    v = validity(med)
    assert not v[('tc',4,1)]['valid'] and not v[('tc',4,1)]['converged'] and v[('tc',4,1)]['above_floor_8']
    assert not v[('marlin',4,2)]['valid'] and not v[('marlin',4,2)]['above_floor_16']
    assert sum(x['valid'] for x in v.values()) == 30
    med[(8,4,'tc',1)] = 2*floor_us('tc',4)*1.029
    assert validity(med)[('tc',4,1)]['valid']
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
