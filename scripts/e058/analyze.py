#!/usr/bin/env python3
"""E058 analyzer. analyze.py DIR --stage prereq|full [--out OUTDIR]; analyze.py --self-test.

prereq: library identity (E050 hash), project checks, traced FP64 validation of every path at every matrix count,
sanitizers, the smoke timing at M = 4 (4/8/16 matrices) against the floor (inspection only, not a result).
full: both series; per-GEMM tables at 4/8/16 matrices; the validity check per cell (path x M x series), amendment 1:
(1) scripts/checks/floorcheck (v2).classify == OK at 8 AND 16 matrices (not BELOW_STRICT / BELOW_MEASURED:
time >= max(bytes/427.2 GB/s, 2MNK/51.84 TFLOP/s); working set >= 2 x L2); (2) |t8/t16 - 1| <= 3%. Unit tests: scripts/e058/test_analyze.py. A cell failing either is reported as still cache-affected, not as a result.
Descriptive quantities (fraction of Marlin speed, tc_ladder/tc, cuBLAS fraction) use the 16-matrix medians;
comparison with E050's 4096^2 numbers (results/e050_20261118/summary.json)."""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'checks'))
import floorcheck as fc  # noqa: E402  shared DRAM/compute floor and working-set check (scripts/checks)

K = N = 4096
MS = [4,16,64,128]
COUNTS = [4,8,16]
NAMES = ['marlin','tc','tc_ladder','cublas_dq']
SEEDS = list(range(11,27))
E050_TC_SHA = 'a24a78fad127230f04a867e4a78e7a2a0453a6d4f0db95ea35900c7adaa971fe'
READ_GBPS = fc.BW_BYTES_PER_S/1e9   # E057 measured read-only bandwidth, c8 (job 344144)
CONVERGENCE_TOL = 0.03              # predeclared |t8/t16 - 1| bound
# Amendment 3 wording: a valid cell is converged and insensitive to the working set between 8 and 16 matrices.
# At M64/M128 the 4-bit floor is the compute floor, so there validity rests on convergence alone; E049 still saw
# 4-6% L2 re-reads at 2.8 x L2, so "valid" never means "pure DRAM".
LABEL_VALID = 'converged, working-set-insensitive (8 vs 16)'
NOTE_FLOOR_CALIBRATION = ('floor calibration: cuBLAS BELOW_MEASURED with a 268-537 MB working set (8-16 x L2); '
                          'a calibration issue of the 427.2 GB/s floor, not caching')
E050_SUMMARY = Path(__file__).resolve().parents[2]/'results/e050_20261118/summary.json'


def kind(name):
    """floorcheck v2 byte-model kind: 'fp16' (FP16 W_hat) for cuBLAS, 'w4' for Marlin/tc/tc_ladder."""
    return fc.kind_of(name)


def floor_us(name,m):
    """floorcheck's measured floor: max(byte model / 427.2 GB/s, 2MNK / 51.84 TFLOP/s)."""
    return fc.floor_us(K,N,m,kind=kind(name))


def strict_floor_us(name,m):
    """floorcheck's strict floor: max(weights(+scales) / 448 GB/s, 2MNK / 56.95 TFLOP/s)."""
    return fc.strict_floor_us(K,N,m,kind=kind(name))


def per_gemm_us(row):
    """Per-GEMM time = median of the cycle samples / number of matrices in the cycle."""
    return statistics.median(row['samples_us'])/row['matrices']


def cell_status(name,m,t,count):
    return fc.classify(K,N,m,t,matrices=count,kind=kind(name))


def cell_valid(name,m,t8,t16):
    """Per cell: floorcheck OK at 8 AND 16 matrices (BELOW_STRICT, BELOW_MEASURED and WS_SMALL all invalid),
    and |t8/t16-1| <= 3%."""
    conv = t8/t16 - 1
    v = dict(floor_us=floor_us(name,m),strict_floor_us=strict_floor_us(name,m),t8=t8,t16=t16,status_8=cell_status(name,m,t8,8),
             status_16=cell_status(name,m,t16,16),t8_over_t16_minus_1=conv,
             converged=abs(t8-t16) <= CONVERGENCE_TOL*t16)  # |t8/t16-1| <= 3% without float-ratio rounding at 3.00%
    v['valid'] = v['status_8'] == fc.OK and v['status_16'] == fc.OK and v['converged']
    # Predeclared (amendment 3): cuBLAS BELOW_MEASURED at 8/16 matrices is reported as floor calibration, not caching.
    calib = kind(name) == 'fp16' and fc.BELOW_MEASURED in (v['status_8'],v['status_16']) and \
        fc.BELOW_STRICT not in (v['status_8'],v['status_16'])
    v['note'] = NOTE_FLOOR_CALIBRATION if calib else None
    return v


def cell_label(v):
    """Per-cell text label: VALID, or CACHE-AFFECTED with the 8/16-matrix statuses and convergence."""
    if v['valid']:
        return 'VALID'
    return f"CACHE-AFFECTED ({v['status_8']}/{v['status_16']}, {'converged' if v['converged'] else 'not converged'})"


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
         'path        count  per-GEMM  floor   time/floor  floorcheck']
    out = {}
    for r in sorted(rs,key=lambda r: (NAMES.index(r['config']),r['matrices'])):
        t = per_gemm_us(r); f = floor_us(r['config'],r['M'])
        st = cell_status(r['config'],r['M'],t,r['matrices'])
        out[f"{r['config']} c{r['matrices']}"] = dict(per_gemm_us=t,floor_us=f,ratio=t/f,status=st)
        L.append(f"{r['config']:<11} {r['matrices']:>5}  {t:8.2f}  {f:6.2f}  {t/f:6.3f}  {st}")
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


def check_series_order(rs,s):
    """File order of a series: (count, M) blocks as predeclared (series 2 fully reversed), paths per order()."""
    cs, ms = (COUNTS, MS) if s == 1 else (COUNTS[::-1], MS[::-1])
    expect = [(c,m,x) for c in cs for m in ms for x in order(c,m,s)]
    assert [(r['matrices'],r['M'],r['config']) for r in rs] == expect, f'series {s} file order'


def cells(series_rows):
    out = {}
    for s,rs in series_rows.items():
        assert len(rs) == 48 and sum(len(r['samples_us']) for r in rs) == 4800
        check_series_order(rs,s)
        assert sorted((r['matrices'],r['M'],r['config']) for r in rs) == sorted(
            (c,m,x) for c in COUNTS for m in MS for x in NAMES)
        for r in rs:
            assert (r['K'],r['N']) == (K,N)
            assert r['series'] == s and r['iters'] == 100 and r['warmup'] == 10 and len(r['samples_us']) == 100
            assert abs(statistics.median(r['samples_us'])-r['median_us']) < 1e-9
            assert r['order'] == order(r['matrices'],r['M'],s).index(r['config'])
            check_validation(r); check_validation(r,'post_timing_validation')
            assert r['post_timing_validation'] == r['validation']
            out[(r['matrices'],r['M'],r['config'],s)] = per_gemm_us(r)
        assert same_bits(rs) == 12
    return out


def validity(med):
    """Per (path, M, series): cell_valid at 8/16 matrices; the 4-matrix control is reported, never validated."""
    out = {}
    for x in NAMES:
        for m in MS:
            for s in (1,2):
                v = cell_valid(x,m,med[(8,m,x,s)],med[(16,m,x,s)])
                v.update(t4=med[(4,m,x,s)],status_4=cell_status(x,m,med[(4,m,x,s)],4))
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
            # A ratio is reported as a result only if both of its cells are valid.
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
    L = [f"E058 per-GEMM cycle medians (us; cycle median / matrices), K=N=4096; floor = scripts/checks/floorcheck.py "
         f"(max(bytes/{READ_GBPS} GB/s, 2MNK/51.84 TFLOP/s); working set >= 2 x 32 MiB L2)",
         '', 'Validity (per path, M, series): floorcheck OK at 8 and 16 matrices, |t8/t16-1| <= 3%; 4-matrix status reported only',
         'path        M    s  floor    t4      t8      t16     t8/t16-1  status4      valid']
    for key,v in summary['validity'].items():
        x,m,s = key.split()
        L.append(f"{x:<11} {m[1:]:<4} {s[1:]}  {v['floor_us']:6.2f}  {v['t4']:6.2f}  {v['t8']:6.2f}  {v['t16']:6.2f}  "
                 f"{v['t8_over_t16_minus_1']:+7.3%}  {v['status_4']:<11}  {cell_label(v)}"
                 + (f"  NOTE {v['note']}" if v['note'] else ''))
    L += ['',f"Valid cells ({LABEL_VALID}): {summary['valid_cells']}/{summary['cells']}",
          'VALID = ' + LABEL_VALID + '; at M64/M128 the 4-bit floor is the compute floor, so validity there rests on '
          'the 8-vs-16 convergence; E049 saw 4-6% L2 re-reads at 2.8 x L2. The 3% bound is from E050 between-process '
          'spread (1.36%) while 8 vs 16 is within one process, so a residual L2 benefit up to 3% can pass. '
          'Counts run in time order (clock may change): read both series side by side.','',
          'Ratios at 16 matrices (fraction = Marlin/path; * = a cell failed validity, not a result)',
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
    """Order rule only; the floor/convergence/divide logic is covered by scripts/e058/test_analyze.py."""
    for c in COUNTS:
        for m in MS:
            assert order(c,m,2) == order(c,m,1)[::-1] and sorted(order(c,m,1)) == sorted(NAMES)
    offsets = [((COUNTS.index(c)+MS.index(m)) % 4) for c in COUNTS for m in MS]
    assert sorted(offsets) == [0]*3+[1]*3+[2]*3+[3]*3
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
