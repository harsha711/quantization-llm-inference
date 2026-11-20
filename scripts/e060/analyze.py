#!/usr/bin/env python3
"""E060 analyzer. analyze.py DIR --stage prereq|full [--out OUTDIR].

prereq: project checks, library hashes, unit tests + mutation check, traced all-row FP64 validation of both arms
at every shape/M with the planned counts (kernel address in its own library, one Marlin kernel per call),
memcheck/synccheck of the patched arm, racecheck of both arms (control's known hazard RECORDED), smoke timing
(4096x11008, 4 matrices, M 1/4) classified by scripts/checks/floorcheck (inspection only, NOT a result).
full: both series (60 rows / 6000 cycle samples, file order checked), per-GEMM = cycle median / matrices,
validity = floorcheck OK (kind 'w4') per arm/shape/M/series, cost = t_patched / t_control - 1, labels by the
predeclared 5% all-series rule, cost range over all valid cells and both series, bits control vs patched recorded.
Logic: scripts/e060/e060lib.py; tests: scripts/e060/test_e060.py."""
import argparse
import importlib.util
import json
import os
import statistics
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'e060lib', os.environ.get('E060_LIB', str(Path(__file__).resolve().parent/'e060lib.py')))
lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lib)
fc = lib.fc

ROWS_PER_SERIES = 2*len(lib.SHAPES)*len(lib.MS)   # 30
ITERS, WARMUP = 100, 10


class Measured(dict):
    """{(K, N, M, arm, series): per-GEMM us} plus counts and the bit-equality record."""
    rows = samples = 0
    bits = None


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def check_validation(r, key='validation'):
    v = r[key]
    assert len(v) == r['matrices'], (key, len(v))
    for x in v:
        assert x['num_bad'] == 0 and x['checked'] == r['M']*r['N'], x


def check_row_identity(r):
    assert r['library_sha256'] == lib.ARM_SHA256[r['arm']], (r['arm'], r['library_sha256'])
    assert lib.address_in(r['kernel_address'], r['library_ranges']), (r['arm'], hex(r['kernel_address']))
    assert r['output_tail_untouched'] and r['workspace_guards']
    assert r['matrices'] == lib.COUNTS[(r['K'], r['N'])] and r['seeds'] == lib.seeds_for(r['K'], r['N'])
    assert r['activation_seed'] == lib.ACTIVATION_SEED
    check_validation(r)


def check_pairs(rs):
    """Per cell: the two arms ran from different libraries (address in own, not in the other's ranges)."""
    by = {(r['K'], r['N'], r['M'], r['arm']): r for r in rs}
    for (k, n, m, arm), r in by.items():
        if arm != 'control':
            continue
        p = by[(k, n, m, 'patched')]
        assert r['kernel_address'] != p['kernel_address'], (k, n, m)
        assert not lib.address_in(r['kernel_address'], p['library_ranges']), (k, n, m)
        assert not lib.address_in(p['kernel_address'], r['library_ranges']), (k, n, m)


def bit_equal_cells(rs):
    """Count cells whose control and patched output hashes agree for every matrix; list the others (recorded)."""
    by = {(r['K'], r['N'], r['M'], r['arm']): [v['output_sha256'] for v in r['validation']] for r in rs}
    equal, diff = 0, []
    for (k, n, m, arm), h in sorted(by.items()):
        if arm != 'control' or (k, n, m, 'patched') not in by:
            continue
        if h == by[(k, n, m, 'patched')]:
            equal += 1
        else:
            diff.append((k, n, m))
    return equal, diff


def check_series(rs, s):
    # The exact file order (30 rows) and len(samples_us) == ITERS per row imply 30 rows / 3000 samples per series.
    assert [(r['K'], r['N'], r['M'], r['arm']) for r in rs] == lib.file_order(s), f'series {s} file order'
    for r in rs:
        assert r['series'] == s and r['iters'] == ITERS and r['warmup'] == WARMUP and len(r['samples_us']) == ITERS
        assert abs(statistics.median(r['samples_us'])-r['median_us']) < 1e-9
        assert r['order'] == lib.arm_order(r['K'], r['N'], r['M'], s).index(r['arm'])
        check_row_identity(r)
        check_validation(r, 'post_timing_validation')
        assert r['post_timing_validation'] == r['validation']
    check_pairs(rs)


def check_full(series_rows):
    med = Measured()
    med.bits = {}
    for s in (1, 2):
        rs = series_rows[s]
        check_series(rs, s)
        med.rows += len(rs)
        med.samples += sum(len(r['samples_us']) for r in rs)
        med.bits[s] = bit_equal_cells(rs)
        for r in rs:
            med[(r['K'], r['N'], r['M'], r['arm'], s)] = lib.per_gemm_us(r)
    return med


def summarize(med):
    cells = {}
    for k, n in lib.SHAPES:
        for m in lib.MS:
            t = {(a, s): med[(k, n, m, a, s)] for a in lib.ARMS for s in (1, 2)}
            cells[(k, n, m)] = lib.cell(k, n, m, lib.COUNTS[(k, n)], t)
    o = lib.overall(cells)
    return dict(cells={f'{k}x{n} M{m}': c for (k, n, m), c in cells.items()}, overall=o, rows=med.rows,
                samples=med.samples, bits={s: dict(equal_cells=b[0], differing=b[1]) for s, b in (med.bits or {}).items()},
                tol=lib.TOL, noise=lib.NOISE,
                disclosure=('control = original E006 body with a reported racecheck hazard (E005/E006): timing it does '
                            'not validate it; passing FP64 is not race freedom; never a default. One node, one GPU '
                            'type, synthetic weights, kernel level, unprofiled. Costs below about 1.5% are within '
                            'E058 run-to-run noise.'))


def pct(x):
    return f'{100*x:+.2f}%'


def text(summ):
    L = ['E060 per-GEMM medians (us; cycle median / matrices); cost = patched / control - 1 (positive = barrier costs time)',
         'validity: scripts/checks/floorcheck classify(kind w4) == OK per arm and series; else NOT A RESULT',
         'cell            c  floor   ctl s1   pat s1   ctl s2   pat s2   cost s1   cost s2  label']
    for key, c in summ['cells'].items():
        t = c['times']
        L.append(f"{key:<15} {c['matrices']:>2}  {c['floor_us']:6.2f}  {t['control s1']:7.2f}  {t['patched s1']:7.2f}  "
                 f"{t['control s2']:7.2f}  {t['patched s2']:7.2f}  {pct(c['cost'][1]):>8}  {pct(c['cost'][2]):>8}  "
                 f"{c['label']}")
    o = summ['overall']
    L += ['', f"Overall (predeclared 5% rule, all valid cells, both series): {o['label']}",
          f"valid cells {o['valid_cells']}/{o['cells']}; not a result: {o['not_result']}; |cost| >= 5% in a series: {o['material']}"]
    if o['cost_min'] is not None:
        L.append(f"cost range over valid cells and both series: {pct(o['cost_min'])} to {pct(o['cost_max'])}"
                 f"{' (all within the ~1.5% E058 noise)' if o['within_noise'] else ''}")
    for s, b in summ['bits'].items():
        L.append(f"series {s}: control and patched outputs bitwise equal in {b['equal_cells']} cells; differing: {b['differing']}")
    L += ['', summ['disclosure']]
    return '\n'.join(L)


def ok(d, label, want='0'):
    got = (Path(d)/f'{label}.exit').read_text().strip()
    assert got == want, (label, got)


def check_untimed(path, shapes):
    rs = rows(path)
    assert sorted((r['K'], r['N'], r['M'], r['arm']) for r in rs) == sorted(
        (k, n, m, a) for k, n in shapes for m in lib.MS for a in lib.ARMS), path
    for r in rs:
        assert r['iters'] == 0
        check_row_identity(r)
    check_pairs(rs)
    return rs


def prereq(d):
    d = Path(d)
    for label in ['unit-tests', 'mutation', 'project-build', 'project-ctest', 'project-gpu', 'smoke', 'validation',
                  'memcheck', 'synccheck', 'racecheck-patched', 'smoke-timing']:
        ok(d, label)
    assert (d/'dependency-diff.txt').read_text() == ''
    assert 'FAILED' not in (d/'e010-inputs-check.log').read_text()
    for arm in lib.ARMS:
        assert lib.ARM_SHA256[arm] in (d/'libraries.sha256').read_text(), arm
    smoke = rows(d/'smoke.jsonl')
    assert {(r['K'], r['N']) for r in smoke} == {(256, 2048)}
    val = check_untimed(d/'validation.jsonl', lib.SHAPES)
    for r in val:
        ks = r['traced_kernels']
        assert len(ks) == 1 and ks[0].startswith('void e005_marlin::Marlin<'), ks
    bits = bit_equal_cells(val)
    for tool in ('memcheck', 'synccheck'):
        rs = rows(d/f'{tool}.jsonl')
        assert sorted((r['K'], r['N'], r['M'], r['arm']) for r in rs) == sorted(
            (k, n, m, 'patched') for k, n in lib.SHAPES for m in lib.MS), tool
        assert 'ERROR SUMMARY: 0 errors' in (d/f'{tool}.log').read_text(), tool
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (d/'racecheck-patched.log').read_text()
    rc_control = (d/'racecheck-control.exit').read_text().strip()
    summary_lines = [x for x in (d/'racecheck-control.log').read_text().splitlines() if 'RACECHECK SUMMARY' in x]
    st = rows(d/'smoke-timing.jsonl')
    assert sorted((r['K'], r['N'], r['M'], r['arm']) for r in st) == sorted(
        (4096, 11008, m, a) for m in (1, 4) for a in lib.ARMS)
    L = ['Smoke timing (prerequisite, series-1 protocol, 4 matrices; inspection only, NOT a result)',
         'arm      M   per-GEMM   floor   time/floor  floorcheck']
    smoke_out = {}
    for r in st:
        assert r['iters'] == ITERS and len(r['samples_us']) == ITERS and r['post_timing_validation'] == r['validation']
        check_validation(r, 'post_timing_validation')
        t = lib.per_gemm_us(r)
        f = fc.floor_us(r['K'], r['N'], r['M'], kind=lib.KIND)
        stt = lib.status(r['K'], r['N'], r['M'], t, r['matrices'])
        smoke_out[f"{r['arm']} M{r['M']}"] = dict(per_gemm_us=t, floor_us=f, status=stt)
        L.append(f"{r['arm']:<8} {r['M']:>3}  {t:8.2f}  {f:6.2f}  {t/f:6.3f}  {stt}")
    print('\n'.join(L))
    # per-arm launch resources from the dry call; the host-stub address differs by construction and is excluded
    resources = {f"{r['K']}x{r['N']} M{r['M']} {r['arm']}": {x: y for x, y in r['resources'].items() if x != 'kernel_address'}
                 for r in val}
    print('PREREQ PASS', json.dumps(dict(validation_rows=len(val), bits_equal_cells=bits[0], bits_differing=bits[1],
          racecheck_control_exit=rc_control, racecheck_control_summary=summary_lines,
          memory_after_prepare=[r['memory_after_prepare'] for r in val[:1]], smoke=smoke_out,
          resources_differ=[k for k in resources if k.endswith('control') and
                            resources[k] != resources[k[:-len('control')]+'patched']],
          resources_control={k: v for k, v in resources.items() if k.endswith('control')})))


def full(d, out):
    d = Path(d)
    out = Path(out) if out else d
    sealed = (d/'all.sha256').exists()
    assert not (sealed and out.resolve().is_relative_to(d.resolve())), \
        f'refusing to write into the sealed job directory {d}; pass --out (e.g. results/e060_20261120/analysis)'
    med = check_full({s: rows(d/f'series{s}.jsonl') for s in (1, 2)})
    summ = summarize(med)
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summ, indent=1, default=str)+'\n')
    print(text(summ))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dir', type=Path)
    ap.add_argument('--stage', choices=['prereq', 'full'], required=True)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    if a.stage == 'prereq':
        prereq(a.dir)
    else:
        full(a.dir, a.out)
