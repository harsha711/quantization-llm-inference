"""E059 pure-Python helpers shared by bench.py and analyze.py; tested by test_e059.py (written first).

Floor/byte model/working-set classification come from the shared checker scripts/checks/floorcheck.py
(BW 427.2e9 B/s, peak 51.84e12 FLOP/s, L2 32 MiB, working set >= 2 x L2); nothing here re-derives them.
"""
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checks'))
import floorcheck as fc  # noqa: E402

MS = [1, 4, 16]
SHAPES = [(1536, 8960), (8960, 1536), (1536, 1536), (4096, 11008), (11008, 4096), (4096, 4096)]   # 4096^2: amendment 2
SPLIT_SHAPES = [(8960, 1536), (11008, 4096), (4096, 11008)]
TALL = [(11008, 4096), (8960, 1536)]
CONTROL = (4096, 11008)
COUNTS = {(1536, 8960): 10, (8960, 1536): 10, (1536, 1536): 56, (4096, 11008): 4, (11008, 4096): 4,
          (4096, 4096): 8}                                                                          # predeclared
# C1 as quoted (cd39962 docs/OURS.md, M=1 read-flush tables): v3 "within +-4% of Marlin on the five large shapes";
# v3 "beats tc by 5-13% on four of six shapes and ties on the other two".
CLAIM_MARLIN_TIE = [(4096, 11008), (11008, 4096), (4096, 4096), (1536, 8960), (8960, 1536)]
CLAIM_V3_VS_TC = {(4096, 11008): 'faster', (11008, 4096): 'faster', (4096, 4096): 'faster', (8960, 1536): 'faster',
                  (1536, 8960): 'comparable', (1536, 1536): 'comparable'}
BALANCE = {(8960, 1536): 3, (11008, 4096): 7, (4096, 11008): 1}                                      # predeclared
BASE = ['marlin', 'tc_ladder', 'v3', 'cublas_dq']
SMS = 36
MIN_MATRICES = 4      # E050's four-matrix cycle is the minimum


def matrix_count(k, n):
    """Smallest count >= 4 whose rotating 4-bit working set is >= L2_MARGIN x L2 (floorcheck constants)."""
    assert k % fc.GROUP == 0, 'K must be a whole number of 128-groups'
    c = MIN_MATRICES
    while fc.working_set(k, n, c) < fc.L2_MARGIN * fc.L2_BYTES:
        c += 1
    return c


# The shared checker exists in two API versions (v1: fp16_weights=, BELOW_FLOOR; v2: kind=, BELOW_STRICT and
# BELOW_MEASURED). Both use the same byte model for the E059 paths; this adapter accepts either.
BELOW = {getattr(fc, x) for x in ('BELOW_FLOOR', 'BELOW_STRICT', 'BELOW_MEASURED') if hasattr(fc, x)}


def _model(name):
    fp16 = name == 'cublas_dq'
    return dict(kind='fp16' if fp16 else 'w4') if hasattr(fc, 'kind_of') else dict(fp16_weights=fp16)


def bytes_for(name, k, n, m):
    return fc.bytes_model(k, n, m, **_model(name))


def floor_for(name, k, n, m):
    return fc.floor_us(k, n, m, **_model(name))


def status(name, k, n, m, t_us, matrices=None, flushed=False):
    """floorcheck.classify with the path's byte model (FP16 W_hat for cuBLAS, 4-bit + scales otherwise)."""
    return fc.classify(k, n, m, t_us, matrices=matrices, flushed=flushed, **_model(name))


def is_below(s):
    return s in BELOW


def valid(s):
    return s in (fc.OK, fc.OK_FLUSHED)


def per_gemm_us(samples_us, matrices):
    """Amortized per-GEMM time: median of the complete-cycle samples / matrices in the cycle."""
    return statistics.median(samples_us) / matrices


def fraction(marlin_us, path_us):
    """Fraction of Marlin speed = Marlin time / path time."""
    return marlin_us / path_us


def r1(a, b):
    """E027/E054 R1 over the series: 'faster' if a <= 0.95 b in every series, 'slower' if b <= 0.95 a in every
    series, 'comparable' if neither is >= 5% below the other in any series, else 'mixed'."""
    if all(x <= 0.95 * y for x, y in zip(a, b)): return 'faster'
    if all(y <= 0.95 * x for x, y in zip(a, b)): return 'slower'
    if all(x > 0.95 * y and y > 0.95 * x for x, y in zip(a, b)): return 'comparable'
    return 'mixed'


def balance(k, n, m, sms=SMS):
    """The teammate's v1 balance rule (ours_v1.cu w4a16_v1_splits) on tc's grid: tiles = N/32 x ceil(M/16),
    load(S) = ceil(tiles*S/SMs)/S, S in [2, min(8, chunks/8)] taken only if load <= 0.9 load(1), smallest best S."""
    tiles = (n // 32) * ((m + 15) // 16)
    load = lambda s: math.ceil(tiles * s / sms) / s
    cutoff, best, best_load = 0.9 * load(1), 1, load(1)
    for s in range(2, min(8, (k // 64) // 8) + 1):
        if load(s) <= cutoff + 1e-9 and load(s) < best_load - 1e-9:
            best, best_load = s, load(s)
    return best


def splits(k, n):
    return sorted({1, 2, 4, 8, BALANCE[(k, n)]}) if (k, n) in SPLIT_SHAPES else []


def paths(k, n, extra=()):
    return BASE + ['splitk_s%d' % s for s in splits(k, n)] + list(extra)


def order(k, n, m, series, extra=()):
    names = paths(k, n, extra)
    shift = ((SHAPES.index((k, n)) if (k, n) in SHAPES else 0) + (MS.index(m) if m in MS else 0)) % len(names)
    names = names[shift:] + names[:shift]
    return names if series == 1 else names[::-1]


def split_s(name):
    return int(name[len('splitk_s'):]) if name.startswith('splitk_s') else None


def launch_splitk(lib, name, ptrs, stream):
    """Call the forced-S launcher with the S named by the path and require that the launcher used that S."""
    s = split_s(name)
    assert s is not None and s >= 1, name
    rc = lib.e059_tc_splitk(*ptrs, s, stream)
    assert rc == 0, (name, rc)
    assert lib.e059_tc_splitk_last_s() == s, ('S not forced', name, lib.e059_tc_splitk_last_s())


def violations(got, ref):
    """CONTRACT elementwise check: count of |got-ref| > 1e-3|ref| + 1e-3 rms(ref), non-finite got counts."""
    rms = math.sqrt(sum(r * r for r in ref) / len(ref))
    return sum(1 for g, r in zip(got, ref) if not math.isfinite(g) or abs(g - r) > 1e-3 * abs(r) + 1e-3 * rms)


def check_row(r, count=None, timed=False, seed0=101):
    """A row is usable only if every row of every matrix passed FP64 (before and after timing, bit-equal),
    v3 ran its own kernel (no v1 fallback), cuBLAS math mode 16, and the per-GEMM time divides by the count."""
    assert 'contract_failure' not in r, r.get('contract_failure')
    c = count or COUNTS[(r['K'], r['N'])]
    assert r['matrices'] == c and r['seeds'] == list(range(seed0, seed0 + c)) and r['activation_seed'] == seed0
    for key in ['validation'] + (['post_timing_validation'] if timed else []):
        assert len(r[key]) == c, key
        for v in r[key]:
            assert v['num_bad'] == 0 and v['checked'] == r['M'] * r['N'], v
    if timed:
        assert r['post_timing_validation'] == r['validation']
        assert len(r['samples_us']) == r['iters'] and abs(statistics.median(r['samples_us']) - r['median_us']) < 1e-9
        assert abs(r['median_per_matrix_us'] - per_gemm_us(r['samples_us'], c)) < 1e-9
    assert r['cublas_math_mode'] == 16 and r['output_tail_untouched'] and r['workspace_guards']
    if r['config'] == 'v3':
        assert r['info']['fast_path'] is True
    if split_s(r['config']):
        assert r['info']['S'] == split_s(r['config']) and r['info']['last_s'] == split_s(r['config'])


def _fr(cell, k, n, m, x, key):
    return [fraction(a, b) for a, b in zip(cell[(k, n, m, 'marlin')][key], cell[(k, n, m, x)][key])]


def _verdict(per):
    """per: {label: dict(verdict='holds'|'fails') or 'open ...'}; claim-level verdict over the scored labels."""
    h = [v['verdict'] == 'holds' for v in per.values() if isinstance(v, dict) and v['verdict'] in ('holds', 'fails')]
    opened = [s for s, v in per.items() if not isinstance(v, dict)]
    out = 'open' if not h else 'supported' if all(h) else 'contradicted' if not any(h) else 'partly supported (shape-dependent)'
    return out + ('; open on %s' % opened if opened else '')


def _largest_two(vals, i, reverse):
    return set(sorted(vals, key=lambda s: vals[s][i], reverse=reverse)[:2])


def readings(cell, key='cyc'):
    """Predeclared C1/C2 readings on one method's per-GEMM medians: cell[(K,N,M,path)] = {key: [s1, s2],
    key+'_valid': bool}. 'cyc' = cyclic (the result), 'rf' = the teammate's read flush (their method)."""
    ok = lambda k, n, m, *xs: all(cell[(k, n, m, x)][key + '_valid'] for x in xs)
    t = lambda k, n, m, x: cell[(k, n, m, x)][key]
    sh = lambda k, n: '%dx%d' % (k, n)
    out = dict(C1a={}, C1b={}, C2_tall_tc_vs_v3={}, C2_tall_fraction={}, C2_split={})
    for k, n in SHAPES:                                         # C1 at M = 1
        if not ok(k, n, 1, 'v3', 'marlin'):
            out['C1a'][sh(k, n)] = 'open (cache-affected cell)'
        else:
            vm = r1(t(k, n, 1, 'v3'), t(k, n, 1, 'marlin'))
            out['C1a'][sh(k, n)] = dict(v3_vs_marlin=vm, verdict=('holds' if vm == 'comparable' else 'fails')
                                        if (k, n) in CLAIM_MARLIN_TIE else 'descriptive (not claimed)')
        if not ok(k, n, 1, 'v3', 'tc_ladder'):
            out['C1b'][sh(k, n)] = 'open (cache-affected cell)'
        else:
            vt = r1(t(k, n, 1, 'v3'), t(k, n, 1, 'tc_ladder'))
            out['C1b'][sh(k, n)] = dict(v3_vs_tc_ladder=vt, claimed=CLAIM_V3_VS_TC[(k, n)],
                                        verdict='holds' if vt == CLAIM_V3_VS_TC[(k, n)] else 'fails')
    out['C1a_verdict'], out['C1b_verdict'] = _verdict(out['C1a']), _verdict(out['C1b'])
    a, b = out['C1a_verdict'].split(';')[0], out['C1b_verdict'].split(';')[0]
    out['C1_verdict'] = a if a == b else 'partly supported (C1a %s, C1b %s)' % (a, b)
    for m in MS:
        shapes = [s for s in SHAPES if ok(*s, m, 'tc_ladder', 'marlin', 'v3')]
        if not all(s in shapes for s in TALL) or len(shapes) < 3:
            out['C2_tall_tc_vs_v3'][m] = out['C2_tall_fraction'][m] = dict(verdict='open (cache-affected cells)')
        else:
            ratio = {s: [a / b for a, b in zip(t(*s, m, 'tc_ladder'), t(*s, m, 'v3'))] for s in shapes}
            frac = {s: _fr(cell, *s, m, 'tc_ladder', key) for s in shapes}
            out['C2_tall_tc_vs_v3'][m] = dict(tc_over_v3={sh(*s): v for s, v in ratio.items()}, verdict='supported'
                                             if all(_largest_two(ratio, i, True) == set(TALL) for i in (0, 1)) else 'not supported')
            out['C2_tall_fraction'][m] = dict(fractions={sh(*s): v for s, v in frac.items()}, verdict='supported'
                                             if all(_largest_two(frac, i, False) == set(TALL) for i in (0, 1)) else 'not supported')
        for k, n in TALL:
            sb = 'splitk_s%d' % BALANCE[(k, n)]
            label = '%dx%d M%d' % (k, n, m)
            if not ok(k, n, m, sb, 'tc_ladder', 'marlin', 'v3') or not ok(*CONTROL, m, 'tc_ladder', 'marlin'):
                out['C2_split'][label] = 'open (cache-affected cell)'; continue
            helps = r1(t(k, n, m, sb), t(k, n, m, 'tc_ladder'))
            closes = all(a >= b for a, b in zip(_fr(cell, k, n, m, sb, key), _fr(cell, *CONTROL, m, 'tc_ladder', key)))
            vs = [s for s in splits(k, n) if ok(k, n, m, 'splitk_s%d' % s)]
            best = min(vs, key=lambda s: sum(t(k, n, m, 'splitk_s%d' % s)))
            out['C2_split'][label] = dict(
                balance_S=BALANCE[(k, n)], balance_vs_tc=helps, balance_vs_v3=r1(t(k, n, m, sb), t(k, n, m, 'v3')),
                tc_vs_v3=r1(t(k, n, m, 'tc_ladder'), t(k, n, m, 'v3')),
                balance_fraction_of_marlin=_fr(cell, k, n, m, sb, key), tc_fraction_of_marlin=_fr(cell, k, n, m, 'tc_ladder', key),
                control_tc_fraction=_fr(cell, *CONTROL, m, 'tc_ladder', key), balance_reaches_control=closes,
                verdict='fixes' if helps == 'faster' and closes else 'helps, does not close' if helps == 'faster' else 'does not help',
                sweep_vs_tc={'S%d' % s: r1(t(k, n, m, 'splitk_s%d' % s), t(k, n, m, 'tc_ladder')) for s in vs},
                best_S_POST_HOC=best, best_fraction_POST_HOC=_fr(cell, k, n, m, 'splitk_s%d' % best, key))
    return out
