#!/usr/bin/env python3
"""E061 mutation check: each mutant of e061lib.py / analyze.py is written to a temp dir and the suite
(scripts/e061/test_e061.py) is run against it via E061_LIB / E061_ANALYZE. Every mutant must fail at least one
test; the unmutated copies must pass. Run from the repo root: python3 scripts/e061/mutate.py"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
L, A = 'lib', 'analyze'
MUTANTS = {
    'wrong divisor (8)': (L, "return statistics.median(row['samples_us']) / row['matrices']",
                          "return statistics.median(row['samples_us']) / 8"),
    'mean not median': (L, "return statistics.median(row['samples_us']) / row['matrices']",
                        "return statistics.mean(row['samples_us']) / row['matrices']"),
    '4-bit byte model': (L, "KIND = fc.kind_of(CONFIG)", "KIND = 'w4'"),
    'flop without factor 2': (L, "return 2 * m * K * N", "return m * K * N"),
    'ridge inverted': (L, "return 'BW' if intensity(m) < fc.PEAK", "return 'TC' if intensity(m) < fc.PEAK"),
    'share always vs compute roof': (L, "    if bound(m) == 'BW':", "    if False:"),
    'cross tolerance 5%': (L, "CROSS_TOL = 0.03", "CROSS_TOL = 0.05"),
    'cross one-sided': (L, "return r, abs(r - 1) <= CROSS_TOL", "return r, r - 1 <= CROSS_TOL"),
    'cross ratio inverted': (L, "r = t_new / t_e050", "r = t_e050 / t_new"),
    'series 2 not reversed': (L, "return list(MS) if series == 1 else list(MS[::-1])", "return list(MS)"),
    'order not checked': (L, "assert [r['M'] for r in rows] == m_order(series)", "assert True or [r['M'] for r in rows] == m_order(series)"),
    'BELOW_MEASURED accepted': (L, "invalid = sorted((k for k, c in cells.items() if c['status'] != fc.OK)",
                                "invalid = sorted((k for k, c in cells.items() if c['status'] not in (fc.OK, fc.BELOW_MEASURED))"),
    'validity from series 1 only': (L, "invalid = sorted((k for k, c in cells.items() if c['status'] != fc.OK)",
                                    "invalid = sorted((k for k, c in cells.items() if c['status'] != fc.OK and k.endswith('s1'))"),
    'valid() ignores floor': (L, "    return status(m, t_us) == fc.OK", "    return status(m, t_us) != fc.WS_SMALL"),
    'old seeds': (L, "SEEDS = [301, 302, 303, 304]", "SEEDS = [7, 8, 9, 10]"),
    '8 matrices': (L, "MATRICES = 4\n", "MATRICES = 8\n"),
    'capacity 512 (no guard rows)': (L, "CAPACITY = 520", "CAPACITY = 512"),
    'analyzer always exits 0': (A, "sys.exit(0 if s['all_valid'] else 1)", "sys.exit(0)"),
    'analyzer exit ignores invalid': (A, "sys.exit(0 if s['all_valid'] else 1)", "sys.exit(0 if s['cells'] else 1)"),
    'sample count not checked': (L, "assert len(r['samples_us']) == SAMPLES, r['M']", "pass"),
    'validation not checked': (L, "assert len(r['validation']) == MATRICES and len(r['post_timing_validation']) == MATRICES, r['M']", "pass"),
    'math mode / capacity not checked': (L, "assert (r['capacity'], r['cublas_math_mode'], r['warmup'], r['iters']) == (CAPACITY, 16, WARMUP, SAMPLES)", "pass"),
    'config/series not checked': (L, "assert (r['K'], r['N'], r['config'], r['series']) == (K, N, CONFIG, series), r['M']", "pass"),
    'seeds/matrices not checked': (L, "assert (r['matrices'], r['seeds'], r['activation_seed']) == (MATRICES, SEEDS, ACTIVATION_SEED), r['M']", "pass"),
    'cross-check vs series 1 E050 always': (L, "ratio, ok = cross_check(cells[f'128|s{s}']['us'], e050[s])",
                                            "ratio, ok = cross_check(cells[f'128|s{s}']['us'], e050[1])"),
    'plan margin 1 x L2': (L, "assert working_set_bytes() >= fc.L2_MARGIN * l2_bytes", "assert working_set_bytes() >= l2_bytes"),
    'E050 wrong M': (A, "(lib.K, lib.N, 128, lib.CONFIG)", "(lib.K, lib.N, 64, lib.CONFIG)"),
    'E050 wrong shape': (A, "(lib.K, lib.N, 128, lib.CONFIG)", "(lib.N, lib.K, 128, lib.CONFIG)"),
    'analyzer swaps series files': (A, "return lib.summarize(_rows(os.path.join(d, 'series1.jsonl')), _rows(os.path.join(d, 'series2.jsonl')), e050_m128())",
                                    "return lib.summarize(_rows(os.path.join(d, 'series2.jsonl')), _rows(os.path.join(d, 'series1.jsonl')), e050_m128())"),
}


def run(lib_path, ana_path):
    env = dict(os.environ, E061_LIB=lib_path, E061_ANALYZE=ana_path, PYTHONPATH=os.path.join(ROOT, 'scripts', 'checks'), E061_ROOT=ROOT)
    p = subprocess.run([sys.executable, '-m', 'unittest', 'scripts/e061/test_e061.py'], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return p.returncode, p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ''


def main():
    src = {L: open(os.path.join(ROOT, 'scripts/e061/e061lib.py')).read(),
           A: open(os.path.join(ROOT, 'scripts/e061/analyze.py')).read()}
    with tempfile.TemporaryDirectory() as d:
        paths = {L: os.path.join(d, 'e061lib.py'), A: os.path.join(d, 'analyze.py')}
        for key in src:
            open(paths[key], 'w').write(src[key])
        rc, last = run(paths[L], paths[A])
        assert rc == 0, f'unmutated copies fail: {last}'
        print('unmutated: pass')
        killed = 0
        for name, (which, old, new) in MUTANTS.items():
            assert src[which].count(old) == 1, f'{name}: pattern found {src[which].count(old)} times'
            open(paths[which], 'w').write(src[which].replace(old, new))
            rc, last = run(paths[L], paths[A])
            open(paths[which], 'w').write(src[which])
            killed += rc != 0
            print(f"{'KILLED ' if rc else 'SURVIVED'} {name}: {last}")
        print(f'{killed}/{len(MUTANTS)} killed')
        sys.exit(0 if killed == len(MUTANTS) else 1)


if __name__ == '__main__':
    main()
