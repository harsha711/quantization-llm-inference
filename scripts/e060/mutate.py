#!/usr/bin/env python3
"""E060 mutation check: writes each mutant of scripts/e060/e060lib.py or scripts/e060/analyze.py to a temporary
directory, runs scripts/e060/test_e060.py against it (E060_LIB / E060_ANALYZE), and prints which tests fail.
Every mutant must fail at least one test; the unmutated copies must pass. Run from the repo root:
python3 scripts/e060/mutate.py"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
L, A = 'lib', 'analyze'
MUTANTS = {
    # cost and labels
    'inverted cost ratio': (L, "return t_patched / t_control - 1", "return t_control / t_patched - 1"),
    '5% boundary (<= instead of <)': (L, "return not abs(t_patched - t_control) < TOL * t_control",
                                      "return not abs(t_patched - t_control) <= TOL * t_control"),
    'material vs t_patched': (L, "return not abs(t_patched - t_control) < TOL * t_control",
                              "return not abs(t_patched - t_control) < TOL * t_patched"),
    'tolerance 10%': (L, "TOL = 0.05 ", "TOL = 0.10 "),
    'label from series 1 only': (L, "    elif sides[1] == sides[2] == 'none':", "    elif sides[1] == 'none':"),
    'any-series no-material label': (L, "    elif sides[1] == sides[2] == 'none':",
                                     "    elif 'none' in (sides[1], sides[2]):"),
    'any-series material-cost label': (L, "    elif sides[1] == sides[2] == 'cost':",
                                       "    elif 'cost' in (sides[1], sides[2]):"),
    'result needs one series': (L, "result = vs[1] and vs[2]", "result = vs[1] or vs[2]"),
    'series validity from control only': (L, "vs = {s: st[('control', s)] == fc.OK and st[('patched', s)] == fc.OK for s in (1, 2)}",
                                          "vs = {s: st[('control', s)] == fc.OK for s in (1, 2)}"),
    'swapped arm labels in cost': (L, "cs = {s: cost(t[('patched', s)], t[('control', s)]) for s in (1, 2)}",
                                   "cs = {s: cost(t[('control', s)], t[('patched', s)]) for s in (1, 2)}"),
    'overall any-cell rule': (L, "elif all(c['label'] == 'no material cost' for c in good.values()):",
                              "elif any(c['label'] == 'no material cost' for c in good.values()):"),
    'overall includes invalid cells': (L, "good = {key: c for key, c in cells.items() if c['result']}",
                                       "good = dict(cells)"),
    'range from series 1 only': (L, "costs = [c['cost'][s] for c in good.values() for s in (1, 2)]",
                                 "costs = [c['cost'][1] for c in good.values()]"),
    'material list from series 1 only': (L, "if any(c['sides'][s] != 'none' for s in (1, 2))",
                                         "if c['sides'][1] != 'none'"),
    'noise bound 3%': (L, "NOISE = 0.015 ", "NOISE = 0.03 "),
    # validity
    'floor check dropped': (L, "    return status(k, n, m, t, matrices) == fc.OK",
                            "    return status(k, n, m, t, matrices) != fc.BELOW_STRICT or True"),
    'BELOW_MEASURED accepted': (L, "vs = {s: st[('control', s)] == fc.OK and st[('patched', s)] == fc.OK for s in (1, 2)}",
                                "vs = {s: all(st[(a, s)] in (fc.OK, fc.BELOW_MEASURED) for a in ARMS) for s in (1, 2)}"),
    'WS_SMALL accepted': (L, "vs = {s: st[('control', s)] == fc.OK and st[('patched', s)] == fc.OK for s in (1, 2)}",
                          "vs = {s: all(st[(a, s)] in (fc.OK, fc.WS_SMALL) for a in ARMS) for s in (1, 2)}"),
    'FP16 byte model': (L, "KIND = fc.kind_of('marlin')", "KIND = fc.kind_of('cublas_dq')"),
    'matrices not passed to classify': (L, "return fc.classify(k, n, m, t, matrices=matrices, kind=KIND)",
                                        "return fc.classify(k, n, m, t, matrices=16, kind=KIND)"),
    # per-GEMM
    'wrong divisor (4)': (L, "return statistics.median(row['samples_us']) / row['matrices']",
                          "return statistics.median(row['samples_us']) / 4"),
    'mean not median': (L, "return statistics.median(row['samples_us']) / row['matrices']",
                        "return statistics.mean(row['samples_us']) / row['matrices']"),
    # plan
    'cycle sized to 1 x L2': (L, "L2_MARGIN = 2.0", "L2_MARGIN = 1.0"),
    '4096^2 with 8 matrices': (L, "COUNTS = {(4096, 4096): 16,", "COUNTS = {(4096, 4096): 8,"),
    'old seeds': (L, "SEEDS = list(range(201, 217))", "SEEDS = list(range(11, 27))"),
    # libraries and identity
    'swapped library hashes': (L, "ARM_SHA256 = {'control': '51cfcf3a8c3758631e2824d0be8f0da4c6a4cf29021a769ccf864afb4d1d652a',",
                               "ARM_SHA256 = {'patched': '51cfcf3a8c3758631e2824d0be8f0da4c6a4cf29021a769ccf864afb4d1d652a',"),
    'hash check dropped': (L, "    assert got == expected, (path, got, expected)\n", ""),
    'swapped library path': (L, "return os.path.join(root, arm, 'libe005.so')",
                             "return os.path.join(root, ARMS[1 - ARMS.index(arm)], 'libe005.so')"),
    'address check dropped': (L, "        assert address_in(addrs[arm], library_ranges(maps_text, paths[arm])), (arm, hex(addrs[arm]))\n", ""),
    'not-in-other check dropped': (L, "        assert not address_in(addrs[arm], library_ranges(maps_text, paths[other])), (arm, hex(addrs[arm]))\n", ""),
    'range end inclusive': (L, "return any(a <= addr < b for a, b in ranges)", "return any(a <= addr <= b for a, b in ranges)"),
    # order
    'series 2 not reversed (cells)': (L, "return list(first) if series == 1 else list(first[::-1])", "return list(first)"),
    'series 2 not reversed (shapes, M)': (L, "shapes, ms = (SHAPES, MS) if series == 1 else (SHAPES[::-1], MS[::-1])",
                                          "shapes, ms = (SHAPES, MS)"),
    'no rotation': (L, "first = ARMS if (SHAPES.index((k, n)) + MS.index(m)) % 2 == 0 else ARMS[::-1]", "first = ARMS"),
    # analyzer
    'no file-order check': (A, "    assert [(r['K'], r['N'], r['M'], r['arm']) for r in rs] == lib.file_order(s), f'series {s} file order'\n", ""),
    'no post-timing equality': (A, "        assert r['post_timing_validation'] == r['validation']\n", ""),
    'no row hash check': (A, "    assert r['library_sha256'] == lib.ARM_SHA256[r['arm']], (r['arm'], r['library_sha256'])\n", ""),
    'no row address check': (A, "    assert lib.address_in(r['kernel_address'], r['library_ranges']), (r['arm'], hex(r['kernel_address']))\n", ""),
    'no pair check': (A, "    check_pairs(rs)\n\n\ndef check_full", "\n\ndef check_full"),
    'no seed check': (A, "assert r['matrices'] == lib.COUNTS[(r['K'], r['N'])] and r['seeds'] == lib.seeds_for(r['K'], r['N'])",
                      "assert r['matrices'] == lib.COUNTS[(r['K'], r['N'])]"),
    'no activation-seed check': (A, "    assert r['activation_seed'] == lib.ACTIVATION_SEED\n", ""),
    'partial row check accepted': (A, "assert x['num_bad'] == 0 and x['checked'] == r['M']*r['N'], x", "assert x['num_bad'] == 0, x"),
    'no series-field check': (A, "assert r['series'] == s and r['iters'] == ITERS", "assert r['iters'] == ITERS"),
    'no warmup check': (A, "and r['warmup'] == WARMUP ", ""),
    'no recorded-median check': (A, "        assert abs(statistics.median(r['samples_us'])-r['median_us']) < 1e-9\n", ""),
    'bits compared on patched only': (A, "        if h == by[(k, n, m, 'patched')]:", "        if True:"),
    'series 2 analysed as series 1': (A, "    for s in (1, 2):\n        rs = series_rows[s]\n        check_series(rs, s)",
                                      "    for s in (1, 2):\n        rs = series_rows[s]\n        check_series(rs, 1) if s == 1 else None"),
}


def run(paths):
    env = dict(os.environ, E060_LIB=paths[L], E060_ANALYZE=paths[A])
    r = subprocess.run([sys.executable, '-m', 'unittest', 'scripts/e060/test_e060.py'], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return r.returncode, re.findall(r'^(?:FAIL|ERROR): (\S+)', r.stderr, re.M)


def main():
    src = {L: open(os.path.join(ROOT, 'scripts/e060/e060lib.py')).read(),
           A: open(os.path.join(ROOT, 'scripts/e060/analyze.py')).read()}
    # the temporary copy of e060lib must still find scripts/checks/floorcheck.py
    src[L] = src[L].replace("os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checks')",
                            repr(os.path.join(ROOT, 'scripts', 'checks')))
    assert repr(os.path.join(ROOT, 'scripts', 'checks')) in src[L]
    survivors = []
    with tempfile.TemporaryDirectory() as d:
        base = {k: os.path.join(d, f'unmutated_{k}.py') for k in src}
        for k in src:
            open(base[k], 'w').write(src[k])
        rc, failed = run(base)
        print(f"unmutated: {'OK' if rc == 0 else 'FAILED'} {failed}")
        assert rc == 0
        for i, (name, (which, a, b)) in enumerate(MUTANTS.items()):
            assert src[which].count(a) == 1, (name, src[which].count(a))
            paths = dict(base)
            paths[which] = os.path.join(d, f'm{i}_{which}.py')
            open(paths[which], 'w').write(src[which].replace(a, b))
            rc, failed = run(paths)
            print(f"{name}: {'KILLED' if rc else 'SURVIVED'} by {failed[:6]}{' ...' if len(failed) > 6 else ''}")
            if not rc:
                survivors.append(name)
    print(f"{len(MUTANTS)-len(survivors)}/{len(MUTANTS)} mutants killed")
    sys.exit(1 if survivors else 0)


if __name__ == '__main__':
    main()
