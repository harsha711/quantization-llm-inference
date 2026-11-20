#!/usr/bin/env python3
"""E059 mutation check: apply one deliberate bug at a time to a temporary copy of e059lib.py (or of the shared
scripts/checks/floorcheck.py) and require that test_e059.py FAILS for every mutation. Exit 0 only if all are caught.
    python3 scripts/e059/mutate.py [--checker PATH]   (default: scripts/checks/floorcheck.py; the job uses the
    committed one, so also run with --checker <(git show HEAD:scripts/checks/floorcheck.py) saved to a file)
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKER = Path(sys.argv[sys.argv.index('--checker')+1]) if '--checker' in sys.argv else HERE.parent/'checks'/'floorcheck.py'
MUTATIONS = [  # (label, file, original text, mutated text)
    ('drop floor check', 'e059lib.py', 'fc.classify(k, n, m, t_us,', 'fc.classify(k, n, m, 1e9,'),
    ('wrong byte model: FP16 bytes for every path', 'e059lib.py', "fp16 = name == 'cublas_dq'", 'fp16 = True'),
    ('wrong byte model: 4-bit bytes for cuBLAS', 'e059lib.py', "fp16 = name == 'cublas_dq'", 'fp16 = False'),
    ('wrong byte model: scales dropped (shared checker)', 'floorcheck.py', 'K * N // 2 + 2 * (K // GROUP) * N', 'K * N // 2'),
    ('wrong divide-by-matrices', 'e059lib.py', 'statistics.median(samples_us) / matrices', 'statistics.median(samples_us) / 1'),
    ('cycle sized to 1 x L2', 'e059lib.py', '< fc.L2_MARGIN * fc.L2_BYTES', '< fc.L2_BYTES'),
    ('S not actually forced', 'e059lib.py', 'lib.e059_tc_splitk(*ptrs, s, stream)', 'lib.e059_tc_splitk(*ptrs, 4, stream)'),
    ('launcher S not verified', 'e059lib.py', "assert lib.e059_tc_splitk_last_s() == s,", 'assert True or lib.e059_tc_splitk_last_s() == s,'),
    ('v3 fallback accepted', 'e059lib.py', "assert r['info']['fast_path'] is True", 'pass'),
    ('contract check: rms term dropped', 'e059lib.py', '1e-3 * abs(r) + 1e-3 * rms', '1e-3 * abs(r)'),
    ('contract check: rows not all required', 'e059lib.py', "v['checked'] == r['M'] * r['N']", "v['checked'] > 0"),
    ('fraction inverted', 'e059lib.py', 'return marlin_us / path_us', 'return path_us / marlin_us'),
    ('analyzer: cycle size ignored in floorcheck', 'analyze.py', "r['median_per_matrix_us'],matrices=r['matrices'])", "r['median_per_matrix_us'],matrices=1)"),
    ('analyzer: series value used as per-GEMM time', 'analyze.py', "c['cyc'][sr-1] = r['median_per_matrix_us']", "c['cyc'][sr-1] = r['median_us']"),
    # Amendment 2: the independent reviewer's survivors (and the new checks), each must be killed.
    ('C1 v3 vs tc from series 1 only', 'e059lib.py', "vt = r1(t(k, n, 1, 'v3'), t(k, n, 1, 'tc_ladder'))",
     "vt = r1(t(k, n, 1, 'v3')[:1], t(k, n, 1, 'tc_ladder')[:1])"),
    ('C1 combined verdict ignores C1b (tc condition)', 'e059lib.py', "out['C1_verdict'] = a if a == b else", "out['C1_verdict'] = a if True else"),
    ('C1b open-check ignores tc_ladder validity', 'e059lib.py', "if not ok(k, n, 1, 'v3', 'tc_ladder'):", "if not ok(k, n, 1, 'v3'):"),
    ('C1a scores the unclaimed 1536^2', 'e059lib.py', "if (k, n) in CLAIM_MARLIN_TIE else 'descriptive (not claimed)'", "if True else ''"),
    ('C2 closes in ANY series', 'e059lib.py', 'closes = all(a >= b', 'closes = any(a >= b'),
    ('C2 ignores control validity', 'e059lib.py', "or not ok(*CONTROL, m, 'tc_ladder', 'marlin')", 'or False'),
    ('C2 tall (tc/v3) from series 1 only', 'e059lib.py', "if all(_largest_two(ratio, i, True) == set(TALL) for i in (0, 1))",
     "if all(_largest_two(ratio, i, True) == set(TALL) for i in (0,))"),
    ('row check: seeds not checked', 'e059lib.py', "r['seeds'] == list(range(seed0, seed0 + c))", 'True'),
    ('row check: median not recomputed', 'e059lib.py', "abs(statistics.median(r['samples_us']) - r['median_us']) < 1e-9", 'True'),
    ('analyzer: cell valid if ANY series passes', 'analyze.py', "c['cyc_valid'] = all(L.valid(v) for v in c['cyc_status'])",
     "c['cyc_valid'] = any(L.valid(v) for v in c['cyc_status'])"),
    ('analyzer: per-cell order not checked', 'analyze.py', "assert [(r['order'],r['config']) for r in cr] == list(enumerate(order(k,n,m,s)))", 'assert True'),
    ('analyzer: file order not checked', 'analyze.py', "assert [(r['K'],r['N'],r['M'],r['config']) for r in rs] == expected_file_order(s)", 'assert True'),
    ('analyzer: S1 bits not compared', 'analyze.py', "if 'splitk_s1' in h and h['splitk_s1'] != h['tc_ladder']:", 'if False:'),
    ('analyzer: read-flush validation not compared', 'analyze.py', "assert rf['validation'] == r['validation'][:1]", 'assert True'),
    ('analyzer: read-flush iters not checked', 'analyze.py', "assert rf['iters'] == 200 and rf['warmup'] == 10 and len(rf['samples_us']) == 200", 'pass'),
    ('analyzer: read-flush median not recomputed', 'analyze.py', "assert abs(statistics.median(rf['samples_us'])-rf['median_us']) < 1e-9", 'pass'),
    ('analyzer: independent violations ignored', 'analyze.py', "assert r['info'].get('independent_violations') == 0", 'pass'),
    ('analyzer: lib/body identity ignored', 'analyze.py', "assert r['body_sha256'] == BODY and r['lib_sha256'] in lib", 'assert True'),
    ('analyzer: GPU identity ignored', 'analyze.py', "assert gpu == ['prereq',GPU_UUID,DRIVER,'/','now',GPU_UUID,DRIVER]", 'assert True'),
    ('analyzer: invalid cells not labelled', 'analyze.py', "+ ('' if c['cyc_valid'] else", "+ ('' if True else"),
    ('rule: 5% boundary', 'e059lib.py', 'x <= 0.95 * y for x, y', 'x < 0.95 * y for x, y'),
]


def run(tmp):
    return subprocess.run([sys.executable, '-m', 'unittest', 'test_e059'], cwd=tmp / 'e059',
                          capture_output=True, text=True)


def main():
    caught = 0
    print(f'checker: {CHECKER}')
    for label, name, old, new in MUTATIONS:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / 'e059').mkdir(); (tmp / 'checks').mkdir()
            for f in ('e059lib.py', 'test_e059.py', 'analyze.py'):
                shutil.copy(HERE / f, tmp / 'e059' / f)
            shutil.copy(CHECKER, tmp / 'checks' / 'floorcheck.py')
            path = tmp / ('checks' if name == 'floorcheck.py' else 'e059') / name
            text = path.read_text()
            assert text.count(old) == 1, (label, 'mutation site not unique/found')
            path.write_text(text.replace(old, new))
            r = run(tmp)
            summary = (r.stderr.strip().splitlines() or ['?'])[-1]
            ok = r.returncode != 0
            caught += ok
            print(f"{'CAUGHT' if ok else 'MISSED'}  {label:48s} -> {summary}")
    with tempfile.TemporaryDirectory() as t:   # unmutated copy must pass (the harness itself is sound)
        tmp = Path(t); (tmp / 'e059').mkdir(); (tmp / 'checks').mkdir()
        for f in ('e059lib.py', 'test_e059.py', 'analyze.py'):
            shutil.copy(HERE / f, tmp / 'e059' / f)
        shutil.copy(CHECKER, tmp / 'checks' / 'floorcheck.py')
        r = run(tmp)
        print(f"BASELINE (no mutation) -> {(r.stderr.strip().splitlines() or ['?'])[-1]}")
    print(f'{caught}/{len(MUTATIONS)} mutations caught')
    sys.exit(0 if caught == len(MUTATIONS) and r.returncode == 0 else 1)


if __name__ == '__main__':
    main()
