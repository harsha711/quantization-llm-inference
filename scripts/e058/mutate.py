#!/usr/bin/env python3
"""E058 mutation check: writes each mutant of scripts/e058/analyze.py to a temporary directory, runs
scripts/e058/test_analyze.py against it (E058_ANALYZE=...), and prints which tests fail. Every mutant must fail
at least one test; the unmutated copy must pass. Run from the repo root: python3 scripts/e058/mutate.py"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
IMP = "import floorcheck as fc  # noqa: E402  shared DRAM/compute floor and working-set check (scripts/checks)\n"
V = "v['valid'] = v['status_8'] == fc.OK and v['status_16'] == fc.OK and v['converged']"
C = "converged=abs(t8-t16) <= CONVERGENCE_TOL*t16)"
MED = "return statistics.median(row['samples_us'])/row['matrices']"
MUTANTS = {
    'drop floor check': (V, "v['valid'] = v['converged']"),
    'drop convergence': (V, "v['valid'] = v['status_8'] == fc.OK and v['status_16'] == fc.OK"),
    'divide by 4': (MED, "return statistics.median(row['samples_us'])/4"),
    'memory-only floor': (IMP, IMP+"fc.PEAK_FLOP_PER_S = fc.STRICT_PEAK_FLOP_PER_S = float('inf')\n"),
    'no working-set rule': (IMP, IMP+"fc.L2_MARGIN = 0.0\n"),
    'mean not median': (MED, "return statistics.mean(row['samples_us'])/row['matrices']"),
    'BELOW_MEASURED accepted': (V, "v['valid'] = all(x in (fc.OK, fc.BELOW_MEASURED) for x in (v['status_8'],v['status_16'])) and v['converged']"),
    'cuBLAS with 4-bit bytes': ("    return fc.kind_of(name)", "    return 'w4'"),
    'drop 8-matrix status': (V, "v['valid'] = v['status_16'] == fc.OK and v['converged']"),
    'drop 16-matrix status': (V, "v['valid'] = v['status_8'] == fc.OK and v['converged']"),
    '<= 3% -> < 3%': (C, "converged=abs(t8-t16) < CONVERGENCE_TOL*t16)"),
    'inverted fraction': ("{f'{x}_fraction_of_marlin_speed': mr/med[(c,m,x,s)]", "{f'{x}_fraction_of_marlin_speed': med[(c,m,x,s)]/mr"),
    'ratio validity without Marlin': ("'tc_fraction': row['valid']['tc'] and row['valid']['marlin']", "'tc_fraction': row['valid']['tc']"),
    't8 classified with count 16': ("status_8=cell_status(name,m,t8,8)", "status_8=cell_status(name,m,t8,16)"),
    't4 reports the 8-matrix median': ("v.update(t4=med[(4,m,x,s)]", "v.update(t4=med[(8,m,x,s)]"),
    'no file-order check': ("        check_series_order(rs,s)\n", ""),
    'raw-ratio convergence': (C, "converged=abs(conv) <= CONVERGENCE_TOL)"),
    'no calibration note': ("v['note'] = NOTE_FLOOR_CALIBRATION if calib else None", "v['note'] = None"),
    # second review
    'cuBLAS ratio uses tc cell': ("'cublas_fraction': row['valid']['cublas_dq'] and", "'cublas_fraction': row['valid']['tc'] and"),
    'tc_ladder ratio uses tc cell': ("'tc_ladder_fraction': row['valid']['tc_ladder'] and", "'tc_ladder_fraction': row['valid']['tc'] and"),
    'label always VALID': ("    if v['valid']:\n        return 'VALID'\n", "    return 'VALID'\n"),
    'c16_over_e050 from 4 matrices': ("row['c16_over_e050'] = {x: med[(16,m,x,s)]", "row['c16_over_e050'] = {x: med[(4,m,x,s)]"),
    'no seed check': ("row['seeds'] == SEEDS[:c] and ", ""),
    'no same_bits in cells': ("        assert same_bits(rs) == 12\n    return out", "    return out"),
    'no post-timing equality': ("            assert r['post_timing_validation'] == r['validation']\n", ""),
    'no series field check': ("assert r['series'] == s and r['iters']", "assert r['iters']"),
    'calibration note on BELOW_STRICT': (" and \\\n        fc.BELOW_STRICT not in (v['status_8'],v['status_16'])", ""),
}


def run(path):
    env = dict(os.environ, E058_ANALYZE=path)
    r = subprocess.run([sys.executable, '-m', 'unittest', 'scripts/e058/test_analyze.py'], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return r.returncode, re.findall(r'^(?:FAIL|ERROR): (\S+)', r.stderr, re.M)


def main():
    src = open(os.path.join(ROOT, 'scripts/e058/analyze.py')).read()
    src = src.replace("Path(__file__).resolve().parents[1]/'checks'", repr(os.path.join(ROOT, 'scripts/checks')))
    src = src.replace("Path(__file__).resolve().parents[2]/'results", f"Path({ROOT!r})/'results")
    survivors = []
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, 'unmutated.py'); open(base, 'w').write(src)
        rc, failed = run(base)
        print(f"unmutated: {'OK' if rc == 0 else 'FAILED'} {failed}")
        assert rc == 0
        for i, (name, (a, b)) in enumerate(MUTANTS.items()):
            assert src.count(a) == 1, name
            p = os.path.join(d, f'm{i}.py'); open(p, 'w').write(src.replace(a, b))
            rc, failed = run(p)
            print(f"{name}: {'KILLED' if rc else 'SURVIVED'} by {failed}")
            if not rc:
                survivors.append(name)
    print(f"{len(MUTANTS)-len(survivors)}/{len(MUTANTS)} mutants killed")
    sys.exit(1 if survivors else 0)


if __name__ == '__main__':
    main()
