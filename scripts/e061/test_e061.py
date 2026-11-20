"""Tests for E061 (scripts/e061/e061lib.py and scripts/e061/analyze.py), written BEFORE the implementation
(2026-11-20, user's test-driven rule). Every expected number is derived by hand in the comment next to it or
read from a raw file another experiment recorded, never by calling the code under test.

E061: cuBLAS on FP16 W_hat (cublas_dq) at K=4096, N=11008, M 128/256/512, 4 matrices per cycle, two series.
Logic under test: plan constants, working set >= 2 x L2, series order, per-GEMM = median / matrices, intensity,
TFLOP/s, roof share (bandwidth below the ridge, Tensor Core above), floorcheck validity (kind 'fp16'), the M128
cross-check against E050, and the analyzer's row checks and summary.

Run (repo root):  python3 -m unittest scripts/e061/test_e061.py -v
Mutation runs:    E061_LIB=... E061_ANALYZE=... python3 -m unittest scripts/e061/test_e061.py
"""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
LIB = os.environ.get('E061_LIB', os.path.join(HERE, 'e061lib.py'))
ANA = os.environ.get('E061_ANALYZE', os.path.join(HERE, 'analyze.py'))
os.environ['E061_LIB'] = LIB  # analyze.py imports the same (possibly mutated) library
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'checks'))
import floorcheck as fc  # noqa: E402  only for its status constants


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lib = load('e061lib', LIB)
an = load('e061_analyze', ANA)
L2 = 32 * 1024 * 1024  # 33,554,432 B


class Plan(unittest.TestCase):
    def test_constants(self):
        self.assertEqual((lib.K, lib.N), (4096, 11008))
        self.assertEqual(lib.MS, [128, 256, 512])
        self.assertEqual(lib.MATRICES, 4)
        self.assertEqual(lib.CAPACITY, 520)   # 512 + 8 guard rows: the untouched-tail check is not empty at M512
        self.assertEqual(lib.SEEDS, [301, 302, 303, 304])   # fresh, never used before
        self.assertEqual(lib.ACTIVATION_SEED, 301)
        self.assertEqual(lib.CONFIG, 'cublas_dq')

    def test_working_set(self):
        # FP16 W_hat: 4 x 2 x 4096 x 11008 = 4 x 90,177,536 = 360,710,144 B (10.75 x L2)
        self.assertEqual(lib.working_set_bytes(), 360_710_144)
        lib.check_plan(L2)                       # >= 2 x L2: passes
        with self.assertRaises(AssertionError):
            lib.check_plan(200 * 1024 * 1024)    # 2 x 209.7 MB = 419 MB > 360.7 MB: must fail

    def test_order(self):
        self.assertEqual(lib.m_order(1), [128, 256, 512])
        self.assertEqual(lib.m_order(2), [512, 256, 128])


class Arithmetic(unittest.TestCase):
    def test_per_gemm(self):
        # median(400, 404, 408) = 404; / 4 matrices = 101.0 us
        self.assertAlmostEqual(lib.per_gemm_us(dict(samples_us=[408, 400, 404], matrices=4)), 101.0)
        # median of an even count: (400 + 404) / 2 = 402; / 4 = 100.5 (median, not mean of 403.5/4)
        self.assertAlmostEqual(lib.per_gemm_us(dict(samples_us=[400, 404, 410, 0], matrices=4)), 100.5)

    def test_flop(self):
        # 2 x 256 x 4096 x 11008 = 23,085,449,216
        self.assertEqual(lib.flop(256), 23_085_449_216)

    def test_intensity(self):
        # bytes = 2KN + 2MK + 2MN (W_hat, A, C), FLOP = 2MKN
        # M128: 11,542,724,608 / (90,177,536 + 1,048,576 + 2,818,048 = 94,044,160) = 122.737
        # M256: 23,085,449,216 / 97,910,784 = 235.780 ; M512: 46,170,898,432 / 105,644,032 = 437.042
        self.assertAlmostEqual(lib.intensity(128), 122.737, places=2)
        self.assertAlmostEqual(lib.intensity(256), 235.780, places=2)
        self.assertAlmostEqual(lib.intensity(512), 437.042, places=2)

    def test_tflops(self):
        # 23,085,449,216 FLOP / 500 us = 46.1709 TFLOP/s
        self.assertAlmostEqual(lib.tflops(256, 500.0), 46.1709, places=3)

    def test_bound_and_share(self):
        # ridge = 51.84e12 / 427.2e9 = 121.35 FLOP/byte
        # M64 (hypothetical): 5,771,362,304 / (90,177,536 + 524,288 + 1,409,024 = 92,110,848) = 62.66 -> BW
        self.assertEqual(lib.bound(64), 'BW')
        self.assertEqual(lib.bound(128), 'TC')   # 122.74 > 121.35 (just past the ridge)
        self.assertEqual(lib.bound(512), 'TC')
        # M256, 500 us: 46.1709 / 51.84 = 89.065 %
        self.assertAlmostEqual(lib.roof_share_pct(256, 500.0), 89.065, places=2)
        # E050 series 1 cuBLAS M128 257.54 us: 11,542,724,608 / 257.54e-6 = 44.819 TFLOP/s = 86.46 % of 51.84
        # (the existing slide's "86% TC")
        self.assertAlmostEqual(lib.roof_share_pct(128, 257.54), 86.46, places=1)
        # M64 hypothetical 250 us: bytes 92,110,848 / 250e-6 = 368.44 GB/s = 86.25 % of 427.2
        self.assertAlmostEqual(lib.roof_share_pct(64, 250.0), 86.25, places=1)


class Validity(unittest.TestCase):
    # floors (fp16, M256): measured max(97,910,784 B / 427.2e9 = 229.19 us, 23,085,449,216 / 51.84e12 = 445.32 us)
    # = 445.32 us; strict max(90,177,536 / 448e9 = 201.29, 23,085,449,216 / 56.95e12 = 405.33) = 405.33 us
    def test_ok(self):
        self.assertEqual(lib.status(256, 500.0), fc.OK)
        self.assertTrue(lib.valid(256, 500.0))

    def test_below_measured(self):
        self.assertEqual(lib.status(256, 440.0), fc.BELOW_MEASURED)   # 405.33 < 440 < 445.32
        self.assertFalse(lib.valid(256, 440.0))

    def test_below_strict(self):
        self.assertEqual(lib.status(256, 400.0), fc.BELOW_STRICT)     # < 405.33
        self.assertFalse(lib.valid(256, 400.0))

    def test_m512_floor(self):
        # 46,170,898,432 / 51.84e12 = 890.64 us: 885 us is below the measured floor
        self.assertEqual(lib.status(512, 885.0), fc.BELOW_MEASURED)
        self.assertEqual(lib.status(512, 1000.0), fc.OK)

    def test_m128_between_floors(self):
        # fp16 strict M128: max(201.29, 11,542,724,608 / 56.95e12 = 202.66) = 202.66 us; measured 222.66 us
        self.assertEqual(lib.status(128, 210.0), fc.BELOW_MEASURED)
        self.assertEqual(lib.status(128, 200.0), fc.BELOW_STRICT)
        # (At M >= 128 the compute floor governs for both byte models, so the fp16 model is pinned by the
        # intensity test instead: 4-bit bytes would give 11,542,724,608 / 27,115,520 = 425.69, not 122.74.)


class CrossCheck(unittest.TestCase):
    def test_consistent(self):
        r, ok = lib.cross_check(260.0, 257.54)       # 260 / 257.54 = 1.00955 -> within 3 %
        self.assertAlmostEqual(r, 1.00955, places=4)
        self.assertTrue(ok)

    def test_inconsistent(self):
        r, ok = lib.cross_check(270.0, 257.54)       # 1.0484 -> outside 3 %
        self.assertFalse(ok)
        r, ok = lib.cross_check(249.0, 257.54)       # 0.96684 -> outside 3 % on the low side
        self.assertFalse(ok)

    def test_near_boundary(self):
        # 2.9 %: 257.54 x 1.029 = 265.00866 -> consistent; 3.1 %: 257.54 x 1.031 = 265.52374 -> not
        # (an exact 3.00 % is not representable in floating point, so the boundary is tested on both sides)
        self.assertTrue(lib.cross_check(265.00866, 257.54)[1])
        self.assertFalse(lib.cross_check(265.52374, 257.54)[1])
        self.assertTrue(lib.cross_check(257.54 * 0.971, 257.54)[1])
        self.assertFalse(lib.cross_check(257.54 * 0.969, 257.54)[1])

    def test_e050_reference(self):
        # E050 full-c8-343308 cuBLAS 4096x11008 M128: series 1 257.540 us, series 2 257.304 us (raw samples / 4)
        ref = an.e050_m128()
        self.assertAlmostEqual(ref[1], 257.540, places=2)
        self.assertAlmostEqual(ref[2], 257.304, places=2)


def row(m, series, us=None, **kw):
    """A synthetic bench row; per-GEMM = `us` (samples all equal to 4 x us)."""
    us = us if us is not None else {128: 258.0, 256: 500.0, 512: 1000.0}[m]
    r = dict(K=4096, N=11008, M=m, capacity=520, matrices=4, seeds=[301, 302, 303, 304], activation_seed=301,
             config='cublas_dq', series=series, cublas_math_mode=16, warmup=10, iters=100,
             samples_us=[4 * us] * 100, validation=[{'num_bad': 0}] * 4, post_timing_validation=[{'num_bad': 0}] * 4)
    r.update(kw)
    return r


def series(s, override=None):
    override = override or {}
    return [row(m, s, **override.get(m, {})) for m in lib.m_order(s)]


class Rows(unittest.TestCase):
    def test_good(self):
        lib.check_rows(series(1), 1)
        lib.check_rows(series(2), 2)

    def bad(self, rows, s):
        with self.assertRaises(AssertionError):
            lib.check_rows(rows, s)

    def test_wrong_order(self):
        self.bad(series(1)[::-1], 1)          # series 1 in series 2's order
        self.bad(series(2)[::-1], 2)

    def test_missing_row(self):
        self.bad(series(1)[:2], 1)

    def test_wrong_fields(self):
        for key, value in [('config', 'marlin'), ('matrices', 8), ('K', 11008), ('N', 4096), ('series', 2),
                           ('cublas_math_mode', 0), ('seeds', [7, 8, 9, 10]), ('activation_seed', 7),
                           ('capacity', 512), ('warmup', 0), ('iters', 50)]:
            rows = series(1)
            rows[1][key] = value
            self.bad(rows, 1)

    def test_sample_count(self):
        rows = series(1)
        rows[0]['samples_us'] = rows[0]['samples_us'][:99]
        self.bad(rows, 1)

    def test_validation_missing(self):
        for key in ('validation', 'post_timing_validation'):
            rows = series(1)
            rows[2][key] = []
            self.bad(rows, 1)


class Summary(unittest.TestCase):
    def setUp(self):
        self.e050 = {1: 257.54, 2: 257.304}

    def test_all_valid(self):
        s = lib.summarize(series(1), series(2), self.e050)
        self.assertTrue(s['all_valid'])
        c = s['cells']['256|s1']
        self.assertAlmostEqual(c['us'], 500.0)
        self.assertEqual(c['status'], fc.OK)
        self.assertAlmostEqual(c['tflops'], 46.1709, places=3)
        self.assertAlmostEqual(c['share_pct'], 89.065, places=2)
        self.assertEqual(c['bound'], 'TC')
        self.assertAlmostEqual(c['flop_per_byte'], 235.780, places=2)
        # cross-check per series: 258 / 257.54 = 1.00179 ; 258 / 257.304 = 1.00270
        self.assertAlmostEqual(s['cross_check']['s1']['ratio'], 1.00179, places=4)
        self.assertAlmostEqual(s['cross_check']['s2']['ratio'], 1.00270, places=4)
        self.assertTrue(s['cross_check']['s1']['consistent'] and s['cross_check']['s2']['consistent'])
        self.assertEqual(len(s['cells']), 6)

    def test_series_2_used(self):
        # series 2 M512 at 1100 us must appear in s2, not s1
        s = lib.summarize(series(1), series(2, {512: dict(us=1100.0)}), self.e050)
        self.assertAlmostEqual(s['cells']['512|s2']['us'], 1100.0)
        self.assertAlmostEqual(s['cells']['512|s1']['us'], 1000.0)

    def test_invalid_cell(self):
        s = lib.summarize(series(1, {256: dict(us=440.0)}), series(2), self.e050)
        self.assertFalse(s['all_valid'])
        self.assertEqual(s['cells']['256|s1']['status'], fc.BELOW_MEASURED)
        self.assertEqual(s['invalid'], ['256|s1'])

    def test_invalid_in_series_2_only(self):
        s = lib.summarize(series(1), series(2, {512: dict(us=885.0)}), self.e050)
        self.assertFalse(s['all_valid'])
        self.assertEqual(s['invalid'], ['512|s2'])

    def test_floor_fields(self):
        # summary floors, fp16, M256: measured 445.32 us, strict 405.33 us (hand-derived in Validity)
        c = lib.summarize(series(1), series(2), self.e050)['cells']['256|s2']
        self.assertAlmostEqual(c['floor_us'], 445.32, places=2)
        self.assertAlmostEqual(c['strict_floor_us'], 405.33, places=2)

    def test_cross_check_flag(self):
        s = lib.summarize(series(1, {128: dict(us=270.0)}), series(2), self.e050)
        self.assertFalse(s['cross_check']['s1']['consistent'])
        self.assertTrue(s['cross_check']['s2']['consistent'])
        self.assertTrue(s['all_valid'])   # the cross-check is not a validity criterion


class Analyzer(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            for s in (1, 2):
                with open(os.path.join(d, f'series{s}.jsonl'), 'w') as fh:
                    for r in series(s):
                        fh.write(json.dumps(r) + '\n')
            s = an.analyze(d)
            self.assertTrue(s['all_valid'])
            self.assertAlmostEqual(s['cells']['512|s1']['tflops'], 46.1709, places=3)   # 46,170,898,432 / 1000 us
            self.assertAlmostEqual(s['cross_check']['s1']['e050_us'], 257.540, places=2)

    def test_end_to_end_rejects_bad_order(self):
        with tempfile.TemporaryDirectory() as d:
            for s in (1, 2):
                rows = series(s)
                if s == 2:
                    rows = rows[::-1]
                with open(os.path.join(d, f'series{s}.jsonl'), 'w') as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + '\n')
            with self.assertRaises(AssertionError):
                an.analyze(d)


def write(d, s1, s2):
    for s, rows in ((1, s1), (2, s2)):
        if rows is None:
            continue
        with open(os.path.join(d, f'series{s}.jsonl'), 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')


class MissingSeries(unittest.TestCase):
    def test_empty_series2(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, series(1), [])
            with self.assertRaises(AssertionError):
                an.analyze(d)

    def test_absent_series2(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, series(1), None)
            with self.assertRaises(FileNotFoundError):
                an.analyze(d)


class CommandLine(unittest.TestCase):
    """The job records the analyzer's exit code: 0 only if every cell is floorcheck OK."""
    def cli(self, d):
        import subprocess
        env = dict(os.environ, E061_LIB=LIB, PYTHONPATH=os.path.join(ROOT, 'scripts', 'checks'))
        return subprocess.run([sys.executable, ANA, d], env=env, capture_output=True, text=True).returncode

    def test_exit_0_all_valid(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, series(1), series(2))
            self.assertEqual(self.cli(d), 0)

    def test_exit_1_invalid_cell(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, series(1), series(2, {256: dict(us=440.0)}))   # BELOW_MEASURED (445.32 us floor)
            self.assertEqual(self.cli(d), 1)

    def test_exit_nonzero_bad_order(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, series(1), series(2)[::-1])
            self.assertNotEqual(self.cli(d), 0)


if __name__ == '__main__':
    unittest.main()
