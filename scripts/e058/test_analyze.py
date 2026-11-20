"""Tests for scripts/e058/analyze.py, written BEFORE the floorcheck-based implementation (2026-11-20).

Every expected number is derived by hand in the comment next to it, never by calling the code under test.
The shared checker scripts/checks/floorcheck.py (v2: kind=, BELOW_STRICT / BELOW_MEASURED) supplies
floor_us/strict_floor_us/classify/kind_of; E058's own logic under test is: per-GEMM time = median(cycle samples)
/ matrices, cell validity = classify(...) == OK at 8 AND 16 matrices AND |t8/t16 - 1| <= 3%; both BELOW_STRICT
(below weights+scales / 448 GB/s spec or FLOP / 56.95 TFLOP/s at 3090 MHz: physically impossible) and
BELOW_MEASURED (below the full byte model / 427.2 GB/s or FLOP / 51.84 TFLOP/s: almost certainly cache) are invalid.

Why E058's floor differs from its first draft (predeclaration amendment 1): the draft used the memory floor
only; floorcheck uses max(memory, compute). At K=N=4096 M64/M128 the compute floor is larger
(2*128*4096^3/... see ComputeFloor below), so the draft would have accepted times the tensor rate forbids.

Run (repo root): python3 -m unittest scripts/e058/test_analyze.py -v
Mutation runs:   E058_ANALYZE=/path/to/mutated_analyze.py python3 -m unittest scripts/e058/test_analyze.py
"""
import importlib.util
import json
import os
import statistics
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
PATH = os.environ.get('E058_ANALYZE', os.path.join(HERE, 'analyze.py'))
spec = importlib.util.spec_from_file_location('e058_analyze', PATH)
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)

E050_S2 = os.path.join(ROOT, 'results/e050_20261118/full-c8-343308/series2.jsonl')
L2 = 32*1024*1024  # 33,554,432 B


def row(samples, matrices):
    return dict(samples_us=samples, matrices=matrices, median_us=statistics.median(samples))


class UsesSharedChecker(unittest.TestCase):
    def test_floorcheck_is_the_floor(self):
        # The analyzer must not carry its own constants: it imports scripts/checks/floorcheck.py.
        self.assertTrue(hasattr(an, 'fc'))
        self.assertEqual(an.fc.BW_BYTES_PER_S, 427.2e9)
        self.assertEqual(an.fc.L2_BYTES, L2)
        self.assertEqual(an.fc.STRICT_BW_BYTES_PER_S, 448e9)   # 128-bit GDDR7 at 28 Gb/s spec
        # 512 FLOP/clk/SM x 36 SMs x 3.090 GHz = 56,954,880,000,000 FLOP/s
        self.assertAlmostEqual(an.fc.STRICT_PEAK_FLOP_PER_S, 56.95488e12, delta=1e6)

    def test_kind_mapping(self):
        # cuBLAS rows use FP16 W_hat bytes; Marlin/tc/tc_ladder 4-bit
        self.assertEqual(an.kind('cublas_dq'), 'fp16')
        for x in ('marlin', 'tc', 'tc_ladder'):
            self.assertEqual(an.kind(x), 'w4')


class PerGemm(unittest.TestCase):
    def test_divides_by_matrices_16(self):
        # samples 390, 400, 410 -> median 400 us per cycle; 16 matrices -> 25.0 us per GEMM
        self.assertAlmostEqual(an.per_gemm_us(row([390.0, 400.0, 410.0], 16)), 25.0)

    def test_divides_by_matrices_8(self):
        # median 400 / 8 = 50.0 us (a constant divide-by-4, E050's count, would give 100)
        self.assertAlmostEqual(an.per_gemm_us(row([400.0, 400.0, 401.0], 8)), 50.0)

    def test_uses_median_not_mean(self):
        # samples 100, 100, 700: median 100, mean 300; 4 matrices -> 25.0 us
        self.assertAlmostEqual(an.per_gemm_us(row([100.0, 100.0, 700.0], 4)), 25.0)


class KnownBad(unittest.TestCase):
    """Cells that MUST be flagged."""

    def test_e050_marlin_m4_from_raw_file(self):
        # E050 series 2, Marlin, 4096^2, M4: per-GEMM = median(cycle)/4 = 15.13 us (E050 table 15.1).
        # Floor = (8,388,608 weights + 262,144 scales + 32,768 A + 32,768 C) = 8,716,288 B / 427.2e9 = 20.403 us.
        # Strict floor = 8,650,752 B (weights + scales) / 448e9 = 19.31 us; 15.13 < 19.31 -> BELOW_STRICT
        # (physically impossible from DRAM). Working set 4 x 8,650,752 = 34,603,008 B = 1.031 x L2.
        with open(E050_S2) as fh:
            r = [json.loads(x) for x in fh]
        hit = [x for x in r if (x['K'], x['N'], x['M'], x['config']) == (4096, 4096, 4, 'marlin')]
        self.assertEqual(len(hit), 1)
        t = an.per_gemm_us(hit[0])
        self.assertAlmostEqual(t, 15.13, delta=0.01)
        self.assertEqual(an.cell_status('marlin', 4, t, 4), an.fc.BELOW_STRICT)

    def test_below_floor_at_8_and_16_invalid_even_if_converged(self):
        # Marlin M4 15.13 at 8 and 15.10 at 16: converged (0.2%) but < 20.403 floor -> invalid
        v = an.cell_valid('marlin', 4, 15.13, 15.10)
        self.assertFalse(v['valid'])
        self.assertTrue(v['converged'])

    def test_below_measured_is_invalid(self):
        # Marlin M16 19.70 us (E050 s2 value) at 8 and 16 matrices: strict floor 8,650,752/448e9 = 19.31 us
        # < 19.70 < measured floor (8,650,752 + 2*16*4096*2 = 262,144) = 8,912,896/427.2e9 = 20.86 us
        # -> BELOW_MEASURED at both counts; converged (0%) but invalid.
        self.assertEqual(an.cell_status('marlin', 16, 19.70, 16), an.fc.BELOW_MEASURED)
        v = an.cell_valid('marlin', 16, 19.70, 19.70)
        self.assertTrue(v['converged'])
        self.assertFalse(v['valid'])

    def test_four_matrices_above_floor_is_ws_small(self):
        # 21.0 us >= 20.403 floor, but 4 matrices = 34,603,008 B < 2 x L2 = 67,108,864 B -> WS_SMALL
        self.assertEqual(an.cell_status('tc', 4, 21.0, 4), an.fc.WS_SMALL)

    def test_not_converged_invalid(self):
        # t8 23.0, t16 22.0: both >= 20.403 and 8 x 8,650,752 = 69,206,016 B = 2.0625 x L2 (OK),
        # but 23.0/22.0 - 1 = 4.545% > 3% -> invalid
        v = an.cell_valid('tc', 4, 23.0, 22.0)
        self.assertFalse(v['valid'])
        self.assertFalse(v['converged'])

    def test_convergence_just_outside(self):
        # t16 22.0, t8 22.682: 22.682/22.0 = 1.0310 -> 3.10% > 3% -> invalid
        self.assertFalse(an.cell_valid('tc', 4, 22.682, 22.0)['valid'])

    def test_compute_floor_m128(self):
        # tc_ladder M128 at 60 us: memory floor (8,650,752 + 2*128*4096*2 = 2,097,152) = 10,747,904 B
        # / 427.2e9 = 25.16 us, but compute floor 2*128*4096*4096 = 4,294,967,296 FLOP / 51.84e12 = 82.85 us.
        # Strict compute floor 4,294,967,296 / 56.95488e12 = 75.41 us; 60 < 75.41 -> BELOW_STRICT
        # (the draft's memory-only floor of 25.16 us would have passed it).
        self.assertEqual(an.cell_status('tc_ladder', 128, 60.0, 16), an.fc.BELOW_STRICT)
        self.assertFalse(an.cell_valid('tc_ladder', 128, 60.0, 60.5)['valid'])
        # 80 us: 75.41 < 80 < 82.85 -> BELOW_MEASURED, still invalid
        self.assertEqual(an.cell_status('tc_ladder', 128, 80.0, 16), an.fc.BELOW_MEASURED)
        self.assertFalse(an.cell_valid('tc_ladder', 128, 80.0, 80.5)['valid'])

    def test_cublas_uses_fp16_bytes(self):
        # cuBLAS M4 at 70 us: FP16 floor (33,554,432 + 32,768 + 32,768) = 33,619,968 B / 427.2e9 = 78.70 us.
        # Strict: 33,554,432 B / 448e9 = 74.90 us; 70 < 74.90 -> BELOW_STRICT (a 4-bit floor of 20.40 us
        # would wrongly accept it). 76 us: 74.90 < 76 < 78.70 -> BELOW_MEASURED.
        self.assertEqual(an.cell_status('cublas_dq', 4, 70.0, 16), an.fc.BELOW_STRICT)
        self.assertEqual(an.cell_status('cublas_dq', 4, 76.0, 16), an.fc.BELOW_MEASURED)


class KnownGood(unittest.TestCase):
    """Cells that must pass."""

    def test_tc_m4_valid(self):
        # 22.0 and 21.8 >= 20.403; 22.0/21.8 - 1 = 0.917% <= 3%; WS 2.06x / 4.1x L2 -> valid
        v = an.cell_valid('tc', 4, 22.0, 21.8)
        self.assertTrue(v['valid'])
        self.assertEqual(an.cell_status('tc', 4, 21.8, 16), an.fc.OK)

    def test_convergence_just_inside(self):
        # 22.638/22.0 = 1.0290 -> 2.90% <= 3% -> valid
        self.assertTrue(an.cell_valid('tc', 4, 22.638, 22.0)['valid'])

    def test_convergence_symmetric(self):
        # t8 below t16: 21.4/22.0 - 1 = -2.73%, |.| <= 3% -> valid (21.4 >= 20.403)
        self.assertTrue(an.cell_valid('tc', 4, 21.4, 22.0)['valid'])

    def test_cublas_m4_valid(self):
        # 86.0 / 86.5 >= 78.70 FP16 floor; 0.58% -> valid
        self.assertTrue(an.cell_valid('cublas_dq', 4, 86.0, 86.5)['valid'])

    def test_m128_valid_above_compute_floor(self):
        # 110.0 / 110.5 >= 82.85 compute floor; 0.45% -> valid
        self.assertTrue(an.cell_valid('tc_ladder', 128, 110.0, 110.5)['valid'])


class TableLevel(unittest.TestCase):
    def test_validity_counts(self):
        # 4 paths x 4 M x 2 series = 32 cells. All at 1.1 x floor_us(...) (hand: above floor, converged)
        # except two: (tc, M4, s1) t8 = 1.04 x t16 -> 4% > 3%; (marlin, M4, s2) t16 = 15.0 < 20.403.
        # Expected valid 30/32.
        base = {}
        hand_floor = {('w4', 4): 20.403, ('w4', 16): 20.863, ('w4', 64): 41.43, ('w4', 128): 82.85,
                      ('fp16', 4): 78.70, ('fp16', 16): 79.16, ('fp16', 64): 81.00, ('fp16', 128): 83.45}
        # hand: M64 compute 2*64*4096^2 = 2,147,483,648 / 51.84e12 = 41.43 us > memory 22.70;
        # cuBLAS M128 memory 35,651,584/427.2e9 = 83.45 us > compute 82.85.
        for x in an.NAMES:
            for m in an.MS:
                f = hand_floor[('fp16' if x == 'cublas_dq' else 'w4', m)]
                for s in (1, 2):
                    for c in an.COUNTS:
                        base[(c, m, x, s)] = 1.1*f
        base[(8, 4, 'tc', 1)] = 1.04*base[(16, 4, 'tc', 1)]
        base[(16, 4, 'marlin', 2)] = 15.0
        v = an.validity(base)
        self.assertEqual(sum(c['valid'] for c in v.values()), 30)
        self.assertFalse(v[('tc', 4, 1)]['valid'])
        self.assertFalse(v[('marlin', 4, 2)]['valid'])
        # the 4-matrix control is reported (WS_SMALL at 1.03 x L2), never part of validity
        self.assertEqual(v[('tc', 4, 2)]['status_4'], an.fc.WS_SMALL)


if __name__ == '__main__':
    unittest.main()
