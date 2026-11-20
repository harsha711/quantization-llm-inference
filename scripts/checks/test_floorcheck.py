"""Tests for floorcheck.py, written BEFORE the implementation (2026-11-20).

Every expected number below is derived by hand in the comment next to it,
never by calling the code under test. Known-bad rows come from raw result
files that an independent audit found cache-served; they MUST be flagged.

Run: python3 -m unittest scripts/checks/test_floorcheck.py -v   (from repo root)
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import floorcheck as fc  # noqa: E402

E050 = os.path.join(ROOT, "results/e050_20261118/full-c8-343308")
E051 = os.path.join(ROOT, "results/e051_20261118/full-c8-343361/analysis.txt")


class Constants(unittest.TestCase):
    def test_constants(self):
        # E057 measured read-only bandwidth and mma.sync peak; RTX 5060 Ti L2.
        self.assertEqual(fc.BW_BYTES_PER_S, 427.2e9)
        self.assertEqual(fc.PEAK_FLOP_PER_S, 51.84e12)
        self.assertEqual(fc.L2_BYTES, 32 * 1024 * 1024)  # 33,554,432
        self.assertEqual(fc.L2_MARGIN, 2.0)


class ByteModel(unittest.TestCase):
    def test_w4_square_m1(self):
        # weights 4096*4096/2 = 8,388,608; scales 2*(4096/128)*4096 = 262,144
        # A 2*1*4096 = 8,192; C 2*1*4096 = 8,192  -> 8,667,136
        self.assertEqual(fc.bytes_model(4096, 4096, 1), 8_667_136)

    def test_w4_rect_m4(self):
        # K=4096 N=11008: weights 22,544,384; scales 2*32*11008 = 704,512
        # A 2*4*4096 = 32,768; C 2*4*11008 = 88,064 -> 23,369,728
        self.assertEqual(fc.bytes_model(4096, 11008, 4), 23_369_728)

    def test_w4_transposed_rect_m4(self):
        # K=11008 N=4096: weights 22,544,384; scales 2*(11008/128=86)*4096 = 704,512
        # A 2*4*11008 = 88,064; C 2*4*4096 = 32,768 -> 23,369,728 (same total)
        self.assertEqual(fc.bytes_model(11008, 4096, 4), 23_369_728)

    def test_fp16_weights(self):
        # cuBLAS on FP16 W-hat: weights 2*4096*11008 = 90,177,536, no scales
        # A 32,768 + C 88,064 -> 90,298,368
        self.assertEqual(fc.bytes_model(4096, 11008, 4, fp16_weights=True), 90_298_368)


class Floors(unittest.TestCase):
    def test_memory_floor_square_m1(self):
        # 8,667,136 B / 427.2e9 B/s = 20.2882 us
        self.assertAlmostEqual(fc.floor_us(4096, 4096, 1), 20.2882, places=3)

    def test_compute_floor_dominates_m128(self):
        # FLOP 2*128*4096*11008 = 11,542,724,608 / 51.84e12 = 222.66 us
        # bytes 22,544,384+704,512+1,048,576+2,818,048 = 27,115,520 -> 63.47 us
        self.assertAlmostEqual(fc.floor_us(4096, 11008, 128), 222.66, places=1)


class WorkingSet(unittest.TestCase):
    def test_square_four_matrices(self):
        # 4 * (8,388,608 + 262,144) = 34,603,008 B = 1.031 x L2
        self.assertEqual(fc.working_set(4096, 4096, 4), 34_603_008)

    def test_rect_four_matrices(self):
        # 4 * (22,544,384 + 704,512) = 92,995,584 B = 2.77 x L2
        self.assertEqual(fc.working_set(4096, 11008, 4), 92_995_584)


class Classify(unittest.TestCase):
    # ---- known-bad: must be flagged ----
    def test_e050_square_marlin_m4_below_floor(self):
        # E050 series 2 Marlin 4096^2 M4 15.13 us < 20.39 us floor
        self.assertEqual(fc.classify(4096, 4096, 4, 15.13, matrices=4), fc.BELOW_FLOOR)

    def test_e050_square_m16_above_floor_still_flagged(self):
        # tc_ladder s2 21.25 us > 20.9 us floor, but working set 1.03 x L2
        self.assertEqual(fc.classify(4096, 4096, 16, 21.25, matrices=4), fc.WS_SMALL)

    def test_e051_single_matrix_marlin(self):
        # E051 b x1 Marlin 27.9 us < 54.70 us floor (23,369,728 B / 427.2e9)
        self.assertEqual(fc.classify(4096, 11008, 4, 27.9, matrices=1), fc.BELOW_FLOOR)

    def test_e051_noflush_m1(self):
        # E051 c tc_ladder no-flush 17.696 us < 20.29 us floor, one matrix
        self.assertEqual(fc.classify(4096, 4096, 1, 17.696, matrices=1), fc.BELOW_FLOOR)

    def test_two_rect_matrices_ws_small(self):
        # 2 * 23,248,896 = 46,497,792 B = 1.39 x L2 < 2 x L2; time 62.2 above floor
        self.assertEqual(fc.classify(4096, 11008, 4, 62.2, matrices=2), fc.WS_SMALL)

    # ---- known-good: must pass ----
    def test_e050_rect_marlin_m4_ok(self):
        self.assertEqual(fc.classify(4096, 11008, 4, 61.3, matrices=4), fc.OK)

    def test_e050_rect_marlin_m128_ok(self):
        # 250.2 us > 222.66 us compute floor
        self.assertEqual(fc.classify(4096, 11008, 128, 250.2, matrices=4), fc.OK)

    def test_flushed_single_matrix_ok(self):
        # E051 c tc_ladder flush M1 33.12 us > 20.29; flushed: working set not applicable
        self.assertEqual(fc.classify(4096, 4096, 1, 33.12, flushed=True), fc.OK_FLUSHED)

    def test_flushed_below_floor_still_flagged(self):
        self.assertEqual(fc.classify(4096, 4096, 1, 15.0, flushed=True), fc.BELOW_FLOOR)

    def test_fp16_cublas_ok(self):
        # cuBLAS W-hat x1 271.6 us; FP16 floor 90,298,368/427.2e9 = 211.4 us -> above,
        # but one 90 MB matrix cannot be resident in a 32 MiB L2 -> OK
        self.assertEqual(fc.classify(4096, 11008, 4, 271.6, matrices=1, fp16_weights=True), fc.OK)

    def test_requires_matrices_or_flushed(self):
        with self.assertRaises(ValueError):
            fc.classify(4096, 4096, 1, 30.0)


class RealFiles(unittest.TestCase):
    """Adapters on raw result files: expectations from the audit, not from the code."""

    def test_e050_square_all_flagged_rects_all_ok(self):
        rows = fc.e050_rows(E050)
        # 2 series x 3 shapes x 4 M x 4 configs (marlin, tc, tc_ladder, cublas) = 96
        self.assertEqual(len(rows), 96)
        sq = [r for r in rows if r["K"] == 4096 and r["N"] == 4096]
        rect = [r for r in rows if not (r["K"] == 4096 and r["N"] == 4096)]
        self.assertEqual(len(sq), 32)
        w4 = [r for r in sq if r["config"] != "cublas_dq"]
        self.assertTrue(all(r["status"] in (fc.BELOW_FLOOR, fc.WS_SMALL) for r in w4))
        # cuBLAS on FP16 W-hat at 4096^2: 4 x 33,554,432 = 134 MB = 4 x L2 cycled,
        # M4 floor (33,554,432+32,768+32,768)/427.2e9 = 78.70 us < measured ~81.7 -> valid
        cub = [r for r in sq if r["config"] == "cublas_dq"]
        self.assertEqual(len(cub), 8)
        self.assertTrue(all(r["status"] == fc.OK for r in cub), cub)
        self.assertTrue(all(r["status"] == fc.OK for r in rect), [r for r in rect if r["status"] != fc.OK])

    def test_e050_known_value(self):
        # per-GEMM median = median(cycle samples)/4; Marlin 4096^2 M4 series 2 = 15.13 us
        rows = fc.e050_rows(E050)
        hit = [r for r in rows if r["K"] == 4096 and r["N"] == 4096 and r["M"] == 4
               and r["config"] == "marlin" and r["series"] == 2]
        self.assertEqual(len(hit), 1)
        self.assertAlmostEqual(hit[0]["time_us"], 15.13, places=1)

    def test_e051_b(self):
        rows = fc.e051_b_rows(E051)
        # 2 shapes x 3 counts (x1,x2,x4) x 2 series x 3 modelled configs
        # (marlin, tc_ladder, cublas_dq; unfused is not modelled) = 36
        self.assertEqual(len(rows), 36)
        x1 = [r for r in rows if r["matrices"] == 1 and r["config"] != "cublas_dq"]
        self.assertTrue(all(r["status"] == fc.BELOW_FLOOR for r in x1))  # 26-44 us < 54.7
        x2 = [r for r in rows if r["matrices"] == 2 and r["config"] != "cublas_dq"]
        self.assertTrue(all(r["status"] == fc.WS_SMALL for r in x2))
        x4 = [r for r in rows if r["matrices"] == 4]
        self.assertTrue(all(r["status"] == fc.OK for r in x4))

    def test_e051_c(self):
        rows = fc.e051_c_rows(E051)
        noflush = {(r["M"], r["config"]): r["status"] for r in rows if not r["flushed"]}
        flush = {(r["M"], r["config"]): r["status"] for r in rows if r["flushed"]}
        self.assertEqual(noflush[(1, "tc_ladder")], fc.BELOW_FLOOR)   # 17.696 < 20.29
        self.assertEqual(noflush[(4, "tc_ladder")], fc.BELOW_FLOOR)   # 17.712 < 20.39
        self.assertEqual(flush[(1, "tc_ladder")], fc.OK_FLUSHED)      # 33.12
        self.assertEqual(flush[(128, "tc_ladder")], fc.OK_FLUSHED)    # 129.3 > 82.85 compute floor (2*128*4096*4096 / 51.84e12)


if __name__ == "__main__":
    unittest.main()
