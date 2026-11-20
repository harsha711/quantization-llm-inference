"""Tests for floorcheck.py, written BEFORE the implementation (v1 2026-11-20;
v2 the same day after the independent review: two-tier floor, unfused/dequant
models, E051 (a) and raw E051 (b) parsers, mutation gaps closed).

Every expected number is derived by hand in the comment next to it or taken
from a value an experiment's own analyzer/bench recorded, never by calling the
code under test. Known-bad rows come from raw files an independent audit found
cache-served; they MUST be flagged.

Run: python3 -m unittest scripts/checks/test_floorcheck.py -v   (from repo root)
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import floorcheck as fc  # noqa: E402

E050 = os.path.join(ROOT, "results/e050_20261118/full-c8-343308")
E051D = os.path.join(ROOT, "results/e051_20261118/full-c8-343361")
E051 = os.path.join(E051D, "analysis.txt")
BELOW = (fc.BELOW_STRICT, fc.BELOW_MEASURED)


class Constants(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(fc.BW_BYTES_PER_S, 427.2e9)         # E057 measured read
        self.assertEqual(fc.STRICT_BW_BYTES_PER_S, 448e9)     # spec 14 Gb/s x 128 bit... 448 GB/s
        self.assertEqual(fc.PEAK_FLOP_PER_S, 51.84e12)        # E057 measured at 2812 MHz
        # strict: 512 FLOP/clk/SM x 36 SM x 3.090 GHz (clocks.max.sm) = 56.95e12
        self.assertAlmostEqual(fc.STRICT_PEAK_FLOP_PER_S, 56.95e12, delta=0.01e12)
        self.assertEqual(fc.L2_BYTES, 32 * 1024 * 1024)
        self.assertEqual(fc.L2_MARGIN, 2.0)


class ByteModel(unittest.TestCase):
    def test_w4_square_m1(self):
        # weights 8,388,608 + scales 262,144 + A 8,192 + C 8,192 = 8,667,136
        self.assertEqual(fc.bytes_model(4096, 4096, 1), 8_667_136)

    def test_w4_rect_m4(self):
        # 22,544,384 + 704,512 + A 32,768 + C 88,064 = 23,369,728
        self.assertEqual(fc.bytes_model(4096, 11008, 4), 23_369_728)

    def test_fp16_weights(self):
        # 90,177,536 + 32,768 + 88,064 = 90,298,368
        self.assertEqual(fc.bytes_model(4096, 11008, 4, kind="fp16"), 90_298_368)

    def test_unfused(self):
        # 11008x4096 M4: 4-bit 23,248,896 read + W-hat 90,177,536 written + read again
        # + A 2*4*11008 = 88,064 + C 2*4*4096 = 32,768 -> 203,724,800
        self.assertEqual(fc.bytes_model(11008, 4096, 4, kind="unfused"), 203_724_800)

    def test_dequant(self):
        # 4096^2: read 8,650,752, write 33,554,432 -> 42,205,184 (no A, no C)
        self.assertEqual(fc.bytes_model(4096, 4096, 1, kind="dequant"), 42_205_184)

    def test_strict_bytes_omit_a_c(self):
        # weights + scales only: 8,650,752
        self.assertEqual(fc.strict_bytes(4096, 4096), 8_650_752)


class Floors(unittest.TestCase):
    def test_measured_floor_square_m1(self):
        # 8,667,136 / 427.2e9 = 20.2882 us
        self.assertAlmostEqual(fc.floor_us(4096, 4096, 1), 20.2882, places=3)

    def test_strict_floor_square(self):
        # 8,650,752 / 448e9 = 19.3097 us ; compute at M16 2*16*4096^2/56.95e12 = 9.4 us
        self.assertAlmostEqual(fc.strict_floor_us(4096, 4096, 16), 19.3097, places=3)

    def test_compute_floor_dominates_m128(self):
        # 2*128*4096*11008 = 11,542,724,608 / 51.84e12 = 222.66 us (> 63.47 memory)
        self.assertAlmostEqual(fc.floor_us(4096, 11008, 128), 222.66, places=1)

    def test_unfused_floor(self):
        # 203,724,800 / 427.2e9 = 476.88 us
        self.assertAlmostEqual(fc.floor_us(11008, 4096, 4, kind="unfused"), 476.88, places=1)


class WorkingSet(unittest.TestCase):
    def test_square_four(self):
        self.assertEqual(fc.working_set(4096, 4096, 4), 34_603_008)   # 4 x 8,650,752

    def test_rect_four(self):
        self.assertEqual(fc.working_set(4096, 11008, 4), 92_995_584)  # 4 x 23,248,896

    def test_unfused_one_temporary(self):
        # 4 x 23,248,896 + one reused 90,177,536 temporary = 183,173,120
        self.assertEqual(fc.working_set(4096, 11008, 4, kind="unfused"), 183_173_120)


class Classify(unittest.TestCase):
    def test_boundary_equal_floor_not_below(self):
        f = fc.floor_us(4096, 11008, 4)
        self.assertEqual(fc.classify(4096, 11008, 4, f, matrices=4), fc.OK)

    def test_e050_marlin_m4_strict(self):
        # 15.13 us < 8,650,752/448e9 = 19.31 us -> impossible from DRAM
        self.assertEqual(fc.classify(4096, 4096, 4, 15.13, matrices=4), fc.BELOW_STRICT)

    def test_e050_marlin_m16_s2_below_measured_only(self):
        # 19.70: above strict 19.31, below measured 20.86 -> BELOW_MEASURED
        self.assertEqual(fc.classify(4096, 4096, 16, 19.70, matrices=4), fc.BELOW_MEASURED)

    def test_e050_m16_above_floor_ws_small(self):
        self.assertEqual(fc.classify(4096, 4096, 16, 21.25, matrices=4), fc.WS_SMALL)

    def test_e051_single_matrix_marlin(self):
        # 27.9 < 23,248,896/448e9 = 51.90 us
        self.assertEqual(fc.classify(4096, 11008, 4, 27.9, matrices=1), fc.BELOW_STRICT)

    def test_two_rect_matrices_ws_small(self):
        # 2 x 23,248,896 = 46,497,792 = 1.39 x L2
        self.assertEqual(fc.classify(4096, 11008, 4, 62.2, matrices=2), fc.WS_SMALL)

    def test_rect_ok(self):
        self.assertEqual(fc.classify(4096, 11008, 4, 61.3, matrices=4), fc.OK)
        self.assertEqual(fc.classify(4096, 11008, 128, 250.2, matrices=4), fc.OK)

    def test_flushed(self):
        self.assertEqual(fc.classify(4096, 4096, 1, 33.12, flushed=True), fc.OK_FLUSHED)
        self.assertEqual(fc.classify(4096, 4096, 1, 15.0, flushed=True), fc.BELOW_STRICT)

    def test_fp16_single_90mb_ok(self):
        # 271.6 > 90,298,368/427.2e9 = 211.4; one 90 MB matrix = 2.7 x L2
        self.assertEqual(fc.classify(4096, 11008, 4, 271.6, matrices=1, kind="fp16"), fc.OK)

    def test_requires_matrices_or_flushed(self):
        with self.assertRaises(ValueError):
            fc.classify(4096, 4096, 1, 30.0)

    def test_margin(self):
        # unfused 11008x4096 x4 477.3 us / 476.88 floor = 1.0009
        self.assertAlmostEqual(fc.margin(11008, 4096, 4, 477.3, kind="unfused"), 1.0009, places=3)


class Parsers(unittest.TestCase):
    def test_divides_by_matrices_not_4(self):
        # synthetic 8-matrix cycle: samples 80,81,82 us -> median 81 / 8 = 10.125
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "series1.jsonl"), "w") as f:
                f.write(json.dumps(dict(K=4096, N=4096, M=4, matrices=8, config="tc",
                                        series=1, samples_us=[80, 82, 81])) + "\n")
            rows = fc.e050_rows(d)
        self.assertEqual(rows[0]["time_us"], 10.125)
        self.assertEqual(rows[0]["matrices"], 8)

    def test_kind_of_config(self):
        self.assertEqual(fc.kind_of("cublas"), "fp16")      # cuBLAS on original FP16 W
        self.assertEqual(fc.kind_of("cublas_dq"), "fp16")   # cuBLAS on W-hat
        self.assertEqual(fc.kind_of("unfused"), "unfused")
        self.assertEqual(fc.kind_of("dequant"), "dequant")
        self.assertEqual(fc.kind_of("tc_ladder"), "w4")
        self.assertEqual(fc.kind_of("marlin"), "w4")


class RealFiles(unittest.TestCase):
    def test_e050_matches_recorded_median(self):
        # bench recorded median_per_matrix_us; recomputed median(samples)/matrices must equal it
        rows = fc.e050_rows(E050)
        for r in rows:
            self.assertAlmostEqual(r["time_us"], r["recorded_us"], places=6)

    def test_e050_shape_labels(self):
        # E050 analyzer: tc_ladder/tc 11008x4096 M64 s1 = 0.87170, 4096x11008 M64 s1 = 0.82650
        rows = fc.e050_rows(E050)
        def t(K, N, cfg):
            return [r["time_us"] for r in rows if (r["K"], r["N"], r["M"], r["config"], r["series"]) == (K, N, 64, cfg, 1)][0]
        self.assertAlmostEqual(t(11008, 4096, "tc_ladder") / t(11008, 4096, "tc"), 0.87170, places=4)
        self.assertAlmostEqual(t(4096, 11008, "tc_ladder") / t(4096, 11008, "tc"), 0.82650, places=4)

    def test_e050_statuses(self):
        rows = fc.e050_rows(E050)
        self.assertEqual(len(rows), 96)  # 2 series x 3 shapes x 4 M x 4 configs
        sq = [r for r in rows if r["K"] == 4096 and r["N"] == 4096]
        w4 = [r for r in sq if r["config"] != "cublas_dq"]
        self.assertEqual(len(w4), 24)
        self.assertTrue(all(r["status"] in BELOW + (fc.WS_SMALL,) for r in w4))
        # cuBLAS W-hat 4096^2: 4 x 33.5 MB cycled, M4 floor 78.70 < ~81.7 -> valid
        self.assertTrue(all(r["status"] == fc.OK for r in sq if r["config"] == "cublas_dq"))
        rect = [r for r in rows if not (r["K"] == 4096 and r["N"] == 4096)]
        self.assertTrue(all(r["status"] == fc.OK for r in rect))

    def test_e051_b_raw(self):
        rows = fc.e051_b_rows(E051D)
        # 2 shapes x 3 counts x 4 configs (incl. unfused) x 2 series = 48
        self.assertEqual(len(rows), 48)
        for r in rows:
            self.assertAlmostEqual(r["time_us"], r["recorded_us"], places=6)
        w4x1 = [r for r in rows if r["matrices"] == 1 and r["config"] in ("marlin", "tc_ladder")]
        self.assertTrue(all(r["status"] == fc.BELOW_STRICT for r in w4x1))  # 26-44 < 51.9
        w4x2 = [r for r in rows if r["matrices"] == 2 and r["config"] in ("marlin", "tc_ladder")]
        self.assertTrue(all(r["status"] == fc.WS_SMALL for r in w4x2))
        x4 = [r for r in rows if r["matrices"] == 4 and r["config"] != "unfused"]
        self.assertTrue(all(r["status"] == fc.OK for r in x4))
        # unfused 11008x4096 x1 476.7 < 476.88 measured floor (above strict 454.5)
        u = [r for r in rows if r["config"] == "unfused" and r["K"] == 11008 and r["matrices"] == 1]
        self.assertTrue(any(r["status"] == fc.BELOW_MEASURED for r in u), u)

    def test_e051_c(self):
        rows = fc.e051_c_rows(E051)
        st = {(r["M"], r["config"], r["flushed"]): r["status"] for r in rows}
        self.assertEqual(st[(1, "tc_ladder", False)], fc.BELOW_STRICT)   # 17.696 < 19.31
        self.assertEqual(st[(4, "tc_ladder", False)], fc.BELOW_STRICT)   # 17.712
        self.assertEqual(st[(4, "cublas", False)], fc.BELOW_STRICT)      # 52.77 < 33,554,432/448e9 = 74.9
        self.assertEqual(st[(1, "cublas", False)], fc.WS_SMALL)          # 83.49 > 78.58; 1 x L2
        self.assertEqual(st[(128, "tc_ladder", False)], fc.WS_SMALL)     # 107.84 > 82.85; 1 matrix
        self.assertEqual(st[(1, "tc_ladder", True)], fc.OK_FLUSHED)      # 33.12
        self.assertEqual(st[(1, "naive", True)], fc.OK_FLUSHED)          # 205.152
        self.assertEqual(st[(128, "tc_ladder", True)], fc.OK_FLUSHED)    # 129.3 > 82.85

    def test_e051_a(self):
        rows = fc.e051_a_rows(E051D)
        self.assertEqual({r["source"] for r in rows},
                         {"a-main-s1.csv", "a-main-s2.csv", "a-held-s1.csv", "a-held-s2.csv"})
        st = {(r["K"], r["N"], r["M"], r["config"], r["flushed"]): r for r in rows
              if r["source"] == "a-main-s1.csv"}
        self.assertTrue(any(r["source"] == "a-held-s1.csv" for r in rows))
        # a-main-s1.csv: tc_ladder 4096^2 M1 noflush 17.984 < strict 19.31
        self.assertEqual(st[(4096, 4096, 1, "tc_ladder", False)]["status"], fc.BELOW_STRICT)
        # cublas_dq M32 noflush ~30.3 << FP16 strict 74.9
        self.assertEqual(st[(4096, 4096, 32, "cublas_dq", False)]["status"], fc.BELOW_STRICT)
        self.assertAlmostEqual(st[(4096, 4096, 1, "naive", True)]["time_us"], 205.152, places=3)
        self.assertEqual(st[(4096, 4096, 1, "tc_ladder", True)]["status"], fc.OK_FLUSHED)
        # held-out rectangles are parsed too
        self.assertTrue(any(r["K"] == 11008 for r in rows))


if __name__ == "__main__":
    unittest.main()
