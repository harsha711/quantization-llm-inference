"""Tests for anchors.py (presentation anchor numbers), written BEFORE anchors.py.

anchors.py recomputes every anchor from RAW files (samples, CSVs, medians).
Expected values here are independent: hand arithmetic (shown) or the value
printed by each experiment's own analyzer / docs/EXPERIMENTS.md record.
Tolerances are rounding of the quoted value only.

Run from repo root: python3 -m unittest scripts/checks/test_anchors.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchors as an  # noqa: E402
import floorcheck as fc  # noqa: E402

A = an.compute_all()


def rng(x):
    return (round(min(x), 4), round(max(x), 4))


class Problem(unittest.TestCase):
    def test_bytes_and_intensity(self):
        p = A["problem"]
        # 4096x11008 M1 W4: 22,544,384 + 704,512 + 8,192 + 22,016 = 23,279,104 B
        self.assertEqual(p["w4_bytes_m1"], 23_279_104)
        # FP16: 90,177,536 + 8,192 + 22,016 = 90,207,744 B
        self.assertEqual(p["fp16_bytes_m1"], 90_207_744)
        # FLOP 2*4096*11008 = 90,177,536 -> 3.874 and 0.9997 FLOP/B
        self.assertAlmostEqual(p["w4_flop_per_byte_m1"], 3.874, places=3)
        self.assertAlmostEqual(p["fp16_flop_per_byte_m1"], 0.9997, places=3)
        # ridge 51.84e12 / 427.2e9 = 121.35
        self.assertAlmostEqual(p["ridge_measured"], 121.35, places=2)


class ModelMotivation(unittest.TestCase):
    def test_marlin_vs_fp16_curve(self):
        # E052 same-job GQA vs GRA decode, % less time, primary cells
        # (audit-verified 45.3/42.3/38.8/32.4/24.9 at 128/2048/4096/8192/16384)
        c = A["model"]["gqa_less_time_than_gra_pct"]
        for ctx, exp in [(128, 45.3), (2048, 42.3), (4096, 38.8), (8192, 32.4), (16384, 24.9)]:
            lo, hi = rng(c[ctx])
            self.assertTrue(abs(lo - exp) < 0.6 and abs(hi - exp) < 0.6, (ctx, lo, hi))

    def test_ours_vs_fp16_and_marlin(self):
        m = A["model"]
        # E052 analyzer comparisons: GTA below GRA at 128: 47.171..47.288 %
        lo, hi = rng(m["gta_less_time_than_gra_pct"][128])
        self.assertAlmostEqual(lo, 47.17, places=1)
        self.assertAlmostEqual(hi, 47.29, places=1)
        # E052 record: ours vs Marlin decode 0.8-3.6 % less time (all contexts)
        allv = [v for ctx in m["gta_less_time_than_gqa_pct"].values() for v in ctx]
        self.assertAlmostEqual(min(allv), 0.8, delta=0.15)
        self.assertAlmostEqual(max(allv), 3.6, delta=0.15)
        # prefill 7-12 % MORE time at 128..4096 (512 from ext jobs: 11-12 %)
        pre = [v for ctx in (128, 512, 2048, 4096) for v in m["gta_more_prefill_than_gqa_pct"][ctx]]
        self.assertGreater(min(pre), 6.5)
        self.assertLess(max(pre), 12.5)


class CPU(unittest.TestCase):
    def test_e041(self):
        c = A["cpu"]
        # E041 record: T1 26.6 / 31.3 / 241.2 ms, Sp16 13.1 / 12.3 / 12.3
        for M, t1, sp in [(1, 26.6, 13.1), (4, 31.3, 12.3), (128, 241.2, 12.3)]:
            self.assertAlmostEqual(c["t1_ms"][M], t1, delta=0.06)
            self.assertAlmostEqual(c["sp16"][M], sp, delta=0.06)

    def test_e057(self):
        c = A["cpu"]
        self.assertAlmostEqual(c["triad_gbps"][1], 45.2, delta=0.05)
        self.assertAlmostEqual(c["triad_gbps"][16], 126.1, delta=0.05)
        self.assertAlmostEqual(c["clock_ghz_p1"], 4.14, delta=0.01)
        # E057 record: 3.25-3.65 GHz at p = 16 over M1/4/128
        self.assertAlmostEqual(min(c["clock_ghz_p16"].values()), 3.25, delta=0.01)
        self.assertAlmostEqual(max(c["clock_ghz_p16"].values()), 3.65, delta=0.01)


class Climb(unittest.TestCase):
    def test_single_matrix_flushed_4096(self):
        c = A["climb"]
        # E051 a-main-s1.csv flush rows (summary record): M1 naive 205.152, tc_ladder 32.4
        self.assertAlmostEqual(c[1]["naive"]["us"][0], 205.152, places=2)   # series 1
        self.assertAlmostEqual(c[1]["tc_ladder"]["us"][0], 32.4, delta=0.1)
        # every climb time must pass the floor check (flushed)
        for M, d in c.items():
            for k, v in d.items():
                self.assertEqual(v["status"], fc.OK_FLUSHED, (M, k, v))


class WhereItEnds(unittest.TestCase):
    def test_e050_rectangles(self):
        e = A["e050"]
        # E050 analysis / audit: rectangles M4/16 fraction 0.862-0.96, M64/128 0.83-0.87
        lo, hi = rng(e["frac_small_m"])
        self.assertAlmostEqual(lo, 0.862, delta=0.002)
        self.assertAlmostEqual(hi, 0.96, delta=0.005)
        lo, hi = rng(e["frac_large_m"])
        self.assertAlmostEqual(lo, 0.83, delta=0.005)
        self.assertAlmostEqual(hi, 0.87, delta=0.005)
        # E050 record: tc_ladder/tc 0.826-0.849 (4096x11008), 0.864-0.872 (11008x4096)
        lo, hi = rng(e["ladder_over_tc_large_m"])
        self.assertAlmostEqual(lo, 0.826, delta=0.001)
        self.assertAlmostEqual(hi, 0.872, delta=0.001)

    def test_e050_only_valid_rows_used(self):
        self.assertTrue(all(s == fc.OK for s in A["e050"]["statuses_used"]))
        self.assertEqual(len(A["e050"]["statuses_used"]), 64)  # 2 rect x 4 M x 4 cfg x 2 series


class Roofline(unittest.TestCase):
    def test_roofs(self):
        r = A["roofline"]
        self.assertAlmostEqual(r["read_gbps"], 427.2, delta=0.1)   # E057 427.249
        self.assertAlmostEqual(r["copy_gbps"], 385.9, delta=0.1)   # E057 385.914
        self.assertAlmostEqual(r["mma_tflops"], 51.84, delta=0.01)

    def test_coordinates(self):
        r = A["roofline"]["points"]
        # 4096x11008 series 1. M64: 2*64*4096*11008 / 152.5 us = 37.8 TFLOP/s = 73 % of 51.84
        self.assertAlmostEqual(r[("tc_ladder", 64)]["tflops"], 37.8, delta=0.1)
        self.assertAlmostEqual(r[("tc_ladder", 64)]["pct_peak"], 73, delta=0.6)
        self.assertAlmostEqual(r[("marlin", 64)]["pct_peak"], 85, delta=0.6)
        self.assertAlmostEqual(r[("tc_ladder", 128)]["pct_peak"], 78, delta=0.6)
        self.assertAlmostEqual(r[("marlin", 128)]["pct_peak"], 89, delta=0.6)
        # M4: 23,369,728 B / 64.4 us = 362.9 GB/s = 85 % of 427.2
        self.assertAlmostEqual(r[("tc_ladder", 4)]["gbps"], 363, delta=1)
        self.assertAlmostEqual(r[("tc_ladder", 4)]["pct_bw"], 85, delta=0.6)
        self.assertAlmostEqual(r[("marlin", 4)]["pct_bw"], 89, delta=0.6)


class Robustness(unittest.TestCase):
    def test_e054(self):
        r = A["e054"]
        # AGENTS E054: 5060 Ti 3.2-3.9 %, A100 3.0-4.4 %, 3060 Ti -0.1-1.2 %
        lo, hi = rng(r["decode_less_time_pct"]["NVIDIA GeForce RTX 5060 Ti"])
        self.assertAlmostEqual(lo, 3.2, delta=0.06)
        self.assertAlmostEqual(hi, 3.9, delta=0.06)
        lo, hi = rng(r["decode_less_time_pct"]["NVIDIA GeForce RTX 3060 Ti"])
        self.assertAlmostEqual(lo, -0.1, delta=0.06)
        # summary.json: 5 GPU models on 10 nodes (records said "6 GPU types": they counted
        # RTX 4060 Ti 8 GB (c20) and 16 GB (c55) separately; same model name in nvidia-smi)
        self.assertEqual(r["n_gpu_models"], 5)
        self.assertEqual(r["n_nodes"], 10)
        self.assertEqual(r["direction_holds"], (28, 30))


class Quality(unittest.TestCase):
    def test_e055(self):
        q = A["quality"]
        self.assertAlmostEqual(q["ppl"]["R"], 9.7350, places=3)
        self.assertAlmostEqual(q["ppl"]["O"], 8.6058, places=3)
        self.assertAlmostEqual(q["blocks_cost_pct"], 13.12, delta=0.01)   # 9.73499/8.60581 - 1
        self.assertAlmostEqual(q["qt_vs_qm_pct"], -0.0006, delta=0.0002)
        self.assertAlmostEqual(q["top1_agreement"], 0.99795, delta=0.0001)


class Standing(unittest.TestCase):
    def test_vllm(self):
        s = A["standing"]
        # E052 ext comparisons: GTA vs V decode more time 31.9-35.3 (128/512), 45.7-46.2 (2048)
        lo, hi = rng(s["more_time_than_vllm_pct"][128] + s["more_time_than_vllm_pct"][512])
        self.assertAlmostEqual(lo, 31.9, delta=0.1)
        self.assertAlmostEqual(hi, 35.3, delta=0.1)
        lo, hi = rng(s["more_time_than_vllm_pct"][2048])
        self.assertAlmostEqual(lo, 45.7, delta=0.1)
        self.assertAlmostEqual(hi, 46.2, delta=0.1)

    def test_e056(self):
        s = A["standing"]["e056"]
        self.assertAlmostEqual(s[128]["GTA_us"], 5007.4, delta=0.5)
        self.assertAlmostEqual(s[2048]["V_us"], 3899, delta=1)
        # GEMM share of gap 17 % (128) / 11 % (2048); other 93 % / 73 %
        self.assertAlmostEqual(s[128]["gemm_share_pct"], 17, delta=1)
        self.assertAlmostEqual(s[2048]["gemm_share_pct"], 11, delta=1)
        self.assertAlmostEqual(s[128]["other_share_pct"], 93, delta=1)
        self.assertAlmostEqual(s[2048]["other_share_pct"], 73, delta=1)


class ReviewGaps(unittest.TestCase):
    """Added after the independent review (2026-11-20), before fixing anchors.py."""

    def test_cli_runs(self):
        import subprocess
        out = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "anchors.py")],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertIn("tc_ladder", out.stdout)

    def test_e054_prefill_2048(self):
        p = A["e054"]["prefill_2048_more_time_pct"]
        # review, by hand from summary.json (negated prefill_pct_ours_below_marlin at context 2048):
        # A6000 87.5-95.5, A100 46.8-47.4, 5060 Ti 8.7-9.3
        lo, hi = rng(p["NVIDIA RTX A6000"]); self.assertAlmostEqual(lo, 87.5, delta=0.1); self.assertAlmostEqual(hi, 95.5, delta=0.1)
        lo, hi = rng(p["NVIDIA A100 80GB PCIe"]); self.assertAlmostEqual(lo, 46.8, delta=0.1); self.assertAlmostEqual(hi, 47.4, delta=0.1)
        lo, hi = rng(p["NVIDIA GeForce RTX 5060 Ti"]); self.assertAlmostEqual(lo, 8.7, delta=0.1); self.assertAlmostEqual(hi, 9.3, delta=0.1)

    def test_e054_decode_extremes(self):
        d = A["e054"]["decode_less_time_pct"]
        # review: 5060 Ti min 3.149 (c9, 2048), max 3.93; 3060 Ti max 1.2 (AGENTS -0.1-1.2)
        lo, hi = rng(d["NVIDIA GeForce RTX 5060 Ti"])
        self.assertAlmostEqual(lo, 3.149, delta=0.002); self.assertAlmostEqual(hi, 3.93, delta=0.01)
        self.assertAlmostEqual(max(d["NVIDIA GeForce RTX 3060 Ti"]), 1.2, delta=0.05)

    def test_e056_extra_launches_and_attention(self):
        s = A["standing"]["e056"]
        # E056 arms at 128: other calls GTA 1124.0 - V 192.11 = 931.89 (not total 1461 - 389.11 = 1071.9)
        self.assertAlmostEqual(s[128]["other_extra_launches"], 931.89, places=2)
        self.assertAlmostEqual(s[128]["total_extra_launches"], 1071.89, places=2)
        # attention share at 128: (223.48 - 342.93) / (5007.4 - 3774.2) = -9.69 %
        self.assertAlmostEqual(s[128]["attention_share_pct"], -9.69, delta=0.02)

    def test_climb_both_series_m128(self):
        c = A["climb"]
        # awk on a-main-s1.csv / a-main-s2.csv, flush rows, 4096^2
        self.assertEqual(c[1]["tc_ladder"]["us"], [32.416, 33.120])
        self.assertEqual(c[128]["tc_ladder"]["us"], [126.656, 129.344])
        self.assertEqual(c[128]["cublas_dq"]["us"], [114.720, 114.688])
        self.assertEqual(c[128]["naive"]["us"], [2553.504, 2563.392])

    def test_blocks_cost_definition(self):
        # R vs O perplexities from E055 analysis.json: 9.734986752455836 / 8.605813234340317 - 1
        self.assertAlmostEqual(A["quality"]["blocks_cost_pct"], 13.12110, places=4)

    def test_processor_ratios_same_node(self):
        r = A["processor_ratios"]
        # E051 (c) analyzer, c8 CPU T1 over flushed tc_ladder: 794.945 / 949.056 / 1866.487
        for M, exp in [(1, 794.945), (4, 949.056), (128, 1866.487)]:
            self.assertAlmostEqual(r[M], exp, delta=0.01)

    def test_model_provenance(self):
        jobs = A["model"]["jobs"]
        self.assertEqual(jobs[512], ["ext-c10-343366", "ext-c5-343367"])
        self.assertEqual(jobs[16384], ["long-c10-343364"])
        self.assertEqual(sorted(jobs[128]), ["core-c10-343363", "ext-c10-343366", "ext-c5-343367", "head-c10-343365"])

    def test_model_prefill_512(self):
        # E052 ext analyzers (review): GTA vs GQA prefill at 512: 10.95-11.97 % more time
        lo, hi = rng(A["model"]["gta_more_prefill_than_gqa_pct"][512])
        self.assertAlmostEqual(lo, 10.95, delta=0.05); self.assertAlmostEqual(hi, 11.97, delta=0.05)

    def test_roofline_off_roof_coordinates_hidden(self):
        r = A["roofline"]["points"]
        self.assertIsNone(r[("tc_ladder", 64)]["pct_bw"])     # compute-bound: GB/s % is not efficiency
        self.assertIsNone(r[("tc_ladder", 4)]["pct_peak"])    # memory-bound: TFLOP/s % is not efficiency
        self.assertIsNotNone(r[("tc_ladder", 16)]["pct_bw"])  # intensity ~62 < ridge 121


if __name__ == "__main__":
    unittest.main()
