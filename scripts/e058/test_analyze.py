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


class Amendment3(unittest.TestCase):
    """Review gaps (amendment 3): each case kills a mutation that survived the amendment-2 suite."""

    def test_only_t8_below_floor(self):
        # M4: t8 20.0 < 20.403 measured floor (and > 19.31 strict) -> BELOW_MEASURED; t16 20.5 >= 20.403 -> OK;
        # 20.0/20.5 - 1 = -2.44% (converged). Invalid because of the 8-matrix status alone.
        v = an.cell_valid('tc', 4, 20.0, 20.5)
        self.assertEqual(v['status_8'], an.fc.BELOW_MEASURED)
        self.assertEqual(v['status_16'], an.fc.OK)
        self.assertTrue(v['converged'])
        self.assertFalse(v['valid'])

    def test_only_t16_below_floor(self):
        # mirror: t8 20.5 OK, t16 20.0 BELOW_MEASURED, +2.5% converged -> invalid
        v = an.cell_valid('tc', 4, 20.5, 20.0)
        self.assertEqual(v['status_8'], an.fc.OK)
        self.assertEqual(v['status_16'], an.fc.BELOW_MEASURED)
        self.assertFalse(v['valid'])

    def test_exactly_three_percent_passes(self):
        # t16 25.0, t8 25.75: difference 0.75 = 3.00% of 25.0 exactly -> converged (bound is <= 3%), valid
        # (both >= 20.403). In floating point 25.75/25 - 1 = 0.030000000000000027, so the comparison must not
        # be a raw ratio; the hand rule is |t8 - t16| <= 0.03 * t16.
        v = an.cell_valid('tc', 4, 25.75, 25.0)
        self.assertTrue(v['converged'])
        self.assertTrue(v['valid'])

    def test_t8_classified_with_eight_matrices(self):
        # With a hypothetical 2.1 x L2 margin: 8 matrices = 69,206,016 B = 2.0625 x L2 < 2.1 -> WS_SMALL,
        # 16 matrices = 4.125 x L2 -> OK. So the 8-matrix status must come from count 8, not 16.
        old = an.fc.L2_MARGIN
        try:
            an.fc.L2_MARGIN = 2.1
            v = an.cell_valid('tc', 4, 22.0, 22.0)
            self.assertEqual(v['status_8'], an.fc.WS_SMALL)
            self.assertEqual(v['status_16'], an.fc.OK)
        finally:
            an.fc.L2_MARGIN = old

    def test_cublas_below_measured_is_floor_calibration(self):
        # cuBLAS M4 77.0 us at 8 and 16: strict 74.90 < 77.0 < measured 78.70 -> BELOW_MEASURED; working set
        # 8 x 33,554,432 = 268 MB (8 x L2): invalid, labelled a floor-calibration issue, not caching.
        v = an.cell_valid('cublas_dq', 4, 77.0, 77.0)
        self.assertFalse(v['valid'])
        self.assertEqual(v['note'], an.NOTE_FLOOR_CALIBRATION)
        # a 4-bit BELOW_MEASURED cell carries no such note
        self.assertIsNone(an.cell_valid('tc', 4, 20.0, 20.0)['note'])
        # cuBLAS BELOW_STRICT (70 < 74.90) is not a calibration question either
        self.assertIsNone(an.cell_valid('cublas_dq', 4, 70.0, 70.0)['note'])

    def test_series_order_from_file(self):
        # series 1 file order: counts 4, 8, 16; within each M 4, 16, 64, 128; paths in order(c, m, 1).
        # series 2: counts 16, 8, 4; M 128, 64, 16, 4; paths in order(c, m, 2) (= reversed).
        def rows_for(s):
            cs, ms = ([4, 8, 16], [4, 16, 64, 128]) if s == 1 else ([16, 8, 4], [128, 64, 16, 4])
            return [dict(matrices=c, M=m, config=x) for c in cs for m in ms for x in an.order(c, m, s)]
        an.check_series_order(rows_for(1), 1)
        an.check_series_order(rows_for(2), 2)
        with self.assertRaises(AssertionError):      # series 2 written in series-1 block order
            an.check_series_order([dict(r) for r in rows_for(1)], 2)
        bad = rows_for(2)
        bad[0], bad[4] = bad[4], bad[0]              # first two (count, M) blocks interleaved
        with self.assertRaises(AssertionError):
            an.check_series_order(bad, 2)


class CellsUsesFileOrder(unittest.TestCase):
    """cells() (the full-stage loader) must enforce the file order, not only the per-row order field."""

    @staticmethod
    def series(s, cs, ms):
        out = []
        for c in cs:
            for m in ms:
                for i, x in enumerate(an.order(c, m, s)):
                    v = [dict(num_bad=0, checked=m*4096, output_sha256=f'{j}-{m}-{c}') for j in range(c)]
                    t = 30.0*c  # cycle median; per-GEMM 30 us
                    out.append(dict(K=4096, N=4096, M=m, matrices=c, config=x, series=s, iters=100, warmup=10,
                                    samples_us=[t]*100, median_us=t, order=i, seeds=an.SEEDS[:c], activation_seed=11,
                                    validation=v, post_timing_validation=v, cublas_math_mode=16,
                                    output_tail_untouched=True, workspace_guards=True, tc_sha256=an.E050_TC_SHA))
        return out

    def test_good_order_loads(self):
        med = an.cells({1: self.series(1, [4, 8, 16], [4, 16, 64, 128]),
                        2: self.series(2, [16, 8, 4], [128, 64, 16, 4])})
        self.assertEqual(len(med), 96)                       # 48 rows x 2 series
        self.assertAlmostEqual(med[(16, 4, 'tc', 2)], 30.0)  # 480 / 16

    def test_series2_in_series1_block_order_rejected(self):
        # series-2 rows (reversed path order per cell) written with counts 4,8,16 and M 4..128: every row's own
        # order field is right, only the block order is wrong -> must be rejected
        with self.assertRaises(AssertionError):
            an.cells({1: self.series(1, [4, 8, 16], [4, 16, 64, 128]),
                      2: self.series(2, [4, 8, 16], [4, 16, 64, 128])})


class Review2(unittest.TestCase):
    """Second-review findings: each case kills a surviving mutant."""

    def med(self, **override):
        # Same base as Summarize: M4/M16 Marlin 24, tc 30, tc_ladder 30, cuBLAS 96; M64/M128 100/125/110/120
        # (all above floors, converged). override: {(count, M, path, series): value}
        med = {}
        for c in an.COUNTS:
            for s in (1, 2):
                for m in an.MS:
                    v = dict(marlin=24.0, tc=30.0, tc_ladder=30.0, cublas_dq=96.0) if m < 64 else \
                        dict(marlin=100.0, tc=125.0, tc_ladder=110.0, cublas_dq=120.0)
                    for x in an.NAMES:
                        med[(c, m, x, s)] = v[x]
        med.update(override)
        return med

    def test_ratio_validity_cublas_own_cell(self):
        # only cuBLAS M4 s1 invalid (t16 = 70 < 74.90 strict): cuBLAS fraction invalid, tc/tc_ladder fractions valid
        sm = an.summarize({**self.med(), (16, 4, 'cublas_dq', 1): 70.0})
        ok = {(r['M'], r['series']): r for r in sm['timing']}[(4, 1)]['c16_ratio_valid']
        self.assertFalse(ok['cublas_fraction'])
        self.assertTrue(ok['tc_fraction'])
        self.assertTrue(ok['tc_ladder_fraction'])
        self.assertTrue(ok['tc_ladder_over_tc'])
        self.assertEqual(sm['valid_cells'], 31)

    def test_ratio_validity_tc_ladder_own_cell(self):
        # only tc_ladder M64 s2 invalid (t8 = 125, t16 = 110: 15/110 = 13.6% > 3%, not converged):
        # tc_ladder fraction and ladder/tc invalid; tc and cuBLAS fractions valid
        sm = an.summarize({**self.med(), (8, 64, 'tc_ladder', 2): 125.0})
        ok = {(r['M'], r['series']): r for r in sm['timing']}[(64, 2)]['c16_ratio_valid']
        self.assertFalse(ok['tc_ladder_fraction'])
        self.assertFalse(ok['tc_ladder_over_tc'])
        self.assertTrue(ok['tc_fraction'])
        self.assertTrue(ok['cublas_fraction'])

    def test_text_label_non_converged_cell(self):
        # the tc_ladder M64 s2 line must say CACHE-AFFECTED and "not converged"; a valid line says VALID
        t = an.text(an.summarize({**self.med(), (8, 64, 'tc_ladder', 2): 125.0}))
        bad = [l for l in t.splitlines() if l.startswith('tc_ladder   64   2')]
        good = [l for l in t.splitlines() if l.startswith('tc_ladder   64   1')]
        self.assertEqual(len(bad), 1)
        self.assertIn('CACHE-AFFECTED', bad[0])
        self.assertIn('not converged', bad[0])
        self.assertNotIn('VALID', bad[0].replace('CACHE-AFFECTED', ''))
        self.assertIn('VALID', good[0])
        self.assertNotIn('CACHE-AFFECTED', good[0])

    def test_c16_over_e050_uses_16_matrices(self):
        # E050 series 1 Marlin 4096^2 M4 = 16.71 us (E050 table 16.7; summary.json 16.712, 4-matrix cycle). With Marlin at 24.0 for
        # counts 4/8 and 20.9 at 16 (M4 s1), c16_over_e050 = 20.9/E050 and c4_over_e050 = 24.0/E050.
        sm = an.summarize({**self.med(), (16, 4, 'marlin', 1): 20.9})
        r = {(x['M'], x['series']): x for x in sm['timing']}[(4, 1)]
        e = r['e050']['marlin_us']
        self.assertAlmostEqual(e, 16.71, delta=0.01)
        self.assertAlmostEqual(r['c16_over_e050']['marlin'], 20.9/e)
        self.assertAlmostEqual(r['c4_over_e050']['marlin'], 24.0/e)

    def test_calibration_note_excludes_below_strict(self):
        # cuBLAS t8 77.0 (BELOW_MEASURED) and t16 70.0 (BELOW_STRICT, < 74.90): no calibration note
        v = an.cell_valid('cublas_dq', 4, 77.0, 70.0)
        self.assertEqual(v['status_16'], an.fc.BELOW_STRICT)
        self.assertIsNone(v['note'])


class CellsIntegrity(unittest.TestCase):
    """cells() must reject bad rows (seed list, tc_ladder bits, post-timing validation, series field)."""

    def good(self):
        return {1: CellsUsesFileOrder.series(1, [4, 8, 16], [4, 16, 64, 128]),
                2: CellsUsesFileOrder.series(2, [16, 8, 4], [128, 64, 16, 4])}

    def test_good_passes(self):
        self.assertEqual(len(an.cells(self.good())), 96)

    def test_wrong_seeds_rejected(self):
        d = self.good(); d[1][0]['seeds'] = list(range(7, 11))   # E050's seeds, not 11-14
        with self.assertRaises(AssertionError):
            an.cells(d)

    def test_tc_ladder_bits_differ_rejected(self):
        d = self.good()
        r = next(x for x in d[2] if x['config'] == 'tc_ladder' and x['matrices'] == 8 and x['M'] == 64)
        r['validation'] = [dict(v, output_sha256='different') for v in r['validation']]
        r['post_timing_validation'] = r['validation']
        with self.assertRaises(AssertionError):
            an.cells(d)

    def test_post_timing_validation_differs_rejected(self):
        d = self.good()
        r = d[1][5]
        r['post_timing_validation'] = [dict(v, max_abs=1.0) for v in r['validation']]
        with self.assertRaises(AssertionError):
            an.cells(d)

    def test_series_field_mismatch_rejected(self):
        d = self.good(); d[2][3]['series'] = 1
        with self.assertRaises(AssertionError):
            an.cells(d)


class OutputLocation(unittest.TestCase):
    """Review 3: the analyzer never writes into a retrieved (sealed: all.sha256 present) job directory, and the
    summary stores repo-relative paths only."""

    def make(self, root):
        import pathlib
        d = pathlib.Path(root)/'full-c8-0'
        d.mkdir()
        rows = {1: CellsUsesFileOrder.series(1, [4, 8, 16], [4, 16, 64, 128]),
                2: CellsUsesFileOrder.series(2, [16, 8, 4], [128, 64, 16, 4])}
        for s_, rs in rows.items():
            (d/f'series{s_}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rs))
        (d/'all.sha256').write_text('sealed\n')
        return d

    @staticmethod
    def snapshot(d):
        import hashlib
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}

    def test_sealed_dir_untouched_with_out(self):
        import contextlib, io, pathlib, tempfile
        with tempfile.TemporaryDirectory() as root:
            d = self.make(root)
            before = self.snapshot(d)
            out = pathlib.Path(root)/'analysis'; out.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                an.full(d, out)
            self.assertEqual(self.snapshot(d), before)                   # input directory unchanged
            self.assertTrue((out/'summary.json').exists())

    def test_sealed_dir_refused_without_out(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            d = self.make(root)
            before = self.snapshot(d)
            with self.assertRaises(AssertionError):
                an.full(d, None)                                         # would default into the sealed dir
            with self.assertRaises(AssertionError):
                an.full(d, d)                                            # explicit, still refused
            self.assertEqual(self.snapshot(d), before)

    def test_summary_paths_repo_relative(self):
        # the stored E050 reference is 'results/e050_20261118/summary.json', never an absolute home path
        sm = an.summarize(Review2().med())
        self.assertEqual(sm['e050_summary'], 'results/e050_20261118/summary.json')
        self.assertNotIn('/home/', json.dumps(sm))


class Summarize(unittest.TestCase):
    """Hand-computed fractions and validity marks at 16 matrices."""

    def setUp(self):
        # Every (count, M, series): M4/M16 Marlin 24, tc 30, tc_ladder 30, cuBLAS 96;
        # M64/M128 Marlin 100, tc 125, tc_ladder 110, cuBLAS 120. All above their floors
        # (4-bit 20.40/20.86/41.43/82.85; cuBLAS 78.70/79.16/81.00/83.45) and converged (identical at 8/16).
        # One defect: Marlin M4 series 2 at 16 matrices = 15.0 (< 19.31 strict) -> Marlin cell invalid there.
        self.med = {}
        for c in an.COUNTS:
            for s in (1, 2):
                for m in an.MS:
                    v = dict(marlin=24.0, tc=30.0, tc_ladder=30.0, cublas_dq=96.0) if m < 64 else \
                        dict(marlin=100.0, tc=125.0, tc_ladder=110.0, cublas_dq=120.0)
                    for x in an.NAMES:
                        self.med[(c, m, x, s)] = v[x]
        self.med[(16, 4, 'marlin', 2)] = 15.0
        self.sm = an.summarize(self.med)
        self.row = {(r['M'], r['series']): r for r in self.sm['timing']}

    def test_fractions(self):
        r = self.row[(4, 1)]['c16']
        self.assertAlmostEqual(r['tc_fraction_of_marlin_speed'], 0.8)         # 24/30
        self.assertAlmostEqual(r['cublas_dq_fraction_of_marlin_speed'], 0.25)  # 24/96
        self.assertAlmostEqual(r['tc_ladder_over_tc'], 1.0)                    # 30/30
        r = self.row[(64, 2)]['c16']
        self.assertAlmostEqual(r['tc_fraction_of_marlin_speed'], 0.8)              # 100/125
        self.assertAlmostEqual(r['tc_ladder_fraction_of_marlin_speed'], 0.909091, places=5)  # 100/110
        self.assertAlmostEqual(r['cublas_dq_fraction_of_marlin_speed'], 0.833333, places=5)  # 100/120
        self.assertAlmostEqual(r['tc_ladder_over_tc'], 0.88)                       # 110/125
        r = self.row[(4, 2)]['c16']
        self.assertAlmostEqual(r['tc_fraction_of_marlin_speed'], 0.5)          # 15/30 (defect cell)

    def test_validity_marks(self):
        # 32 cells, only Marlin M4 s2 invalid -> 31 valid
        self.assertEqual(self.sm['valid_cells'], 31)
        ok = self.row[(4, 2)]['c16_ratio_valid']
        self.assertFalse(ok['tc_fraction'])          # needs the Marlin cell
        self.assertFalse(ok['tc_ladder_fraction'])
        self.assertFalse(ok['cublas_fraction'])
        self.assertTrue(ok['tc_ladder_over_tc'])     # tc and tc_ladder cells both valid
        self.assertTrue(all(self.row[(4, 1)]['c16_ratio_valid'].values()))

    def test_t4_is_the_four_matrix_median(self):
        # change only the 4-matrix value; validity must report it as t4 and not mix it with t8
        med = dict(self.med)
        med[(4, 16, 'tc', 1)] = 12.5
        v = an.validity(med)[('tc', 16, 1)]
        self.assertEqual(v['t4'], 12.5)
        self.assertEqual(v['t8'], 30.0)
        self.assertEqual(v['status_4'], an.fc.BELOW_STRICT)  # 12.5 < 19.31

    def test_wording(self):
        # valid cells are labelled converged/working-set-insensitive, never "DRAM-regime"
        t = an.text(self.sm)
        self.assertIn(an.LABEL_VALID, t)
        self.assertEqual(an.LABEL_VALID, 'converged, working-set-insensitive (8 vs 16)')
        self.assertNotIn('DRAM-regime', t)


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
