"""E059 unit tests, written BEFORE scripts/e059/e059lib.py was implemented (2026-11-20, coordinator's test-driven rule).

Every expected number is derived by hand in the comment next to it, never by calling the code under test.
Known-bad cases MUST be flagged; known-good cases must pass. Pure Python (no numpy/torch), runs anywhere:
    python3 -m unittest scripts/e059/test_e059.py -v        (from the repo root)
Mutation check: python3 scripts/e059/mutate.py (each deliberate mutation must make this suite fail).
Reuses the shared checker scripts/checks/floorcheck.py (bytes_model / floor_us / classify, BW 427.2e9, L2 32 MiB).
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'checks'))

import floorcheck as fc  # noqa: E402
import e059lib as L      # noqa: E402


class SharedCheckerOnE059Shapes(unittest.TestCase):
    """floorcheck's byte model on the 1536-wide shapes (K/128 must be integral: 1536/128 = 12, 8960/128 = 70)."""

    def test_groups_integral(self):
        for k in (1536, 8960, 4096, 11008):
            self.assertEqual(k % 128, 0)          # 12, 70, 32, 86 groups
        with self.assertRaises(AssertionError):   # a K that is not a multiple of 128 is refused, not floored
            L.matrix_count(1600, 1536)

    def test_bytes_8960x1536_m1(self):
        # codes 8960*1536/2 = 6,881,280; scales 2*70*1536 = 215,040; A 2*8960 = 17,920; C 2*1536 = 3,072
        self.assertEqual(L.bytes_for('v3', 8960, 1536, 1), 7_117_312)

    def test_bytes_1536x8960_m1(self):
        # codes 6,881,280; scales 2*12*8960 = 215,040; A 2*1536 = 3,072; C 2*8960 = 17,920 -> same total
        self.assertEqual(L.bytes_for('v3', 1536, 8960, 1), 7_117_312)

    def test_bytes_1536sq_m1(self):
        # codes 1,179,648; scales 2*12*1536 = 36,864; A 3,072; C 3,072 -> 1,222,656
        self.assertEqual(L.bytes_for('v3', 1536, 1536, 1), 1_222_656)

    def test_floors(self):
        # 7,117,312 / 427.2e9 = 16.6603 us ; 1,222,656 / 427.2e9 = 2.8620 us
        self.assertAlmostEqual(L.floor_for('v3', 8960, 1536, 1), 16.6603, places=3)
        self.assertAlmostEqual(L.floor_for('v3', 1536, 1536, 1), 2.8620, places=3)
        # 11008x4096 M16: 23,248,896 + 2*16*11008 (352,256) + 2*16*4096 (131,072) = 23,732,224 -> 55.553 us;
        # compute floor 2*16*4096*11008 / 51.84e12 = 27.83 us is smaller
        self.assertAlmostEqual(L.floor_for('v3', 11008, 4096, 16), 55.553, places=2)
        # cuBLAS FP16 8960x1536 M16: 27,525,120 + 286,720 + 49,152 = 27,860,992 -> 65.218 us
        self.assertAlmostEqual(L.floor_for('cublas_dq', 8960, 1536, 16), 65.218, places=2)


class CycleSizing(unittest.TestCase):
    """c = smallest count >= 4 whose 4-bit working set is >= 2 x 33,554,432 = 67,108,864 B."""

    def test_1536sq(self):
        # 1,216,512 B/matrix: 55 -> 66,908,160 (< 2 L2), 56 -> 68,124,672 (>=)
        self.assertEqual(L.matrix_count(1536, 1536), 56)

    def test_tall_and_wide_model(self):
        # 7,096,320 B/matrix: 9 -> 63,866,880 (<), 10 -> 70,963,200 (>=)
        self.assertEqual(L.matrix_count(8960, 1536), 10)
        self.assertEqual(L.matrix_count(1536, 8960), 10)

    def test_rectangles_minimum_four(self):
        # 23,248,896 B/matrix: 3 -> 69,746,688 would suffice; the E050 minimum of 4 applies -> 4 (92,995,584 B)
        self.assertEqual(L.matrix_count(4096, 11008), 4)
        self.assertEqual(L.matrix_count(11008, 4096), 4)

    def test_square_4096(self):
        # 8,650,752 B/matrix: 7 -> 60,555,264 (<), 8 -> 69,206,016 (>=); E058's "8 = about 2.1 x L2"
        self.assertEqual(L.matrix_count(4096, 4096), 8)

    def test_predeclared_counts(self):
        # amendment 2 adds 4096^2 with 8 matrices (69,206,016 B >= 2 x L2; E058: 4 matrices are cache-served)
        self.assertEqual(L.COUNTS, {(1536, 8960): 10, (8960, 1536): 10, (1536, 1536): 56, (4096, 11008): 4, (11008, 4096): 4,
                                    (4096, 4096): 8})
        for (k, n), c in L.COUNTS.items():
            self.assertEqual(L.matrix_count(k, n), c)


class Floor(unittest.TestCase):
    # ---- known-bad: must be flagged ----
    def test_single_4096sq_m1_below_floor(self):
        # one 4096^2 matrix at M1 timed 17.7 us < 20.29 us floor (8,667,136 B / 427.2e9)
        self.assertTrue(L.is_below(L.status('tc_ladder', 4096, 4096, 1, 17.7, matrices=1)))
        self.assertFalse(L.valid(L.status('tc_ladder', 4096, 4096, 1, 17.7, matrices=1)))

    def test_readflush_below_floor_flagged(self):
        self.assertTrue(L.is_below(L.status('v3', 4096, 4096, 1, 17.7, flushed=True)))

    def test_small_cycle_flagged(self):
        # four 4096^2 matrices = 34,603,008 B < 67,108,864: above the floor (25 us > 20.39) but cache possible
        self.assertEqual(L.status('marlin', 4096, 4096, 4, 25.0, matrices=4), fc.WS_SMALL)
        # ten 1536^2 matrices = 12,165,120 B: fits L2
        self.assertEqual(L.status('v3', 1536, 1536, 1, 8.0, matrices=10), fc.WS_SMALL)
        self.assertFalse(L.valid(fc.WS_SMALL))

    def test_cublas_uses_fp16_bytes(self):
        # cuBLAS 8960x1536 M1 floor = (27,525,120 + 17,920 + 3,072) / 427.2e9 = 64.48 us: 60 us is impossible
        self.assertTrue(L.is_below(L.status('cublas_dq', 8960, 1536, 1, 60.0, matrices=10)))

    # ---- known-good: must pass ----
    def test_v3_tall_ok(self):
        # 22.5 us > 16.66 us; 10 x 7,096,320 = 70.96 MB >= 2 x L2
        self.assertEqual(L.status('v3', 8960, 1536, 1, 22.5, matrices=10), fc.OK)
        self.assertTrue(L.valid(fc.OK))

    def test_readflush_ok(self):
        self.assertEqual(L.status('marlin', 8960, 1536, 1, 22.5, flushed=True), fc.OK_FLUSHED)
        self.assertTrue(L.valid(fc.OK_FLUSHED))

    def test_cublas_ok(self):
        # 70 us > 64.48 us; 10 FP16 matrices = 275 MB
        self.assertEqual(L.status('cublas_dq', 8960, 1536, 1, 70.0, matrices=10), fc.OK)

    def test_4bit_path_not_fp16_model(self):
        # a 4-bit path at 22.5 us must NOT be judged with the FP16 byte model (floor 64.48 us would flag it)
        for name in ('marlin', 'tc_ladder', 'v3', 'splitk_s4'):
            self.assertEqual(L.status(name, 8960, 1536, 1, 22.5, matrices=10), fc.OK)


class PerGemmAndFraction(unittest.TestCase):
    def test_per_gemm_odd(self):
        # median of [400, 404, 396, 500, 100] = 400; four matrices -> 100.0 us per GEMM
        self.assertEqual(L.per_gemm_us([400, 404, 396, 500, 100], 4), 100.0)

    def test_per_gemm_even(self):
        # median of [560, 570, 580, 590] = (570+580)/2 = 575; 56 matrices -> 10.267857 us
        self.assertAlmostEqual(L.per_gemm_us([560, 570, 580, 590], 56), 10.267857, places=5)

    def test_fraction(self):
        # Marlin 61.3 us, path 64.4 us -> 61.3/64.4 = 0.95186 (fraction of Marlin speed < 1 = slower)
        self.assertAlmostEqual(L.fraction(61.3, 64.4), 0.95186, places=4)
        # path faster than Marlin -> fraction > 1: 30/25 = 1.2
        self.assertAlmostEqual(L.fraction(30.0, 25.0), 1.2)


class Rule(unittest.TestCase):
    """E027/E054 R1 over two series (a = candidate, b = reference)."""

    def test_rule(self):
        self.assertEqual(L.r1([90, 91], [100, 100]), 'faster')      # 0.90, 0.91 <= 0.95
        self.assertEqual(L.r1([110, 111], [100, 100]), 'slower')    # 100 <= 0.95*110 = 104.5
        self.assertEqual(L.r1([100, 103], [100, 100]), 'comparable')
        self.assertEqual(L.r1([94, 100], [100, 100]), 'mixed')      # one series faster, one tied
        self.assertEqual(L.r1([95.0], [100.0]), 'faster')           # boundary: exactly 5% below counts
        self.assertEqual(L.r1([95.1], [100.0]), 'comparable')


class Splits(unittest.TestCase):
    def test_balance_rule(self):
        # 11008x4096: tiles 128, load(1) = ceil(128/36) = 4, cutoff 3.6; S5 ceil(640/36)=18 -> 3.6;
        # S7 ceil(896/36)=25 -> 3.571 (best); S8 ceil(1024/36)=29 -> 3.625  => 7
        self.assertEqual(L.balance(11008, 4096, 1), 7)
        # 8960x1536: tiles 48, load(1) = 2; S3 ceil(144/36)=4 -> 1.333 (best, S6 8/6 ties, not smaller) => 3
        self.assertEqual(L.balance(8960, 1536, 1), 3)
        # 4096x11008: tiles 344, load(1) = 10, cutoff 9; best S7 67/7 = 9.57 > 9 => 1
        self.assertEqual(L.balance(4096, 11008, 1), 1)
        # M 4/16 use one 16-row tile -> same tiles -> same S
        self.assertEqual([L.balance(11008, 4096, m) for m in (4, 16)], [7, 7])

    def test_paths_and_order(self):
        self.assertEqual(L.paths(8960, 1536), ['marlin', 'tc_ladder', 'v3', 'cublas_dq', 'splitk_s1', 'splitk_s2',
                                               'splitk_s3', 'splitk_s4', 'splitk_s8'])
        self.assertEqual(len(L.paths(11008, 4096)), 9)          # S 1,2,4,7,8
        self.assertEqual(len(L.paths(4096, 11008)), 8)          # balance S = 1 already in the set
        self.assertEqual(L.paths(1536, 1536), ['marlin', 'tc_ladder', 'v3', 'cublas_dq'])
        # 8960x1536 is shape index 1, M1 index 0 -> rotate by 1: starts at tc_ladder, marlin last
        o = L.order(8960, 1536, 1, 1)
        self.assertEqual((o[0], o[-1]), ('tc_ladder', 'marlin'))
        self.assertEqual(L.order(8960, 1536, 1, 2), o[::-1])
        # rows per series: (4+9+4+8+9+4) paths x 3 M = 38 x 3 = 114 (amendment 2 adds 4096^2 with the four base paths)
        self.assertEqual(L.paths(4096, 4096), ['marlin', 'tc_ladder', 'v3', 'cublas_dq'])
        self.assertEqual(sum(len(L.paths(k, n)) for k, n in L.SHAPES) * 3, 114)

    def test_split_s(self):
        self.assertEqual(L.split_s('splitk_s7'), 7)
        self.assertEqual(L.split_s('splitk_s1'), 1)
        self.assertIsNone(L.split_s('tc_ladder'))


class FakeLib:
    """Records the S passed to e059_tc_splitk; last_s() reports what the (fake) launcher used."""

    def __init__(self, honour=True):
        self.calls, self.honour, self._last = [], honour, 0

    def e059_tc_splitk(self, a, q, s, c, m, n, k, g, S, stream):
        self.calls.append(S)
        self._last = S if self.honour else 4
        return 0

    def e059_tc_splitk_last_s(self):
        return self._last


class ForcedS(unittest.TestCase):
    PTRS = (1, 2, 3, 4, 1, 1536, 8960, 128)

    def test_each_s_forced(self):
        lib = FakeLib()
        for s in (1, 2, 3, 4, 8):
            L.launch_splitk(lib, f'splitk_s{s}', self.PTRS, 0)
        self.assertEqual(lib.calls, [1, 2, 3, 4, 8])

    def test_launcher_ignoring_s_detected(self):
        # a launcher that always uses its own S (E047's rule = 4) must be caught for S != 4
        with self.assertRaises(AssertionError):
            L.launch_splitk(FakeLib(honour=False), 'splitk_s2', self.PTRS, 0)


class Tolerance(unittest.TestCase):
    """CONTRACT: |got-ref| <= 1e-3|ref| + 1e-3 rms(ref); NaN/Inf fail."""

    REF = [1.0, -2.0, 0.0, 3.0]   # rms = sqrt((1+4+0+9)/4) = sqrt(3.5) = 1.870829 -> 1e-3 rms = 0.0018708

    def test_good(self):
        # bounds: 0.0028708, 0.0038708, 0.0018708, 0.0048708
        self.assertEqual(L.violations([1.002, -2.003, 0.0018, 3.004], self.REF), 0)

    def test_bad(self):
        self.assertEqual(L.violations([1.003, -2.0, 0.0, 3.0], self.REF), 1)    # 0.003 > 0.0028708
        self.assertEqual(L.violations([1.0, -2.0, 0.0019, 3.0], self.REF), 1)   # 0.0019 > 0.0018708
        self.assertEqual(L.violations([float('nan'), -2.0, 0.0, float('inf')], self.REF), 2)


def row(**kw):
    """A passing synthetic timed row (8960x1536, M4, 10 matrices); tests change one field to break it."""
    r = dict(K=8960, N=1536, M=4, config='v3', matrices=10, seeds=list(range(101, 111)), activation_seed=101,
             validation=[dict(num_bad=0, checked=4 * 1536, output_sha256='h%d' % i) for i in range(10)],
             cublas_math_mode=16, output_tail_untouched=True, workspace_guards=True,
             info=dict(fast_path=True), iters=3, samples_us=[300.0, 310.0, 320.0], median_us=310.0,
             median_per_matrix_us=31.0)
    r.update(kw)
    if 'post_timing_validation' not in kw:   # keep post == pre so that only the field under test is broken
        r['post_timing_validation'] = [dict(v) for v in r['validation']]
    return r


class RowCheck(unittest.TestCase):
    def test_good_row(self):
        L.check_row(row(), timed=True)

    def test_bad_rows(self):
        bad = [row(contract_failure='AssertionError(num_bad=3)'),
               row(validation=[dict(num_bad=1, checked=6144, output_sha256='x')] * 10),   # one element out of bound
               row(validation=[dict(num_bad=0, checked=1536, output_sha256='x')] * 10),   # only 1 of 4 rows checked
               row(validation=[dict(num_bad=0, checked=6144, output_sha256='x')] * 9),    # a matrix not checked
               row(info=dict(fast_path=False)),                                            # v3 fell back to v1
               row(median_per_matrix_us=310.0),                                            # not divided by 10
               row(post_timing_validation=[dict(num_bad=0, checked=6144, output_sha256='y')] * 10),  # bits changed
               row(cublas_math_mode=0),
               row(seeds=list(range(102, 112))),                                          # wrong seed list
               row(median_us=305.0, median_per_matrix_us=30.5)]                           # median not of samples
        for r in bad:
            with self.assertRaises(AssertionError, msg=str(r)[:120]):
                L.check_row(r, timed=True)


TIE = {(1536, 8960), (1536, 1536)}   # shapes where the teammate reports v3 and tc "tie" (OURS.md M=1 table)


def synthetic(over=None, valid=None):
    """Per-GEMM medians [series 1, series 2]. Base (C1 and C2 both supported by construction): Marlin 100, v3 100;
    tc_ladder 108 (v3 7.4% below: 'faster'), 102 on the two claimed-tie shapes ('comparable'), 115 on the tall
    shapes; split S_bal 100.5 on the tall shapes, other split arms 110; cuBLAS 150. over/valid: {(K,N,M,path): ...}."""
    cell = {}
    for k, n in L.SHAPES:
        for m in L.MS:
            for x in L.paths(k, n):
                t = {'marlin': 100.0, 'v3': 100.0, 'cublas_dq': 150.0}.get(x, 110.0)
                if x == 'tc_ladder':
                    t = 115.0 if (k, n) in L.TALL else 102.0 if (k, n) in TIE else 108.0
                if (k, n) in L.TALL and x == 'splitk_s%d' % L.BALANCE[(k, n)]:
                    t = 100.5
                cell[(k, n, m, x)] = dict(cyc=[t, t], rf=[t, t], cyc_valid=True, rf_valid=True)
    for key, ts in (over or {}).items():
        cell[key]['cyc'] = list(ts)
    for key in (valid or ()):
        cell[key]['cyc_valid'] = False
    return cell


class Readings(unittest.TestCase):
    def test_c1_supported_as_quoted(self):
        r = L.readings(synthetic())
        self.assertEqual((r['C1a_verdict'], r['C1b_verdict'], r['C1_verdict']), ('supported', 'supported', 'supported'))
        self.assertEqual(r['C1a']['1536x1536']['verdict'], 'descriptive (not claimed)')
        self.assertEqual(sorted(L.CLAIM_V3_VS_TC.values()).count('faster'), 4)   # "faster on four shapes"

    def test_1536sq_marlin_not_scored(self):
        # v3 130 vs Marlin 100 on 1536^2: Marlin 23% below -> 'slower', but the tie was not claimed there
        r = L.readings(synthetic({(1536, 1536, 1, 'v3'): (130, 130), (1536, 1536, 1, 'tc_ladder'): (130, 130)}))
        self.assertEqual(r['C1a_verdict'], 'supported')
        self.assertEqual(r['C1a']['1536x1536']['v3_vs_marlin'], 'slower')

    def test_c1b_series_asymmetric(self):
        # 4096x11008: v3 100 / 106 vs tc 108 / 108: s1 100 <= 102.6 (faster), s2 106 > 102.6 and 108 > 100.7 -> mixed
        r = L.readings(synthetic({(4096, 11008, 1, 'v3'): (100, 106), (4096, 11008, 1, 'marlin'): (100, 106)}))
        self.assertEqual(r['C1b']['4096x11008']['v3_vs_tc_ladder'], 'mixed')
        self.assertEqual(r['C1b']['4096x11008']['verdict'], 'fails')
        self.assertTrue(r['C1b_verdict'].startswith('partly'))

    def test_c1b_v3_slower_than_tc(self):
        # 8960x1536: v3 120 vs tc 115: tc 4.2% below -> comparable, claimed 'faster' -> fails
        r = L.readings(synthetic({(8960, 1536, 1, 'v3'): (120, 120), (8960, 1536, 1, 'marlin'): (120, 120)}))
        self.assertEqual(r['C1b']['8960x1536']['verdict'], 'fails')
        # v3 130 vs tc 115: tc 11.5% below -> 'slower'
        r = L.readings(synthetic({(8960, 1536, 1, 'v3'): (130, 130), (8960, 1536, 1, 'marlin'): (130, 130)}))
        self.assertEqual(r['C1b']['8960x1536']['v3_vs_tc_ladder'], 'slower')

    def test_c1b_claimed_tie_shape_but_v3_faster(self):
        # 1536x8960 claimed tie; tc 108 there -> v3 'faster' -> the claim (comparable) fails on that shape
        r = L.readings(synthetic({(1536, 8960, 1, 'tc_ladder'): (108, 108)}))
        self.assertEqual(r['C1b']['1536x8960']['verdict'], 'fails')

    def test_c1a_series_asymmetric(self):
        # 11008x4096: v3 100 / 110 vs Marlin 100: s2 Marlin 9.1% below -> mixed -> tie fails there
        r = L.readings(synthetic({(11008, 4096, 1, 'v3'): (100, 110)}))
        self.assertEqual(r['C1a']['11008x4096']['verdict'], 'fails')
        self.assertTrue(r['C1a_verdict'].startswith('partly'))

    def test_c1_combined_needs_both(self):
        # every v3 120 (and tc 130 so C1b keeps its labels on 'faster' shapes): C1a contradicted -> C1 not supported
        over = {(k, n, 1, 'v3'): (120, 120) for k, n in L.SHAPES}
        r = L.readings(synthetic(over))
        self.assertEqual(r['C1a_verdict'], 'contradicted')
        self.assertNotEqual(r['C1_verdict'], 'supported')
        # Marlin tie holds everywhere but tc = v3 (100) on the four claimed-'faster' shapes: C1b fails there
        over = {(k, n, 1, 'tc_ladder'): (100, 100) for (k, n), lab in L.CLAIM_V3_VS_TC.items() if lab == 'faster'}
        r = L.readings(synthetic(over))
        self.assertEqual((r['C1a_verdict'], r['C1b_verdict']), ('supported', 'partly supported (shape-dependent)'))
        self.assertTrue(r['C1_verdict'].startswith('partly'))

    def test_c1_open_on_invalid_tc(self):
        r = L.readings(synthetic(valid=[(4096, 4096, 1, 'tc_ladder')]))
        self.assertTrue(r['C1b']['4096x4096'].startswith('open'))
        self.assertEqual(r['C1a']['4096x4096']['verdict'], 'holds')         # Marlin part still scored
        self.assertIn('open', r['C1b_verdict'])
        r = L.readings(synthetic(valid=[(4096, 4096, 1, 'marlin')]))
        self.assertTrue(r['C1a']['4096x4096'].startswith('open'))

    def test_c2_tall_both_framings(self):
        r = L.readings(synthetic())
        # tc/v3: tall 1.15, others 1.08/1.02 -> tall largest; tc fraction of Marlin: tall 0.870 lowest
        self.assertTrue(all(v['verdict'] == 'supported' for v in r['C2_tall_tc_vs_v3'].values()))
        self.assertTrue(all(v['verdict'] == 'supported' for v in r['C2_tall_fraction'].values()))
        # series-asymmetric: 4096x11008 tc 108 / 125 -> in series 2 its tc/v3 1.25 is largest -> not supported
        r = L.readings(synthetic({(4096, 11008, 1, 'tc_ladder'): (108, 125)}))
        self.assertEqual(r['C2_tall_tc_vs_v3'][1]['verdict'], 'not supported')
        self.assertEqual(r['C2_tall_fraction'][1]['verdict'], 'not supported')

    def test_c2_split(self):
        r = L.readings(synthetic())
        # S_bal 100.5 vs tc 115: 12.6% below -> faster; fraction 0.995 >= control 100/108 = 0.926 -> fixes
        self.assertTrue(all(v['verdict'] == 'fixes' for v in r['C2_split'].values()))
        self.assertEqual(r['C2_split']['8960x1536 M1']['balance_vs_v3'], 'comparable')
        r = L.readings(synthetic({(8960, 1536, m, 'splitk_s3'): (112, 112) for m in L.MS}))   # 2.6% below tc
        self.assertEqual(r['C2_split']['8960x1536 M1']['verdict'], 'does not help')

    def test_c2_split_closes_needs_all_series(self):
        # 8960x1536 S3 100.5 / 112 vs tc 115: s1 12.6%, s2 2.6% below -> mixed -> does not help
        r = L.readings(synthetic({(8960, 1536, 1, 'splitk_s3'): (100.5, 112)}))
        self.assertEqual(r['C2_split']['8960x1536 M1']['verdict'], 'does not help')
        # tc 130/130, S3 100 / 120: faster in both (0.77, 0.92); fraction 1.0 / 0.833 vs control 0.926: closes only s1
        r = L.readings(synthetic({(8960, 1536, 1, 'tc_ladder'): (130, 130), (8960, 1536, 1, 'splitk_s3'): (100, 120)}))
        self.assertEqual(r['C2_split']['8960x1536 M1']['verdict'], 'helps, does not close')

    def test_c2_open_on_invalid_control(self):
        r = L.readings(synthetic(valid=[(4096, 11008, 4, 'tc_ladder')]))
        self.assertTrue(r['C2_split']['8960x1536 M4'].startswith('open'))
        r = L.readings(synthetic(valid=[(8960, 1536, 4, 'splitk_s3')]))
        self.assertTrue(r['C2_split']['8960x1536 M4'].startswith('open'))


BODY = 'e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7'
LIBSHA = 'a' * 64


def series_rows(sr):
    """Rows of one series in the predeclared file order (series 2: shapes, M and each cell's order reversed)."""
    shapes = L.SHAPES if sr == 1 else L.SHAPES[::-1]
    ms = L.MS if sr == 1 else L.MS[::-1]
    out = []
    for k, n in shapes:
        c = L.COUNTS[(k, n)]
        for m in ms:
            for pos, x in enumerate(L.order(k, n, m, sr)):
                big = k * n > 20_000_000
                per = 300.0 if x == 'cublas_dq' else 80.0 if big else 40.0   # above every floor (<= 55.6 / 212.2 us)
                vals = [dict(num_bad=0, checked=m * n, output_sha256='%s%d' % ('t' if x in ('tc_ladder', 'splitk_s1') else x, i))
                        for i in range(c)]
                info = dict(fast_path=True, independent_violations=0)
                if L.split_s(x):
                    info.update(S=L.split_s(x), last_s=L.split_s(x))
                out.append(dict(K=k, N=n, M=m, config=x, series=sr, order=pos, matrices=c, seeds=list(range(101, 101 + c)),
                    activation_seed=101, validation=vals, post_timing_validation=[dict(v) for v in vals], cublas_math_mode=16,
                    output_tail_untouched=True, workspace_guards=True, info=info, iters=100, warmup=10,
                    samples_us=[per * c] * 100, median_us=per * c, median_per_matrix_us=per, body_sha256=BODY, lib_sha256=LIBSHA,
                    readflush=dict(iters=200, warmup=10, samples_us=[per + 1] * 200, median_us=per + 1, mean_us=per + 1,
                                   validation=[dict(v) for v in vals[:1]])))
    return out


def run_full(mutate=None):
    """Write synthetic full-job files, apply mutate(rows1, rows2, d), run analyze.full, return summary.json."""
    import contextlib, io, json, tempfile
    from pathlib import Path
    import analyze
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        rs = {1: series_rows(1), 2: series_rows(2)}
        (d / 'prereq-binaries.sha256').write_text('%s  /x/prereq-c8-1/libe059.so\n' % LIBSHA)
        (d / 'gpu-identity.txt').write_text('prereq GPU-a2aa0491-62c7-f289-6ed1-00a6eb3ea767 610.57.04 / now '
                                            'GPU-a2aa0491-62c7-f289-6ed1-00a6eb3ea767 610.57.04\n')
        if mutate:
            mutate(rs[1], rs[2], d)
        for sr in (1, 2):
            (d / ('series%d.jsonl' % sr)).write_text(''.join(json.dumps(r) + '\n' for r in rs[sr]))
        with contextlib.redirect_stdout(io.StringIO()):
            analyze.full(d, d)
        return json.loads((d / 'summary.json').read_text())


def find(rows, k, n, m, x):
    return next(r for r in rows if (r['K'], r['N'], r['M'], r['config']) == (k, n, m, x))


class AnalyzerEndToEnd(unittest.TestCase):
    def test_good(self):
        out = run_full()
        self.assertEqual(out['invalid_cyclic'], [])
        self.assertEqual(out['cells']['8960x1536|1|v3']['cyc'], [40.0, 40.0])        # 400/10, not 400
        self.assertEqual(out['s1_bit_mismatches'], [])

    def test_below_floor_flagged(self):
        def f(r1, r2, d):
            find(r1, 1536, 1536, 1, 'marlin').update(median_per_matrix_us=2.0, median_us=112.0, samples_us=[112.0] * 100)
        out = run_full(f)   # 2.0 us < 2.862 us floor in series 1 only: the cell is invalid (all series must pass)
        self.assertEqual(out['invalid_cyclic'], ['1536x1536 M1 marlin'])
        self.assertTrue(out['readings']['cyclic']['C1a']['1536x1536'].startswith('open'))
        self.assertIn('marlin', out['label_lines']['1536x1536 M1 marlin'])
        self.assertIn('CACHE-AFFECTED', out['label_lines']['1536x1536 M1 marlin'])
        self.assertNotIn('CACHE-AFFECTED', out['label_lines']['1536x1536 M1 v3'])

    def test_4096sq_four_matrix_values_would_be_flagged(self):
        def f(r1, r2, d):   # E050's cache-served Marlin 15.13 us at M4, here in the 8-matrix cycle: below 20.39 us
            for r in (r1, r2):
                find(r, 4096, 4096, 4, 'marlin').update(median_per_matrix_us=15.13, median_us=121.04, samples_us=[121.04] * 100)
        self.assertIn('4096x4096 M4 marlin', run_full(f)['invalid_cyclic'])

    def test_s1_mismatch_recorded_not_fatal(self):
        def f(r1, r2, d):
            find(r1, 8960, 1536, 4, 'splitk_s1')['validation'][0]['output_sha256'] = 'zz'
            find(r1, 8960, 1536, 4, 'splitk_s1')['post_timing_validation'][0]['output_sha256'] = 'zz'
            find(r1, 8960, 1536, 4, 'splitk_s1')['readflush']['validation'][0]['output_sha256'] = 'zz'
        self.assertEqual(run_full(f)['s1_bit_mismatches'], [[1, 8960, 1536, 4]])

    def test_integrity_failures(self):
        def seeds(r1, r2, d): find(r2, 1536, 8960, 16, 'v3')['seeds'][0] = 99
        def file_order(r1, r2, d): r1[0], r1[1] = r1[1], r1[0]
        def cell_order(r1, r2, d): find(r1, 1536, 8960, 1, 'v3')['order'] += 1
        def rf_validation(r1, r2, d): find(r1, 1536, 8960, 1, 'v3')['readflush']['validation'][0]['output_sha256'] = 'q'
        def violations(r1, r2, d): find(r1, 1536, 8960, 1, 'v3')['info']['independent_violations'] = 1
        def rf_iters(r1, r2, d):
            rf = find(r1, 1536, 8960, 1, 'v3')['readflush']; rf['iters'] = 199; rf['samples_us'] = rf['samples_us'][:199]
        def rf_median(r1, r2, d): find(r1, 1536, 8960, 1, 'v3')['readflush']['median_us'] = 30.0
        def median(r1, r2, d): find(r1, 1536, 8960, 1, 'v3')['median_us'] = 399.0
        def lib(r1, r2, d): find(r2, 1536, 8960, 1, 'v3')['lib_sha256'] = 'b' * 64
        def body(r1, r2, d): find(r2, 1536, 8960, 1, 'marlin')['body_sha256'] = 'c' * 64
        def gpu(r1, r2, d): (d / 'gpu-identity.txt').write_text('prereq GPU-a 610.57.04 / now GPU-b 610.57.04\n')
        def missing(r1, r2, d): r2.pop()
        def cell_moved(r1, r2, d):   # a whole cell (4 rows of 1536x8960 M1) moved to the end: only the file order changes
            r1[:] = r1[4:] + r1[:4]
        for f in (cell_moved, seeds, file_order, cell_order, rf_validation, violations, rf_iters, rf_median, median, lib, body, gpu, missing):
            with self.assertRaises(AssertionError, msg=f.__name__):
                run_full(f)


class SassNames(unittest.TestCase):
    """Prerequisite 344190 found the splitk SASS check comparing names that embed the translation unit's file name
    (the E059 launcher includes E047's .cu, so its anonymous namespace is named after splitk_forced.cu)."""

    def test_same_kernel_different_tu(self):
        import analyze
        bench = '_ZN51_GLOBAL__N__84022bb0_18_w4a16_tc_splitk_cu_6bf79c6222w4a16_tc_splitk_kernelILb1EEEvPK6__halfPKjS3_PS1_iiiiPfPj'
        lib = '_ZN49_GLOBAL__N__e5a3fbd6_16_splitk_forced_cu_6bf79c6222w4a16_tc_splitk_kernelILb1EEEvPK6__halfPKjS3_PS1_iiiiPfPj'
        self.assertEqual(analyze.sass_name(bench), analyze.sass_name(lib))

    def test_different_kernels_stay_different(self):
        import analyze
        t0 = '_ZN51_GLOBAL__N__84022bb0_18_w4a16_tc_splitk_cu_6bf79c6222w4a16_tc_splitk_kernelILb0EEEvPK6__halfPKjS3_PS1_iiiiPfPj'
        t1 = '_ZN51_GLOBAL__N__84022bb0_18_w4a16_tc_splitk_cu_6bf79c6222w4a16_tc_splitk_kernelILb1EEEvPK6__halfPKjS3_PS1_iiiiPfPj'
        tc = '_ZN44_GLOBAL__N__a9541ffe_11_w4a16_tc_cu_599bad6515w4a16_tc_kernelILb1ELb1EEEvPK6__halfPKjS3_PS1_iiii'
        m32 = '_ZN48_GLOBAL__N__80b228c8_15_w4a16_tc_m32_cu_935f40ed19w4a16_tc_m32_kernelILb1EEEvPK6__halfPKjS3_PS1_iiii'
        self.assertEqual(len({analyze.sass_name(x) for x in (t0, t1, tc, m32)}), 4)


if __name__ == '__main__':
    unittest.main()
