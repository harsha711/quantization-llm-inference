"""Tests for E060 (scripts/e060/e060lib.py and scripts/e060/analyze.py), written BEFORE the implementation
(2026-11-20, user's test-driven rule). Every expected number is derived by hand in the comment next to it or
read from a raw file another experiment recorded, never by calling the code under test.

E060 times E006's two matched Marlin binaries (control = original body, patched = corrected body with one added
__syncthreads) in E058's cyclic harness. Logic under test: plan (shapes, matrix counts, working set >= 2 x L2,
seeds), arm-to-library mapping and hash check, kernel-address-in-own-library check, arm order and series file
order (series 2 fully reversed), per-GEMM time = median(cycle samples) / matrices, floorcheck validity
(classify == OK with kind 'w4'), cost = t_patched / t_control - 1, per-cell labels (5% rule, both series),
overall label (all valid cells), cost range (all valid cells, both series), and the analyzer's row checks.

Run (repo root):  python3 -m unittest scripts/e060/test_e060.py -v
Mutation runs:    E060_LIB=... E060_ANALYZE=... python3 -m unittest scripts/e060/test_e060.py
"""
import copy
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
LIB = os.environ.get('E060_LIB', os.path.join(HERE, 'e060lib.py'))
ANA = os.environ.get('E060_ANALYZE', os.path.join(HERE, 'analyze.py'))
os.environ['E060_LIB'] = LIB  # analyze.py imports the same (possibly mutated) library
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'checks'))
import floorcheck as fc  # noqa: E402  shared checker; used here only for its status constants


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lib = load('e060lib', LIB)
an = load('e060_analyze', ANA)

L2 = 32 * 1024 * 1024  # 33,554,432 B (RTX 5060 Ti, floorcheck.L2_BYTES)
CONTROL_SHA = '51cfcf3a8c3758631e2824d0be8f0da4c6a4cf29021a769ccf864afb4d1d652a'  # E006 control/libe005.so
PATCHED_SHA = 'e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7'  # E006 patched/libe005.so
E058_S1 = os.path.join(ROOT, 'results/e058_20261120/full-c8-344197/series1.jsonl')


# ---------------------------------------------------------------- plan
class Plan(unittest.TestCase):
    def test_shapes_ms_arms(self):
        self.assertEqual(lib.SHAPES, [(4096, 4096), (4096, 11008), (11008, 4096)])
        self.assertEqual(lib.MS, [1, 4, 16, 64, 128])
        self.assertEqual(lib.ARMS, ['control', 'patched'])

    def test_counts(self):
        # predeclared: 4096^2 x 16, rectangles x 4
        self.assertEqual(lib.COUNTS, {(4096, 4096): 16, (4096, 11008): 4, (11008, 4096): 4})

    def test_working_set_square(self):
        # 4096*4096/2 = 8,388,608 codes + 2*(4096/128)*4096 = 262,144 scales = 8,650,752 per matrix; x16 = 138,412,032
        self.assertEqual(lib.working_set_bytes(4096, 4096, 16), 138_412_032)

    def test_working_set_rect(self):
        # 4096*11008/2 = 22,544,384 + 2*32*11008 = 704,512 -> 23,248,896; x4 = 92,995,584
        self.assertEqual(lib.working_set_bytes(4096, 11008, 4), 92_995_584)
        # 11008*4096/2 = 22,544,384 + 2*86*4096 = 704,512 -> same 23,248,896; x4
        self.assertEqual(lib.working_set_bytes(11008, 4096, 4), 92_995_584)

    def test_plan_passes_at_32mib(self):
        # 2 x L2 = 67,108,864 B: 138,412,032 and 92,995,584 both >= it
        p = lib.plan(L2)
        self.assertEqual([(x['K'], x['N'], x['matrices']) for x in p],
                         [(4096, 4096, 16), (4096, 11008, 4), (11008, 4096, 4)])

    def test_plan_rejects_small_working_set(self):
        # Known bad: a hypothetical 64 MiB L2 -> 2 x L2 = 134,217,728 B; the rectangles' 92,995,584 B is below it.
        with self.assertRaises(AssertionError):
            lib.plan(64 * 1024 * 1024)

    def test_plan_rejects_one_l2_margin(self):
        # Known bad for a 1 x L2 rule: an L2 of 47,000,000 B gives 2 x L2 = 94,000,000 > 92,995,584 -> must raise
        # (a mutant that sizes against 1 x L2 = 47,000,000 would accept it).
        with self.assertRaises(AssertionError):
            lib.plan(47_000_000)

    def test_seeds(self):
        # fresh seeds 201-216 for 4096^2 (16), the first four (201-204) for each rectangle
        self.assertEqual(lib.seeds_for(4096, 4096), list(range(201, 217)))
        self.assertEqual(lib.seeds_for(4096, 11008), [201, 202, 203, 204])
        self.assertEqual(lib.seeds_for(11008, 4096), [201, 202, 203, 204])
        self.assertEqual(lib.ACTIVATION_SEED, 201)
        # never used before: e004_generate seeds 1-10 (E001-E051), 11-26 (E058), 101-156 (E059)
        self.assertGreater(min(lib.SEEDS), 156)


# ---------------------------------------------------------------- arms and libraries
class Libraries(unittest.TestCase):
    def test_expected_hashes(self):
        self.assertEqual(lib.ARM_SHA256['control'], CONTROL_SHA)
        self.assertEqual(lib.ARM_SHA256['patched'], PATCHED_SHA)

    def test_arm_to_path(self):
        root = '/home/hpuvvad/csc548-w4a16-e006-20261106/results/e006_20261106'
        self.assertEqual(lib.library_path('control', root), root + '/control/libe005.so')
        self.assertEqual(lib.library_path('patched', root), root + '/patched/libe005.so')

    def test_check_hash_known_good(self):
        # sha256(b'abc') = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad (FIPS 180-2 example)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b'abc')
        try:
            self.assertEqual(lib.check_hash(f.name, 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'),
                             'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
            with self.assertRaises(AssertionError):  # known bad: wrong expected hash
                lib.check_hash(f.name, PATCHED_SHA)
            for arm in ('control', 'patched'):       # known bad: neither E006 binary
                with self.assertRaises(AssertionError):
                    lib.check_library(arm, f.name)
        finally:
            os.unlink(f.name)

    def test_check_library_rejects_swapped_arm(self):
        # A file whose bytes hash to the patched value must fail when loaded as 'control'. Fake it by checking that
        # check_library compares against ARM_SHA256[arm]: write a file and patch the table copy.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b'abc')
        saved = dict(lib.ARM_SHA256)
        try:
            lib.ARM_SHA256['patched'] = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
            lib.check_library('patched', f.name)            # good
            with self.assertRaises(AssertionError):
                lib.check_library('control', f.name)        # bad: the patched bytes under the control label
        finally:
            lib.ARM_SHA256.clear(); lib.ARM_SHA256.update(saved)
            os.unlink(f.name)


MAPS = ("7f1000000000-7f1000100000 r--p 00000000 08:01 11 /x/e006/control/libe005.so\n"
        "7f1000100000-7f1000300000 r-xp 00100000 08:01 11 /x/e006/control/libe005.so\n"
        "7f2000000000-7f2000100000 r--p 00000000 08:01 12 /x/e006/patched/libe005.so\n"
        "7f2000100000-7f2000300000 r-xp 00100000 08:01 12 /x/e006/patched/libe005.so\n"
        "7f3000000000-7f3000100000 r-xp 00000000 08:01 13 /x/libtorch.so\n")
PATHS = {'control': '/x/e006/control/libe005.so', 'patched': '/x/e006/patched/libe005.so'}


class Addresses(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(lib.library_ranges(MAPS, PATHS['control']),
                         [[0x7f1000000000, 0x7f1000100000], [0x7f1000100000, 0x7f1000300000]])

    def test_in_own_library(self):
        # 0x7f1000100500 lies in control's r-xp range [0x7f1000100000, 0x7f1000300000)
        self.assertTrue(lib.address_in(0x7f1000100500, lib.library_ranges(MAPS, PATHS['control'])))
        self.assertFalse(lib.address_in(0x7f1000100500, lib.library_ranges(MAPS, PATHS['patched'])))
        # end is exclusive: 0x7f1000300000 is outside control
        self.assertFalse(lib.address_in(0x7f1000300000, lib.library_ranges(MAPS, PATHS['control'])))

    def test_check_addresses_good(self):
        lib.check_addresses({'control': 0x7f1000100500, 'patched': 0x7f2000100500}, MAPS, PATHS)

    def test_check_addresses_swapped(self):
        # known bad: each arm's kernel lives in the other arm's library (swapped labels)
        with self.assertRaises(AssertionError):
            lib.check_addresses({'control': 0x7f2000100500, 'patched': 0x7f1000100500}, MAPS, PATHS)

    def test_check_addresses_same_library(self):
        # known bad: both arms resolved to one library (e.g. dlopen returned the first handle)
        with self.assertRaises(AssertionError):
            lib.check_addresses({'control': 0x7f2000100400, 'patched': 0x7f2000100500}, MAPS, PATHS)

    def test_check_addresses_one_library_two_labels(self):
        # added after the first mutation run (survivor 'not-in-other check dropped'): both arms' paths name the same
        # library (both 'control'); distinct addresses inside it must still be rejected.
        with self.assertRaises(AssertionError):
            lib.check_addresses({'control': 0x7f1000100400, 'patched': 0x7f1000100500}, MAPS,
                                {'control': PATHS['control'], 'patched': PATHS['control']})

    def test_check_addresses_outside(self):
        # known bad: an address in libtorch, in neither E006 library
        with self.assertRaises(AssertionError):
            lib.check_addresses({'control': 0x7f3000000500, 'patched': 0x7f2000100500}, MAPS, PATHS)


# ---------------------------------------------------------------- order
class Order(unittest.TestCase):
    def test_series1_cells(self):
        # (shape index + M index) even -> [control, patched], odd -> [patched, control]
        self.assertEqual(lib.arm_order(4096, 4096, 1, 1), ['control', 'patched'])     # 0+0
        self.assertEqual(lib.arm_order(4096, 4096, 4, 1), ['patched', 'control'])     # 0+1
        self.assertEqual(lib.arm_order(4096, 11008, 1, 1), ['patched', 'control'])    # 1+0
        self.assertEqual(lib.arm_order(11008, 4096, 128, 1), ['control', 'patched'])  # 2+4

    def test_series2_reversed(self):
        self.assertEqual(lib.arm_order(4096, 4096, 1, 2), ['patched', 'control'])
        self.assertEqual(lib.arm_order(4096, 4096, 4, 2), ['control', 'patched'])

    def test_balance(self):
        # control first in series 1 where (si+mi) even: si=0: mi 0,2,4 (3); si=1: mi 1,3 (2); si=2: 0,2,4 (3) = 8 of 15
        firsts = [lib.arm_order(k, n, m, 1)[0] for (k, n) in lib.SHAPES for m in lib.MS]
        self.assertEqual(firsts.count('control'), 8)
        self.assertEqual(firsts.count('patched'), 7)

    def test_file_order_series1(self):
        o = lib.file_order(1)
        self.assertEqual(len(o), 30)
        self.assertEqual(o[:4], [(4096, 4096, 1, 'control'), (4096, 4096, 1, 'patched'),
                                 (4096, 4096, 4, 'patched'), (4096, 4096, 4, 'control')])
        self.assertEqual(o[-2:], [(11008, 4096, 128, 'control'), (11008, 4096, 128, 'patched')])

    def test_file_order_series2(self):
        # series 2: shapes reversed, M reversed, each cell's arm order reversed
        o = lib.file_order(2)
        self.assertEqual(len(o), 30)
        self.assertEqual(o[:2], [(11008, 4096, 128, 'patched'), (11008, 4096, 128, 'control')])
        self.assertEqual(o[2:4], [(11008, 4096, 64, 'control'), (11008, 4096, 64, 'patched')])  # 2+3 odd -> s1 [p,c]
        self.assertEqual(o[-2:], [(4096, 4096, 1, 'patched'), (4096, 4096, 1, 'control')])
        self.assertEqual(sorted(o), sorted(lib.file_order(1)))


# ---------------------------------------------------------------- per-GEMM time and validity
def prow(samples, matrices):
    return dict(samples_us=samples, matrices=matrices)


class PerGemm(unittest.TestCase):
    def test_divide_16(self):
        # median(390, 400, 410) = 400 us per cycle / 16 matrices = 25.0 us
        self.assertAlmostEqual(lib.per_gemm_us(prow([390.0, 400.0, 410.0], 16)), 25.0)

    def test_divide_4(self):
        # median(240, 244, 250) = 244 / 4 = 61.0 us
        self.assertAlmostEqual(lib.per_gemm_us(prow([240.0, 244.0, 250.0], 4)), 61.0)

    def test_median_not_mean(self):
        # samples 100, 100, 700: median 100 (mean 300) / 4 = 25.0
        self.assertAlmostEqual(lib.per_gemm_us(prow([100.0, 100.0, 700.0], 4)), 25.0)


class Validity(unittest.TestCase):
    def test_kind(self):
        self.assertEqual(lib.KIND, 'w4')
        self.assertEqual(fc.kind_of('marlin'), 'w4')

    def test_e058_real_four_matrix_cell_below_strict(self):
        # Known bad, from a raw file: E058 series 1, Marlin, 4096^2, M4, 4 matrices: per-GEMM 15.104 us (cycle median
        # / 4). Strict floor = 8,650,752 B / 448e9 = 19.31 us -> BELOW_STRICT.
        with open(E058_S1) as fh:
            hit = [json.loads(x) for x in fh]
        hit = [r for r in hit if (r['config'], r['M'], r['matrices']) == ('marlin', 4, 4)]
        self.assertEqual(len(hit), 1)
        t = lib.per_gemm_us(hit[0])
        self.assertAlmostEqual(t, 15.104, places=2)
        self.assertEqual(lib.status(4096, 4096, 4, t, 4), fc.BELOW_STRICT)
        self.assertFalse(lib.valid(4096, 4096, 4, t, 4))

    def test_e058_real_sixteen_matrix_cell_ok(self):
        # Known good, raw: E058 series 1 Marlin 4096^2 M4 16 matrices 24.137 us > 20.403 us floor
        # (8,716,288 B / 427.2e9); working set 138,412,032 B >= 67,108,864 -> OK
        with open(E058_S1) as fh:
            hit = [json.loads(x) for x in fh]
        hit = [r for r in hit if (r['config'], r['M'], r['matrices']) == ('marlin', 4, 16)]
        t = lib.per_gemm_us(hit[0])
        self.assertAlmostEqual(t, 24.137, places=2)
        self.assertEqual(lib.status(4096, 4096, 4, t, 16), fc.OK)
        self.assertTrue(lib.valid(4096, 4096, 4, t, 16))

    def test_below_measured_invalid(self):
        # 4096^2 M4: strict 19.31 < 20.0 < measured floor 20.403 -> BELOW_MEASURED, invalid
        self.assertEqual(lib.status(4096, 4096, 4, 20.0, 16), fc.BELOW_MEASURED)
        self.assertFalse(lib.valid(4096, 4096, 4, 20.0, 16))

    def test_ws_small_invalid(self):
        # 4096^2 M4 at 25.0 us (above 20.403) but 4 matrices = 34,603,008 B < 67,108,864 -> WS_SMALL, invalid
        self.assertEqual(lib.status(4096, 4096, 4, 25.0, 4), fc.WS_SMALL)
        self.assertFalse(lib.valid(4096, 4096, 4, 25.0, 4))

    def test_rect_compute_floor(self):
        # 4096x11008 M128: compute floor 2*128*4096*11008 / 51.84e12 = 11,542,724,608 / 51.84e12 = 222.66 us;
        # strict 11,542,724,608 / 56.95e12 = 202.66 us. 210 us -> BELOW_MEASURED; 250.16 (E050 Marlin s1) -> OK
        self.assertEqual(lib.status(4096, 11008, 128, 210.0, 4), fc.BELOW_MEASURED)
        self.assertEqual(lib.status(4096, 11008, 128, 250.16, 4), fc.OK)

    def test_rect_m4_ok(self):
        # 4096x11008 M4 floor 23,369,728 B / 427.2e9 = 54.70 us; E050 Marlin 61.26 us with 4 matrices -> OK
        self.assertTrue(lib.valid(4096, 11008, 4, 61.26, 4))
        # 11008x4096 M1 floor (23,248,896 + 22,016 + 8,192 = 23,279,104 B) / 427.2e9 = 54.49 us; 54.0 -> invalid
        self.assertFalse(lib.valid(11008, 4096, 1, 54.0, 4))


# ---------------------------------------------------------------- cost and labels
class Cost(unittest.TestCase):
    def test_direction(self):
        # cost = t_patched / t_control - 1: patched 105 vs control 100 -> +0.05 (inverted ratio would give -0.0476)
        self.assertAlmostEqual(lib.cost(105.0, 100.0), 0.05)
        self.assertAlmostEqual(lib.cost(99.0, 100.0), -0.01)

    def test_boundary_exactly_5pct_is_material(self):
        # predeclared: |t_p - t_c| < 0.05 * t_c is required for "no material"; 5.00% exactly is material
        self.assertTrue(lib.material(105.0, 100.0))
        self.assertFalse(lib.material(104.99, 100.0))
        self.assertTrue(lib.material(95.0, 100.0))    # 5% gain, also material
        self.assertFalse(lib.material(95.01, 100.0))


def times(c1, p1, c2, p2):
    return {('control', 1): c1, ('patched', 1): p1, ('control', 2): c2, ('patched', 2): p2}


class CellLabel(unittest.TestCase):
    # 4096x11008 M4, 4 matrices: floor 54.70 us, so every time >= 55 below is OK unless stated.
    def cell(self, t):
        return lib.cell(4096, 11008, 4, 4, t)

    def test_no_material(self):
        # s1 61.3/61.0 - 1 = +0.4918%; s2 61.0/61.2 - 1 = -0.3268%
        c = self.cell(times(61.0, 61.3, 61.2, 61.0))
        self.assertEqual(c['label'], 'no material cost')
        self.assertAlmostEqual(c['cost'][1], 0.3 / 61.0)
        self.assertAlmostEqual(c['cost'][2], -0.2 / 61.2)
        self.assertTrue(c['result'])

    def test_material_cost_both_series(self):
        # s1 63.0/60.0 = +5.00% exactly (material), s2 63.6/60.0 = +6%
        self.assertEqual(self.cell(times(60.0, 63.0, 60.0, 63.6))['label'], 'material cost')

    def test_series_disagree_s1_material(self):
        # s1 +6% (63.6/60), s2 +1% (60.6/60) -> mixed (a one-series reading of s1 would say 'material cost')
        self.assertEqual(self.cell(times(60.0, 63.6, 60.0, 60.6))['label'], 'mixed (series disagree)')

    def test_series_disagree_s2_material(self):
        # s1 +1%, s2 +6% -> mixed (a one-series reading of s1 would say 'no material cost')
        self.assertEqual(self.cell(times(60.0, 60.6, 60.0, 63.6))['label'], 'mixed (series disagree)')

    def test_gain(self):
        # s1 57/60 - 1 = -5.00%, s2 56/60 - 1 = -6.67% -> material gain
        self.assertEqual(self.cell(times(60.0, 57.0, 60.0, 56.0))['label'], 'material gain (patched faster)')

    def test_cost_and_gain_is_mixed(self):
        # s1 +6%, s2 -6% -> not "material cost", not "gain": mixed
        self.assertEqual(self.cell(times(60.0, 63.6, 60.0, 56.4))['label'], 'mixed (series disagree)')

    def test_not_a_result_control_below_floor(self):
        # s2 control 50.0 < strict 51.89 us -> BELOW_STRICT -> the whole cell is not a result
        c = self.cell(times(61.0, 61.3, 50.0, 61.0))
        self.assertFalse(c['result'])
        self.assertTrue(c['label'].startswith('NOT A RESULT'))
        self.assertIn(fc.BELOW_STRICT, c['label'])

    def test_not_a_result_patched_only(self):
        # only the patched arm is below the floor (s1 patched 54.0 < 54.70): cost validity needs BOTH arms
        c = self.cell(times(61.0, 54.0, 61.2, 61.0))
        self.assertFalse(c['result'])
        self.assertFalse(c['valid_series'][1])
        self.assertTrue(c['valid_series'][2])

    def test_not_a_result_control_only(self):
        c = self.cell(times(54.0, 61.0, 61.2, 61.0))
        self.assertFalse(c['result'])

    def test_ws_small_cell(self):
        # 4096^2 with 4 matrices (34.6 MB < 2 x L2): above the floor but WS_SMALL -> not a result
        c = lib.cell(4096, 4096, 4, 4, times(25.0, 25.1, 25.0, 25.1))
        self.assertFalse(c['result'])
        self.assertIn(fc.WS_SMALL, c['label'])


class Overall(unittest.TestCase):
    def cells(self):
        a = lib.cell(4096, 11008, 4, 4, times(61.0, 61.3, 61.2, 61.0))   # +0.4918% / -0.3268%
        b = lib.cell(4096, 11008, 16, 4, times(60.0, 60.6, 60.0, 59.7))  # +1.0% / -0.5%
        return {(4096, 11008, 4): a, (4096, 11008, 16): b}

    def test_no_material(self):
        o = lib.overall(self.cells())
        self.assertEqual(o['label'], 'no material cost')
        self.assertEqual(o['valid_cells'], 2)
        self.assertEqual(o['not_result'], [])

    def test_range_both_series(self):
        # min over a(s1 +0.004918, s2 -0.003268), b(s1 +0.010, s2 -0.005) = -0.005; max = +0.010
        o = lib.overall(self.cells())
        self.assertAlmostEqual(o['cost_min'], -0.005)
        self.assertAlmostEqual(o['cost_max'], 0.010)

    def test_within_noise(self):
        # max |cost| 0.010 < 0.015 (E058 noise) -> within noise
        self.assertTrue(lib.overall(self.cells())['within_noise'])
        cs = self.cells()
        cs[(11008, 4096, 4)] = lib.cell(11008, 4096, 4, 4, times(60.0, 61.2, 60.0, 61.2))  # +2% both
        self.assertFalse(lib.overall(cs)['within_noise'])
        self.assertEqual(lib.overall(cs)['label'], 'no material cost')

    def test_one_material_cell(self):
        cs = self.cells()
        cs[(11008, 4096, 4)] = lib.cell(11008, 4096, 4, 4, times(60.0, 63.6, 60.0, 63.0))  # +6% / +5%
        o = lib.overall(cs)
        self.assertNotEqual(o['label'], 'no material cost')
        self.assertEqual(o['material'], [(11008, 4096, 4)])

    def test_one_series_material_is_listed(self):
        # a 'mixed' cell (+6% s1, +1% s2) breaks the all-cells rule and is listed as material in a series
        cs = self.cells()
        cs[(11008, 4096, 4)] = lib.cell(11008, 4096, 4, 4, times(60.0, 63.6, 60.0, 60.6))
        o = lib.overall(cs)
        self.assertNotEqual(o['label'], 'no material cost')
        self.assertEqual(o['material'], [(11008, 4096, 4)])

    def test_series2_only_material_is_listed(self):
        # added after the first mutation run (survivor 'material list from series 1 only'): s1 +1% (60.6/60),
        # s2 +6% (63.6/60) -> listed as |cost| >= 5% in a series
        cs = self.cells()
        cs[(11008, 4096, 4)] = lib.cell(11008, 4096, 4, 4, times(60.0, 60.6, 60.0, 63.6))
        self.assertEqual(lib.overall(cs)['material'], [(11008, 4096, 4)])

    def test_invalid_cell_excluded_and_counted(self):
        # a not-a-result cell with a +20% cost does not enter the label or the range, but is named
        cs = self.cells()
        cs[(11008, 4096, 1)] = lib.cell(11008, 4096, 1, 4, times(50.0, 60.0, 50.0, 60.0))  # control 50 < 51.89 strict
        o = lib.overall(cs)
        self.assertEqual(o['label'], 'no material cost')
        self.assertEqual(o['not_result'], [(11008, 4096, 1)])
        self.assertAlmostEqual(o['cost_max'], 0.010)

    def test_no_valid_cells(self):
        cs = {(11008, 4096, 1): lib.cell(11008, 4096, 1, 4, times(50.0, 60.0, 50.0, 60.0))}
        self.assertEqual(lib.overall(cs)['label'], 'no result')


# ---------------------------------------------------------------- analyzer over synthetic series files
# Per-GEMM control times above every floor by hand (floors: 4096^2 20.29/20.40/20.86/41.43/82.85,
# rectangles 54.49/54.70/55.55/111.33/222.66 us at M 1/4/16/64/128).
BASE = {(4096, 4096): {1: 24.0, 4: 24.2, 16: 25.3, 64: 55.3, 128: 97.8},
        (4096, 11008): {1: 60.0, 4: 61.0, 16: 64.0, 64: 131.0, 128: 250.0},
        (11008, 4096): {1: 58.0, 4: 59.0, 16: 61.0, 64: 129.0, 128: 245.0}}
ADDR = {'control': 0x7f1000100500, 'patched': 0x7f2000100500}
RANGES = {'control': [[0x7f1000000000, 0x7f1000300000]], 'patched': [[0x7f2000000000, 0x7f2000300000]]}
SHA = {'control': CONTROL_SHA, 'patched': PATCHED_SHA}


def fake_series(s, factor):
    """Rows in the predeclared file order; patched = control x factor[s]."""
    out = []
    for pos_cell in lib_file_cells(s):
        k, n, m, arm, pos = pos_cell
        c = {(4096, 4096): 16}.get((k, n), 4)
        seeds = list(range(201, 201 + c))
        t = BASE[(k, n)][m] * (factor if arm == 'patched' else 1.0)
        samples = [t * c] * 100
        val = [dict(num_bad=0, checked=m * n, output_sha256=f'h{k}{n}{m}{i}') for i in range(c)]
        out.append(dict(K=k, N=n, M=m, matrices=c, seeds=seeds, activation_seed=201, arm=arm, series=s, order=pos,
                        warmup=10, iters=100, samples_us=samples, median_us=statistics.median(samples),
                        validation=val, post_timing_validation=copy.deepcopy(val), library_sha256=SHA[arm],
                        kernel_address=ADDR[arm], library_ranges=RANGES[arm], output_tail_untouched=True,
                        workspace_guards=True, lock_reset=True))
    return out


def lib_file_cells(s):
    """Hand-written predeclared file order (independent of lib.file_order): shapes and M in order for series 1,
    both reversed for series 2; per cell [control, patched] if (si+mi) even else [patched, control]; series 2
    reverses each cell."""
    shapes = [(4096, 4096), (4096, 11008), (11008, 4096)]
    ms = [1, 4, 16, 64, 128]
    out = []
    for k, n in (shapes if s == 1 else shapes[::-1]):
        for m in (ms if s == 1 else ms[::-1]):
            arms = ['control', 'patched'] if (shapes.index((k, n)) + ms.index(m)) % 2 == 0 else ['patched', 'control']
            if s == 2:
                arms = arms[::-1]
            out += [(k, n, m, a, i) for i, a in enumerate(arms)]
    return out


class Analyzer(unittest.TestCase):
    def good(self):
        return {1: fake_series(1, 1.01), 2: fake_series(2, 0.995)}

    def test_good_passes_and_summarizes(self):
        summ = an.summarize(an.check_full(self.good()))
        # every cell: s1 cost +1.0%, s2 -0.5% -> 15 valid cells, no material cost, range [-0.005, +0.010]
        self.assertEqual(summ['overall']['label'], 'no material cost')
        self.assertEqual(summ['overall']['valid_cells'], 15)
        self.assertAlmostEqual(summ['overall']['cost_min'], -0.005)
        self.assertAlmostEqual(summ['overall']['cost_max'], 0.010)
        self.assertEqual(summ['rows'], 60)
        self.assertEqual(summ['samples'], 6000)

    def test_per_gemm_from_samples(self):
        med = an.check_full(self.good())
        # 4096^2 M1 control: cycle median 24.0 * 16 = 384.0 us / 16 = 24.0 us
        self.assertAlmostEqual(med[(4096, 4096, 1, 'control', 1)], 24.0)
        # 4096x11008 M128 patched s1: 250.0 * 1.01 = 252.5 us
        self.assertAlmostEqual(med[(4096, 11008, 128, 'patched', 1)], 252.5)

    def test_bits_recorded(self):
        # fake rows give identical output hashes for both arms in every cell -> 15 equal cells per series
        g = self.good()
        self.assertEqual(an.bit_equal_cells(g[1]), (15, []))
        g[1][0]['validation'][0]['output_sha256'] = 'different'
        g[1][0]['post_timing_validation'][0]['output_sha256'] = 'different'
        n, diff = an.bit_equal_cells(g[1])
        self.assertEqual(n, 14)
        self.assertEqual(diff, [(4096, 4096, 1)])

    def test_material_from_full_files(self):
        g = {1: fake_series(1, 1.06), 2: fake_series(2, 1.06)}  # +6% everywhere
        o = an.summarize(an.check_full(g))['overall']
        self.assertNotEqual(o['label'], 'no material cost')
        self.assertEqual(len(o['material']), 15)

    def bad(self, edit, s=1):
        g = self.good()
        edit(g[s])
        with self.assertRaises(AssertionError):
            an.check_full(g)

    def test_rejects_wrong_order(self):
        self.bad(lambda rs: rs.__setitem__(slice(0, 2), rs[0:2][::-1]))

    def test_rejects_series2_not_reversed(self):
        g = self.good()
        g[2] = [dict(r, series=2) for r in fake_series(1, 0.995)]
        with self.assertRaises(AssertionError):
            an.check_full(g)

    def test_rejects_missing_row(self):
        self.bad(lambda rs: rs.pop())

    def test_rejects_wrong_seeds(self):
        self.bad(lambda rs: rs[0].__setitem__('seeds', list(range(11, 27))))

    def test_rejects_wrong_activation_seed(self):
        self.bad(lambda rs: rs[0].__setitem__('activation_seed', 11))

    def test_rejects_post_timing_change(self):
        self.bad(lambda rs: rs[3]['post_timing_validation'][0].__setitem__('output_sha256', 'x'))

    def test_rejects_failed_check(self):
        self.bad(lambda rs: rs[3]['validation'][0].__setitem__('num_bad', 1))

    def test_rejects_partial_check(self):
        # every output row: checked must equal M*N
        # (fixed after the first mutation run, survivor 'partial row check accepted': the edit must hit both the
        # pre- and post-timing records, else the equality check alone rejects it)
        def edit(rs):
            rs[3]['validation'][0]['checked'] = 1
            rs[3]['post_timing_validation'][0]['checked'] = 1
        self.bad(edit)

    def test_rejects_swapped_library(self):
        # a control row carrying the patched hash (swapped arm labels)
        self.bad(lambda rs: rs[0].__setitem__('library_sha256', PATCHED_SHA))

    def test_rejects_address_outside_own_library(self):
        self.bad(lambda rs: rs[0].__setitem__('kernel_address', ADDR['patched' if rs[0]['arm'] == 'control' else 'control']))

    def test_rejects_address_in_neither_library(self):
        # added after the first mutation run (survivor 'no row address check'): a control kernel address in libtorch
        # (0x7f3000000500) is in neither E006 library; the pair check alone would pass (differs, not in patched)
        self.bad(lambda rs: rs[0].__setitem__('kernel_address', 0x7f3000000500))

    def test_rejects_same_address_both_arms(self):
        def edit(rs):
            for r in rs[:2]:
                r['kernel_address'] = ADDR['control']; r['library_ranges'] = RANGES['control']
        self.bad(edit)

    def test_rejects_wrong_series_field(self):
        self.bad(lambda rs: rs[5].__setitem__('series', 2))

    def test_rejects_recorded_median_mismatch(self):
        self.bad(lambda rs: rs[5].__setitem__('median_us', rs[5]['median_us'] + 1.0))

    def test_rejects_short_samples(self):
        self.bad(lambda rs: rs[5].__setitem__('samples_us', rs[5]['samples_us'][:99]))

    def test_rejects_wrong_warmup(self):
        self.bad(lambda rs: rs[5].__setitem__('warmup', 0))

    def test_invalid_cell_reported_not_a_result(self):
        g = self.good()
        for s in (1, 2):
            for r in g[s]:
                if (r['K'], r['N'], r['M'], r['arm']) == (4096, 11008, 1, 'control'):
                    r['samples_us'] = [50.0 * 4] * 100; r['median_us'] = 200.0  # 50 us < 51.89 strict
        summ = an.summarize(an.check_full(g))
        self.assertEqual(summ['overall']['not_result'], [(4096, 11008, 1)])
        self.assertEqual(summ['overall']['valid_cells'], 14)
        self.assertIn('NOT A RESULT', an.text(summ))


if __name__ == '__main__':
    unittest.main()
