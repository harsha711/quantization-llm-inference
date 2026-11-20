"""E060 pure logic: plan, arms/libraries, order, per-GEMM time, floorcheck validity, cost and labels.

Imported by scripts/e060/bench.py (on the GPU node) and scripts/e060/analyze.py. Tests (written first):
scripts/e060/test_e060.py; mutation check: scripts/e060/mutate.py. Predeclaration: docs/EXPERIMENTS.md E060.
"""
import hashlib
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checks'))
import floorcheck as fc  # noqa: E402  shared DRAM/compute floor and working-set check (scripts/checks, v2)

SHAPES = [(4096, 4096), (4096, 11008), (11008, 4096)]
COUNTS = {(4096, 4096): 16, (4096, 11008): 4, (11008, 4096): 4}
MS = [1, 4, 16, 64, 128]
ARMS = ['control', 'patched']
SEEDS = list(range(201, 217))      # fresh: e004_generate used 1-10 (E001-E051), 11-26 (E058), 101-156 (E059)
ACTIVATION_SEED = SEEDS[0]
ARM_SHA256 = {'control': '51cfcf3a8c3758631e2824d0be8f0da4c6a4cf29021a769ccf864afb4d1d652a',   # E006 original body
              'patched': 'e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7'}   # E006 corrected body
KIND = fc.kind_of('marlin')        # 'w4': 4-bit weights + FP16 scales + A + C byte model
TOL = 0.05                         # project 5% rule
NOISE = 0.015                      # E058 same-harness run-to-run spread, about 0.3-1.5%
L2_MARGIN = 2.0


def working_set_bytes(k, n, matrices):
    """4-bit codes + FP16 scales rotated per cycle."""
    return matrices * (k * n // 2 + 2 * (k // 128) * n)


def plan(l2_bytes):
    out = []
    for k, n in SHAPES:
        c = COUNTS[(k, n)]
        assert working_set_bytes(k, n, c) >= L2_MARGIN * l2_bytes, (k, n, c, l2_bytes)
        out.append(dict(K=k, N=n, matrices=c, seeds=seeds_for(k, n)))
    return out


def seeds_for(k, n):
    return SEEDS[:COUNTS[(k, n)]]


def library_path(arm, root):
    assert arm in ARMS, arm
    return os.path.join(root, arm, 'libe005.so')


def check_hash(path, expected):
    with open(path, 'rb') as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    assert got == expected, (path, got, expected)
    return got


def check_library(arm, path):
    return check_hash(path, ARM_SHA256[arm])


def library_ranges(maps_text, path):
    """[start, end) address ranges of every /proc/self/maps line whose pathname is `path`."""
    out = []
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[5].strip() == path:
            a, b = parts[0].split('-')
            out.append([int(a, 16), int(b, 16)])
    return out


def address_in(addr, ranges):
    return any(a <= addr < b for a, b in ranges)


def check_addresses(addrs, maps_text, paths):
    """Each arm's kernel (host-stub) address lies in its own library and not in the other's (which also makes the
    two addresses differ)."""
    for arm in ARMS:
        other = ARMS[1 - ARMS.index(arm)]
        assert address_in(addrs[arm], library_ranges(maps_text, paths[arm])), (arm, hex(addrs[arm]))
        assert not address_in(addrs[arm], library_ranges(maps_text, paths[other])), (arm, hex(addrs[arm]))


def arm_order(k, n, m, series):
    first = ARMS if (SHAPES.index((k, n)) + MS.index(m)) % 2 == 0 else ARMS[::-1]
    return list(first) if series == 1 else list(first[::-1])


def file_order(series):
    shapes, ms = (SHAPES, MS) if series == 1 else (SHAPES[::-1], MS[::-1])
    return [(k, n, m, arm) for k, n in shapes for m in ms for arm in arm_order(k, n, m, series)]


def per_gemm_us(row):
    """Per-GEMM time = median of the cycle samples / matrices in the cycle."""
    return statistics.median(row['samples_us']) / row['matrices']


def status(k, n, m, t, matrices):
    return fc.classify(k, n, m, t, matrices=matrices, kind=KIND)


def valid(k, n, m, t, matrices):
    return status(k, n, m, t, matrices) == fc.OK


def cost(t_patched, t_control):
    return t_patched / t_control - 1


def material(t_patched, t_control):
    """|t_p - t_c| >= 5% of t_c (exactly 5.00% is material)."""
    return not abs(t_patched - t_control) < TOL * t_control


def _side(tp, tc):
    if not material(tp, tc):
        return 'none'
    return 'cost' if tp > tc else 'gain'


def cell(k, n, m, matrices, t):
    """t = {(arm, series): per-GEMM us}. A series counts only if both arms are OK; the label needs both series."""
    st = {(a, s): status(k, n, m, t[(a, s)], matrices) for a in ARMS for s in (1, 2)}
    vs = {s: st[('control', s)] == fc.OK and st[('patched', s)] == fc.OK for s in (1, 2)}
    cs = {s: cost(t[('patched', s)], t[('control', s)]) for s in (1, 2)}
    sides = {s: _side(t[('patched', s)], t[('control', s)]) for s in (1, 2)}
    result = vs[1] and vs[2]
    if not result:
        label = 'NOT A RESULT (' + ', '.join(f'{a} s{s} {st[(a, s)]}' for s in (1, 2) for a in ARMS
                                             if st[(a, s)] != fc.OK) + ')'
    elif sides[1] == sides[2] == 'none':
        label = 'no material cost'
    elif sides[1] == sides[2] == 'cost':
        label = 'material cost'
    elif sides[1] == sides[2] == 'gain':
        label = 'material gain (patched faster)'
    else:
        label = 'mixed (series disagree)'
    return dict(K=k, N=n, M=m, matrices=matrices, times={f'{a} s{s}': t[(a, s)] for a in ARMS for s in (1, 2)},
                status={f'{a} s{s}': st[(a, s)] for a in ARMS for s in (1, 2)},
                floor_us=fc.floor_us(k, n, m, kind=KIND), strict_floor_us=fc.strict_floor_us(k, n, m, kind=KIND),
                valid_series=vs, cost=cs, sides=sides, result=result, label=label)


def overall(cells):
    """cells = {(K, N, M): cell(...)}. Label over valid cells only (all cells, both series); range likewise."""
    good = {key: c for key, c in cells.items() if c['result']}
    not_result = sorted(key for key, c in cells.items() if not c['result'])
    material_cells = sorted(key for key, c in good.items() if any(c['sides'][s] != 'none' for s in (1, 2)))
    costs = [c['cost'][s] for c in good.values() for s in (1, 2)]
    if not good:
        label = 'no result'
    elif all(c['label'] == 'no material cost' for c in good.values()):
        label = 'no material cost'
    else:
        label = 'material difference in some cells'
    return dict(label=label, valid_cells=len(good), cells=len(cells), not_result=not_result,
                material=material_cells, cost_min=min(costs) if costs else None,
                cost_max=max(costs) if costs else None,
                within_noise=bool(costs) and max(abs(x) for x in costs) < NOISE)
