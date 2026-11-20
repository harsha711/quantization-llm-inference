"""E061 pure logic: plan, series order, per-GEMM time, roofline coordinates, floorcheck validity, E050 cross-check,
row checks and summary. Imported by scripts/e061/bench.py (on the GPU node) and scripts/e061/analyze.py.
Tests (written first): scripts/e061/test_e061.py; mutation check: scripts/e061/mutate.py. Predeclaration:
docs/EXPERIMENTS.md E061."""
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checks'))
import floorcheck as fc  # noqa: E402  shared DRAM/compute floor and working-set check (scripts/checks, v2)

K, N = 4096, 11008
MS = [128, 256, 512]
MATRICES = 4
CAPACITY = 520           # 512 + 8 guard rows (output row-tail check is not empty at M512)
SEEDS = [301, 302, 303, 304]   # fresh: e004_generate used 1-10, 11-26, 101-156, 201-216
ACTIVATION_SEED = SEEDS[0]
CONFIG = 'cublas_dq'
KIND = fc.kind_of(CONFIG)      # 'fp16': W_hat + A + C byte model
CROSS_TOL = 0.03
SAMPLES = 100
WARMUP = 10


def working_set_bytes():
    return fc.working_set(K, N, MATRICES, kind=KIND)


def check_plan(l2_bytes):
    assert working_set_bytes() >= fc.L2_MARGIN * l2_bytes, (working_set_bytes(), l2_bytes)


def m_order(series):
    return list(MS) if series == 1 else list(MS[::-1])


def per_gemm_us(row):
    """Per-GEMM time = median of the cycle samples / matrices in the cycle."""
    return statistics.median(row['samples_us']) / row['matrices']


def flop(m):
    return 2 * m * K * N


def intensity(m):
    return flop(m) / fc.bytes_model(K, N, m, kind=KIND)


def tflops(m, t_us):
    return flop(m) / t_us / 1e6


def bound(m):
    return 'BW' if intensity(m) < fc.PEAK_FLOP_PER_S / fc.BW_BYTES_PER_S else 'TC'


def roof_share_pct(m, t_us):
    """Share of the roof the point sits under (as scripts/checks/anchors.py roofline())."""
    if bound(m) == 'BW':
        return 100 * fc.bytes_model(K, N, m, kind=KIND) / (t_us * 1e-6) / fc.BW_BYTES_PER_S
    return 100 * tflops(m, t_us) * 1e12 / fc.PEAK_FLOP_PER_S


def status(m, t_us):
    return fc.classify(K, N, m, t_us, matrices=MATRICES, kind=KIND)


def valid(m, t_us):
    return status(m, t_us) == fc.OK


def cross_check(t_new, t_e050):
    r = t_new / t_e050
    return r, abs(r - 1) <= CROSS_TOL


def check_rows(rows, series):
    assert [r['M'] for r in rows] == m_order(series), [r['M'] for r in rows]
    for r in rows:
        assert (r['K'], r['N'], r['config'], r['series']) == (K, N, CONFIG, series), r['M']
        assert (r['matrices'], r['seeds'], r['activation_seed']) == (MATRICES, SEEDS, ACTIVATION_SEED), r['M']
        assert (r['capacity'], r['cublas_math_mode'], r['warmup'], r['iters']) == (CAPACITY, 16, WARMUP, SAMPLES)
        assert len(r['samples_us']) == SAMPLES, r['M']
        assert len(r['validation']) == MATRICES and len(r['post_timing_validation']) == MATRICES, r['M']


def summarize(rows1, rows2, e050):
    """rows per series (checked), e050 = {series: E050 per-GEMM us at M128}."""
    check_rows(rows1, 1)
    check_rows(rows2, 2)
    cells, cross = {}, {}
    for s, rows in ((1, rows1), (2, rows2)):
        for r in rows:
            m, t = r['M'], per_gemm_us(r)
            cells[f'{m}|s{s}'] = dict(M=m, series=s, us=t, status=status(m, t), tflops=tflops(m, t),
                                      flop_per_byte=intensity(m), bound=bound(m), share_pct=roof_share_pct(m, t),
                                      floor_us=fc.floor_us(K, N, m, kind=KIND),
                                      strict_floor_us=fc.strict_floor_us(K, N, m, kind=KIND))
        ratio, ok = cross_check(cells[f'128|s{s}']['us'], e050[s])
        cross[f's{s}'] = dict(us=cells[f'128|s{s}']['us'], e050_us=e050[s], ratio=ratio, consistent=ok)
    invalid = sorted((k for k, c in cells.items() if c['status'] != fc.OK), key=lambda k: (k[-1], int(k.split('|')[0])))
    return dict(cells=cells, cross_check=cross, invalid=invalid, all_valid=not invalid)
