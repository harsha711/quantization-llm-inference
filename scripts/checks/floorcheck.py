"""DRAM-floor and working-set check for W4A16 GEMM timings (2026-11-20).

A timing is physically a DRAM-regime measurement only if
  1. time >= floor = max(modelled bytes / BW, 2*M*N*K / PEAK), and
  2. the bytes rotated between reuses (working set) are >= L2_MARGIN x L2,
     unless the run flushed the cache (then only 1 applies; a store flush
     does not prove eviction, so OK_FLUSHED is weaker than OK).
Constants: E057 measured roofs on RTX 5060 Ti (c8); 32 MiB L2.

Usage: python3 scripts/checks/floorcheck.py   (prints every row of E050 and E051 b/c)
Tests: python3 -m unittest scripts/checks/test_floorcheck.py -v
"""
import json
import glob
import os
import re
import statistics

BW_BYTES_PER_S = 427.2e9
PEAK_FLOP_PER_S = 51.84e12
L2_BYTES = 32 * 1024 * 1024
L2_MARGIN = 2.0
GROUP = 128

OK = "OK"
OK_FLUSHED = "OK_FLUSHED"
BELOW_FLOOR = "BELOW_FLOOR"   # impossible from DRAM: cache-served
WS_SMALL = "WS_SMALL"         # above floor, but working set < 2 x L2: cache possible


def matrix_bytes(K, N, fp16_weights=False):
    if fp16_weights:
        return 2 * K * N
    return K * N // 2 + 2 * (K // GROUP) * N


def bytes_model(K, N, M, fp16_weights=False):
    return matrix_bytes(K, N, fp16_weights) + 2 * M * K + 2 * M * N


def floor_us(K, N, M, fp16_weights=False):
    mem = bytes_model(K, N, M, fp16_weights) / BW_BYTES_PER_S
    comp = 2 * M * N * K / PEAK_FLOP_PER_S
    return max(mem, comp) * 1e6


def working_set(K, N, matrices, fp16_weights=False):
    return matrices * matrix_bytes(K, N, fp16_weights)


def classify(K, N, M, time_us, matrices=None, flushed=False, fp16_weights=False):
    if matrices is None and not flushed:
        raise ValueError("need matrices (cycle size) or flushed=True")
    if time_us < floor_us(K, N, M, fp16_weights):
        return BELOW_FLOOR
    if flushed:
        return OK_FLUSHED
    if working_set(K, N, matrices, fp16_weights) < L2_MARGIN * L2_BYTES:
        return WS_SMALL
    return OK


def _row(K, N, M, t, config, **kw):
    fp16 = config.startswith("cublas")
    r = dict(K=K, N=N, M=M, time_us=t, config=config, **kw)
    r["floor_us"] = floor_us(K, N, M, fp16)
    r["status"] = classify(K, N, M, t, matrices=kw.get("matrices"),
                           flushed=kw.get("flushed", False), fp16_weights=fp16)
    return r


def e050_rows(directory):
    """E050 cyclic: per-GEMM time = median of cycle samples / matrices."""
    rows = []
    for f in sorted(glob.glob(os.path.join(directory, "series*.jsonl"))):
        with open(f) as fh:
            lines = fh.readlines()
        for line in lines:
            r = json.loads(line)
            t = statistics.median(r["samples_us"]) / r["matrices"]
            rows.append(_row(r["K"], r["N"], r["M"], t, r["config"],
                             matrices=r["matrices"], series=r["series"]))
    return rows


_B = re.compile(r"^(\d+)x(\d+) x(\d+) (.*)$")


def e051_b_rows(analysis_txt):
    """E051 (b) E009 cycle at M=4: 'KxN xC cfg s1|s2 ...' lines; unfused not modelled."""
    rows, on = [], False
    with open(analysis_txt) as fh:
        lines = fh.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("(b)"):
            on = True
            continue
        if on and line.startswith("("):
            break
        m = _B.match(line) if on else None
        if not m:
            continue
        K, N, C = int(m[1]), int(m[2]), int(m[3])
        toks = m[4].split()
        for cfg, val in zip(toks[0::2], toks[1::2]):
            if cfg not in ("marlin", "tc_ladder", "cublas_dq"):
                continue
            for s, v in enumerate(val.split("|"), 1):
                rows.append(_row(K, N, 4, float(v), cfg, matrices=C, series=s))
    return rows


def e051_c_rows(analysis_txt):
    """E051 (c) single 4096^2 matrix, flush and no-flush GPU times (JSON lines)."""
    rows, on = [], False
    with open(analysis_txt) as fh:
        lines = fh.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("(c)"):
            on = True
            continue
        if on and line.startswith("("):
            break
        if not (on and line.startswith("{")):
            continue
        d = json.loads(line)
        for k, v in d.items():
            m = re.match(r"^(.*)_(flush|noflush)_us$", k)
            if not m:
                continue
            fl = m[2] == "flush"
            rows.append(_row(4096, 4096, d["M"], v, m[1],
                             matrices=None if fl else 1, flushed=fl))
    return rows


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    e051 = os.path.join(root, "results/e051_20261118/full-c8-343361/analysis.txt")
    sets = [("E050", e050_rows(os.path.join(root, "results/e050_20261118/full-c8-343308"))),
            ("E051b", e051_b_rows(e051)), ("E051c", e051_c_rows(e051))]
    for name, rows in sets:
        for r in rows:
            print(f"{name:6} {r['K']:>5}x{r['N']:<5} M{r['M']:<4} {r['config']:<15} "
                  f"x{r.get('matrices') or '-'} s{r.get('series', '-')} "
                  f"{'flush' if r.get('flushed') else '':5} {r['time_us']:9.2f} "
                  f"floor {r['floor_us']:8.2f}  {r['status']}")
