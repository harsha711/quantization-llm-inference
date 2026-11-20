"""DRAM-floor and working-set check for W4A16 GEMM timings (v2, 2026-11-20).

Statuses, strongest first:
  BELOW_STRICT    time < weights(+scales) / 448 GB/s spec and < FLOP / 56.95 TFLOP/s
                  (max clock): physically impossible from DRAM -> cache-served.
  BELOW_MEASURED  time < full byte model / 427.2 GB/s (E057 best measured read) or
                  FLOP / 51.84 TFLOP/s (E057, 2812 MHz): almost certainly cache-served,
                  but not a proof (A shared across cycled matrices, C write-back, clocks).
  WS_SMALL        above the floor but bytes rotated between reuses < 2 x L2:
                  DRAM regime NOT CONFIRMED (heuristic margin; not evidence of caching).
  OK              above floor, working set >= 2 x L2: mostly DRAM (E049 still saw
                  4-6 % L2 re-reads at 2.8 x L2).
  OK_FLUSHED      above floor, store-flush before each launch: eviction not proven,
                  and flush write-back may INFLATE the time.
Constants are for ONE RTX 5060 Ti (c8, E057); parsers refuse other GPUs where the
row names its GPU. FP32-accumulate HMMA assumed (Marlin, tc*, cuBLAS mode 16).

Usage: python3 scripts/checks/floorcheck.py
Tests: python3 -m unittest scripts/checks/test_floorcheck.py -v
"""
import csv
import glob
import json
import os
import re
import statistics

BW_BYTES_PER_S = 427.2e9
STRICT_BW_BYTES_PER_S = 448e9
PEAK_FLOP_PER_S = 51.84e12
STRICT_PEAK_FLOP_PER_S = 512 * 36 * 3.090e9
L2_BYTES = 32 * 1024 * 1024
L2_MARGIN = 2.0
GROUP = 128
GPU = "RTX 5060 Ti"

OK = "OK"
OK_FLUSHED = "OK_FLUSHED"
BELOW_STRICT = "BELOW_STRICT"
BELOW_MEASURED = "BELOW_MEASURED"
WS_SMALL = "WS_SMALL"


def kind_of(config):
    if config.startswith("cublas"):
        return "fp16"
    if config in ("unfused", "dequant"):
        return config
    return "w4"


def _w4(K, N):
    return K * N // 2 + 2 * (K // GROUP) * N


def strict_bytes(K, N, kind="w4"):
    """Weights (+scales) that must cross DRAM once; A, C, temporaries omitted."""
    return 2 * K * N if kind == "fp16" else _w4(K, N)


def bytes_model(K, N, M, kind="w4"):
    if kind == "dequant":
        return _w4(K, N) + 2 * K * N
    act = 2 * M * K + 2 * M * N
    if kind == "fp16":
        return 2 * K * N + act
    if kind == "unfused":
        return _w4(K, N) + 2 * (2 * K * N) + act
    return _w4(K, N) + act


def _flop(K, N, M, kind):
    return 0 if kind == "dequant" else 2 * M * N * K


def floor_us(K, N, M, kind="w4"):
    return max(bytes_model(K, N, M, kind) / BW_BYTES_PER_S,
               _flop(K, N, M, kind) / PEAK_FLOP_PER_S) * 1e6


def strict_floor_us(K, N, M, kind="w4"):
    return max(strict_bytes(K, N, kind) / STRICT_BW_BYTES_PER_S,
               _flop(K, N, M, kind) / STRICT_PEAK_FLOP_PER_S) * 1e6


def margin(K, N, M, time_us, kind="w4"):
    return time_us / floor_us(K, N, M, kind)


def working_set(K, N, matrices, kind="w4"):
    if kind == "fp16":
        return matrices * 2 * K * N
    if kind in ("unfused", "dequant"):
        return matrices * _w4(K, N) + 2 * K * N  # one reused W-hat temporary
    return matrices * _w4(K, N)


def classify(K, N, M, time_us, matrices=None, flushed=False, kind="w4"):
    if matrices is None and not flushed:
        raise ValueError("need matrices (cycle size) or flushed=True")
    if time_us < strict_floor_us(K, N, M, kind):
        return BELOW_STRICT
    if time_us < floor_us(K, N, M, kind):
        return BELOW_MEASURED
    if flushed:
        return OK_FLUSHED
    if working_set(K, N, matrices, kind) < L2_MARGIN * L2_BYTES:
        return WS_SMALL
    return OK


def _row(K, N, M, t, config, matrices=None, flushed=False, **kw):
    kind = kind_of(config)
    r = dict(K=K, N=N, M=M, time_us=t, config=config, matrices=matrices,
             flushed=flushed, kind=kind, **kw)
    r["floor_us"] = floor_us(K, N, M, kind)
    r["margin"] = t / r["floor_us"]
    r["status"] = classify(K, N, M, t, matrices=matrices, flushed=flushed, kind=kind)
    return r


def _jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def e050_rows(directory):
    """E050-style cyclic series: per-GEMM time = median(cycle samples) / matrices."""
    rows = []
    for f in sorted(glob.glob(os.path.join(directory, "series*.jsonl"))):
        for r in _jsonl(f):
            t = statistics.median(r["samples_us"]) / r["matrices"]
            rows.append(_row(r["K"], r["N"], r["M"], t, r["config"], matrices=r["matrices"],
                             series=r["series"], recorded_us=r.get("median_per_matrix_us")))
    return rows


def e051_b_rows(directory):
    """E051 (b) raw b-series*.jsonl (E009 cycle at M4), same schema as E050."""
    rows = []
    for f in sorted(glob.glob(os.path.join(directory, "b-series*.jsonl"))):
        for r in _jsonl(f):
            t = statistics.median(r["samples_us"]) / r["matrices"]
            rows.append(_row(r["K"], r["N"], r["M"], t, r["config"], matrices=r["matrices"],
                             series=r["series"], recorded_us=r.get("median_per_matrix_us")))
    return rows


def e051_c_rows(analysis_txt):
    """E051 (c) single 4096^2 matrix GPU row, flush and no-flush (analyzer JSON lines)."""
    with open(analysis_txt) as fh:
        lines = [x.strip() for x in fh]
    rows, on = [], False
    for line in lines:
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
            if m:
                fl = m[2] == "flush"
                rows.append(_row(4096, 4096, d["M"], v, m[1],
                                 matrices=None if fl else 1, flushed=fl))
    return rows


def e051_a_rows(directory):
    """E051 (a) single-matrix sweeps a-main-s*.csv and a-held-s*.csv (bench CSV)."""
    rows = []
    files = glob.glob(os.path.join(directory, "a-main-s*.csv")) + glob.glob(os.path.join(directory, "a-held-s*.csv"))
    for f in sorted(files):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if GPU not in r["gpu"]:
                    raise ValueError(f"constants are for {GPU}, row is {r['gpu']}")
                fl = r["flush_l2"] == "1"
                rows.append(_row(int(r["K"]), int(r["N"]), int(r["M"]), float(r["median_us"]),
                                 r["kernel"], matrices=None if fl else 1, flushed=fl,
                                 source=os.path.basename(f)))
    return rows


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    d51 = os.path.join(root, "results/e051_20261118/full-c8-343361")
    sets = [("E050", e050_rows(os.path.join(root, "results/e050_20261118/full-c8-343308"))),
            ("E051a", e051_a_rows(d51)), ("E051b", e051_b_rows(d51)),
            ("E051c", e051_c_rows(os.path.join(d51, "analysis.txt")))]
    for name, rows in sets:
        for r in rows:
            print(f"{name:6} {r['K']:>5}x{r['N']:<5} M{r['M']:<4} {r['config']:<20} "
                  f"x{r['matrices'] or '-'} s{r.get('series', '-')} {'flush' if r['flushed'] else '':5} "
                  f"{r['time_us']:9.2f} floor {r['floor_us']:8.2f} x{r['margin']:.3f}  {r['status']}")


if __name__ == "__main__":
    main()
