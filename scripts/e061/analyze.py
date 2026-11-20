#!/usr/bin/env python3
"""E061 analyzer. analyze.py DIR  -> JSON summary (series1.jsonl, series2.jsonl in DIR); exit 1 unless every
cell is floorcheck OK. Logic in scripts/e061/e061lib.py; tests scripts/e061/test_e061.py."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('E061_ROOT', os.path.abspath(os.path.join(HERE, '..', '..')))
spec = importlib.util.spec_from_file_location('e061lib', os.environ.get('E061_LIB', os.path.join(HERE, 'e061lib.py')))
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)
E050 = os.path.join(ROOT, 'results/e050_20261118/full-c8-343308')


def e050_m128():
    """E050's cuBLAS cell at the same shape and M128, per series: median(raw samples) / 4."""
    out = {}
    for f in ('series1.jsonl', 'series2.jsonl'):
        with open(os.path.join(E050, f)) as fh:
            for line in fh:
                r = json.loads(line)
                if (r['K'], r['N'], r['M'], r['config']) == (lib.K, lib.N, 128, lib.CONFIG):
                    assert r['series'] not in out
                    out[r['series']] = lib.per_gemm_us(r)
    assert sorted(out) == [1, 2], out
    return out


def _rows(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def analyze(d):
    return lib.summarize(_rows(os.path.join(d, 'series1.jsonl')), _rows(os.path.join(d, 'series2.jsonl')), e050_m128())


if __name__ == '__main__':
    s = analyze(sys.argv[1])
    print(json.dumps(s, indent=1))
    for k, c in s['cells'].items():
        print(f"M{c['M']:<4} s{c['series']} {c['us']:9.2f} us  {c['tflops']:6.2f} TFLOP/s  {c['flop_per_byte']:6.1f} "
              f"FLOP/B  {c['share_pct']:5.1f}% {c['bound']}  floor {c['floor_us']:.2f}  {c['status']}", file=sys.stderr)
    print('E061', 'ALL VALID' if s['all_valid'] else f"INVALID {s['invalid']}", file=sys.stderr)
    sys.exit(0 if s['all_valid'] else 1)
