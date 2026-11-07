#!/usr/bin/env python3
"""Release E012 only after original identity and fresh target correctness pass."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'e011'))
spec = importlib.util.spec_from_file_location('e011_analysis', Path(__file__).resolve().parents[1] / 'e011/analyze.py')
e011 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e011)
SHAPES = [(4096, 11008), (11008, 4096)]
MS = [4, 128]


def trace_checked(row, old):
    e011.matched(row, old)
    assert (row['M'], row['K'], row['N'], row['config']) == (old['M'], old['K'], old['N'], 'marlin')
    for key in ('launch_skip', 'launch_count', 'target_cycle', 'matrix_position', 'kernels_by_matrix', 'cycle_kernels'):
        assert row[key] == old[key], key
    assert row['launch_skip'] == 399 and row['launch_count'] == 1


def main(root, old):
    assert (root / 'e011-handoff-check.log').read_text().count(': OK') == 362
    for label in ('query-source', 'query-warpsampling', 'source_M4', 'source_M128', 'report-interface-resume',
                  'identity', 'runtime', 'smoke', 'library-identity', 'retained-project-gpu'):
        assert (root / f'{label}.exit').read_text().strip() == '0', label
    for name in ('dependency-diff.txt',):
        assert not (root / name).read_text(), name
    assert (root / 'reused-library-check.log').read_text().count(': OK') == 6
    assert (root / 'reused-tooling-check.log').read_text().count(': OK') == 12
    assert (root / 'retained-project-gpu.log').read_text().count('ERROR SUMMARY: 0 errors') == 6
    for filename, collection in [('instruction-metrics.txt', 'query-source.log'), ('sampling-metrics.txt', None)]:
        text = (root / collection).read_text() if collection else '\n'.join((root / f'query-{x}.log').read_text() for x in ('source', 'warpsampling', 'stats'))
        for metric in (Path(__file__).parent / filename).read_text().splitlines():
            assert any(line.startswith(metric + ' ') for line in text.splitlines()), metric
    for k, n in SHAPES:
        for m in MS:
            name = f'trace_marlin_M{m}_K{k}_N{n}'
            assert (root / f'{name}.exit').read_text().strip() == '0', name
            trace_checked(json.loads((root / f'{name}.json').read_text()), json.loads((old / f'{name}.json').read_text()))
    print('PASS: original source/binary/environment, complete targets and full-output E011 agreement; same-environment safety evidence reused')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('old', type=Path)
    args = parser.parse_args()
    main(args.root, args.old)
