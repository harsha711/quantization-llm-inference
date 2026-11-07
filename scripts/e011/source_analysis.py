#!/usr/bin/env python3
"""Finite source-derived partition/FP32-scratch model, not measured traffic."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

from plan import MS, SHAPES

root = Path(sys.argv[1])
repo = Path(__file__).resolve().parents[2]
relative = 'external/patched/csrc/quantization/marlin/marlin_template.h'
baseline = repo / 'results/e006_20261106'
with tarfile.open(baseline / 'matched-source.tar.gz') as archive:
    source = archive.extractfile(relative).read()
expected = next(line.split()[0] for line in (baseline / 'patched/build-source.sha256').read_text().splitlines()
                if line.split()[-1] == relative)
assert hashlib.sha256(source).hexdigest() == expected
text = source.decode()
for formula in ('int global_mn_tiles = parallel * n_tiles;', 'part2_mn_tiles * 3 <= gridDim.x',
                'int iters = div_ceil(k_tiles * part2_mn_tiles, gridDim.x);',
                'constexpr int active_threads = 32 * tb_n_warps;',
                'k += (m_block_size_8 ? 2 : 1)'):
    assert formula in text, formula
ceil = lambda a, b: (a + b - 1) // b
rows = []
for k, n in SHAPES:
    for m in MS:
        trace = json.loads((root / f'trace_marlin_M{m}_K{k}_N{n}.json').read_text())
        a = trace['resources'][0]
        tile_m = 8 if a['m8'] else 16 * a['m_blocks']
        parallel = m // tile_m if m > tile_m else 1
        kt, nt, grid = k // a['thread_k'], n // a['thread_n'], a['grid']
        assert kt * a['thread_k'] == k and nt * a['thread_n'] == n
        tiles = parallel * nt
        part2 = tiles
        if tiles > grid:
            part2 = tiles % grid
            if 3 * part2 <= grid:
                part2 += grid
        part1 = (tiles - part2) // grid
        stripe = ceil(kt * part2, grid)
        ratio = 128 // a['thread_k']
        stripe = ratio * ceil(stripe, ratio)
        pieces = [[] for _ in range(part2)]
        for block in range(grid):
            lo, hi = block * stripe, min((block + 1) * stripe, part2 * kt)
            for column in range(part2):
                overlap = max(0, min(hi, (column + 1) * kt) - max(lo, column * kt))
                if overlap:
                    pieces[column].append((block, overlap))
        assert all(sum(count for _, count in parts) == kt for parts in pieces)
        links = sum(len(parts) - 1 for parts in pieces)
        active_threads = 32 * (a['thread_n'] // 16 // 4)
        th_size = a['m_blocks'] * 32 * 4 // 16
        partial_bytes = active_threads * ceil(th_size, 2 if a['m8'] else 1) * 16
        rows.append(dict(K=k, N=n, M=m, tile_m=tile_m, tile_k=a['thread_k'], tile_n=a['thread_n'],
                         parallel_m_tiles=parallel, global_mn_tiles=tiles, full_tiles_per_cta=part1,
                         striped_mn_tiles=part2, stripe_k_steps=stripe, reduction_links=links,
                         fp32_partial_bytes_per_link_direction=partial_bytes,
                         modeled_scratch_read_plus_write_bytes=2 * links * partial_bytes))
for a, b in zip(rows[::2], rows[1::2]):
    assert a['M'] == 4 and b['M'] == 128
    assert a['global_mn_tiles'] == b['global_mn_tiles'] and a['reduction_links'] == b['reduction_links']
    assert b['stripe_k_steps'] == 2 * a['stripe_k_steps']
    assert b['modeled_scratch_read_plus_write_bytes'] == 16 * a['modeled_scratch_read_plus_write_bytes']
result = dict(source=relative, source_sha256=expected, source_lines=[326, 378, 1376, 1418],
              interpretation='Finite source request model, includes padded rows. Not measured L2/DRAM traffic or execution time.', rows=rows)
(root / 'source-partitions.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
