#!/usr/bin/env python3
"""E019 timing gate: Q decode median >=5% below fresh R in BOTH reversed series, per context. Stdlib only."""
import json
import sys
from pathlib import Path
from statistics import median

root, stage = Path(sys.argv[1]), sys.argv[2]
cells, result = {}, dict(stage=stage, contexts={})
for context in (128, 2048):
    series = []
    for s in (1, 2):
        row = {}
        for v in ('R', 'Q'):
            rows = [json.loads(line) for line in (root / f'{stage}_{v}_{context}_s{s}.jsonl').read_text().splitlines()]
            assert len(rows) == 20 and all(r['variant'] == v and r['context'] == context for r in rows)
            row[v] = {k: median(r[k] for r in rows) for k in ('decode_s', 'ttft_s', 'e2e_s')}
            row[v].update(decode_min=min(r['decode_s'] for r in rows), decode_max=max(r['decode_s'] for r in rows),
                          peak_allocated=max(r['peak_allocated'] for r in rows), peak_reserved=max(r['peak_reserved'] for r in rows))
        row['gain_pct'] = {k: 100 * (1 - row['Q'][k] / row['R'][k]) for k in ('decode_s', 'ttft_s', 'e2e_s')}
        series.append(row)
    result['contexts'][context] = dict(series=series, passes=all(x['gain_pct']['decode_s'] >= 5 for x in series))
result['heldout_required'] = stage == 'primary' and any(c['passes'] for c in result['contexts'].values())
print(json.dumps(result, indent=1))
