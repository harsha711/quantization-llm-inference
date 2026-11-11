#!/usr/bin/env python3
"""E026 analysis: analyze.py DIR CONTEXT... — decode gains of the new decode attention per (context, passage, series)."""
import json
import sys
from pathlib import Path

root, contexts = Path(sys.argv[1]), [int(c) for c in sys.argv[2:]]
cells, passes = [], {}
for context in contexts:
    ok = True
    for prompt in ('primary', 'heldout'):
        for s in (1, 2):
            m = {v: json.loads((root / f'{prompt}_{v}_{context}_s{s}.json').read_text()) for v in ('GR', 'GQ', 'GRA', 'GQA')}
            assert all(len(x['rows']) == 23 for x in m.values())
            gain = lambda a, b: round(100 * (1 - m[a]['decode_s'] / m[b]['decode_s']), 3)
            row = dict(context=context, prompt=prompt, series=s, decode_ms={v: round(1e3 * m[v]['decode_s'], 3) for v in m},
                       decode_range_ms={v: [round(1e3 * m[v]['decode_min'], 3), round(1e3 * m[v]['decode_max'], 3)] for v in m},
                       e2e_ms={v: round(1e3 * m[v]['e2e_s'], 3) for v in m},
                       gain_pct=dict(GQA_vs_GQ=gain('GQA', 'GQ'), GRA_vs_GR=gain('GRA', 'GR'), GQA_vs_GRA=gain('GQA', 'GRA'),
                                     GQA_vs_GR=gain('GQA', 'GR'), GQ_vs_GR=gain('GQ', 'GR')))
            ok &= row['gain_pct']['GQA_vs_GQ'] >= 5
            cells.append(row)
    passes[context] = ok
print(json.dumps(dict(cells=cells, passes_5pct_vs_GQ_all_cells=passes), indent=1))
