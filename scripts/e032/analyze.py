#!/usr/bin/env python3
"""E032 analysis: analyze.py DIR — perplexities, relative changes, and top-1 agreement between variants."""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
q = {v: json.loads((root / f'quality-{v}.json').read_text()) for v in ('O', 'OH', 'R', 'RH')}
t = {v: np.load(root / f'top1-{v}.npz')['top1'] for v in q}
assert len({(x['text_sha256'], x['scored_tokens']) for x in q.values()}) == 1
p = {v: q[v]['perplexity'] for v in q}
rise = lambda a, b: round(100 * (p[a] / p[b] - 1), 3)
agree = lambda a, b: round(100 * float((t[a] == t[b]).mean()), 3)
print(json.dumps(dict(scored_tokens=q['O']['scored_tokens'], perplexity=p, top1_accuracy_pct={v: round(100 * q[v]['top1_accuracy'], 3) for v in q},
                      perplexity_rise_pct=dict(head_on_quantized_blocks=rise('RH', 'R'), head_on_original=rise('OH', 'O'),
                                               blocks=rise('R', 'O'), blocks_and_head=rise('RH', 'O')),
                      top1_agreement_pct=dict(RH_vs_R=agree('RH', 'R'), OH_vs_O=agree('OH', 'O'), R_vs_O=agree('R', 'O'), RH_vs_O=agree('RH', 'O'))), indent=1))
