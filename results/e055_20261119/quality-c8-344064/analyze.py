#!/usr/bin/env python3
"""E055 analysis (criteria predeclared in docs/EXPERIMENTS.md E055): analyze.py S -> JSON on stdout."""
import json
import sys
from pathlib import Path

import numpy as np

S = Path(sys.argv[1])
q = {v: json.loads((S / f'quality-{v}.json').read_text()) for v in ('O', 'R', 'QM', 'QT')}
z = {v: np.load(S / f'top1-{v}.npz') for v in q}
assert len({(r['tokens'], r['windows'], r['scored_tokens'], r['text_sha256']) for r in q.values()}) == 1


def label(x):
    return 'negligible' if abs(x) <= 1 else 'modest' if abs(x) <= 5 else 'substantial'


out = dict(variants={v: dict(perplexity=r['perplexity'], top1_accuracy=r['top1_accuracy'], body=r.get('body')) for v, r in q.items()},
           scored=q['R']['scored_tokens'], r_reproduces_e032=round(q['R']['perplexity'], 3) == 9.735, comparisons={})
for a, b in (('QM', 'R'), ('QT', 'R'), ('QT', 'QM'), ('R', 'O'), ('QT', 'O'), ('QM', 'O')):
    pct = 100 * (q[a]['perplexity'] / q[b]['perplexity'] - 1)
    d = z[a]['window_nll'] - z[b]['window_nll']
    out['comparisons'][f'{a}_vs_{b}'] = dict(ppl_pct=pct, label=label(pct), top1_agreement=float((z[a]['top1'] == z[b]['top1']).mean()),
                                            window_nll_diff_mean=float(d.mean()), window_nll_diff_max_abs=float(np.abs(d).max()),
                                            windows_a_higher=int((d > 0).sum()), windows_a_lower=int((d < 0).sum()), windows_equal=int((d == 0).sum()))
c = out['comparisons']
out['kernels_negligible'] = c['QM_vs_R']['label'] == 'negligible' and c['QT_vs_R']['label'] == 'negligible'
out['no_quality_difference_between_kernels'] = abs(c['QT_vs_QM']['ppl_pct']) <= 0.1 and c['QT_vs_QM']['top1_agreement'] >= 0.99
print(json.dumps(out, indent=1))
