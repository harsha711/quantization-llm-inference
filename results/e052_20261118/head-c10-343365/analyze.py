#!/usr/bin/env python3
"""E052 analysis: analyze.py S STAGE — reads S/cells.jsonl (arm, file per timed cell), writes per-cell medians and the
predeclared comparisons. Decode = 63-step decode median; prefill = time to first token median (ours) / 1-token request (vLLM)."""
import json
import sys
from pathlib import Path

S, stage = Path(sys.argv[1]), sys.argv[2]
cells = {}
for line in (S / 'cells.jsonl').read_text().splitlines():
    c = json.loads(line)
    d = json.loads(Path(c['file']).read_text())
    assert len(d['rows']) == 23, c
    cells[(c['arm'], c['context'], c['prompt'], c['series'])] = dict(
        decode_ms=1e3 * d['decode_s'], prefill_ms=1e3 * d.get('ttft_s', d.get('first_s', float('nan'))), e2e_ms=1e3 * d['e2e_s'],
        decode_range_ms=[1e3 * d['decode_min'], 1e3 * d['decode_max']], peak_allocated=d.get('peak_allocated'),
        tokens_agree_min=d.get('tokens_agree_min'))
contexts = sorted({k[1] for k in cells})
arms = sorted({k[0] for k in cells})


def keys(context):
    return [(p, s) for p in ('primary', 'heldout') for s in (1, 2)]


def gain(a, b, context, metric='decode_ms'):
    """Percent by which arm a is below arm b in each of the four cells (positive = a faster)."""
    return [100 * (1 - cells[(a, context, p, s)][metric] / cells[(b, context, p, s)][metric]) for p, s in keys(context)
            if (a, context, p, s) in cells and (b, context, p, s) in cells]


def outcome(g):
    """E027 rule: faster if a >= 5% below b in all four cells, slower if b >= 5% below a in all four, comparable if all within +-5%."""
    if len(g) != 4:
        return 'incomplete'
    if all(x >= 5 for x in g):
        return 'faster'
    if all((1 - x / 100) >= 1 / 0.95 for x in g):  # b >= 5% below a  <=>  b/a <= 0.95  <=>  a/b >= 1/0.95
        return 'slower'
    if all(abs(x) < 5 for x in g):  # a within +-5% of b in every cell
        return 'comparable'
    return 'mixed'


COMPARE = dict(
    core=[('GTA', 'GRA', 'H1'), ('GTA', 'GQA', 'H2'), ('GT', 'GQ', ''), ('TA', 'QA', ''), ('GTA', 'TA', ''), ('GQA', 'QA', ''),
          ('GTA', 'GR', ''), ('GQA', 'GRA', ''), ('GT', 'GR', ''), ('GRA', 'GR', '')],
    long=[('GTA', 'GRA', 'H1'), ('GTA', 'GQA', 'H2'), ('GQA', 'GRA', '')],
    head=[('GTAH', 'GTA', 'H5'), ('GTAH', 'GQAH', 'H2'), ('GTA', 'GQA', 'H2'), ('GQAH', 'GQA', ''), ('VH', 'V', ''),
          ('GTA', 'V', 'H7'), ('GTAH', 'VH', 'H7'), ('cGTAH', 'cGTA', 'H6'), ('cGTA', 'cGQA', 'H2'), ('cGTAH', 'cGQAH', 'H2'),
          ('cGQAH', 'cGQA', ''), ('cVH', 'cV', '')],
    ext=[('GTA', 'GQA', 'H2'), ('GTA', 'V', 'H7'), ('GTA', 'OQ', 'H7'), ('GTA', 'LQ', 'H7'), ('GQA', 'OQ', ''), ('GQA', 'LQ', ''),
         ('GQA', 'V', '')])[stage]
result = dict(stage=stage, arms=arms, contexts=contexts, comparisons=[],
              medians={f'{a}|{c}|{p}|s{s}': v for (a, c, p, s), v in sorted(cells.items(), key=str)})
for c in contexts:
    for a, b, h in COMPARE:
        for metric in ('decode_ms', 'prefill_ms'):
            g = gain(a, b, c, metric)
            if not g:
                continue
            row = dict(context=c, a=a, b=b, metric=metric, hypothesis=h, pct_a_below_b=[round(x, 3) for x in g], outcome=outcome(g))
            if h in ('H1', 'H5', 'H6') and metric == 'decode_ms':
                row['passes_5pct_all_cells'] = len(g) == 4 and all(x >= 5 for x in g)
            result['comparisons'].append(row)
if stage == 'head':  # E038 equal speed: calibrated within 5% of plain-rounded, per arm and cell
    result['equal_speed'] = {a: [[round(x, 3) for x in gain('c' + a, a, c)] for c in contexts] for a in ('GTA', 'GTAH', 'GQA', 'GQAH', 'V', 'VH')
                             if any(('c' + a, c, p, s) in cells for c in contexts for p, s in keys(c))}
print(json.dumps(result, indent=1))
