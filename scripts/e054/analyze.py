#!/usr/bin/env python3
"""E054 per-node analysis (criteria predeclared in docs/EXPERIMENTS.md E054): analyze.py S
Reads S/cells.jsonl (one row per timed cell) and S/prof-{T,M}/ours.json (E030 gprofile output). Writes JSON to stdout.
Cells per node/context: 2 passages x 4 series = 8. Decode = median of the 20 per-request 63-step decode times."""
import json
import sys
from pathlib import Path

import numpy as np

S = Path(sys.argv[1])
cells = {}
for line in (S / 'cells.jsonl').read_text().splitlines():
    c = json.loads(line)
    d = json.loads(Path(c['file']).read_text())
    assert len(d['rows']) == 23, c
    kept = d['rows'][3:]
    dec = np.array([r['decode_s'] for r in kept]) * 1e3
    assert abs(np.median(dec) - 1e3 * d['decode_s']) < 1e-9, c
    cells[(c['arm'], c['context'], c['prompt'], c['series'])] = dict(
        decode_ms=1e3 * d['decode_s'], prefill_ms=1e3 * d['ttft_s'], samples=dec.tolist(), gpu=d.get('gpu'))
contexts = sorted({k[1] for k in cells})
KEYS = [(p, s) for p in ('primary', 'heldout') for s in (1, 2, 3, 4)]
rng = np.random.default_rng(53)


def gains(a, b, c, metric='decode_ms'):
    return [100 * (1 - cells[(a, c, p, s)][metric] / cells[(b, c, p, s)][metric]) for p, s in KEYS]


def outcome(g):
    """E027 rule over all cells: faster/slower if >= 5% in every cell, comparable if all within +-5%, else mixed."""
    if len(g) != len(KEYS):
        return 'incomplete'
    if all(x >= 5 for x in g):
        return 'faster'
    if all((1 - x / 100) >= 1 / 0.95 for x in g):
        return 'slower'
    if all(abs(x) < 5 for x in g):
        return 'comparable'
    return 'mixed'


def boot(a, b, c, p, s, n=10000):
    """95% bootstrap interval of median(a)/median(b) per cell (independent resamples of each arm's 20 requests)."""
    x, y = np.array(cells[(a, c, p, s)]['samples']), np.array(cells[(b, c, p, s)]['samples'])
    r = np.median(rng.choice(x, (n, x.size)), axis=1) / np.median(rng.choice(y, (n, y.size)), axis=1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


out = dict(node=S.name, gpu=sorted({v['gpu'] for v in cells.values()}), contexts=contexts, cells=len(cells), per_context={})
for c in contexts:
    assert all((arm, c, p, s) in cells for arm in ('GRA', 'GQ', 'GQA', 'GT', 'GTA') for p, s in KEYS), c
    row = {}
    for a, b, tag in (('GTA', 'GQA', 'R1'), ('GT', 'GQ', 'stock'), ('GTA', 'GRA', 'H1'), ('GQA', 'GRA', 'marlin_vs_fp16')):
        for metric in ('decode_ms', 'prefill_ms'):
            g = gains(a, b, c, metric)
            row[f'{a}_vs_{b}_{metric}'] = dict(tag=tag, pct_a_below_b=[round(x, 3) for x in g], min=round(min(g), 3), max=round(max(g), 3),
                                              outcome=outcome(g))
        if tag == 'H1':
            row['H1_pass'] = all(x >= 5 for x in gains(a, b, c))
    for a, b in (('GTA', 'GQA'), ('GT', 'GQ')):  # R2: direction with per-cell bootstrap intervals
        ci = {f'{p}_s{s}': boot(a, b, c, p, s) for p, s in KEYS}
        ratios = [cells[(a, c, p, s)]['decode_ms'] / cells[(b, c, p, s)]['decode_ms'] for p, s in KEYS]
        row[f'R2_{a}_vs_{b}'] = dict(ratio_min=round(min(ratios), 5), ratio_max=round(max(ratios), 5), ratio_median=round(float(np.median(ratios)), 5),
                                     all_cells_lower=all(x < 1 for x in ratios), all_ci_below_1=all(v[1] < 1 for v in ci.values()),
                                     holds=all(x < 1 for x in ratios) and all(v[1] < 1 for v in ci.values()), ci95=ci)
    row['median_ms'] = {arm: round(float(np.median([cells[(arm, c, p, s)]['decode_ms'] for p, s in KEYS])), 3) for arm in ('GRA', 'GQ', 'GQA', 'GT', 'GTA')}
    row['median_prefill_ms'] = {arm: round(float(np.median([cells[(arm, c, p, s)]['prefill_ms'] for p, s in KEYS])), 3) for arm in ('GRA', 'GQA', 'GTA')}
    out['per_context'][c] = row

# Attribution: per-step device time by kernel class from E030's gprofile (64-token minus 8-token request, /56)
prof = {}
for k in ('T', 'M'):
    f = S / f'prof-{k}' / 'ours.json'
    if f.exists():
        prof[k] = json.loads(f.read_text())
if len(prof) == 2:
    out['attribution'] = {}
    for c in prof['T']:
        split = {}
        for k in ('T', 'M'):
            table = prof[k][c]['kernel_us_and_calls_per_step']
            gemm = {n: v for n, v in table.items() if ('w4a16_tc' in n) or ('Marlin' in n) or ('inblock' in n)}
            split[k] = dict(total_us=prof[k][c]['device_us_per_step'], gemm_us=sum(v[0] for v in gemm.values()),
                            gemm_calls=sum(v[1] for v in gemm.values()), gemm_kernels=sorted(gemm),
                            other_us=prof[k][c]['device_us_per_step'] - sum(v[0] for v in gemm.values()))
        gemm_diff = split['M']['gemm_us'] - split['T']['gemm_us']
        ci = int(c)
        med = out['per_context'].get(ci, out['per_context'].get(str(ci)))
        step_diff = None if med is None else 1e3 * (med['median_ms']['GQA'] - med['median_ms']['GTA']) / 63
        out['attribution'][c] = dict(ours=split['T'], marlin=split['M'], gemm_diff_us_per_step=gemm_diff, decode_diff_us_per_step=step_diff,
                                     other_diff_us_per_step=split['M']['other_us'] - split['T']['other_us'],
                                     within_25pct=None if not step_diff else abs(gemm_diff - step_diff) <= 0.25 * abs(step_diff))
print(json.dumps(out, indent=1))
