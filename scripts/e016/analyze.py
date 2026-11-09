#!/usr/bin/env python3
"""Validate complete fresh E016 records and compare matched decode medians."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from checks import checked, passes, traversal
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'e014'))
from timeline import summarize


def read(path):
    return json.loads(path.read_text())


def successful(root, tag):
    assert (root/(tag+'.exit')).read_text().strip() == '0', tag
    return read(root/(tag+'.json'))


def validate(root):
    assert not (root/'dependency-diff.txt').read_text()
    eager = successful(root, 'validation_G_128')['identity']
    summaries = []
    for length, variant in traversal(1):
        tag = f'validation_{variant}_{length}'
        cell = successful(root, tag)
        assert (cell['variant'], cell['context'], cell['stage']) == (variant, length, 'validation')
        for key in ('weights_sha256', 'model_revision', 'source', 'gpu', 'torch', 'transformers', 'attention', 'threads'):
            assert cell['identity'][key] == eager[key], key
        assert cell['identity']['parameter_count'] == 1543714304
        assert cell['identity']['parameter_bytes'] == 3087428608
        assert cell['identity']['model_revision'] == '8faed761d45a263340a0528343f099c05c9a4323'
        assert len(cell['warmups']) == 3
        for row in cell['warmups']:
            ref = read(root/f'validation_primary_{length}_64-reference.json')
            checked(row, ref, length)
        for counters in (cell['compiler_before'], cell['compiler_after']):
            assert not counters.get('stats', {}).get('unique_graphs', 0)
            assert not counters.get('graph_break')
        assert cell['compiler_before'] == cell['compiler_after']
        assert [v['prompt'] for v in cell['validation']] == ['primary', 'heldout', 'primary']
        for index, item in enumerate(cell['validation']):
            ref = read(root/f"validation_{item['prompt']}_{length}_64-reference.json")
            checked(item['record'], ref, length)
            errors = item['errors']
            assert errors['finite'] and errors['bad_logits'] == 0 and errors['relative_l2'] <= .003
            assert errors['logits_checked'] == 9723904
            if variant == 'G':
                historical = item['historical']
                assert historical['finite'] and historical['bad_logits'] == 0 and historical['relative_l2'] <= .003
            assert errors == read(root/f"{tag}_{item['prompt']}_{index}-errors.json")
            assert (root/f"{tag}_{item['prompt']}_{index}.npz").stat().st_size > 0
            if variant == 'G' and index == 0:
                assert [v['step'] for v in item['uncached']] == [0, 63]
                assert all(v['bad_logits'] == 0 and v['relative_l2'] <= .003 for v in item['uncached'])
        with gzip.open(root/(tag+'-trace.json.gz'), 'rt') as f:
            events = json.load(f)['traceEvents']
        summary = summarize(events, 'E016/decode')
        assert summary['graph_launch_calls'] == (63 if variant == 'G' else 1)
        assert not any('Torch-Compiled' in name for name in summary['compiled_markers'])
        kernels = [e for e in events if e.get('cat') == 'kernel']
        assert sum('fmha_cutlassF_f16' in e['name'] for e in kernels) == 1792
        assert not any('flash' in e['name'].lower() for e in kernels)
        ref = read(root/f'validation_primary_{length}_64-reference.json')
        checked(cell['trace_record'], ref, length)
        marker = next(e for e in events if e.get('name') == 'E016/decode' and e.get('cat') == 'user_annotation')
        counts = Counter(e['name'] for e in kernels if marker['ts'] <= e['ts'] < marker['ts']+marker['dur'])
        summaries.append(dict(context=length, variant=variant, decode=summary, kernel_counts=dict(counts),
                              validation=[v['errors'] for v in cell['validation']]))
    parity = []
    for length in (128, 2048):
        pair = {v['variant']:v for v in summaries if v['context'] == length}
        parity.append(dict(context=length, same_kernel_multiset=pair['G']['kernel_counts'] == pair['H']['kernel_counts']))
    output = dict(status='full numerical and replay validation passed', cases=4, requests=12,
                  identity=eager, summaries=summaries, kernel_parity=parity)
    (root/'validation-analysis.json').write_text(json.dumps(output, indent=2)+'\n')
    return output


def phase(root, name, identity):
    assert not (root/(name+'-dependency-diff.txt')).read_text()
    rows = []
    for series in (1, 2):
        for length, variant in traversal(series):
            tag = f'{name}_s{series}_{variant}_{length}'
            cell = successful(root, tag)
            assert (cell['variant'], cell['context'], cell['stage']) == (variant, length, name)
            for key in ('weights_sha256', 'source', 'gpu', 'torch', 'transformers', 'attention', 'threads'):
                assert cell['identity'][key] == identity[key]
            assert cell['compiler_before'] == cell['compiler_after']
            assert not cell['compiler_after'].get('stats', {}).get('unique_graphs', 0)
            samples = [json.loads(x) for x in (root/(tag+'.jsonl')).read_text().splitlines()]
            assert len(samples) == 20 and [v['sample'] for v in samples] == list(range(20))
            prompt = 'primary' if name == 'primary' else 'heldout'
            ref = read(root/f'validation_{prompt}_{length}_64-reference.json')
            for row in samples:
                checked(row, ref, length)
                assert row['prompt'] == prompt and row['series'] == series and row['variant'] == variant
                assert row['weights_sha256'] == identity['weights_sha256']
                assert row['gpu'] == identity['gpu'] and row['backend'] == identity['attention']
            assert len(cell['warmups']) == 3
            primary_ref = read(root/f'validation_primary_{length}_64-reference.json')
            for row in cell['warmups']:
                checked(row, primary_ref, length)
            metrics = ('decode_s', 'ttft_s', 'e2e_s', 'peak_allocated', 'peak_reserved')
            rows.append(dict(phase=name, series=series, context=length, variant=variant,
                **{k:statistics.median(v[k] for v in samples) for k in metrics},
                decode_min_s=min(v['decode_s'] for v in samples), decode_max_s=max(v['decode_s'] for v in samples),
                decode_p90_s=sorted(v['decode_s'] for v in samples)[17],
                static_setup_s=cell['static_setup_s'], graph_setup_s=cell['graph_setup_s'],
                first_request_s=cell['warmups'][0]['e2e_s']))
    comparisons = []
    for length in (128, 2048):
        gains, changes = [], []
        for series in (1, 2):
            pair = {v['variant']:v for v in rows if v['context'] == length and v['series'] == series}
            gains.append(100*(1-pair['H']['decode_s']/pair['G']['decode_s']))
            changes.append({key:100*(pair['H'][key]/pair['G'][key]-1) for key in ('ttft_s', 'e2e_s', 'peak_reserved')})
        comparisons.append(dict(context=length, decode_gain_pct=gains, passes=passes(gains), other_changes_pct=changes))
    return rows, comparisons


def main(root, stage):
    validation = validate(root)
    if stage == 'validation':
        print('PASS four full numerical/replay cases, twelve reset/alternating-prompt requests')
        print('Kernel parity:', validation['kernel_parity'])
        return
    for name in ('primary',) if stage == 'primary' else ('primary', 'heldout'):
        for line in (root/(name+'-tooling.sha256')).read_text().splitlines():
            digest, path = line.split(maxsplit=1)
            assert hashlib.sha256((Path(__file__).resolve().parents[2]/path).read_bytes()).hexdigest() == digest, path
    rows, primary = phase(root, 'primary', validation['identity'])
    required = any(v['passes'] for v in primary)
    heldout = []
    if stage == 'heldout':
        assert required
        extra, heldout = phase(root, 'heldout', validation['identity'])
        rows += extra
    summary = dict(stage=stage, primary=primary, heldout=heldout, heldout_required=required,
        request_samples=len(rows)*20, confirmed_contexts=[p['context'] for p,h in zip(primary, heldout) if p['passes'] and h['passes']])
    (root/(stage+'-analysis.json')).write_text(json.dumps(summary, indent=2)+'\n')
    with (root/(stage+'-summary.csv')).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    p.add_argument('--stage', choices=('validation', 'primary', 'heldout'), default='validation')
    a = p.parse_args()
    main(a.results, a.stage)
