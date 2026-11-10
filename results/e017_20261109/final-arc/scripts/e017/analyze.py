#!/usr/bin/env python3
"""Offline E017 evidence gates and fresh matched latency comparison (stdlib only)."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

BODY = 'e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7'
WHEEL = 'e634b80e72078e964f8bd2ac541d66f2a900920773cf993544609612190bd9a9'
BRIDGE = '85f332951a8c24b3eb06594977b3e3d62b7954fa01b986da41ffeff851bfbd3f'
ORIGINAL = 'e2317857636f95884c753ae8f8f8ed09e58ef2760c8d6e07a3c8e61cf146571f'
REVISION = '8faed761d45a263340a0528343f099c05c9a4323'
REPRESENTATIVE = 'model.layers.0.mlp.down_proj'
SHAPES = {'q_proj': (1536, 1536), 'k_proj': (1536, 256), 'v_proj': (1536, 256),
          'o_proj': (1536, 1536), 'gate_proj': (1536, 8960),
          'up_proj': (1536, 8960), 'down_proj': (8960, 1536)}


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def functions():
    path = Path(__file__).with_name('infer.py')
    selected = [n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name in ('checked', 'accept_logits')]
    assert len(selected) == 2
    namespace = dict(math=math)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['checked'], namespace['accept_logits']


checked, accept_logits = functions()


def success(root, label):
    assert (root / (label + '.exit')).read_text().strip() == '0', label
    return (root / (label + '.log')).read_text()


def finite(value, positive=False):
    assert isinstance(value, (int, float)) and math.isfinite(value)
    assert value > 0 if positive else value >= 0


def coverage(identity):
    expected = {}
    for index in range(28):
        for leaf, (k, n) in SHAPES.items():
            block = 'self_attn' if leaf in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp'
            name = f'model.layers.{index}.{block}.{leaf}'
            expected[name] = (k, n, leaf in ('q_proj', 'k_proj', 'v_proj'))
    rows = identity['coverage']
    assert len(rows) == 196
    actual = {r['name']: (r['K'], r['N'], r['bias']) for r in rows}
    assert len(actual) == len(rows) and actual == expected


def identity_checked(identity, variant, common=None, same_variant=None):
    assert identity['variant'] == variant
    assert identity['original_weights_sha256'] == ORIGINAL
    assert identity['model_revision'] == REVISION
    assert identity['model_id'] == 'Qwen/Qwen2.5-1.5B'
    assert identity['torch'] == '2.14.0+cu130' and identity['transformers'] == '4.57.1'
    assert identity['attention'] == 'sdpa/EFFICIENT_ATTENTION only'
    assert identity['threads'] == 16 and identity['graphs'] is False
    assert identity['parameter_count'] == 1543714304 and identity['parameter_bytes'] == 3087428608
    assert identity['corrected_body_sha256'] == (BODY if variant == 'Q' else None)
    assert bool(identity['artifacts_sha256']) == (variant != 'O')
    assert identity['linear_bias_policy'] == 'FP16 GEMM followed by separate FP16 bias addition in O/R/Q'
    coverage(identity)
    if common is not None:
        for key in ('original_weights_sha256', 'model_revision', 'source', 'gpu', 'torch',
                    'transformers', 'attention', 'threads', 'coverage', 'linear_bias_policy'):
            assert identity[key] == common[key], key
        if variant != 'O' and common['variant'] != 'O':
            assert identity['artifacts_sha256'] == common['artifacts_sha256']
    if same_variant is not None:
        for key in ('weights_sha256', 'artifacts_sha256', 'corrected_body_sha256'):
            assert identity[key] == same_variant[key], key
    resident = identity['resident']
    assert resident['bytes'] == sum(x['bytes'] for x in resident['allocations']) > 0
    assert len({(x['device'], x['pointer']) for x in resident['allocations']}) == len(resident['allocations'])
    names = {name for allocation in resident['allocations'] for name in allocation['names']}
    for row in identity['coverage']:
        name = row['name']
        if variant == 'Q':
            assert name + '.weight' not in names
            assert name + '.qweight' in names and name + '.scales' in names
        else:
            assert name + '.weight' in names
            assert name + '.qweight' not in names and name + '.scales' not in names


def compiler(cell):
    assert cell['compiler_before'] == cell['compiler_after']
    assert not cell['compiler_after'].get('stats', {}).get('unique_graphs', 0)
    assert not cell['compiler_after'].get('graph_break')
    assert cell['weights_unchanged'] is True


def runner_storage(cell, length):
    resident = cell['runner_resident']
    allocations = resident['allocations']
    assert len(allocations) == len({(a['device'], a['pointer']) for a in allocations}) == 64
    assert all(a['pointer'] > 0 and isinstance(a['bytes'], int) and a['bytes'] > 0 for a in allocations)
    capacity = length + 64
    cache_bytes = 56 * 2 * capacity * 128 * 2
    logits_bytes = 64 * 151936 * 2
    buffer_bytes = (capacity + 1 + 1 + 64 + 63 + 63) * 8 + 56
    assert resident['bytes'] == sum(a['bytes'] for a in allocations) == cache_bytes + logits_bytes + buffer_bytes
    model = {(a['device'], a['pointer']) for a in cell['identity']['resident']['allocations']}
    assert not model.intersection((a['device'], a['pointer']) for a in allocations)


def row_checked(row, reference, length, variant, generated=64):
    checked(row, reference, length, generated)
    assert row['variant'] == variant
    workspace = row['operator_workspace']
    if variant == 'Q':
        assert set(workspace) == {'locks_zero', 'lock_guards', 'scratch_guards'} and all(workspace.values())
    else:
        assert workspace is None
    assert len(row['input_ids']) == length and len(row['output_ids']) == generated
    assert row['input_sha256'] == hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest()
    assert row['eos_policy'] == 'fixed length, EOS recorded and not used to stop'
    for key in ('allocated', 'reserved'):
        finite(row[key], positive=True)
    assert row['allocated'] <= row['reserved'] <= row['peak_reserved']
    assert row['allocated'] <= row['peak_allocated']
    if 'eos_positions' in reference:
        assert row['eos_positions'] == [i for i in reference['eos_positions'] if i < generated]


def binaries(root):
    rows = [line.split(maxsplit=1) for line in (root / 'binaries.sha256').read_text().splitlines()]
    hashes = {digest for digest, _ in rows}
    assert {BODY, WHEEL, BRIDGE} <= hashes


def source_manifest(root, stage):
    path = root / f'source-{stage}.sha256'
    project = Path(__file__).resolve().parents[2]
    rows = [line.split(maxsplit=1) for line in path.read_text().splitlines()]
    assert rows
    for digest, name in rows:
        relative = Path(name.removeprefix('*'))
        assert not relative.is_absolute() and '..' not in relative.parts
        assert hashlib.sha256((project / relative).read_bytes()).hexdigest() == digest, name


def operator_case(root, m, name, metadata, activations, tool=None, first_layer=False):
    base = f'one_layer_M{m}' if first_layer else f'operator_{name}_M{m}'
    tag = base + (f'-{tool}' if tool else '')
    label = (f'one-layer-{tool}-M{m}' if tool else f'one-layer-M{m}') if first_layer else tag
    log = success(root, label)
    if tool:
        assert '--error-exitcode 99' in log
        if first_layer or tool == 'racecheck':
            assert '--kernel-name kns=e005_marlin' in log
        else:
            assert '--kernel-name' not in log
        summary = 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors'
        assert summary in log
        assert not re.search(r'(?:ERROR SUMMARY:|RACECHECK SUMMARY:) [1-9]', log)
    records = [read(root / f'{tag}_{i}.json') for i in range(3)]
    expected_activation = activations[name][f'A_{128 if m in (1, 128) else 2048}_{m}']
    for index, row in enumerate(records):
        assert row['iteration'] == index and row['finite'] and row['bad'] == 0
        assert row['checked'] == m * metadata['N']
        for key in ('max_abs', 'rms', 'rel_l2'):
            finite(row[key])
        assert all(row['workspace'].values()) and len(row['workspace']) == 3
        assert row['weight_hashes'] == metadata['hashes']
        assert row['activation_sha256'] == expected_activation
        resource = row['resources']
        assert resource['candidate'] == 0 and resource['sms'] == 36
        assert resource['grid'] > 0 and resource['grid'] % resource['sms'] == 0
        assert resource['thread_k'] in (64, 128) and resource['thread_n'] in (64, 128, 256)
        assert resource['threads'] in (128, 256) and resource['stages'] == 4
        assert 1 <= resource['m_blocks'] <= 4 and resource['m8'] == (m <= 8)
        assert resource['kernel_address'] > 0 and resource['max_blocks_per_sm'] > 0
        assert resource['binary_version'] == 120
        assert row['resources'] == records[0]['resources']
        assert (root / f'{tag}_{index}-outputs.npz').stat().st_size > 0
    for key in ('candidate_sha256', 'reference_sha256', 'activation_sha256'):
        assert len({row[key] for row in records}) == 1
    for index in range(2):
        row = read(root / f'{tag}_module_{index}.json')
        assert row['bitwise_equal'] and all(row['workspace'].values())
        assert row['bias'] == (name.rsplit('.', 1)[1] in ('q_proj', 'k_proj', 'v_proj'))
        assert (root / f'{tag}_module_{index}.npz').stat().st_size > 0
    if not tool:
        trace = read(root / f'{tag}-trace-summary.json')
        assert trace['expected_launches'] == len(trace['kernels']) == (2 if m == 2048 and metadata['N'] > 4096 else 1)
        assert all('e005_marlin::Marlin<' in name for name in trace['kernels'])
        assert trace['resources'] == records[0]['resources']
        assert (root / f'{tag}-trace.json.gz').stat().st_size > 0
    return dict(name=name, M=m, tool=tool, max_abs=max(r['max_abs'] for r in records))


def prerequisite(root):
    binaries(root)
    assert not (root / 'dependency-diff.txt').read_text()
    for label in ('quantizer-check', 'inference-local-check', 'runtime',
                  'retained-build', 'retained-ctest', 'retained-gpu-check', 'capture'):
        log = success(root, label)
        if label in ('quantizer-check', 'inference-local-check', 'runtime', 'capture'):
            assert 'PASS' in log
    meta = read(root / 'one-layer.json')
    assert meta['name'] == REPRESENTATIVE and (meta['K'], meta['N']) == (8960, 1536)
    identity_checked(meta['identity'], 'O')
    cases = [operator_case(root, m, REPRESENTATIVE, meta, meta['activations'], tool, True)
             for tool in (None, 'memcheck', 'synccheck', 'racecheck')
             for m in ((1, 128) if tool == 'racecheck' else (1, 128, 2048))]
    return dict(status='one-layer numerical, module-path and eight sanitizer cases passed', cases=cases)


def validation(root):
    prerequisite(root)
    source_manifest(root, 'validation')
    manifest_path = root / 'artifacts/manifest.json'
    manifest = read(manifest_path)
    assert manifest['original_weights_sha256'] == ORIGINAL and len(manifest['layers']) == 196
    artifact_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    activations = read(root / 'one-layer.json')['activations']
    for name in ('model.layers.0.self_attn.q_proj', 'model.layers.0.self_attn.k_proj',
                 'model.layers.0.mlp.gate_proj', REPRESENTATIVE):
        for tool in (None, 'memcheck', 'synccheck', 'racecheck'):
            for m in ((1, 128) if tool == 'racecheck' else (1, 128, 2048)):
                operator_case(root, m, name, manifest['layers'][name], activations, tool)
    cells = {}
    for length in (128, 2048):
        for variant in ('O', 'R', 'Q'):
            tag = f'validation_{variant}_{length}'
            success(root, tag)
            cell = read(root / (tag + '.json'))
            assert (cell['stage'], cell['variant'], cell['context']) == ('validation', variant, length)
            identity_checked(cell['identity'], variant, cells.get((128, 'O'), cell)['identity'],
                             cells.get((128, variant), cell)['identity'])
            assert cell['identity']['coverage'] == manifest['coverage']
            if variant != 'O':
                assert cell['identity']['artifacts_sha256'] == artifact_hash
            compiler(cell)
            runner_storage(cell, length)
            records = cell['records']
            assert len(records) == 4 and [r['prompt'] for r in records] == ['primary', 'heldout'] * 2
            raw = lines(root / (tag + '.jsonl'))
            assert [r['sample'] for r in raw] == list(range(4))
            for index, item in enumerate(records):
                prompt = item['prompt']; row = item['record']
                assert raw[index]['record'] == row and row['prompt'] == prompt
                case = f'{tag}_{prompt}_{index}'
                assert row == read(root / (case + '-output.json'))
                assert item == read(root / (case + '-diagnostics.json'))
                assert read(root / (case + '-weights.json'))['unchanged'] is True
                assert len(row['cache_hashes']) == 56
                assert all(re.fullmatch(r'[0-9a-f]{64}', h) for h in row['cache_hashes'])
                first = records[index % 2]
                ref = cells[(length, 'R')]['records'][index % 2]['record'] if variant == 'Q' else first['record']
                row_checked(row, ref, length, variant)
                repeat = item['repetition']
                accept_logits(repeat)
                assert repeat == read(root / (case + '-repeat-errors.json'))
                assert repeat['bitwise_equal'] and repeat['logits_checked'] == 9723904
                assert repeat['max_abs'] == repeat['relative_l2'] == 0
                assert repeat['candidate_sha256'] == repeat['reference_sha256'] == first['repetition']['candidate_sha256']
                assert row['cache_hashes'] == first['record']['cache_hashes']
                assert (root / (case + '-output.npz')).stat().st_size > 0
                assert (root / (case + '-repeat-comparison.npz')).stat().st_size > 0
                if variant == 'Q':
                    error = item['errors']; accept_logits(error)
                    finite(error['relative_l2']); finite(error['max_abs'])
                    assert error == read(root / (case + '-reference-errors.json'))
                    assert error['logits_checked'] == 9723904
                    assert error['candidate_sha256'] == repeat['candidate_sha256']
                    assert error['reference_sha256'] == cells[(length, 'R')]['records'][index % 2]['repetition']['candidate_sha256']
                    assert (root / (case + '-reference-comparison.npz')).stat().st_size > 0
                else:
                    assert item['errors'] is None
                if variant == 'R':
                    change = item['quantization_changes']
                    assert change == read(root / (case + '-quantization.json'))
                    assert change['shape_equal'] and change['finite'] and change['logits_checked'] == 9723904
                    original = cells[(length, 'O')]['records'][index % 2]['record']
                    assert change['tokens_equal'] == (row['output_ids'] == original['output_ids'])
                    # Original-model differences are observations, not equivalence gates.
                    finite(change['relative_l2']); finite(change['max_abs'])
            cells[(length, variant)] = cell
        assert cells[(length, 'Q')]['identity']['artifacts_sha256'] == cells[(length, 'R')]['identity']['artifacts_sha256']
        trace = read(root / f'trace-coverage-{length}.json')
        assert trace['passed'] and trace['modulecoverage'] == 196
        assert trace['candidateprefillMarlinlaunches'] == (196 if length == 128 else 252)
        assert trace['decode'] == 1372
    return cells


def traversal(series):
    cells = [(c, v) for c in (128, 2048) for v in ('R', 'Q')]
    return cells if series == 1 else cells[::-1]


def phase(root, name, validated):
    source_manifest(root, name)
    expected = [f'{name}_{variant}_{length}_s{series}'
                for series in (1, 2) for length, variant in traversal(series)]
    order = lines(root / f'order-{name}.jsonl')
    assert [row['label'] for row in order] == expected
    for row in order:
        assert isinstance(row['started_utc_ns'], int) and isinstance(row['finished_utc_ns'], int)
        assert 0 < row['started_utc_ns'] < row['finished_utc_ns']
    assert all(a['finished_utc_ns'] <= b['started_utc_ns'] for a, b in zip(order, order[1:]))
    rows = []
    for series in (1, 2):
        for length, variant in traversal(series):
            tag = f'{name}_{variant}_{length}_s{series}'
            success(root, tag)
            cell = read(root / (tag + '.json')); compiler(cell)
            runner_storage(cell, length)
            assert (cell['stage'], cell['variant'], cell['context'], cell['series']) == (name, variant, length, series)
            identity_checked(cell['identity'], variant, validated[(length, variant)]['identity'], validated[(length, variant)]['identity'])
            prompt = 'primary' if name == 'primary' else 'heldout'
            reference = validated[(length, 'R')]['records'][0 if prompt == 'primary' else 1]['record']
            warm = lines(root / (tag + '-warmups.jsonl'))
            assert len(warm) == 3 and [r['sample'] for r in warm] == list(range(3))
            for row in warm:
                row_checked(row['record'], reference, length, variant)
                assert row['record']['prompt'] == prompt
            samples = lines(root / (tag + '.jsonl'))
            assert len(samples) == 20 and [r['sample'] for r in samples] == list(range(20))
            for row in samples:
                row_checked(row, reference, length, variant)
                assert row['prompt'] == prompt and row['series'] == series
                assert row['identity'] == cell['identity']
            metric = dict(phase=name, series=series, context=length, variant=variant)
            for key in ('decode_s', 'ttft_s', 'e2e_s', 'peak_allocated', 'peak_reserved', 'allocated', 'reserved'):
                metric[key] = statistics.median(r[key] for r in samples)
            metric.update(decode_min_s=min(r['decode_s'] for r in samples),
                          decode_max_s=max(r['decode_s'] for r in samples),
                          decode_p90_s=sorted(r['decode_s'] for r in samples)[17],
                          first_request_s=warm[0]['record']['e2e_s'],
                          model_resident_bytes=cell['identity']['resident']['bytes'],
                          runner_resident_bytes=cell['runner_resident']['bytes'],
                          load_s=cell['identity']['load_s'], replace_s=cell['identity']['replace_s'],
                          total_runner_setup_s=cell['total_runner_setup_s'])
            rows.append(metric)
    comparisons = []
    for length in (128, 2048):
        gains, changes = [], []
        for series in (1, 2):
            pair = {r['variant']: r for r in rows if r['context'] == length and r['series'] == series}
            gains.append(100 * (1 - pair['Q']['decode_s'] / pair['R']['decode_s']))
            changes.append({key: 100 * (pair['Q'][key] / pair['R'][key] - 1)
                            for key in ('ttft_s', 'e2e_s', 'peak_allocated', 'peak_reserved', 'model_resident_bytes')})
        comparisons.append(dict(context=length, decode_gain_pct=gains,
                                passes=all(math.isfinite(g) and g >= 5 for g in gains), other_changes_pct=changes))
    return rows, comparisons


def stopped(root):
    for phase_name in ('primary', 'heldout'):
        assert all(not p.read_text().strip() for p in root.glob(f'{phase_name}_*_s[12].jsonl')), 'timing exists'
    failures = []
    for path in sorted(root.glob('*.exit')):
        code = path.read_text().strip()
        if code and code != '0':
            failures.append(dict(file=path.name, exit=code))
    for path in sorted(root.glob('failure-*.json')):
        value = read(path)
        assert value.get('error')
        failures.append(dict(file=path.name, error=value['error']))
    for path in sorted(root.glob('one_layer_M*_*.json')):
        row = read(path)
        if 'bad' in row and (row['bad'] or not row.get('finite', True)):
            failures.append(dict(file=path.name, bad=row['bad'], finite=row.get('finite')))
    assert failures, 'no concrete preserved failure'
    return dict(status='stopped before performance release', failures=failures, request_samples=0)


def main(root, stage):
    root = Path(root)
    if stage == 'stopped':
        result = stopped(root)
    elif stage == 'prerequisite':
        result = prerequisite(root)
    else:
        validated = validation(root)
        if stage == 'validation':
            result = dict(status='full model numerical/state/identity and trace gates passed', cases=6, requests=24)
        else:
            rows, primary = phase(root, 'primary', validated)
            required = any(p['passes'] for p in primary)
            heldout = []
            if not required:
                assert all(not p.read_text().strip() for p in root.glob('heldout_*_s[12].jsonl')), 'unreleased held-out timing exists'
            if stage == 'heldout':
                assert required, 'no primary context qualified'
                extra, heldout = phase(root, 'heldout', validated)
                rows += extra
            result = dict(stage=stage, primary=primary, heldout=heldout, heldout_required=required,
                          request_samples=len(rows) * 20,
                          confirmed_contexts=[p['context'] for p, h in zip(primary, heldout) if p['passes'] and h['passes']])
            with (root / f'{stage}-summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader(); writer.writerows(rows)
    (root / f'{stage}-analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--stage', choices=('prerequisite', 'validation', 'primary', 'heldout', 'stopped'), required=True)
    args = parser.parse_args()
    main(args.results, args.stage)
