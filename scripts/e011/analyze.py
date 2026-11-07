#!/usr/bin/env python3
"""Validate E011 operation coverage and fresh matched endpoint evidence."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

from plan import CYCLES, MS, NAMES, SHAPES, WARMUP, target

spec = importlib.util.spec_from_file_location('e010_analysis', Path(__file__).resolve().parents[1] / 'e010/analyze.py')
e010 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e010)


def raw(path):
    rows = list(csv.reader(path.open()))
    assert len(rows) >= 3 and all(len(row) == len(rows[0]) for row in rows)
    return dict(zip(rows[0], rows[1])), [dict(zip(rows[0], row)) for row in rows[2:]]


def number(row, key):
    value = float(row[key].replace(',', ''))
    assert math.isfinite(value), (key, row[key])
    return value


def matched(row, reference):
    assert row['body_sha256'] == reference['body_sha256'] == e010.e008.BODY
    assert row['cublas_math_mode'] == reference['cublas_math_mode'] == 16
    assert row['activation_prefix_sha256'] == reference['activation_prefix_sha256']
    assert (row['capacity'], row['matrices'], row['seeds'], row['group']) == (128, 4, [3, 4, 5, 6], 128)
    assert row['direct_cycles'] == row['poisoned_graph_cycles'] == 3
    assert len(row['allocations']) == len(row['validation']) == 4
    ranges = [row['allocations'][0]['activation_range']]
    for operand, old, check, old_check in zip(row['allocations'], reference['allocations'], row['validation'], reference['validation']):
        for key in ('operand_sha256', 'packed_sha256', 'scales_sha256', 'wh_sha256'):
            assert operand[key] == old[key], key
        assert operand['activation_range'] == row['allocations'][0]['activation_range']
        for key in ('packed', 'scales', 'wh', 'output', 'locks', 'scratch'):
            ranges.append(operand[key + '_range'])
        assert check['num_bad'] == 0 and check['checked'] == row['M'] * row['N']
        assert check['output_sha256'] == old_check['output_sha256']
    e010.e008.disjoint(ranges)
    assert len(row['resources']) == (4 if row['config'] == 'marlin' else 0)
    for current, old in zip(row['resources'], reference['resources']):
        assert {k: v for k, v in current.items() if k != 'kernel_address'} == {k: v for k, v in old.items() if k != 'kernel_address'}


def main(root, stage):
    def success(label):
        assert (root / (label + '.exit')).read_text().strip() == '0', label

    old = [json.loads(line) for line in (root / 'e010-validation-reference.jsonl').read_text().splitlines()]
    lookup = {(v['K'], v['N'], v['M'], v['config']): v for v in old}
    success('calibration')
    _, calibration = raw(root / 'ncu/calibration.csv')
    assert len(calibration) == 1 and 'add' in calibration[0]['Kernel Name'].lower(), calibration
    assert '--launch-skip 399 --launch-count 1' in (root / 'calibration.log').read_text()
    query = (root / 'supported-metrics.txt').read_text() + (root / 'launch-metrics.txt').read_text()
    available = {line.split()[0] for line in query.splitlines() if line.strip()}
    metrics = Path(__file__).with_name('metrics.txt').read_text().splitlines()
    assert len(metrics) == len(set(metrics)) == 33
    assert all(metric in available for metric in metrics), set(metrics) - available
    assert (root / 'e010-handoff-check.log').read_text().count(': OK') == 153
    assert not (root / 'dependency-diff.txt').read_text()
    for label in ('identity', 'identity-validation', 'runtime', 'smoke', 'library-identity'):
        success(label)
    assert all(json.loads(line)['validation'][i]['num_bad'] == 0 for line in (root / 'smoke.jsonl').read_text().splitlines() for i in range(4))
    assert (root / 'validation-library-check.log').read_text().count(': OK') == 6
    assert (root / 'validation-input-check.log').read_text().count(': OK') == 3
    traces = {}
    for k, n in SHAPES:
        for m in MS:
            for name in NAMES:
                tag = f'{name}_M{m}_K{k}_N{n}'
                success('trace_' + tag)
                row = json.loads((root / ('trace_' + tag + '.json')).read_text())
                assert (row['K'], row['N'], row['M'], row['config']) == (k, n, m, name)
                matched(row, lookup[k, n, m, name])
                plan = target(row['kernels_by_matrix'])
                assert all(row[key] == value for key, value in plan.items())
                assert row['cycle_kernels'] == row['kernels_by_matrix'][0] * 4
                for suffix, expected in [('target', row['kernels_by_matrix'][3]), ('cycle', row['cycle_kernels'])]:
                    trace = json.loads((root / f'trace_{tag}.{suffix}.trace.json').read_text())
                    assert [event['name'] for event in trace['traceEvents'] if event.get('cat') == 'kernel'] == expected
                if name == 'marlin':
                    assert plan['launch_count'] == 1
                    a = row['resources'][0]
                    assert e010.template_values(row['kernels_by_matrix'][3][0])[-8:] == [a['threads'], a['m_blocks'], a['thread_n']//16, a['thread_k']//16, a['m8'], a['stages'], 8, 0]
                traces[tag] = row
    if stage == 'validation':
        print('PASS: eight complete-operation traces, E010 operands/outputs/resources, profiler-window calibration, 33 supported metrics')
        return
    timings = []
    for series, shapes, ms in [(1, SHAPES, MS), (2, SHAPES[::-1], MS[::-1])]:
        success(f'series{series}')
        rows = [json.loads(line) for line in (root / f'series{series}.jsonl').read_text().splitlines()]
        assert [(v['K'], v['N'], v['M'], v['config'], v['order']) for v in rows] == e010.traversal(shapes, ms, series)
        for row in rows:
            e010.checked(row)
            matched(row, lookup[row['K'], row['N'], row['M'], row['config']])
            assert row['warmup'] == WARMUP and row['iters'] == CYCLES and row['series'] == series
            samples = row['samples_us']
            assert len(samples) == CYCLES and all(math.isfinite(x) and x > 0 for x in samples)
            assert row['median_us'] == statistics.median(samples)
            assert (row['min_us'], row['max_us'], row['p90_us']) == (min(samples), max(samples), sorted(samples)[89])
            assert row['post_timing_validation'] == row['validation']
        timings.extend(rows)
    comparisons = []
    for k, n in SHAPES:
        for m in MS:
            for series in (1, 2):
                pair = {v['config']: v for v in timings if (v['K'], v['N'], v['M'], v['series']) == (k, n, m, series)}
                a, b = pair['marlin']['median_us'], pair['cublas_dq']['median_us']
                comparisons.append(dict(K=k, N=n, M=m, series=series, marlin_cycle_us=a, cublas_cycle_us=b,
                                        speedup=b/a, marlin_gain_pct=100*(1-a/b)))
    with (root / 'comparisons.csv').open('w') as out:
        writer = csv.DictWriter(out, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    profiles = []
    for tag, trace in traces.items():
        success('profile_' + tag)
        text = (root / ('profile_' + tag + '.log')).read_text()
        passes = [json.loads(line.split('E011_PASS ', 1)[1]) for line in text.splitlines() if line.startswith('E011_PASS ')]
        assert passes, tag
        for row in passes:
            matched(row, trace)
            assert row['post_profile_validation'] == row['validation']
        unit, values = raw(root / f'ncu/{tag}.csv')
        assert len(values) == trace['launch_count'], (tag, len(values))
        kernels = []
        for i, value in enumerate(values):
            expected = trace['kernels_by_matrix'][3][i]
            if trace['config'] == 'marlin':
                assert e010.template_values(value['Kernel Name']) == e010.template_values(expected)
            else:
                assert value['Kernel Name'].replace(' ', '') == expected.replace(' ', '')
            normalized = {metric: number(value, metric) for metric in metrics}
            assert unit['gpu__time_duration.sum'] == 'ns'
            normalized['duration_us'] = normalized['gpu__time_duration.sum']/1000
            kernels.append(dict(name=value['Kernel Name'], units={metric: unit[metric] for metric in metrics}, metrics=normalized))
        # Keep per-kernel utilization/resources. Sum only sequential, additive work.
        additive = ['duration_us', 'dram__bytes_op_read.sum', 'dram__bytes_op_write.sum', 'lts__t_bytes.sum',
                    'smsp__inst_executed.sum', 'sm__inst_executed_pipe_tensor.sum',
                    'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum', 'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']
        profiles.append(dict(K=trace['K'], N=trace['N'], M=trace['M'], config=trace['config'],
                             kernels=kernels, totals={metric: sum(kernel['metrics'][metric] for kernel in kernels) for metric in additive},
                             validated_application_executions=len(passes)))
    (root / 'profile-analysis.json').write_text(json.dumps(profiles, indent=2) + '\n')
    print('PASS: 16 fresh endpoint timing rows/1600 complete-cycle samples and eight context-preserving complete-operation profiles')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--stage', choices=['validation', 'complete'], default='complete')
    args = parser.parse_args()
    main(args.results, args.stage)
