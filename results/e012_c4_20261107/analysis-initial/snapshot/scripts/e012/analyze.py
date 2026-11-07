#!/usr/bin/env python3
"""Check bounded E012 evidence and report work/sample shares, never phase time."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

from regions import resolve_region, source_map, verified_sources
from check_validation import SHAPES, MS, e011, main as validation_checked

SAMPLE = 'smsp__pcsamp_sample_count'
COORDINATION = {'block_reduction', 'global_reduction_fp16_branch', 'global_reduction_fp32',
                'global_reduction_control', 'lock_coordination', 'atomic_coordination',
                'epilogue', 'stripe_drain_barrier'}
AMBIGUOUS = {'unmapped', 'inline_or_external_ambiguous', 'phase_ambiguous'}


def kernel_checked(symbol, expected):
    assert e011.kernel_identity(symbol)[0] == e011.kernel_identity(expected)[0] == 'Marlin'
    assert e011.e010.template_values(symbol) == e011.e010.template_values(expected)


def records():
    cases = [(m, k, n) for k, n in SHAPES for m in MS]
    return ([('instruction', 1, *case) for case in cases] +
            [('sampling', 1, *case) for case in cases] +
            [('sampling', 2, *case) for case in reversed(cases)])


def profile_checked(root, kind, series, m, k, n, sources):
    tag = f'{kind}_s{series}_M{m}_K{k}_N{n}'
    for label in (tag, 'export_' + tag):
        assert (root / f'{label}.exit').read_text().strip() == '0', label
    report = root / 'ncu' / (tag + '.ncu-repz')
    assert report.stat().st_size > 0
    trace = json.loads((root / f'trace_marlin_M{m}_K{k}_N{n}.json').read_text())
    log = (root / f'{tag}.log').read_text()
    for text in ('--replay-mode application', '--app-replay-mode strict --app-replay-match grid',
                 '--profile-from-start off --graph-profiling node --cache-control none --clock-control none',
                 '--pipeline-boost-state dynamic --disable-extra-suffixes --launch-skip 399 --launch-count 1'):
        assert text in log, (tag, text)
    if kind == 'sampling':
        assert '--warp-sampling-interval 0 --warp-sampling-max-passes 1 --warp-sampling-buffer-size 33554432' in log
    passes = [json.loads(line.split('E011_PASS ', 1)[1]) for line in log.splitlines() if line.startswith('E011_PASS ')]
    assert passes, tag
    for row in passes:
        e011.matched(row, trace)
        assert (row['M'], row['K'], row['N'], row['config']) == (m, k, n, 'marlin')
        assert (row['warmup'], row['graph_cycles'], row['replay'], row['cache_control'], row['clock_control']) == (10, 100, 'application', 'none', 'none')
        assert row['policy'] == 'no_artificial_flush' and row['post_profile_validation'] == row['validation']
    units, raw = e011.raw(root / 'ncu' / f'{tag}.raw.csv')
    assert len(raw) == 1 and raw[0]['CC'] == '12.0'
    kernel_checked(raw[0]['Kernel Name'], trace['kernels_by_matrix'][3][0])
    assert len(passes) == int(e011.number(raw[0], 'profiler__replayer_passes'))
    data = json.loads(gzip.open(root / 'ncu' / f'{tag}.pc.json.gz', 'rt').read())
    assert Path(data['report']).name == report.name and data['source_sha256'] == sources
    kernel_checked(data['name'], trace['kernels_by_matrix'][3][0])
    requested = (Path(__file__).parent / f'{kind}-metrics.txt').read_text().splitlines()
    assert list(data['metric_metadata']) == requested
    for key, meta in data['metric_metadata'].items():
        assert units[key] == meta['unit'], key
        assert e011.number(raw[0], key) == meta['value'], key
    mapping, name = source_map(root / 'ncu' / f'{tag}.source.csv')
    kernel_checked(name, data['name'])
    metric = 'inst_executed' if kind == 'instruction' else SAMPLE
    assert data['metric_metadata'][metric]['instances'] > 0
    if kind == 'instruction':
        assert data['metric_metadata'][metric]['unit'] == 'inst'
    counts = {}
    wait_counts = {}
    covered, total = 0, 0
    seen = set()
    for pc in data['pcs']:
        address = pc['pc']
        assert type(address) is int and address not in seen
        seen.add(address)
        sites = mapping.get(address, {}).get('locations', [])
        assert pc['locations'] == sites and pc['region'] == resolve_region(sites)
        for key, value in pc['counts'].items():
            assert key in requested and type(value) is int and value >= 0
        value = pc['counts'].get(metric, 0)
        total += value
        if any(site['file'] and site['line'] > 0 for site in sites):
            covered += value
        counts[pc['region']] = counts.get(pc['region'], 0) + value
        for key, value in pc['counts'].items():
            if key.startswith('smsp__pcsamp_warps_issue_stalled_'):
                waits = wait_counts.setdefault(key, {})
                waits[pc['region']] = waits.get(pc['region'], 0) + value
    assert total > 0
    for key in requested:
        if key == 'inst_executed' or key == SAMPLE or key.startswith('smsp__pcsamp_warps_issue_stalled_'):
            values = [pc['counts'][key] for pc in data['pcs'] if key in pc['counts']]
            meta = data['metric_metadata'][key]
            assert len(values) == meta['instances'] and sum(values) == meta['value'], key
    quality = []
    if covered / total < .95:
        quality.append('source coverage below 95%')
    result = dict(kind=kind, series=series, M=m, K=k, N=n, total=total,
                  source_coverage_pct=100 * covered / total, validated_application_executions=len(passes),
                  counts=counts, wait_counts=wait_counts, quality=quality)
    if kind == 'instruction':
        result['hmma_instructions'] = sum(pc['counts'].get(metric, 0) for pc in data['pcs'] if 'HMMA.' in pc['sass'])
    if kind == 'sampling':
        meta = data['metric_metadata']
        for suffix, expected in [('buffer_overflow', 0), ('dropped_bytes', 0), ('aggregated_passes', 1)]:
            if meta['smsp__pcsamp_' + suffix]['value'] != expected:
                quality.append(f'{suffix} differs from {expected}')
        interval = meta['smsp__pcsamp_interval_cycles']['value']
        assert isinstance(interval, (int, float)) and math.isfinite(interval) and interval > 0
        result['interval_cycles'] = interval
        result['dropped_bytes'] = meta['smsp__pcsamp_dropped_bytes']['value']
        result['buffer_overflow'] = meta['smsp__pcsamp_buffer_overflow']['value']
        result['unmapped_pc_samples'] = counts.get('unmapped', 0)
        if total < 1000:
            quality.append('fewer than 1000 total samples')
    return result


def main(root, old, e006):
    validation_checked(root, old)
    assert (root / 'COLLECTION_COMPLETE').exists()
    for label in ('identity-measurement', 'identity-final'):
        assert (root / f'{label}.exit').read_text().strip() == '0'
    repo = Path(__file__).resolve().parents[2]
    tooling = (root / 'measurement-tooling.sha256').read_text().splitlines()
    for line in tooling:
        digest, path = line.split(maxsplit=1)
        assert hashlib.sha256((repo / path).read_bytes()).hexdigest() == digest, path
    for name, count in [('final-input-check.log', 3), ('final-library-check.log', 6), ('final-tooling-check.log', len(tooling))]:
        assert (root / name).read_text().count(': OK') == count, name
    sources = verified_sources(e006)
    rows = [profile_checked(root, *record, sources) for record in records()]
    instructions = [row for row in rows if row['kind'] == 'instruction']
    samples = [row for row in rows if row['kind'] == 'sampling']
    baseline = {(row['K'], row['N'], row['M']): row for row in json.loads((old/'profile-analysis.json').read_text()) if row['config']=='marlin'}
    sanity = []
    for row in instructions:
        expected = baseline[row['K'], row['N'], row['M']]['totals']['sm__inst_executed_pipe_tensor.sum']
        assert row['hmma_instructions'] == expected
        sanity.append(f"K={row['K']} N={row['N']} M={row['M']}: source HMMA count {row['hmma_instructions']} exactly matches E011 Tensor instruction count. This does not establish unperturbed timing.")
    (root/'instruction-sanity.log').write_text('\n'.join(sanity)+'\n')
    intervals = {row['interval_cycles'] for row in samples}
    if len(intervals) != 1:
        for row in samples:
            row['quality'].append('actual sampling intervals differ across cases')
    for k, n in SHAPES:
        for m in MS:
            a, b = [row for row in samples if (row['K'], row['N'], row['M']) == (k, n, m)]
            shifts = {phase: 100 * (b['counts'].get(phase, 0) / b['total'] - a['counts'].get(phase, 0) / a['total'])
                      for phase in a['counts'].keys() | b['counts'].keys()}
            a['series_shift_pp'] = b['series_shift_pp'] = shifts
            if max(abs(value) for value in shifts.values()) > 10:
                a['quality'].append('region share shifts more than 10 percentage points')
                b['quality'].append('region share shifts more than 10 percentage points')
    deltas = []
    for k, n in SHAPES:
        a, b = [row for row in instructions if row['K'] == k and row['N'] == n]
        total = b['total'] - a['total']
        known_a, known_b = (row['counts'].get('main_loop', 0) for row in (a, b))
        unknown_a, unknown_b = (sum(value for phase, value in row['counts'].items() if phase in AMBIGUOUS) for row in (a, b))
        low, high = known_b - known_a - unknown_a, known_b + unknown_b - known_a
        supported = total > 0 and low > .5 * total and not a['quality'] and not b['quality']
        deltas.append(dict(K=k, N=n, total_instruction_increase=total,
                           known_main_loop_increase=known_b-known_a,
                           main_loop_increase_bounds=[low, high], hypothesis_supported=supported))
    summary = []
    for row in rows:
        summary.append(dict(kind=row['kind'], series=row['series'], M=row['M'], K=row['K'], N=row['N'],
                            total=row['total'], main_loop_pct=100*row['counts'].get('main_loop',0)/row['total'],
                            coordination_epilogue_pct=100*sum(v for p,v in row['counts'].items() if p in COORDINATION)/row['total'],
                            ambiguous_pct=100*sum(v for p,v in row['counts'].items() if p in AMBIGUOUS)/row['total'],
                            source_coverage_pct=row['source_coverage_pct'], quality='; '.join(row['quality']) or 'pass'))
    with (root / 'attribution-summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (root / 'attribution.json').write_text(json.dumps(dict(rows=rows, deltas=deltas), indent=2, allow_nan=False)+'\n')
    table = ['Instruction counts are warp-level counts ignoring instruction predicates. Millions mean 10^6. Shares describe diagnostic instruction counts, not runtime. Known regions exclude ambiguous inline callers.', '',
             '| K / N | M | Warp instructions M | Known main loop % | Known coordination/epilogue % | Ambiguous phase % | Source coverage % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary[:4]:
        table.append(f"| {row['K']} / {row['N']} | {row['M']} | {row['total']/1e6:.6f} | {row['main_loop_pct']:.2f} | {row['coordination_epilogue_pct']:.2f} | {row['ambiguous_pct']:.2f} | {row['source_coverage_pct']:.2f} |")
    (root / 'instruction-table.md').write_text('\n'.join(table)+'\n')
    table = ['Sampling uses one pass at 32 cycles with a 32 MiB buffer. All eight captures fail the zero-dropped-data gate. M128 also fails 95% source coverage. These counts document a measurement limitation, not accepted stall attribution.', '',
             '| K / N | M | Series | Recorded samples | Dropped bytes | Unmapped all-ones PC samples | Source coverage % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in samples:
        table.append(f"| {row['K']} / {row['N']} | {row['M']} | {row['series']} | {row['total']} | {row['dropped_bytes']} | {row['unmapped_pc_samples']} | {row['source_coverage_pct']:.2f} |")
    (root / 'sampling-quality-table.md').write_text('\n'.join(table)+'\n')

    (root / 'ANALYSIS_COMPLETE').write_text('12 structurally complete profiles; quality gates and ambiguity bounds are recorded in attribution.json. No optimization accepted.\n')
    print('PASS: 4 instruction and 8 sampling profiles structurally complete, all replay outputs and exact sources checked')
    print('Quality:', sum(not row['quality'] for row in rows), '/12 cases pass predeclared per-case/series gates')
    print('Instruction hypothesis supported in both orientations:', all(row['hypothesis_supported'] for row in deltas))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'old', 'e006'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    main(args.root, args.old, args.e006)
