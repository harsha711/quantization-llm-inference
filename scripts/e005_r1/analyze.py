#!/usr/bin/env python3
"""Check complete corrected A/B/C evidence and apply the unchanged E005 criteria."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('results', type=Path)
a = p.parse_args()
r = a.results
repo = Path(__file__).resolve().parents[2]
body = 'e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7'
policies = ('store_flush','no_flush')
shapes = ((4096,4096),(4096,11008),(11008,4096))

def write_csv(name, rows):
    with (r/name).open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

def rows(name, ms, dims, seed):
    assert (r/(name+'.exit')).read_text().strip() == '0', name
    data = [json.loads(line) for line in (r/(name+'.jsonl')).read_text().splitlines()]
    expected = {(m,k,n,c,policy) for m,(k,n),c,policy in itertools.product(ms,dims,'ABC',policies)}
    assert len(data) == len(expected) and {(v['M'],v['K'],v['N'],v['config'],v['policy']) for v in data} == expected
    assert body in (r/(name+'.log')).read_text()
    for v in data:
        assert v['seed'] == seed and v['group'] == 128 and v['pattern'] == 'mixed'
        assert v['num_bad'] == 0 and v['checked'] == v['M']*v['N'] and v['graph_check']
        assert v['repeated_launches'] == 3 and v['method'] == 'graph_events'
        assert all(v[k] for k in ('codes_match','scales_match','reconstruction_match','workspace_guards','lock_reset'))
        attr = v['resources']
        if v['M'] <= 8:
            grid,shared,capacity = {'A':(36,101376,1),'B':(36,49664,2),'C':(72,49664,2)}[v['config']]
            assert (attr['grid'],attr['dynamic_shared'],attr['max_blocks_per_sm']) == (grid,shared,capacity)
            assert attr['registers'] == 126 and attr['local_bytes'] == 32
        else:
            assert attr['candidate'] == 0 and attr['grid'] == 36 and attr['dynamic_shared'] == 101376
        assert attr['threads'] == 256 and attr['static_shared'] == 0
        samples = v['samples_us']
        assert v['warmup'] == 10 and v['iters'] == len(samples) == 100
        assert all(math.isfinite(x) and x > 0 for x in samples)
        assert statistics.median(samples) == v['median_us']
        assert sorted(samples)[89] == v['p90_us'] and min(samples) == v['min_us'] and max(samples) == v['max_us']
    for m,(k,n) in itertools.product(ms,dims):
        group = [v for v in data if (v['M'],v['K'],v['N']) == (m,k,n)]
        assert len({v['resources']['kernel_address'] for v in group}) == 1
        assert len({json.dumps(v['operand_sha256'],sort_keys=True) for v in group}) == 1
    return data

series = {name:rows(name,ms,dims,seed) for name,ms,dims,seed in [
    ('series1',(1,8),shapes[:1],1),('series2',(1,8),shapes[:1],1),
    ('fallback1',(16,64),shapes[:1],1),('fallback2',(16,64),shapes[:1],1),
    ('heldout',(2,4),shapes,2)]}
comparison = []
sample_stats = []
for name,data in series.items():
    for v in data:
        base = next(x for x in data if (x['M'],x['K'],x['N'],x['policy'],x['config']) == (v['M'],v['K'],v['N'],v['policy'],'A'))
        b = next(x for x in data if (x['M'],x['K'],x['N'],x['policy'],x['config']) == (v['M'],v['K'],v['N'],v['policy'],'B'))
        comparison.append(dict(series=name,M=v['M'],K=v['K'],N=v['N'],policy=v['policy'],config=v['config'],
            median_us=v['median_us'],p90_us=v['p90_us'],A_median_us=base['median_us'],
            change_vs_A_pct=100*(v['median_us']/base['median_us']-1),A_over_config=base['median_us']/v['median_us'],
            change_vs_B_pct=100*(v['median_us']/b['median_us']-1)))
        sample_stats.append(dict(series=name,M=v['M'],K=v['K'],N=v['N'],policy=v['policy'],config=v['config'],
            median_us=v['median_us'],min_us=v['min_us'],max_us=v['max_us'],p90_us=v['p90_us'],
            # Twelve significant digits keep derived CSVs stable across the
            # local Python3.10 and ARC Python3.11 statistics implementations.
            sample_variance_us2=float(format(statistics.variance(v['samples_us']),'.12g')),
            sample_stdev_us=float(format(statistics.stdev(v['samples_us']),'.12g'))))
write_csv('comparisons.csv',comparison)
write_csv('sample_statistics.csv',sample_stats)
repeat = []
for stem in ('series','fallback'):
    first,second = series[stem+'1'],series[stem+'2']
    for x in first:
        y = next(v for v in second if (v['M'],v['policy'],v['config']) == (x['M'],x['policy'],x['config']))
        assert x['operand_sha256'] == y['operand_sha256']
        assert x['order']+y['order'] == 2
        repeat.append(dict(series=stem,M=x['M'],policy=x['policy'],config=x['config'],
            first_us=x['median_us'],second_us=y['median_us'],change_pct=100*(y['median_us']/x['median_us']-1)))
write_csv('repeat_variability.csv',repeat)
criteria = {}
for config in 'BC':
    primary = [v for v in comparison if v['series'].startswith('series') and v['config'] == config]
    fallback = [v for v in comparison if v['series'].startswith('fallback') and v['config'] == config]
    nf = all(v['change_vs_A_pct'] <= -5 for v in primary if v['policy'] == 'no_flush')
    sf = all(v['change_vs_A_pct'] <= 5 for v in primary if v['policy'] == 'store_flush')
    reproducible = any(all(v['change_vs_A_pct'] > 5 for v in fallback if v['M'] == m and v['policy'] == policy) for m,policy in itertools.product((16,64),policies))
    criteria[config] = dict(no_flush_5pct_both_M_both_series=nf,store_flush_regression_at_most_5pct=sf,
        fallback_no_reproducible_5pct_regression=not reproducible,meets_latency_criteria=nf and sf and not reproducible)

profiles = []
units = {'byte':1,'Kbyte':1000,'Mbyte':1000000,'Gbyte':1000000000,'ns':.001,'us':1,'ms':1000}
for config in 'ABC':
    assert (r/f'profile-{config}.exit').read_text().strip() == '0'
    assert body in (r/f'profile-{config}.log').read_text()
    checked = [json.loads(x) for x in (r/f'profile-{config}.jsonl').read_text().splitlines()]
    assert len(checked) == 1 and checked[0]['num_bad'] == 0 and checked[0]['checked'] == 32768
    assert checked[0]['config'] == config and checked[0]['warmup'] == 10 and checked[0]['iters'] == 0
    assert checked[0]['graph_check'] and all(checked[0][k] for k in ('codes_match','scales_match','reconstruction_match','workspace_guards','lock_reset'))
    values = list(csv.reader((r/f'ncu/{config}_M8_reused_kernel_replay_metrics.csv').open()))
    assert len(values) == 3 and len(values[0]) == len(values[1]) == len(values[2])
    unit = dict(zip(values[0],values[1])); data = dict(zip(values[0],values[2]))
    mangled = list(csv.DictReader((r/f'ncu/{config}_mangled.csv').open()))
    assert len(mangled) == 2 and mangled[1]['Kernel Name'].startswith('_ZN11e005_marlin6MarlinI')
    # The default demangled raw exporter strips namespaces. Preserve a second
    # mangled-name export from the same report to verify the actual local symbol.
    assert 'Marlin<' in data['Kernel Name'] and ', 256, 1, 8, 8, 1, 4, 8, 0>' in data['Kernel Name'] and data['CC'] == '12.0'
    out = dict(config=config,M=8,K=4096,N=4096,kernel=data['Kernel Name'])
    for metric in (r/'profile-metrics.txt').read_text().splitlines()+['launch__occupancy_limit_shared_mem','launch__occupancy_limit_registers','profiler__replayer_passes']:
        value = float(data[metric].replace(',',''))
        assert math.isfinite(value), metric
        out[metric] = value*units.get(unit[metric].split('/')[0],1)
    assert out['launch__grid_size'] == (72 if config == 'C' else 36)
    assert out['launch__registers_per_thread'] == 126 and out['launch__block_size'] == 256
    assert out['launch__shared_mem_per_block_dynamic'] == (101376 if config == 'A' else 49664)
    profiles.append(out)
assert len({v['kernel'] for v in profiles}) == 1
write_csv('profile_summary.csv',profiles)

assert (r/'identity.exit').read_text().strip() == (r/'smoke.exit').read_text().strip() == '0'
assert (r/'identity-final.exit').read_text().strip() == '0'
smoke = [json.loads(x) for x in (r/'smoke.jsonl').read_text().splitlines()]
assert len(smoke) == 27 and {(v['M'],v['pattern'],v['config']) for v in smoke} == set(itertools.product((1,5,8),('mixed','zero_a','zero_w'),'ABC'))
assert all(v['num_bad'] == 0 and v['checked'] == v['M']*v['N'] and v['graph_check'] for v in smoke)
assert (r/'project.exit').read_text().strip() == '127' and (r/'project-retry.exit').read_text().strip() == '0'
assert (r/'project-retry.log').read_text().count('ERROR SUMMARY: 0 errors') == 6
assert (r/'dependency-diff.txt').read_text() == ''
assert (r/'e006-handoff-check.log').read_text().count(': OK') == 113
assert (r/'e006-handoff-final-check.log').read_text().count(': OK') == 113
assert (r/'e006-binaries-check.log').read_text().count(': OK') == 4
assert (r/'e006-binaries-final-check.log').read_text().count(': OK') == 4
for name in ('measurement-input-check.log','profile-input-check.log','final-input-check.log'):
    assert (r/name).read_text().count(': OK') == 5
for name in ('measurement-tooling.sha256','profile-tooling.sha256'):
    for line in (r/name).read_text().splitlines():
        expected,path = line.split(maxsplit=1)
        assert hashlib.sha256((repo/path).read_bytes()).hexdigest() == expected, path
summary = dict(status='accepted' if criteria['C']['meets_latency_criteria'] else 'rejected under predeclared performance criteria',
    baseline='corrected A, same E006 library for A/B/C',body_sha256=body,criteria=criteria,
    timing_rows={k:len(v) for k,v in series.items()},timed_samples=sum(len(v['samples_us']) for data in series.values() for v in data),
    smoke_rows=27,profiles=3,E006_reused_numerical_rows=135,E006_reused_sanitizer_rows=162,
    max_abs=max(v['max_abs'] for data in series.values() for v in data),max_rel_l2=max(v['rel_l2'] for data in series.values() for v in data),
    next_task='stop after E005-R1, no new optimization authorized')
(r/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
