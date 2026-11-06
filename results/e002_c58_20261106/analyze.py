"""Validate E002 evidence and recompute comparisons without changing raw records."""
import csv
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / 'timings.csv').open()))
kernels = ('naive', 'inblock', 'inblock_specialized', 'cublas', 'cublas_dq', 'dequant', 'unfused')
expected = {(str(m), str(flush), k) for m in (1, 8, 32, 128) for flush in (0, 1) for k in kernels}
assert len(rows) == len(expected) == 56
assert {(r['M'], r['flush_l2'], r['kernel']) for r in rows} == expected
assert all(r['pass'] == '1' and r['w_hat_mismatches'] == '0' and r['warmup'] == '10'
           and int(r['iters']) >= 100 and r['seed'] == '1' and r['N'] == r['K'] == '4096'
           and r['group'] == '128' for r in rows)
comparisons = []
for flush in ('1', '0'):
    for m in ('1', '8', '32', '128'):
        data = {r['kernel']: r for r in rows if r['M'] == m and r['flush_l2'] == flush}
        old, new = data['inblock'], data['inblock_specialized']
        a, b = float(old['median_us']), float(new['median_us'])
        comparisons.append(dict(cache_policy='store_flush' if flush == '1' else 'no_flush', M=int(m),
            e001_median_us=a, e001_p90_us=float(old['p90_us']), e002_median_us=b,
            e002_p90_us=float(new['p90_us']), speedup_over_e001=a/b,
            speedup_over_naive=float(data['naive']['median_us'])/b,
            cublas_median_us=float(data['cublas']['median_us']),
            cublas_dq_median_us=float(data['cublas_dq']['median_us']),
            unfused_median_us=float(data['unfused']['median_us'])))
with (root / 'comparisons.csv').open('w') as out:
    writer = csv.DictWriter(out, fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)
log = (root / 'timings.log').read_text()
bit_checks = re.findall(r'inblock_specialized vs inblock: (\d+) / (\d+) FP16 outputs differ', log)
assert len(bit_checks) == 8 and all(int(bad) == 0 for bad, _ in bit_checks)
assert sorted(int(n) for _, n in bit_checks) == sorted([m*4096 for m in (1,8,32,128)]*2)
check_log = (root / 'check.log').read_text()
assert check_log.count('ERROR SUMMARY: 0 errors') == 6
assert 'COMPUTE-SANITIZER' in check_log
criteria = dict(all_56_rows_pass=True, bitwise_all_outputs_match=True, selected_sanitizers_pass=True,
    large_M_speedup_at_least_1_05=all(r['speedup_over_e001'] >= 1.05 for r in comparisons if r['M'] in (32,128)),
    M1_latency_no_more_than_1_05_baseline=all(r['e002_median_us'] <= 1.05*r['e001_median_us'] for r in comparisons if r['M'] == 1))
criteria['predeclared_criterion_met_in_this_series'] = all(criteria.values())
(root / 'criteria.json').write_text(json.dumps(criteria, indent=2) + '\n')
with (root / 'profile_summary.csv').open('w') as out:
    writer = csv.writer(out)
    writer.writerow(['profile', 'metric', 'value', 'unit'])
    files = sorted((root / 'ncu').glob('*metrics.csv'))
    assert len(files) == 6
    for file in files:
        raw = list(csv.reader(file.open()))
        assert len(raw) == 3
        units, data = dict(zip(raw[0], raw[1])), dict(zip(raw[0], raw[2]))
        for name in raw[0]:
            if name.startswith(('launch__', 'sm__', 'smsp__', 'dram__', 'lts__', 'gpu__time', 'gpu__compute')):
                assert data[name] not in ('n/a', 'N/A'), (file, name)
                writer.writerow([file.stem, name, data[name], units[name]])
# Verify source hashes where the checkout is present. Build hashes remain provenance records.
repo = root.parent.parent
for line in (root / 'source-build.sha256').read_text().splitlines():
    expected_hash, name = line.split('  ', 1)
    if not name.startswith('build/'):
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected_hash, name
print(json.dumps(criteria, indent=2))
for row in comparisons:
    print(row['cache_policy'], row['M'], row['e001_median_us'], row['e002_median_us'],
          f"{row['speedup_over_e001']:.3f}x over E001")
