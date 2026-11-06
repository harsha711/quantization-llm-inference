"""Recompute timing ratios and normalized profiler summaries from saved raw CSVs."""
import csv
from pathlib import Path

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / 'timings.csv').open()))
expected = {(str(m), str(flush), k) for m in (1, 8, 32, 128) for flush in (0, 1)
            for k in ('naive', 'inblock', 'cublas', 'cublas_dq', 'dequant', 'unfused')}
assert len(rows) == len(expected) == 48
assert {(r['M'], r['flush_l2'], r['kernel']) for r in rows} == expected
assert all(r['pass'] == '1' and r['w_hat_mismatches'] == '0' and r['warmup'] == '10'
           and int(r['iters']) >= 100 and r['seed'] == '1' and r['N'] == r['K'] == '4096'
           and r['group'] == '128' for r in rows)
with (root / 'speedups.csv').open('w') as out:
    writer = csv.writer(out)
    writer.writerow(['cache_policy', 'M', 'naive_median_us', 'naive_p90_us',
                     'inblock_median_us', 'inblock_p90_us', 'speedup_over_naive',
                     'inblock_over_naive_latency', 'cublas_median_us',
                     'cublas_dq_median_us', 'unfused_median_us'])
    for flush in ('1', '0'):
        for m in ('1', '8', '32', '128'):
            d = {r['kernel']: r for r in rows if r['M'] == m and r['flush_l2'] == flush}
            naive, candidate = float(d['naive']['median_us']), float(d['inblock']['median_us'])
            writer.writerow(['store_flush' if flush == '1' else 'no_flush', m,
                             naive, d['naive']['p90_us'], candidate, d['inblock']['p90_us'],
                             f'{naive/candidate:.6f}', f'{candidate/naive:.6f}',
                             d['cublas']['median_us'], d['cublas_dq']['median_us'],
                             d['unfused']['median_us']])
with (root / 'profile_summary.csv').open('w') as out:
    writer = csv.writer(out)
    writer.writerow(['profile', 'metric', 'value', 'unit'])
    for file in sorted((root / 'ncu').glob('*metrics.csv')):
        raw = list(csv.reader(file.open()))
        assert len(raw) == 3, file
        units, data = dict(zip(raw[0], raw[1])), dict(zip(raw[0], raw[2]))
        assert data['Kernel Name'] and data['CC'] == '12.0'
        for name in raw[0]:
            if name.startswith(('launch__', 'sm__', 'smsp__', 'dram__', 'lts__',
                                'gpu__time', 'gpu__compute')):
                assert data[name] not in ('n/a', 'N/A'), (file, name)
                writer.writerow([file.stem, name, data[name], units[name]])
print('PASS: all 48 measurements validated. Recomputed speedups.csv and profile_summary.csv.')
