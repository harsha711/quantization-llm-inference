"""Recompute E003 comparisons and assert complete timing, validation, and profile evidence."""
import csv
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
shapes = (1, 8, 32, 128, 5)  # M=5 is the supplemental incomplete-row-tile case.
kernels = ('naive', 'inblock', 'inblock_specialized', 'inblock_m2', 'inblock_m4',
           'cublas', 'cublas_dq', 'dequant', 'unfused')
rows = list(csv.DictReader((root / 'timings.csv').open()))
expected = {(str(m), str(f), k) for m in shapes for f in (0, 1) for k in kernels}
assert len(rows) == len(expected) == 90
assert {(r['M'], r['flush_l2'], r['kernel']) for r in rows} == expected
assert all(r['pass'] == '1' and r['w_hat_mismatches'] == '0' and r['warmup'] == '10'
           and r['iters'] == '100' and r['seed'] == '1' and r['N'] == r['K'] == '4096'
           and r['group'] == '128' for r in rows)
log = (root / 'timings.log').read_text()
for name, baseline in [('inblock_specialized', 'inblock'), ('inblock_m2', 'inblock_specialized'),
                       ('inblock_m4', 'inblock_specialized')]:
    bits = re.findall(rf'{name} vs {baseline}: (\d+) / (\d+) FP16 outputs differ', log)
    assert len(bits) == 10 and all(int(b) == 0 for b, _ in bits), (name, bits)
    assert sorted(int(n) for _, n in bits) == sorted([m*4096 for m in shapes]*2)
check = (root / 'check.log').read_text()
assert check.count('ERROR SUMMARY: 0 errors') == 6
assert not re.search(r'vs inblock(?:_specialized)?: [1-9]\d* /', check)
for m in (1, 2, 3, 4, 5, 7):
    for k in (136, 137):
        assert f'shape: M={m} N=33 K={k} group=128 seed=7' in check
comparisons = []
for f in ('1', '0'):
    for m in shapes:
        data = {r['kernel']: r for r in rows if r['M'] == str(m) and r['flush_l2'] == f}
        for kernel in ('inblock_m2', 'inblock_m4'):
            candidate, control, naive = [data[k] for k in (kernel, 'inblock_specialized', 'naive')]
            a, b, n = [float(r['median_us']) for r in (control, candidate, naive)]
            comparisons.append(dict(cache_policy='store_flush' if f == '1' else 'no_flush',
                M=m, kernel=kernel, e002_median_us=a, e002_p90_us=float(control['p90_us']),
                naive_median_us=n, naive_p90_us=float(naive['p90_us']),
                candidate_median_us=b, candidate_p90_us=float(candidate['p90_us']),
                speedup_over_e002=a/b, speedup_over_naive=n/b))
with (root / 'comparisons.csv').open('w') as out:
    writer = csv.DictWriter(out, fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)
criteria = dict(all_90_rows_pass=True, all_output_bits_match=True, selected_sanitizers_pass=True)
for k in ('inblock_m2', 'inblock_m4'):
    criteria[k + '_large_M_criterion_met'] = all(r['speedup_over_e002'] >= 1.05
        for r in comparisons if r['kernel'] == k and r['M'] in (32, 128))
(root / 'criteria.json').write_text(json.dumps(criteria, indent=2) + '\n')
# Nsight exports decimal-scaled byte/time units. Normalize requested evidence explicitly.
def numeric(value, unit):
    scales = {'byte': 1, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9,
              'nsecond': 1e-9, 'usecond': 1e-6, 'msecond': 1e-3, 'second': 1,
              'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1,
              'byte/block': 1, 'Kbyte/block': 1e3, 'Mbyte/block': 1e6}
    return float(value.replace(',', '')) * scales.get(unit, 1)
profiles = []
files = sorted((root / 'ncu').glob('*metrics.csv'))
assert len(files) == 20
metrics = dict(registers_per_thread='launch__registers_per_thread',
    static_shared_bytes='launch__shared_mem_per_block_static', grid_blocks='launch__grid_size',
    executed_warp_instructions='smsp__inst_executed.sum', l2_bytes='lts__t_bytes.sum',
    dram_read_bytes='dram__bytes_op_read.sum', profiler_seconds='gpu__time_duration.sum',
    active_warps_pct='sm__warps_active.avg.pct_of_peak_sustained_active',
    eligible_warps='smsp__warps_eligible.avg.per_cycle_active',
    issue_active_pct='smsp__issue_active.avg.pct_of_peak_sustained_active')
for file in files:
    raw = list(csv.reader(file.open()))
    assert len(raw) == 3, (file, len(raw))
    unit, data = dict(zip(raw[0], raw[1])), dict(zip(raw[0], raw[2]))
    kernel, m = re.match(r'(.*)_M(\d+)_ncu_invalidated_metrics', file.stem).groups()
    expected_name = r'w4a16_inblock_rows_kernel<(?:\(int\))?' + kernel[-1] + ',' if kernel in ('inblock_m2', 'inblock_m4') else 'w4a16_' + kernel + '_kernel'
    assert re.search(expected_name, data['Kernel Name']), (file, data['Kernel Name'])
    record = dict(kernel=kernel, M=int(m))
    for short, full in metrics.items():
        assert data[full] not in ('n/a', 'N/A', ''), (file, full)
        record[short] = numeric(data[full], unit[full])
    tile = int(kernel[-1]) if kernel in ('inblock_m2', 'inblock_m4') else 1
    blocks = (4096 // (256 if kernel == 'naive' else 32)) * ((int(m) + tile - 1) // tile)
    assert record['grid_blocks'] == blocks, (file, record['grid_blocks'], blocks)
    assert record['static_shared_bytes'] == (0 if kernel == 'naive' else tile * 1024)
    profiles.append(record)
assert {(r['kernel'], r['M']) for r in profiles} == {(k, m) for k in ('naive', 'inblock_specialized', 'inblock_m2', 'inblock_m4') for m in shapes}
with (root / 'profile_summary.csv').open('w') as out:
    writer = csv.DictWriter(out, fieldnames=list(profiles[0]))
    writer.writeheader()
    writer.writerows(profiles)
repo = root.parent.parent
for line in (root / 'source-build.sha256').read_text().splitlines():
    digest, path = line.split('  ', 1)
    if not path.startswith('build/'):
        assert hashlib.sha256((repo / path).read_bytes()).hexdigest() == digest, path
print(json.dumps(criteria, indent=2))
for row in comparisons:
    print(row['cache_policy'], row['M'], row['kernel'], row['candidate_median_us'],
          f"{row['speedup_over_e002']:.3f}x E002", f"{row['speedup_over_naive']:.3f}x naive")

# The first timing series completed before a profiler-filter failure. It is a separate repeat.
first = root.parent / 'e003_c58_20261106_attempt1'
assert (first / 'check.sha256').read_text() == (root / 'check.sha256').read_text()
prior = list(csv.DictReader((first / 'timings.csv').open()))
assert len(prior) == 90 and all(r['pass'] == '1' and r['w_hat_mismatches'] == '0' for r in prior)
repeat = []
for row in rows:
    old = next(r for r in prior if all(r[k] == row[k] for k in ('kernel', 'M', 'flush_l2')))
    assert all(old[k] == row[k] for k in ('gpu', 'sm', 'sms', 'driver', 'runtime', 'seed', 'N', 'K', 'group', 'warmup', 'iters', 'check_rows', 'l2_mib', 'flush_mib', 'logical_bytes')), (row['kernel'], row['M'])
    repeat.append(dict(kernel=row['kernel'], M=row['M'], flush_l2=row['flush_l2'],
        attempt1_median_us=old['median_us'], final_median_us=row['median_us'],
        final_over_attempt1=float(row['median_us'])/float(old['median_us'])))
with (root / 'repeat_comparison.csv').open('w') as out:
    writer = csv.DictWriter(out, fieldnames=list(repeat[0]))
    writer.writeheader()
    writer.writerows(repeat)
