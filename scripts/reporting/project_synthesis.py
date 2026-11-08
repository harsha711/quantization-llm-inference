#!/usr/bin/env python3
"""Recompute matched comparisons and render four offline synthesis figures."""
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics

os.environ.setdefault('MPLCONFIGDIR', '/tmp/w4a16-synthesis-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/reporting/project_synthesis'
FIG = ROOT / 'docs/figures/project_synthesis'
SHAPES = [(4096, 11008), (11008, 4096)]
BODY = 'e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7'
INPUTS = {}


def read(path):
    data = (ROOT / path).read_bytes()
    INPUTS[path] = hashlib.sha256(data).hexdigest()
    return data.decode()


def csv_rows(path):
    return list(csv.DictReader(read(path).splitlines()))


def sample_median(row):
    samples = row['samples_us']
    assert len(samples) == row['iters'] == 100
    assert all(math.isfinite(x) and x > 0 for x in samples)
    median = statistics.median(samples)
    assert median == row['median_us'], 'saved median differs from samples'
    assert median / row['matrices'] == row['median_per_matrix_us']
    return median


def comparison(experiment, series, k, n, m, policy, count, baseline, candidate, a, b, source):
    assert a > 0 and b > 0
    return dict(experiment=experiment, series=series, K=k, N=n, M=m,
                policy=policy, matrices=count, baseline=baseline, candidate=candidate,
                baseline_us=a, candidate_us=b, speedup=a/b, gain_pct=100*(1-b/a),
                baseline_amortized_us=a/count, candidate_amortized_us=b/count,
                source=source)


def progression():
    specs = [
        ('E001', 1, 'inblock_c58_20261106_final', ['naive', 'inblock'], [1, 8, 32, 128]),
        ('E002', 1, 'e002_c58_20261106', ['inblock', 'inblock_specialized'], [1, 8, 32, 128]),
        ('E003', 1, 'e003_c58_20261106_attempt1', ['inblock_specialized', 'inblock_m2', 'inblock_m4'], [1, 5, 8, 32, 128]),
        ('E003', 2, 'e003_c58_20261106', ['inblock_specialized', 'inblock_m2', 'inblock_m4'], [1, 5, 8, 32, 128]),
    ]
    result = []
    for exp, series, directory, kernels, ms in specs:
        path = f'results/{directory}/timings.csv'
        rows = csv_rows(path)
        assert len(rows) == {'E001': 48, 'E002': 56, 'E003': 90}[exp]
        by_key = {(r['kernel'], int(r['M']), int(r['flush_l2'])): r for r in rows}
        assert len(by_key) == len(rows)
        assert all(r['pass'] == '1' and r['w_hat_mismatches'] == '0' for r in rows)
        for m, flush in itertools.product(ms, [0, 1]):
            baseline = by_key[kernels[0], m, flush]
            for name in kernels[1:]:
                candidate = by_key[name, m, flush]
                for field in ('gpu', 'sm', 'sms', 'driver', 'runtime', 'K', 'N', 'M', 'group', 'seed', 'flush_l2', 'warmup', 'iters'):
                    assert baseline[field] == candidate[field]
                assert (baseline['K'], baseline['N'], baseline['group'], baseline['seed'], baseline['warmup'], baseline['iters']) == ('4096', '4096', '128', '1', '10', '100')
                result.append(comparison(exp, series, 4096, 4096, m,
                    'store_flush' if flush else 'no_flush', 1, kernels[0], name,
                    float(baseline['median_us']), float(candidate['median_us']), path))
    return result


def cyclic(exp, directory, configs, ms, counts):
    result = []
    for series in [1, 2]:
        path = f'results/{directory}/series{series}.jsonl'
        rows = [json.loads(s) for s in read(path).splitlines()]
        keys = {(r['K'], r['N'], r['M'], r['matrices'], r['config']): r for r in rows}
        expected = {(k, n, m, count, config) for (k, n), m, count, config in itertools.product(SHAPES, ms, counts, configs)}
        assert set(keys) == expected and len(rows) == len(keys)
        for row in rows:
            sample_median(row)
            assert row['body_sha256'] == BODY
            assert (row['group'], row['seeds'], row['warmup']) == (128, [3, 4, 5, 6], 10)
            assert (row['method'], row['policy']) == ('complete_cycle_graph_events', 'no_artificial_flush')
            if exp != 'E008':
                assert row['series'] == series and row['cublas_math_mode'] == 16
            assert all(row[k] for k in ('codes_match', 'scales_match', 'reconstruction_match', 'lock_reset', 'workspace_guards'))
            for checks in [row['validation'], row['post_timing_validation']]:
                assert len(checks) == row['matrices']
                assert all(v['num_bad'] == 0 and v['checked'] == row['M']*row['N'] for v in checks)
        for (k, n), m, count in itertools.product(SHAPES, ms, counts):
            candidate = keys[k, n, m, count, configs[0]]
            for baseline_name in configs[1:]:
                baseline = keys[k, n, m, count, baseline_name]
                assert candidate['allocations'] == baseline['allocations'], 'unmatched operands/storage'
                result.append(comparison(exp, series, k, n, m, 'no_artificial_flush', count,
                    baseline_name, configs[0], sample_median(baseline), sample_median(candidate), path))
    # Independent cross-check against the retained analyzer's table. Never rewrite it.
    saved = csv_rows(f'results/{directory}/comparisons.csv')
    for row in result:
        matches = [r for r in saved if (int(r['K']), int(r['N']), int(r['series']), int(r.get('M', 4)), int(r.get('matrices', 4))) ==
                   (row['K'], row['N'], row['series'], row['M'], row['matrices'])]
        assert len(matches) == 1
        s = matches[0]
        a, b = {'E008': ('A_cycle_us', 'C_cycle_us'),
                'E009': (row['baseline']+'_cycle_us', 'marlin_cycle_us'),
                'E010': ('cublas_cycle_us', 'marlin_cycle_us')}[exp]
        assert math.isclose(float(s[a]), row['baseline_us'], rel_tol=1e-14)
        assert math.isclose(float(s[b]), row['candidate_us'], rel_tol=1e-14)
    return result


def save(fig, name, title, footer):
    fig.suptitle(title, fontsize=11, y=0.99)
    fig.text(0.5, 0.012, footer, ha='center', va='bottom', fontsize=7)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(FIG / (name+'.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(FIG / (name+'.png'), dpi=240)
    plt.close(fig)


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=0.6)
    ax.set_axisbelow(True)


def figures(data):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'lines.linewidth': 1.4, 'lines.markersize': 4})
    colors = ['#0072B2', '#D55E00', '#009E73']
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4))
    for col, exp in enumerate(['E001', 'E002', 'E003']):
        for i, policy in enumerate(['store_flush', 'no_flush']):
            ax = axes[i, col]
            group = [r for r in data if r['experiment'] == exp and r['policy'] == policy]
            ms = sorted({r['M'] for r in group})
            for j, candidate in enumerate(sorted({r['candidate'] for r in group})):
                for s in sorted({r['series'] for r in group}):
                    rows = sorted([r for r in group if r['candidate'] == candidate and r['series'] == s], key=lambda r:r['M'])
                    label = {'inblock': '8 K slices', 'inblock_specialized': 'Aligned specialization', 'inblock_m2': '2 rows', 'inblock_m4': '4 rows'}[candidate]
                    ax.plot(range(len(ms)), [r['speedup'] for r in rows], marker='o' if s == 1 else 's',
                            linestyle='-' if s == 1 else '--', color=colors[j],
                            markerfacecolor=colors[j] if s == 1 else 'white', label=f'{label}, S{s}')
            style(ax)
            ax.axhline(1, color='#555555', linewidth=0.8)
            ax.set_xticks(range(len(ms)), [str(m)+('*' if m == 5 else '') for m in ms])
            ax.set_xlabel('Batch M (* row tail)')
            ax.set_ylabel('Control / candidate median')
            control = {'E001':'naive', 'E002':'E001', 'E003':'E002'}[exp]
            ax.set_title(f'{exp}: control {control}\n'+policy.replace('_', ' '))
            ax.set_ylim(0.4 if exp == 'E003' else 0.65, 4.2 if exp == 'E001' else 1.9)
            ax.legend(fontsize=6, loc='upper right' if exp == 'E001' else 'lower right' if exp == 'E002' else 'best')
    save(fig, '01_kernel_progression', 'Matched changes to our CUDA kernels | RTX 5060 Ti, c58 | K=N=4096, G128, seed 1',
         'Higher is better. Separate experiment denominators, no multiplied speedup. E001/E002: one series. E003: both saved series.\n100 launch-event samples/point. Store flush writes 64 MiB outside timing and does not prove eviction. No confidence intervals.')

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7), sharey=True)
    for ax, (k, n) in zip(axes, SHAPES):
        for s in [1, 2]:
            rows = sorted([r for r in data if r['experiment']=='E008' and (r['K'],r['N'],r['series'])==(k,n,s)], key=lambda r:r['matrices'])
            ax.plot([1,2,4], [r['gain_pct'] for r in rows], marker='o' if s==1 else 's', linestyle='-' if s==1 else '--', color=colors[s-1], label=f'Series {s}')
        style(ax)
        ax.axhline(0, color='#555555', linewidth=0.8)
        ax.axhline(5, color='#777777', linestyle=':', label='5% gate')
        ax.set_xticks([1,2,4], ['1\n22.17', '2\n44.34', '4\n88.69'])
        ax.set_xlabel('Distinct matrices/cycle\nActive compressed weights + scales (MiB)')
        ax.set_title(f'K={k}, N={n}')
        ax.legend(fontsize=7)
    axes[0].set_ylabel('C lower cycle median vs A (%)')
    save(fig, '02_reuse_distance', 'E008 | RTX 5060 Ti, c10 | M=4, G128 | No artificial flush',
         '100 × (1 − C/A). Both reversed series shown. Negative means C is slower. 100 whole-cycle samples/point.\n22.17 MiB per matrix, measured L2 capacity 32 MiB. Logical footprint does not establish eviction.')

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7), sharey=True)
    names = [('marlin','Corrected Marlin A'), ('cublas_dq','Pre-materialized FP16 cuBLAS'), ('unfused','GPU dequant + cuBLAS')]
    for ax, (k, n) in zip(axes, SHAPES):
        for j, (name, label) in enumerate(names):
            for s in [1, 2]:
                group = [r for r in data if r['experiment']=='E009' and (r['K'],r['N'],r['series'])==(k,n,s) and r['baseline']==('cublas_dq' if name=='marlin' else name)]
                group.sort(key=lambda r:r['matrices'])
                field = 'candidate_amortized_us' if name=='marlin' else 'baseline_amortized_us'
                ax.plot([1,2,4], [r[field] for r in group], marker='o' if s==1 else 's', linestyle='-' if s==1 else '--', color=colors[j], markerfacecolor=colors[j] if s==1 else 'white', label=label if s==1 else None)
        style(ax)
        ax.set_xticks([1,2,4])
        ax.set_xlabel('Distinct matrices per complete cycle')
        ax.set_title(f'K={k}, N={n}')
        ax.set_ylim(0,625)
    axes[0].set_ylabel('Cycle median / matrix count (µs/GEMM)')
    axes[0].legend(fontsize=6.5, loc='center right', bbox_to_anchor=(1, 0.66))
    save(fig, '03_cyclic_baselines', 'E009 | RTX 5060 Ti, c4 | M=4, G128 | No artificial flush',
         'Lower is better. Filled circles/solid = series 1. Open squares/dashed = reversed series 2 (often overlap).\n100 complete-cycle samples/point. Per-GEMM values are derived amortization. All paths use W_hat, cuBLAS mode 16.')

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7), sharey=True)
    for ax, (k, n) in zip(axes, SHAPES):
        for s in [1,2]:
            rows = sorted([r for r in data if r['experiment']=='E010' and (r['K'],r['N'],r['series'])==(k,n,s)], key=lambda r:r['M'])
            ax.plot([r['M'] for r in rows], [r['speedup'] for r in rows], marker='o' if s==1 else 's', linestyle='-' if s==1 else '--', color=colors[s-1], label=f'Series {s}')
        style(ax)
        ax.set_xscale('log', base=2)
        ax.set_xticks([4,16,64,128], ['4','16','64','128'])
        ax.axhline(1, color='#555555', linewidth=0.8, label='Equal median')
        ax.axhline(1/0.95, color='#777777', linestyle=':', label='5% lower latency')
        ax.set_ylim(0.9,4.7)
        ax.set_xlabel('Batch M')
        ax.set_title(f'K={k}, N={n}')
        ax.legend(fontsize=6.5)
        endpoint = [r['speedup'] for r in data if r['experiment']=='E010' and (r['K'],r['N'],r['M'])==(k,n,128)]
        ax.text(0.04, 0.09, f'M128: {min(endpoint):.3f}–{max(endpoint):.3f}×', transform=ax.transAxes)
    axes[0].set_ylabel('FP16 cuBLAS / corrected Marlin median')
    save(fig, '04_batch_speedup', 'E010 | RTX 5060 Ti, c4 | Four distinct matrices/cycle, G128 | No artificial flush',
         'Higher is better. Fresh same-series cycle medians, 100 whole-cycle samples/point, cuBLAS mode 16.\nBoth reversed series shown. Guide lines connect tested points only. No confidence intervals or extrapolation above M128.')


def supporting_evidence():
    """Audit headline calculations outside the four plotted comparisons."""
    result = {}
    for exp, directory, a, b in [
        ('E005-R1', 'e005_r1_20261106', 'A_median_us', 'median_us'),
        ('E005-R2', 'e005_r2_c10_20261106', 'A_us', 'C_us'),
        ('E007', 'e007_20261106', 'control_us', 'candidate_us'),
    ]:
        path = f'results/{directory}/comparisons.csv'
        rows = csv_rows(path)
        if exp == 'E005-R1':
            rows = [r for r in rows if r['config']=='C' and r['series'] in ('series1','series2') and int(r['M']) in (1,8)]
        result[exp] = [dict(source=path, series=r['series'], K=int(r['K']), N=int(r['N']),
                            M=int(r.get('M',4)), policy=r['policy'], baseline_us=float(r[a]),
                            candidate_us=float(r[b]), gain_pct=100*(1-float(r[b])/float(r[a]))) for r in rows]
    profiles = csv_rows('results/e003_c58_20261106/profile_summary.csv')
    base = next(r for r in profiles if r['kernel']=='inblock_specialized' and r['M']=='128')
    result['E003_profile_changes'] = [dict(candidate=r['kernel'],
        instruction_reduction_pct=100*(1-float(r['executed_warp_instructions'])/float(base['executed_warp_instructions'])),
        l2_reduction_pct=100*(1-float(r['l2_bytes'])/float(base['l2_bytes'])))
        for r in profiles if r['M']=='128' and r['kernel'] in ('inblock_m2','inblock_m4')]
    profiles = json.loads(read('results/e011_c4_20261107/profile-analysis.json'))
    result['E011'] = []
    for k,n in SHAPES:
        endpoints = {r['M']:r for r in profiles if (r['K'],r['N'],r['config'])==(k,n,'marlin')}
        a,b = endpoints[4]['totals'], endpoints[128]['totals']
        result['E011'].append(dict(K=k,N=n,
            dram_read_increase_pct=100*(b['dram__bytes_op_read.sum']/a['dram__bytes_op_read.sum']-1),
            warp_instruction_ratio=b['smsp__inst_executed.sum']/a['smsp__inst_executed.sum'],
            tensor_instruction_ratio=b['sm__inst_executed_pipe_tensor.sum']/a['sm__inst_executed_pipe_tensor.sum'],
            endpoints=[dict(M=m,dram_read_MiB=r['totals']['dram__bytes_op_read.sum']/2**20,
                            warp_instructions=r['totals']['smsp__inst_executed.sum'],
                            tensor_instructions=r['totals']['sm__inst_executed_pipe_tensor.sum']) for m,r in endpoints.items()]))
    attribution = json.loads(read('results/e012_c4_20261107/attribution.json'))
    result['E012_instruction_increase'] = [dict(K=d['K'],N=d['N'],
        known_main_increase_share_pct=100*d['known_main_loop_increase']/d['total_instruction_increase'],
        conservative_main_lower_share_pct=100*d['main_loop_increase_bounds'][0]/d['total_instruction_increase']) for d in attribution['deltas']]
    result['E012_sampling'] = [dict(K=r['K'],N=r['N'],M=r['M'],series=r['series'],
        dropped_bytes=r['dropped_bytes'],source_coverage_pct=r['source_coverage_pct'],quality=r['quality'],
        unmapped_pc_samples=r['unmapped_pc_samples']) for r in attribution['rows'] if r['kind']=='sampling']
    assert len(result['E012_sampling']) == 8 and all(r['dropped_bytes']>0 and r['quality'] for r in result['E012_sampling'])
    return result


def self_test():
    row = dict(samples_us=list(range(1,101)), iters=100, median_us=50.5, matrices=2, median_per_matrix_us=25.25)
    assert sample_median(row) == 50.5
    for broken in [dict(row, median_us=51), dict(row, samples_us=[1]*99), dict(row, samples_us=[float('nan')]*100)]:
        try:
            sample_median(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupt samples accepted')
    r = comparison('test',1,1,1,1,'test',4,'a','b',200,100,'test')
    assert (r['speedup'],r['gain_pct'],r['candidate_amortized_us']) == (2,50,25)
    assert 1 < 102/100 < 1/0.95  # A positive speedup can fail the practical gate.


def main():
    self_test()
    manifest = OUT/'figure-inputs.sha256'
    recorded_inputs = None
    if manifest.exists():
        recorded_inputs = {path: sha for sha, path in (line.split('  ', 1) for line in manifest.read_text().splitlines())}
        for path, expected in recorded_inputs.items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, 'frozen input changed: '+path
    data = progression()
    data += cyclic('E008','e008_c10_20261106',['C','A'],[4],[1,2,4])
    data += cyclic('E009','e009_c4_20261107',['marlin','cublas_dq','unfused'],[4],[1,2,4])
    data += cyclic('E010','e010_c4_20261107',['marlin','cublas_dq'],[4,16,64,128],[4])
    assert len(data) == 108
    supporting = supporting_evidence()
    assert recorded_inputs is None or INPUTS == recorded_inputs, 'input set changed'
    OUT.mkdir(parents=True,exist_ok=True)
    FIG.mkdir(parents=True,exist_ok=True)
    with (OUT/'matched-comparisons.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)
    figures(data)
    (OUT/'supporting-evidence.json').write_text(json.dumps(supporting,indent=2)+'\n')
    for path, expected in INPUTS.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, 'input modified'
    manifest.write_text(''.join(f'{h}  {p}\n' for p,h in sorted(INPUTS.items())))
    summary = {exp: {'comparisons':len(rows), 'speedup_min':min(r['speedup'] for r in rows), 'speedup_max':max(r['speedup'] for r in rows)}
               for exp in ['E001','E002','E003','E008','E009','E010'] if (rows := [r for r in data if r['experiment']==exp])}
    (OUT/'figure-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'PASS: {len(data)} matched comparisons, {len(INPUTS)} unchanged inputs, four PDF/PNG figures')


if __name__ == '__main__':
    main()
