#!/usr/bin/env python3
"""Assert E004 series completeness and regenerate compact comparisons from samples."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SHAPES=[(4096,4096),(4096,11008),(11008,4096)]
MS=[1,2,4,8,16,32,64,128]
KERNELS=['marlin','naive','inblock_specialized','inblock_m2','inblock_m4','cublas','cublas_dq','unfused']
POLICIES=['store_flush','no_flush']


def key(r):return r['K'],r['N'],r['M'],r['policy'],r['kernel']


def load(path):
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    assert len({key(r) for r in rows})==len(rows),f'duplicate rows in {path}'
    for r in rows:
        assert r['pass_check'] and r['num_bad']==0 and r['checked']==r['M']*r['N']
        assert r['repack_codes_match'] and r['repack_scale_bits_match'] and r['full_control_match']
        assert r['reconstructed_elements']==r['K']*r['N']
        assert r['group']==128 and r['seed']==1 and r['cublas_math_mode']==16
        assert r['warmup']==10
        s=r['samples_us'];assert len(s)==r['iters'] and all(math.isfinite(x) and x>0 for x in s)
        assert statistics.median(s)==r['median_us']
        assert sorted(s)[math.floor(.9*(len(s)-1))]==r['p90_us']
        assert min(s)==r['min_us'] and max(s)==r['max_us']
    return rows


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(root):
    rows=load(root/'screen.jsonl')
    expected={(k,n,m,p,c) for k,n in SHAPES for m in MS for p in POLICIES for c in KERNELS}
    assert {key(r) for r in rows}==expected and len(rows)==384
    assert all(r['iters']==20 and r['method']=='graph_events' for r in rows)
    lookup={key(r):r for r in rows}
    table=[]
    for k,n in SHAPES:
        for m in MS:
            row={'K':k,'N':n,'M':m}
            for policy in POLICIES:
                d={c:lookup[k,n,m,policy,c]['median_us'] for c in KERNELS}
                fused=min(['naive','inblock_specialized','inblock_m2','inblock_m4'],key=d.get)
                row.update({f'{policy}_marlin_us':d['marlin'],f'{policy}_cublas_us':d['cublas'],
                            f'{policy}_cublas_dq_us':d['cublas_dq'],f'{policy}_best_retained':fused,
                            f'{policy}_best_retained_us':d[fused],f'{policy}_retained_over_marlin':d[fused]/d['marlin'],
                            f'{policy}_cublas_dq_over_marlin':d['cublas_dq']/d['marlin']})
            table.append(row)
    write_csv(root/'performance_map.csv',table)
    # Each logical weight set is identical across all comparators/M/policies.
    for k,n in SHAPES:
        signatures={json.dumps(r['operand_sha256'],sort_keys=True) for r in rows if (r['K'],r['N'])==(k,n)}
        assert len(signatures)==1
    summary=dict(screen_rows=len(rows),screen_complete=True,all_outputs_checked=True,
                 max_relative_l2=max(r['rel_l2'] for r in rows),max_absolute_error=max(r['max_abs'] for r in rows))
    # Repeated target sets are checked after the diagnostic selection is recorded.
    for name in ['repeat1','repeat2','eager_diagnostic','graph_diagnostic']:
        path=root/(name+'.jsonl')
        if not path.exists():continue
        other=load(path)
        assert all(r['iters']==100 for r in other)
        for r in other:
            assert key(r) in lookup
            base=lookup[key(r)]
            for operand in ['W','Q','S','W_hat']:
                assert base['operand_sha256'][operand]==r['operand_sha256'][operand]
        summary[name+'_rows']=len(other)
    if (root/'repeat1.jsonl').exists() and (root/'repeat2.jsonl').exists():
        a={key(r):r for r in load(root/'repeat1.jsonl')};b={key(r):r for r in load(root/'repeat2.jsonl')}
        assert a.keys()==b.keys()
        comps=[]
        for x in a:
            r,s=a[x],b[x]
            assert r['method']==s['method']=='graph_events'
            assert r['operand_sha256']==s['operand_sha256']
            comps.append(dict(K=x[0],N=x[1],M=x[2],policy=x[3],kernel=x[4],screen_us=lookup[x]['median_us'],repeat1_us=r['median_us'],repeat2_us=s['median_us'],repeat2_over_repeat1=s['median_us']/r['median_us']))
        write_csv(root/'repeat_comparison.csv',comps)
    profiles=[]
    factors={'byte':1,'byte/block':1,'Kbyte':1000,'Kbyte/block':1000,'Mbyte':1000000,'us':1,'inst':1,'register/thread':1,'block':1,'warp':1,'%':1,'pass':1,'SM':1}
    fields=['gpu__time_duration.sum','launch__grid_size','launch__block_size','launch__registers_per_thread',
            'launch__registers_per_thread_allocated','launch__shared_mem_per_block_dynamic',
            'launch__shared_mem_per_block_allocated','launch__occupancy_limit_shared_mem',
            'launch__occupancy_limit_registers','sm__warps_active.avg.pct_of_peak_sustained_active',
            'smsp__warps_eligible.avg.per_cycle_active','smsp__issue_active.avg.pct_of_peak_sustained_active',
            'sm__inst_executed_pipe_tensor.sum','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
            'smsp__inst_executed.sum','smsp__sass_thread_inst_executed_op_ffma_pred_on.sum',
            'smsp__sass_thread_inst_executed_op_integer_pred_on.sum',
            'smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed',
            'smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_elapsed',
            'smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_elapsed',
            'smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_elapsed',
            'smsp__warps_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_elapsed',
            'dram__throughput.avg.pct_of_peak_sustained_elapsed','dram__bytes_op_read.sum','dram__bytes_op_write.sum','lts__t_bytes.sum','profiler__replayer_passes']
    for m in [8,16,64]:
        path=root/'ncu'/f'marlin_M{m}_reused_kernel_replay_metrics.csv'
        if not path.exists():continue
        lines=list(csv.reader(path.open()))
        assert len(lines)==3 and len(lines[0])==len(lines[1])==len(lines[2])
        unit=dict(zip(lines[0],lines[1]));data=dict(zip(lines[0],lines[2]))
        assert 'Marlin<' in data['Kernel Name'] and data['CC']=='12.0'
        assert data['Grid Size']=='(36, 1, 1)' and data['Block Size']=='(256, 1, 1)'
        row=dict(M=m,K=4096,N=4096,kernel=data['Kernel Name'],cache_control='none',warmup=10,replay='kernel')
        for field in fields:
            v=float(data[field].replace(',',''));assert math.isfinite(v)
            row[field]=v*factors.get(unit[field],1)
        assert row['launch__shared_mem_per_block_dynamic']==101376
        assert row['smsp__sass_thread_inst_executed_op_ffma_pred_on.sum']==0
        assert row['sm__inst_executed_pipe_tensor.sum']==m*8192
        profiles.append(row)
    if profiles:
        assert len(profiles)==3
        write_csv(root/'profile_summary.csv',profiles)
        summary['profiles']=len(profiles)
    # Validate the source subset against its pinned provenance, independent of current build.
    import tarfile
    provenance=json.loads((root/'provenance.json').read_text())
    with tarfile.open(root/'marlin-source.tar.gz') as archive:
        for file,digest in provenance['files'].items():
            assert hashlib.sha256(archive.extractfile(file).read()).hexdigest()==digest,file
    binary_hashes=[]
    repo=Path(__file__).resolve().parents[2]
    for manifest in ['screen-source.sha256','repeat-source.sha256']:
        for line in (root/manifest).read_text().splitlines():
            digest,file=line.split(None,1)
            file=file.strip()
            if file.endswith('/libe004.so'):
                binary_hashes.append(digest)
            elif file.startswith(('src/','include/')) or file in ['scripts/e004/bench.py','scripts/e004/bridge.cu','scripts/e004/build_bridge.sh','scripts/e004/run_arc.sh']:
                assert hashlib.sha256((repo/file).read_bytes()).hexdigest()==digest,file
    assert len(binary_hashes)==2 and len(set(binary_hashes))==1
    assert provenance['operator_library_sha256'] in (root/'torch-smoke.log').read_text()
    summary['bridge_binary_sha256']=binary_hashes[0]
    summary['operator_library_sha256']=provenance['operator_library_sha256']
    assert 'ERROR SUMMARY: 0 errors' in (root/'project-check.log').read_text()
    assert (root/'project-check.log').read_text().count('ERROR SUMMARY: 0 errors')==6
    summary['source_archive_verified']=True
    (root/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args();main(a.results)
