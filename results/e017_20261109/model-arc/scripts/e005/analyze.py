#!/usr/bin/env python3
"""Assert the E005 stopped-gate evidence. Never turn a failed racecheck into acceptance."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path
import safety


def rows(root,name,count):
    data=[json.loads(x) for x in (root/(name+'.jsonl')).read_text().splitlines()]
    assert len(data)==count,(name,len(data),count)
    assert len({(x['config'],x['M'],x['K'],x['N'],x['pattern'],x['policy']) for x in data})==count,name
    for r in data:
        assert r['num_bad']==0 and r['checked']==r['M']*r['N']
        assert all(r[x] for x in ('codes_match','scales_match','reconstruction_match','lock_reset','workspace_guards'))
        assert r['group']==128 and r['iters']==0 and 'samples_us' not in r
        v=r['resources']
        if v and r['K']>=4096 and r['M']<=8:
            assert (v['grid'],v['dynamic_shared'],v['threads'],v['registers'])==({'A':(36,101376,256,126),'B':(36,49664,256,126),'C':(72,49664,256,126)}[r['config']])
        if v and r['M'] in (16,64):
            assert v['candidate']==0 and v['grid']==36 and v['dynamic_shared']==101376
    return data


def main(root):
    assert not (root/'VALIDATED').exists()
    assert not any((root/f'{name}.jsonl').exists() for name in ('series1','series2','heldout','wheel_diagnostic','fallback1','fallback2'))
    data={name:rows(root,name,count) for name,count in (
        ('smoke',8),('smoke_candidate',18),('validation_small',180),('validation_main',126),
        ('sanitizer_memcheck',18),('sanitizer_racecheck',18),('sanitizer_synccheck',18),
        ('sanitizer_memcheck_rect',6),('race_diagnostic_A',1),('race_diagnostic_wheel',1),('wheel_selection',2))}
    assert {r['M'] for r in data['validation_main']}=={1,2,4,5,8,16,64}
    assert {(r['K'],r['N']) for r in data['validation_main']}=={(4096,4096),(4096,11008),(11008,4096)}
    assert {r['pattern'] for r in data['validation_small']}=={'mixed','zero_a','zero_w'}
    assert all(r['repeated_launches']==5 and r['graph_check'] for name in ('validation_small','validation_main') for r in data[name])
    for name in ('sanitizer_memcheck','sanitizer_synccheck','sanitizer_memcheck_rect'):
        log=(root/(name+'.log')).read_text()
        assert 'ERROR SUMMARY: 0 errors' in log,name
        assert 'COMPUTE-SANITIZER' in log and 'kns=e005_marlin' in log
    race=(root/'sanitizer_racecheck.log').read_text()
    assert 'RACECHECK SUMMARY: 36 hazards displayed (36 errors, 0 warnings)' in race
    assert 'marlin_template.h:843' in race and 'marlin_template.h:1237' in race
    for name in ('A','wheel'):
        assert (root/f'race_diagnostic_{name}.exit').read_text().strip()=='99'
        log=(root/f'race_diagnostic_{name}.log').read_text()
        assert 'RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)' in log
        assert ('void e005_marlin::Marlin<' if name=='A' else 'void marlin::Marlin<') in log
    assert 'KERNEL wheel' in (root/'wheel_selection.log').read_text()
    assert "['void marlin::Marlin<" in (root/'wheel_selection.log').read_text()
    assert 'libe005.so' not in '\n'.join(x for x in (root/'wheel_selection.log').read_text().splitlines() if '/' in x and 'ENV ' not in x and 'Launch cmd =' not in x), 'local library mapped into isolated wheel process'
    attrs=(root/'attributes.log').read_text();attrs=json.loads(attrs[attrs.index('[\n'):])
    for m in (1,8):
        v=[r for r in attrs if r['M']==m]
        assert len({r['kernel_address'] for r in v})==1
        assert [(r['grid'],r['dynamic_shared'],r['max_blocks_per_sm']) for r in v]==[(36,101376,1),(36,49664,2),(72,49664,2)]
    for m in (16,64):
        v=[{k:v for k,v in r.items() if k!='config'} for r in attrs if r['M']==m]
        assert v[0]==v[1]==v[2]
    assert json.loads((root/'safety-model.json').read_text())==safety.check()
    # No device-body changes are hidden in the generated source patch.
    patch=(root/'e005-host.patch').read_text()
    assert 'marlin_template.h' not in patch and 'marlin_mma.h' not in patch
    with tarfile.open(root/'pinned-source.tar.gz') as t:
        with tarfile.open(root.parent/'e004_20261106/marlin-source.tar.gz') as old:
            for header in ('marlin_template.h','marlin.cuh','marlin_dtypes.cuh','dequant.h','marlin_mma.h','kernel.h'):
                path='csrc/quantization/marlin/'+header
                assert t.extractfile('pinned/'+path).read()==old.extractfile(path).read()
    env=[]
    for name in data:
        log=(root/(name+'.log')).read_text()
        line=next(x for x in log.splitlines() if x.startswith('ENV '))
        env.append(json.loads(line[4:]))
    assert len({x['local_sha256'] for x in env})==1
    assert all(x['wheel_sha256']=='e634b80e72078e964f8bd2ac541d66f2a900920773cf993544609612190bd9a9' for x in env)
    assert (root/'project-check.log').read_text().count('ERROR SUMMARY: 0 errors')==6
    final=dict(status='no-go at Gate 3',performance='not measured',
               criterion='all selected sanitizer checks pass',criterion_met=False,
               row_counts={k:len(v) for k,v in data.items()},
               max_main_rel_l2=max(r['rel_l2'] for r in data['validation_main']),
               max_main_abs=max(r['max_abs'] for r in data['validation_main']),
               all_numerical_checks_pass=True,primary_series_released=False,
               heldout_seed2_released=False,nsight_profiles_released=False,
               racecheck_small_errors=36,racecheck_main_A_errors=2,racecheck_main_wheel_errors=2,
               local_library_sha256=env[0]['local_sha256'],
               source_archive_sha256=hashlib.sha256((root/'pinned-source.tar.gz').read_bytes()).hexdigest())
    (root/'analysis.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(final,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
