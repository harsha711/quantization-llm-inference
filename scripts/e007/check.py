#!/usr/bin/env python3
"""Release E007 stages only after complete source/SASS and numerical evidence."""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re
import tarfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path)
p.add_argument('--stage',choices=['build','smoke','validation'],required=True)
p.add_argument('--e006',type=Path,default=Path(__file__).resolve().parents[2]/'results/e006_20261106')
a=p.parse_args();r=a.root

def functions(path):
    return {b.splitlines()[0].strip():b for b in gzip.open(path,'rt').read().split('Function : ')[1:]}

control=functions(r/'control/sass.txt.gz');candidate=functions(r/'candidate/sass.txt.gz')
assert control==functions(a.e006/'patched/sass.txt.gz') and control.keys()==candidate.keys() and len(control)==16
changed=[k for k in control if control[k]!=candidate[k]]
assert len(changed)==1 and 'ELi256ELi1ELi8ELi8ELb1ELi4ELi8ELb0E' in changed[0],changed
counts={}
for variant,functions_ in [('control',control),('candidate',candidate)]:
    body=functions_[changed[0]]
    ops=re.findall(r'/\*[0-9a-f]+\*/\s+([^;]+);',body)
    counts[variant]=dict(instructions=len(ops),barriers=body.count('BAR.SYNC'),
        shared_loads=sum(bool(re.match(r'(?:@\S+\s+)?LDSM?(?:\.|\s)',x)) for x in ops),
        unpredicated_shared_loads=sum(x.startswith(('LDS','LDSM')) for x in ops),
        predicated_shared_loads=sum(bool(re.match(r'@\S+\s+LDSM?(?:\.|\s)',x)) for x in ops),
        always_false_shared_loads=sum(bool(re.match(r'@!PT\s+LDSM?(?:\.|\s)',x)) for x in ops),
        sass_sha256=hashlib.sha256(body.encode()).hexdigest())
with tarfile.open(r/'matched-source.tar.gz') as t:
    files={m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
for name,content in files.items():
    if name.startswith('external/control/'):
        other=files[name.replace('external/control/','external/candidate/',1)]
        if name.endswith('/marlin_template.h'):
            assert content!=other
            assert content.count(b'__syncthreads();')==other.count(b'__syncthreads();')
            assert b'k != b_sh_wr_iters - 1 || slice_iters > 1' in other
            assert b'Join all shared readers and drained copy groups before overlay reuse.' in other
        else:assert content==other,name
with tarfile.open(a.e006/'matched-source.tar.gz') as t:
    for name,content in files.items():
        if name.startswith('external/control/'):
            assert content==t.extractfile(name.replace('external/control/','external/patched/',1)).read(),name
hashes={variant:(r/variant/'build-source.sha256').read_text().split()[0] for variant in ('control','candidate')}
for variant in hashes:
    path=r/variant/'libe005.so'
    if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[variant]
build=dict(control_sass_matches_e006=True,changed_specializations=changed,counts=counts,binary_sha256=hashes)
(r/'build-check.json').write_text(json.dumps(build,indent=2)+'\n')
if a.stage=='build':
    print(json.dumps(build,indent=2));raise SystemExit(0)

def load(label,expected,repeats):
    assert (r/(label+'.exit')).read_text().strip()=='0',label
    rows=[json.loads(x) for x in (r/(label+'.jsonl')).read_text().splitlines()]
    assert len(rows)==len(expected) and {(v['M'],v['K'],v['N'],v['pattern'],v['config']) for v in rows}==expected,label
    variant='control' if label.startswith('control') else 'candidate'
    assert hashes[variant] in (r/(label+'.log')).read_text()
    for v in rows:
        assert v['num_bad']==0 and v['checked']==v['M']*v['N'] and v['warmup']==v['iters']==0
        assert v['repeated_launches']==repeats and v['graph_check'] and len(v['output_sha256'])==64
        assert all(v[k] for k in ('codes_match','scales_match','reconstruction_match','workspace_guards','lock_reset'))
        attr=v['resources']
        if v['M']<=8 and v['N']>=2048:
            assert (attr['grid'],attr['dynamic_shared'])=={'A':(36,101376),'B':(36,49664),'C':(72,49664)}[v['config']]
        else:assert attr['grid']==36 and attr['dynamic_shared']==101376
    return {(v['M'],v['K'],v['N'],v['pattern'],v['config']):v for v in rows}

shapes=((4096,4096),(4096,11008),(11008,4096))
if a.stage=='smoke':
    expected={(m,256,2048,p,c) for m,p,c in itertools.product((1,5,8),('mixed','zero_a','zero_w'),'ABC')}
    base=load('control_smoke',expected,3);trial=load('candidate_smoke',expected,3)
    for key in base:
        assert base[key]['output_sha256']==trial[key]['output_sha256'],key
        assert base[key]['operand_sha256']==trial[key]['operand_sha256'],key
    print('PASS: 27 smoke cases per body, exact all-output bit agreement and strict numerical/workspace checks')
    raise SystemExit(0)
small={(m,256,n,p,c) for m,n,p,c in itertools.product((1,5,8,16,64),(128,2048),('mixed','zero_a','zero_w'),'ABC')}
main={(m,k,n,'mixed',c) for m,(k,n),c in itertools.product((1,5,8,16,64),shapes,'ABC')}
checks={}
for part,expected in [('small',small),('main',main)]:
    base=load('control_'+part,expected,5);trial=load('candidate_'+part,expected,5)
    for key in base:
        assert base[key]['output_sha256']==trial[key]['output_sha256'],key
        assert base[key]['operand_sha256']==trial[key]['operand_sha256'],key
    checks[part]=dict(rows_per_body=len(base),all_output_bits_match=True)
for tool in ('memcheck','racecheck','synccheck'):
    for part,expected,repeats in [('small',{v for v in small if v[0] in (1,5,8) and v[2]==2048},2),('main',{v for v in main if v[0] in (8,16,64)},1)]:
        label=tool+'_'+part;data=load(label,expected,repeats)
        log=(r/(label+'.log')).read_text()
        assert len(re.findall(r'(?:RACECHECK SUMMARY:|ERROR SUMMARY:)',log))==1
        assert re.search(r'(?:RACECHECK SUMMARY:|ERROR SUMMARY:).*0 errors',log)
        if tool=='racecheck':assert '0 warnings' in log
        checks[label]=dict(rows=len(data),errors=0)
(r/'validation-check.json').write_text(json.dumps(dict(build=build,checks=checks),indent=2)+'\n')
print('PASS: matched source/SASS, 270 numerical rows with full-output bit identity, 162 clean sanitizer rows')
