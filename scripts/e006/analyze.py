#!/usr/bin/env python3
"""Assert E006's bounded correction evidence, without releasing E005 timing."""
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import tarfile

root = Path(sys.argv[1])
repo = Path(__file__).resolve().parents[2]
e005 = repo/'results/e005_20261106'

def read_json(name):
    return json.loads((root/name).read_text())

def exit_code(name, expected):
    assert int((root/(name+'.exit')).read_text()) == expected, name

def summary(name, clean):
    text = (root/(name+'.log')).read_text()
    lines = re.findall(r'(?:RACECHECK SUMMARY:|ERROR SUMMARY:).*', text)
    assert len(lines) == 1, (name, lines)
    if clean:
        assert '0 errors' in lines[0], lines
        if 'RACECHECK' in lines[0]:
            assert '0 warnings' in lines[0], lines
    else:
        assert re.search(r'\b[1-9][0-9]* errors?\b', lines[0]), lines
    return lines[0]

for name in ('original_square','original_small_analysis','control_square'):
    exit_code(name,99)
    summary(name,False)
    r=read_json(name+'.json')
    assert r['num_bad']==0 and r['checked']==8*r['N'] and r['direct_launches']==1
    assert all(r[k] for k in ('codes_match','scales_match','reconstruction_match','workspace_guards','lock_reset'))
exit_code('original_square_filter0',0)
exit_code('original_small',124)
assert not (root/'original_small.json').exists(), 'Timed-out record must not be counted as a validated reproducer'
exit_code('patched_square',0)
summary('patched_square',True)
patched_case=read_json('patched_square.json')
assert patched_case['num_bad']==0 and patched_case['checked']==32768
assert patched_case['operand_sha256']==read_json('control_square.json')['operand_sha256']==read_json('original_square.json')['operand_sha256']

def kernels(path):
    return {b.splitlines()[0].strip():b for b in gzip.open(path,'rt').read().split('Function : ')[1:]}

old=kernels(e005/'sass.txt.gz')
control=kernels(root/'control/sass.txt.gz')
patched=kernels(root/'patched/sass.txt.gz')
assert old==control and old.keys()==patched.keys() and len(old)==16
sass={}
for name in control:
    a,b=control[name],patched[name]
    expected=0 if 'MarlinDefault' in name else 1
    assert b.count('BAR.SYNC')-a.count('BAR.SYNC')==expected, name
    sass[name]={}
    for variant,body in [('control',a),('patched',b)]:
        instructions=re.findall(r'/\*[0-9a-f]+\*/\s+([^;]+);',body)
        sass[name][variant]=dict(static_instructions=len(instructions), nop=sum('NOP' in x for x in instructions),
            block_barriers=body.count('BAR.SYNC'),sha256=hashlib.sha256(body.encode()).hexdigest())

# Verify the exact compiled source archives differ by one barrier only.
with tarfile.open(root/'matched-source.tar.gz') as archive:
    files={m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
oldline=b'      cp_async_wait<0>();\n      bool last = slice_idx == slice_count - 1;'
newline=b'      cp_async_wait<0>();\n      // Join all shared readers and drained copy groups before overlay reuse.\n      __syncthreads();\n      bool last = slice_idx == slice_count - 1;'
for name,content in files.items():
    if not name.startswith('external/control/'):
        continue
    other=files[name.replace('external/control/','external/patched/',1)]
    if name.endswith('/marlin_template.h'):
        assert content.count(oldline)==1 and content.replace(oldline,newline)==other
    else:
        assert content==other,name
with tarfile.open(e005/'pinned-source.tar.gz') as archive:
    for name in ('marlin_template.h','marlin.cuh','marlin_dtypes.cuh','dequant.h','marlin_mma.h','kernel.h'):
        assert files['external/control/csrc/quantization/marlin/'+name]==archive.extractfile('pinned/csrc/quantization/marlin/'+name).read()

# Byte-accurate displayed hazards must map to ordinary B reads and first writes.
text=(root/'original_square.log').read_text()
pattern=r'Potential (\w+) hazard detected at __shared__ (0x[0-9a-f]+) in block \((\d+),0,0\) :\n=========     Read Thread \((\d+),0,0\).*?const\+(0x[0-9a-f]+).*?\n=========     Write Thread \((\d+),0,0\).*?const\+(0x[0-9a-f]+)'
hazards=re.findall(pattern,text,re.S)
assert len(hazards)==100
for kind,addr,block,reader,rpc,writer,wpc in hazards:
    t,u=int(reader),int(writer)
    assert kind=='WAR' and rpc=='0x2bc0' and wpc=='0x4a00' and t//32!=u//32
    rd=1024+16*(512+t+64*(t//64))
    wr=1024+16*(512*(u//64-2)+u%64)
    assert rd<=int(addr,16)<rd+16 and wr<=int(addr,16)<wr+16
hazard_details=dict(displayed=len(hazards),addresses=[hex(min(int(x[1],16) for x in hazards)),hex(max(int(x[1],16) for x in hazards))],
    blocks=sorted({int(x[2]) for x in hazards}),reader_threads=sorted({int(x[3]) for x in hazards}),writer_threads=sorted({int(x[5]) for x in hazards}))

# Source-level double buffering: exactly one terminal lookahead is discarded.
for length in range(1,87):
    buffers={0:(0,0)}
    consumed=[]
    for tile in range(length):
        for k in range(2):
            buffers[(k+1)%2]=(tile+(k==1),(k+1)%2)
            consumed.append(buffers[k%2])
    assert consumed==list(itertools.product(range(length),range(2)))
    assert buffers[0]==(length,0) and (length,0) not in consumed

binary_sha={name:(root/name/'build-source.sha256').read_text().split()[0] for name in ('control','patched')}
assert binary_sha['control']!=binary_sha['patched']
assert binary_sha['patched'] in (root/'patched_square.log').read_text()
small={(m,k,n,p,c) for m,k,n,p,c in itertools.product((1,5,8,16,64),(256,),(128,2048),('mixed','zero_a','zero_w'),'ABC')}
main={(m,k,n,'mixed',c) for m,(k,n),c in itertools.product((1,5,8,16,64),((4096,4096),(4096,11008),(11008,4096)),'ABC')}
sani_small={(m,256,2048,p,c) for m,p,c in itertools.product((1,5,8),('mixed','zero_a','zero_w'),'ABC')}
sani_main={v for v in main if v[0] in (8,16,64)}
matrix={}
for name,expected,repeats in [('validation_small',small,5),('validation_main',main,5)]+[(tool+'_'+part,expected,repeats) for tool in ('memcheck','racecheck','synccheck') for part,expected,repeats in [('small',sani_small,2),('main',sani_main,1)]]:
    exit_code(name,0)
    rows=[json.loads(x) for x in (root/(name+'.jsonl')).read_text().splitlines()]
    assert len(rows)==len(expected)
    assert {(r['M'],r['K'],r['N'],r['pattern'],r['config']) for r in rows}==expected
    assert binary_sha['patched'] in (root/(name+'.log')).read_text()
    for r in rows:
        assert r['num_bad']==0 and r['checked']==r['M']*r['N'] and r['iters']==r['warmup']==0
        assert r['repeated_launches']==repeats and r['graph_check'] and r['policy']=='no_flush'
        assert all(r[k] for k in ('codes_match','scales_match','reconstruction_match','workspace_guards','lock_reset'))
        assert 'samples_us' not in r and 'median_us' not in r
        attr=r['resources']
        if r['M']<=8 and r['N']>=2048:
            assert (attr['grid'],attr['dynamic_shared'],attr['candidate'])=={'A':(36,101376,0),'B':(36,49664,1),'C':(72,49664,1)}[r['config']]
            assert attr['registers']==126 and attr['local_bytes']==32 and attr['static_shared']==0
        else:
            assert attr['candidate']==0 and attr['grid']==36 and attr['dynamic_shared']==101376
    # A/B/C must share the compiled entrypoint for each problem in one process.
    for key in {(r['M'],r['K'],r['N'],r['pattern']) for r in rows}:
        group=[r for r in rows if (r['M'],r['K'],r['N'],r['pattern'])==key]
        assert len({r['resources']['kernel_address'] for r in group})==1
    matrix[name]=dict(rows=len(rows),all_output_checks=True,direct_repeats=repeats,poisoned_graph_replays=repeats,
        max_abs=max(r['max_abs'] for r in rows),max_rel_l2=max(r['rel_l2'] for r in rows))
    if name.startswith(('memcheck','racecheck','synccheck')):
        matrix[name]['sanitizer_summary']=summary(name,True)

# E005's historical failed gate remains immutable. Nothing enables timing here.
assert not (e005/'VALIDATED').exists() and not (root/'VALIDATED').exists()
assert not list(root.glob('*series*')) and not list(root.glob('*ncu*'))
out=dict(classification='source-supported synchronization correction, reported hazard absent in tested cases',
    incorrect_original_output_demonstrated=False,performance_measured=False,binary_sha256=binary_sha,
    original_detailed_summary=summary('original_square',False),displayed_hazards=hazard_details,
    matrix=matrix,sass=sass,control_sass_identical_to_e005=True,
    minimized_reproducer=dict(M=8,K=256,N=2048,summary=summary('original_small_analysis',False)),
    limits=['one GPU and seed','terminal-unused dataflow established only for the analyzed specialization',
        'official-wheel exact source-PC mapping unestablished','default detailed print limit 100','no latency or achieved-residency measurements'])
(root/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS: E006 evidence consistent. E005 timing remains disabled.')
print(json.dumps(dict(classification=out['classification'],matrix=matrix,displayed_hazards=hazard_details),indent=2))
