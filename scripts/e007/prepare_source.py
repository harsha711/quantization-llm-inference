#!/usr/bin/env python3
"""Preserve corrected E006 and guard only the analyzed terminal A/B lookahead."""
import argparse
import difflib
import itertools
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('corrected',type=Path)
p.add_argument('out',type=Path)
a=p.parse_args()
a.out.mkdir(exist_ok=False)
control=a.out/'control';candidate=a.out/'candidate'
shutil.copytree(a.corrected,control);shutil.copytree(control,candidate)
f=candidate/'csrc/quantization/marlin/marlin_template.h'
before=f.read_text()
assert 'Join all shared readers and drained copy groups before overlay reuse.' in before
old='        fetch_to_registers(k + 1, pipe % stages);'
new='''        if constexpr (a_type_id == vllm::kFloat16.id() &&
                      b_type_id == vllm::kU4B8.id() &&
                      c_type_id == vllm::kFloat16.id() &&
                      s_type_id == vllm::kFloat16.id() &&
                      threads == 256 && thread_m_blocks == 1 &&
                      thread_n_blocks == 8 && thread_k_blocks == 8 &&
                      m_block_size_8 && stages == 4 && group_blocks == 8 &&
                      !is_zp_float) {
          // A new stripe replaces buffer0 before use, so omit its last lookahead.
          if (k != b_sh_wr_iters - 1 || slice_iters > 1)
            fetch_to_registers(k + 1, pipe % stages);
        } else {
          fetch_to_registers(k + 1, pipe % stages);
        }'''
assert before.count(old)==1
after=before.replace(old,new)
assert after.replace(new,old)==before and after.count('__syncthreads();')==before.count('__syncthreads();')
f.write_text(after)
(a.out/'e007-lookahead.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/marlin_template.h',tofile='b/marlin_template.h')))
for file in control.rglob('*'):
    if file.is_file() and file.name!='marlin_template.h':
        assert file.read_bytes()==(candidate/file.relative_to(control)).read_bytes()
model=[]
for length in range(1,87):
    # Both A/B fragments use the same alternating register-buffer tags.
    for variant in ('control','candidate'):
        buffers={0:(0,0)};used=[];loads=[]
        for tile in range(length):
            for k in range(2):
                if variant=='control' or k!=1 or length-tile>1:
                    tag=(tile+(k==1),(k+1)%2)
                    buffers[(k+1)%2]=tag;loads.append(tag)
                used.append(buffers[k%2])
        assert used==list(itertools.product(range(length),range(2)))
        assert len(loads)==2*length-(variant=='candidate')
        if variant=='candidate':assert (length,0) not in loads
    model.append(dict(stripe_tiles=length,consumed=2*length,discarded_fetches_elided=1))
(a.out/'buffer-model.json').write_text(json.dumps(model,indent=2)+'\n')
print('PASS: one bounded source guard, unchanged barriers, identical consumed A/B tags for 86 stripe lengths')
