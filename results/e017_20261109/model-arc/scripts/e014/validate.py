#!/usr/bin/env python3
"""Validate one variant/context against fixed dynamic reference, then audit replay."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch._dynamo.utils import counters

from model import Workload, Runner, tensor_sha


def compiler_counters():
    return {key:dict(value) for key,value in counters.items() if value}


def validate_logits(got,ref):
    assert got.shape == ref.shape and torch.isfinite(got).all() and torch.isfinite(ref).all()
    delta = (got.float()-ref.float()).abs()
    bad = int((delta > 0.03+0.01*ref.float().abs()).sum())
    relative_l2 = float(torch.linalg.vector_norm(delta.double())/torch.linalg.vector_norm(ref.double()))
    assert bad == 0 and relative_l2 <= 0.003,(bad,relative_l2,float(delta.max()))
    top = ref.float().topk(2,dim=-1).values
    return dict(bad_logits=bad,logits_checked=ref.numel(),max_abs=float(delta.max()),
        relative_l2=relative_l2,min_top2_margin=float((top[...,0]-top[...,1]).min()),
        reference_logits_sha256=tensor_sha(ref),candidate_logits_sha256=tensor_sha(got))


def uncached(work,runner,record,logits):
    checks = []
    with torch.inference_mode():
        for step in (0,63):
            ids = record['input_ids']+record['output_ids'][:step]
            inputs = torch.tensor([ids],dtype=torch.long,device='cuda')
            position = torch.arange(len(ids),device='cuda')
            output = work.model(input_ids=inputs,position_ids=position.view(1,-1),cache_position=position,
                attention_mask=runner.bias(position,len(ids)),use_cache=False,logits_to_keep=1).logits[:,0,:].cpu()
            assert output.argmax(dim=-1).item() == record['output_ids'][step]
            checks.append(dict(step=step,**validate_logits(logits[step],output)))
    return checks


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('results',type=Path)
    p.add_argument('--variant',choices=['A','B','C'],required=True)
    p.add_argument('--context',type=int,choices=[128,2048],required=True)
    a = p.parse_args()
    tag = f'validation_{a.variant}_{a.context}'
    assert not (a.results/(tag+'.json')).exists()
    work = Workload(a.results)
    runner = Runner(work,a.variant,a.context)
    cold = []
    for i in range(3):
        record = runner.request('primary')
        record.pop('logits')
        cold.append(record)
        print('WARMUP',i,record['e2e_s'],compiler_counters(),flush=True)
    validations = []
    for prompt in ('primary','heldout'):
        record = runner.request(prompt,capture_logits=True)
        logits = record.pop('logits')
        path = a.results/f'reference_{prompt}_{a.context}.npz'
        meta = a.results/f'reference_{prompt}_{a.context}.json'
        if a.variant == 'A':
            assert not path.exists() and not meta.exists()
            np.savez_compressed(path,logits=logits.numpy())
            meta.write_text(json.dumps(dict(record=record,weights_sha256=work.weights_sha256),indent=2)+'\n')
            ref = logits
            uncached_checks = uncached(work,runner,record,logits) if prompt == 'primary' else []
        else:
            saved = json.loads(meta.read_text())
            assert saved['weights_sha256'] == work.weights_sha256
            assert saved['record']['input_ids'] == record['input_ids']
            assert saved['record']['output_ids'] == record['output_ids'],'greedy tokens diverged'
            with np.load(path) as data:
                ref = torch.from_numpy(data['logits'])
            uncached_checks = []
        validations.append(dict(prompt=prompt,record=record,uncached=uncached_checks,
                                **validate_logits(logits,ref)))
        print('PASS numerical/output',a.variant,a.context,prompt,validations[-1]['max_abs'],flush=True)
    before_trace = compiler_counters()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
        trace_record = runner.request('primary',generated=8,markers=True)
    trace_record.pop('logits')
    prof.export_chrome_trace(str(a.results/f'timeline_{a.variant}_{a.context}_validated.json'))
    after_trace = compiler_counters()
    if a.variant == 'C':
        assert after_trace.get('stats',{}).get('unique_graphs',0) >= 1
        assert not after_trace.get('graph_break')
    else:
        assert after_trace.get('stats',{}).get('unique_graphs',0) == 0
    work.weights_unchanged()
    payload = dict(variant=a.variant,context=a.context,identity=work.identity,
        static_setup_s=runner.setup_s,cold_requests=cold,validation=validations,
        compiler_before_trace=before_trace,compiler_after_trace=after_trace,trace_request=trace_record)
    (a.results/(tag+'.json')).write_text(json.dumps(payload,indent=2)+'\n')
    print('PASS',tag,after_trace,flush=True)
