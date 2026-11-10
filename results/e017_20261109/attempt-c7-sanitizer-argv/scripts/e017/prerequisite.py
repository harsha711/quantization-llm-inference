#!/usr/bin/env python3
"""Captured actual-layer operands and fail-fast corrected-Marlin integration gate."""
import argparse
import ctypes as ct
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from integration import Workload, BaseRunner, Operator, Linear, save, tensor_sha, array_sha, repack, load_packer, e004, read_artifact

REPRESENTATIVE='model.layers.0.mlp.down_proj'


def artifact(path,data,k,n):
    np.savez(path,**{x:data[x] for x in ('packed','scales','marlin_packed','marlin_scales')})
    return dict(K=k,N=n,hashes=data['hashes'],file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),repack_s=data['repack_s'],quantize_s=data['quantize_s'])


@torch.inference_mode()
def capture(root):
    work=Workload(root,'O')
    selected=[r for r in work.identity['coverage'] if r['name'].startswith('model.layers.0.')]
    arrays={r['name']:{} for r in selected};handles=[];current=[128]
    for r in selected:
        def hook(module,args,name=r['name']):
            x=args[0].reshape(-1,module.k).contiguous()
            key=f'A_{current[0]}_{x.shape[0]}'
            assert key not in arrays[name]
            arrays[name][key]=x.cpu().numpy().copy()
        handles.append(work.model.get_submodule(r['name']).register_forward_pre_hook(hook))
    for context in (128,2048):
        current[0]=context
        runner=BaseRunner(work,'B',context)
        row=runner.request('primary',generated=2)
        row.pop('logits');save(root/f'capture_{context}.json',row)
        del runner
    for handle in handles: handle.remove()
    for name,values in arrays.items():
        np.savez(root/f'{name}-activations.npz',**values)
    source,wheel=load_packer()
    weight=work.model.get_submodule(REPRESENTATIVE).weight.detach().cpu().numpy().T.copy()
    data=repack(weight,source,wheel)
    meta=artifact(root/'one-layer.npz',data,*weight.shape)
    save(root/'one-layer.json',dict(name=REPRESENTATIVE,**meta,identity=work.identity,
        activations={name:{key:array_sha(x) for key,x in a.items()} for name,a in arrays.items()}))
    work.weights_unchanged()
    print('PASS actual model activation capture, original inventory and lossless representative repack',flush=True)


@torch.inference_mode()
def check(root,m,name=REPRESENTATIVE,all_layers=False,suffix="",trace_enabled=True):
    if all_layers:
        manifest=json.loads((root/'artifacts/manifest.json').read_text());metadata=manifest['layers'][name]
        path=root/'artifacts'/f'{name}.npz'
    else:
        metadata=json.loads((root/'one-layer.json').read_text());path=root/'one-layer.npz'
    data=read_artifact(path,metadata);k,n=data['K'],data['N']
    with np.load(root/f'{name}-activations.npz') as f: a=f[f'A_{128 if m in (1,128) else 2048}_{m}'].copy()
    codes=e004.unpack_project(data['packed'],k,n)
    wh=((codes.astype(np.float32)-8)*data['scales'].repeat(128,axis=0).astype(np.float32)).astype(np.float16)
    assert array_sha(wh)==data['hashes']['reconstructed']
    reference=np.empty((m,n),np.float64)
    bridge=e004.bridge(Path(os.environ['E017_BRIDGE']))
    bridge.e004_reference(a.ctypes.data,wh.ctypes.data,m,n,k,reference.ctypes.data)
    op=Operator();x=torch.from_numpy(a).cuda();q=torch.from_numpy(data['marlin_packed']).cuda();s=torch.from_numpy(data['marlin_scales']).cuda()
    output=torch.empty((m,n),dtype=torch.float16,device='cuda')
    resources=op.call(x,q,s,output,True)
    tag=f"operator_{name}_M{m}" if all_layers else f'one_layer_M{m}'
    tag += suffix
    assert not (root/f'{tag}_0-outputs.npz').exists() and not (root/f'{tag}_0.json').exists()
    records=[]
    for index in range(3):
        output.fill_(float('nan'));op.call(x,q,s,output);torch.cuda.synchronize()
        got=output.cpu().numpy();err=np.abs(got.astype(np.float64)-reference)
        rms=float(np.sqrt(np.mean(reference*reference)))
        bad=~np.isfinite(got)|(err>1e-3*np.abs(reference)+1e-3*rms)
        row=dict(iteration=index,checked=int(got.size),bad=int(bad.sum()),max_abs=float(err.max()),rms=rms,
            rel_l2=float(np.linalg.norm(err)/np.linalg.norm(reference)),resources=resources,workspace=op.audit(),
            activation_sha256=array_sha(a),weight_hashes=data['hashes'],candidate_sha256=array_sha(got),reference_sha256=array_sha(reference))
        row['finite']=bool(np.isfinite(got).all() and np.isfinite(reference).all())
        for key in ('max_abs','rms','rel_l2'):
            if not np.isfinite(row[key]):row[key]=None
        # Preserve actual failed payloads, workspace diagnostics and reference before gates.
        np.savez_compressed(root/f'{tag}_{index}-outputs.npz',candidate=got,reference=reference,activation=a)
        save(root/f'{tag}_{index}.json',row)
        assert row['finite'] and not row['bad'] and all(row['workspace'].values()),row
        records.append(row)
    assert len({r['candidate_sha256'] for r in records})==1
    # Invoke the same module.forward used by full inference, including retained bias.
    bias=None
    if name.rsplit('.',1)[1] in ('q_proj','k_proj','v_proj'):
        with np.load(root/'artifacts'/f'{name}.npz') as f: bias=torch.nn.Parameter(torch.from_numpy(f['bias']).cuda(),requires_grad=False)
    module=Linear(name,bias,data,'Q',op)
    for index in range(2):
        wrapped=module(x);torch.cuda.synchronize()
        expected=output if bias is None else output+bias
        equal=torch.equal(wrapped.view(torch.int16),expected.view(torch.int16))
        np.savez_compressed(root/f'{tag}_module_{index}.npz',candidate=wrapped.cpu().numpy(),reference=expected.cpu().numpy())
        save(root/f'{tag}_module_{index}.json',dict(bitwise_equal=equal,workspace=op.audit(),bias=bias is not None))
        assert equal and all(op.audit().values())
    if not trace_enabled:
        print('PASS',tag,records[-1],flush=True)
        return
    op.markers=True
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        module(x);torch.cuda.synchronize()
    trace=root/f'{tag}-trace.json';prof.export_chrome_trace(str(trace))
    gpu=[e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA and 'Marlin<' in e.name]
    save(root/f'{tag}-trace-summary.json',dict(kernels=gpu,resources=resources,expected_launches=2 if m==2048 and n>4096 else 1))
    with trace.open('rb') as f,gzip.open(str(trace)+'.gz','wb') as z:z.write(f.read())
    trace.unlink()
    assert len(gpu)==(2 if m==2048 and n>4096 else 1) and all('e005_marlin::Marlin<' in x for x in gpu),gpu
    print('PASS',tag,records[-1],flush=True)


@torch.inference_mode()
def prepare(root):
    assert (root/'ONE_LAYER_INSPECTED').exists()
    dest=root/'artifacts';dest.mkdir()
    work=Workload(root,'O');source,wheel=load_packer();started=time.perf_counter()
    ns=e004.helpers(source)
    perm_path=dest/'packing-permutations.npz'
    np.savez(perm_path,weight=ns['get_weight_perm'](4).numpy(),scales=np.asarray(ns['get_scale_perms']()[0]))
    rows=work.identity['coverage'];manifest=dict(original_weights_sha256=work.original_identity['weights_sha256'],coverage=rows,layers={})
    for row in rows:
        name=row['name'];module=work.model.get_submodule(name);path=dest/f'{name}.npz'
        start=time.perf_counter()
        if name==REPRESENTATIVE:
            old=json.loads((root/'one-layer.json').read_text())
            assert old['identity']['original_weights_sha256']==work.original_identity['weights_sha256']
            data=read_artifact(root/'one-layer.npz',old)
            meta={key:old[key] for key in ('K','N','hashes','repack_s','quantize_s')}
        else:
            weight=module.weight.detach().cpu().numpy().T.copy()
            data=repack(weight,source,wheel)
            meta=dict(K=row['K'],N=row['N'],hashes=data['hashes'],repack_s=data['repack_s'],quantize_s=data['quantize_s'])
            del weight
        arrays={key:data[key] for key in ('packed','scales','marlin_packed','marlin_scales')}
        if module.bias is not None:arrays['bias']=module.bias.detach().cpu().numpy().copy()
        np.savez(path,**arrays)
        meta.update(file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),preparation_s=time.perf_counter()-start)
        manifest['layers'][name]=meta
        save(dest/f'{name}.json',meta)
        del data,arrays
        print('PREPARED',name,meta,flush=True)
    manifest['packing_permutations_sha256']=hashlib.sha256(perm_path.read_bytes()).hexdigest()
    manifest['total_preparation_s']=time.perf_counter()-started
    save(dest/'manifest.json',manifest)
    work.weights_unchanged()
    print('PASS deterministic one-time artifact preparation',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--mode',choices=('capture','check','prepare'),required=True)
    p.add_argument('--m',type=int,choices=(1,128,2048),default=1);p.add_argument('--name',default=REPRESENTATIVE);p.add_argument('--all-layers',action='store_true')
    p.add_argument('--suffix',default='');p.add_argument('--no-trace',action='store_true')
    a=p.parse_args()
    if a.mode=='capture':capture(a.results)
    elif a.mode=='prepare':prepare(a.results)
    else:check(a.results,a.m,a.name,a.all_layers,a.suffix,not a.no_trace)
