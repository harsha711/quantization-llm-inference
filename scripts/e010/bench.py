#!/usr/bin/env python3
"""E010: four-matrix corrected Marlin / pre-materialized FP16 batch crossover."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('e008',Path(__file__).resolve().parents[1]/'e008/bench.py')
e008 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e008)
e005, e004 = e008.e005, e008.e005.e004
BODY = 'e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7'
SHAPES = [(4096,11008),(11008,4096)]
MS = [4,16,64,128]
NAMES = ['marlin','cublas_dq']
CAPACITY = 128


def sha(tensor):
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def order(k,n,m,series):
    # Stable parity also balances a paired-shape midpoint run.
    shift = (SHAPES.index((k,n)) if (k,n) in SHAPES else 0) + (MS.index(m) if m in MS else 0)
    names = NAMES[shift%2:]+NAMES[:shift%2]
    return names if series == 1 else names[::-1]


def prepare(args,k,n):
    cases = [e005.Case(args,k,n,[CAPACITY],seed,'mixed') for seed in (3,4,5,6)]
    metadata = []
    for case in cases:
        case.a = cases[0].a
        case.hashes['A'] = cases[0].hashes['A']
        codes = e004.unpack_marlin(case.packed.cpu().numpy(),k,n,case.ns['get_weight_perm'](4).numpy())
        inv = np.argsort(np.array(case.ns['get_scale_perms']()[0]))
        scales = case.scales.cpu().numpy().reshape(-1,64)[:,inv].reshape(k//128,n)
        wh = ((codes.astype(np.float32)-8)*scales.repeat(128,axis=0).astype(np.float32)).astype(np.float16)
        assert hashlib.sha256(wh.tobytes()).hexdigest() == case.hashes['W_hat']
        if case.seed != 3:
            activation = case.a.cpu().numpy()
            case.bridge.e004_reference(activation.ctypes.data,wh.ctypes.data,CAPACITY,n,k,case.ref.ctypes.data)
        case.wh = torch.from_numpy(wh).cuda()
        del case.flush_buf
        metadata.append(dict(seed=case.seed,operand_sha256=case.hashes,
            **{name+'_range':e008.allocation(tensor) for name,tensor in
               [('packed',case.packed),('scales',case.scales),('wh',case.wh),('output',case.c),
                ('locks',case.locks_storage),('scratch',case.tmp_storage),('activation',case.a)]},
            **{name+'_sha256':sha(tensor) for name,tensor in
               [('packed',case.packed),('scales',case.scales),('wh',case.wh)]}))
    ranges = [item[key+'_range'] for item in metadata for key in ('packed','scales','wh','output','locks','scratch')]
    e008.distinct(ranges+[metadata[0]['activation_range']])
    assert len({v['operand_sha256']['A'] for v in metadata}) == 1
    for key in ('packed_sha256','scales_sha256','wh_sha256'):
        assert len({v[key] for v in metadata}) == 4
    return cases,metadata


def main(args):
    assert not args.out.exists()
    assert hashlib.sha256(args.local.read_bytes()).hexdigest() == BODY
    bridge = e004.bridge(args.bridge)
    handle = bridge.e004_handle()
    cublas = ct.CDLL('libcublas.so.13')
    cublas.cublasSetMathMode.argtypes = [ct.c_void_p,ct.c_int]
    cublas.cublasGetMathMode.argtypes = [ct.c_void_p,ct.POINTER(ct.c_int)]
    mode = ct.c_int()
    assert cublas.cublasSetMathMode(handle,16) == 0
    stream = torch.cuda.Stream()
    print('ENV',json.dumps(dict(body_sha256=BODY,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),stream=int(stream.cuda_stream))),flush=True)
    with torch.cuda.stream(stream),args.out.open('x') as out:
        for k,n in args.shapes:
            cases,metadata = prepare(args,k,n)
            for m in args.ms:
                resources = [case.call('A',m,True) for case in cases]
                kernels = []
                if args.trace:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                        cases[0].call('A',m)
                        torch.cuda.synchronize()
                    kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                    assert len(kernels) == 1 and kernels[0].startswith('void e005_marlin::Marlin<'),kernels
                    prof.export_chrome_trace(str(args.out.parent/f'{args.out.stem}_M{m}_K{k}_N{n}.trace.json'))
                    print('KERNEL',m,k,n,kernels,resources[0],flush=True)
                for position,name in enumerate(order(k,n,m,args.series)):
                    if name not in args.implementations:
                        continue
                    def launch():
                        for case in cases:
                            if name == 'marlin':
                                case.call('A',m)
                            else:
                                bridge.e004_launch(5,case.a.data_ptr(),None,None,case.wh.data_ptr(),
                                    case.wh.data_ptr(),case.c.data_ptr(),m,n,k,stream.cuda_stream,handle)
                    def poison():
                        for case in cases:
                            case.c.fill_(float('nan'))
                    def verify():
                        checks = []
                        for case in cases:
                            case.check_workspace()
                            assert torch.all(torch.isnan(case.c[m:])).item(),'output row-tail overwritten'
                            checks.append(dict(**e005.stats(case.c[:m],case.ref[:m]),output_sha256=sha(case.c[:m])))
                        assert cublas.cublasGetMathMode(handle,ct.byref(mode)) == 0 and mode.value == 16
                        return checks
                    for _ in range(3):
                        poison(); launch(); torch.cuda.synchronize()
                        checks = verify()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        launch()
                    for _ in range(3):
                        poison(); graph.replay(); torch.cuda.synchronize()
                        checks = verify()
                    row = dict(K=k,N=n,M=m,capacity=CAPACITY,matrices=4,group=128,seeds=[3,4,5,6],
                        config=name,series=args.series,order=position,method='complete_cycle_graph_events',
                        policy='no_artificial_flush',warmup=args.warmup,iters=args.iters,body_sha256=BODY,
                        cublas_math_mode=mode.value,resources=resources if name == 'marlin' else [],
                        traced_kernels=kernels if name == 'marlin' else [],allocations=metadata,
                        activation_prefix_sha256=sha(cases[0].a[:m]),
                        live_tensor_storage_bytes=sum(b-a for a,b in [metadata[0]['activation_range']]+
                            [v[key+'_range'] for v in metadata for key in ('packed','scales','wh','output','locks','scratch')]),
                        framework_allocated_bytes=torch.cuda.memory_allocated(),framework_reserved_bytes=torch.cuda.memory_reserved(),
                        library_workspace='cuBLAS default library-managed, size not exposed',
                        active_weight_bytes=4*(2*k*n if name == 'cublas_dq' else k*n//2+k*n//64),
                        l2_bytes=bridge.e004_l2(),codes_match=True,scales_match=True,reconstruction_match=True,
                        lock_reset=True,workspace_guards=True,output_tail_untouched=True,
                        direct_cycles=3,poisoned_graph_cycles=3,validation=checks)
                    if args.iters:
                        poison()
                        row.update(e004.timed(launch,None,args.warmup,args.iters,cases[0].c[:m]))
                        row['post_timing_validation'] = verify()
                        row['median_per_matrix_us'] = row['median_us']/4
                    out.write(json.dumps(row)+'\n'); out.flush()
                    print('PASS',m,k,n,name,row.get('median_us'),flush=True)
                    del graph
            del cases
    bridge.e004_destroy(handle)
    print('MAPPED_LIBRARIES','\n'.join(x for x in Path('/proc/self/maps').read_text().splitlines()
        if any(v in x for v in ('libe005','libe004','libcublas','libcudart','libtorch'))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source','local','wheel','bridge','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shapes',nargs='+',default=['4096,11008','11008,4096'])
    p.add_argument('--ms',nargs='+',type=int,default=MS)
    p.add_argument('--implementations',nargs='+',choices=NAMES,default=NAMES)
    p.add_argument('--series',type=int,choices=[1,2],default=1)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iters',type=int,default=100)
    p.add_argument('--trace',action='store_true')
    args = p.parse_args()
    args.configs = ['A']
    args.shapes = [tuple(map(int,x.split(','))) for x in args.shapes]
    assert len(set(args.ms)) == len(args.ms) and all(0 < m <= CAPACITY for m in args.ms)
    assert args.iters >= 0 and args.warmup >= 0 and not (args.trace and args.iters)
    main(args)
