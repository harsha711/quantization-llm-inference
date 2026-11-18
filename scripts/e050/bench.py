#!/usr/bin/env python3
"""E050: `tc_ladder` (tc_m32 at M >= 64, else tc) versus `tc`, corrected Marlin A and W_hat cuBLAS in
E045's four-matrix cycle, on FRESH weight/activation seeds 7-10 and three shapes (adds K=N=4096).

Copy of E045's/E049's timing loop. prepare() is E010's + E045's preparation with the seed list as a
parameter (E010 hard-codes 3-6); the activation is the first seed's (seed 7), as E010 uses seed 3's."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

# E010 module (E008 allocation helpers, E005 Case, E004 bridge/timer), imported unchanged.
spec = importlib.util.spec_from_file_location('e010',Path(__file__).resolve().parents[1]/'e010/bench.py')
e010 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e010)
e008, e005, e004 = e010.e008, e010.e005, e010.e004
BODY, CAPACITY, MS, sha = e010.BODY, e010.CAPACITY, e010.MS, e010.sha
SHAPES = [(4096,11008),(11008,4096),(4096,4096)]
SEEDS = [7,8,9,10]
NAMES = ['marlin','tc','tc_ladder','cublas_dq']


def order(k,n,m,series):
    # 3 shapes x 4 M: each rotation offset occurs three times per series.
    shift = ((SHAPES.index((k,n)) if (k,n) in SHAPES else 0) + (MS.index(m) if m in MS else 0)) % 4
    names = NAMES[shift:]+NAMES[:shift]
    return names if series == 1 else names[::-1]


def expect_kernel(name,m,kernels):
    if name == 'marlin':
        assert len(kernels) == 1 and kernels[0].startswith('void e005_marlin::Marlin<'),kernels
    elif name == 'tc' or (name == 'tc_ladder' and m < 64):
        assert len(kernels) == 1 and 'w4a16_tc_kernel<true' in kernels[0],kernels
    elif name == 'tc_ladder':
        assert len(kernels) == 1 and 'w4a16_tc_m32_kernel<false>' in kernels[0],kernels


def prepare(args,k,n):
    cases = [e005.Case(args,k,n,[CAPACITY],seed,'mixed') for seed in SEEDS]
    metadata = []
    for case in cases:
        case.a = cases[0].a
        case.hashes['A'] = cases[0].hashes['A']
        codes = e004.unpack_marlin(case.packed.cpu().numpy(),k,n,case.ns['get_weight_perm'](4).numpy())
        inv = np.argsort(np.array(case.ns['get_scale_perms']()[0]))
        scales = case.scales.cpu().numpy().reshape(-1,64)[:,inv].reshape(k//128,n)
        wh = ((codes.astype(np.float32)-8)*scales.repeat(128,axis=0).astype(np.float32)).astype(np.float16)
        assert hashlib.sha256(wh.tobytes()).hexdigest() == case.hashes['W_hat']
        if case.seed != SEEDS[0]:
            activation = case.a.cpu().numpy()
            case.bridge.e004_reference(activation.ctypes.data,wh.ctypes.data,CAPACITY,n,k,case.ref.ctypes.data)
        case.wh = torch.from_numpy(wh).cuda()
        del case.flush_buf
        # Project-layout codes/scales of the same weight (E045).
        cpu = [np.empty((4,k),np.float16),np.empty((k,n),np.float16),
               np.empty((k//8,n),np.uint32),np.empty((k//128,n),np.float16),np.empty((k,n),np.float16)]
        assert case.bridge.e004_generate(4,n,k,case.seed,*[x.ctypes.data for x in cpu]) == 0
        for key,array in zip(['Q','S','W_hat'],cpu[2:]):
            assert hashlib.sha256(array.tobytes()).hexdigest() == case.hashes[key]
        case.project_q = torch.from_numpy(cpu[2]).cuda()
        case.project_s = torch.from_numpy(cpu[3]).cuda()
        metadata.append(dict(seed=case.seed,operand_sha256=case.hashes,
            **{name+'_range':e008.allocation(tensor) for name,tensor in
               [('packed',case.packed),('scales',case.scales),('wh',case.wh),('output',case.c),('locks',case.locks_storage),
                ('scratch',case.tmp_storage),('activation',case.a),('project_q',case.project_q),('project_s',case.project_s)]},
            **{name+'_sha256':sha(tensor) for name,tensor in
               [('packed',case.packed),('scales',case.scales),('wh',case.wh),('project_q',case.project_q),('project_s',case.project_s)]}))
    assert all(case.a.data_ptr() % 16 == 0 for case in cases)  # else the tc launchers fall back
    e008.distinct([v[key+'_range'] for v in metadata for key in
                   ('packed','scales','wh','output','locks','scratch','project_q','project_s')]
                  +[metadata[0]['activation_range']])
    assert len({v['operand_sha256']['A'] for v in metadata}) == 1
    for key in ('packed_sha256','scales_sha256','wh_sha256'):
        assert len({v[key] for v in metadata}) == 4
    return cases,metadata


def main(args):
    assert not args.out.exists()
    assert hashlib.sha256(args.local.read_bytes()).hexdigest() == BODY
    bridge = e004.bridge(args.bridge)
    handle = bridge.e004_handle()
    lib = ct.CDLL(str(args.tc.resolve()))
    for fn in (lib.e050_tc,lib.e050_tc_ladder):
        fn.argtypes = [ct.c_void_p]*4+[ct.c_int]*4+[ct.c_void_p]
        fn.restype = ct.c_int
    cublas = ct.CDLL('libcublas.so.13')
    cublas.cublasSetMathMode.argtypes = [ct.c_void_p,ct.c_int]
    cublas.cublasGetMathMode.argtypes = [ct.c_void_p,ct.POINTER(ct.c_int)]
    mode = ct.c_int()
    assert cublas.cublasSetMathMode(handle,16) == 0
    stream = torch.cuda.Stream()
    tc_sha = hashlib.sha256(args.tc.read_bytes()).hexdigest()
    print('ENV',json.dumps(dict(body_sha256=BODY,tc_sha256=tc_sha,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),stream=int(stream.cuda_stream))),flush=True)
    with torch.cuda.stream(stream),args.out.open('x') as out:
        for k,n in args.shapes:
            cases,metadata = prepare(args,k,n)
            for m in args.ms:
                resources = [case.call('A',m,True) for case in cases]
                for position,name in enumerate(order(k,n,m,args.series)):
                    if name not in args.implementations:
                        continue
                    def call(case):
                        if name == 'marlin':
                            case.call('A',m)
                        elif name in ('tc','tc_ladder'):
                            fn = lib.e050_tc if name == 'tc' else lib.e050_tc_ladder
                            assert fn(case.a.data_ptr(),case.project_q.data_ptr(),case.project_s.data_ptr(),
                                case.c.data_ptr(),m,n,k,128,stream.cuda_stream) == 0
                        else:
                            bridge.e004_launch(5,case.a.data_ptr(),None,None,case.wh.data_ptr(),
                                case.wh.data_ptr(),case.c.data_ptr(),m,n,k,stream.cuda_stream,handle)
                    def launch():
                        for case in cases:
                            call(case)
                    kernels = []
                    if args.trace:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA]) as prof:
                            call(cases[0])
                            torch.cuda.synchronize()
                        kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                        expect_kernel(name,m,kernels)
                        print('KERNEL',name,m,k,n,kernels,flush=True)
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
                        assert verify() == checks
                    row = dict(K=k,N=n,M=m,capacity=CAPACITY,matrices=4,group=128,seeds=SEEDS,activation_seed=SEEDS[0],
                        config=name,series=args.series,order=position,method='complete_cycle_graph_events',
                        policy='no_artificial_flush',warmup=args.warmup,iters=args.iters,body_sha256=BODY,
                        tc_sha256=tc_sha,cublas_math_mode=mode.value,resources=resources if name == 'marlin' else [],
                        traced_kernels=kernels,allocations=metadata,activation_prefix_sha256=sha(cases[0].a[:m]),
                        framework_allocated_bytes=torch.cuda.memory_allocated(),framework_reserved_bytes=torch.cuda.memory_reserved(),
                        active_weight_bytes=4*(2*k*n if name == 'cublas_dq' else k*n//2+k*n//64),
                        l2_bytes=bridge.e004_l2(),lock_reset=True,workspace_guards=True,output_tail_untouched=True,
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
        if any(v in x for v in ('libe005','libe004','libe050','libcublas','libcudart','libtorch'))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source','local','wheel','bridge','tc','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shapes',nargs='+',default=['4096,11008','11008,4096','4096,4096'])
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
