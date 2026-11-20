#!/usr/bin/env python3
"""E058: E050's 4096x4096 cyclic cells repeated with 4, 8 and 16 distinct matrices per cycle, so the rotating
4-bit working set (4/8/16 x 8.65 MB = 34.6/69.2/138.4 MB) is about, 2x and 4x the 32 MiB L2.

Copy of E050's bench.py (itself E045/E049's timing loop). Changes, all in the harness only: one shape (K=N=4096),
the matrix count is a loop level (--counts), 16 fresh weight seeds 11-26 (count c uses the first c of them,
nested), the activation of seed 11 (first seed, as E050/E010). Same paths, binaries, checks and timing method:
tc, tc_ladder (E050 library), corrected Marlin A (E006 body), cuBLAS on pre-materialized W_hat in math mode 16;
10 complete-cycle warm-ups, 100 graph-event cycle samples; per-GEMM time = cycle median / count."""
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
SEEDS = list(range(11,27))  # fresh: E001-E051 used e004_generate seeds 1-10 only
COUNTS = [4,8,16]
NAMES = ['marlin','tc','tc_ladder','cublas_dq']
E050_TC_SHA = 'a24a78fad127230f04a867e4a78e7a2a0453a6d4f0db95ea35900c7adaa971fe'


def order(count,m,series):
    # 3 counts x 4 M: each rotation offset occurs three times per series (as E050's 3 shapes x 4 M).
    shift = ((COUNTS.index(count) if count in COUNTS else 0) + (MS.index(m) if m in MS else 0)) % 4
    names = NAMES[shift:]+NAMES[:shift]
    return names if series == 1 else names[::-1]


def prepare(args,k,n,count):
    """E050's prepare() with the seed list cut to `count` matrices."""
    seeds = SEEDS[:count]
    cases = [e005.Case(args,k,n,[CAPACITY],seed,'mixed') for seed in seeds]
    metadata = []
    for case in cases:
        case.a = cases[0].a
        case.hashes['A'] = cases[0].hashes['A']
        codes = e004.unpack_marlin(case.packed.cpu().numpy(),k,n,case.ns['get_weight_perm'](4).numpy())
        inv = np.argsort(np.array(case.ns['get_scale_perms']()[0]))
        scales = case.scales.cpu().numpy().reshape(-1,64)[:,inv].reshape(k//128,n)
        wh = ((codes.astype(np.float32)-8)*scales.repeat(128,axis=0).astype(np.float32)).astype(np.float16)
        assert hashlib.sha256(wh.tobytes()).hexdigest() == case.hashes['W_hat']
        if case.seed != seeds[0]:
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
        assert len({v[key] for v in metadata}) == count
    return cases,metadata


def main(args):
    assert not args.out.exists()
    assert hashlib.sha256(args.local.read_bytes()).hexdigest() == BODY
    tc_sha = hashlib.sha256(args.tc.read_bytes()).hexdigest()
    assert tc_sha == E050_TC_SHA, 'not the E050 library'
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
    k,n = args.shape
    print('ENV',json.dumps(dict(body_sha256=BODY,tc_sha256=tc_sha,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),stream=int(stream.cuda_stream))),flush=True)
    with torch.cuda.stream(stream),args.out.open('x') as out:
        # One preparation of the largest count; smaller counts cycle over its first matrices (nested seeds).
        cases_all,metadata_all = prepare(args,k,n,max(args.counts))
        free,total = torch.cuda.mem_get_info()
        memory = dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),free=free,total=total)
        print('MEMORY',json.dumps(memory),flush=True)
        for count in args.counts:
            cases, metadata = cases_all[:count], metadata_all[:count]
            for m in args.ms:
                resources = [case.call('A',m,True) for case in cases]
                for position,name in enumerate(order(count,m,args.series)):
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
                        print('KERNEL',name,m,count,kernels,flush=True)
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
                    row = dict(K=k,N=n,M=m,capacity=CAPACITY,matrices=count,group=128,seeds=SEEDS[:count],activation_seed=SEEDS[0],
                        config=name,series=args.series,order=position,method='complete_cycle_graph_events',
                        policy='no_artificial_flush',warmup=args.warmup,iters=args.iters,body_sha256=BODY,
                        tc_sha256=tc_sha,cublas_math_mode=mode.value,resources=resources if name == 'marlin' else [],
                        traced_kernels=kernels,allocations=metadata,activation_prefix_sha256=sha(cases[0].a[:m]),
                        framework_allocated_bytes=torch.cuda.memory_allocated(),framework_reserved_bytes=torch.cuda.memory_reserved(),
                        memory_after_prepare=memory,
                        active_weight_bytes=count*(2*k*n if name == 'cublas_dq' else k*n//2+k*n//64),
                        l2_bytes=bridge.e004_l2(),lock_reset=True,workspace_guards=True,output_tail_untouched=True,
                        direct_cycles=3,poisoned_graph_cycles=3,validation=checks)
                    if args.iters:
                        poison()
                        row.update(e004.timed(launch,None,args.warmup,args.iters,cases[0].c[:m]))
                        row['post_timing_validation'] = verify()
                        row['median_per_matrix_us'] = row['median_us']/count
                    out.write(json.dumps(row)+'\n'); out.flush()
                    print('PASS',count,m,name,row.get('median_per_matrix_us'),flush=True)
                    del graph
        del cases_all
    bridge.e004_destroy(handle)
    print('MAPPED_LIBRARIES','\n'.join(x for x in Path('/proc/self/maps').read_text().splitlines()
        if any(v in x for v in ('libe005','libe004','libe050','libcublas','libcudart','libtorch'))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source','local','wheel','bridge','tc','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shape',default='4096,4096')
    p.add_argument('--counts',nargs='+',type=int,default=COUNTS)
    p.add_argument('--ms',nargs='+',type=int,default=MS)
    p.add_argument('--implementations',nargs='+',choices=NAMES,default=NAMES)
    p.add_argument('--series',type=int,choices=[1,2],default=1)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iters',type=int,default=100)
    p.add_argument('--trace',action='store_true')
    args = p.parse_args()
    args.configs = ['A']
    args.shape = tuple(map(int,args.shape.split(',')))
    assert len(set(args.ms)) == len(args.ms) and all(0 < m <= CAPACITY for m in args.ms)
    assert len(set(args.counts)) == len(args.counts) and all(c in COUNTS for c in args.counts)
    assert args.iters >= 0 and args.warmup >= 0 and not (args.trace and args.iters)
    main(args)
