#!/usr/bin/env python3
"""E049: copy of E045's four-matrix cycle with `tc_m32` as a fourth path, plus E011-style profile mode.

Timing/validation mode is E045's main loop with four paths. --profile runs ONE implementation/shape/M:
validation, 10 direct complete-cycle warm-ups, then a 100-cycle event graph between cudaProfilerStart/Stop
(ncu --launch-skip 399 --launch-count 1 selects the fourth GEMM of the last cycle; one kernel per GEMM)."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

# E045 harness (prepare: E010 operands plus project-layout Q/S), imported unchanged.
spec = importlib.util.spec_from_file_location('e045',Path(__file__).resolve().parents[1]/'e045/bench.py')
e045 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e045)
e010, e008, e005, e004 = e045.e010, e045.e008, e045.e005, e045.e004
BODY, CAPACITY, SHAPES, MS, sha = e045.BODY, e045.CAPACITY, e045.SHAPES, e045.MS, e045.sha
NAMES = ['marlin','tc','tc_m32','cublas_dq']
WARMUP, CYCLES = 10, 100


def order(k,n,m,series):
    shift = ((SHAPES.index((k,n)) if (k,n) in SHAPES else 0) + (MS.index(m) if m in MS else 0)) % 4
    names = NAMES[shift:]+NAMES[:shift]
    return names if series == 1 else names[::-1]


def expect_kernel(name,m,kernels):
    if name == 'marlin':
        assert len(kernels) == 1 and kernels[0].startswith('void e005_marlin::Marlin<'),kernels
    elif name == 'tc' or (name == 'tc_m32' and m < 32):
        # tc_m32 dispatches M < 32 to the unchanged tc launcher (E048 predeclaration).
        assert len(kernels) == 1 and 'w4a16_tc_kernel<true' in kernels[0],kernels
    elif name == 'tc_m32':
        assert len(kernels) == 1 and 'w4a16_tc_m32_kernel<false>' in kernels[0],kernels


def main(args):
    # Application replay reruns the whole process once per pass: profile mode appends one row per pass.
    assert args.profile or not args.out.exists()
    assert hashlib.sha256(args.local.read_bytes()).hexdigest() == BODY
    bridge = e004.bridge(args.bridge)
    handle = bridge.e004_handle()
    lib = ct.CDLL(str(args.tc.resolve()))
    for fn in (lib.e049_tc,lib.e049_tc_m32):
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
    with torch.cuda.stream(stream),args.out.open('a' if args.profile else 'x') as out:
        for k,n in args.shapes:
            cases,metadata = e045.prepare(args,k,n)
            for m in args.ms:
                resources = [case.call('A',m,True) for case in cases]
                for position,name in enumerate(order(k,n,m,args.series)):
                    if name not in args.implementations:
                        continue
                    def call(case):
                        if name == 'marlin':
                            case.call('A',m)
                        elif name in ('tc','tc_m32'):
                            fn = lib.e049_tc if name == 'tc' else lib.e049_tc_m32
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
                    row = dict(K=k,N=n,M=m,capacity=CAPACITY,matrices=4,group=128,seeds=[3,4,5,6],
                        config=name,series=args.series,order=position,method='complete_cycle_graph_events',
                        policy='no_artificial_flush',warmup=args.warmup,iters=args.iters,body_sha256=BODY,
                        tc_sha256=tc_sha,cublas_math_mode=mode.value,resources=resources if name == 'marlin' else [],
                        traced_kernels=kernels,allocations=metadata,activation_prefix_sha256=sha(cases[0].a[:m]),
                        framework_allocated_bytes=torch.cuda.memory_allocated(),framework_reserved_bytes=torch.cuda.memory_reserved(),
                        active_weight_bytes=4*(2*k*n if name == 'cublas_dq' else k*n//2+k*n//64),
                        l2_bytes=bridge.e004_l2(),lock_reset=True,workspace_guards=True,output_tail_untouched=True,
                        direct_cycles=3,poisoned_graph_cycles=3,validation=checks)
                    if args.profile:
                        # E011 profile workload: ten direct warm-ups, then 100 event-delimited graph cycles.
                        for _ in range(WARMUP):
                            launch()
                        torch.cuda.synchronize()
                        starts = [torch.cuda.Event(enable_timing=True,external=True) for _ in range(CYCLES)]
                        stops = [torch.cuda.Event(enable_timing=True,external=True) for _ in range(CYCLES)]
                        measured = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(measured,stream=stream):
                            for start,stop in zip(starts,stops):
                                start.record(); launch(); stop.record()
                        poison(); torch.cuda.synchronize()
                        runtime = ct.CDLL('libcudart.so.13')
                        assert runtime.cudaProfilerStart() == 0
                        measured.replay()
                        torch.cuda.synchronize()
                        assert runtime.cudaProfilerStop() == 0
                        row.update(profile_warmup=WARMUP,profile_cycles=CYCLES,launch_skip=CYCLES*4-1,launch_count=1,
                                   post_profile_validation=verify())
                        assert row['post_profile_validation'] == checks
                    elif args.iters:
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
        if any(v in x for v in ('libe005','libe004','libe049','libcublas','libcudart','libtorch'))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source','local','wheel','bridge','tc','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shapes',nargs='+',default=['4096,11008','11008,4096'])
    p.add_argument('--ms',nargs='+',type=int,default=MS)
    p.add_argument('--implementations',nargs='+',choices=NAMES,default=NAMES)
    p.add_argument('--series',type=int,choices=[1,2],default=1)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iters',type=int,default=100)
    p.add_argument('--trace',action='store_true')
    p.add_argument('--profile',action='store_true')
    args = p.parse_args()
    args.configs = ['A']
    args.shapes = [tuple(map(int,x.split(','))) for x in args.shapes]
    assert len(set(args.ms)) == len(args.ms) and all(0 < m <= CAPACITY for m in args.ms)
    assert args.iters >= 0 and args.warmup >= 0 and not (args.trace and args.iters)
    # One profiled process = one case, so the launch window contains only that path's 400 GEMMs.
    assert not args.profile or (len(args.shapes) == 1 and len(args.ms) == 1 and len(args.implementations) == 1
                                and args.iters == 0 and not args.trace)
    main(args)
