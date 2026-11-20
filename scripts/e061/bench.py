#!/usr/bin/env python3
"""E061: cuBLAS on pre-materialized FP16 W_hat (cublas_dq, E004 bridge id 5, math mode 16) at K=4096, N=11008,
M 128/256/512, 4 matrices per cycle, fresh seeds 301-304, activation/output capacity 520 rows (512 + 8 guard rows).

Copy of E058's bench.py method (E010's cycle; E004/E005/E008/E010 modules imported unchanged; complete-cycle
graph events, no artificial flush, 10 warm-ups, 100 cycle samples, per-GEMM = cycle median / matrices).
Changes, harness only: one arm (cuBLAS), one shape, capacity 520 (8 guard rows past M512), M 128/256/512. Logic in e061lib.py."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e010', HERE.parent/'e010/bench.py')
e010 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e010)
e008, e005, e004, sha = e010.e008, e010.e005, e010.e004, e010.sha
spec = importlib.util.spec_from_file_location('e061lib', HERE/'e061lib.py')
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)
CAPACITY = lib.CAPACITY


def prepare(args,k,n):
    """E058's prepare() without the Marlin/tc tensors: W_hat on the GPU, one shared activation (first seed),
    FP64 references of all CAPACITY rows against that activation."""
    cases = [e005.Case(args,k,n,[CAPACITY],seed,'mixed') for seed in lib.SEEDS]
    metadata = []
    for case in cases:
        case.a = cases[0].a
        case.hashes['A'] = cases[0].hashes['A']
        codes = e004.unpack_marlin(case.packed.cpu().numpy(),k,n,case.ns['get_weight_perm'](4).numpy())
        inv = np.argsort(np.array(case.ns['get_scale_perms']()[0]))
        scales = case.scales.cpu().numpy().reshape(-1,64)[:,inv].reshape(k//128,n)
        wh = ((codes.astype(np.float32)-8)*scales.repeat(128,axis=0).astype(np.float32)).astype(np.float16)
        assert hashlib.sha256(wh.tobytes()).hexdigest() == case.hashes['W_hat']
        if case.seed != lib.SEEDS[0]:
            activation = case.a.cpu().numpy()
            case.bridge.e004_reference(activation.ctypes.data,wh.ctypes.data,CAPACITY,n,k,case.ref.ctypes.data)
        case.wh = torch.from_numpy(wh).cuda()
        del case.flush_buf
        metadata.append(dict(seed=case.seed,operand_sha256=case.hashes,
            **{name+'_range':e008.allocation(tensor) for name,tensor in
               [('wh',case.wh),('output',case.c),('activation',case.a)]},
            wh_sha256=sha(case.wh)))
    e008.distinct([v[key+'_range'] for v in metadata for key in ('wh','output')]+[metadata[0]['activation_range']])
    assert len({v['operand_sha256']['A'] for v in metadata}) == 1
    assert len({v['wh_sha256'] for v in metadata}) == lib.MATRICES
    return cases,metadata


def main(args):
    assert not args.out.exists()
    bridge = e004.bridge(args.bridge)
    handle = bridge.e004_handle()
    cublas = ct.CDLL('libcublas.so.13')
    cublas.cublasSetMathMode.argtypes = [ct.c_void_p,ct.c_int]
    cublas.cublasGetMathMode.argtypes = [ct.c_void_p,ct.POINTER(ct.c_int)]
    mode = ct.c_int()
    assert cublas.cublasSetMathMode(handle,16) == 0
    l2 = bridge.e004_l2()
    lib.check_plan(l2)                                   # FP16 working set >= 2 x L2
    stream = torch.cuda.Stream()
    k,n = lib.K,lib.N
    print('ENV',json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        stream=int(stream.cuda_stream),l2_bytes=l2,working_set_bytes=lib.working_set_bytes())),flush=True)
    with torch.cuda.stream(stream),args.out.open('x') as out:
        cases,metadata = prepare(args,k,n)
        free,total = torch.cuda.mem_get_info()
        memory = dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),free=free,total=total)
        print('MEMORY',json.dumps(memory),flush=True)
        for m in (lib.m_order(args.series) if args.iters else args.ms):
            def call(case):
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
                assert kernels, 'no CUDA kernel traced'
                print('KERNEL',m,kernels,flush=True)
            def poison():
                for case in cases:
                    case.c.fill_(float('nan'))
            def verify():
                checks = []
                for case in cases:
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
            row = dict(K=k,N=n,M=m,capacity=CAPACITY,matrices=lib.MATRICES,group=128,seeds=lib.SEEDS,
                activation_seed=lib.ACTIVATION_SEED,config=lib.CONFIG,series=args.series,
                method='complete_cycle_graph_events',policy='no_artificial_flush',warmup=args.warmup,iters=args.iters,
                cublas_math_mode=mode.value,traced_kernels=kernels,allocations=metadata,
                activation_prefix_sha256=sha(cases[0].a[:m]),memory_after_prepare=memory,
                working_set_bytes=lib.working_set_bytes(),l2_bytes=l2,output_tail_untouched=True,
                direct_cycles=3,poisoned_graph_cycles=3,validation=checks)
            if args.iters:
                poison()
                row.update(e004.timed(launch,None,args.warmup,args.iters,cases[0].c[:m]))
                row['post_timing_validation'] = verify()
                assert row['post_timing_validation'] == checks, 'outputs changed during timing'
                row['median_per_matrix_us'] = lib.per_gemm_us(row)
            out.write(json.dumps(row)+'\n'); out.flush()
            print('PASS',m,row.get('median_per_matrix_us'),flush=True)
            del graph
        del cases
    bridge.e004_destroy(handle)
    print('MAPPED_LIBRARIES','\n'.join(x for x in Path('/proc/self/maps').read_text().splitlines()
        if any(v in x for v in ('libe004','libcublas','libcudart','libtorch_cuda'))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source','local','wheel','bridge','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ms',nargs='+',type=int,default=lib.MS,help='untimed runs only; timing uses the series order')
    p.add_argument('--series',type=int,choices=[1,2],default=1)
    p.add_argument('--warmup',type=int,default=lib.WARMUP)
    p.add_argument('--iters',type=int,default=lib.SAMPLES)
    p.add_argument('--trace',action='store_true')
    args = p.parse_args()
    args.configs = ['A']
    assert all(0 < m <= CAPACITY for m in args.ms)
    assert args.iters in (0,lib.SAMPLES) and args.warmup in (0,lib.WARMUP) and not (args.trace and args.iters)
    main(args)
