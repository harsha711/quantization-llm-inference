#!/usr/bin/env python3
"""E059: the teammate's v3 (origin/samarth/bandwidth-roofline cd39962, unchanged) next to tc_ladder, corrected
Marlin A and cuBLAS on FP16 W_hat in a CACHE-VALID cyclic harness (rotating 4-bit working set >= 2 x L2), plus a
diagnostic split-K sweep of E047's unchanged tc_splitk kernel with S forced, and the teammate's read-flush
single-matrix method on the same operands in the same process.

Copy of E050's bench.py (E045/E049's timing loop: E010's cycle, live FP16 W_hat, guarded per-matrix Marlin
workspace/output, no artificial flush, graph-event complete-cycle samples; E004/E005/E008/E010 modules imported
unchanged). Harness changes: five shapes; per-shape matrix count c = max(4, ceil(2 x L2 / 4-bit bytes)); fresh
weight seeds 101..100+c (activation of seed 101); CAPACITY 16 (M <= 16); per-shape path list; per-GEMM time =
cycle median / c; after each row's cyclic timing, the same path on matrix 0 alone under the teammate's read-only
flush (their timed(): flush in warm-ups and before every sample, graph events around one launch)."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('e010',Path(__file__).resolve().parents[1]/'e010/bench.py')
e010 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e010)
e008, e005, e004 = e010.e008, e010.e005, e010.e004
BODY, sha = e010.BODY, e010.sha
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e059lib as L  # noqa: E402  (tested pure-Python helpers: sizing, balance rule, paths/order, forced-S call)
CAPACITY = 16
MS, SHAPES, COUNTS, BALANCE, SMS = L.MS, L.SHAPES, L.COUNTS, L.BALANCE, L.SMS
SEED0 = 101
L2 = L.fc.L2_BYTES
paths, order, balance = L.paths, L.order, L.balance


def weight_bytes(k,n):
    return k*n//2 + (k//128)*n*2


def expect_kernel(name,kernels):
    one = lambda s: len(kernels) == 1 and s in kernels[0]
    if name == 'marlin':
        assert len(kernels) == 1 and kernels[0].startswith('void e005_marlin::Marlin<'),kernels
    elif name == 'tc_ladder':     # M < 64: the unchanged tc launcher
        assert one('w4a16_tc_kernel<true'),kernels
    elif name in ('v1','v2','v3'):
        assert one(f'w4a16_{name}_kernel'),kernels
    elif name.startswith('splitk_s'):
        assert one('w4a16_tc_splitk_kernel'),kernels


def prepare(args,k,n,c):
    seeds = list(range(SEED0,SEED0+c))
    cases = []
    for seed in seeds:
        case = e005.Case(args,k,n,[CAPACITY],seed,'mixed')
        del case.flush_buf            # E050 deletes it too; here per case so 56 x 64 MiB never coexist
        cases.append(case)
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
    assert all(case.a.data_ptr() % 16 == 0 for case in cases)
    e008.distinct([v[key+'_range'] for v in metadata for key in
                   ('packed','scales','wh','output','locks','scratch','project_q','project_s')]
                  +[metadata[0]['activation_range']])
    assert len({v['operand_sha256']['A'] for v in metadata}) == 1
    for key in ('packed_sha256','scales_sha256','wh_sha256'):
        assert len({v[key] for v in metadata}) == c
    return cases,metadata,seeds


def timed_readflush(launch,flush,warmup,iters,output):
    """The teammate's timed() (scripts/ours/marlin_compare.py at cd39962), plus raw samples."""
    for _ in range(warmup):
        flush(); launch()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True,external=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True,external=True) for _ in range(iters)]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=torch.cuda.current_stream()):
        for a,z in zip(starts,stops):
            flush(); a.record(); launch(); z.record()
    output.fill_(float('nan'))
    g.replay()
    torch.cuda.synchronize()
    us = [a.elapsed_time(z)*1000 for a,z in zip(starts,stops)]
    ordered = sorted(us)
    del g
    return dict(samples_us=us,median_us=float(np.median(us)),mean_us=float(np.mean(us)),min_us=ordered[0],
                p90_us=ordered[math.floor(0.9*(iters-1))],max_us=ordered[-1],warmup=warmup,iters=iters)


def main(args):
    assert not args.out.exists()
    assert not os.environ.get('W4A16_V1_SPLITS') and not os.environ.get('W4A16_V2_SPLITS')
    assert hashlib.sha256(args.local.read_bytes()).hexdigest() == BODY
    bridge = e004.bridge(args.bridge)
    handle = bridge.e004_handle()
    assert bridge.e004_l2() == L2 == 33554432 and torch.cuda.get_device_properties(0).multi_processor_count == SMS
    lib = ct.CDLL(str(args.lib.resolve()))
    for fn in (lib.e050_tc_ladder,lib.e059_v1,lib.e059_v2,lib.e059_v3):
        fn.argtypes = [ct.c_void_p]*4+[ct.c_int]*4+[ct.c_void_p]
        fn.restype = ct.c_int
    lib.e059_tc_splitk.argtypes = [ct.c_void_p]*4+[ct.c_int]*5+[ct.c_void_p]
    lib.e059_tc_splitk.restype = ct.c_int
    lib.e059_tc_splitk_last_s.restype = ct.c_int
    lib.e059_v2_fast_path.argtypes = [ct.c_void_p]*4+[ct.c_int]*2
    for fn in (lib.e059_v2_splits,lib.e059_v1_splits):
        fn.argtypes = [ct.c_int]*3
    lib.e059_flush_read.argtypes = [ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_void_p]
    lib.e059_flush_read.restype = ct.c_int
    cublas = ct.CDLL('libcublas.so.13')
    cublas.cublasSetMathMode.argtypes = [ct.c_void_p,ct.c_int]
    cublas.cublasGetMathMode.argtypes = [ct.c_void_p,ct.POINTER(ct.c_int)]
    mode = ct.c_int()
    assert cublas.cublasSetMathMode(handle,16) == 0
    stream = torch.cuda.Stream()
    lib_sha = hashlib.sha256(args.lib.read_bytes()).hexdigest()
    flush_buf = torch.empty(2*L2,dtype=torch.uint8,device='cuda')      # never written (their method)
    sink = torch.zeros(1,dtype=torch.int32,device='cuda')
    occupancy = lib.e059_tc_splitk_occupancy()
    print('ENV',json.dumps(dict(body_sha256=BODY,lib_sha256=lib_sha,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),stream=int(stream.cuda_stream),splitk_occupancy=occupancy)),flush=True)
    def flush():
        assert lib.e059_flush_read(flush_buf.data_ptr(),flush_buf.numel(),sink.data_ptr(),stream.cuda_stream) == 0
    failures = []
    with torch.cuda.stream(stream),args.out.open('x') as out:
        for k,n in args.shapes:
            c = args.count or L.matrix_count(k,n)
            assert args.count or c == COUNTS[(k,n)]
            for m in MS:
                assert (k,n) not in BALANCE or balance(k,n,m) == BALANCE[(k,n)]
            cases,metadata,seeds = prepare(args,k,n,c)
            for m in args.ms:
                resources = [case.call('A',m,True) for case in cases]
                cell = {}
                for position,name in enumerate(order(k,n,m,args.series,args.extra)):
                    if name not in args.implementations and name not in args.extra:
                        continue
                    try:
                        info = {}
                        if name in ('v1','v2','v3'):
                            info = dict(fast_path=bool(lib.e059_v2_fast_path(cases[0].a.data_ptr(),cases[0].project_q.data_ptr(),
                                            cases[0].project_s.data_ptr(),cases[0].c.data_ptr(),n,k)),
                                        v2_splits=lib.e059_v2_splits(m,n,k),v1_splits=lib.e059_v1_splits(m,n,k))
                            assert name == 'v1' or info['fast_path']
                        if name.startswith('splitk_s'):
                            s = int(name[8:])
                            info = dict(S=s,balance_rule_S=BALANCE[(k,n)],is_balance=s == BALANCE[(k,n)],
                                        chunks=k//64,tiles=(n//32)*((m+15)//16))
                        def call(case):
                            if name == 'marlin':
                                case.call('A',m)
                            elif name == 'cublas_dq':
                                bridge.e004_launch(5,case.a.data_ptr(),None,None,case.wh.data_ptr(),
                                    case.wh.data_ptr(),case.c.data_ptr(),m,n,k,stream.cuda_stream,handle)
                            else:
                                ptrs = (case.a.data_ptr(),case.project_q.data_ptr(),case.project_s.data_ptr(),case.c.data_ptr(),m,n,k,128)
                                if name == 'tc_ladder':
                                    rc = lib.e050_tc_ladder(*ptrs,stream.cuda_stream)
                                elif name.startswith('splitk_s'):
                                    L.launch_splitk(lib,name,ptrs,stream.cuda_stream)   # asserts grid z == S
                                    rc = 0
                                else:
                                    rc = getattr(lib,'e059_'+name)(*ptrs,stream.cuda_stream)
                                assert rc == 0,(name,rc)
                        def launch():
                            for case in cases:
                                call(case)
                        def poison():
                            for case in cases:
                                case.c.fill_(float('nan'))
                        def verify(subset=None):
                            checks = []
                            for case in (subset or cases):
                                case.check_workspace()
                                assert torch.all(torch.isnan(case.c[m:])).item(),'output row-tail overwritten'
                                checks.append(dict(**e005.stats(case.c[:m],case.ref[:m]),output_sha256=sha(case.c[:m])))
                            assert cublas.cublasGetMathMode(handle,ct.byref(mode)) == 0 and mode.value == 16
                            return checks
                        # First call outside any capture (grows the v1-v3 / split-K workspaces before capture).
                        poison(); launch(); torch.cuda.synchronize()
                        if name.startswith('splitk_s'):
                            info['last_s'] = lib.e059_tc_splitk_last_s()
                        # Independent re-check of matrix 0 (all M rows) with the tested pure-Python CONTRACT bound.
                        info['independent_violations'] = L.violations(cases[0].c[:m].float().cpu().flatten().tolist(),
                                                                      cases[0].ref[:m].ravel().tolist())
                        assert info['independent_violations'] == 0,(name,info['independent_violations'])
                        kernels = []
                        if args.trace:
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                                call(cases[0])
                                torch.cuda.synchronize()
                            kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                            expect_kernel(name,kernels)
                            print('KERNEL',name,m,k,n,kernels,flush=True)
                        for _ in range(3):
                            poison(); launch(); torch.cuda.synchronize()
                            checks = verify()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=stream):
                            launch()
                        for _ in range(3):
                            poison(); graph.replay(); torch.cuda.synchronize()
                            assert verify() == checks
                        del graph
                        row = dict(K=k,N=n,M=m,capacity=CAPACITY,matrices=c,group=128,seeds=seeds,activation_seed=seeds[0],
                            config=name,series=args.series,order=position,method='complete_cycle_graph_events',
                            policy='no_artificial_flush',warmup=args.warmup,iters=args.iters,body_sha256=BODY,
                            lib_sha256=lib_sha,cublas_math_mode=mode.value,info=info,
                            resources=resources if name == 'marlin' else [],traced_kernels=kernels,allocations=metadata,
                            activation_prefix_sha256=sha(cases[0].a[:m]),
                            framework_allocated_bytes=torch.cuda.memory_allocated(),framework_reserved_bytes=torch.cuda.memory_reserved(),
                            active_weight_bytes=c*(2*k*n if name == 'cublas_dq' else weight_bytes(k,n)),
                            l2_bytes=L2,lock_reset=True,workspace_guards=True,output_tail_untouched=True,
                            direct_cycles=3,poisoned_graph_cycles=3,validation=checks)
                        if args.iters:
                            poison()
                            row.update(e004.timed(launch,None,args.warmup,args.iters,cases[0].c[:m]))
                            row['post_timing_validation'] = verify()
                            assert row['post_timing_validation'] == checks
                            row['median_per_matrix_us'] = row['median_us']/c
                        if args.rf_iters:
                            poison()
                            rf = timed_readflush(lambda: call(cases[0]),flush,args.rf_warmup,args.rf_iters,cases[0].c[:m])
                            rf['validation'] = verify(cases[:1])
                            assert rf['validation'] == checks[:1]
                            rf['flush_bytes'] = flush_buf.numel()
                            row['readflush'] = rf
                        cell[name] = [v['output_sha256'] for v in checks]
                        out.write(json.dumps(row)+'\n'); out.flush()
                        print('PASS',m,k,n,name,c,row.get('median_per_matrix_us'),row.get('readflush',{}).get('median_us'),flush=True)
                    except AssertionError as error:   # contract/validation failure: record, stop this path only
                        torch.cuda.synchronize()
                        failures.append((k,n,m,name))
                        out.write(json.dumps(dict(K=k,N=n,M=m,matrices=c,config=name,series=args.series,order=position,
                            contract_failure=repr(error)[:4000]))+'\n'); out.flush()
                        print('FAIL',m,k,n,name,repr(error)[:400],flush=True)
                # Bitwise: S = 1 split kernel == tc (tc_ladder runs tc at M < 64); v3 == v2 (the teammate's claim).
                if 'splitk_s1' in cell and 'tc_ladder' in cell:
                    equal = cell['splitk_s1'] == cell['tc_ladder']
                    # Predeclared: a mismatch (e.g. a -0 output) is RECORDED, never fatal (the analyzer lists it);
                    # contract failures (the failures list) still fail the job.
                    print('BITWISE splitk_s1 == tc_ladder',k,n,m,equal,flush=True)
                if 'v2' in cell and 'v3' in cell:
                    print('BITWISE v3 == v2',k,n,m,cell['v2'] == cell['v3'],flush=True)
            del cases
    bridge.e004_destroy(handle)
    if failures:
        print('FAILURES',failures,flush=True)
        raise SystemExit(3)
    print('MAPPED_LIBRARIES','\n'.join(x for x in Path('/proc/self/maps').read_text().splitlines()
        if any(v in x for v in ('libe005','libe004','libe059','libcublas','libcudart','libtorch'))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source','local','wheel','bridge','lib','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shapes',nargs='+',default=[f'{k},{n}' for k,n in SHAPES])
    p.add_argument('--ms',nargs='+',type=int,default=MS)
    p.add_argument('--implementations',nargs='+',default=None)
    p.add_argument('--extra',nargs='*',choices=['v1','v2'],default=[])   # untimed contract checks only
    p.add_argument('--count',type=int,default=0)                        # override (prerequisite/sanitizers only)
    p.add_argument('--series',type=int,choices=[1,2],default=1)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iters',type=int,default=100)
    p.add_argument('--rf-warmup',type=int,default=10)
    p.add_argument('--rf-iters',type=int,default=200)
    p.add_argument('--trace',action='store_true')
    args = p.parse_args()
    args.configs = ['A']
    args.shapes = [tuple(map(int,x.split(','))) for x in args.shapes]
    assert all(s in SHAPES for s in args.shapes) and all(m in MS for m in args.ms)
    allp = {x for k,n in SHAPES for x in paths(k,n)}
    args.implementations = args.implementations or sorted(allp)
    assert set(args.implementations) <= allp
    assert args.iters >= 0 and args.warmup >= 0 and args.rf_iters >= 0
    assert not (args.trace and (args.iters or args.rf_iters)) and not (args.extra and (args.iters or args.rf_iters))
    assert not (args.count and args.iters > 20)          # an overridden count is never a measurement
    main(args)
