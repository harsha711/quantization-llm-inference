#!/usr/bin/env python3
"""E060: E006's original (control) and corrected (patched) Marlin bodies timed in E058's cyclic harness.

Copy of E058's bench.py method (E010's cycle; E004/E005/E008/E010 modules imported unchanged, E008/E010
bench.py hashes asserted in inner.sh; complete-cycle graph events, no artificial flush, 10 warm-ups, 100 cycle
samples, per-GEMM = cycle median / matrices). Changes, harness only: two Marlin arms = E006's matched binaries
(each opened with its own ctypes.CDLL; same e005_call entrypoint, mode 0 = corrected A host policy), three shapes
with fixed matrix counts (4096^2 x 16, 4096x11008 x 4, 11008x4096 x 4), M 1/4/16/64/128, fresh seeds 201-216,
no tc/cuBLAS arm. Pure logic (plan, hashes, address identity, order, per-GEMM) in scripts/e060/e060lib.py."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e010', HERE.parent/'e010/bench.py')
e010 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e010)
e008, e005, e004 = e010.e008, e010.e005, e010.e004
CAPACITY, sha = e010.CAPACITY, e010.sha
spec = importlib.util.spec_from_file_location('e060lib', HERE/'e060lib.py')
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)
assert e010.BODY == lib.ARM_SHA256['patched']  # E010's corrected body is E060's patched arm


def open_library(path):
    h = ct.CDLL(path)
    h.e005_call.argtypes = [ct.c_int,ct.c_int]+[ct.c_void_p]*6+[ct.c_int]*3+[ct.c_void_p,ct.POINTER(ct.c_int64)]
    h.e005_call.restype = ct.c_int
    return h


def marlin(h,case,m,dry=False):
    """E005 Case.call('A', ...) with an explicit library handle (mode 0 = corrected A host policy)."""
    rc = h.e005_call(0,int(dry),case.a.data_ptr(),case.packed.data_ptr(),case.scales.data_ptr(),case.c.data_ptr(),
                     case.tmp.data_ptr(),case.locks.data_ptr(),m,case.n,case.k,torch.cuda.current_stream().cuda_stream,
                     case.info)
    assert rc == 0,rc
    return dict(zip(e005.FIELDS,map(int,case.info)))


def prepare(args,k,n,seeds):
    """E058's prepare() without the tc/cuBLAS tensors: one shared activation (first seed), FP64 references."""
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
        del case.flush_buf
        metadata.append(dict(seed=case.seed,operand_sha256=case.hashes,
            **{name+'_range':e008.allocation(tensor) for name,tensor in
               [('packed',case.packed),('scales',case.scales),('output',case.c),('locks',case.locks_storage),
                ('scratch',case.tmp_storage),('activation',case.a)]},
            **{name+'_sha256':sha(tensor) for name,tensor in [('packed',case.packed),('scales',case.scales)]}))
    e008.distinct([v[key+'_range'] for v in metadata for key in ('packed','scales','output','locks','scratch')]
                  +[metadata[0]['activation_range']])
    assert len({v['operand_sha256']['A'] for v in metadata}) == 1
    for key in ('packed_sha256','scales_sha256'):
        assert len({v[key] for v in metadata}) == len(seeds)
    return cases,metadata


def main(args):
    assert not args.out.exists()
    paths = {arm: os.path.realpath(lib.library_path(arm,str(args.e006_root))) for arm in lib.ARMS}
    shas = {arm: lib.check_library(arm,paths[arm]) for arm in lib.ARMS}      # both hashes, always
    assert os.path.realpath(args.local) == paths['patched']                  # Case's own handle = patched
    bridge = e004.bridge(args.bridge)
    l2 = bridge.e004_l2()
    plan = {(p['K'],p['N']): p for p in lib.plan(l2)}                         # working set >= 2 x L2 asserted
    handles = {arm: open_library(paths[arm]) for arm in args.arms}
    maps = Path('/proc/self/maps').read_text()
    ranges = {arm: lib.library_ranges(maps,paths[arm]) for arm in args.arms}
    assert all(ranges[arm] for arm in args.arms),ranges
    stream = torch.cuda.Stream()
    print('ENV',json.dumps(dict(library_sha256=shas,library_path=paths,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),stream=int(stream.cuda_stream),l2_bytes=l2)),flush=True)
    with torch.cuda.stream(stream),args.out.open('x') as out:
        for k,n in args.shapes:
            in_plan = (k,n) in plan
            count = args.matrices or lib.COUNTS[(k,n)]
            if args.iters:
                assert in_plan and count == lib.COUNTS[(k,n)],'timing only with the predeclared shapes/counts'
            seeds = lib.SEEDS[:count]
            cases,metadata = prepare(args,k,n,seeds)
            free,total = torch.cuda.mem_get_info()
            memory = dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),free=free,total=total)
            print('MEMORY',k,n,json.dumps(memory),flush=True)
            for m in args.ms:
                resources = {arm: [marlin(handles[arm],case,m,True) for case in cases] for arm in args.arms}
                address = {arm: resources[arm][0]['kernel_address'] for arm in args.arms}
                for arm in args.arms:
                    assert len({r['kernel_address'] for r in resources[arm]}) == 1
                    assert lib.address_in(address[arm],ranges[arm]),(arm,hex(address[arm]))
                if len(args.arms) == 2:
                    lib.check_addresses(address,maps,paths)
                arms = lib.arm_order(k,n,m,args.series) if in_plan else list(lib.ARMS)
                for position,arm in enumerate(arms):
                    if arm not in args.arms:
                        continue
                    h = handles[arm]
                    def launch():
                        for case in cases:
                            marlin(h,case,m)
                    kernels = []
                    if args.trace:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA]) as prof:
                            marlin(h,cases[0],m)
                            torch.cuda.synchronize()
                        kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                        assert len(kernels) == 1 and kernels[0].startswith('void e005_marlin::Marlin<'),kernels
                        print('KERNEL',arm,k,n,m,kernels,flush=True)
                    def poison():
                        for case in cases:
                            case.c.fill_(float('nan'))
                    def verify():
                        checks = []
                        for case in cases:
                            case.check_workspace()
                            assert torch.all(torch.isnan(case.c[m:])).item(),'output row-tail overwritten'
                            checks.append(dict(**e005.stats(case.c[:m],case.ref[:m]),output_sha256=sha(case.c[:m])))
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
                    row = dict(K=k,N=n,M=m,capacity=CAPACITY,matrices=count,group=128,seeds=seeds,
                        activation_seed=seeds[0],arm=arm,series=args.series,order=position,
                        method='complete_cycle_graph_events',policy='no_artificial_flush',warmup=args.warmup,
                        iters=args.iters,library_path=paths[arm],library_sha256=shas[arm],kernel_address=address[arm],
                        library_ranges=ranges[arm],resources=resources[arm][0],traced_kernels=kernels,
                        allocations=metadata,activation_prefix_sha256=sha(cases[0].a[:m]),memory_after_prepare=memory,
                        working_set_bytes=lib.working_set_bytes(k,n,count),l2_bytes=l2,lock_reset=True,
                        workspace_guards=True,output_tail_untouched=True,direct_cycles=3,poisoned_graph_cycles=3,
                        validation=checks)
                    if args.iters:
                        poison()
                        row.update(e004.timed(launch,None,args.warmup,args.iters,cases[0].c[:m]))
                        row['post_timing_validation'] = verify()
                        row['median_per_matrix_us'] = lib.per_gemm_us(row)
                    out.write(json.dumps(row)+'\n'); out.flush()
                    print('PASS',k,n,m,arm,row.get('median_per_matrix_us'),flush=True)
                    del graph
            del cases
            torch.cuda.empty_cache()
    print('MAPPED_LIBRARIES','\n'.join(x for x in Path('/proc/self/maps').read_text().splitlines()
        if any(v in x for v in ('libe005','libe004','libcudart','libtorch_cuda'))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source','local','wheel','bridge','e006_root','out'):
        p.add_argument('--'+key.replace('_','-'),dest=key,type=Path,required=True)
    p.add_argument('--shapes',nargs='+',default=[f'{k},{n}' for k,n in lib.SHAPES])
    p.add_argument('--matrices',type=int,default=0,help='override the count (untimed runs only)')
    p.add_argument('--ms',nargs='+',type=int,default=lib.MS)
    p.add_argument('--arms',nargs='+',choices=lib.ARMS,default=lib.ARMS)
    p.add_argument('--series',type=int,choices=[1,2],default=1)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iters',type=int,default=100)
    p.add_argument('--trace',action='store_true')
    args = p.parse_args()
    args.configs = ['A']
    args.shapes = [tuple(map(int,s.split(','))) for s in args.shapes]
    assert len(set(args.ms)) == len(args.ms) and all(0 < m <= CAPACITY for m in args.ms)
    assert args.iters >= 0 and args.warmup >= 0 and not (args.trace and args.iters)
    assert not (args.matrices and args.iters)
    main(args)
