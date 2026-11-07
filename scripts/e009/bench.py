#!/usr/bin/env python3
"""E009: unchanged corrected A versus W_hat cuBLAS and timed dequant + cuBLAS."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('e008', Path(__file__).resolve().parents[1]/'e008/bench.py')
e008 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e008)
e004 = e008.e005.e004
NAMES = ['marlin', 'cublas_dq', 'unfused']
BODY = 'e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7'


def sha(tensor):
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def order(k, n, count, series):
    cell = [(a,b,c) for a,b in [(4096,11008),(11008,4096)] for c in (1,2,4)]
    shift = cell.index((k,n,count)) % 3 if (k,n,count) in cell else (1,2,4).index(count)
    names = NAMES[shift:]+NAMES[:shift]
    return names if series == 1 else names[::-1]


def prepare(args, k, n):
    cases, metadata = e008.prepare(args,k,n)
    for case, item in zip(cases,metadata):
        cpu = [np.empty((4,k),np.float16),np.empty((k,n),np.float16),
               np.empty((k//8,n),np.uint32),np.empty((k//128,n),np.float16),np.empty((k,n),np.float16)]
        assert case.bridge.e004_generate(4,n,k,case.seed,*[x.ctypes.data for x in cpu]) == 0
        for key, array in zip(['Q','S','W_hat'],cpu[2:]):
            assert hashlib.sha256(array.tobytes()).hexdigest() == case.hashes[key]
        case.project_q = torch.from_numpy(cpu[2]).cuda()
        case.project_s = torch.from_numpy(cpu[3]).cuda()
        case.wh = torch.from_numpy(cpu[4]).cuda()
        for name, tensor in [('project_q',case.project_q),('project_s',case.project_s),('wh',case.wh)]:
            item[name+'_range'] = e008.allocation(tensor)
            item[name+'_sha256'] = sha(tensor)
        del cpu
    temporary_storage = torch.full((k*n+32,),123.,dtype=torch.float16,device='cuda')
    temporary = temporary_storage[16:-16].view(k,n)
    ranges = [item[key] for item in metadata for key in
              ('packed_range','scales_range','output_range','locks_range','scratch_range',
               'project_q_range','project_s_range','wh_range')]
    e008.distinct(ranges+[e008.allocation(temporary_storage),metadata[0]['activation_range']])
    return cases, metadata, temporary_storage, temporary


def main(args):
    assert not args.out.exists(), args.out
    assert hashlib.sha256(args.local.read_bytes()).hexdigest() == BODY
    bridge = e004.bridge(args.bridge)
    handle = bridge.e004_handle()
    cublas = ct.CDLL('libcublas.so.13')
    cublas.cublasSetMathMode.argtypes = [ct.c_void_p,ct.c_int]
    cublas.cublasGetMathMode.argtypes = [ct.c_void_p,ct.POINTER(ct.c_int)]
    mode = ct.c_int()
    assert cublas.cublasSetMathMode(handle,16) == 0
    assert cublas.cublasGetMathMode(handle,ct.byref(mode)) == 0 and mode.value == 16
    stream = torch.cuda.Stream()
    print('ENV',json.dumps(dict(body_sha256=BODY,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),stream=int(stream.cuda_stream),cublas_math_mode=mode.value)),flush=True)
    with torch.cuda.stream(stream), args.out.open('x') as out:
        for k,n in args.shapes:
            cases, metadata, temp_storage, temp = prepare(args,k,n)
            def bridge_call(case, selector):
                bridge.e004_launch(selector,case.a.data_ptr(),case.project_q.data_ptr(),case.project_s.data_ptr(),
                    case.wh.data_ptr(),temp.data_ptr() if selector in (6,7) else case.wh.data_ptr(),
                    case.c.data_ptr(),4,n,k,stream.cuda_stream,handle)
            def guards():
                for case in cases:
                    case.check_workspace()
                assert torch.all(temp_storage[:16] == 123.).item()
                assert torch.all(temp_storage[-16:] == 123.).item()
            # Check every reconstruction individually before composing cycles.
            for case in cases:
                temp.fill_(float('nan'))
                bridge_call(case,7)
                torch.cuda.synchronize()
                assert sha(temp) == case.hashes['W_hat']
                guards()
            for count in args.counts:
                active = cases[:count]
                for position, name in enumerate(order(k,n,count,args.series)):
                    if name not in args.implementations:
                        continue
                    resources = [case.call('A',4,True) for case in active] if name == 'marlin' else []
                    def launch():
                        for case in active:
                            if name == 'marlin':
                                case.call('A',4)
                            else:
                                bridge_call(case,5 if name == 'cublas_dq' else 6)
                    def poison():
                        for case in active:
                            case.c.fill_(float('nan'))
                        if name == 'unfused':
                            temp.fill_(float('nan'))
                    def verify():
                        guards()
                        result = e008.verify(active)
                        if name == 'unfused':
                            assert sha(temp) == active[-1].hashes['W_hat']
                        assert cublas.cublasGetMathMode(handle,ct.byref(mode)) == 0 and mode.value == 16
                        return result
                    for _ in range(3):
                        poison()
                        launch()
                        torch.cuda.synchronize()
                        checks = verify()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        launch()
                    for _ in range(3):
                        poison()
                        graph.replay()
                        torch.cuda.synchronize()
                        checks = verify()
                    live_ranges = [item[key] for item in metadata for key in
                        ('packed_range','scales_range','output_range','locks_range','scratch_range',
                         'project_q_range','project_s_range','wh_range')]
                    live_ranges += [e008.allocation(temp_storage),metadata[0]['activation_range']]
                    row = dict(K=k,N=n,M=4,group=128,seeds=[3,4,5,6],config=name,series=args.series,
                        order=position,matrices=count,method='complete_cycle_graph_events',policy='no_artificial_flush',
                        warmup=args.warmup,iters=args.iters,body_sha256=BODY,cublas_math_mode=mode.value,
                        resources=resources,allocations=metadata,temporary_range=e008.allocation(temp_storage),
                        temporary_bytes=temp.numel()*temp.element_size(),temporary_buffers=1,
                        live_tensor_storage_bytes=sum(b-a for a,b in live_ranges),
                        framework_allocated_bytes=torch.cuda.memory_allocated(),framework_reserved_bytes=torch.cuda.memory_reserved(),
                        library_workspace='cuBLAS default library-managed, size not exposed',
                        active_weight_bytes=count*(2*k*n if name == 'cublas_dq' else k*n//2+2*(k//128)*n),
                        l2_bytes=bridge.e004_l2(),codes_match=True,scales_match=True,reconstruction_match=True,
                        exact_individual_gpu_reconstructions=4,lock_reset=True,workspace_guards=True,temporary_guards=True,
                        final_temporary_match=(name == 'unfused'),direct_cycles=3,poisoned_graph_cycles=3,validation=checks)
                    if args.iters:
                        poison()
                        row.update(e004.timed(launch,None,args.warmup,args.iters,active[0].c))
                        row['post_timing_validation'] = verify()
                        row['median_per_matrix_us'] = row['median_us']/count
                    out.write(json.dumps(row)+'\n')
                    out.flush()
                    print('PASS',k,n,count,name,row.get('median_us'),flush=True)
                    del graph
            del active, cases, temp, temp_storage
    bridge.e004_destroy(handle)
    print('MAPPED_LIBRARIES','\n'.join(x for x in Path('/proc/self/maps').read_text().splitlines()
        if any(v in x for v in ('libe005','libe004','libcublas','libcudart','libtorch'))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source','local','wheel','bridge','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shapes',nargs='+',default=['4096,11008','11008,4096'])
    p.add_argument('--counts',nargs='+',type=int,choices=[1,2,4],default=[1,2,4])
    p.add_argument('--implementations',nargs='+',choices=NAMES,default=NAMES)
    p.add_argument('--series',type=int,choices=[1,2],default=1)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iters',type=int,default=100)
    args = p.parse_args()
    args.configs = ['A']
    args.shapes = [tuple(map(int,x.split(','))) for x in args.shapes]
    assert args.iters >= 0 and args.warmup >= 0
    main(args)
