#!/usr/bin/env python3
"""E011 target validation and application-replay workload, unchanged E010 operands."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from plan import CYCLES, MATRICES, MS, NAMES, SHAPES, WARMUP, target

spec = importlib.util.spec_from_file_location('e010', Path(__file__).resolve().parents[1] / 'e010/bench.py')
e010 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e010)
e005, e004 = e010.e005, e010.e004


def main(args):
    assert hashlib.sha256(args.local.read_bytes()).hexdigest() == e010.BODY
    args.configs = ['A']
    bridge = e004.bridge(args.bridge)
    handle = bridge.e004_handle()
    cublas = ct.CDLL('libcublas.so.13')
    cublas.cublasSetMathMode.argtypes = [ct.c_void_p, ct.c_int]
    cublas.cublasGetMathMode.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
    mode = ct.c_int()
    assert cublas.cublasSetMathMode(handle, 16) == 0
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        cases, metadata = e010.prepare(args, args.k, args.n)
        resources = [case.call('A', args.m, True) for case in cases]

        def operation(case):
            if args.implementation == 'marlin':
                case.call('A', args.m)
            else:
                bridge.e004_launch(5, case.a.data_ptr(), None, None, case.wh.data_ptr(),
                                   case.wh.data_ptr(), case.c.data_ptr(), args.m, args.n, args.k,
                                   stream.cuda_stream, handle)

        def cycle():
            for case in cases:
                operation(case)

        def poison():
            for case in cases:
                case.c.fill_(float('nan'))

        def verify():
            checks = []
            for case in cases:
                case.check_workspace()
                assert torch.all(torch.isnan(case.c[args.m:])).item()
                checks.append(dict(**e005.stats(case.c[:args.m], case.ref[:args.m]),
                                   output_sha256=e010.sha(case.c[:args.m])))
            assert cublas.cublasGetMathMode(handle, ct.byref(mode)) == 0 and mode.value == 16
            return checks

        for _ in range(3):
            poison()
            cycle()
            torch.cuda.synchronize()
            validation = verify()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cycle()
        for _ in range(3):
            poison()
            graph.replay()
            torch.cuda.synchronize()
            assert verify() == validation
        row = dict(K=args.k, N=args.n, M=args.m, config=args.implementation,
                   capacity=128, matrices=MATRICES, seeds=[3, 4, 5, 6], group=128,
                   body_sha256=e010.BODY, cublas_math_mode=mode.value,
                   allocations=metadata, activation_prefix_sha256=e010.sha(cases[0].a[:args.m]),
                   resources=resources if args.implementation == 'marlin' else [],
                   direct_cycles=3, poisoned_graph_cycles=3, validation=validation,
                   warmup=WARMUP, graph_cycles=CYCLES, policy='no_artificial_flush',
                   replay='application', cache_control='none', clock_control='none')
        if args.trace:
            assert args.out is not None and not args.out.exists()
            kernels = []
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            for position, case in enumerate(cases):
                with torch.profiler.profile(activities=activities) as prof:
                    operation(case)
                    torch.cuda.synchronize()
                kernels.append([event.name for event in prof.events()
                                if event.device_type == torch.autograd.DeviceType.CUDA])
                if position == 3:
                    prof.export_chrome_trace(str(args.out.with_suffix('.target.trace.json')))
            with torch.profiler.profile(activities=activities) as prof:
                graph.replay()
                torch.cuda.synchronize()
            graph_kernels = [event.name for event in prof.events()
                             if event.device_type == torch.autograd.DeviceType.CUDA]
            assert graph_kernels == [name for names in kernels for name in names]
            prof.export_chrome_trace(str(args.out.with_suffix('.cycle.trace.json')))
            row.update(kernels_by_matrix=kernels, cycle_kernels=graph_kernels, **target(kernels))
            assert verify() == validation
            args.out.write_text(json.dumps(row, indent=2) + '\n')
        else:
            # Match E004/E010's ten direct warmups and 100 event-delimited graph cycles.
            for _ in range(WARMUP):
                cycle()
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(CYCLES)]
            stops = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(CYCLES)]
            measured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(measured, stream=stream):
                for start, stop in zip(starts, stops):
                    start.record()
                    cycle()
                    stop.record()
            cases[0].c[:args.m].fill_(float('nan'))
            torch.cuda.synchronize()
            runtime = ct.CDLL('libcudart.so.13')
            assert runtime.cudaProfilerStart() == 0
            measured.replay()
            torch.cuda.synchronize()
            assert runtime.cudaProfilerStop() == 0
            row['post_profile_validation'] = verify()
            assert row['post_profile_validation'] == validation
        print('E011_PASS ' + json.dumps(row), flush=True)
    bridge.e004_destroy(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'local', 'wheel', 'bridge'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--m', type=int, choices=MS, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--implementation', choices=NAMES, required=True)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    assert (args.k, args.n) in SHAPES
    assert args.trace == (args.out is not None)
    main(args)
