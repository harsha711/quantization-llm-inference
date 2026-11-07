#!/usr/bin/env python3
"""E008: complete cycles over distinct weights, unchanged corrected A/C body."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('e005', Path(__file__).resolve().parents[1]/'e005/bench.py')
e005 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e005)


def allocation(tensor):
    storage = tensor.untyped_storage()
    return [storage.data_ptr(), storage.data_ptr()+storage.nbytes()]


def distinct(ranges):
    ordered = sorted(ranges)
    assert all(a < b for a, b in ordered)
    assert all(b <= c for (_, b), (c, _) in zip(ordered, ordered[1:])), ranges


def prepare(args, k, n):
    cases = [e005.Case(args, k, n, [4], 3+j, 'mixed') for j in range(4)]
    for case in cases:
        # Only weight reuse distance changes. All matrices use the same activation.
        case.a = cases[0].a
        case.hashes['A'] = cases[0].hashes['A']
        codes = e005.e004.unpack_marlin(case.packed.cpu().numpy(), k, n,
                                        case.ns['get_weight_perm'](4).numpy())
        inv = np.argsort(np.array(case.ns['get_scale_perms']()[0]))
        scales = case.scales.cpu().numpy().reshape(-1, 64)[:, inv].reshape(k//128, n)
        wh = ((codes.astype(np.float32)-8)*scales.repeat(128, axis=0).astype(np.float32)).astype(np.float16)
        assert hashlib.sha256(wh.tobytes()).hexdigest() == case.hashes['W_hat']
        activation = case.a.cpu().numpy()
        assert hashlib.sha256(activation.tobytes()).hexdigest() == case.hashes['A']
        case.bridge.e004_reference(activation.ctypes.data, wh.ctypes.data, 4, n, k, case.ref.ctypes.data)
        del case.flush_buf
    metadata = []
    for case in cases:
        metadata.append(dict(seed=case.seed, operand_sha256=case.hashes,
            packed_range=allocation(case.packed), scales_range=allocation(case.scales),
            packed_bytes=case.packed.numel()*case.packed.element_size(),
            scales_bytes=case.scales.numel()*case.scales.element_size(),
            output_range=allocation(case.c), locks_range=allocation(case.locks_storage),
            scratch_range=allocation(case.tmp_storage), activation_range=allocation(case.a),
            packed_sha256=hashlib.sha256(case.packed.cpu().numpy().tobytes()).hexdigest(),
            scales_sha256=hashlib.sha256(case.scales.cpu().numpy().tobytes()).hexdigest()))
    distinct([v[key] for v in metadata for key in ('packed_range', 'scales_range', 'output_range', 'locks_range', 'scratch_range')])
    for key in ('packed_sha256', 'scales_sha256'):
        assert len({v[key] for v in metadata}) == 4
    assert len({v['operand_sha256']['A'] for v in metadata}) == 1
    return cases, metadata


def launch(cases, name):
    for case in cases:
        case.call(name, 4)


def verify(cases):
    result = []
    for case in cases:
        case.check_workspace()
        result.append(dict(**e005.stats(case.c, case.ref),
            output_sha256=hashlib.sha256(case.c.cpu().numpy().tobytes()).hexdigest()))
    return result


def check_cycles(cases, name):
    def poison():
        for case in cases:
            case.c.fill_(float('nan'))
    for _ in range(3):
        poison()
        launch(cases, name)
        torch.cuda.synchronize()
        verify(cases)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        launch(cases, name)
    for _ in range(3):
        poison()
        graph.replay()
        torch.cuda.synchronize()
        result = verify(cases)
    return result


def main(args):
    assert not args.out.exists(), args.out
    stream = torch.cuda.Stream()
    body = hashlib.sha256(args.local.read_bytes()).hexdigest()
    assert body == 'e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7'
    print('ENV', json.dumps(dict(body_sha256=body, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), stream=int(stream.cuda_stream))), flush=True)
    with torch.cuda.stream(stream), args.out.open('x') as out:
        for k, n in args.shapes:
            cases, metadata = prepare(args, k, n)
            for count in args.counts:
                active = cases[:count]
                for order, name in enumerate(args.configs):
                    resources = [case.call(name, 4, True) for case in active]
                    checks = check_cycles(active, name)
                    row = dict(K=k, N=n, M=4, group=128, seeds=[3,4,5,6], config=name,
                        order=order, matrices=count, method='complete_cycle_graph_events',
                        policy='no_artificial_flush', warmup=args.warmup, iters=args.iters,
                        body_sha256=body, resources=resources, allocations=metadata,
                        active_weight_bytes=sum(v['packed_bytes']+v['scales_bytes'] for v in metadata[:count]),
                        l2_bytes=cases[0].bridge.e004_l2(), codes_match=True, scales_match=True,
                        reconstruction_match=True, lock_reset=True, workspace_guards=True,
                        direct_cycles=3, poisoned_graph_cycles=3, validation=checks)
                    if args.iters:
                        for case in active:
                            case.c.fill_(float('nan'))
                        row.update(e005.e004.timed(lambda: launch(active, name), None,
                            args.warmup, args.iters, active[0].c))
                        row['post_timing_validation'] = verify(active)
                        row['median_per_matrix_us'] = row['median_us']/count
                    out.write(json.dumps(row)+'\n')
                    out.flush()
                    print('PASS', k, n, count, name, row.get('median_us'), flush=True)
            del active, cases
    print('MAPPED_LIBRARIES', '\n'.join(x for x in Path('/proc/self/maps').read_text().splitlines()
        if any(v in x for v in ('libe005','libe004','libcudart','libtorch'))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source','local','wheel','bridge','out'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--shapes', nargs='+', default=['4096,11008','11008,4096'])
    p.add_argument('--counts', nargs='+', type=int, choices=[1,2,4], default=[1,2,4])
    p.add_argument('--configs', nargs='+', choices=['A','C'], default=['A','C'])
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=100)
    args = p.parse_args()
    args.shapes = [tuple(map(int, x.split(','))) for x in args.shapes]
    assert args.iters >= 0 and args.warmup >= 0
    main(args)
