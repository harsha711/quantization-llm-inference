#!/usr/bin/env python3
"""Cheap check of profiler-window counting on the installed graph-node replay."""
import ctypes as ct
import torch

stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    buffers = [torch.empty(n, device='cuda') for n in (256, 1024, 4096, 16384)]

    def cycle():
        for position, buffer in enumerate(buffers):
            buffer.fill_(position + 1)

    for _ in range(10):
        cycle()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(99):
            cycle()
        for position, buffer in enumerate(buffers[:3]):
            buffer.fill_(position + 1)
        buffers[-1].add_(0)
    runtime = ct.CDLL('libcudart.so.13')
    assert runtime.cudaProfilerStart() == 0
    graph.replay()
    torch.cuda.synchronize()
    assert runtime.cudaProfilerStop() == 0
    for position, buffer in enumerate(buffers):
        assert torch.all(buffer == position + 1).item()
print('PASS: ten direct warmups, 100 four-operation graph cycles, unique final add_ target')
