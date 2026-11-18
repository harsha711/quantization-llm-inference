#!/usr/bin/env python3
"""E049 window check: E011's four-fill calibration plus the timing events of the real profile graph.

100 event-delimited cycles of four fills, the last operation replaced by a unique add_; ncu --launch-skip 399
--launch-count 1 must select that add_ (event-record nodes are then not counted as launches)."""
import ctypes as ct
import torch

stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    buffers = [torch.empty(n, device='cuda') for n in (256, 1024, 4096, 16384)]
    for _ in range(10):
        for position, buffer in enumerate(buffers):
            buffer.fill_(position + 1)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(100)]
    stops = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(100)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for cycle, (start, stop) in enumerate(zip(starts, stops)):
            start.record()
            for position, buffer in enumerate(buffers):
                buffer.add_(0) if (cycle, position) == (99, 3) else buffer.fill_(position + 1)
            stop.record()
    runtime = ct.CDLL('libcudart.so.13')
    assert runtime.cudaProfilerStart() == 0
    graph.replay()
    torch.cuda.synchronize()
    assert runtime.cudaProfilerStop() == 0
    for position, buffer in enumerate(buffers):
        assert torch.all(buffer == position + 1).item()
print('PASS: 100 event-delimited four-operation graph cycles, unique final add_ target')
