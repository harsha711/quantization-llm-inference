"""E014 static decoding with matched mask staging and optional direct replay."""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'e014'))
from model import Workload, Runner as StaticRunner, tensor_sha
import torch


class Runner(StaticRunner):
    def __init__(self, work, variant, length):
        assert variant in ('E', 'G')
        super().__init__(work, 'B', length)
        assert all(type(layer).__name__ == 'StaticLayer' for layer in self.cache.layers)
        self.mode = variant
        self.graph = None
        self.graph_setup_s = 0.0
        self.replays = 0
        self.mask = None
        self.decode = self.staged_decode
        self.cache_pointers = self.pointers()

    def pointers(self):
        return [p.data_ptr() for layer in self.cache.layers for p in (layer.keys, layer.values)]

    def staged_decode(self, token, position, bias):
        if self.mask is None:
            self.mask = torch.empty_strided(bias.size(), bias.stride(), dtype=bias.dtype, device=bias.device)
        assert token.data_ptr() == self.input_token.data_ptr()
        assert position.data_ptr() == self.position.data_ptr()
        self.mask.copy_(bias)
        if self.mode == 'E':
            return self.eager_decode(token, position, self.mask)
        if self.graph is None:
            start = time.perf_counter()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.eager_decode(token, position, self.mask)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.graph_output = self.eager_decode(token, position, self.mask)
            torch.cuda.synchronize()
            self.graph_setup_s = time.perf_counter()-start
            self.input_pointers = [token.data_ptr(), position.data_ptr(), self.mask.data_ptr()]
        assert self.input_pointers == [token.data_ptr(), position.data_ptr(), self.mask.data_ptr()]
        self.graph.replay()
        self.replays += 1
        return self.graph_output

    def request(self, *args, **kwargs):
        before = self.replays
        row = super().request(*args, **kwargs)
        assert self.pointers() == self.cache_pointers
        assert self.replays-before == (row['decode_forwards'] if self.mode == 'G' else 0)
        row.update(variant=self.mode, graph_replays=self.replays-before,
                   graph_setup_s=self.graph_setup_s, cache_pointers_stable=True)
        return row
