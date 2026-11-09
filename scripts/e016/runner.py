"""E015 per-token replay versus one explicitly unrolled decode graph."""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.e015.runner import Runner as TokenRunner, Workload, tensor_sha
from model import mark
import torch


class Runner(TokenRunner):
    def __init__(self, work, variant, length):
        assert variant in ('G', 'H')
        super().__init__(work, 'G', length)
        self.mode = variant
        self.whole_graph = None
        self.fixed_generated = None
        self.tokens = torch.empty(64, dtype=torch.long, device='cuda')
        self.logits = torch.empty((64, 1, self.model.config.vocab_size), dtype=torch.float16, device='cuda')
        self.positions_seen = torch.empty(63, dtype=torch.long, device='cuda')
        self.feedback = torch.empty(63, dtype=torch.long, device='cuda')
        bias = self.bias(self.position, self.capacity)
        self.mask = torch.empty_strided(bias.size(), bias.stride(), dtype=bias.dtype, device=bias.device)
        self.storage = self.addresses()
        self.restoration = None

    def cache_tensors(self):
        return [p for layer in self.cache.layers for p in (layer.keys, layer.values)]

    def addresses(self):
        return self.pointers()+[p.data_ptr() for p in (self.input_token, self.position, self.mask,
            self.tokens, self.logits, self.positions_seen, self.feedback)]

    def steps(self, generated):
        for i in range(1, generated):
            self.position.fill_(self.length+i-1)
            self.positions_seen[i-1].copy_(self.position[0])
            self.feedback[i-1].copy_(self.input_token[0, 0])
            bias = self.bias(self.position, self.capacity)
            if self.mode == 'G':
                logits, token = self.staged_decode(self.input_token, self.position, bias)
            else:
                self.mask.copy_(bias)
                logits, token = self.eager_decode(self.input_token, self.position, self.mask)
            self.logits[i].copy_(logits)
            self.tokens[i].copy_(token[0, 0])
            self.input_token.copy_(token)

    def construct(self, generated):
        start = time.perf_counter()
        cache = [p.clone() for p in self.cache_tensors()]
        buffers = (self.input_token, self.position, self.tokens, self.logits, self.positions_seen, self.feedback, self.mask)
        saved = [p.clone() for p in buffers]

        def restore():
            for dest, source in zip(self.cache_tensors()+list(buffers), cache+saved):
                dest.copy_(source)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                restore()
                self.steps(generated)
        torch.cuda.current_stream().wait_stream(stream)
        restore()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.steps(generated)
        torch.cuda.synchronize()
        restore()
        torch.cuda.synchronize()
        # Snapshot equality is checked after capture, before the first genuine replay.
        restored = all(torch.equal(a, b) for a, b in zip(self.cache_tensors()+list(buffers), cache+saved))
        self.restoration = dict(post_prefill_bits_restored=restored,
            cache_lengths=[int(layer.get_seq_length()) for layer in self.cache.layers],
            tail_zero=all(not bool(p[:, :, self.length:].count_nonzero()) for p in self.cache_tensors()),
            input_token=int(self.input_token.item()), pointers_stable=self.addresses() == self.storage)
        print('CAPTURE_RESTORATION', self.restoration, flush=True)
        assert restored and self.restoration['tail_zero'] and self.restoration['pointers_stable'], self.restoration
        assert self.restoration['cache_lengths'] == [self.length]*len(self.cache.layers)
        self.whole_graph = graph
        self.graph_setup_s = time.perf_counter()-start

    def cache_state(self):
        result = []
        for layer in self.cache.layers:
            pair = []
            for tensor in (layer.keys, layer.values):
                value = tensor.cpu().contiguous()
                pair.append(dict(sha256=tensor_sha(value),
                    decode_positions=[tensor_sha(value[:, :, i]) for i in range(self.length, self.length+self.fixed_generated-1)]))
            result.append(pair)
        return result

    @torch.inference_mode()
    def request(self, prompt_name, generated=64, capture_logits=False, markers=False):
        assert generated in (8, 64)
        if self.fixed_generated is None:
            self.fixed_generated = generated
        assert generated == self.fixed_generated
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with mark('E016/input_and_cache', markers):
            _, ids = self.workload.prompt(prompt_name, self.length)
            inputs = torch.tensor([ids], dtype=torch.long, device='cuda')
            self.cache.reset()
            self.tokens.zero_()
            self.logits.zero_()
            self.positions_seen.fill_(-1)
            self.feedback.fill_(-1)
            self.position.zero_()
            self.mask.zero_()
            positions = torch.arange(self.length, device='cuda')
            bias = self.bias(positions, self.capacity)
        prefill_start = time.perf_counter()
        with mark('E016/prefill', markers):
            output = self.model(input_ids=inputs, position_ids=positions.view(1, -1),
                attention_mask=bias, past_key_values=self.cache, cache_position=positions,
                use_cache=True, logits_to_keep=1)
            logits = output.logits[:, 0, :]
            next_token = logits.argmax(dim=-1).view(1, 1)
            self.tokens[0].copy_(next_token[0, 0])
            first = int(next_token.item())
        first_end = time.perf_counter()
        self.logits[0].copy_(logits)
        self.input_token.copy_(next_token)
        decode_start = time.perf_counter()
        with mark('E016/decode', markers):
            if self.mode == 'H':
                if self.whole_graph is None:
                    self.construct(generated)
                self.whole_graph.replay()
                replays = 1
            else:
                before = self.replays
                self.steps(generated)
                replays = self.replays-before
            torch.cuda.synchronize()
        decode_end = time.perf_counter()
        with mark('E016/output', markers):
            output_ids = self.tokens[:generated].cpu().tolist()
            output_text = self.workload.tokenizer.decode(output_ids, skip_special_tokens=False)
        end = time.perf_counter()
        # All audit transfers occur after the request timers in both variants.
        positions_seen = self.positions_seen[:generated-1].cpu().tolist()
        feedback = self.feedback[:generated-1].cpu().tolist()
        cache_lengths = [int(layer.get_seq_length()) for layer in self.cache.layers]
        stable = self.addresses() == self.storage
        return dict(prompt=prompt_name, context=self.length, generated=generated, decode_forwards=generated-1,
            input_ids=ids, input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            output_ids=output_ids, output_text=output_text,
            eos_positions=[i for i, x in enumerate(output_ids) if x in self.workload.eos],
            eos_policy='fixed length, EOS recorded and not used to stop', variant=self.mode,
            ttft_s=first_end-start, prefill_to_first_s=first_end-prefill_start,
            preparation_s=prefill_start-start, decode_s=decode_end-decode_start, e2e_s=end-start,
            post_first_boundary_s=decode_start-first_end, output_s=end-decode_end,
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            device_total=torch.cuda.get_device_properties(0).total_memory,
            cache_class=type(self.cache).__name__, static_capacity=self.capacity, cache_length=cache_lengths[0],
            cache_lengths=cache_lengths, positions_seen=positions_seen, feedback=feedback,
            cache_pointers_stable=stable, storage_stable=stable, graph_replays=replays,
            graph_setup_s=self.graph_setup_s, restoration=self.restoration,
            cache_state=self.cache_state() if capture_logits else None,
            logits=self.logits[:generated].cpu() if capture_logits else None)
