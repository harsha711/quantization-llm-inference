#!/usr/bin/env python3
"""E017 common eager static-cache validation and synchronized request timing."""
import argparse
from contextlib import nullcontext
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch._dynamo.utils import counters

from integration import BaseRunner, Workload, save, tensor_sha


def compiler_counters():
    return {key: dict(value) for key, value in counters.items() if value}


def accept_logits(stats):
    assert stats['shape_equal'] and stats['finite'], stats
    norm = stats['relative_l2']
    assert stats['bad_logits'] == 0 and isinstance(norm, (int, float)) and math.isfinite(norm) and norm <= .003, stats


def checked(row, reference, length, generated=64):
    assert row['context'] == length and row['generated'] == generated
    assert row['decode_forwards'] == generated - 1
    assert row['input_ids'] == reference['input_ids']
    assert row['output_ids'] == reference['output_ids'][:generated], 'greedy tokens diverged'
    assert row['positions_seen'] == list(range(length, length + generated - 1))
    assert row['feedback'] == row['output_ids'][:-1]
    assert row['first_token'] == row['output_ids'][0]
    assert row['static_capacity'] == length + 64 and row['cache_class'] == 'StaticCache'
    assert row['cache_lengths'] == [length + generated - 1] * 28
    assert row['cache_reset_zero'] == [True] * 56
    assert row['cache_tail_zero'] and row['cache_finite'] and row['storage_stable'] and row['logits_finite']
    assert row['variant'] in ('O', 'R', 'Q') and row['graph_replays'] == 0
    for key in ('decode_s', 'ttft_s', 'e2e_s', 'preparation_s', 'prefill_to_first_s', 'post_first_boundary_s', 'output_s'):
        assert math.isfinite(row[key]) and row[key] >= 0, key
    assert row['decode_s'] > 0 and row['ttft_s'] > 0
    total = sum(row[key] for key in ('preparation_s', 'prefill_to_first_s', 'post_first_boundary_s', 'decode_s', 'output_s'))
    assert math.isclose(row['e2e_s'], total, rel_tol=1e-8, abs_tol=1e-7)
    assert math.isclose(row['ttft_s'], row['preparation_s'] + row['prefill_to_first_s'], abs_tol=1e-7)
    assert 0 < row['peak_allocated'] <= row['peak_reserved'] < row['device_total'] * .75


class Runner(BaseRunner):
    def __init__(self, work, variant, length):
        start = time.perf_counter()
        super().__init__(work, 'B', length)
        self.mode = variant
        self.tokens = torch.empty(64, dtype=torch.long, device='cuda')
        self.logits = torch.empty((64, 1, self.model.config.vocab_size), dtype=torch.float16, device='cuda')
        self.positions_seen = torch.empty(63, dtype=torch.long, device='cuda')
        self.feedback = torch.empty(63, dtype=torch.long, device='cuda')
        self.reset_zero = torch.empty(56, dtype=torch.bool, device='cuda')
        self.storage = self.addresses()
        torch.cuda.synchronize()
        self.total_setup_s = time.perf_counter() - start

    def cache_tensors(self):
        return [tensor for layer in self.cache.layers for tensor in (layer.keys, layer.values)]

    def addresses(self):
        return [tensor.data_ptr() for tensor in self.cache_tensors() + [self.input_token, self.position,
            self.tokens, self.logits, self.positions_seen, self.feedback, self.reset_zero]]

    @torch.inference_mode()
    def request(self, prompt_name, generated=64, capture_logits=False, markers=False):
        assert generated in (8, 64)
        if self.workload.operator: self.workload.operator.markers=markers
        mark = torch.profiler.record_function if markers else lambda name: nullcontext()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with mark('E017/input_and_cache'):
            _, ids = self.workload.prompt(prompt_name, self.length)
            inputs = torch.tensor([ids], dtype=torch.long, device='cuda')
            self.cache.reset()
            # Device-only reset observations; all CPU diagnostics remain after timers.
            for index, tensor in enumerate(self.cache_tensors()):
                self.reset_zero[index].copy_(torch.count_nonzero(tensor) == 0)
            self.tokens.zero_()
            self.logits.zero_()
            self.positions_seen.fill_(-1)
            self.feedback.fill_(-1)
            self.position.zero_()
            positions = torch.arange(self.length, device='cuda')
            bias = self.bias(positions, self.capacity)
        prefill_start = time.perf_counter()
        with mark('E017/prefill'):
            output = self.model(input_ids=inputs, position_ids=positions.view(1, -1),
                attention_mask=bias, past_key_values=self.cache, cache_position=positions,
                use_cache=True, logits_to_keep=1)
            logits = output.logits[:, 0, :]
            token = logits.argmax(dim=-1).view(1, 1)
            self.tokens[0].copy_(token[0, 0])
            first = int(token.item())
        first_end = time.perf_counter()
        self.logits[0].copy_(logits)
        self.input_token.copy_(token)
        decode_start = time.perf_counter()
        with mark('E017/decode'):
            for index in range(1, generated):
                self.position.fill_(self.length + index - 1)
                self.positions_seen[index - 1].copy_(self.position[0])
                self.feedback[index - 1].copy_(self.input_token[0, 0])
                bias = self.bias(self.position, self.capacity)
                logits, token = self.eager_decode(self.input_token, self.position, bias)
                self.logits[index].copy_(logits)
                self.tokens[index].copy_(token[0, 0])
                self.input_token.copy_(token)
            torch.cuda.synchronize()
        decode_end = time.perf_counter()
        with mark('E017/output'):
            output_ids = self.tokens[:generated].cpu().tolist()
            output_text = self.workload.tokenizer.decode(output_ids, skip_special_tokens=False)
        end = time.perf_counter()
        peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        tensors = self.cache_tensors()
        row = dict(prompt=prompt_name, variant=self.mode, context=self.length, generated=generated,
            decode_forwards=generated - 1, input_ids=ids,
            input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            output_ids=output_ids, output_text=output_text, first_token=first,
            eos_positions=[index for index, value in enumerate(output_ids) if value in self.workload.eos],
            eos_policy='fixed length, EOS recorded and not used to stop',
            ttft_s=first_end-start, prefill_to_first_s=first_end-prefill_start,
            preparation_s=prefill_start-start, decode_s=decode_end-decode_start, e2e_s=end-start,
            post_first_boundary_s=decode_start-first_end, output_s=end-decode_end,
            peak_allocated=peak_allocated, peak_reserved=peak_reserved,
            allocated=allocated, reserved=reserved, device_total=torch.cuda.get_device_properties(0).total_memory,
            cache_class=type(self.cache).__name__, static_capacity=self.capacity,
            cache_lengths=[int(layer.get_seq_length()) for layer in self.cache.layers],
            cache_reset_zero=self.reset_zero.cpu().tolist(),
            cache_tail_zero=all(not bool(value[:, :, self.length + generated - 1:].count_nonzero()) for value in tensors),
            cache_finite=all(bool(torch.isfinite(value).all()) for value in tensors),
            storage_stable=self.addresses() == self.storage,
            positions_seen=self.positions_seen[:generated-1].cpu().tolist(),
            feedback=self.feedback[:generated-1].cpu().tolist(), graph_replays=0,
            logits_finite=bool(torch.isfinite(self.logits[:generated]).all()),
            cache_hashes=[tensor_sha(value) for value in tensors] if capture_logits else None)
        return row, self.logits[:generated].cpu() if capture_logits else None


def compare(root, tag, got, reference):
    """Preserve both tensors and diagnostics before applying any numerical gate."""
    np.savez_compressed(root / (tag + '-comparison.npz'), candidate=got.numpy(), reference=reference.numpy())
    same_shape = got.shape == reference.shape
    finite = bool(torch.isfinite(got).all() and torch.isfinite(reference).all())
    candidate_hash, reference_hash = tensor_sha(got), tensor_sha(reference)
    result = dict(shape_equal=same_shape, finite=finite, candidate_sha256=candidate_hash,
        reference_sha256=reference_hash, bitwise_equal=same_shape and candidate_hash == reference_hash)
    if same_shape:
        delta = (got.float() - reference.float()).abs()
        bad = ~torch.isfinite(got) | ~torch.isfinite(reference) | (delta > .03 + .01 * reference.float().abs())
        numerator = float(torch.linalg.vector_norm(delta.double()))
        denominator = float(torch.linalg.vector_norm(reference.double()))
        norm = numerator / denominator if denominator else (0.0 if numerator == 0 else float('inf'))
        maximum = float(delta.max())
        result.update(bad_logits=int(bad.sum()), logits_checked=reference.numel(),
            max_abs=maximum if math.isfinite(maximum) else None,
            relative_l2=norm if math.isfinite(norm) else None)
        if not finite or result['bad_logits'] or not math.isfinite(norm) or norm > .003:
            np.savez_compressed(root / (tag + '-failed-indices.npz'), indices=bad.nonzero().numpy())
    else:
        result.update(bad_logits=None, logits_checked=reference.numel(), max_abs=None, relative_l2=None)
    save(root / (tag + '-errors.json'), result)
    return result


def main(args):
    root = args.results
    timed = args.stage in ('primary', 'heldout')
    if timed:
        assert args.variant in ('R', 'Q')
        assert (root / 'FULL_VALIDATION_INSPECTED').is_file(), 'full inspected validation required'
        if args.stage == 'heldout':
            assert (root / 'HELDOUT_RELEASED').is_file(), 'primary performance gate required'
    tag = f'{args.stage}_{args.variant}_{args.context}' + (f'_s{args.series}' if timed else '')
    assert not (root / (tag + '.json')).exists() and not (root / (tag + '.jsonl')).exists()
    work = Workload(root, args.variant)
    runner = Runner(work, args.variant, args.context)
    work.weights_unchanged()
    torch.cuda.empty_cache()
    before = compiler_counters()
    assert not before.get('stats', {}).get('unique_graphs', 0) and not before.get('graph_break')
    setup = dict(identity=work.identity, variant=args.variant, context=args.context, stage=args.stage,
        series=args.series, static_setup_s=runner.setup_s, total_runner_setup_s=runner.total_setup_s,
        compiler_before=before)
    save(root / (tag + '-setup.json'), setup)
    records = []
    with (root / (tag + '.jsonl')).open('x') as log:
        if args.stage == 'validation':
            first_seen = {}
            for index, prompt in enumerate(('primary', 'heldout', 'primary', 'heldout')):
                row, logits = runner.request(prompt, capture_logits=True)
                case = f'{tag}_{prompt}_{index}'
                np.savez_compressed(root / (case + '-output.npz'), logits=logits.numpy())
                save(root / (case + '-output.json'), row)
                log.write(json.dumps(dict(sample=index, record=row)) + '\n'); log.flush()
                reference_tag = f'validation_R_{args.context}_{prompt}_0' if prompt == 'primary' else f'validation_R_{args.context}_{prompt}_1'
                own_tag = case if prompt not in first_seen else first_seen[prompt]
                with np.load(root / (own_tag + '-output.npz')) as archive:
                    own_logits = torch.from_numpy(archive['logits'])
                own_row = json.loads((root / (own_tag + '-output.json')).read_text())
                repetition = compare(root, case + '-repeat', logits, own_logits)
                errors = None
                if args.variant == 'Q':
                    with np.load(root / (reference_tag + '-output.npz')) as archive:
                        ref_logits = torch.from_numpy(archive['logits'])
                    ref_row = json.loads((root / (reference_tag + '-output.json')).read_text())
                    errors = compare(root, case + '-reference', logits, ref_logits)
                else:
                    ref_row = own_row
                result = dict(prompt=prompt, record=row, repetition=repetition, errors=errors)
                # Quantization descriptors are observations, never equivalence gates.
                if args.variant == 'R':
                    original_tag = f'validation_O_{args.context}_{prompt}_{0 if prompt == "primary" else 1}'
                    with np.load(root / (original_tag + '-output.npz')) as archive:
                        original = torch.from_numpy(archive['logits'])
                    original_row = json.loads((root / (original_tag + '-output.json')).read_text())
                    result['quantization_changes'] = compare(root, case + '-versus-original', logits, original)
                    result['quantization_changes'].update(
                        tokens_equal=row['output_ids'] == original_row['output_ids'],
                        conditioning='each model follows its own greedy token history; not teacher-forced quality evidence')
                    save(root / (case + '-quantization.json'), result['quantization_changes'])
                save(root / (case + '-diagnostics.json'), result)
                checked(row, ref_row, args.context)
                accept_logits(repetition)
                assert repetition['bitwise_equal'] and row['cache_hashes'] == own_row['cache_hashes'], 'same-path replay changed'
                if errors is not None:
                    accept_logits(errors)
                first_seen.setdefault(prompt, case)
                records.append(result)
                assert compiler_counters() == before
                print('PASS', case, flush=True)
        elif args.stage == 'trace':
            warm, _ = runner.request('primary', generated=8)
            save(root / (tag + '-warmup.json'), warm)
            checked(warm, warm, args.context, generated=8)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as profiler:
                row, logits = runner.request('primary', generated=8, capture_logits=True, markers=True)
            trace = root / (tag + '-trace.json')
            profiler.export_chrome_trace(str(trace))
            with trace.open('rb') as source, gzip.open(str(trace) + '.gz', 'wb') as target:
                target.write(source.read())
            trace.unlink()
            np.savez_compressed(root / (tag + '-output.npz'), logits=logits.numpy())
            log.write(json.dumps(row) + '\n'); log.flush()
            checked(row, warm, args.context, generated=8)
            records.append(row)
        else:
            prompt = 'primary' if args.stage == 'primary' else 'heldout'
            ref_index = 0 if prompt == 'primary' else 1
            ref_tag = f'validation_R_{args.context}_{prompt}_{ref_index}'
            reference = json.loads((root / (ref_tag + '-output.json')).read_text())
            with (root / (tag + '-warmups.jsonl')).open('x') as warm_log:
                for index in range(3):
                    row, _ = runner.request(prompt)
                    warm_log.write(json.dumps(dict(sample=index, record=row)) + '\n'); warm_log.flush()
                    checked(row, reference, args.context)
                    assert compiler_counters() == before
            for index in range(20):
                row, _ = runner.request(prompt)
                row.update(sample=index, series=args.series, identity=work.identity)
                log.write(json.dumps(row) + '\n'); log.flush()
                checked(row, reference, args.context)
                assert compiler_counters() == before
                print('PASS', tag, index, row['decode_s'], row['e2e_s'], flush=True)
    work.weights_unchanged()
    after = compiler_counters()
    save(root / (tag + '.json'), dict(**setup, records=records, compiler_after=after, weights_unchanged=True))
    assert before == after


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('--variant', choices=('O', 'R', 'Q'), required=True)
    parser.add_argument('--context', type=int, choices=(128, 2048), required=True)
    parser.add_argument('--stage', choices=('validation', 'primary', 'heldout', 'trace'), required=True)
    parser.add_argument('--series', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as exception:
        path = args.results / f'failure-{args.stage}-{args.variant}-{args.context}-s{args.series}-{time.time_ns()}.json'
        diagnostics = dict(error=repr(exception), compiler=compiler_counters())
        if torch.cuda.is_initialized():
            diagnostics.update(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
        save(path, diagnostics)
        raise
