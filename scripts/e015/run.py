#!/usr/bin/env python3
"""One fresh-process E015 smoke, full validation or timing cell."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

from runner import Workload, Runner, tensor_sha
from checks import checked
from validate import compiler_counters, uncached
import numpy as np
import torch


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def compare(root, tag, got, ref):
    # Preserve tensors/statistics before any numerical assertion can stop the run.
    assert not (root/(tag+'.npz')).exists()
    np.savez_compressed(root/(tag+'.npz'), candidate=got.numpy(), reference=ref.numpy())
    assert got.shape == ref.shape
    delta = (got.float()-ref.float()).abs()
    bad = delta > .03+.01*ref.float().abs()
    finite = bool(torch.isfinite(got).all() and torch.isfinite(ref).all())
    norm = float(torch.linalg.vector_norm(delta.double())/torch.linalg.vector_norm(ref.double()))
    result = dict(finite=finite, bad_logits=int(bad.sum()), logits_checked=ref.numel(),
                  max_abs=float(delta.max()), relative_l2=norm,
                  candidate_sha256=tensor_sha(got), reference_sha256=tensor_sha(ref),
                  bitwise_equal=bool(torch.equal(got, ref)))
    if not finite or result['bad_logits'] or norm > .003:
        np.savez_compressed(root/(tag+'-failed-indices.npz'), indices=bad.nonzero().numpy())
    save(root/(tag+'-errors.json'), result)
    assert finite and result['bad_logits'] == 0 and norm <= .003, result
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    p.add_argument('--stage', choices=('smoke', 'validation', 'primary', 'heldout'), required=True)
    p.add_argument('--variant', choices=('E', 'G'), required=True)
    p.add_argument('--context', type=int, choices=(128, 2048), required=True)
    p.add_argument('--series', type=int, choices=(1, 2), default=1)
    a = p.parse_args()
    tag = f'{a.stage}_{a.variant}_{a.context}' if a.stage in ('smoke', 'validation') else f'{a.stage}_s{a.series}_{a.variant}_{a.context}'
    assert not (a.results/(tag+'.json')).exists() and not (a.results/(tag+'.jsonl')).exists()
    work = Workload(a.results)
    runner = Runner(work, a.variant, a.context)
    generated = 8 if a.stage == 'smoke' else 64
    warmups = []
    for _ in range(3):
        row = runner.request('primary', generated=generated)
        row.pop('logits')
        warmups.append(row)
    before = compiler_counters()
    assert not before.get('stats', {}).get('unique_graphs', 0) and not before.get('graph_break')
    output = dict(identity=work.identity, variant=a.variant, context=a.context, stage=a.stage,
                  warmups=warmups, static_setup_s=runner.setup_s, graph_setup_s=runner.graph_setup_s)
    if a.stage in ('smoke', 'validation'):
        records = []
        prompts = ('primary', 'heldout', 'primary') if a.stage == 'validation' else ('primary', 'heldout')
        for index, prompt in enumerate(prompts):
            row = runner.request(prompt, generated=generated, capture_logits=True)
            logits = row.pop('logits')
            case = f'{a.stage}_{prompt}_{a.context}_{generated}'
            reference_path = a.results/(case+'-reference.npz')
            record_path = a.results/(case+'-reference.json')
            if a.variant == 'E' and index < 2:
                assert not reference_path.exists() and not record_path.exists()
                np.savez_compressed(reference_path, logits=logits.numpy())
                save(record_path, row)
            ref_row = json.loads(record_path.read_text())
            with np.load(reference_path) as data:
                ref = torch.from_numpy(data['logits'])
            save(a.results/f'{tag}_{prompt}_{index}-output.json', row)
            errors = compare(a.results, f'{tag}_{prompt}_{index}', logits, ref)
            checked(row, ref_row, a.context, generated)
            if a.stage == 'validation' and a.variant == 'E' and index < 2:
                historical = json.loads((a.results/f'e014-B-{a.context}.json').read_text())
                old = next(v for v in historical['validation'] if v['prompt'] == prompt)
                assert historical['identity']['weights_sha256'] == work.weights_sha256
                assert old['record']['input_ids'] == row['input_ids']
                row['historical_tokens_equal'] = old['record']['output_ids'] == row['output_ids']
                row['historical_logits_equal'] = old['candidate_logits_sha256'] == tensor_sha(logits)
                if historical['identity']['gpu'] == work.identity['gpu']:
                    assert row['historical_tokens_equal'] and row['historical_logits_equal']
            uncached_checks = uncached(work, runner, row, logits) if a.stage == 'validation' and a.variant == 'E' and index == 0 else []
            records.append(dict(prompt=prompt, record=row, errors=errors, uncached=uncached_checks))
            print('PASS', tag, prompt, errors, flush=True)
        if a.stage == 'validation':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True, profile_memory=True) as prof:
                trace_row = runner.request('primary', generated=8, markers=True)
            trace_row.pop('logits')
            trace_path = a.results/(tag+'-trace.json')
            prof.export_chrome_trace(str(trace_path))
            with trace_path.open('rb') as src, gzip.open(str(trace_path)+'.gz', 'wb') as dst:
                dst.write(src.read())
            trace_path.unlink()
            output['trace_record'] = trace_row
        output['validation'] = records
    else:
        prompt = 'primary' if a.stage == 'primary' else 'heldout'
        ref = json.loads((a.results/f'validation_{prompt}_{a.context}_64-reference.json').read_text())
        for row in warmups:
            # Warmups use primary text in every cell; timed rows use the declared passage.
            primary_ref = json.loads((a.results/f'validation_primary_{a.context}_64-reference.json').read_text())
            checked(row, primary_ref, a.context)
        with (a.results/(tag+'.jsonl')).open('x') as f:
            for index in range(20):
                row = runner.request(prompt)
                row.pop('logits')
                checked(row, ref, a.context)
                assert compiler_counters() == before
                row.update(sample=index, series=a.series, weights_sha256=work.weights_sha256,
                           gpu=work.identity['gpu'], backend=work.identity['attention'])
                f.write(json.dumps(row)+'\n'); f.flush()
                print('PASS', tag, index, row['decode_s'], row['e2e_s'], flush=True)
    work.weights_unchanged()
    output['compiler_before'] = before
    output['compiler_after'] = compiler_counters()
    assert output['compiler_after'] == before
    save(a.results/(tag+'.json'), output)
