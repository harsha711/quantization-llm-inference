#!/usr/bin/env python3
"""Fresh-process prerequisite, full validation or matched timing cell."""
import argparse
import gzip
import json
from pathlib import Path

from runner import Runner, Workload, tensor_sha
from checks import checked
from scripts.e015.run import compare, save
from validate import compiler_counters, uncached
import numpy as np
import torch


def main(a):
    small = a.stage == 'smoke'
    generated = 8 if small else 64
    validation = a.stage in ('smoke', 'validation')
    tag = f'{a.stage}_{a.variant}_{a.context}' if validation else f'{a.stage}_s{a.series}_{a.variant}_{a.context}'
    assert not (a.results/(tag+'.json')).exists() and not (a.results/(tag+'.jsonl')).exists()
    work = Workload(a.results)
    runner = Runner(work, a.variant, a.context)
    warmups = []
    for _ in range(3):
        row = runner.request('primary', generated=generated)
        row.pop('logits')
        warmups.append(row)
    before = compiler_counters()
    assert not before.get('stats', {}).get('unique_graphs', 0) and not before.get('graph_break')
    output = dict(identity=work.identity, variant=a.variant, context=a.context, stage=a.stage,
        warmups=warmups, static_setup_s=runner.setup_s, graph_setup_s=runner.graph_setup_s)
    if validation:
        records = []
        for index, prompt in enumerate(('primary', 'heldout', 'primary')):
            row = runner.request(prompt, generated=generated, capture_logits=True)
            logits = row.pop('logits')
            case = f'{a.stage}_{prompt}_{a.context}_{generated}'
            reference_path = a.results/(case+'-reference.npz')
            record_path = a.results/(case+'-reference.json')
            if a.variant == 'G' and index < 2:
                assert not reference_path.exists() and not record_path.exists()
                np.savez_compressed(reference_path, logits=logits.numpy())
                save(record_path, row)
            ref_row = json.loads(record_path.read_text())
            with np.load(reference_path) as data:
                ref = torch.from_numpy(data['logits'])
            save(a.results/f'{tag}_{prompt}_{index}-output.json', row)
            errors = compare(a.results, f'{tag}_{prompt}_{index}', logits, ref)
            checked(row, ref_row, a.context, generated)
            endpoints, historical = [], None
            if a.variant == 'G' and not small:
                old = json.loads((a.results/f'e015-validation_{prompt}_{a.context}-reference.json').read_text())
                assert old['input_ids'] == row['input_ids'] and old['output_ids'] == row['output_ids']
                # Preserved E015 logits supply a full reference, not just matching token IDs.
                with np.load(a.results/f'e015-validation_{prompt}_{a.context}-reference.npz') as data:
                    old_logits = torch.from_numpy(data['logits'])
                historical = compare(a.results, f'{tag}_{prompt}_{index}-historical', logits, old_logits)
                if index == 0:
                    endpoints = uncached(work, runner, row, logits)
            records.append(dict(prompt=prompt, record=row, errors=errors, uncached=endpoints, historical=historical))
            print('PASS', tag, prompt, errors, flush=True)
        # Trace the exact fixed-length path (including the full candidate graph).
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            trace_row = runner.request('primary', generated=generated, markers=True)
        trace_row.pop('logits')
        trace_path = a.results/(tag+'-trace.json')
        prof.export_chrome_trace(str(trace_path))
        with trace_path.open('rb') as src, gzip.open(str(trace_path)+'.gz', 'wb') as dst:
            dst.write(src.read())
        trace_path.unlink()
        output.update(validation=records, trace_record=trace_row)
    else:
        prompt = 'primary' if a.stage == 'primary' else 'heldout'
        ref = json.loads((a.results/f'validation_{prompt}_{a.context}_64-reference.json').read_text())
        primary = json.loads((a.results/f'validation_primary_{a.context}_64-reference.json').read_text())
        for row in warmups:
            checked(row, primary, a.context)
        with (a.results/(tag+'.jsonl')).open('x') as f:
            for index in range(20):
                row = runner.request(prompt)
                row.pop('logits')
                checked(row, ref, a.context)
                assert compiler_counters() == before
                row.update(sample=index, series=a.series, weights_sha256=work.weights_sha256,
                    gpu=work.identity['gpu'], backend=work.identity['attention'])
                f.write(json.dumps(row)+'\n')
                f.flush()
                print('PASS', tag, index, row['decode_s'], row['e2e_s'], flush=True)
    work.weights_unchanged()
    output.update(compiler_before=before, compiler_after=compiler_counters())
    assert output['compiler_after'] == before
    save(a.results/(tag+'.json'), output)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    p.add_argument('--stage', choices=('smoke', 'validation', 'primary', 'heldout'), required=True)
    p.add_argument('--variant', choices=('G', 'H'), required=True)
    p.add_argument('--context', choices=(128, 2048), type=int, required=True)
    p.add_argument('--series', choices=(1, 2), type=int, default=1)
    a = p.parse_args()
    try:
        main(a)
    except BaseException as exc:
        # Preserve available memory diagnostics as well as the traced exception.
        if torch.cuda.is_initialized():
            failure = dict(error=repr(exc), allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(), peak_reserved=torch.cuda.max_memory_reserved())
            (a.results/f'failure-{a.stage}-{a.variant}-{a.context}.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise
