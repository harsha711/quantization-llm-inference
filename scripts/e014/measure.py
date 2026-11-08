#!/usr/bin/env python3
"""One fresh-process warmed inference cell, retaining every synchronized wall sample."""
import argparse
import json
from pathlib import Path

from model import Workload, Runner
from validate import compiler_counters


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('results',type=Path)
    p.add_argument('--variant',choices=['A','B','C'],required=True)
    p.add_argument('--context',type=int,choices=[128,2048],required=True)
    p.add_argument('--series',type=int,choices=[1,2],required=True)
    p.add_argument('--prompt',choices=['primary','heldout'],required=True)
    a = p.parse_args()
    tag = f"{a.prompt}_s{a.series}_{a.variant}_{a.context}"
    out = a.results/(tag+'.jsonl')
    assert not out.exists()
    work = Workload(a.results)
    runner = Runner(work,a.variant,a.context)
    reference = json.loads((a.results/f'reference_{a.prompt}_{a.context}.json').read_text())
    assert reference['weights_sha256'] == work.weights_sha256
    warmups = []
    for _ in range(3):
        row = runner.request(a.prompt)
        row.pop('logits')
        assert row['output_ids'] == reference['record']['output_ids']
        warmups.append(row)
    before = compiler_counters()
    if a.variant == 'C':
        assert before.get('stats',{}).get('unique_graphs',0) >= 1 and not before.get('graph_break')
    else:
        assert before.get('stats',{}).get('unique_graphs',0) == 0
    with out.open('x') as f:
        for index in range(20):
            row = runner.request(a.prompt)
            row.pop('logits')
            assert row['input_ids'] == reference['record']['input_ids']
            assert row['output_ids'] == reference['record']['output_ids']
            assert compiler_counters() == before,'compilation/recompilation during warmed timing'
            assert row['peak_reserved'] < row['device_total']*0.75,'memory margin exceeded'
            row.update(sample=index,variant=a.variant,series=a.series,
                weights_sha256=work.weights_sha256,model_revision=work.identity['model_revision'],
                backend=work.identity['attention'],gpu=work.identity['gpu'])
            f.write(json.dumps(row)+'\n');f.flush()
            print('PASS',tag,index,row['decode_s'],row['ttft_s'],row['e2e_s'],flush=True)
    after = compiler_counters()
    assert after == before
    work.weights_unchanged()
    (a.results/(tag+'-context.json')).write_text(json.dumps(dict(identity=work.identity,
        variant=a.variant,context=a.context,prompt=a.prompt,series=a.series,
        static_setup_s=runner.setup_s,warmups=warmups,
        compiler_before=before,compiler_after=after),indent=2)+'\n')
