#!/usr/bin/env python3
"""Check matched full-model evidence and apply the predeclared decode-latency gate."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

from timeline import summarize

CONTEXTS = [128,2048]
VARIANTS = ['A','B','C']
PAIRS = [('A','B'),('B','C'),('A','C')]
REVISION = '8faed761d45a263340a0528343f099c05c9a4323'


def read(path):
    return json.loads(path.read_text())


def events(path):
    if path.exists():
        return read(path)['traceEvents']
    with gzip.open(str(path)+'.gz','rt') as f:
        return json.load(f)['traceEvents']


def gain(base,candidate):
    assert all(math.isfinite(x) and x > 0 for x in (base,candidate))
    return 100*(1-candidate/base)


def passes(gains):
    return len(gains) == 2 and all(math.isfinite(g) and g >= 5 for g in gains)


def traversal(series):
    contexts,variants = (CONTEXTS,VARIANTS) if series == 1 else (CONTEXTS[::-1],VARIANTS[::-1])
    return list(itertools.product(contexts,variants))


def checked_sample(row, reference, context):
    assert row['context'] == context and row['generated'] == 64 and row['decode_forwards'] == 63
    assert row['cache_length'] == context+63
    assert row['input_ids'] == reference['input_ids'] and len(row['input_ids']) == context
    assert row['output_ids'] == reference['output_ids'] and len(row['output_ids']) == 64
    assert row['input_sha256'] == reference['input_sha256']
    assert row['eos_positions'] == reference['eos_positions']
    assert row['eos_policy'] == 'fixed length, EOS recorded and not used to stop'
    for key in ('decode_s','ttft_s','prefill_to_first_s','preparation_s','e2e_s','output_s','post_first_boundary_s'):
        assert math.isfinite(row[key]) and row[key] >= 0,key
    assert row['decode_s'] > 0 and row['ttft_s'] > 0
    total = sum(row[k] for k in ('preparation_s','prefill_to_first_s','post_first_boundary_s','decode_s','output_s'))
    assert math.isclose(total,row['e2e_s'],abs_tol=1e-7,rel_tol=0)
    assert math.isclose(row['preparation_s']+row['prefill_to_first_s'],row['ttft_s'],abs_tol=1e-7,rel_tol=0)
    assert 0 < row['peak_allocated'] <= row['peak_reserved'] < row['device_total']*0.75
    assert row['cache_class'] == ('DynamicCache' if row['variant'] == 'A' else 'StaticCache')
    assert row['static_capacity'] == (None if row['variant'] == 'A' else context+64)


def stopped(root):
    """A failed numerical gate is complete evidence of a stopped investigation."""
    baseline = read(root/'baseline-identity.json')
    assert baseline['model_revision'] == REVISION
    assert baseline['parameter_count'] == 1543714304
    assert baseline['parameter_bytes'] == 2*baseline['parameter_count']
    assert (root/'diagnose.exit').read_text().strip() == '0'
    assert (root/'validation_C_2048.exit').read_text().strip() == '1'
    assert not (root/'validation_C_2048.json').exists()
    assert not list(root.glob('primary_s*.jsonl')) and not list(root.glob('heldout_s*.jsonl'))
    assert not (root/'VALIDATION_INSPECTED').exists()
    assert not (root/'PRIMARY_COMPLETE').exists()
    log = (root/'validation_C_2048.log').read_text()
    matches = re.findall(r'AssertionError: \((\d+), ([0-9.eE+-]+), ([0-9.eE+-]+)\)',log)
    assert len(matches) == 1 and 'validate_logits(logits,ref)' in log
    bad,relative,max_abs = int(matches[0][0]),float(matches[0][1]),float(matches[0][2])
    assert bad > 0 and 0 <= relative <= 0.003 and max_abs > 0.03
    assert 'Recording function=partition_0' in log and "'unique_graphs': 1" in log
    assert 'WARMUP 2' in log
    validations = []
    for length,variant in [(128,'A'),(128,'B'),(128,'C'),(2048,'A'),(2048,'B')]:
        tag = f'validation_{variant}_{length}'
        assert (root/(tag+'.exit')).read_text().strip() == '0'
        v = read(root/(tag+'.json'))
        assert (v['variant'],v['context']) == (variant,length)
        identity = v['identity']
        for key in ('weights_sha256','source','gpu','torch','transformers','attention','threads'):
            assert identity[key] == baseline[key],key
        assert len(v['cold_requests']) == 3
        assert [c['prompt'] for c in v['validation']] == ['primary','heldout']
        for check in v['validation']:
            assert check['bad_logits'] == 0 and check['relative_l2'] <= 0.003
            assert check['logits_checked'] == 64*151936
            assert math.isfinite(check['max_abs']) and math.isfinite(check['relative_l2'])
            ref = read(root/f"reference_{check['prompt']}_{length}.json")
            assert ref['weights_sha256'] == baseline['weights_sha256']
            record = dict(check['record'],variant=variant)
            checked_sample(record,ref['record'],length)
            assert len(ref['record']['output_ids']) == 64 and len(ref['record']['input_ids']) == length
            if variant == 'A' and check['prompt'] == 'primary':
                assert [c['step'] for c in check['uncached']] == [0,63]
                assert all(c['bad_logits'] == 0 and c['relative_l2'] <= 0.003 for c in check['uncached'])
        trace = events(root/f'timeline_{variant}_{length}_validated.json')
        kernels = [e['name'] for e in trace if e.get('cat') == 'kernel']
        assert sum('fmha_cutlassF_f16' in k for k in kernels) == 224
        assert not any('flash' in k.lower() for k in kernels)
        decode = summarize(trace,'E014/decode')
        stats = v['compiler_after_trace']
        assert not stats.get('graph_break')
        if variant == 'C':
            assert stats.get('stats',{}).get('unique_graphs') == 1
            assert decode['compiled_markers'] and decode['graph_launch_calls'] > 0
            assert v['compiler_before_trace'] == stats
        else:
            assert stats.get('stats',{}).get('unique_graphs',0) == 0
            assert not decode['compiled_markers'] and decode['graph_launch_calls'] == 0
        validations.append(dict(variant=variant,context=length,
            checks=[{k:c[k] for k in ('prompt','bad_logits','max_abs','relative_l2','min_top2_margin')} for c in v['validation']],
            decode_trace=decode,compiler=stats,model_load_s=identity['load_s'],static_setup_s=v['static_setup_s'],
            warmup_e2e_s=[r['e2e_s'] for r in v['cold_requests']],
            peak_reserved_bytes=max(r['peak_reserved'] for r in v['cold_requests']+
                                    [check['record'] for check in v['validation']])))
    diagnostics = []
    for length in CONTEXTS:
        trace = events(root/f'timeline_A_{length}.json')
        summary = [summarize(trace,marker) for marker in ('E014/prefill','E014/decode')]
        assert all(s['graph_launch_calls'] == 0 and not s['compiled_markers'] for s in summary)
        diagnostics.append(dict(context=length,timeline=summary))
    outcome = dict(status='stopped at full-model numerical gate, no timing campaign released',
        failed_variant='C',failed_context=2048,bad_logits=bad,logits_checked=64*151936,
        failing_fraction=bad/(64*151936),relative_l2=relative,max_abs=max_abs,
        greedy_token_assertion_preceded_failure=True,comparison_timing_samples=0,
        heldout_timing_samples=0,accepted_speedup=None,
        unchanged_precision='FP16 parameters and KV across all variants',
        diagnostics=diagnostics,completed_validations=validations,
        limitation='Failed C2048 tensor values were not saved by the original fail-fast validator. Counts/norm/max and compiler diagnostics survive. No post-failure weight rehash or C2048 timeline is claimed.')
    (root/'stopped-analysis.json').write_text(json.dumps(outcome,indent=2)+'\n')
    print(json.dumps({k:v for k,v in outcome.items() if k not in ('diagnostics','completed_validations')},indent=2))


def main(root,stage):
    def success(name):
        assert (root/(name+'.exit')).read_text().strip() == '0',name
    success('diagnose')
    baseline = read(root/'baseline-identity.json')
    assert baseline['model_revision'] == REVISION and baseline['model_id'] == 'Qwen/Qwen2.5-1.5B'
    assert baseline['parameter_count'] == 1543714304
    assert baseline['parameter_bytes'] == 2*baseline['parameter_count']
    assert baseline['attention'] == 'sdpa/EFFICIENT_ATTENTION only'
    identities = []
    audits = []
    for length,variant in itertools.product(CONTEXTS,VARIANTS):
        tag = f'validation_{variant}_{length}'
        success(tag)
        v = read(root/(tag+'.json'))
        identity = v['identity'];identities.append(identity)
        assert identity['weights_sha256'] == baseline['weights_sha256']
        assert identity['source'] == baseline['source'] and identity['gpu'] == baseline['gpu']
        assert identity['torch'] == baseline['torch'] and identity['transformers'] == baseline['transformers']
        assert len(v['cold_requests']) == 3
        assert [x['prompt'] for x in v['validation']] == ['primary','heldout']
        for check in v['validation']:
            assert check['bad_logits'] == 0 and check['relative_l2'] <= 0.003
            assert check['logits_checked'] == 64*151936
            assert math.isfinite(check['max_abs']) and math.isfinite(check['relative_l2'])
            ref = read(root/f"reference_{check['prompt']}_{length}.json")
            assert ref['weights_sha256'] == baseline['weights_sha256']
            assert check['record']['output_ids'] == ref['record']['output_ids']
            if variant == 'A' and check['prompt'] == 'primary':
                assert [c['step'] for c in check['uncached']] == [0,63]
                assert all(c['bad_logits'] == 0 and c['relative_l2'] <= 0.003 for c in check['uncached'])
        trace = events(root/f'timeline_{variant}_{length}_validated.json')
        kernels = [e['name'] for e in trace if e.get('cat') == 'kernel']
        attention = [k for k in kernels if 'fmha_cutlassF_f16' in k]
        assert len(attention) == 28*8,(variant,length,len(attention))
        assert not any('flash' in k.lower() for k in kernels)
        decode = summarize(trace,'E014/decode')
        stats = v['compiler_after_trace']
        if variant == 'C':
            assert stats.get('stats',{}).get('unique_graphs',0) >= 1
            assert not stats.get('graph_break')
            assert decode['compiled_markers'],'compiled region missing in actual timeline'
            assert decode['graph_launch_calls'] > 0,'no observed CUDA graph replay'
        else:
            assert stats.get('stats',{}).get('unique_graphs',0) == 0
            assert not decode['compiled_markers'] and decode['graph_launch_calls'] == 0
        audits.append(dict(variant=variant,context=length,decode=decode,compiler=stats,
            attention_kernels=len(attention),load_s=identity['load_s'],static_setup_s=v['static_setup_s'],
            warmup_e2e_s=[x['e2e_s'] for x in v['cold_requests']]))
    (root/'execution-audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    if stage == 'validation':
        print('PASS: all primary/held-out logits/tokens, fixed weights, six cache paths, fused attention and actual compiled graph replay')
        return
    assert not (root/'measurement-dependency-diff.txt').read_text()
    repo = Path(__file__).resolve().parents[2]
    for line in (root/'measurement-tooling.sha256').read_text().splitlines():
        expected,name = line.split(maxsplit=1)
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest() == expected,name
    def phase(prompt):
        result = []
        summaries = []
        for series in (1,2):
            for length,variant in traversal(series):
                tag = f'{prompt}_s{series}_{variant}_{length}'
                success(tag)
                ctx = read(root/(tag+'-context.json'))
                assert (ctx['prompt'],ctx['series'],ctx['context'],ctx['variant']) == (prompt,series,length,variant)
                identity = ctx['identity'];identities.append(identity)
                for key in ('weights_sha256','source','gpu','torch','transformers','attention','threads'):
                    assert identity[key] == baseline[key],key
                assert ctx['compiler_before'] == ctx['compiler_after']
                assert not ctx['compiler_before'].get('graph_break')
                graphs = ctx['compiler_before'].get('stats',{}).get('unique_graphs',0)
                assert (graphs >= 1) if variant == 'C' else graphs == 0
                rows = [json.loads(x) for x in (root/(tag+'.jsonl')).read_text().splitlines()]
                assert len(rows) == 20 and [x['sample'] for x in rows] == list(range(20))
                ref = read(root/f'reference_{prompt}_{length}.json')['record']
                for row in rows:
                    checked_sample(row,ref,length)
                    assert (row['prompt'],row['series'],row['variant']) == (prompt,series,variant)
                    assert row['weights_sha256'] == baseline['weights_sha256'] and row['model_revision'] == REVISION
                    assert row['backend'] == baseline['attention'] and row['gpu'] == baseline['gpu']
                assert len(ctx['warmups']) == 3
                for warm in ctx['warmups']:
                    warm['variant'] = variant
                    checked_sample(warm,ref,length)
                metrics = ('decode_s','ttft_s','e2e_s','prefill_to_first_s','peak_allocated','peak_reserved')
                summaries.append(dict(prompt=prompt,series=series,context=length,variant=variant,
                    **{key:statistics.median(v[key] for v in rows) for key in metrics},
                    decode_min_s=min(v['decode_s'] for v in rows),decode_max_s=max(v['decode_s'] for v in rows),
                    decode_p90_s=sorted(v['decode_s'] for v in rows)[17],load_s=identity['load_s'],
                    first_warmup_e2e_s=ctx['warmups'][0]['e2e_s'],static_setup_s=ctx['static_setup_s']))
        for length,(base,candidate) in itertools.product(CONTEXTS,PAIRS):
            gains = []
            regressions = []
            for series in (1,2):
                pair = {v['variant']:v for v in summaries if v['series'] == series and v['context'] == length}
                gains.append(gain(pair[base]['decode_s'],pair[candidate]['decode_s']))
                regressions.append({key:100*(pair[candidate][key]/pair[base][key]-1)
                    for key in ('ttft_s','e2e_s','prefill_to_first_s','peak_allocated','peak_reserved')})
            result.append(dict(context=length,baseline=base,candidate=candidate,gains_pct=gains,
                               passes=passes(gains),other_metric_changes_pct=regressions))
        return result,summaries
    primary,summaries = phase('primary')
    required = any(v['passes'] for v in primary)
    heldout = []
    if stage == 'complete' and required:
        assert not (root/'heldout-dependency-diff.txt').read_text()
        heldout,heldout_summaries = phase('heldout')
        summaries += heldout_summaries
    elif stage == 'complete':
        assert not list(root.glob('heldout_s*.jsonl'))
    summary = dict(stage=stage,primary=primary,heldout=heldout,heldout_required=required,
        confirmed=[dict(context=p['context'],baseline=p['baseline'],candidate=p['candidate'])
                   for p,h in zip(primary,heldout) if p['passes'] and h['passes']],
        request_samples=len(summaries)*20,generated_tokens=len(summaries)*20*64,
        model=baseline['model_id'],revision=REVISION,weights_sha256=baseline['weights_sha256'],
        attribution='upstream Transformers caches and PyTorch compiler/fusion/graph replay; no Marlin kernel change')
    output = 'primary-analysis.json' if stage == 'primary' else 'analysis.json'
    (root/output).write_text(json.dumps(summary,indent=2)+'\n')
    with (root/('primary-summary.csv' if stage == 'primary' else 'summary.csv')).open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(summaries[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(summaries)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    p.add_argument('--stage',choices=['stopped','validation','primary','complete'],default='stopped')
    a = p.parse_args()
    if a.stage == 'stopped':
        stopped(a.results)
    else:
        main(a.results,a.stage)
