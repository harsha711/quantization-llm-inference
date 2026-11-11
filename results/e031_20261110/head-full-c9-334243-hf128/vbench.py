#!/usr/bin/env python3
"""E027: one vLLM timing cell. vbench.py MODELDIR REFDIR OUT CONTEXT PROMPT SERIES

Batch 1, greedy, fixed 64 tokens, the exact input ids of our validated runs. Each sample is a 1-token request
followed by a 64-token request; decode_s is their difference (63 decode steps including vLLM's per-step engine work).
"""
import json
import statistics
import sys
import time
from pathlib import Path


def main():
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    model, refdir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    context, prompt, series = int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
    label = sys.argv[7] if len(sys.argv) > 7 else 'V'  # E031: V against our GQA tokens, VH (4-bit head checkpoint) against GQAH
    ref = json.loads((refdir / f'reference-{context}-{prompt}.json').read_text())[sys.argv[8] if len(sys.argv) > 8 else 'GQ']
    assert len(ref['input_ids']) == context and len(ref['output_ids']) == 64
    llm = LLM(model=model, dtype='float16', max_model_len=context + 64, gpu_memory_utilization=0.6,
              enable_prefix_caching=False, generation_config='vllm', seed=0)

    def run(n):
        params = SamplingParams(temperature=0, max_tokens=n, min_tokens=n, ignore_eos=True, detokenize=False)
        start = time.perf_counter()
        result = llm.generate([{'prompt_token_ids': ref['input_ids']}], params, use_tqdm=False)[0]
        return time.perf_counter() - start, list(result.outputs[0].token_ids), result

    rows = []
    for index in range(23):  # 3 warmups + 20 samples
        first_s, one, _ = run(1)
        e2e_s, ids, result = run(64)
        assert len(one) == 1 and len(ids) == 64
        agree = next((i for i, (a, b) in enumerate(zip(ids, ref['output_ids'])) if a != b), 64)
        assert agree > 0, 'first token differs from our Marlin path: checkpoint or request is wrong'
        rows.append(dict(sample=index - 3, first_s=first_s, e2e_s=e2e_s, decode_s=e2e_s - first_s, tokens_agree=agree))
    kept = rows[3:]
    summary = {k: statistics.median(r[k] for r in kept) for k in ('decode_s', 'first_s', 'e2e_s')}
    summary.update(decode_min=min(r['decode_s'] for r in kept), decode_max=max(r['decode_s'] for r in kept), rows=rows,
                   tokens_agree_min=min(r['tokens_agree'] for r in rows), output_ids=ids, metrics=repr(getattr(result, 'metrics', None)),
                   vllm=vllm.__version__, torch=torch.__version__)
    (out / f'{prompt}_{label}_{context}_s{series}.json').write_text(json.dumps(summary, indent=1))
    print('DONE', prompt, 'V', context, series, summary['decode_s'], summary['tokens_agree_min'], flush=True)


if __name__ == '__main__':  # vLLM starts its engine in a child process
    main()
