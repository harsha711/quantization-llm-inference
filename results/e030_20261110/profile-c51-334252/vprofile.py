#!/usr/bin/env python3
"""E030 diagnostic: GPU kernel time per decode step of vLLM on the same weights, engine run in-process.

vprofile.py MODELDIR OUT CONTEXT...  — reads OUT/ours.json for the input ids and our tokens; same 2-token / 34-token
difference as gprofile.py. Requires identical tokens.
"""
import collections
import json
import os
import sys
from pathlib import Path

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # the profiler must see the engine's kernels


def kernels(prof):
    total = collections.defaultdict(lambda: [0, 0])
    for e in prof.events():
        if str(e.device_type).endswith('CUDA'):
            total[e.name[:120]][0] += e.time_range.end - e.time_range.start
            total[e.name[:120]][1] += 1
    return total


def per_step(short, long, steps=32):
    rows = {k: [(long[k][0] - short[k][0]) / steps, (long[k][1] - short[k][1]) / steps] for k in set(short) | set(long)}
    return dict(sorted(rows.items(), key=lambda x: -x[1][0]))


def main():
    from torch.profiler import ProfilerActivity, profile
    from vllm import LLM, SamplingParams
    model, out, contexts = sys.argv[1], Path(sys.argv[2]), [int(c) for c in sys.argv[3:]]
    ours = json.loads((out / 'ours.json').read_text())
    llm = LLM(model=model, dtype='float16', max_model_len=max(contexts) + 64, gpu_memory_utilization=0.6,
              enable_prefix_caching=False, generation_config='vllm', seed=0)

    def run(ids, n):
        params = SamplingParams(temperature=0, max_tokens=n, min_tokens=n, ignore_eos=True, detokenize=False)
        return list(llm.generate([{'prompt_token_ids': ids}], params, use_tqdm=False)[0].outputs[0].token_ids)

    summary = {}
    for context in contexts:
        ref = ours[str(context)]
        for _ in range(3):
            assert run(ref['input_ids'], 34) == ref['output_ids']
        prof = {}
        for n in (2, 34):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                assert run(ref['input_ids'], n) == ref['output_ids'][:n]
            prof[n] = kernels(p)
        table = per_step(prof[2], prof[34])
        summary[context] = dict(device_us_per_step=sum(v[0] for v in table.values()), kernel_us_and_calls_per_step=table)
        print('PROFILED', context, summary[context]['device_us_per_step'], flush=True)
    (out / 'vllm.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
