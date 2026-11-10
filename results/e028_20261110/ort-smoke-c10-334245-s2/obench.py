#!/usr/bin/env python3
"""E028: one ONNX Runtime GenAI timing cell (CUDA provider). obench.py MODELDIR REFDIR OUT VARIANT CONTEXT PROMPT SERIES

Batch 1, greedy, 64 tokens, the exact input ids of our validated runs. The first generate_next_token is prefill plus
one token; decode_s covers the next 63. VARIANT OF = FP16 export (compared with our GR tokens), OQ = the model builder's
own INT4 quantization (compared with our GQ tokens; different quantizer, agreement is informational).
ORT GenAI stops at EOS and has no ignore option: a request that ends before 64 tokens fails the cell.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch  # noqa: F401  loads the CUDA/cuDNN libraries ONNX Runtime needs
import onnxruntime_genai as og

model_dir, refdir, out, variant = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
context, prompt, series = int(sys.argv[5]), sys.argv[6], int(sys.argv[7])
ref = json.loads((refdir / f'reference-{context}-{prompt}.json').read_text())[dict(OF='GR', OQ='GQ')[variant]]
assert len(ref['input_ids']) == context
model = og.Model(model_dir)
params = og.GeneratorParams(model)
params.set_search_options(do_sample=False, max_length=context + 64, batch_size=1)
rows = []
for index in range(23):  # 3 warmups + 20 samples
    generator = og.Generator(model, params)
    start = time.perf_counter()
    generator.append_tokens(ref['input_ids'])
    generator.generate_next_token()
    first = time.perf_counter()
    for _ in range(63):
        assert not generator.is_done(), 'stopped before 64 tokens'
        generator.generate_next_token()
    end = time.perf_counter()
    ids = [int(t) for t in generator.get_sequence(0)[context:]]
    assert len(ids) == 64
    agree = next((i for i, (a, b) in enumerate(zip(ids, ref['output_ids'])) if a != b), 64)
    assert agree > 0 or variant == 'OQ', 'FP16 ONNX Runtime disagrees with FP16 Transformers on the first token'
    rows.append(dict(sample=index - 3, first_s=first - start, e2e_s=end - start, decode_s=end - first, tokens_agree=agree))
    del generator
kept = rows[3:]
summary = {k: statistics.median(r[k] for r in kept) for k in ('decode_s', 'first_s', 'e2e_s')}
summary.update(decode_min=min(r['decode_s'] for r in kept), decode_max=max(r['decode_s'] for r in kept), rows=rows,
               tokens_agree_min=min(r['tokens_agree'] for r in rows), output_ids=ids, version=og.__version__)
(out / f'{prompt}_{variant}_{context}_s{series}.json').write_text(json.dumps(summary, indent=1))
print('DONE', prompt, variant, context, series, summary['decode_s'], summary['tokens_agree_min'], flush=True)
