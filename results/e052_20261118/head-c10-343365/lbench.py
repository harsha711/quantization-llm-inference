#!/usr/bin/env python3
"""E028: one llama.cpp timing cell (llama-cpp-python, all layers on the GPU). lbench.py GGUF REFDIR OUT VARIANT CONTEXT PROMPT SERIES

Batch 1, greedy, fixed 64 tokens, the exact input ids of our validated runs. decode_s covers 63 single-token
llama_decode calls plus argmax, like our decode metric. VARIANT LF = FP16 GGUF (compared with our GR tokens),
LQ = llama.cpp's own Q4_K_M quantization (compared with our GQ tokens; different quantizer, agreement is informational).
"""
import json
import statistics
import sys
import time
from pathlib import Path

import llama_cpp
import numpy as np

model, refdir, out, variant = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
context, prompt, series = int(sys.argv[5]), sys.argv[6], int(sys.argv[7])
ref = json.loads((refdir / f'reference-{context}-{prompt}.json').read_text())[dict(LF='GR', LQ='GQ')[variant]]
assert len(ref['input_ids']) == context
llm = llama_cpp.Llama(model_path=model, n_ctx=context + 64, n_batch=context + 64, n_gpu_layers=-1, n_threads=16, seed=0, verbose=True)
vocab = llm.n_vocab()


def pick():
    logits = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1), shape=(vocab,))
    assert np.isfinite(logits).all()
    return int(logits.argmax())


rows = []
for index in range(23):  # 3 warmups + 20 samples
    llm.reset()
    start = time.perf_counter()
    llm.eval(ref['input_ids'])
    ids = [pick()]
    first = time.perf_counter()
    for _ in range(63):
        llm.eval(ids[-1:])
        ids.append(pick())
    end = time.perf_counter()
    agree = next((i for i, (a, b) in enumerate(zip(ids, ref['output_ids'])) if a != b), 64)
    assert agree > 0 or variant == 'LQ', 'FP16 llama.cpp disagrees with FP16 Transformers on the first token'
    rows.append(dict(sample=index - 3, first_s=first - start, e2e_s=end - start, decode_s=end - first, tokens_agree=agree))
kept = rows[3:]
summary = {k: statistics.median(r[k] for r in kept) for k in ('decode_s', 'first_s', 'e2e_s')}
summary.update(decode_min=min(r['decode_s'] for r in kept), decode_max=max(r['decode_s'] for r in kept), rows=rows,
               tokens_agree_min=min(r['tokens_agree'] for r in rows), output_ids=ids, version=llama_cpp.__version__)
(out / f'{prompt}_{variant}_{context}_s{series}.json').write_text(json.dumps(summary, indent=1))
print('DONE', prompt, variant, context, series, summary['decode_s'], summary['tokens_agree_min'], flush=True)
