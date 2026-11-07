#!/usr/bin/env python3
"""CPU-only E011 evidence/selection self-checks, never a CUDA validation claim."""
import copy
import json
from pathlib import Path

import analyze
import plan

plan.self_test()
path = Path(__file__).resolve().parents[2] / 'results/e010_c4_20261107/validation.jsonl'
row = json.loads(path.read_text().splitlines()[0])
analyze.matched(row, row)
for change in ('alias', 'mode', 'operand', 'output', 'prefix'):
    candidate = copy.deepcopy(row)
    if change == 'alias':
        candidate['allocations'][1]['packed_range'] = candidate['allocations'][0]['packed_range']
    elif change == 'mode':
        candidate['cublas_math_mode'] = 0
    elif change == 'operand':
        candidate['allocations'][0]['operand_sha256']['Q'] = '0' * 64
    elif change == 'output':
        candidate['validation'][0]['num_bad'] = 1
    else:
        candidate['activation_prefix_sha256'] = '0' * 64
    try:
        analyze.matched(candidate, row)
    except AssertionError:
        pass
    else:
        raise AssertionError(change)
for value in ('nan', 'inf', 'N/A'):
    try:
        analyze.number({'metric': value}, 'metric')
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError(value)
print('PASS: CPU-only target selection and real-reference evidence checks, five corruptions/nonfinite metrics rejected')
