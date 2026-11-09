#!/usr/bin/env python3
"""Reject damaged real E016 numerical, replay, identity and timing records."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import analyze

source = Path(sys.argv[1]).resolve()
stage = sys.argv[2] if len(sys.argv) > 2 else 'validation'
with tempfile.TemporaryDirectory(prefix='e016-negative-') as temp:
    root = Path(temp)/'evidence'
    shutil.copytree(source, root)
    def check(expected):
        try:
            with redirect_stdout(io.StringIO()):
                analyze.main(root, stage)
        except (AssertionError, FileNotFoundError):
            assert not expected, 'valid evidence rejected'
        else:
            assert expected, 'damaged evidence accepted'
    check(True)
    path = root/'validation_H_128.json'
    original = path.read_text()
    for mutation in ('logits', 'tokens', 'replay', 'identity', 'compiler', 'position', 'feedback', 'restoration', 'cache'):
        value = json.loads(original)
        if mutation == 'position': value['validation'][0]['record']['positions_seen'][1] -= 1
        if mutation == 'feedback': value['validation'][0]['record']['feedback'][1] += 1
        if mutation == 'restoration': value['validation'][0]['record']['restoration']['tail_zero'] = False
        if mutation == 'cache': value['validation'][0]['record']['cache_state'][0][0]['decode_positions'][0] = '0'*64
        if mutation == 'logits': value['validation'][0]['errors']['bad_logits'] = 1
        if mutation == 'tokens': value['validation'][0]['record']['output_ids'][0] += 1
        if mutation == 'replay': value['validation'][0]['record']['graph_replays'] = 0
        if mutation == 'identity': value['identity']['weights_sha256'] = '0'*64
        if mutation == 'compiler': value['compiler_after']['stats'] = {'unique_graphs':1}
        path.write_text(json.dumps(value)); check(False); path.write_text(original)
    if stage != 'validation':
        path = root/'primary_s1_H_128.jsonl'
        original = path.read_text()
        path.write_text('\n'.join(original.splitlines()[:-1])+'\n'); check(False)
        path.write_text(original)
        rows = [json.loads(x) for x in original.splitlines()]
        rows[0]['decode_s'] = float('nan')
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows)); check(False)
        path.write_text(original)
    check(True)
print('PASS real evidence accepted and numerical/token/replay/identity/compiler/position/feedback/restoration/cache corruptions rejected')
