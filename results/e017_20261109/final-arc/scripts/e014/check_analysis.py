#!/usr/bin/env python3
"""Accept the stopped E014 record and reject false pass/fallback/timing claims."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

import analyze

source = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix='e014-negative-') as tmp:
    root = Path(tmp)/'evidence'
    shutil.copytree(source,root)
    def check(expected):
        try:
            with redirect_stdout(io.StringIO()):
                analyze.stopped(root)
        except AssertionError:
            assert not expected,'valid stopped record rejected'
        else:
            assert expected,'damaged record accepted'
    check(True)
    path = root/'validation_C_2048.exit';original = path.read_text()
    path.write_text('0\n');check(False);path.write_text(original)
    path = root/'validation_C_2048.log';original = path.read_text()
    path.write_text(original.replace('AssertionError: (415,','AssertionError: (0,'))
    check(False);path.write_text(original)
    path = root/'validation_B_128.json';original = path.read_text();data = json.loads(original)
    data['validation'][0]['record']['output_ids'][0] += 1
    path.write_text(json.dumps(data));check(False);path.write_text(original)
    path = root/'validation_C_128.json';original = path.read_text();data = json.loads(original)
    data['compiler_after_trace'] = {}
    path.write_text(json.dumps(data));check(False);path.write_text(original)
    path = root/'primary_s1_A_128.jsonl';path.write_text('{}\n');check(False);path.unlink()
    original_reader = analyze.events
    def no_replay(path):
        values = original_reader(path)
        return [e for e in values if 'GraphLaunch' not in e.get('name','')]
    analyze.events = no_replay
    check(False)
    analyze.events = original_reader
    check(True)
print('PASS: stopped evidence accepted, false pass, erased failure, token mismatch, eager fallback, released timing and missing replay rejected')
