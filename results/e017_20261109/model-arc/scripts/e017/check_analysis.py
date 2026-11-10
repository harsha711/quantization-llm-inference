#!/usr/bin/env python3
"""Reject damaged E017 JSON evidence without copying large tensor archives."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

import analyze


def main(source, stage):
    source = Path(source).resolve()
    with tempfile.TemporaryDirectory(prefix='e017-negative-') as directory:
        root = Path(directory) / 'evidence'; root.mkdir()
        # Only editable JSON/JSONL is copied. Tensor payloads and immutable logs are links.
        for path in source.iterdir():
            target = root / path.name
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.exit'):
                shutil.copyfile(path, target)
            else:
                target.symlink_to(path, target_is_directory=path.is_dir())

        def check(expected):
            try:
                with redirect_stdout(io.StringIO()):
                    analyze.main(root, stage)
            except (AssertionError, KeyError, ValueError, TypeError, FileNotFoundError):
                assert not expected, 'valid evidence rejected'
            else:
                assert expected, 'damaged evidence accepted'

        check(True)
        mutations = []
        if stage == 'stopped':
            path = root / 'primary_Q_128_s1.jsonl'
            original = path.read_text() if path.exists() else None
            path.write_text('{"sample":0,"decode_s":1.0}\n')
            check(False)
            if original is None:
                path.unlink()
            else:
                path.write_text(original)
            check(True)
            print('PASS stopped evidence and premature timing rejection')
            return
        if stage == 'prerequisite':
            path = root / 'one_layer_M1_0.json'
            for key, value in (('bad', 1), ('finite', False), ('activation_sha256', '0' * 64),
                               ('candidate_sha256', '0' * 64)):
                mutations.append((path, lambda x, key=key, value=value: x.update({key: value})))
            mutations.append((path, lambda x: x['resources'].update(candidate=1)))
            mutations.append((path, lambda x: x['workspace'].update(locks_zero=False)))
        elif stage in ('validation', 'primary', 'heldout'):
            path = root / 'validation_Q_128.json'
            mutations += [
                (path, lambda x: x['records'][0]['errors'].update(bad_logits=1)),
                (path, lambda x: x['records'][0]['record']['output_ids'].__setitem__(0, 999999)),
                (path, lambda x: x['records'][0]['record']['cache_lengths'].__setitem__(0, 1)),
                (path, lambda x: x['identity'].update(artifacts_sha256='0' * 64)),
                (path, lambda x: x['identity']['coverage'][0].update(K=128)),
                (path, lambda x: x['compiler_after'].update(stats={'unique_graphs': 1})),
                (path, lambda x: x['records'][2]['repetition'].update(bitwise_equal=False)),
                (root / 'validation_Q_128_primary_0-weights.json', lambda x: x.update(unchanged=False)),
            ]
        count = 0
        for path, mutate in mutations:
            original = path.read_text(); value = json.loads(original)
            mutate(value); path.write_text(json.dumps(value))
            check(False); path.write_text(original); count += 1
        if stage in ('primary', 'heldout'):
            path = root / 'primary_Q_128_s1.jsonl'; original = path.read_text()
            rows = [json.loads(line) for line in original.splitlines() if line.strip()]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows[:-1])); check(False); count += 1
            for key, value in (('decode_s', float('nan')), ('sample', 99), ('variant', 'R')):
                changed = copy.deepcopy(rows); changed[0][key] = value
                path.write_text(''.join(json.dumps(r) + '\n' for r in changed)); check(False); count += 1
            changed = copy.deepcopy(rows); changed[0].pop('ttft_s')
            path.write_text(''.join(json.dumps(r) + '\n' for r in changed)); check(False); count += 1
            path.write_text(original)
            path = root / 'order-primary.jsonl'; original = path.read_text()
            values = [json.loads(line) for line in original.splitlines() if line.strip()]
            values[0]['finished_utc_ns'] = values[0]['started_utc_ns']
            path.write_text(''.join(json.dumps(r) + '\n' for r in values)); check(False); count += 1
            path.write_text(original)
        check(True)
    print(f'PASS real {stage} evidence and {count} damaged-data scenarios')


if __name__ == '__main__':
    source = Path(sys.argv[1])
    stage = sys.argv[2] if len(sys.argv) > 2 else 'validation'
    assert stage in ('prerequisite', 'validation', 'primary', 'heldout', 'stopped')
    main(source, stage)
