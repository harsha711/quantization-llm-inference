"""Check retained E001 code and summarize emitted E002 instructions."""
from collections import Counter
from pathlib import Path
import json
import re

root = Path(__file__).resolve().parent

def functions(path):
    result = {}
    for block in path.read_text().split('Function : ')[1:]:
        result[block.splitlines()[0]] = [
            ' '.join(line.split()) for line in block.splitlines()
            if re.match(r'\s*/\*', line)
        ]
    return result

saved = next(iter(functions(root / 'e001_saved.sass').values()))
rebuilt = next(iter(functions(root / 'e001_rebuilt.sass').values()))
assert saved and saved == rebuilt, 'Retained E001 emitted code changed'
specialized = functions(root / 'e002.sass')
aligned = next(v for k, v in specialized.items() if 'ILb0EE' in k)
tail = next(v for k, v in specialized.items() if 'ILb1EE' in k)
assert tail == rebuilt, 'E002 tail code differs from retained E001'
summary = {}
for name, lines in [('E001', rebuilt), ('E002_aligned', aligned), ('E002_tail', tail)]:
    instructions = [re.sub(r'/\*.*?\*/', '', line).strip() for line in lines]
    instructions = [s for s in instructions if s]
    ops = Counter(re.sub(r'^@!?U?P\d+\s+', '', s).split()[0].rstrip(';') for s in instructions)
    summary[name] = {'static_instructions_including_padding': len(instructions),
                     'opcode_histogram': dict(sorted(ops.items()))}
(root / 'sass_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(f'PASS: E001 matches {len(saved)} saved SASS lines including encodings/control words, ignoring display whitespace.')
print('PASS: E002 tail specialization matches retained E001 SASS.')
for name, stats in summary.items():
    print(name, stats['static_instructions_including_padding'], 'static instructions including padding')
