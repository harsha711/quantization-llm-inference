"""Assert retained E002 and naive control code matches saved code, including encodings."""
from pathlib import Path
import json
import re

root = Path(__file__).resolve().parent

def functions(path):
    return {re.search(r'w4a16_.*', block.splitlines()[0]).group(): [' '.join(line.split()) for line in block.splitlines()
            if re.match(r'\s*/\*', line)]
            for block in path.read_text().split('Function : ')[1:]}

summary = {}
for kernel in ('e002', 'naive'):
    saved, rebuilt = [functions(root / f'{kernel}_{tag}.sass') for tag in ('saved', 'rebuilt')]
    assert saved and saved == rebuilt, f'{kernel} device code changed'
    summary[kernel] = {name: len(lines) for name, lines in saved.items()}
    print(f'PASS: {kernel} control matches saved SASS, including encoding/control words.')
candidates = functions(root / 'e003.sass')
assert len(candidates) == 8 and all(candidates.values()), 'Expected eight row/K-tail specializations'
summary['e003_sass_lines'] = {name: len(lines) for name, lines in candidates.items()}
(root / 'sass_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('PASS: all eight E003 specializations emitted.')
