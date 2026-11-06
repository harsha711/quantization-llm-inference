"""Compare exact emitted instructions/control words, ignoring display whitespace."""
from pathlib import Path
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

original = next(iter(functions(root / 'naive_original.sass').values()))
aligned = next(v for k, v in functions(root / 'naive_specialized.sass').items() if 'ILb0EE' in k)
assert original and original == aligned, 'Aligned naive SASS differs from original'
print(f'PASS: {len(original)} SASS lines, including instruction/address/opcode and control words, match after whitespace normalization.')
