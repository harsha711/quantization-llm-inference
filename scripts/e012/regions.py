"""Conservative single-source-line regions of the unchanged E006 kernel."""
import hashlib
from pathlib import Path
import tarfile

# Inline helper sites shared by phases remain ambiguous instead of guessing callers.
RANGES = [
    (79, 113, 'main_loop'),
    (116, 148, 'math_helper_ambiguous'),
    (151, 182, 'lock_coordination'),
    (184, 211, 'atomic_coordination'),
    (214, 672, 'setup_partition'),
    (673, 1196, 'main_loop'),
    (1202, 1260, 'block_reduction'),
    (1263, 1375, 'global_reduction_fp16_branch'),
    (1378, 1418, 'global_reduction_fp32'),
    (1423, 1542, 'epilogue'),
    (1545, 1659, 'main_loop'),
    (1660, 1662, 'stripe_drain_barrier'),
    (1663, 1763, 'post_accumulation_other'),
    (1765, 1774, 'global_reduction_control'),
    (1776, 1786, 'post_bias_atomic_other'),
    (1787, 1790, 'epilogue'),
    (1791, 1837, 'setup_partition'),
]
HELPERS = {'marlin_mma.h': 'main_loop', 'dequant.h': 'main_loop',
           'marlin.cuh': 'async_helper_ambiguous', 'marlin_dtypes.cuh': 'dtype_helper_ambiguous'}


def region(filename, line):
    name = Path(filename).name
    if not filename or line <= 0:
        return 'unmapped'
    if filename != name and '/quantization/marlin/' not in filename:
        return 'external_or_ambiguous'
    if name == 'marlin_template.h':
        return next((label for start, end, label in RANGES if start <= line <= end), 'unclassified')
    return HELPERS.get(name, 'external_or_ambiguous')


def verified_sources(baseline):
    hashes = {}
    with tarfile.open(baseline / 'matched-source.tar.gz') as archive:
        for entry in (baseline / 'patched/build-source.sha256').read_text().splitlines():
            digest, name = entry.split(maxsplit=1)
            if name.startswith('external/patched/'):
                source = archive.extractfile(name).read()
                assert hashlib.sha256(source).hexdigest() == digest, name
                hashes[name] = digest
                if name.endswith('/marlin_template.h'):
                    lines = source.decode().splitlines()
                    for line, marker in [(151, 'barrier_acquire'), (166, 'barrier_release'),
                                         (1202, 'thread_block_reduce'), (1378, 'global_reduce_fp32'),
                                         (1423, 'write_result'), (1545, 'start_pipes'),
                                         (1662, '__syncthreads'), (1767, 'barrier_acquire')]:
                        assert marker in lines[line - 1], (line, marker)
    assert len(hashes) > 5
    return hashes


def source_map(path):
    """Keep every inlined source location while counting each static PC once."""
    import csv
    filename, line = '', 0
    pcs, functions = {}, set()
    with path.open() as f:
        for row in csv.reader(f):
            if len(row) == 2 and row[0] == 'File Path':
                filename, line = row[1], 0
            elif len(row) == 2 and row[0] == 'Function Name':
                functions.add(row[1])
            elif len(row) >= 4:
                if row[0].isdigit():
                    line = int(row[0])
                if row[2].startswith('0x'):
                    assert filename, row
                    pc, sass = int(row[2], 16), row[3].strip()
                    assert sass, row
                    entry = pcs.setdefault(pc, dict(sass=sass, locations=[]))
                    assert entry['sass'] == sass, pc
                    site = dict(file=filename, line=line, region=region(filename, line))
                    if site not in entry['locations']:
                        entry['locations'].append(site)
    assert len(functions) == 1 and pcs, 'Expected one complete kernel source export'
    return pcs, next(iter(functions))


def resolve_region(locations):
    """Use an unambiguous project caller, retaining conflicting inline phases."""
    ambiguous = {'unmapped', 'external_or_ambiguous', 'unclassified', 'math_helper_ambiguous',
                 'async_helper_ambiguous', 'dtype_helper_ambiguous'}
    known = {site['region'] for site in locations} - ambiguous
    if known & {'lock_coordination', 'global_reduction_fp32', 'global_reduction_fp16_branch'}:
        known.discard('global_reduction_control')
    if len(known) == 1:
        return next(iter(known))
    if len(known) > 1:
        return 'phase_ambiguous'
    return 'inline_or_external_ambiguous' if locations else 'unmapped'
