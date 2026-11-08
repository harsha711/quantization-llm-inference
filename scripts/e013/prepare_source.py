#!/usr/bin/env python3
"""Extract verified corrected E006 source and add one M128 host tile selector."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ARCHIVE_SHA = 'e08be8352dfb500f0417bc8e8eedc74fbf3036ed646a2215713018ebac846e6a'
OLD = '    int num_threads = thread_tfg.num_threads;'
NEW = '''    if (e005_mode != 0) {
      STD_TORCH_CHECK(e005_mode == 1 && prob_m == 128 &&
          prob_m_split == 128 && thread_m_blocks == 4 && !m_block_size_8 &&
          a_type == vllm::kFloat16 && b_type == vllm::kU4B8 &&
          c_type == vllm::kFloat16 && s_type == vllm::kFloat16 &&
          group_size == 128 && stages == 4 && !has_bias && !has_zp &&
          !is_zp_float && !use_atomic_add && use_fp32_reduce,
          "E013 supports only the declared M128 FP16/uint4b8/G128 case");
      // Compare one existing tile, preserving grid, reservation and device code.
      thread_tfg = {64, 128, 128};
    }
    int num_threads = thread_tfg.num_threads;'''


def prepare(archive, out):
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA
    out.mkdir(parents=True, exist_ok=False)
    originals = {}
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            if not member.name.startswith('external/patched/') or not member.isfile():
                continue
            name = PurePosixPath(member.name).relative_to('external/patched')
            assert not name.is_absolute() and '..' not in name.parts
            content = source.extractfile(member).read()
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            originals[str(name)] = hashlib.sha256(content).hexdigest()
    host = out / 'csrc/quantization/marlin/marlin.cu'
    before = host.read_text()
    assert before.count(OLD) == 1
    after = before.replace(OLD, NEW)
    assert after.replace(NEW, OLD) == before
    host.write_text(after)
    for name, digest in originals.items():
        if name != 'csrc/quantization/marlin/marlin.cu':
            assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    template = (out / 'csrc/quantization/marlin/marlin_template.h').read_text()
    assert 'Join all shared readers and drained copy groups before overlay reuse.' in template
    (out / 'e013-host-tile.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='a/marlin.cu', tofile='b/marlin.cu')))
    (out / 'e013-source.json').write_text(json.dumps(dict(
        archive_sha256=ARCHIVE_SHA, original_sha256=originals,
        generated_sha256={name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                          for name in originals}, device_sources_unchanged=True), indent=2)+'\n')
    print('PASS: exact corrected E006 device source, one host-only tile selection')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    prepare(a.archive, a.out)
