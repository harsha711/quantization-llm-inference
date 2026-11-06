#!/usr/bin/env python3
"""Generate a host-only patch and a bounded FP16/uint4b8/G128 local build."""
import argparse
import difflib
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('out', type=Path)
a = p.parse_args()
assert not a.out.exists(), a.out
shutil.copytree(a.source, a.out)
f = a.out/'csrc/quantization/marlin/marlin.cu'
original = f.read_text()
s = original

def replace(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)

replace('namespace marlin {\n', '''namespace marlin {
// E005 host-only experimental state. Each call sets it and restores it on exit.
thread_local int e005_mode = 0;
thread_local bool e005_small_layout = false;
thread_local bool e005_dry_run = false;
thread_local int64_t e005_info[20] = {};
''')
replace('int tb_m = thread_m_blocks * 16;', 'int tb_m = e005_small_layout ? 8 : thread_m_blocks * 16;')
needle = '''    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;
'''
replace(needle, needle+'''
    // Only the inspected dense FP16 uint4b8 G128 eight-row specialization.
    bool e005_eligible = major_capability == 12 && minor_capability == 0 &&
        prob_m <= 8 && prob_m_split <= 8 && m_block_size_8 &&
        thread_m_blocks == 1 && thread_k == 128 && thread_n == 128 &&
        num_threads == 256 && stages == 4 && group_size == 128 &&
        a_type == vllm::kFloat16 && b_type == vllm::kU4B8 &&
        c_type == vllm::kFloat16 && s_type == vllm::kFloat16 &&
        !has_bias && !has_zp && !is_zp_float && !use_atomic_add && use_fp32_reduce;
    int e005_budget = max_shared_mem / 2 - 1024;
    if (e005_mode && e005_eligible) {
      e005_small_layout = true;
      max_shared_mem_new = e005_budget;
      if (e005_mode == 2) blocks = 2 * sms;
    }
''')
needle = '''    bool part_use_atomic_add =
        use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;
'''
replace(needle, needle+'''
    cudaFuncAttributes attr{};
    STD_TORCH_CHECK(cudaFuncGetAttributes(&attr, kernel) == cudaSuccess);
    int e005_occupancy = 0;
    STD_TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &e005_occupancy, kernel, num_threads, max_shared_mem_new) == cudaSuccess);
    if (e005_mode && e005_eligible) {
      STD_TORCH_CHECK(e005_budget >= 41984 && attr.sharedSizeBytes == 0);
      STD_TORCH_CHECK(e005_occupancy >= 2,
                      "E005 candidate cannot fit two blocks per SM");
    }
    int64_t details[] = {blocks, num_threads, max_shared_mem_new, attr.numRegs,
        (int64_t)attr.sharedSizeBytes, (int64_t)attr.localSizeBytes,
        e005_occupancy, sms, e005_eligible && e005_mode != 0,
        thread_k, thread_n, stages, thread_m_blocks, m_block_size_8,
        max_shared_mem, e005_budget, (int64_t)attr.maxThreadsPerBlock,
        (int64_t)attr.binaryVersion, (int64_t)attr.ptxVersion,
        (int64_t)reinterpret_cast<uintptr_t>(kernel)};
    std::copy(details, details + 20, e005_info);
    if (e005_dry_run) return;
''')
f.write_text(s)
(a.out/'e005-host.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), s.splitlines(True), fromfile='a/csrc/quantization/marlin/marlin.cu', tofile='b/csrc/quantization/marlin/marlin.cu')))
# This build supports only the experiment's format. Selection priorities and all
# host checks stay upstream. No unrelated quantization kernels are instantiated.
select = []
for tm in (1,2,3,4):
    for small in ((True,False) if tm == 1 else (False,)):
        for tk,tn,th in (((128,128,256),(64,128,128),(128,64,128)) if tm==1 else ((64,256,256),(64,128,128),(128,64,128))):
            params=f'vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), {th}, {tm}, {tn//16}, {tk//16}, {str(small).lower()}, 4, 8, false'
            cond=f'a_type == vllm::kFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && thread_m_blocks == {tm} && thread_n_blocks == {tn//16} && thread_k_blocks == {tk//16} && m_block_size_8 == {str(small).lower()} && !has_zp && group_blocks == 8 && threads == {th} && !is_zp_float && stages == 4'
            select.append(f'if ({cond}) kernel = Marlin<{params}>;\n')
(f.parent/'kernel_selector.h').write_text('else '.join(select))
print('Preserved original source separately. Generated host-only patch and',len(select),'bounded specializations.')
