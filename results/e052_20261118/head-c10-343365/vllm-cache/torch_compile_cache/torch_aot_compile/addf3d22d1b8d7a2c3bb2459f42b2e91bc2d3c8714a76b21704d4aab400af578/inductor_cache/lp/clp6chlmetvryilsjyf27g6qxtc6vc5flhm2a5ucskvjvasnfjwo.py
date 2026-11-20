r"""
Compile-time auto-tuning block: 

import torch
from math import inf, nan
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.async_compile import AsyncCompile

async_compile = AsyncCompile()
generate_example_value = AlgorithmSelectorCache.generate_example_value
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
get_raw_stream = torch._C._cuda_getCurrentRawStream


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/addf3d22d1b8d7a2c3bb2459f42b2e91bc2d3c8714a76b21704d4aab400af578/inductor_cache/ns/cnsy3tr52fwbseligbb7yi5uukxyn3jx5aiq3avhs5gnukrgfdut.py
# Topologically Sorted Source Nodes: [long, embedding, rms_norm_default], Original ATen: [aten._to_copy, aten.embedding, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   long => convert_element_type
#   rms_norm_default => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %arg0_1 : Tensor "i32[s72][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "f16[151936, 1536][1536, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %embedding : Tensor "f16[s72, 1536][1536, 1]cuda:0" = PlaceHolder[target=embedding]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "f16[1536][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type : Tensor "i64[s72][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int64), kwargs = {})
#   %embedding : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.float16), kwargs = {})
#   %mul_tensor_1 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, %arg3_1), kwargs = {})
#   return %embedding,%buf1,%mul_tensor_1
triton_red_fused__to_copy_embedding_rms_norm_0 = async_compile.triton('triton_red_fused__to_copy_embedding_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_embedding_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 32768, 'r0_': 100666368}, 'kernel_num_gb': 0.075533312, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_embedding_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 1536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tmp0.to(tl.int64)
        tl.device_assert(((0 <= tmp1) & (tmp1 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp1 < 151936")
        tmp3 = tl.load(in_ptr1 + (r0_1 + 1536*tmp1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tl.store(out_ptr0 + (r0_1 + 1536*x0), tmp3, r0_mask & xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tl.debug_barrier()
        tmp9 = tl.load(out_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tl.full([1, 1], 1536.0, tl.float32)
        tmp12 = (tmp7 / tmp11)
        tmp13 = tl.full([1, 1], 1e-06, tl.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp17 * tmp18
        tl.store(out_ptr2 + (r0_1 + 1536*x0), tmp19, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/addf3d22d1b8d7a2c3bb2459f42b2e91bc2d3c8714a76b21704d4aab400af578/inductor_cache/so/csosnbxex3q456t3nqcecnt2ygphzq5jdvfde53grj32mabo6hdk.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, view_1, chunk_2, unsqueeze_2, mul_4, unsqueeze_3, mul_5, sub_1, mul_6, mul_7, add_1, cat_2], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
# Source node to ATen node mapping:
#   add_1 => add_143
#   cat_2 => cat_1
#   chunk => split
#   chunk_2 => split_2
#   index_select => index
#   mul_4 => mul_81
#   mul_5 => mul_84
#   mul_6 => mul_89
#   mul_7 => mul_92
#   split => split_with_sizes
#   sub_1 => sub_38
#   unsqueeze_2 => unsqueeze_2
#   unsqueeze_3 => unsqueeze_3
#   view_1 => view_4
# Graph fragment:
#   %marlin_gemm : Tensor "f16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %arg8_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f16[131072, 128][128, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%marlin_gemm, [1536, 256, 256], -1), kwargs = {})
#   %index : Tensor "f16[s72, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg9_1, [%arg8_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 64, -1), kwargs = {})
#   %view_4 : Tensor "f16[s72, 2, 128][2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [%arg1_1, -1, 128]), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_4, 64, -1), kwargs = {})
#   %unsqueeze_2 : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_81 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_84 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_3), kwargs = {})
#   %sub_38 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_81, %mul_84), kwargs = {})
#   %mul_89 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_2), kwargs = {})
#   %mul_92 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_3), kwargs = {})
#   %add_143 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %mul_92), kwargs = {})
#   %cat_1 : Tensor "f16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_38, %add_143], -1), kwargs = {})
#   return %cat_1
triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1 = async_compile.triton('triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 21037056}, 'kernel_num_gb': 0.037814272, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 2)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = (x0).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x0).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 64, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (1536 + 128*x1 + 2048*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = (tl.full([XBLOCK], 131072, tl.int32)).to(tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp8 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp8)
    tl.device_assert(((0 <= tl.broadcast_to(tmp12, [XBLOCK])) & (tl.broadcast_to(tmp12, [XBLOCK]) < 131072)) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp12, [XBLOCK]) < 131072")
    tmp14 = tl.load(in_ptr2 + (128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr0 + (1600 + 128*x1 + 2048*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (64 + 128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 - tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tmp0 >= tmp5
    tmp23 = tl.full([1], 128, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr0 + (1600 + 128*x1 + 2048*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = (tl.full([XBLOCK], 131072, tl.int32)).to(tl.int32)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp26 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp26)
    tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 131072)) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 131072")
    tmp32 = tl.load(in_ptr2 + (128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.load(in_ptr0 + (1536 + 128*x1 + 2048*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr2 + (64 + 128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp22, tmp37, tmp38)
    tmp40 = tl.where(tmp6, tmp21, tmp39)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
''', device_str='cuda')


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/addf3d22d1b8d7a2c3bb2459f42b2e91bc2d3c8714a76b21704d4aab400af578/inductor_cache/rk/crkp4zathslstazwxaeav6cmhhr755nqin4a7zgd6otzpnxjfnzs.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, view, chunk_1, unsqueeze, mul, unsqueeze_1, mul_1, sub, mul_2, mul_3, add, cat], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
# Source node to ATen node mapping:
#   add => add_81
#   cat => cat
#   chunk => split
#   chunk_1 => split_1
#   index_select => index
#   mul => mul_42
#   mul_1 => mul_45
#   mul_2 => mul_50
#   mul_3 => mul_53
#   split => split_with_sizes
#   sub => sub_21
#   unsqueeze => unsqueeze
#   unsqueeze_1 => unsqueeze_1
#   view => view_2
# Graph fragment:
#   %marlin_gemm : Tensor "f16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %arg8_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f16[131072, 128][128, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%marlin_gemm, [1536, 256, 256], -1), kwargs = {})
#   %index : Tensor "f16[s72, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg9_1, [%arg8_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 64, -1), kwargs = {})
#   %view_2 : Tensor "f16[s72, 12, 128][2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [%arg1_1, -1, 128]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_2, 64, -1), kwargs = {})
#   %unsqueeze : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_42 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_5, %unsqueeze), kwargs = {})
#   %unsqueeze_1 : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_45 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_6, %unsqueeze_1), kwargs = {})
#   %sub_21 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_42, %mul_45), kwargs = {})
#   %mul_50 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_6, %unsqueeze), kwargs = {})
#   %mul_53 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_5, %unsqueeze_1), kwargs = {})
#   %add_81 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %mul_53), kwargs = {})
#   %cat : Tensor "f16[s72, 12, 128][1536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_21, %add_81], -1), kwargs = {})
#   return %cat
triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2 = async_compile.triton('triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 125894656}, 'kernel_num_gb': 0.226557952, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 12)
    x2 = xindex // 1536
    x4 = xindex
    tmp0 = (x0).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x0).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 64, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (128*x1 + 2048*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = (tl.full([XBLOCK], 131072, tl.int32)).to(tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp8 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp8)
    tl.device_assert(((0 <= tl.broadcast_to(tmp12, [XBLOCK])) & (tl.broadcast_to(tmp12, [XBLOCK]) < 131072)) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp12, [XBLOCK]) < 131072")
    tmp14 = tl.load(in_ptr2 + (128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr0 + (64 + 128*x1 + 2048*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (64 + 128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 - tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tmp0 >= tmp5
    tmp23 = tl.full([1], 128, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr0 + (64 + 128*x1 + 2048*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = (tl.full([XBLOCK], 131072, tl.int32)).to(tl.int32)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp26 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp26)
    tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 131072)) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 131072")
    tmp32 = tl.load(in_ptr2 + (128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.load(in_ptr0 + (128*x1 + 2048*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr2 + (64 + 128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp22, tmp37, tmp38)
    tmp40 = tl.where(tmp6, tmp21, tmp39)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
''', device_str='cuda')

async_compile.wait(globals())
del async_compile

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
with torch.cuda._DeviceGuard(0):
    raw_stream0 = get_raw_stream(0)
raw_stream0 = get_raw_stream(0)
arg0_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int32, 0, (8192,))
arg2_1 = generate_example_value((151936, 1536), (1536, 1), 'cuda:0', torch.float16, 0, (151936, 1536))
arg3_1 = generate_example_value((1536,), (1,), 'cuda:0', torch.float16, 0, (1536,))
buf0 = generate_example_value((8192, 1536), (1536, 1), 'cuda:0', torch.float16, 0, (8192, 1536))
buf2 = generate_example_value((8192, 1536), (1536, 1), 'cuda:0', torch.float16, 0, (8192, 1536))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_embedding_rms_norm_0.run(arg0_1, arg2_1, arg3_1, buf0, buf2, 8192, 1536, stream=raw_stream0)
del arg0_1, arg2_1, arg3_1, buf0, buf2

from torch._inductor.runtime.runtime_utils import assert_tensor_metadata
raw_stream0 = get_raw_stream(0)
buf3 = generate_example_value((8192, 2048), (2048, 1), 'cuda:0', torch.float16, 0, (8192, 2048))
arg8_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int64, 0, (8192,))
arg9_1 = generate_example_value((131072, 128), (128, 1), 'cuda:0', torch.float16, 0, (131072, 128))
buf4 = generate_example_value((8192, 2, 128), (256, 128, 1), 'cuda:0', torch.float16, 0, (8192, 2, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1.run(buf3, arg8_1, arg9_1, buf4, 2097152, stream=raw_stream0)
del buf4

raw_stream0 = get_raw_stream(0)
buf5 = generate_example_value((8192, 12, 128), (1536, 128, 1), 'cuda:0', torch.float16, 0, (8192, 12, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf3, arg8_1, arg9_1, buf5, 12582912, stream=raw_stream0)
del buf3, arg8_1, arg9_1, buf5

"""
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.runtime_utils import assert_tensor_metadata

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/addf3d22d1b8d7a2c3bb2459f42b2e91bc2d3c8714a76b21704d4aab400af578/inductor_cache/ns/cnsy3tr52fwbseligbb7yi5uukxyn3jx5aiq3avhs5gnukrgfdut.py
# Topologically Sorted Source Nodes: [long, embedding, rms_norm_default], Original ATen: [aten._to_copy, aten.embedding, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   long => convert_element_type
#   rms_norm_default => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %arg0_1 : Tensor "i32[s72][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "f16[151936, 1536][1536, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %embedding : Tensor "f16[s72, 1536][1536, 1]cuda:0" = PlaceHolder[target=embedding]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "f16[1536][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type : Tensor "i64[s72][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int64), kwargs = {})
#   %embedding : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.float16), kwargs = {})
#   %mul_tensor_1 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, %arg3_1), kwargs = {})
#   return %embedding,%buf1,%mul_tensor_1
triton_red_fused__to_copy_embedding_rms_norm_0 = async_compile.triton('triton_red_fused__to_copy_embedding_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_embedding_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 32768, 'r0_': 100666368}, 'kernel_num_gb': 0.075533312, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_embedding_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 1536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tmp0.to(tl.int64)
        tl.device_assert(((0 <= tmp1) & (tmp1 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp1 < 151936")
        tmp3 = tl.load(in_ptr1 + (r0_1 + 1536*tmp1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tl.store(out_ptr0 + (r0_1 + 1536*x0), tmp3, r0_mask & xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tl.debug_barrier()
        tmp9 = tl.load(out_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tl.full([1, 1], 1536.0, tl.float32)
        tmp12 = (tmp7 / tmp11)
        tmp13 = tl.full([1, 1], 1e-06, tl.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp17 * tmp18
        tl.store(out_ptr2 + (r0_1 + 1536*x0), tmp19, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/addf3d22d1b8d7a2c3bb2459f42b2e91bc2d3c8714a76b21704d4aab400af578/inductor_cache/so/csosnbxex3q456t3nqcecnt2ygphzq5jdvfde53grj32mabo6hdk.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, view_1, chunk_2, unsqueeze_2, mul_4, unsqueeze_3, mul_5, sub_1, mul_6, mul_7, add_1, cat_2], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
# Source node to ATen node mapping:
#   add_1 => add_143
#   cat_2 => cat_1
#   chunk => split
#   chunk_2 => split_2
#   index_select => index
#   mul_4 => mul_81
#   mul_5 => mul_84
#   mul_6 => mul_89
#   mul_7 => mul_92
#   split => split_with_sizes
#   sub_1 => sub_38
#   unsqueeze_2 => unsqueeze_2
#   unsqueeze_3 => unsqueeze_3
#   view_1 => view_4
# Graph fragment:
#   %marlin_gemm : Tensor "f16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %arg8_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f16[131072, 128][128, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%marlin_gemm, [1536, 256, 256], -1), kwargs = {})
#   %index : Tensor "f16[s72, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg9_1, [%arg8_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 64, -1), kwargs = {})
#   %view_4 : Tensor "f16[s72, 2, 128][2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [%arg1_1, -1, 128]), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_4, 64, -1), kwargs = {})
#   %unsqueeze_2 : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_81 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_84 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_3), kwargs = {})
#   %sub_38 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_81, %mul_84), kwargs = {})
#   %mul_89 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_2), kwargs = {})
#   %mul_92 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_3), kwargs = {})
#   %add_143 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %mul_92), kwargs = {})
#   %cat_1 : Tensor "f16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_38, %add_143], -1), kwargs = {})
#   return %cat_1
triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1 = async_compile.triton('triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 21037056}, 'kernel_num_gb': 0.037814272, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 2)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = (x0).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x0).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 64, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (1536 + 128*x1 + 2048*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = (tl.full([XBLOCK], 131072, tl.int32)).to(tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp8 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp8)
    tl.device_assert(((0 <= tl.broadcast_to(tmp12, [XBLOCK])) & (tl.broadcast_to(tmp12, [XBLOCK]) < 131072)) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp12, [XBLOCK]) < 131072")
    tmp14 = tl.load(in_ptr2 + (128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr0 + (1600 + 128*x1 + 2048*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (64 + 128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 - tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tmp0 >= tmp5
    tmp23 = tl.full([1], 128, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr0 + (1600 + 128*x1 + 2048*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = (tl.full([XBLOCK], 131072, tl.int32)).to(tl.int32)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp26 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp26)
    tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 131072)) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 131072")
    tmp32 = tl.load(in_ptr2 + (128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.load(in_ptr0 + (1536 + 128*x1 + 2048*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr2 + (64 + 128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp22, tmp37, tmp38)
    tmp40 = tl.where(tmp6, tmp21, tmp39)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
''', device_str='cuda')


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/addf3d22d1b8d7a2c3bb2459f42b2e91bc2d3c8714a76b21704d4aab400af578/inductor_cache/rk/crkp4zathslstazwxaeav6cmhhr755nqin4a7zgd6otzpnxjfnzs.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, view, chunk_1, unsqueeze, mul, unsqueeze_1, mul_1, sub, mul_2, mul_3, add, cat], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
# Source node to ATen node mapping:
#   add => add_81
#   cat => cat
#   chunk => split
#   chunk_1 => split_1
#   index_select => index
#   mul => mul_42
#   mul_1 => mul_45
#   mul_2 => mul_50
#   mul_3 => mul_53
#   split => split_with_sizes
#   sub => sub_21
#   unsqueeze => unsqueeze
#   unsqueeze_1 => unsqueeze_1
#   view => view_2
# Graph fragment:
#   %marlin_gemm : Tensor "f16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %arg8_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f16[131072, 128][128, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%marlin_gemm, [1536, 256, 256], -1), kwargs = {})
#   %index : Tensor "f16[s72, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg9_1, [%arg8_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 64, -1), kwargs = {})
#   %view_2 : Tensor "f16[s72, 12, 128][2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [%arg1_1, -1, 128]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_2, 64, -1), kwargs = {})
#   %unsqueeze : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_42 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_5, %unsqueeze), kwargs = {})
#   %unsqueeze_1 : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_45 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_6, %unsqueeze_1), kwargs = {})
#   %sub_21 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_42, %mul_45), kwargs = {})
#   %mul_50 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_6, %unsqueeze), kwargs = {})
#   %mul_53 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_5, %unsqueeze_1), kwargs = {})
#   %add_81 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %mul_53), kwargs = {})
#   %cat : Tensor "f16[s72, 12, 128][1536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_21, %add_81], -1), kwargs = {})
#   return %cat
triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2 = async_compile.triton('triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 125894656}, 'kernel_num_gb': 0.226557952, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 12)
    x2 = xindex // 1536
    x4 = xindex
    tmp0 = (x0).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x0).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 64, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (128*x1 + 2048*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = (tl.full([XBLOCK], 131072, tl.int32)).to(tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp8 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp8)
    tl.device_assert(((0 <= tl.broadcast_to(tmp12, [XBLOCK])) & (tl.broadcast_to(tmp12, [XBLOCK]) < 131072)) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp12, [XBLOCK]) < 131072")
    tmp14 = tl.load(in_ptr2 + (128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr0 + (64 + 128*x1 + 2048*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (64 + 128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 - tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tmp0 >= tmp5
    tmp23 = tl.full([1], 128, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr0 + (64 + 128*x1 + 2048*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = (tl.full([XBLOCK], 131072, tl.int32)).to(tl.int32)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp26 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp26)
    tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 131072)) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 131072")
    tmp32 = tl.load(in_ptr2 + (128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.load(in_ptr0 + (128*x1 + 2048*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr2 + (64 + 128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp22, tmp37, tmp38)
    tmp40 = tl.where(tmp6, tmp21, tmp39)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
        args.clear()
        s72 = arg1_1
        s80 = s72
        assert_size_stride_grouped((arg0_1, arg2_1, arg3_1), ((s72, ), (151936, 1536), (1536, )), ((1, ), (1536, 1), (1, )), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            buf0 = empty_strided_cuda((s72, 1536), (1536, 1), torch.float16)
            buf2 = empty_strided_cuda((s72, 1536), (1536, 1), torch.float16)
            # Topologically Sorted Source Nodes: [long, embedding, rms_norm_default], Original ATen: [aten._to_copy, aten.embedding, vllm_ir.rms_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_embedding_rms_norm_0.run(arg0_1, arg2_1, arg3_1, buf0, buf2, s72, 1536, stream=raw_stream0)
            del arg0_1
            del arg2_1
            del arg3_1
            assert_size_stride_grouped((arg4_1, arg5_1, arg6_1), ((96, 4096), (2048, ), (12, 2048)), ((4096, 1), (1, ), (2048, 1)), 'input')
            arg4_1 = copy_if_misaligned(arg4_1)
            arg5_1 = copy_if_misaligned(arg5_1)
            arg6_1 = copy_if_misaligned(arg6_1)
            arg7_1 = copy_if_misaligned(arg7_1)
            # Topologically Sorted Source Nodes: [rms_norm_default, marlin_gemm], Original ATen: [vllm_ir.rms_norm, vllm.marlin_gemm]
            buf3 = torch.ops.vllm.marlin_gemm.default(buf2, arg4_1, arg5_1, arg6_1, None, None, arg7_1, None, 1125899907892224, s72, 2048, 1536, False, True)
            assert_tensor_metadata(buf3, (s72, 2048), (2048, 1), torch.float16, 'torch.ops.vllm.marlin_gemm.default')
            assert_alignment(buf3, 16, 'torch.ops.vllm.marlin_gemm.default')
            del arg4_1
            del arg5_1
            del arg6_1
            del arg7_1
            assert_size_stride_grouped((arg8_1, arg9_1), ((s72, ), (131072, 128)), ((1, ), (128, 1)), 'input')
            arg8_1 = copy_if_misaligned(arg8_1)
            arg9_1 = copy_if_misaligned(arg9_1)
            buf4 = empty_strided_cuda((s72, 2, 128), (256, 128, 1), torch.float16)
            # Topologically Sorted Source Nodes: [split, index_select, chunk, view_1, chunk_2, unsqueeze_2, mul_4, unsqueeze_3, mul_5, sub_1, mul_6, mul_7, add_1, cat_2], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1_xnumel = 256*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1.run(buf3, arg8_1, arg9_1, buf4, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1_xnumel, stream=raw_stream0)
            buf5 = reinterpret_tensor(buf2, (s72, 12, 128), (1536, 128, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [split, index_select, chunk, view, chunk_1, unsqueeze, mul, unsqueeze_1, mul_1, sub, mul_2, mul_3, add, cat], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel = 1536*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2.run(buf3, arg8_1, arg9_1, buf5, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_2_xnumel, stream=raw_stream0)
            del arg8_1
            del arg9_1
            buf6 = empty_strided_cuda((s72, 1536), (1536, 1), torch.float16)
        return (buf4, reinterpret_tensor(buf3, (s72, 2, 128), (2048, 128, 1), 1792), buf5, reinterpret_tensor(buf6, (s72, 12, 128), (1536, 128, 1), 0), buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg1_1 = 8192
    arg2_1 = rand_strided((151936, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((96, 4096), (4096, 1), device='cuda:0', dtype=torch.int32)
    arg5_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((12, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((0, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg8_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg9_1 = rand_strided((131072, 128), (128, 1), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
