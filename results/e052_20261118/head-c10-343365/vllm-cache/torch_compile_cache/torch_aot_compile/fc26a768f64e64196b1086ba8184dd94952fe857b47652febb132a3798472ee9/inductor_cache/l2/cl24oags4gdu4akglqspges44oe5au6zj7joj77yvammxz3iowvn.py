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


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/fc26a768f64e64196b1086ba8184dd94952fe857b47652febb132a3798472ee9/inductor_cache/so/cso4ygvjnttcjtsnzvj4qutxi5qhjh4iucpkb2622psd3fmbeb6b.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace], Original ATen: [vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   fused_add_rms_norm_maybe_inplace => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_7, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
# Graph fragment:
#   %marlin_gemm : Tensor "f16[s72, 1536][1536, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %arg6_1 : Tensor "f16[s72, 1536][1536, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg5_1 : Tensor "f16[1536][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convert_element_type_default_4 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_1), kwargs = {})
#   %convert_element_type_default_7 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_2, torch.float16), kwargs = {})
#   %mul_tensor_3 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_7, %arg5_1), kwargs = {})
#   return %buf1,%mul_tensor_3
triton_red_fused_fused_add_rms_norm_0 = async_compile.triton('triton_red_fused_fused_add_rms_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_fused_add_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 100666368}, 'kernel_num_gb': 0.075500544, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_fused_add_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 1536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tl.full([1, 1], 1536.0, tl.float32)
        tmp15 = (tmp7 / tmp14)
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tl.store(out_ptr1 + (r0_1 + 1536*x0), tmp22, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/fc26a768f64e64196b1086ba8184dd94952fe857b47652febb132a3798472ee9/inductor_cache/uu/cuubuigon3k5toqeujddklksynpm2upokmyzg3okcxf5mvosk5pv.py
# Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul], Original ATen: [aten.slice, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   getitem_2 => slice_1
#   getitem_3 => slice_2
#   mul => mul_36
#   silu => add_30, convert_element_type, convert_element_type_1, div, exp, neg
# Graph fragment:
#   %marlin_gemm_1 : Tensor "f16[s72, 17920][17920, 1]cuda:0" = PlaceHolder[target=marlin_gemm_1]
#   %slice_1 : Tensor "f16[s72, 8960][17920, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%marlin_gemm_1, 1, 0, 8960), kwargs = {})
#   %convert_element_type : Tensor "f32[s72, 8960][8960, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[s72, 8960][8960, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type,), kwargs = {})
#   %exp : Tensor "f32[s72, 8960][8960, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_30 : Tensor "f32[s72, 8960][8960, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[s72, 8960][8960, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %add_30), kwargs = {})
#   %convert_element_type_1 : Tensor "f16[s72, 8960][8960, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   %slice_2 : Tensor "f16[s72, 8960][17920, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%marlin_gemm_1, 1, 8960, 9223372036854775807), kwargs = {})
#   %mul_36 : Tensor "f16[s72, 8960][8960, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %slice_2), kwargs = {})
#   return %mul_36
triton_poi_fused_mul_silu_slice_1 = async_compile.triton('triton_poi_fused_mul_silu_slice_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_silu_slice_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 587202560}, 'kernel_num_gb': 0.44040192, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_slice_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8960)
    x1 = xindex // 8960
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 17920*x1), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (8960 + x0 + 17920*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/fc26a768f64e64196b1086ba8184dd94952fe857b47652febb132a3798472ee9/inductor_cache/oa/coaz6wor6ebdnzwtbswvc6dygltbfi62srffi2klorfuw46gqf7k.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, fused_add_rms_norm_maybe_inplace_1], Original ATen: [vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   fused_add_rms_norm_maybe_inplace => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6
#   fused_add_rms_norm_maybe_inplace_1 => add_tensor, add_tensor_1, convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %marlin_gemm_2 : Tensor "f16[s72, 1536][1536, 1]cuda:0" = PlaceHolder[target=marlin_gemm_2]
#   %marlin_gemm : Tensor "f16[s72, 1536][1536, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %arg6_1 : Tensor "f16[s72, 1536][1536, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %buf7 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf7]
#   %arg13_1 : Tensor "f16[1536][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %convert_element_type_default_4 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.float16), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm_2, torch.float32), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_6, torch.float32), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default, %convert_element_type_default_1), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor, torch.float16), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_3 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.float16), kwargs = {})
#   %mul_tensor_1 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_3, %arg13_1), kwargs = {})
#   return %buf7,%convert_element_type_default_2,%mul_tensor_1
triton_red_fused_fused_add_rms_norm_2 = async_compile.triton('triton_red_fused_fused_add_rms_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_fused_add_rms_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 176163840}, 'kernel_num_gb': 0.125832192, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_fused_add_rms_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 1536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tmp9.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 1536*x0), tmp14, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp16 + tmp23
        tmp25 = tl.full([1, 1], 1536.0, tl.float32)
        tmp26 = (tmp12 / tmp25)
        tmp27 = tl.full([1, 1], 1e-06, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp33 = tmp31 * tmp32
        tl.store(out_ptr2 + (r0_1 + 1536*x0), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/fc26a768f64e64196b1086ba8184dd94952fe857b47652febb132a3798472ee9/inductor_cache/5w/c5wrtvml5p5tccmpuwe6qtwvt7s5cfu6iipie46adkeqj2vhanht.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, view_2, chunk_2, unsqueeze_2, mul_5, unsqueeze_3, mul_6, sub_1, mul_7, mul_8, add_1, cat_2], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
# Source node to ATen node mapping:
#   add_1 => add_190
#   cat_2 => cat_1
#   chunk => split
#   chunk_2 => split_2
#   index_select => index
#   mul_5 => mul_133
#   mul_6 => mul_136
#   mul_7 => mul_141
#   mul_8 => mul_144
#   split => split_with_sizes
#   sub_1 => sub_53
#   unsqueeze_2 => unsqueeze_2
#   unsqueeze_3 => unsqueeze_3
#   view_2 => view_11
# Graph fragment:
#   %marlin_gemm_3 : Tensor "f16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=marlin_gemm_3]
#   %arg18_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f16[131072, 128][128, 1]cuda:0" = PlaceHolder[target=arg19_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%marlin_gemm_3, [1536, 256, 256], -1), kwargs = {})
#   %index : Tensor "f16[s72, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg19_1, [%arg18_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 64, -1), kwargs = {})
#   %view_11 : Tensor "f16[s72, 2, 128][2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_5, [%arg1_1, -1, 128]), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_11, 64, -1), kwargs = {})
#   %unsqueeze_2 : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_7, -2), kwargs = {})
#   %mul_133 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_11, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_8, -2), kwargs = {})
#   %mul_136 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_12, %unsqueeze_3), kwargs = {})
#   %sub_53 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_133, %mul_136), kwargs = {})
#   %mul_141 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_12, %unsqueeze_2), kwargs = {})
#   %mul_144 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_11, %unsqueeze_3), kwargs = {})
#   %add_190 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_141, %mul_144), kwargs = {})
#   %cat_1 : Tensor "f16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_53, %add_190], -1), kwargs = {})
#   return %cat_1
triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_3 = async_compile.triton('triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 21037056}, 'kernel_num_gb': 0.037814272, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/fc26a768f64e64196b1086ba8184dd94952fe857b47652febb132a3798472ee9/inductor_cache/up/cupyxkiq3fz75plkoqfcwe4oph64ghmbyc3qzcoyh65zwxemndl3.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, view_1, chunk_1, unsqueeze, mul_1, unsqueeze_1, mul_2, sub, mul_3, mul_4, add, cat], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
# Source node to ATen node mapping:
#   add => add_128
#   cat => cat
#   chunk => split
#   chunk_1 => split_1
#   index_select => index
#   mul_1 => mul_94
#   mul_2 => mul_97
#   mul_3 => mul_102
#   mul_4 => mul_105
#   split => split_with_sizes
#   sub => sub_36
#   unsqueeze => unsqueeze
#   unsqueeze_1 => unsqueeze_1
#   view_1 => view_9
# Graph fragment:
#   %marlin_gemm_3 : Tensor "f16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=marlin_gemm_3]
#   %arg18_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f16[131072, 128][128, 1]cuda:0" = PlaceHolder[target=arg19_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%marlin_gemm_3, [1536, 256, 256], -1), kwargs = {})
#   %index : Tensor "f16[s72, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg19_1, [%arg18_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 64, -1), kwargs = {})
#   %view_9 : Tensor "f16[s72, 12, 128][2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [%arg1_1, -1, 128]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_9, 64, -1), kwargs = {})
#   %unsqueeze : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_7, -2), kwargs = {})
#   %mul_94 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_9, %unsqueeze), kwargs = {})
#   %unsqueeze_1 : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_8, -2), kwargs = {})
#   %mul_97 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_10, %unsqueeze_1), kwargs = {})
#   %sub_36 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_94, %mul_97), kwargs = {})
#   %mul_102 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_10, %unsqueeze), kwargs = {})
#   %mul_105 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_9, %unsqueeze_1), kwargs = {})
#   %add_128 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_102, %mul_105), kwargs = {})
#   %cat : Tensor "f16[s72, 12, 128][1536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_36, %add_128], -1), kwargs = {})
#   return %cat
triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_4 = async_compile.triton('triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 125894656}, 'kernel_num_gb': 0.226557952, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
from torch._inductor.runtime.runtime_utils import assert_tensor_metadata
raw_stream0 = get_raw_stream(0)
buf0 = generate_example_value((8192, 1536), (1536, 1), 'cuda:0', torch.float16, 0, (8192, 1536))
arg6_1 = generate_example_value((8192, 1536), (1536, 1), 'cuda:0', torch.float16, 0, (8192, 1536))
arg5_1 = generate_example_value((1536,), (1,), 'cuda:0', torch.float16, 0, (1536,))
buf2 = generate_example_value((8192, 1536), (1536, 1), 'cuda:0', torch.float16, 0, (8192, 1536))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_fused_add_rms_norm_0.run(buf0, arg6_1, arg5_1, buf2, 8192, 1536, stream=raw_stream0)
del arg5_1, buf2

raw_stream0 = get_raw_stream(0)
buf3 = generate_example_value((8192, 17920), (17920, 1), 'cuda:0', torch.float16, 0, (8192, 17920))
buf4 = generate_example_value((8192, 8960), (8960, 1), 'cuda:0', torch.float16, 0, (8192, 8960))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_mul_silu_slice_1.run(buf3, buf4, 73400320, stream=raw_stream0)
del buf3, buf4

raw_stream0 = get_raw_stream(0)
buf5 = generate_example_value((8192, 1536), (1536, 1), 'cuda:0', torch.float16, 0, (8192, 1536))
arg13_1 = generate_example_value((1536,), (1,), 'cuda:0', torch.float16, 0, (1536,))
buf6 = generate_example_value((8192, 1536), (1536, 1), 'cuda:0', torch.float16, 0, (8192, 1536))
buf8 = generate_example_value((8192, 1536), (1536, 1), 'cuda:0', torch.float16, 0, (8192, 1536))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_fused_add_rms_norm_2.run(buf5, buf0, arg6_1, arg13_1, buf6, buf8, 8192, 1536, stream=raw_stream0)
del buf0, arg6_1, buf5, arg13_1, buf6, buf8

raw_stream0 = get_raw_stream(0)
buf9 = generate_example_value((8192, 2048), (2048, 1), 'cuda:0', torch.float16, 0, (8192, 2048))
arg18_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int64, 0, (8192,))
arg19_1 = generate_example_value((131072, 128), (128, 1), 'cuda:0', torch.float16, 0, (131072, 128))
buf10 = generate_example_value((8192, 2, 128), (256, 128, 1), 'cuda:0', torch.float16, 0, (8192, 2, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_3.run(buf9, arg18_1, arg19_1, buf10, 2097152, stream=raw_stream0)
del buf10

raw_stream0 = get_raw_stream(0)
buf11 = generate_example_value((8192, 12, 128), (1536, 128, 1), 'cuda:0', torch.float16, 0, (8192, 12, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_4.run(buf9, arg18_1, arg19_1, buf11, 12582912, stream=raw_stream0)
del buf9, arg18_1, arg19_1, buf11

"""
# AOT ID: ['1_inference']
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


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/fc26a768f64e64196b1086ba8184dd94952fe857b47652febb132a3798472ee9/inductor_cache/so/cso4ygvjnttcjtsnzvj4qutxi5qhjh4iucpkb2622psd3fmbeb6b.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace], Original ATen: [vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   fused_add_rms_norm_maybe_inplace => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_7, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
# Graph fragment:
#   %marlin_gemm : Tensor "f16[s72, 1536][1536, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %arg6_1 : Tensor "f16[s72, 1536][1536, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg5_1 : Tensor "f16[1536][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convert_element_type_default_4 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_1), kwargs = {})
#   %convert_element_type_default_7 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_2, torch.float16), kwargs = {})
#   %mul_tensor_3 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_7, %arg5_1), kwargs = {})
#   return %buf1,%mul_tensor_3
triton_red_fused_fused_add_rms_norm_0 = async_compile.triton('triton_red_fused_fused_add_rms_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_fused_add_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 100666368}, 'kernel_num_gb': 0.075500544, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_fused_add_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 1536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tl.full([1, 1], 1536.0, tl.float32)
        tmp15 = (tmp7 / tmp14)
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tl.store(out_ptr1 + (r0_1 + 1536*x0), tmp22, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/fc26a768f64e64196b1086ba8184dd94952fe857b47652febb132a3798472ee9/inductor_cache/uu/cuubuigon3k5toqeujddklksynpm2upokmyzg3okcxf5mvosk5pv.py
# Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul], Original ATen: [aten.slice, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   getitem_2 => slice_1
#   getitem_3 => slice_2
#   mul => mul_36
#   silu => add_30, convert_element_type, convert_element_type_1, div, exp, neg
# Graph fragment:
#   %marlin_gemm_1 : Tensor "f16[s72, 17920][17920, 1]cuda:0" = PlaceHolder[target=marlin_gemm_1]
#   %slice_1 : Tensor "f16[s72, 8960][17920, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%marlin_gemm_1, 1, 0, 8960), kwargs = {})
#   %convert_element_type : Tensor "f32[s72, 8960][8960, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[s72, 8960][8960, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type,), kwargs = {})
#   %exp : Tensor "f32[s72, 8960][8960, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_30 : Tensor "f32[s72, 8960][8960, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[s72, 8960][8960, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %add_30), kwargs = {})
#   %convert_element_type_1 : Tensor "f16[s72, 8960][8960, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   %slice_2 : Tensor "f16[s72, 8960][17920, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%marlin_gemm_1, 1, 8960, 9223372036854775807), kwargs = {})
#   %mul_36 : Tensor "f16[s72, 8960][8960, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %slice_2), kwargs = {})
#   return %mul_36
triton_poi_fused_mul_silu_slice_1 = async_compile.triton('triton_poi_fused_mul_silu_slice_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_silu_slice_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 587202560}, 'kernel_num_gb': 0.44040192, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_slice_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8960)
    x1 = xindex // 8960
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 17920*x1), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (8960 + x0 + 17920*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/fc26a768f64e64196b1086ba8184dd94952fe857b47652febb132a3798472ee9/inductor_cache/oa/coaz6wor6ebdnzwtbswvc6dygltbfi62srffi2klorfuw46gqf7k.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, fused_add_rms_norm_maybe_inplace_1], Original ATen: [vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   fused_add_rms_norm_maybe_inplace => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6
#   fused_add_rms_norm_maybe_inplace_1 => add_tensor, add_tensor_1, convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
# Graph fragment:
#   %marlin_gemm_2 : Tensor "f16[s72, 1536][1536, 1]cuda:0" = PlaceHolder[target=marlin_gemm_2]
#   %marlin_gemm : Tensor "f16[s72, 1536][1536, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %arg6_1 : Tensor "f16[s72, 1536][1536, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %buf7 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf7]
#   %arg13_1 : Tensor "f16[1536][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %convert_element_type_default_4 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.float16), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm_2, torch.float32), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_6, torch.float32), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default, %convert_element_type_default_1), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor, torch.float16), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_3 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.float16), kwargs = {})
#   %mul_tensor_1 : Tensor "f16[s72, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_3, %arg13_1), kwargs = {})
#   return %buf7,%convert_element_type_default_2,%mul_tensor_1
triton_red_fused_fused_add_rms_norm_2 = async_compile.triton('triton_red_fused_fused_add_rms_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=36, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_fused_add_rms_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 176163840}, 'kernel_num_gb': 0.125832192, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_fused_add_rms_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 1536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tmp9.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 1536*x0), tmp14, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp16 + tmp23
        tmp25 = tl.full([1, 1], 1536.0, tl.float32)
        tmp26 = (tmp12 / tmp25)
        tmp27 = tl.full([1, 1], 1e-06, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp33 = tmp31 * tmp32
        tl.store(out_ptr2 + (r0_1 + 1536*x0), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/fc26a768f64e64196b1086ba8184dd94952fe857b47652febb132a3798472ee9/inductor_cache/5w/c5wrtvml5p5tccmpuwe6qtwvt7s5cfu6iipie46adkeqj2vhanht.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, view_2, chunk_2, unsqueeze_2, mul_5, unsqueeze_3, mul_6, sub_1, mul_7, mul_8, add_1, cat_2], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
# Source node to ATen node mapping:
#   add_1 => add_190
#   cat_2 => cat_1
#   chunk => split
#   chunk_2 => split_2
#   index_select => index
#   mul_5 => mul_133
#   mul_6 => mul_136
#   mul_7 => mul_141
#   mul_8 => mul_144
#   split => split_with_sizes
#   sub_1 => sub_53
#   unsqueeze_2 => unsqueeze_2
#   unsqueeze_3 => unsqueeze_3
#   view_2 => view_11
# Graph fragment:
#   %marlin_gemm_3 : Tensor "f16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=marlin_gemm_3]
#   %arg18_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f16[131072, 128][128, 1]cuda:0" = PlaceHolder[target=arg19_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%marlin_gemm_3, [1536, 256, 256], -1), kwargs = {})
#   %index : Tensor "f16[s72, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg19_1, [%arg18_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 64, -1), kwargs = {})
#   %view_11 : Tensor "f16[s72, 2, 128][2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_5, [%arg1_1, -1, 128]), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_11, 64, -1), kwargs = {})
#   %unsqueeze_2 : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_7, -2), kwargs = {})
#   %mul_133 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_11, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_8, -2), kwargs = {})
#   %mul_136 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_12, %unsqueeze_3), kwargs = {})
#   %sub_53 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_133, %mul_136), kwargs = {})
#   %mul_141 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_12, %unsqueeze_2), kwargs = {})
#   %mul_144 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_11, %unsqueeze_3), kwargs = {})
#   %add_190 : Tensor "f16[s72, 2, 64][128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_141, %mul_144), kwargs = {})
#   %cat_1 : Tensor "f16[s72, 2, 128][256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_53, %add_190], -1), kwargs = {})
#   return %cat_1
triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_3 = async_compile.triton('triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 21037056}, 'kernel_num_gb': 0.037814272, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118/head-c10-343365/vllm-cache/torch_compile_cache/torch_aot_compile/fc26a768f64e64196b1086ba8184dd94952fe857b47652febb132a3798472ee9/inductor_cache/up/cupyxkiq3fz75plkoqfcwe4oph64ghmbyc3qzcoyh65zwxemndl3.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, view_1, chunk_1, unsqueeze, mul_1, unsqueeze_1, mul_2, sub, mul_3, mul_4, add, cat], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
# Source node to ATen node mapping:
#   add => add_128
#   cat => cat
#   chunk => split
#   chunk_1 => split_1
#   index_select => index
#   mul_1 => mul_94
#   mul_2 => mul_97
#   mul_3 => mul_102
#   mul_4 => mul_105
#   split => split_with_sizes
#   sub => sub_36
#   unsqueeze => unsqueeze
#   unsqueeze_1 => unsqueeze_1
#   view_1 => view_9
# Graph fragment:
#   %marlin_gemm_3 : Tensor "f16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=marlin_gemm_3]
#   %arg18_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f16[131072, 128][128, 1]cuda:0" = PlaceHolder[target=arg19_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%marlin_gemm_3, [1536, 256, 256], -1), kwargs = {})
#   %index : Tensor "f16[s72, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg19_1, [%arg18_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 64, -1), kwargs = {})
#   %view_9 : Tensor "f16[s72, 12, 128][2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [%arg1_1, -1, 128]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_9, 64, -1), kwargs = {})
#   %unsqueeze : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_7, -2), kwargs = {})
#   %mul_94 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_9, %unsqueeze), kwargs = {})
#   %unsqueeze_1 : Tensor "f16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_8, -2), kwargs = {})
#   %mul_97 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_10, %unsqueeze_1), kwargs = {})
#   %sub_36 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_94, %mul_97), kwargs = {})
#   %mul_102 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_10, %unsqueeze), kwargs = {})
#   %mul_105 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_9, %unsqueeze_1), kwargs = {})
#   %add_128 : Tensor "f16[s72, 12, 64][768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_102, %mul_105), kwargs = {})
#   %cat : Tensor "f16[s72, 12, 128][1536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_36, %add_128], -1), kwargs = {})
#   return %cat
triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_4 = async_compile.triton('triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 125894656}, 'kernel_num_gb': 0.226557952, 'kernel_flop': 0, 'backend_hash': 'D8F6EFE4642956DC043F0763331FDCB3506CB688F61A4DC31BD7C5ACAE2719F2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1 = args
        args.clear()
        s72 = arg1_1
        s80 = s72
        assert_size_stride_grouped((arg0_1, arg2_1, arg3_1), ((s72, 12, 128), (96, 3072), (12, 1536)), ((1536, 128, 1), (3072, 1), (1536, 1)), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            arg4_1 = copy_if_misaligned(arg4_1)
            # Topologically Sorted Source Nodes: [view, marlin_gemm], Original ATen: [aten.view, vllm.marlin_gemm]
            buf0 = torch.ops.vllm.marlin_gemm.default(reinterpret_tensor(arg0_1, (s72, 1536), (1536, 1), 0), arg2_1, None, arg3_1, None, None, arg4_1, None, 1125899907892224, s72, 1536, 1536, False, True)
            assert_tensor_metadata(buf0, (s72, 1536), (1536, 1), torch.float16, 'torch.ops.vllm.marlin_gemm.default')
            assert_alignment(buf0, 16, 'torch.ops.vllm.marlin_gemm.default')
            del arg0_1
            del arg2_1
            del arg3_1
            del arg4_1
            assert_size_stride_grouped((arg6_1, arg5_1), ((s72, 1536), (1536, )), ((1536, 1), (1, )), 'input')
            arg6_1 = copy_if_misaligned(arg6_1)
            arg5_1 = copy_if_misaligned(arg5_1)
            buf2 = empty_strided_cuda((s72, 1536), (1536, 1), torch.float16)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace], Original ATen: [vllm_ir.fused_add_rms_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_fused_add_rms_norm_0.run(buf0, arg6_1, arg5_1, buf2, s72, 1536, stream=raw_stream0)
            del arg5_1
            assert_size_stride_grouped((arg7_1, arg8_1), ((96, 35840), (12, 17920)), ((35840, 1), (17920, 1)), 'input')
            arg7_1 = copy_if_misaligned(arg7_1)
            arg8_1 = copy_if_misaligned(arg8_1)
            arg9_1 = copy_if_misaligned(arg9_1)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, marlin_gemm_1], Original ATen: [vllm_ir.fused_add_rms_norm, vllm.marlin_gemm]
            buf3 = torch.ops.vllm.marlin_gemm.default(buf2, arg7_1, None, arg8_1, None, None, arg9_1, None, 1125899907892224, s72, 17920, 1536, False, True)
            assert_tensor_metadata(buf3, (s72, 17920), (17920, 1), torch.float16, 'torch.ops.vllm.marlin_gemm.default')
            assert_alignment(buf3, 16, 'torch.ops.vllm.marlin_gemm.default')
            del arg7_1
            del arg8_1
            del arg9_1
            del buf2
            buf4 = empty_strided_cuda((s72, 8960), (8960, 1), torch.float16)
            # Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul], Original ATen: [aten.slice, aten.silu, aten.mul]
            triton_poi_fused_mul_silu_slice_1_xnumel = 8960*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_slice_1.run(buf3, buf4, triton_poi_fused_mul_silu_slice_1_xnumel, stream=raw_stream0)
            del buf3
            assert_size_stride_grouped((arg10_1, arg11_1), ((560, 3072), (70, 1536)), ((3072, 1), (1536, 1)), 'input')
            arg10_1 = copy_if_misaligned(arg10_1)
            arg11_1 = copy_if_misaligned(arg11_1)
            arg12_1 = copy_if_misaligned(arg12_1)
            # Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul, marlin_gemm_2], Original ATen: [aten.slice, aten.silu, aten.mul, vllm.marlin_gemm]
            buf5 = torch.ops.vllm.marlin_gemm.default(buf4, arg10_1, None, arg11_1, None, None, arg12_1, None, 1125899907892224, s72, 1536, 8960, False, True)
            assert_tensor_metadata(buf5, (s72, 1536), (1536, 1), torch.float16, 'torch.ops.vllm.marlin_gemm.default')
            assert_alignment(buf5, 16, 'torch.ops.vllm.marlin_gemm.default')
            del arg10_1
            del arg11_1
            del arg12_1
            del buf4
            assert_size_stride(arg13_1, (1536, ), (1, ), 'input')
            arg13_1 = copy_if_misaligned(arg13_1)
            buf6 = empty_strided_cuda((s72, 1536), (1536, 1), torch.float16)
            buf8 = empty_strided_cuda((s72, 1536), (1536, 1), torch.float16)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, fused_add_rms_norm_maybe_inplace_1], Original ATen: [vllm_ir.fused_add_rms_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_fused_add_rms_norm_2.run(buf5, buf0, arg6_1, arg13_1, buf6, buf8, s72, 1536, stream=raw_stream0)
            del arg13_1
            del arg6_1
            del buf0
            assert_size_stride_grouped((arg14_1, arg15_1, arg16_1), ((96, 4096), (2048, ), (12, 2048)), ((4096, 1), (1, ), (2048, 1)), 'input')
            arg14_1 = copy_if_misaligned(arg14_1)
            arg15_1 = copy_if_misaligned(arg15_1)
            arg16_1 = copy_if_misaligned(arg16_1)
            arg17_1 = copy_if_misaligned(arg17_1)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, fused_add_rms_norm_maybe_inplace_1, marlin_gemm_3], Original ATen: [vllm_ir.fused_add_rms_norm, vllm.marlin_gemm]
            buf9 = torch.ops.vllm.marlin_gemm.default(buf8, arg14_1, arg15_1, arg16_1, None, None, arg17_1, None, 1125899907892224, s72, 2048, 1536, False, True)
            assert_tensor_metadata(buf9, (s72, 2048), (2048, 1), torch.float16, 'torch.ops.vllm.marlin_gemm.default')
            assert_alignment(buf9, 16, 'torch.ops.vllm.marlin_gemm.default')
            del arg14_1
            del arg15_1
            del arg16_1
            del arg17_1
            assert_size_stride_grouped((arg18_1, arg19_1), ((s72, ), (131072, 128)), ((1, ), (128, 1)), 'input')
            arg18_1 = copy_if_misaligned(arg18_1)
            arg19_1 = copy_if_misaligned(arg19_1)
            buf10 = empty_strided_cuda((s72, 2, 128), (256, 128, 1), torch.float16)
            # Topologically Sorted Source Nodes: [split, index_select, chunk, view_2, chunk_2, unsqueeze_2, mul_5, unsqueeze_3, mul_6, sub_1, mul_7, mul_8, add_1, cat_2], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_3_xnumel = 256*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_3.run(buf9, arg18_1, arg19_1, buf10, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_3_xnumel, stream=raw_stream0)
            buf11 = reinterpret_tensor(buf8, (s72, 12, 128), (1536, 128, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [split, index_select, chunk, view_1, chunk_1, unsqueeze, mul_1, unsqueeze_1, mul_2, sub, mul_3, mul_4, add, cat], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_4_xnumel = 1536*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_4.run(buf9, arg18_1, arg19_1, buf11, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_4_xnumel, stream=raw_stream0)
            del arg18_1
            del arg19_1
            buf12 = buf5; del buf5  # reuse
        return (buf10, reinterpret_tensor(buf9, (s72, 2, 128), (2048, 128, 1), 1792), buf11, reinterpret_tensor(buf12, (s72, 12, 128), (1536, 128, 1), 0), buf6, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 12, 128), (1536, 128, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = 8192
    arg2_1 = rand_strided((96, 3072), (3072, 1), device='cuda:0', dtype=torch.int32)
    arg3_1 = rand_strided((12, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((0, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg5_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((8192, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((96, 35840), (35840, 1), device='cuda:0', dtype=torch.int32)
    arg8_1 = rand_strided((12, 17920), (17920, 1), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((0, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg10_1 = rand_strided((560, 3072), (3072, 1), device='cuda:0', dtype=torch.int32)
    arg11_1 = rand_strided((70, 1536), (1536, 1), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((0, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg13_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg14_1 = rand_strided((96, 4096), (4096, 1), device='cuda:0', dtype=torch.int32)
    arg15_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg16_1 = rand_strided((12, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg17_1 = rand_strided((0, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg18_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg19_1 = rand_strided((131072, 128), (128, 1), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
