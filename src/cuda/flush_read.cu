// Read-only L2 flush (bench --flush-read; docs/OURS.md "Timing caveat").
//
// The default flush (launch_l2_flush in dequant.cu) *writes* 2x L2 bytes. That
// leaves L2 full of dirty lines, which may be written back to DRAM while the
// timed kernel is reading its weights, so the two compete for bandwidth. This
// variant only reads a never-written buffer: it still pushes the kernel's inputs
// out of L2, but leaves nothing dirty behind.
#include "kernels.cuh"

namespace {

__global__ void l2_flush_read_kernel(const uint4* __restrict__ buf, size_t n, uint32_t* sink) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  uint32_t acc = 0;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const uint4 v = buf[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;  // keeps the loads alive; practically never taken
}

}  // namespace

void launch_l2_flush_read(const void* buf, size_t nbytes, uint32_t* sink, cudaStream_t stream) {
  constexpr int kThreads = 256;
  l2_flush_read_kernel<<<2048, kThreads, 0, stream>>>(static_cast<const uint4*>(buf),
                                                      nbytes / sizeof(uint4), sink);
  CUDA_CHECK(cudaGetLastError());
}
