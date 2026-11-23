cmake_minimum_required(VERSION 3.22)
project(w4a16 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(w4a16_host STATIC src/quant.cpp src/synthetic.cpp src/reference.cpp src/cpu_gemm.cpp)
target_include_directories(w4a16_host PUBLIC include)
target_compile_options(w4a16_host PRIVATE -Wall -Wextra)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(w4a16_host PUBLIC OpenMP::OpenMP_CXX)
endif()

enable_testing()
add_executable(test_cpu tests/test_cpu.cpp)
target_link_libraries(test_cpu PRIVATE w4a16_host)
add_test(NAME test_cpu COMMAND test_cpu)
add_executable(cpu_bench tools/cpu_bench.cpp)
target_link_libraries(cpu_bench PRIVATE w4a16_host)
# E041: small fused CPU GEMM must pass the FP64 contract check (exit 0)
add_test(NAME cpu_bench_small COMMAND cpu_bench 5 200 264 8 3 2)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # Must be checked before enable_language(CUDA), which silently picks a default.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
      "Set -DCMAKE_CUDA_ARCHITECTURES to the target GPU's compute capability without the dot "
      "(see `nvidia-smi --query-gpu=compute_cap --format=csv`), or use scripts/build.sh.")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)

  add_executable(w4a16_bench
    src/cuda/bench.cu
    src/cuda/w4a16_naive.cu
    src/cuda/w4a16_inblock.cu
    src/cuda/w4a16_inblock_specialized.cu
    src/cuda/w4a16_inblock_rows.cu
    src/cuda/w4a16_inblock_vec.cu
    src/cuda/w4a16_tc.cu
    src/cuda/w4a16_tc_splitk.cu
    src/cuda/w4a16_tc_m32.cu
    src/cuda/dequant.cu
    src/cuda/cublas_gemm.cu
    src/cuda/ours_v0.cu
    src/cuda/ours_v1.cu
    src/cuda/ours_v2.cu
    src/cuda/ours_v3.cu
    src/cuda/ours_v3_m2.cu
    src/cuda/flush_read.cu)
  target_include_directories(w4a16_bench PRIVATE src/cuda)
  target_link_libraries(w4a16_bench PRIVATE w4a16_host CUDA::cudart CUDA::cublas)
  # -lineinfo maps SASS back to source lines in Nsight Compute without disabling optimization.
  target_compile_options(w4a16_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

  # Achievable DRAM read bandwidth: the ceiling for the bench's pct_floor column.
  add_executable(w4a16_bwprobe src/cuda/bw_probe.cu)
  target_include_directories(w4a16_bwprobe PRIVATE src/cuda)
  target_link_libraries(w4a16_bwprobe PRIVATE CUDA::cudart)
else()
  message(STATUS "No CUDA compiler found: building host-only targets (test_cpu).")
endif()
