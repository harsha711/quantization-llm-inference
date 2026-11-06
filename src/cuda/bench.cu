// Correctness check and timing for one GEMM shape.
//
//   w4a16_bench --M 16 --N 4096 --K 4096 [--group 128] [--seed 1] [--warmup 10]
//               [--iters 100] [--kernels naive,inblock,cublas,cublas_dq,dequant,unfused]
//               [--no-flush-l2] [--check-rows 64] [--nocheck] [--csv results/out.csv]
//
// Kernels
//   naive      fused W4A16, dequant in registers
//   inblock    fused W4A16, eight K slices reduced within each block
//   cublas     cuBLAS FP16 GEMM on the original FP16 weights
//   cublas_dq  cuBLAS FP16 GEMM on pre-dequantized FP16 W_hat (already on device)
//   dequant    GPU unpack+dequant into a K x N FP16 buffer
//   unfused    GPU dequant then cuBLAS, timed as one region (no flush between)
//
// Reported metrics (see also README)
//   tflops     2*M*N*K / median_s / 1e12     (0 for dequant; not a GEMM)
//   eff_gbps   logical_bytes / median_s / 1e9
//              logical_bytes is the minimum bytes if each named buffer is
//              read or written once. It is NOT measured DRAM traffic.
//     naive      |A| + |C| + |qweight| + |scales|
//     cublas     |A| + |C| + |W_fp16|
//     cublas_dq  |A| + |C| + |W_hat|
//     dequant    |qweight| + |scales| + |W_hat|          (W_hat is a write)
//     unfused    |A| + |C| + |qweight| + |scales| + |W_hat|
//                (W_hat is written by dequant and read by GEMM; counted once)
//   pass        every checked C element satisfies
//                 |got - ref| <= rtol*|ref| + atol,  atol = atol_frac * rms(ref)
//               with rtol = atol_frac = 1e-3. got is the FP16 output promoted
//               to float; ref is FP64 CPU GEMM. rel_l2 = ||got-ref||_2 / ||ref||_2.
//               rel_l2_vs_fp16 uses the original-weight CPU GEMM as ref
//               (quantization error, not a kernel bug).
//
// Timing: CUDA events around the kernel(s) only. Allocations, H2D, host quant,
// and the L2 flush are outside the timed interval. Default experiment is
// cold-L2: a store kernel writes 2*L2 bytes on the same stream immediately
// before the start event.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "kernels.cuh"
#include "w4a16/fp16_host.h"
#include "w4a16/quant.h"
#include "w4a16/reference.h"
#include "w4a16/synthetic.h"

using namespace w4a16;

namespace {

struct Options {
  int M = 16, N = 4096, K = 4096, group = 128;
  uint64_t seed = 1;
  int warmup = 10, iters = 100, check_rows = 64;
  bool flush_l2 = true;
  bool nocheck = false;
  bool zero_weights = false;
  std::vector<std::string> kernels = {"naive", "inblock", "cublas", "cublas_dq", "dequant", "unfused"};
  std::string csv;
};

[[noreturn]] void usage(const char* prog) {
  std::fprintf(stderr,
               "usage: %s [--M m] [--N n] [--K k] [--group g] [--seed s] [--warmup w]\n"
               "          [--iters i] [--kernels naive,inblock,cublas,cublas_dq,dequant,unfused]\n"
               "          [--no-flush-l2] [--check-rows r] [--nocheck] [--zero-weights] [--csv path]\n",
               prog);
  std::exit(EXIT_FAILURE);
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage(argv[0]);
      return argv[++i];
    };
    if (a == "--M") o.M = std::stoi(val());
    else if (a == "--N") o.N = std::stoi(val());
    else if (a == "--K") o.K = std::stoi(val());
    else if (a == "--group") o.group = std::stoi(val());
    else if (a == "--seed") o.seed = std::stoull(val());
    else if (a == "--warmup") o.warmup = std::stoi(val());
    else if (a == "--iters") o.iters = std::stoi(val());
    else if (a == "--check-rows") o.check_rows = std::stoi(val());
    else if (a == "--no-flush-l2") o.flush_l2 = false;
    else if (a == "--nocheck") o.nocheck = true;
    else if (a == "--zero-weights") o.zero_weights = true;
    else if (a == "--csv") o.csv = val();
    else if (a == "--kernels") {
      o.kernels.clear();
      std::stringstream ss(val());
      std::string k;
      while (std::getline(ss, k, ',')) o.kernels.push_back(k);
    } else usage(argv[0]);
  }
  if (o.M <= 0 || o.iters <= 0 || o.warmup < 0 || o.check_rows < 1) usage(argv[0]);
  validate_shape(o.K, o.N, o.group);
  return o;
}

template <typename T>
T* to_device(const std::vector<T>& h) {
  T* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

struct Timing {
  double median_us = 0, min_us = 0, max_us = 0, p90_us = 0;
};

Timing time_kernel(const std::function<void()>& launch, int warmup, int iters, void* flush_buf,
                   size_t flush_bytes) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<cudaEvent_t> start(iters), stop(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventCreate(&start[i]));
    CUDA_CHECK(cudaEventCreate(&stop[i]));
  }
  for (int i = 0; i < iters; ++i) {
    if (flush_buf) launch_l2_flush(flush_buf, flush_bytes, 0);
    CUDA_CHECK(cudaEventRecord(start[i]));
    launch();
    CUDA_CHECK(cudaEventRecord(stop[i]));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> us(iters);
  for (int i = 0; i < iters; ++i) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start[i], stop[i]));
    us[i] = ms * 1e3;
    CUDA_CHECK(cudaEventDestroy(start[i]));
    CUDA_CHECK(cudaEventDestroy(stop[i]));
  }
  std::sort(us.begin(), us.end());
  Timing t;
  t.min_us = us.front();
  t.max_us = us.back();
  t.median_us = iters % 2 ? us[iters / 2] : 0.5 * (us[iters / 2 - 1] + us[iters / 2]);
  t.p90_us = us[static_cast<size_t>(0.9 * (iters - 1))];
  return t;
}

double logical_bytes(const std::string& name, int M, int N, int K, int group) {
  const double a = static_cast<double>(fp16_bytes(M, K));
  const double c = static_cast<double>(fp16_bytes(M, N));
  const double w16 = static_cast<double>(fp16_bytes(K, N));
  const double qw = static_cast<double>(qweight_bytes(K, N));
  const double sc = static_cast<double>(scales_bytes(K, N, group));
  if (name == "naive" || name == "inblock") return a + c + qw + sc;
  if (name == "cublas" || name == "cublas_dq") return a + c + w16;
  if (name == "dequant") return qw + sc + w16;
  if (name == "unfused") return a + c + qw + sc + w16;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options o = parse(argc, argv);

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int driver = 0, runtime = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));

  const double w16_mib = fp16_bytes(o.K, o.N) / 1048576.0;
  const double qw_mib = qweight_bytes(o.K, o.N) / 1048576.0;
  const double sc_mib = scales_bytes(o.K, o.N, o.group) / 1048576.0;
  const double l2_mib = prop.l2CacheSize / 1048576.0;

  std::printf("device: %s (sm_%d%d, %d SMs, L2 %.2f MiB from cudaDeviceProp, %.2f GiB)\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount, l2_mib,
              prop.totalGlobalMem / 1073741824.0);
  std::printf("software: driver API %d, runtime %d\n", driver, runtime);
  std::printf("shape: M=%d N=%d K=%d group=%d seed=%llu flush_l2=%d warmup=%d iters=%d check_rows=%d\n",
              o.M, o.N, o.K, o.group, static_cast<unsigned long long>(o.seed), o.flush_l2, o.warmup,
              o.iters, o.check_rows);
  std::printf("footprint: FP16 weights %.4f MiB, packed INT4 %.4f MiB, FP16 scales %.4f MiB\n",
              w16_mib, qw_mib, sc_mib);
  std::printf("launch: naive grid=((N+255)/256, M)=(%d, %d) block=256; dequant grid=(%d, %d) block=256\n",
              (o.N + 255) / 256, o.M, (o.N + 255) / 256, (o.K + 7) / 8);

  std::printf("launch: inblock grid=(%d, %d) block=(32, 8), shared partials=1024 bytes\n",
              (o.N + 31) / 32, o.M);

  Problem p = make_problem(o.M, o.N, o.K, o.group, o.seed);
  if (o.zero_weights) {
    std::fill(p.W.begin(), p.W.end(), 0);
    p.qw = quantize_int4_groupwise(p.W.data(), o.K, o.N, o.group);
    std::printf("input: zero weights\n");
  }
  const std::vector<uint16_t> W_hat16 = dequantize_all_fp16(p.qw);
  const std::vector<float> W_hat = dequantize_all(p.qw);
  std::vector<float> W_orig(p.W.size());
  for (size_t i = 0; i < p.W.size(); ++i) W_orig[i] = half_to_float(p.W[i]);

  double sum_w2 = 0, sum_e2 = 0, sum_s2 = 0;
  for (size_t i = 0; i < W_orig.size(); ++i) {
    double w = W_orig[i], h = W_hat[i];
    sum_w2 += w * w;
    sum_e2 += (w - h) * (w - h);
  }
  for (uint16_t sb : p.qw.scales) {
    double s = half_to_float(sb);
    sum_s2 += s * s;
  }
  const double w_rel_l2 = sum_w2 > 0 ? std::sqrt(sum_e2 / sum_w2) : 0;
  const double pred_rel =
      sum_w2 > 0 ? std::sqrt((sum_s2 / p.qw.scales.size() / 12.0) / (sum_w2 / W_orig.size())) : 0;
  std::printf("weight quant: rel_l2(W_hat, W)=%.4e  uniform-step prediction sqrt(mean(s^2/12)/mean(w^2))=%.4e\n",
              w_rel_l2, pred_rel);
  std::printf("  (prediction assumes independent Uniform[-s/2,s/2] error per weight using that group's s)\n");

  const std::vector<int> rows = pick_check_rows(o.M, o.check_rows);
  std::vector<double> ref_quant, ref_fp16;
  if (!o.nocheck) {
    ref_quant = reference_gemm_rows(p.A.data(), W_hat.data(), o.M, o.N, o.K, rows);
    ref_fp16 = reference_gemm_rows(p.A.data(), W_orig.data(), o.M, o.N, o.K, rows);
  }

  auto* dA = reinterpret_cast<__half*>(to_device(p.A));
  auto* dW = reinterpret_cast<__half*>(to_device(p.W));
  uint32_t* dQ = to_device(p.qw.qweight);
  auto* dS = reinterpret_cast<__half*>(to_device(p.qw.scales));
  __half* dW_hat = nullptr;
  __half* dC = nullptr;
  const size_t c_elems = static_cast<size_t>(o.M) * o.N;
  const size_t w_elems = static_cast<size_t>(o.K) * o.N;
  CUDA_CHECK(cudaMalloc(&dW_hat, w_elems * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dC, c_elems * sizeof(__half)));

  void* flush_buf = nullptr;
  size_t flush_bytes = 0;
  if (o.flush_l2) {
    flush_bytes = 2 * static_cast<size_t>(prop.l2CacheSize);
    CUDA_CHECK(cudaMalloc(&flush_buf, flush_bytes));
  }
  if (o.flush_l2)
    std::printf("cache: intended cold-L2, flush buffer %.2f MiB (2x L2), store kernel (not memset)\n",
                flush_bytes / 1048576.0);
  else
    std::printf("cache: warm/reused inputs, no flush between launches\n");

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  bool need_what = !o.nocheck;
  for (const auto& name : o.kernels)
    if (name == "cublas_dq" || name == "unfused" || name == "dequant") need_what = true;

  size_t what_mismatch = 0, first = 0;
  if (need_what) {
    launch_dequant_w4a16(dQ, dS, dW_hat, o.N, o.K, o.group, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!o.nocheck) {
      std::vector<uint16_t> gpu_what(w_elems);
      CUDA_CHECK(cudaMemcpy(gpu_what.data(), dW_hat, w_elems * sizeof(__half), cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < w_elems; ++i)
        if (gpu_what[i] != W_hat16[i]) {
          if (what_mismatch == 0) first = i;
          ++what_mismatch;
        }
      std::printf("W_hat GPU vs CPU: %zu / %zu FP16 bits differ%s", what_mismatch, w_elems,
                  what_mismatch ? "" : " (bit-exact)\n");
      if (what_mismatch)
        std::printf("  first idx %zu cpu=0x%04x gpu=0x%04x\n", first, W_hat16[first], gpu_what[first]);
    } else {
      std::printf("W_hat GPU vs CPU: skipped (--nocheck)\n");
    }
  } else {
    std::printf("W_hat GPU vs CPU: skipped (not needed for this kernel list)\n");
  }

  if (what_mismatch) return EXIT_FAILURE;

  const double rtol = 1e-3, atol_frac = 1e-3;
  const double flops = 2.0 * o.M * o.N * o.K;

  FILE* csv = nullptr;
  if (!o.csv.empty()) {
    bool fresh = false;
    if (FILE* f = std::fopen(o.csv.c_str(), "r")) std::fclose(f);
    else fresh = true;
    csv = std::fopen(o.csv.c_str(), "a");
    if (!csv) {
      std::perror(o.csv.c_str());
      return EXIT_FAILURE;
    }
    if (fresh)
      std::fprintf(csv,
                   "gpu,sm,sms,driver,runtime,kernel,M,N,K,group,seed,flush_l2,warmup,iters,"
                   "check_rows,l2_mib,flush_mib,logical_bytes,median_us,min_us,max_us,p90_us,"
                   "tflops,eff_gbps,pass,max_abs_err,rel_l2_err,rel_l2_vs_fp16,w_hat_mismatches,"
                   "w_rel_l2,w_rel_l2_uniform_pred\n");
  }

  bool all_pass = what_mismatch == 0;
  std::printf("%-10s %10s %10s %10s %8s %9s %5s %10s %10s %12s\n", "kernel", "median_us", "min_us",
              "p90_us", "TFLOP/s", "eff_GB/s", "pass", "max_abs", "rel_l2", "rel_l2_fp16");
  for (const std::string& name : o.kernels) {
    std::function<void()> launch;
    const std::vector<double>* ref = nullptr;
    bool is_gemm = true;
    if (name == "naive") {
      launch = [&] { launch_w4a16_naive(dA, dQ, dS, dC, o.M, o.N, o.K, o.group, 0); };
      ref = &ref_quant;
    } else if (name == "inblock") {
      launch = [&] { launch_w4a16_inblock(dA, dQ, dS, dC, o.M, o.N, o.K, o.group, 0); };
      ref = &ref_quant;
    } else if (name == "cublas") {
      launch = [&] { launch_cublas_fp16(handle, dA, dW, dC, o.M, o.N, o.K); };
      ref = &ref_fp16;
    } else if (name == "cublas_dq") {
      launch = [&] { launch_cublas_fp16(handle, dA, dW_hat, dC, o.M, o.N, o.K); };
      ref = &ref_quant;
    } else if (name == "dequant") {
      launch = [&] { launch_dequant_w4a16(dQ, dS, dW_hat, o.N, o.K, o.group, 0); };
      is_gemm = false;
    } else if (name == "unfused") {
      launch = [&] {
        launch_dequant_w4a16(dQ, dS, dW_hat, o.N, o.K, o.group, 0);
        launch_cublas_fp16(handle, dA, dW_hat, dC, o.M, o.N, o.K);
      };
      ref = &ref_quant;
    } else {
      std::fprintf(stderr, "unknown kernel '%s'\n", name.c_str());
      return EXIT_FAILURE;
    }

    ErrorStats st{}, vs16{};
    bool pass = is_gemm ? false : (what_mismatch == 0);
    if (is_gemm && !o.nocheck) {
      CUDA_CHECK(cudaMemset(dC, 0xff, c_elems * sizeof(__half)));
      launch();
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<uint16_t> hC(c_elems);
      CUDA_CHECK(cudaMemcpy(hC.data(), dC, c_elems * sizeof(__half), cudaMemcpyDeviceToHost));
      st = compare_rows(hC.data(), *ref, rows, o.N, rtol, atol_frac);
      vs16 = compare_rows(hC.data(), ref_fp16, rows, o.N, rtol, atol_frac);
      pass = st.pass();
    } else if (is_gemm && o.nocheck) {
      pass = true;
    }
    all_pass = all_pass && pass;

    if (!pass) {
      std::fprintf(stderr, "%s failed: %zu/%zu bad outputs, max_abs=%.4e rel_l2=%.4e "
                           "rtol=%.0e atol=%.4e; stopping before timing\n",
                   name.c_str(), st.num_bad, st.count, st.max_abs, st.rel_l2, rtol, st.atol);
      if (csv) std::fclose(csv);
      return EXIT_FAILURE;
    }

    const Timing t = time_kernel(launch, o.warmup, o.iters, flush_buf, flush_bytes);
    const double bytes = logical_bytes(name, o.M, o.N, o.K, o.group);
    const double tflops = is_gemm ? flops / (t.median_us * 1e-6) / 1e12 : 0.0;
    const double gbps = bytes / (t.median_us * 1e-6) / 1e9;
    std::printf("%-10s %10.2f %10.2f %10.2f %8.3f %9.1f %5s %10.3e %10.3e %12.3e\n", name.c_str(),
                t.median_us, t.min_us, t.p90_us, tflops, gbps, pass ? "yes" : "NO", st.max_abs,
                st.rel_l2, vs16.rel_l2);
    if (is_gemm && !o.nocheck && !pass)
      std::printf("  %zu of %zu checked outputs outside |err| <= %.0e*|ref| + %.3e\n", st.num_bad,
                  st.count, rtol, st.atol);
    if (csv)
      std::fprintf(csv,
                   "\"%s\",%d%d,%d,%d,%d,%s,%d,%d,%d,%d,%llu,%d,%d,%d,%d,%.4f,%.4f,%.0f,"
                   "%.3f,%.3f,%.3f,%.3f,%.4f,%.2f,%d,%.4e,%.4e,%.4e,%zu,%.6e,%.6e\n",
                   prop.name, prop.major, prop.minor, prop.multiProcessorCount, driver, runtime,
                   name.c_str(), o.M, o.N, o.K, o.group, static_cast<unsigned long long>(o.seed),
                   o.flush_l2, o.warmup, o.iters, o.check_rows, l2_mib, flush_bytes / 1048576.0,
                   bytes, t.median_us, t.min_us, t.max_us, t.p90_us, tflops, gbps, pass ? 1 : 0,
                   st.max_abs, st.rel_l2, vs16.rel_l2, what_mismatch, w_rel_l2, pred_rel);
  }
  if (!o.nocheck)
    std::printf("checked %zu of %d rows against the FP64 CPU reference (reconstructed W_hat for "
                "naive/inblock/cublas_dq/unfused; original W for cublas)\n",
                rows.size(), o.M);

  if (!o.nocheck) {
    std::printf("checked row indices:");
    for (int row : rows) std::printf(" %d", row);
    std::printf("\n");
  }

  if (csv) std::fclose(csv);
  CUBLAS_CHECK(cublasDestroy(handle));
  for (void* ptr : {static_cast<void*>(dA), static_cast<void*>(dW), static_cast<void*>(dQ),
                    static_cast<void*>(dS), static_cast<void*>(dW_hat), static_cast<void*>(dC),
                    flush_buf})
    if (ptr) CUDA_CHECK(cudaFree(ptr));
  return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
