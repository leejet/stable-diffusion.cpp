// Tests whether forcing softmax to run on CPU (and feeding the result into a
// CUDA mul_mat) closes the V·softmax matmul drift in Gemma's attention.
//
// Loads real Gemma layer-0 _attn_kq_masked.bin and _attn_v.bin, then computes:
//   ref_cpu  = mul_mat(v, softmax_cpu(kq))            on CPU
//   pure_cuda = mul_mat(v, softmax_cuda(kq))           on CUDA
//   hybrid   = mul_mat(v, softmax_cpu(kq) → CUDA)     softmax on CPU, mul_mat on CUDA
//
// If `hybrid` matches `ref_cpu` much better than `pure_cuda` does, then forcing
// softmax to CPU is the win we're looking for.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef SD_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "ggml.h"

static std::vector<float> run_softmax(ggml_backend_t backend,
                                      const std::vector<float>& kq_data,
                                      int K, int N, int B) {
    ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
    ggml_context* ctx = ggml_init(params);
    auto src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, B);
    auto dst = ggml_soft_max(ctx, src);
    ggml_set_output(dst);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(src, kq_data.data(), 0, ggml_nbytes(src));
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> out(ggml_nelements(dst));
    ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));
    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static std::vector<float> run_mul_mat(ggml_backend_t backend,
                                      const std::vector<float>& src0_data,
                                      const std::vector<float>& src1_data,
                                      int K, int M, int N, int B, int r2,
                                      bool prec_f32) {
    ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
    ggml_context* ctx = ggml_init(params);
    auto src0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, B / r2);
    auto src1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, B);
    auto dst  = ggml_mul_mat(ctx, src0, src1);
    if (prec_f32) ggml_mul_mat_set_prec(dst, GGML_PREC_F32);
    ggml_set_output(dst);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(src0, src0_data.data(), 0, ggml_nbytes(src0));
    ggml_backend_tensor_set(src1, src1_data.data(), 0, ggml_nbytes(src1));
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> out(ggml_nelements(dst));
    ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));
    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static std::vector<float> load_bin(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) { std::fprintf(stderr, "fatal: cannot open %s\n", path.c_str()); std::exit(1); }
    std::fseek(fp, 0, SEEK_END);
    long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::vector<float> out(sz / sizeof(float));
    std::fread(out.data(), 1, sz, fp);
    std::fclose(fp);
    return out;
}

struct Stats {
    double max_abs   = 0;
    double mean_abs  = 0;
    double cpu_mag   = 0;
};

static Stats diff(const std::vector<float>& a, const std::vector<float>& b) {
    Stats s;
    double sum_abs = 0, sum_a_abs = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > s.max_abs) s.max_abs = d;
        sum_abs   += d;
        sum_a_abs += std::fabs((double) a[i]);
    }
    s.mean_abs = sum_abs / a.size();
    s.cpu_mag  = sum_a_abs / a.size();
    return s;
}

int main() {
    const std::string dir = std::getenv("SD_GEMMA_TAPS_DIR") ? std::getenv("SD_GEMMA_TAPS_DIR") : "/tmp/gemma_taps";
    auto kq_masked = load_bin(dir + "/_attn_kq_masked.bin");
    auto v         = load_bin(dir + "/_attn_v.bin");

    // Shapes: kq=[128, 128, 16] (K=L_k, N=L_q, B=n_head*N), v=[128, 256, 8] (K=L_k, M=d_head, B/r2=n_kv*N)
    const int K = 128, N = 128, B = 16;
    const int M = 256, B0 = 8;
    if ((int) kq_masked.size() != K * N * B) {
        std::fprintf(stderr, "kq_masked size mismatch: got %zu floats\n", kq_masked.size()); return 1;
    }
    if ((int) v.size() != K * M * B0) {
        std::fprintf(stderr, "v size mismatch: got %zu floats\n", v.size()); return 1;
    }

    ggml_backend_t cpu_bk = ggml_backend_cpu_init();
    int cuda_dev = std::getenv("SD_CUDA_DEVICE") ? std::atoi(std::getenv("SD_CUDA_DEVICE")) : 1;
    ggml_backend_t cuda_bk = ggml_backend_cuda_init(cuda_dev);
    if (!cuda_bk) { std::fprintf(stderr, "fatal: CUDA init failed\n"); return 1; }

    std::printf("Step 1: softmax on CPU\n");
    auto sm_cpu  = run_softmax(cpu_bk,  kq_masked, K, N, B);
    std::printf("Step 2: softmax on CUDA\n");
    auto sm_cuda = run_softmax(cuda_bk, kq_masked, K, N, B);
    {
        auto s = diff(sm_cpu, sm_cuda);
        std::printf("  softmax CPU vs CUDA: max=%.6e mean=%.6e cpu_mag=%.6e\n",
                    s.max_abs, s.mean_abs, s.cpu_mag);
    }

    std::printf("Step 3: kqv on CPU using softmax_cpu (reference)\n");
    auto kqv_cpu = run_mul_mat(cpu_bk, v, sm_cpu, K, M, N, B, /*r2=*/2, true);

    std::printf("Step 4a: kqv on CUDA using softmax_cuda (current)\n");
    auto kqv_cuda_pure = run_mul_mat(cuda_bk, v, sm_cuda, K, M, N, B, 2, true);
    {
        auto s = diff(kqv_cpu, kqv_cuda_pure);
        std::printf("  kqv CPU vs CUDA(pure): max=%.6e mean=%.6e cpu_mag=%.6e\n",
                    s.max_abs, s.mean_abs, s.cpu_mag);
    }

    std::printf("Step 4b: kqv on CUDA using softmax_cpu (hybrid)\n");
    auto kqv_cuda_hybrid = run_mul_mat(cuda_bk, v, sm_cpu, K, M, N, B, 2, true);
    {
        auto s = diff(kqv_cpu, kqv_cuda_hybrid);
        std::printf("  kqv CPU vs CUDA(hybrid): max=%.6e mean=%.6e cpu_mag=%.6e\n",
                    s.max_abs, s.mean_abs, s.cpu_mag);
    }

    ggml_backend_free(cuda_bk);
    ggml_backend_free(cpu_bk);
    return 0;
}
