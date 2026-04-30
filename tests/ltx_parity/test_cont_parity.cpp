// Standalone parity test: ggml_cont(ggml_permute(...)) on CPU vs CUDA.
// Loads the same byte-identical v_proj output onto both backends, applies the
// exact reshape/permute/cont chain Gemma uses, and diffs the resulting
// contiguous tensor.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef SD_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "ggml.h"

static std::vector<float> run(ggml_backend_t backend,
                              const std::vector<float>& src_data,
                              int K, int H, int T, int N) {
    struct ggml_init_params params = {};
    params.mem_size                = 16 * 1024 * 1024;
    params.mem_buffer              = nullptr;
    params.no_alloc                = true;
    ggml_context* ctx = ggml_init(params);

    // src layout matches Gemma: ggml_reshape_4d(v_proj_out, head_dim, num_kv_heads, n_token, N)
    // = [K=head_dim, H=num_kv_heads, T=n_token, N=batch]
    auto src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, H, T, N);
    ggml_set_name(src, "src");

    // Same chain as ggml_ext_attention_ext when v comes in 4D:
    // v = ggml_ext_cont(ggml_permute(v, 1, 2, 0, 3))   → [N, n_kv_head, d_head, L_k]
    // v = ggml_reshape_3d(v, L_k, d_head, n_kv_head*N) → [N*n_kv_head, d_head, L_k]
    auto permuted = ggml_permute(ctx, src, 1, 2, 0, 3);
    auto cont     = ggml_cont(ctx, permuted);
    ggml_set_name(cont, "cont");
    ggml_set_output(cont);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cont);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "fatal: alloc_ctx_tensors failed\n");
        std::exit(1);
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, ggml_nbytes(src));

    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(gallocr, gf)) {
        std::fprintf(stderr, "fatal: alloc_graph failed\n");
        std::exit(1);
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "fatal: compute failed\n");
        std::exit(1);
    }

    std::vector<float> out(ggml_nelements(cont));
    ggml_backend_tensor_get(cont, out.data(), 0, ggml_nbytes(cont));

    ggml_gallocr_free(gallocr);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

int main(int argc, char** argv) {
    int K = 256, H = 8, T = 128, N = 1;
    if (argc >= 5) {
        K = std::atoi(argv[1]);
        H = std::atoi(argv[2]);
        T = std::atoi(argv[3]);
        N = std::atoi(argv[4]);
    }
    std::printf("Shape src=[K=%d, H=%d, T=%d, N=%d]\n", K, H, T, N);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 50.0f);  // match Gemma v_proj scale
    std::vector<float> src(K * H * T * N);
    for (auto& x : src) x = dist(rng);

    std::printf("Running CPU forward...\n");
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    auto cpu_out = run(cpu_backend, src, K, H, T, N);
    ggml_backend_free(cpu_backend);

#ifdef SD_USE_CUDA
    int cuda_device = 1;
    if (const char* d = std::getenv("SD_CUDA_DEVICE")) cuda_device = std::atoi(d);
    std::printf("Running CUDA forward (device %d)...\n", cuda_device);
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(cuda_device);
    if (!cuda_backend) {
        std::fprintf(stderr, "fatal: CUDA init failed for device %d\n", cuda_device);
        return 1;
    }
    auto cuda_out = run(cuda_backend, src, K, H, T, N);
    ggml_backend_free(cuda_backend);
#endif

    if (cpu_out.size() != cuda_out.size()) {
        std::fprintf(stderr, "fatal: output size mismatch\n");
        return 1;
    }

    double max_abs = 0.0, sum_abs = 0.0, sum_cpu_abs = 0.0;
    int    argmax  = 0;
    int    n_diff  = 0;
    for (size_t i = 0; i < cpu_out.size(); ++i) {
        double diff = std::fabs((double) cpu_out[i] - (double) cuda_out[i]);
        if (diff > 0) n_diff++;
        if (diff > max_abs) { max_abs = diff; argmax = (int) i; }
        sum_abs     += diff;
        sum_cpu_abs += std::fabs(cpu_out[i]);
    }
    std::printf("Diff: max=%.6e mean=%.6e cpu_mean_mag=%.6e n_diff=%d/%zu\n",
                max_abs, sum_abs / cpu_out.size(), sum_cpu_abs / cpu_out.size(),
                n_diff, cpu_out.size());
    std::printf("Argmax [idx %d]: CPU=%+.9e CUDA=%+.9e\n", argmax, cpu_out[argmax], cuda_out[argmax]);
    std::printf("First 6 elements:\n");
    for (int i = 0; i < 6 && i < (int) cpu_out.size(); ++i) {
        std::printf("  [%2d] CPU=%+.9e CUDA=%+.9e diff=%+.3e\n",
                    i, cpu_out[i], cuda_out[i], cuda_out[i] - cpu_out[i]);
    }
    return 0;
}
