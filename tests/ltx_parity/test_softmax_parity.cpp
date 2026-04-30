// Standalone softmax parity test on the same inputs the Gemma attention
// path computes for kq just before softmax. Loads _attn_kq_masked.bin
// from a CPU dump (or generates synthetic), runs ggml_soft_max on CPU and
// CUDA backends, diffs.

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
                              int K, int N, int B) {
    struct ggml_init_params params = {};
    params.mem_size                = 16 * 1024 * 1024;
    params.mem_buffer              = nullptr;
    params.no_alloc                = true;
    ggml_context* ctx = ggml_init(params);

    auto src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, B);
    ggml_set_name(src, "src");

    auto dst = ggml_soft_max(ctx, src);
    ggml_set_name(dst, "dst");
    ggml_set_output(dst);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) { std::fprintf(stderr, "fatal: alloc failed\n"); std::exit(1); }
    ggml_backend_tensor_set(src, src_data.data(), 0, ggml_nbytes(src));

    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(gallocr, gf)) { std::fprintf(stderr, "fatal: alloc graph\n"); std::exit(1); }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "fatal: compute failed\n"); std::exit(1);
    }
    std::vector<float> out(ggml_nelements(dst));
    ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));

    ggml_gallocr_free(gallocr);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

int main(int argc, char** argv) {
    int K = 128, N = 128, B = 16;
    std::vector<float> src;

    const char* load_path = std::getenv("SD_SOFTMAX_LOAD");
    if (load_path) {
        FILE* fp = std::fopen(load_path, "rb");
        if (!fp) { std::fprintf(stderr, "fatal: cannot open %s\n", load_path); return 1; }
        std::fseek(fp, 0, SEEK_END);
        long sz = std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        src.resize(sz / sizeof(float));
        std::fread(src.data(), 1, sz, fp);
        std::fclose(fp);
        std::printf("[load] read %ld bytes from %s\n", sz, load_path);
        // Need to also know shape. Hardcode kq shape for now.
        K = 128; N = 128; B = 16;
        if ((int)(K * N * B) != (int) src.size()) {
            std::fprintf(stderr, "size mismatch: K*N*B=%d but file has %zu floats\n",
                         K * N * B, src.size());
            return 1;
        }
    } else {
        // Synthetic kq-like values: roughly [-15, 15] like scaled attention logits
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 5.0f);
        src.resize(K * N * B);
        for (auto& x : src) x = dist(rng);
    }

    std::printf("Shape: K=%d N=%d B=%d\n", K, N, B);

    std::printf("Running CPU forward...\n");
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    auto cpu_out = run(cpu_backend, src, K, N, B);
    ggml_backend_free(cpu_backend);

#ifdef SD_USE_CUDA
    int cuda_device = 1;
    if (const char* d = std::getenv("SD_CUDA_DEVICE")) cuda_device = std::atoi(d);
    std::printf("Running CUDA forward (device %d)...\n", cuda_device);
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(cuda_device);
    if (!cuda_backend) { std::fprintf(stderr, "fatal: CUDA init failed\n"); return 1; }
    auto cuda_out = run(cuda_backend, src, K, N, B);
    ggml_backend_free(cuda_backend);
#endif

    if (cpu_out.size() != cuda_out.size()) { std::fprintf(stderr, "size mismatch\n"); return 1; }

    double max_abs = 0.0, sum_abs = 0.0, sum_cpu_abs = 0.0;
    int    argmax  = 0, n_diff = 0;
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
    return 0;
}
