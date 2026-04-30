// Minimal F32 mul_mat parity test: builds a tiny graph with one mul_mat
// node, runs it on the CPU and CUDA backends with deterministic synthetic
// inputs, and diffs the outputs. Used to isolate where Gemma's attention
// kqv = mul_mat(v, softmax(kq)) drift comes from.
//
// Usage:
//   sd-mm-f32-parity [shape: K M N B]   # default 128 256 128 16
//
// Where the matmul computed is:
//   dst[i, j, b] = sum_k src0[k, i, b/r2] * src1[k, j, b]
// with src0=[K,M,B/r2], src1=[K,N,B] in ggml convention.
//
// The default shape matches Gemma 3 12B's V·softmax matmul at batch=128.

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

struct Shape {
    int K;
    int M;
    int N;
    int B;     // ne[2] of src1
    int r2;    // src1->ne[2] / src0->ne[2]
};

static std::vector<float> run(ggml_backend_t backend,
                              const std::vector<float>& src0_data,
                              const std::vector<float>& src1_data,
                              const Shape& s,
                              bool prec_f32) {
    // Allocate ggml context
    struct ggml_init_params params = {};
    params.mem_size                = 16 * 1024 * 1024;
    params.mem_buffer              = nullptr;
    params.no_alloc                = true;
    ggml_context* ctx = ggml_init(params);

    const int B0 = s.B / s.r2;
    auto src0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.K, s.M, B0);
    auto src1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.K, s.N, s.B);
    ggml_set_name(src0, "src0");
    ggml_set_name(src1, "src1");

    auto dst = ggml_mul_mat(ctx, src0, src1);
    if (prec_f32) ggml_mul_mat_set_prec(dst, GGML_PREC_F32);
    ggml_set_name(dst, "dst");
    ggml_set_output(dst);

    // Build graph
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    // Allocate backend buffer
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "fatal: ggml_backend_alloc_ctx_tensors failed\n");
        std::exit(1);
    }

    // Upload inputs
    ggml_backend_tensor_set(src0, src0_data.data(), 0, ggml_nbytes(src0));
    ggml_backend_tensor_set(src1, src1_data.data(), 0, ggml_nbytes(src1));

    // Allocate compute buffer
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(gallocr, gf)) {
        std::fprintf(stderr, "fatal: ggml_gallocr_alloc_graph failed\n");
        std::exit(1);
    }

    // Compute
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "fatal: graph compute failed\n");
        std::exit(1);
    }

    // Read output
    std::vector<float> out(ggml_nelements(dst));
    ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));

    ggml_gallocr_free(gallocr);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

int main(int argc, char** argv) {
    Shape s{128, 256, 128, 16, 2};
    if (argc >= 5) {
        s.K = std::atoi(argv[1]);
        s.M = std::atoi(argv[2]);
        s.N = std::atoi(argv[3]);
        s.B = std::atoi(argv[4]);
    }
    if (argc >= 6) s.r2 = std::atoi(argv[5]);

    std::printf("Shape: K=%d M=%d N=%d B=%d r2=%d  (src0=[%d,%d,%d], src1=[%d,%d,%d])\n",
                s.K, s.M, s.N, s.B, s.r2,
                s.K, s.M, s.B / s.r2, s.K, s.N, s.B);

    // Deterministic inputs. Default = small magnitudes; SD_MM_DIST=gemma uses
    // wider v values and proper softmax (heavy-tailed) distribution to mirror
    // Gemma 3 12B attention's V·softmax matmul.
    std::mt19937 rng(42);
    std::normal_distribution<float> dist0(0.0f, 1.0f);
    std::normal_distribution<float> dist1(0.0f, 0.05f);

    const int n0 = s.K * s.M * (s.B / s.r2);
    const int n1 = s.K * s.N * s.B;
    std::vector<float> src0(n0), src1(n1);

    const char* dist_mode = std::getenv("SD_MM_DIST");
    const char* load_dir  = std::getenv("SD_MM_LOAD");
    if (load_dir != nullptr) {
        // Load src0 = "_attn_v.bin" and src1 = "_attn_softmax.bin" from disk.
        // Override shape from the .shape file alongside.
        auto load_bin = [](const std::string& path, std::vector<float>& out) {
            FILE* fp = std::fopen(path.c_str(), "rb");
            if (!fp) { std::fprintf(stderr, "fatal: cannot open %s\n", path.c_str()); std::exit(1); }
            std::fseek(fp, 0, SEEK_END);
            long sz = std::ftell(fp);
            std::fseek(fp, 0, SEEK_SET);
            out.resize(sz / sizeof(float));
            std::fread(out.data(), 1, sz, fp);
            std::fclose(fp);
        };
        auto load_shape = [](const std::string& path) {
            FILE* fp = std::fopen(path.c_str(), "r");
            if (!fp) { std::fprintf(stderr, "fatal: cannot open %s\n", path.c_str()); std::exit(1); }
            long ne[4]; char tname[32];
            std::fscanf(fp, "%ld %ld %ld %ld %31s", &ne[0], &ne[1], &ne[2], &ne[3], tname);
            std::fclose(fp);
            return std::vector<long>{ne[0], ne[1], ne[2], ne[3]};
        };
        std::string base = load_dir;
        load_bin(base + "/_attn_v.bin", src0);
        load_bin(base + "/_attn_softmax.bin", src1);
        auto sh0 = load_shape(base + "/_attn_v.shape");
        auto sh1 = load_shape(base + "/_attn_softmax.shape");
        // src0 shape is [K, M, B/r2], src1 shape is [K, N, B]. From dumps:
        // _attn_v shape [128, 256, 8, 1] → K=128, M=256, B/r2=8
        // _attn_softmax shape [128, 128, 16, 1] → K=128, N=128, B=16
        s.K = (int) sh0[0];
        s.M = (int) sh0[1];
        s.N = (int) sh1[1];
        s.B = (int) sh1[2];
        s.r2 = s.B / (int) sh0[2];
        std::printf("[load] src0 shape=[%ld,%ld,%ld] from %s/_attn_v.bin\n", sh0[0], sh0[1], sh0[2], load_dir);
        std::printf("[load] src1 shape=[%ld,%ld,%ld] from %s/_attn_softmax.bin\n", sh1[0], sh1[1], sh1[2], load_dir);
    } else if (dist_mode && std::strcmp(dist_mode, "gemma") == 0) {
        // v values: ~N(0, 50) like Gemma's v_proj output (mean_mag ~ 54)
        std::normal_distribution<float> dv(0.0f, 50.0f);
        for (int i = 0; i < n0; ++i) src0[i] = dv(rng);
        // softmax weights: per row of K, exp normalised. Sums to 1 per row.
        for (int b = 0; b < s.B; ++b) {
            for (int n = 0; n < s.N; ++n) {
                float* row = &src1[b * s.N * s.K + n * s.K];
                float max = -1e30f;
                for (int k = 0; k < s.K; ++k) {
                    row[k] = std::normal_distribution<float>(0.0f, 5.0f)(rng);
                    if (row[k] > max) max = row[k];
                }
                float sum = 0.0f;
                for (int k = 0; k < s.K; ++k) {
                    row[k] = std::exp(row[k] - max);
                    sum += row[k];
                }
                for (int k = 0; k < s.K; ++k) row[k] /= sum;
            }
        }
    } else {
        for (int i = 0; i < n0; ++i) src0[i] = dist0(rng);
        for (int i = 0; i < n1; ++i) src1[i] = dist1(rng);
    }

    // CPU
    std::printf("Running CPU forward...\n");
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    auto cpu_out = run(cpu_backend, src0, src1, s, /*prec_f32=*/true);
    ggml_backend_free(cpu_backend);

    // CUDA
#ifdef SD_USE_CUDA
    int cuda_device = 1;
    if (const char* d = std::getenv("SD_CUDA_DEVICE")) cuda_device = std::atoi(d);
    std::printf("Running CUDA forward (device %d)...\n", cuda_device);
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(cuda_device);
    if (!cuda_backend) {
        std::fprintf(stderr, "fatal: CUDA backend init failed for device %d\n", cuda_device);
        return 1;
    }
    auto cuda_out = run(cuda_backend, src0, src1, s, /*prec_f32=*/true);
    ggml_backend_free(cuda_backend);
#else
    std::fprintf(stderr, "fatal: built without SD_USE_CUDA\n");
    return 1;
#endif

    // Diff
    if (cpu_out.size() != cuda_out.size()) {
        std::fprintf(stderr, "fatal: output size mismatch\n");
        return 1;
    }
    double max_abs = 0.0, sum_abs = 0.0, sum_cpu_abs = 0.0;
    int    argmax  = 0;
    for (size_t i = 0; i < cpu_out.size(); ++i) {
        double diff = std::fabs((double) cpu_out[i] - (double) cuda_out[i]);
        if (diff > max_abs) {
            max_abs = diff;
            argmax  = (int) i;
        }
        sum_abs     += diff;
        sum_cpu_abs += std::fabs(cpu_out[i]);
    }
    const double mean_abs   = sum_abs / cpu_out.size();
    const double cpu_mag    = sum_cpu_abs / cpu_out.size();
    std::printf("Diff: max_abs=%.6e mean_abs=%.6e cpu_mean_mag=%.6e (rel_max=%.3e rel_mean=%.3e)\n",
                max_abs, mean_abs, cpu_mag,
                cpu_mag > 0 ? max_abs / cpu_mag : 0.0,
                cpu_mag > 0 ? mean_abs / cpu_mag : 0.0);
    std::printf("Argmax element [idx %d]: CPU=%+.9e CUDA=%+.9e\n", argmax, cpu_out[argmax], cuda_out[argmax]);
    std::printf("First 6 elements:\n");
    for (int i = 0; i < 6 && i < (int) cpu_out.size(); ++i) {
        std::printf("  [%2d] CPU=%+.9e CUDA=%+.9e diff=%+.3e\n",
                    i, cpu_out[i], cuda_out[i], cuda_out[i] - cpu_out[i]);
    }

    return 0;
}
