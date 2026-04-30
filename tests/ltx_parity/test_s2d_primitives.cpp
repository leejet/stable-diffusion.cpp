// Standalone test: verify our axis-W / axis-H / axis-T SpaceToDepth and
// DepthToSpace ggml recipes against Python einops `rearrange(...)` outputs
// dumped by dump_s2d.py. Composition tests cover the stride patterns used
// by the VAE: (2,2,2), (1,2,2), (2,1,1).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "ltxvae_primitives.hpp"

namespace {

constexpr int B = 1;
constexpr int C = 3;
constexpr int T = 4;
constexpr int H = 6;
constexpr int W = 8;
constexpr int FACTOR = 2;

std::vector<float> load_bin(const std::string& path, size_t expected_numel) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    std::vector<float> buf(expected_numel);
    f.read(reinterpret_cast<char*>(buf.data()), expected_numel * sizeof(float));
    if (!f.good()) { std::fprintf(stderr, "short read on %s\n", path.c_str()); std::exit(2); }
    return buf;
}

enum Kind { S2D_W, S2D_H, S2D_T, S2D_222, S2D_122, S2D_211,
            D2S_W, D2S_H, D2S_T, D2S_222, D2S_122, D2S_211,
            PIXEL_NORM, PCS_NORMALIZE, PCS_UNNORMALIZE };

struct CaseSpec {
    const char* name;
    std::vector<int64_t> in_shape_ne;
    std::vector<int64_t> expected_shape_ne;
    Kind kind;
};

bool run_case(const CaseSpec& cs, const std::string& ref_dir) {
    size_t in_numel = 1, out_numel = 1;
    for (auto d : cs.in_shape_ne) in_numel *= d;
    for (auto d : cs.expected_shape_ne) out_numel *= d;

    const bool is_d2s = (cs.kind >= D2S_W && cs.kind <= D2S_211);
    const bool is_pn  = (cs.kind == PIXEL_NORM);
    const bool is_pcs = (cs.kind == PCS_NORMALIZE || cs.kind == PCS_UNNORMALIZE);
    std::string in_file, exp_file;
    if (is_pn) {
        in_file  = ref_dir + "/tensors/pn_input.bin";
        exp_file = ref_dir + "/tensors/pn_expected.bin";
    } else if (is_pcs) {
        in_file  = ref_dir + "/tensors/pcs_input.bin";
        exp_file = ref_dir + "/tensors/"
            + std::string(cs.kind == PCS_NORMALIZE ? "pcs_normalize_expected.bin"
                                                   : "pcs_unnormalize_expected.bin");
    } else {
        in_file  = ref_dir + "/tensors/" + (is_d2s ? "dinput_"    : "input_")    + cs.name + ".bin";
        exp_file = ref_dir + "/tensors/" + (is_d2s ? "dexpected_" : "expected_") + cs.name + ".bin";
    }
    auto in_data  = load_bin(in_file,  in_numel);
    auto expected = load_bin(exp_file, out_numel);

    size_t mem_size = 128 * 1024 * 1024;
    std::vector<uint8_t> mem_buf(mem_size);
    ggml_init_params params{mem_size, mem_buf.data(), false};
    ggml_context* ctx = ggml_init(params);

    ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                        cs.in_shape_ne[0], cs.in_shape_ne[1],
                                        cs.in_shape_ne[2], cs.in_shape_ne[3]);
    std::memcpy(x->data, in_data.data(), in_numel * sizeof(float));

    ggml_tensor* y = nullptr;
    ggml_tensor* mu_t = nullptr;
    ggml_tensor* sigma_t = nullptr;
    std::vector<float> mu_data, sigma_data;
    switch (cs.kind) {
        case S2D_W:    y = LTXVAE::space_to_depth_axisW(ctx, x, FACTOR); break;
        case S2D_H:    y = LTXVAE::space_to_depth_axisH(ctx, x, FACTOR); break;
        case S2D_T:    y = LTXVAE::space_to_depth_axisT(ctx, x, FACTOR); break;
        case S2D_222:  y = LTXVAE::space_to_depth(ctx, x, FACTOR, FACTOR, FACTOR); break;
        case S2D_122:  y = LTXVAE::space_to_depth(ctx, x, 1, FACTOR, FACTOR); break;
        case S2D_211:  y = LTXVAE::space_to_depth(ctx, x, FACTOR, 1, 1); break;
        case D2S_W:    y = LTXVAE::depth_to_space_axisW(ctx, x, FACTOR); break;
        case D2S_H:    y = LTXVAE::depth_to_space_axisH(ctx, x, FACTOR); break;
        case D2S_T:    y = LTXVAE::depth_to_space_axisT(ctx, x, FACTOR); break;
        case D2S_222:  y = LTXVAE::depth_to_space(ctx, x, FACTOR, FACTOR, FACTOR); break;
        case D2S_122:  y = LTXVAE::depth_to_space(ctx, x, 1, FACTOR, FACTOR); break;
        case D2S_211:  y = LTXVAE::depth_to_space(ctx, x, FACTOR, 1, 1); break;
        case PIXEL_NORM: y = LTXVAE::pixel_norm(ctx, x, 1e-8f); break;
        case PCS_NORMALIZE:
        case PCS_UNNORMALIZE: {
            int64_t C = cs.in_shape_ne[3];
            mu_data    = load_bin(ref_dir + "/tensors/pcs_mu.bin",    (size_t)C);
            sigma_data = load_bin(ref_dir + "/tensors/pcs_sigma.bin", (size_t)C);
            mu_t    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            sigma_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            std::memcpy(mu_t->data,    mu_data.data(),    C * sizeof(float));
            std::memcpy(sigma_t->data, sigma_data.data(), C * sizeof(float));
            y = (cs.kind == PCS_NORMALIZE)
                    ? LTXVAE::pcs_normalize(ctx, x, mu_t, sigma_t)
                    : LTXVAE::pcs_unnormalize(ctx, x, mu_t, sigma_t);
        } break;
    }

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    bool shape_ok = true;
    for (int i = 0; i < 4; i++) if (y->ne[i] != cs.expected_shape_ne[i]) { shape_ok = false; break; }
    if (!shape_ok) {
        std::printf("  %-18s  SHAPE_FAIL  got=[%lld,%lld,%lld,%lld] exp=[%lld,%lld,%lld,%lld]\n",
                    cs.name,
                    (long long)y->ne[0], (long long)y->ne[1], (long long)y->ne[2], (long long)y->ne[3],
                    (long long)cs.expected_shape_ne[0], (long long)cs.expected_shape_ne[1],
                    (long long)cs.expected_shape_ne[2], (long long)cs.expected_shape_ne[3]);
        ggml_free(ctx);
        return false;
    }

    const float* got = (const float*)y->data;
    float max_abs = 0.f;
    int64_t first_diff = -1;
    for (size_t i = 0; i < out_numel; i++) {
        float d = std::abs(got[i] - expected[i]);
        if (d > max_abs) { max_abs = d; if (first_diff < 0) first_diff = (int64_t)i; }
    }
    // PixelNorm / PCS involve f32 divides & rms; relax tolerance slightly.
    float tol  = (cs.kind >= PIXEL_NORM) ? 5e-6f : 1e-6f;
    bool  pass = max_abs < tol;
    std::printf("  %-18s  %s  max_abs=%.3e", cs.name, pass ? "PASS" : "FAIL", max_abs);
    if (!pass && first_diff >= 0) {
        std::printf("  first_diff_idx=%lld got=%.6f exp=%.6f",
                    (long long)first_diff, got[first_diff], expected[first_diff]);
    }
    std::printf("\n");

    ggml_free(ctx);
    return pass;
}

}  // namespace

int main() {
    const std::string ref_dir = "/tmp/s2d_ref";

    std::vector<CaseSpec> cases = {
        // SpaceToDepth
        {"axisW",   {W*FACTOR, H, T, C},                   {W, H, T, C*FACTOR},   S2D_W},
        {"axisH",   {W, H*FACTOR, T, C},                   {W, H, T, C*FACTOR},   S2D_H},
        {"axisT",   {W, H, T*FACTOR, C},                   {W, H, T, C*FACTOR},   S2D_T},
        {"full222", {W*FACTOR, H*FACTOR, T*FACTOR, C},     {W, H, T, C*8},        S2D_222},
        {"full122", {W*FACTOR, H*FACTOR, T, C},            {W, H, T, C*4},        S2D_122},
        {"full211", {W, H, T*FACTOR, C},                   {W, H, T, C*2},        S2D_211},
        // DepthToSpace (input has extra channels)
        {"axisW",   {W, H, T, C*FACTOR},                   {W*FACTOR, H, T, C},   D2S_W},
        {"axisH",   {W, H, T, C*FACTOR},                   {W, H*FACTOR, T, C},   D2S_H},
        {"axisT",   {W, H, T, C*FACTOR},                   {W, H, T*FACTOR, C},   D2S_T},
        {"full222", {W, H, T, C*8},                        {W*FACTOR, H*FACTOR, T*FACTOR, C}, D2S_222},
        {"full122", {W, H, T, C*4},                        {W*FACTOR, H*FACTOR, T, C},        D2S_122},
        {"full211", {W, H, T, C*2},                        {W, H, T*FACTOR, C},               D2S_211},
        // PixelNorm (dim=channel) and PerChannelStatistics
        {"pn",           {W, H, T, 5}, {W, H, T, 5}, PIXEL_NORM},
        {"pcs_norm",     {W, H, T, 6}, {W, H, T, 6}, PCS_NORMALIZE},
        {"pcs_unnorm",   {W, H, T, 6}, {W, H, T, 6}, PCS_UNNORMALIZE},
    };

    std::printf("SpaceToDepth primitive parity:\n");
    int pass = 0;
    for (size_t i = 0; i < cases.size(); i++) {
        if (i == 6)  std::printf("\nDepthToSpace primitive parity:\n");
        if (i == 12) std::printf("\nNorm primitives parity:\n");
        if (run_case(cases[i], ref_dir)) pass++;
    }
    std::printf("\n%d / %zu cases passed.\n", pass, cases.size());
    return (pass == (int)cases.size()) ? 0 : 3;
}
