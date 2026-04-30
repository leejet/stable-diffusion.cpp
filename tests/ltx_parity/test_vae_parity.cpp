// LTX-2 VAE C++ parity test.
//
// Loads /tmp/vae_ref/state_dict.safetensors (from dump_vae.py) plus the per-stage
// reference trace tensors, runs our C++ VideoEncoder + VideoDecoder, and diffs
// each stage against the Python reference.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ltxvae.hpp"

// Standalone GGMLRunner that wraps a single LTXVAE::TimestepEmbedder block so we can
// isolate the sinusoidal + 2-linear path from the full VAE pipeline.
struct TERunner : public GGMLRunner {
    LTXVAE::TimestepEmbedder te;

    TERunner(ggml_backend_t backend, bool offload, const String2TensorStorage& tsm,
             const std::string& prefix, int embedding_dim)
        : GGMLRunner(backend, offload), te(embedding_dim) {
        te.init(params_ctx, tsm, prefix);
    }
    std::string get_desc() override { return "ltx2_vae_te_probe"; }
    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) {
        te.get_param_tensors(tensors, prefix);
    }
    sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& timestep) {
        auto get_g = [&]() -> ggml_cgraph* {
            ggml_cgraph* gf = ggml_new_graph(compute_ctx);
            ggml_tensor* t  = make_input(timestep);
            auto runner_ctx = get_context();
            auto out = te.forward(&runner_ctx, t);
            ggml_build_forward_expand(gf, out);
            return gf;
        };
        return take_or_empty(GGMLRunner::compute<float>(get_g, n_threads, true));
    }
};

// Runs JUST the ada-values reshape+slice on a pre-computed time_embed. Returns one of
// the 4 slices (chosen by `which` in 0..3 → shift1, scale1, shift2, scale2). This
// isolates the PyTorch `timestep.reshape(B, 4, -1, 1, 1, 1)` → unbind(dim=1) path
// in pure GGML ops to verify memory-order correctness.
struct ShiftProbeRunner : public GGMLRunner {
    int in_channels;
    int which;
    ShiftProbeRunner(ggml_backend_t backend, bool offload, int in_ch, int which)
        : GGMLRunner(backend, offload), in_channels(in_ch), which(which) {}
    std::string get_desc() override { return "ltx2_vae_shift_probe"; }
    sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& time_embed) {
        auto get_g = [&]() -> ggml_cgraph* {
            ggml_cgraph* gf = ggml_new_graph(compute_ctx);
            ggml_tensor* te = make_input(time_embed);  // ne=[4*C, 1]
            auto re  = ggml_reshape_2d(compute_ctx, te, in_channels, 4);  // [C, 4]
            auto out = ggml_ext_slice(compute_ctx, re, 1, which, which + 1);  // [C, 1]
            out = ggml_cont(compute_ctx, out);
            ggml_build_forward_expand(gf, out);
            return gf;
        };
        return take_or_empty(GGMLRunner::compute<float>(get_g, n_threads, true));
    }
};
#include "model.h"
#include "tensor.hpp"

namespace {

sd::Tensor<float> load_raw_bin(const std::string& path, const std::vector<int64_t>& shape) {
    sd::Tensor<float> t(shape);
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::fprintf(stderr, "fatal: cannot open %s\n", path.c_str());
        std::exit(2);
    }
    f.read(reinterpret_cast<char*>(t.data()),
           static_cast<std::streamsize>(t.numel() * sizeof(float)));
    if (!f.good()) {
        std::fprintf(stderr, "fatal: short read on %s (expected %ld floats)\n",
                     path.c_str(), t.numel());
        std::exit(2);
    }
    return t;
}

struct DiffStats {
    float max_abs = 0.f, mean_abs = 0.f, max_rel = 0.f;
    int64_t max_abs_idx = -1;
};

DiffStats diff_fp32(const float* a, const float* b, int64_t n) {
    DiffStats s;
    double sum_abs = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        float abs_err = std::fabs(a[i] - b[i]);
        float rel_err = abs_err / (std::fabs(b[i]) + 1e-8f);
        if (abs_err > s.max_abs) { s.max_abs = abs_err; s.max_abs_idx = i; }
        s.max_rel = std::max(s.max_rel, rel_err);
        sum_abs += abs_err;
    }
    s.mean_abs = static_cast<float>(sum_abs / (n > 0 ? n : 1));
    return s;
}

bool compare(const std::string& tag, const sd::Tensor<float>& got,
             const std::string& ref_path, const std::vector<int64_t>& ref_shape,
             float tol_max_abs, float tol_mean_abs) {
    auto ref = load_raw_bin(ref_path, ref_shape);
    if (got.numel() != ref.numel()) {
        std::printf("  %-20s SHAPE_FAIL got_numel=%ld ref_numel=%ld\n",
                    tag.c_str(), got.numel(), ref.numel());
        return false;
    }
    auto s = diff_fp32(got.data(), ref.data(), got.numel());
    bool pass = s.max_abs < tol_max_abs && s.mean_abs < tol_mean_abs;
    std::printf("  %-20s %s  max_abs=%.3e mean_abs=%.3e  n=%ld\n",
                tag.c_str(), pass ? "PASS" : "FAIL", s.max_abs, s.mean_abs, got.numel());
    return pass;
}

}  // namespace

int main() {
    // Enable library logging so load_tensors shape mismatches surface on stderr.
    sd_set_log_callback(
        [](enum sd_log_level_t /*level*/, const char* text, void* /*data*/) {
            std::fputs(text, stderr);
        },
        nullptr);

    const std::string ref_dir    = "/tmp/vae_ref";
    const std::string state_path = ref_dir + "/state_dict.safetensors";

    ModelLoader loader;
    if (!loader.init_from_file(state_path)) {
        std::fprintf(stderr, "fatal: init_from_file failed for %s\n", state_path.c_str());
        return 1;
    }
    const auto& tsm = loader.get_tensor_storage_map();
    std::printf("[state_dict] loaded %zu tensors from %s\n", tsm.size(), state_path.c_str());

    // Tiny config from dump_vae.py: in=3, latent=8, patch=2, base_ch=8.
    // Encoder: compress_space_res(×2), compress_time_res(×2), res_x(1 layer).
    // Decoder: compress_space(m=1), compress_time(m=1), res_x(1 layer, timestep_cond=True).
    const int in_ch      = 3;
    const int latent_ch  = 8;
    const int base_ch    = 8;
    const int patch      = 2;
    const int B = 1, F = 9, H = 16, W_ = 16;

    std::vector<LTXVAE::EncoderBlockSpec> enc_specs = {
        {LTXVAE::EncoderBlockKind::COMPRESS_SPACE_RES, 1, 2},
        {LTXVAE::EncoderBlockKind::COMPRESS_TIME_RES,  1, 2},
        {LTXVAE::EncoderBlockKind::RES_X,              1, 1},
    };
    std::vector<LTXVAE::DecoderBlockSpec> dec_specs = {
        {LTXVAE::DecoderBlockKind::COMPRESS_SPACE, 1, 1},
        {LTXVAE::DecoderBlockKind::COMPRESS_TIME,  1, 1},
        {LTXVAE::DecoderBlockKind::RES_X,          1, 1},
    };

    ggml_backend_t backend = ggml_backend_cpu_init();

    // --- Encoder ---
    LTXVAE::VAEEncoderRunner enc_runner(backend, /*offload=*/false, tsm,
                                        /*prefix=*/"vae.encoder",
                                        in_ch, latent_ch, patch, enc_specs);
    enc_runner.alloc_params_buffer();
    std::map<std::string, ggml_tensor*> enc_params;
    enc_runner.get_param_tensors(enc_params, "vae.encoder");
    std::printf("[enc] requesting %zu param tensors\n", enc_params.size());
    if (!loader.load_tensors(enc_params)) {
        std::fprintf(stderr, "fatal: encoder load_tensors failed\n");
        return 1;
    }

    // Load video input. Python shape (1, 3, 9, 16, 16) → GGML ne=[W=16, H=16, T=9, C=3].
    auto video_in = load_raw_bin(ref_dir + "/tensors/video_in.bin", {W_, H, F, in_ch});
    std::printf("[enc] running encoder (traced)\n");

    bool pass = true;
    struct Stage { int idx; const char* name; std::vector<int64_t> shape; float abs_tol, mean_tol; };
    // Dump order & shapes (PyTorch-majored):
    //   0 post_patchify      (1,12,9,8,8)   → ne=[8,8,9,12]
    //   1 post_conv_in       (1,8,9,8,8)    → ne=[8,8,9,8]
    //   2 down_block[0] (cs) (1,16,9,4,4)   → ne=[4,4,9,16]
    //   3 down_block[1] (ct) (1,32,5,4,4)   → ne=[4,4,5,32]
    //   4 down_block[2] (res)(1,32,5,4,4)   → ne=[4,4,5,32]
    //   5 post_norm          (1,32,5,4,4)   → ne=[4,4,5,32]
    //   6 post_conv_out      (1,9,5,4,4)    → ne=[4,4,5,9]
    //   7 means_preNorm      (1,8,5,4,4)    → ne=[4,4,5,8]
    //   8 latent             (1,8,5,4,4)    → ne=[4,4,5,8]
    // Conv3d weights are stored f16 in the block — each conv boundary introduces a
    // fp16-quantization step (~1e-3 abs per layer). Tolerances are set accordingly.
    std::vector<Stage> stages = {
        {0, "enc_post_patchify",  {8, 8, F, 12},        1e-6f, 1e-7f},   // pure rearrange
        {1, "enc_post_conv_in",   {8, 8, F, 8},         2e-3f, 3e-4f},
        {2, "enc_block_0",        {4, 4, F, 16},        3e-3f, 5e-4f},
        {3, "enc_block_1",        {4, 4, 5, 32},        5e-3f, 8e-4f},
        {4, "enc_block_2",        {4, 4, 5, 32},        5e-3f, 1e-3f},
        {5, "enc_post_norm",      {4, 4, 5, 32},        5e-3f, 1e-3f},
        {6, "enc_post_conv_out",  {4, 4, 5, 9},         5e-3f, 1e-3f},
        {7, "enc_means_preNorm",  {4, 4, 5, 8},         5e-3f, 1e-3f},
        {8, "latent",             {4, 4, 5, 8},         5e-3f, 1e-3f},
    };
    for (const auto& s : stages) {
        auto got = enc_runner.compute(1, video_in, s.idx);
        pass &= compare(s.name, got, ref_dir + "/tensors/" + s.name + ".bin", s.shape,
                        s.abs_tol, s.mean_tol);
    }

    // --- Decoder ---
    LTXVAE::VAEDecoderRunner dec_runner(backend, /*offload=*/false, tsm,
                                        /*prefix=*/"vae.decoder",
                                        latent_ch, in_ch, patch, base_ch,
                                        /*timestep_cond=*/true, dec_specs);
    dec_runner.alloc_params_buffer();
    std::map<std::string, ggml_tensor*> dec_params;
    dec_runner.get_param_tensors(dec_params, "vae.decoder");
    std::printf("[dec] requesting %zu param tensors\n", dec_params.size());

    // Diagnose any name/shape mismatches.
    std::set<std::string> file_keys;
    for (const auto& kv : tsm) file_keys.insert(kv.first);
    int missing = 0;
    for (const auto& pt : dec_params) {
        auto it = file_keys.find(pt.first);
        if (it == file_keys.end()) {
            if (missing < 10) std::printf("[dec] missing: %s\n", pt.first.c_str());
            missing++;
        }
    }
    std::printf("[dec] %d / %zu tensors missing from file\n", missing, dec_params.size());

    if (!loader.load_tensors(dec_params)) {
        std::fprintf(stderr, "fatal: decoder load_tensors failed\n");
        return 1;
    }

    // Feed the Python reference latent to the decoder so its diffs are independent of
    // encoder errors. Once encoder parity is green we can chain them.
    auto latent_ref = load_raw_bin(ref_dir + "/tensors/latent.bin", {4, 4, 5, latent_ch});
    sd::Tensor<float> timestep_t({1});
    timestep_t.data()[0] = 0.05f;
    // TimestepEmbedder micro-probe: bypass the full decoder and run just the
    // up_blocks[0].time_embedder on timestep=0.05 to verify the sinusoidal + linear path.
    {
        TERunner te_runner(backend, false, tsm, "vae.decoder.up_blocks.0.time_embedder", 256);
        te_runner.alloc_params_buffer();
        std::map<std::string, ggml_tensor*> te_params;
        te_runner.get_param_tensors(te_params, "vae.decoder.up_blocks.0.time_embedder");
        if (!loader.load_tensors(te_params)) {
            std::fprintf(stderr, "fatal: TE load failed\n");
            return 1;
        }
        auto te_out = te_runner.compute(1, timestep_t);
        // Python dumps shape [B=1, 256] → innermost 256. sd::Tensor stores innermost-first,
        // so shape is {256, 1}. Same numel.
        pass &= compare("TimestepEmbedder", te_out, ref_dir + "/tensors/te_probe_up0.bin",
                        {256, 1}, 1e-4f, 1e-5f);

        // Verify the ada-values reshape+slice: Python does `te.reshape(B,4,-1,1,1,1)` →
        // unbind(dim=1). The four unbound slices should be te[0:64], te[64:128], te[128:192],
        // te[192:256]. Run each slice through the C++ reshape+slice path and byte-compare.
        auto te_ref = load_raw_bin(ref_dir + "/tensors/te_probe_up0.bin", {256, 1});
        const char* which_names[] = {"shift1", "scale1", "shift2", "scale2"};
        for (int w = 0; w < 4; w++) {
            ShiftProbeRunner sp(backend, false, /*in_ch=*/64, w);
            auto slice = sp.compute(1, te_ref);
            float maxd = 0.f;
            for (int i = 0; i < 64; i++) {
                float d = std::fabs(slice.data()[i] - te_ref.data()[w * 64 + i]);
                if (d > maxd) maxd = d;
            }
            std::printf("  shift-probe %-7s max_abs vs te[%d:%d]=%.3e\n",
                        which_names[w], w * 64, (w + 1) * 64, maxd);
            pass &= (maxd < 1e-6f);
        }
    }

    // Per-stage trace now includes intermediates pushed INSIDE the first res_x block:
    //   0 post_unnorm, 1 post_conv_in, 2 time_embed, 3 post_norm1, 4 shift1, 5 scale1,
    //   6 post_adaln1, 7 post_conv1, 8 post_norm2, 9 up_block[0] out, ...
    auto got_norm1 = dec_runner.compute(1, latent_ref, timestep_t, 3);
    pass &= compare("resblock0 post_norm1", got_norm1,
                    ref_dir + "/tensors/dec_resblock0_post_norm1.bin",
                    {4, 4, 5, 64}, 2e-3f, 5e-4f);
    auto got_adaln1 = dec_runner.compute(1, latent_ref, timestep_t, 6);
    pass &= compare("resblock0 post_adaln1", got_adaln1,
                    ref_dir + "/tensors/dec_resblock0_post_adaln1.bin",
                    {4, 4, 5, 64}, 5e-3f, 1e-3f);
    auto got_conv1 = dec_runner.compute(1, latent_ref, timestep_t, 7);
    pass &= compare("resblock0 post_conv1", got_conv1,
                    ref_dir + "/tensors/dec_resblock0_post_conv1.bin",
                    {4, 4, 5, 64}, 5e-3f, 1e-3f);
    auto got_norm2 = dec_runner.compute(1, latent_ref, timestep_t, 8);
    pass &= compare("resblock0 post_norm2", got_norm2,
                    ref_dir + "/tensors/dec_resblock0_post_norm2.bin",
                    {4, 4, 5, 64}, 1e-2f, 2e-3f);

    // After causal=false + reflect-padding fixes, trace indices in the decoder have shifted.
    // New layout:
    //   0 post_unnorm   1 post_conv_in   2 time_embed   3 post_norm1  4 shift1
    //   5 scale1        6 post_adaln1    7 post_conv1   8 post_norm2  9 up_block[0] out
    //   10 up_block[1]  11 up_block[2]   12 post_pixel_norm  13 post_ada
    //   14 post_conv_out 15 video_out
    struct Stage2 { int idx; const char* name; std::vector<int64_t> shape; float atol, mtol; };
    std::vector<Stage2> stages2 = {
        // Shapes in ne-order (W, H, T, C) after each decoder block.
        // Compress_time expands T 5→9; compress_space expands spatial 4→8 (patch=2 still
        // to apply at the very end via unpatchify).
        { 9, "dec_block_0",       {4, 4, 5, 64},   1e-2f, 2e-3f},
        {10, "dec_block_1",       {4, 4, F, 64},   1e-2f, 2e-3f},
        {11, "dec_block_2",       {8, 8, F, 64},   2e-2f, 4e-3f},
        {12, "dec_post_pixel_norm", {8, 8, F, 64}, 2e-2f, 4e-3f},
        {13, "dec_post_ada",      {8, 8, F, 64},   2e-2f, 4e-3f},
        {14, "dec_post_conv_out", {8, 8, F, 12},   2e-2f, 4e-3f},
    };
    for (const auto& s : stages2) {
        auto got = dec_runner.compute(1, latent_ref, timestep_t, s.idx);
        pass &= compare(s.name, got,
                        ref_dir + "/tensors/" + std::string(s.name) + ".bin",
                        s.shape, s.atol, s.mtol);
    }

    auto decoded = dec_runner.compute(1, latent_ref, timestep_t);
    pass &= compare("dec video", decoded, ref_dir + "/tensors/video_out.bin", {W_, H, F, in_ch}, 1e-2f, 2e-3f);

    // Per-stage probe on the same runner (since GGMLRunner can be reused across
    // multiple computes, as the encoder path does 9 times without issue).
    if (!pass) {
        std::printf("\n[dec] per-stage probe:\n");
        const char* stage_names[] = {
            "dec_post_unnorm", "dec_post_conv_in", "dec_block_0", "dec_block_1", "dec_block_2",
            "dec_post_pixel_norm", "dec_post_ada", "dec_post_conv_out", "video_out"
        };
        for (int idx = 0; idx < 9; idx++) {
            std::printf("  [%d] stage=%s computing...\n", idx, stage_names[idx]); std::fflush(stdout);
            auto out = dec_runner.compute(1, latent_ref, timestep_t, idx);
            std::string tag = stage_names[idx];
            std::string ref_path = ref_dir + "/tensors/" + tag + ".bin";
            std::ifstream check(ref_path);
            if (check.good()) {
                check.close();
                std::vector<int64_t> shape = {out.shape()[0], out.shape()[1], out.shape()[2], out.shape()[3]};
                auto ref = load_raw_bin(ref_path, shape);
                if (ref.numel() != out.numel()) {
                    std::printf("  [%d] %-20s SHAPE_MISMATCH got=%ld ref=%ld (shape=%ld,%ld,%ld,%ld)\n",
                                idx, tag.c_str(), out.numel(), ref.numel(),
                                shape[0], shape[1], shape[2], shape[3]);
                    continue;
                }
                auto s = diff_fp32(out.data(), ref.data(), out.numel());
                std::printf("  [%d] %-20s max_abs=%.3e mean_abs=%.3e\n",
                            idx, tag.c_str(), s.max_abs, s.mean_abs);
            } else {
                float m0 = 0.f, m1 = 0.f;
                for (int64_t i = 0; i < out.numel(); i++) { float a = std::fabs(out.data()[i]); m0 = std::max(m0, a); m1 += a; }
                m1 /= out.numel() > 0 ? out.numel() : 1;
                std::printf("  [%d] %-20s (no ref) shape=[%ld,%ld,%ld,%ld] max_abs=%.3f mean_abs=%.3f\n",
                            idx, tag.c_str(),
                            out.shape()[0], out.shape()[1], out.shape()[2], out.shape()[3], m0, m1);
            }
        }
    }

    std::printf("\n%s\n", pass ? "ALL VAE PARITY: PASS" : "ALL VAE PARITY: FAIL");
    (void)B;
    return pass ? 0 : 3;
}
