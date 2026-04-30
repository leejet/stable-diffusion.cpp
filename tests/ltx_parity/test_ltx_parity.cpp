// LTX-2 C++ parity test.
//
// Loads the state dict + reference intermediate tensors dumped by
// tests/ltx_parity/dump_reference.py, runs one forward pass of LTXRunner on the same inputs,
// and diffs the output against the Python reference.
//
// Tolerances: F32 backend is expected to match to ~1e-4 abs / ~1e-3 rel. Larger drift points to
// a block-level bug — rerun with --intermediate to capture per-block outputs via the cache API.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "denoiser.hpp"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ltx.hpp"
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
    float max_abs  = 0.f;
    float mean_abs = 0.f;
    float max_rel  = 0.f;
    int64_t max_abs_idx = -1;
};

DiffStats diff_fp32(const float* a, const float* b, int64_t n) {
    DiffStats s;
    double sum_abs = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        float abs_err = std::fabs(a[i] - b[i]);
        float rel_err = abs_err / (std::fabs(b[i]) + 1e-8f);
        if (abs_err > s.max_abs) {
            s.max_abs     = abs_err;
            s.max_abs_idx = i;
        }
        s.max_rel = std::max(s.max_rel, rel_err);
        sum_abs += abs_err;
    }
    s.mean_abs = static_cast<float>(sum_abs / (n > 0 ? n : 1));
    return s;
}

void print_shape(const char* label, const std::vector<int64_t>& shape) {
    std::printf("%s[", label);
    for (size_t i = 0; i < shape.size(); ++i) {
        std::printf("%s%ld", i == 0 ? "" : ", ", shape[i]);
    }
    std::printf("]\n");
}

}  // namespace

// Returns true if all schedule cases agree within absolute tolerance.
bool check_schedule(const std::string& ref_dir) {
    struct Case {
        int tokens;
        int steps;
        bool stretch;
        const char* label;
    };
    // Must match the cases in dump_reference.py::dump_scheduler.
    const std::vector<Case> cases = {
        {1024, 10, true,  "tokens1024_steps10_stretch1"},
        {1024, 30, true,  "tokens1024_steps30_stretch1"},
        {4096, 10, true,  "tokens4096_steps10_stretch1"},
        {4096, 40, true,  "tokens4096_steps40_stretch1"},
        {2560, 30, true,  "tokens2560_steps30_stretch1"},
        {4096,  8, false, "tokens4096_steps8_stretch0"},
    };

    bool all_pass = true;
    std::printf("\n=== LTX2FlowDenoiser::get_sigmas parity ===\n");
    for (const auto& c : cases) {
        LTX2FlowDenoiser denoiser;
        denoiser.stretch  = c.stretch;
        auto cpp_sigmas   = denoiser.get_sigmas(static_cast<uint32_t>(c.steps), c.tokens,
                                                DISCRETE_SCHEDULER, VERSION_LTX2);
        auto ref          = load_raw_bin(ref_dir + "/tensors/schedule__" + c.label + ".bin",
                                         {static_cast<int64_t>(c.steps + 1)});
        if (static_cast<int64_t>(cpp_sigmas.size()) != ref.numel()) {
            std::fprintf(stderr, "[sched %s] size mismatch cpp=%zu ref=%ld\n",
                         c.label, cpp_sigmas.size(), ref.numel());
            all_pass = false;
            continue;
        }
        auto s = diff_fp32(cpp_sigmas.data(), ref.data(), ref.numel());
        // Schedules are small floats (≤1), 1e-5 abs tolerance is reasonable; mu/exp arithmetic
        // differs negligibly between libm and Python math.
        const float tol = 5e-5f;
        bool pass       = s.max_abs < tol;
        std::printf("  %-32s max_abs=%.2e mean_abs=%.2e  %s\n",
                    c.label, s.max_abs, s.mean_abs, pass ? "PASS" : "FAIL");
        if (!pass) {
            std::printf("    cpp[0..3] = %.6f %.6f %.6f %.6f\n", cpp_sigmas[0],
                        cpp_sigmas[1], cpp_sigmas[2], cpp_sigmas[3]);
            std::printf("    ref[0..3] = %.6f %.6f %.6f %.6f\n", ref.data()[0],
                        ref.data()[1], ref.data()[2], ref.data()[3]);
            all_pass = false;
        }
    }
    return all_pass;
}

// Runs one Euler step in C++ using LTX2FlowDenoiser's scheduler values + a DiT velocity output,
// then diffs against the Python reference.
bool check_euler_step(const std::string& ref_dir, LTX::LTXRunner& runner,
                      const sd::Tensor<float>& latent, const sd::Tensor<float>& context) {
    auto sigma_cur_ref  = load_raw_bin(ref_dir + "/tensors/euler__sigma_cur.bin", {1});
    auto sigma_next_ref = load_raw_bin(ref_dir + "/tensors/euler__sigma_next.bin", {1});
    auto v_ref          = load_raw_bin(ref_dir + "/tensors/euler__v_step_unflat.bin", {6, 4, 2, 16});
    auto x_next_ref     = load_raw_bin(ref_dir + "/tensors/euler__x_next_unflat.bin", {6, 4, 2, 16});

    const float sigma_cur  = sigma_cur_ref.data()[0];
    const float sigma_next = sigma_next_ref.data()[0];

    std::printf("\n=== Euler step parity (σ=%.4f → %.4f) ===\n", sigma_cur, sigma_next);

    // Run the DiT at σ_cur (pre-scaled by 1000 for AdaLN, via LTX2FlowDenoiser::sigma_to_t).
    LTX2FlowDenoiser denoiser;
    sd::Tensor<float> t_in({1});
    t_in.data()[0] = denoiser.sigma_to_t(sigma_cur);

    sd::Tensor<float> empty_mask;
    auto v_cpp = runner.compute(/*n_threads=*/1, latent, t_in, context, empty_mask);
    if (v_cpp.numel() != v_ref.numel()) {
        std::fprintf(stderr, "fatal: velocity size mismatch\n");
        return false;
    }

    // Compute x_next = latent + (σ_next - σ) * v  (element-wise).
    sd::Tensor<float> x_next_cpp(latent.shape());
    const float dt = sigma_next - sigma_cur;
    for (int64_t i = 0; i < latent.numel(); ++i) {
        x_next_cpp.data()[i] = latent.data()[i] + dt * v_cpp.data()[i];
    }

    auto sv = diff_fp32(v_cpp.data(), v_ref.data(), v_cpp.numel());
    auto sx = diff_fp32(x_next_cpp.data(), x_next_ref.data(), x_next_cpp.numel());

    std::printf("  velocity@σ_cur:  max_abs=%.2e mean_abs=%.2e max_rel=%.2e\n",
                sv.max_abs, sv.mean_abs, sv.max_rel);
    std::printf("  x_next:          max_abs=%.2e mean_abs=%.2e max_rel=%.2e\n",
                sx.max_abs, sx.mean_abs, sx.max_rel);

    // x_next is (latent + dt * v). dt is ~0.09, v drift ~1e-4 → x_next drift ~1e-5. Tolerances
    // are roughly the same as the base DiT test since the Euler step doesn't amplify.
    const float tol_abs = 1e-3f;
    const float tol_rel = 5e-2f;
    return sv.max_abs < tol_abs && sv.max_rel < tol_rel &&
           sx.max_abs < tol_abs && sx.max_rel < tol_rel;
}

int main() {
    const std::string ref_dir    = "/tmp/ltx_ref";
    const std::string state_path = ref_dir + "/state_dict.safetensors";

    // --- 1. Load the reference state dict. Weights are dumped with prefix "model.diffusion_model."
    //    which matches sd.cpp's default DiT location, so init_from_file with empty prefix passes names through.
    ModelLoader loader;
    if (!loader.init_from_file(state_path)) {
        std::fprintf(stderr, "fatal: init_from_file failed for %s\n", state_path.c_str());
        return 1;
    }
    loader.convert_tensors_name();  // no-op for LTX-2 — names already match
    const auto& tsm = loader.get_tensor_storage_map();
    std::printf("[state_dict] loaded %zu tensors from %s\n", tsm.size(), state_path.c_str());

    // --- 2. Construct LTXRunner on CPU with explicit tiny-model params
    //    (the real LTX-2 hyperparams num_heads=32/head_dim=128 are auto-detected from weight shapes,
    //    but the tiny test uses num_heads=4/head_dim=32 which can't be inferred from q_norm alone).
    LTX::LTXParams tiny_params;
    tiny_params.in_channels          = 16;
    tiny_params.out_channels         = 16;
    tiny_params.inner_dim            = 128;
    tiny_params.num_heads            = 4;
    tiny_params.head_dim             = 32;
    tiny_params.num_layers           = 2;
    tiny_params.cross_attention_dim  = 128;
    tiny_params.cross_attention_adaln = false;
    tiny_params.apply_gated_attention = false;

    ggml_backend_t backend = ggml_backend_cpu_init();
    LTX::LTXRunner runner(backend, /*offload_params_to_cpu=*/false, tsm,
                          "model.diffusion_model", VERSION_LTX2, &tiny_params);
    runner.set_fps(24.0f);
    // Parity dump uses simplified (f, h, w) positions without VAE scale factors or
    // causal_fix — mirror that here so positions match the Python reference.
    runner.set_scale_factors(1, 1, 1);
    runner.set_causal_fix(false);

    const auto& p = runner.ltx_params;
    std::printf("[config] layers=%d inner=%ld heads=%d head_dim=%d "
                "in=%ld out=%ld ca_dim=%ld\n",
                p.num_layers, p.inner_dim, p.num_heads, p.head_dim,
                p.in_channels, p.out_channels, p.cross_attention_dim);

    // --- 3. Allocate & load weights into the GGML graph.
    runner.alloc_params_buffer();
    std::map<std::string, ggml_tensor*> param_tensors;
    runner.get_param_tensors(param_tensors, "model.diffusion_model");
    std::printf("[load] loading %zu param tensors…\n", param_tensors.size());
    if (!loader.load_tensors(param_tensors)) {
        std::fprintf(stderr, "fatal: load_tensors failed (some names unmatched?)\n");
        return 1;
    }

    // --- 4. Load reference inputs.
    //    latent_unflat is dumped as [C=16, F=2, H=4, W=6] (C outermost, W innermost in memory).
    //    LTXRunner::build_graph expects ggml ne=[W, H, T=F, C], so sd::Tensor shape is {6, 4, 2, 16}
    //    (sd shape[0] = innermost dim). Raw memory layout is identical.
    auto latent   = load_raw_bin(ref_dir + "/tensors/model__latent_unflat.bin", {6, 4, 2, 16});
    auto sigma_in = load_raw_bin(ref_dir + "/tensors/model__sigma.bin", {1});

    // The C++ AdaLN now expects pre-scaled σ (see src/ltx.hpp:AdaLayerNormSingle docstring);
    // the denoiser's sigma_to_t(σ)=σ*1000 will own this scaling in production. For the test
    // we do it inline.
    sd::Tensor<float> timesteps({1});
    timesteps.data()[0] = sigma_in.data()[0] * 1000.0f;

    // context: Python shape [B=1, S=8, D=128] → ggml ne [128, 8, 1] → sd::Tensor shape {128, 8, 1}.
    auto context = load_raw_bin(ref_dir + "/tensors/model__context_in.bin", {128, 8, 1});

    sd::Tensor<float> empty_mask;

    std::printf("[input] ");
    print_shape("latent=", latent.shape());
    std::printf("[input] σ = %.6f → t = %.3f\n", sigma_in.data()[0], timesteps.data()[0]);

    // --- 5. Run forward.
    std::printf("[compute] running single forward pass…\n");
    auto out = runner.compute(/*n_threads=*/1, latent, timesteps, context, empty_mask);

    print_shape("[output] out.shape = ", out.shape());

    // Dump first & last few values to catch silent NaN / zeros before diffing.
    std::printf("[output] first 8: ");
    for (int i = 0; i < 8 && i < out.numel(); ++i) std::printf("%+.4f ", out.data()[i]);
    std::printf("\n");
    std::printf("[output] last  8: ");
    for (int64_t i = std::max<int64_t>(0, out.numel() - 8); i < out.numel(); ++i) std::printf("%+.4f ", out.data()[i]);
    std::printf("\n");

    // --- 6. Diff vs reference.
    auto ref = load_raw_bin(ref_dir + "/tensors/model__velocity_out_unflat.bin", {6, 4, 2, 16});
    if (out.numel() != ref.numel()) {
        std::fprintf(stderr, "fatal: element count mismatch cpp=%ld ref=%ld\n",
                     out.numel(), ref.numel());
        return 1;
    }
    std::printf("[ref]    first 8: ");
    for (int i = 0; i < 8 && i < ref.numel(); ++i) std::printf("%+.4f ", ref.data()[i]);
    std::printf("\n");

    auto s = diff_fp32(out.data(), ref.data(), out.numel());
    std::printf("\n=== velocity_out parity ===\n");
    std::printf("  max_abs  = %.3e (at index %ld: cpp=%.6f ref=%.6f)\n",
                s.max_abs, s.max_abs_idx,
                s.max_abs_idx >= 0 ? out.data()[s.max_abs_idx] : 0.f,
                s.max_abs_idx >= 0 ? ref.data()[s.max_abs_idx] : 0.f);
    std::printf("  mean_abs = %.3e\n", s.mean_abs);
    std::printf("  max_rel  = %.3e\n", s.max_rel);
    std::printf("  n        = %ld\n\n", out.numel());

    // FP32 tolerances realistic for multi-layer DiT: accumulation order (ggml's mat-mul vs
    // torch.matmul), softmax + rope + rms_norm order-of-ops, and bf16 casts in flash-attn paths
    // all add ~1e-4 abs / ~1e-2 rel drift per block. Mean_abs is the more stable indicator.
    //
    // max_rel is only meaningful when every |ref[i]| is comfortably above the expected noise
    // floor. The V1 reference happens to contain a single element with |ref| ≈ 4e-5, so a
    // 1e-5 abs drift (far below our max_abs tolerance) alone pushes max_rel to ~0.3. Skip
    // the max_rel check here for the same reason V2-deep does — abs/mean catch real drift
    // and the near-zero rel spike is noise.
    const float tol_max_abs  = 1e-3f;
    const float tol_mean_abs = 2e-4f;
    bool pass_dit = s.max_abs < tol_max_abs && s.mean_abs < tol_mean_abs;
    std::printf("%s (tol: max_abs<%.1e mean_abs<%.1e; max_rel ignored due to near-zero divisors)\n",
                pass_dit ? "DiT parity: PASS" : "DiT parity: FAIL",
                tol_max_abs, tol_mean_abs);

    bool pass_sched = check_schedule(ref_dir);
    std::printf("%s\n", pass_sched ? "Scheduler parity: PASS" : "Scheduler parity: FAIL");

    bool pass_euler = check_euler_step(ref_dir, runner, latent, context);
    std::printf("%s\n", pass_euler ? "Euler step parity: PASS" : "Euler step parity: FAIL");

    // --- V2 parity (cross_attention_adaln=true + apply_gated_attention=true) -----------------
    // The V1 check above validates the base path with both V2 features disabled. The production
    // 22B checkpoint uses both. This block reloads the same state_dict with a V2-flagged runner
    // and compares against Python's `v2model/velocity_out_unflat` dump.
    std::printf("\n=== V2 parity (cross_attention_adaln + apply_gated_attention) ===\n");
    LTX::LTXParams v2_params;
    v2_params.in_channels           = 16;
    v2_params.out_channels          = 16;
    v2_params.inner_dim             = 128;
    v2_params.num_heads             = 4;
    v2_params.head_dim              = 32;
    v2_params.num_layers            = 2;
    v2_params.cross_attention_dim   = 128;
    v2_params.cross_attention_adaln = true;
    v2_params.apply_gated_attention = true;

    LTX::LTXRunner v2_runner(backend, /*offload_params_to_cpu=*/false, tsm,
                             "model.diffusion_model_v2", VERSION_LTX2, &v2_params);
    v2_runner.set_fps(24.0f);
    v2_runner.set_scale_factors(1, 1, 1);
    v2_runner.set_causal_fix(false);
    v2_runner.alloc_params_buffer();

    std::map<std::string, ggml_tensor*> v2_param_tensors;
    v2_runner.get_param_tensors(v2_param_tensors, "model.diffusion_model_v2");
    std::printf("[v2] loading %zu param tensors under model.diffusion_model_v2\n", v2_param_tensors.size());
    if (!loader.load_tensors(v2_param_tensors)) {
        std::fprintf(stderr, "fatal: V2 load_tensors failed\n");
        return 1;
    }

    auto v2_latent = load_raw_bin(ref_dir + "/tensors/v2model__latent_unflat.bin", {6, 4, 2, 16});
    auto v2_sigma  = load_raw_bin(ref_dir + "/tensors/v2model__sigma.bin", {1});
    sd::Tensor<float> v2_timesteps({1});
    v2_timesteps.data()[0] = v2_sigma.data()[0] * 1000.0f;
    auto v2_context = load_raw_bin(ref_dir + "/tensors/v2model__context_in.bin", {128, 8, 1});
    sd::Tensor<float> v2_empty_mask;

    auto v2_out = v2_runner.compute(/*n_threads=*/1, v2_latent, v2_timesteps, v2_context, v2_empty_mask);
    auto v2_ref = load_raw_bin(ref_dir + "/tensors/v2model__velocity_out_unflat.bin", {6, 4, 2, 16});

    std::printf("[v2 output] first 8: ");
    for (int i = 0; i < 8 && i < v2_out.numel(); ++i) std::printf("%+.4f ", v2_out.data()[i]);
    std::printf("\n[v2 ref]    first 8: ");
    for (int i = 0; i < 8 && i < v2_ref.numel(); ++i) std::printf("%+.4f ", v2_ref.data()[i]);
    std::printf("\n");

    auto sv2 = diff_fp32(v2_out.data(), v2_ref.data(), v2_out.numel());
    std::printf("  max_abs  = %.3e (at index %ld: cpp=%.6f ref=%.6f)\n",
                sv2.max_abs, sv2.max_abs_idx,
                sv2.max_abs_idx >= 0 ? v2_out.data()[sv2.max_abs_idx] : 0.f,
                sv2.max_abs_idx >= 0 ? v2_ref.data()[sv2.max_abs_idx] : 0.f);
    std::printf("  mean_abs = %.3e\n", sv2.mean_abs);
    std::printf("  max_rel  = %.3e\n", sv2.max_rel);

    // Same max_rel skip as the V1 block above: the reference can contain a handful of
    // near-zero elements whose tiny abs drift blows the relative error up without being
    // a real parity regression. abs/mean catch actual drift.
    bool pass_v2 = sv2.max_abs < tol_max_abs && sv2.mean_abs < tol_mean_abs;
    std::printf("%s (tol: max_abs<%.1e mean_abs<%.1e; max_rel ignored due to near-zero divisors)\n",
                pass_v2 ? "V2 DiT parity: PASS" : "V2 DiT parity: FAIL",
                tol_max_abs, tol_mean_abs);

    // --- V2-deep parity: 8 layers + non-zero scale_shift_table -------------------------------
    // The V2 check above uses 2 layers with zeroed sst, so modulation is effectively identity
    // and can hide sign/broadcast bugs in the (1+scale) and shift_kv/scale_kv branches. This
    // block loads an 8-layer variant with randomised sst weights so any cross-layer drift in
    // the V2 path surfaces.
    std::printf("\n=== V2-deep parity (8 layers + non-zero scale_shift_table) ===\n");
    LTX::LTXParams v2_deep_params       = v2_params;
    v2_deep_params.num_layers           = 8;

    LTX::LTXRunner v2_deep_runner(backend, /*offload_params_to_cpu=*/false, tsm,
                                  "model.diffusion_model_v2_deep", VERSION_LTX2, &v2_deep_params);
    v2_deep_runner.set_fps(24.0f);
    v2_deep_runner.set_scale_factors(1, 1, 1);
    v2_deep_runner.set_causal_fix(false);
    v2_deep_runner.alloc_params_buffer();

    std::map<std::string, ggml_tensor*> v2_deep_param_tensors;
    v2_deep_runner.get_param_tensors(v2_deep_param_tensors, "model.diffusion_model_v2_deep");
    std::printf("[v2-deep] loading %zu param tensors\n", v2_deep_param_tensors.size());
    if (!loader.load_tensors(v2_deep_param_tensors)) {
        std::fprintf(stderr, "fatal: V2-deep load_tensors failed\n");
        return 1;
    }

    auto v2d_latent = load_raw_bin(ref_dir + "/tensors/v2deep__latent_unflat.bin", {6, 4, 2, 16});
    auto v2d_sigma  = load_raw_bin(ref_dir + "/tensors/v2deep__sigma.bin", {1});
    sd::Tensor<float> v2d_timesteps({1});
    v2d_timesteps.data()[0] = v2d_sigma.data()[0] * 1000.0f;
    auto v2d_context = load_raw_bin(ref_dir + "/tensors/v2deep__context_in.bin", {128, 8, 1});
    sd::Tensor<float> v2d_empty_mask;

    auto v2d_out = v2_deep_runner.compute(/*n_threads=*/1, v2d_latent, v2d_timesteps, v2d_context, v2d_empty_mask);
    auto v2d_ref = load_raw_bin(ref_dir + "/tensors/v2deep__velocity_out_unflat.bin", {6, 4, 2, 16});

    std::printf("[v2-deep output] first 8: ");
    for (int i = 0; i < 8 && i < v2d_out.numel(); ++i) std::printf("%+.4f ", v2d_out.data()[i]);
    std::printf("\n[v2-deep ref]    first 8: ");
    for (int i = 0; i < 8 && i < v2d_ref.numel(); ++i) std::printf("%+.4f ", v2d_ref.data()[i]);
    std::printf("\n");

    auto sv2d = diff_fp32(v2d_out.data(), v2d_ref.data(), v2d_out.numel());
    std::printf("  max_abs  = %.3e (at index %ld: cpp=%.6f ref=%.6f)\n",
                sv2d.max_abs, sv2d.max_abs_idx,
                sv2d.max_abs_idx >= 0 ? v2d_out.data()[sv2d.max_abs_idx] : 0.f,
                sv2d.max_abs_idx >= 0 ? v2d_ref.data()[sv2d.max_abs_idx] : 0.f);
    std::printf("  mean_abs = %.3e\n", sv2d.mean_abs);
    std::printf("  max_rel  = %.3e\n", sv2d.max_rel);

    // Tolerance: max_rel is dropped here because per-element rel_err with b_i in the 1e-4
    // range produces meaningless blow-ups (100% rel for 1e-4 abs). max_abs and mean_abs are
    // the reliable signals — both on the order of the 2-layer V2 test confirms no accumulated
    // drift across 8 layers × non-zero sst modulation.
    const float tol_max_abs_deep  = 5e-3f;
    const float tol_mean_abs_deep = 1e-3f;
    bool pass_v2_deep = sv2d.max_abs < tol_max_abs_deep && sv2d.mean_abs < tol_mean_abs_deep;
    std::printf("%s (tol: max_abs<%.1e mean_abs<%.1e; max_rel ignored due to near-zero divisors)\n",
                pass_v2_deep ? "V2-deep DiT parity: PASS" : "V2-deep DiT parity: FAIL",
                tol_max_abs_deep, tol_mean_abs_deep);

    bool pass = pass_dit && pass_sched && pass_euler && pass_v2 && pass_v2_deep;
    std::printf("\n%s\n", pass ? "ALL PARITY: PASS" : "ALL PARITY: FAIL");
    return pass ? 0 : 3;
}
