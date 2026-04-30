// Gemma 3 C++ parity test.
//
// Loads /tmp/gemma_ref/{state_dict.safetensors, tensors/*.bin} produced by
// tests/ltx_parity/dump_gemma.py, runs one LLMRunner forward pass on the same
// input_ids, and diffs each of the N+1 hidden states (embedding + per-layer +
// post-final-norm last) against the Python reference.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llm.hpp"
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

}  // namespace

int main() {
    const std::string ref_dir    = "/tmp/gemma_ref";
    const std::string state_path = ref_dir + "/state_dict.safetensors";

    // --- 1. Load state dict.
    ModelLoader loader;
    if (!loader.init_from_file(state_path)) {
        std::fprintf(stderr, "fatal: init_from_file failed for %s\n", state_path.c_str());
        return 1;
    }
    // Skip convert_tensors_name(): the "text_encoder" prefix is remapped to
    // "cond_stage_model.transformer." by the conversion table (see name_conversion.cpp:1112),
    // which would break our direct-load parity test. We match key names exactly.
    const auto& tsm = loader.get_tensor_storage_map();
    std::printf("[state_dict] loaded %zu tensors from %s\n", tsm.size(), state_path.c_str());

    // --- 2. Construct LLMRunner with GEMMA3 arch. Hyperparams auto-detect from tensor shapes.
    ggml_backend_t backend = ggml_backend_cpu_init();
    LLM::LLMRunner runner(LLM::LLMArch::GEMMA3, backend, /*offload_params_to_cpu=*/false,
                          tsm, /*prefix=*/"text_encoder", /*enable_vision=*/false);

    const auto& p = runner.params;
    std::printf("[config] layers=%ld hidden=%ld heads=%d kv_heads=%d head_dim=%d "
                "ff=%ld vocab=%ld sw=%d pattern=%d\n",
                p.num_layers, p.hidden_size, p.num_heads, p.num_kv_heads, p.head_dim,
                p.intermediate_size, p.vocab_size, p.sliding_window, p.sliding_window_pattern);

    // --- 3. Load params buffer and weights.
    runner.alloc_params_buffer();
    std::map<std::string, ggml_tensor*> param_tensors;
    runner.get_param_tensors(param_tensors, "text_encoder");
    std::printf("[load] loading %zu param tensors…\n", param_tensors.size());

    // Collect tsm keys for diffing.
    std::set<std::string> tsm_keys;
    for (const auto& kv : tsm) tsm_keys.insert(kv.first);

    // Dump any param_tensor keys not present in the file (for diagnosing name mismatches).
    int missing_shown = 0;
    for (const auto& pt : param_tensors) {
        if (tsm_keys.find(pt.first) == tsm_keys.end()) {
            if (missing_shown < 5) {
                std::printf("[load] missing in file: %s\n", pt.first.c_str());
                missing_shown++;
            }
        }
    }

    if (!loader.load_tensors(param_tensors)) {
        std::fprintf(stderr, "fatal: load_tensors failed (some names unmatched?)\n");
        return 1;
    }

    // --- 4. Load reference inputs. Dumper saved input_ids as f32 for simplicity.
    const int64_t B = 1, T = 8, H = p.hidden_size;
    auto input_ids_f32 = load_raw_bin(ref_dir + "/tensors/input_ids.bin", {T, B});
    sd::Tensor<int32_t> input_ids({T, B});
    for (int64_t i = 0; i < T; ++i) {
        input_ids.data()[i] = (int32_t)input_ids_f32.data()[i];
    }
    sd::Tensor<float> empty_mask;

    std::printf("[input] input_ids: ");
    for (int i = 0; i < T; i++) std::printf("%d ", input_ids.data()[i]);
    std::printf("\n");

    // Override the window size to match the tiny-config dump.
    runner.params.sliding_window = 4;
    // The tiny-config Python dump does NOT apply linear rope_scaling (that's only
    // wired for the "deep" variant which mirrors the real 12B).  Disable scaling
    // on the C++ side so we compare apples-to-apples.
    runner.params.rope_scaling_factor_global = 1.0f;

    // --- 5a. First test the basic forward path (returns just last_hidden_state after norm).
    std::printf("[compute] basic forward (last_hidden_state only)…\n");
    std::fflush(stdout);
    auto basic = runner.compute(/*n_threads=*/1, input_ids, empty_mask, {}, {});
    std::printf("[compute] basic forward done, numel=%ld first=%.4f\n", basic.numel(), basic.numel() > 0 ? basic.data()[0] : 0.f);
    std::fflush(stdout);

    // --- 5b. Compute all N+1 hidden states.
    std::printf("[compute] running forward pass with all-hidden-states path…\n");
    std::fflush(stdout);
    auto stacked = runner.compute_all_hidden_states(/*n_threads=*/1, input_ids, empty_mask);
    std::printf("[compute] done, stacked shape=[%zu dims] numel=%ld\n", stacked.shape().size(), stacked.numel());
    std::fflush(stdout);

    // stacked has shape (sd::Tensor layout = innermost-first): {N+1, H, T, B}.
    const int64_t N_plus_1 = p.num_layers + 1;
    if (stacked.numel() != N_plus_1 * H * T * B) {
        std::fprintf(stderr, "fatal: stacked numel mismatch got=%ld expected=%ld\n",
                     stacked.numel(), N_plus_1 * H * T * B);
        return 1;
    }
    std::printf("[output] stacked shape=[%ld,%ld,%ld,%ld] numel=%ld\n",
                N_plus_1, H, T, B, stacked.numel());

    // --- 6. Slice each layer out of the stacked tensor and diff.
    //     Memory layout: innermost=N+1, so all layers for one (h, t, b) are adjacent.
    //     For a given layer_idx l: layer_data[b][t][h] = stacked[((b*T + t)*H + h)*(N+1) + l].
    //     Ref layer is stored with innermost=H, shape [B, T, H]. So we reconstruct layer l by
    //     scattering.
    // Tolerances reflect the fp16 cast inside ggml_ext_attention_ext (K/V go through
    // GGML_TYPE_F16 before the softmax). Reference Python stays in fp32, so ~1e-3 abs
    // drift per attention layer is baked in. For 6 stacked layers we budget ~6× that.
    // max_rel is skipped — small reference values blow up relative error even when
    // absolute agreement is fine.
    const float tol_max_abs  = 1e-2f;
    const float tol_mean_abs = 2e-3f;

    bool all_pass = true;
    std::printf("\n=== Gemma hidden-state parity ===\n");
    std::printf("%-18s %11s %11s %11s\n", "tag", "max_abs", "mean_abs", "max_rel");

    std::vector<float> layer_buf(B * T * H);
    for (int l = 0; l < N_plus_1; l++) {
        // Gather layer l from the stacked tensor.
        const float* src = stacked.data();
        for (int64_t b = 0; b < B; b++) {
            for (int64_t t = 0; t < T; t++) {
                for (int64_t h = 0; h < H; h++) {
                    int64_t stacked_idx = ((b * T + t) * H + h) * N_plus_1 + l;
                    int64_t ref_idx     = (b * T + t) * H + h;
                    layer_buf[ref_idx]  = src[stacked_idx];
                }
            }
        }
        std::string tag = (l == 0) ? "hs_embed" : ("hs_" + std::string(l < 10 ? "0" : "") + std::to_string(l));
        auto ref        = load_raw_bin(ref_dir + "/tensors/" + tag + ".bin", {H, T, B});
        auto s          = diff_fp32(layer_buf.data(), ref.data(), (int64_t)layer_buf.size());
        bool pass       = s.max_abs < tol_max_abs && s.mean_abs < tol_mean_abs;
        std::printf("  %-16s %.3e %.3e %.3e  %s\n",
                    tag.c_str(), s.max_abs, s.mean_abs, s.max_rel, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }
    std::printf("\n%s (tol: max_abs<%.1e mean_abs<%.1e)\n",
                all_pass ? "Gemma parity: PASS" : "Gemma parity: FAIL",
                tol_max_abs, tol_mean_abs);

    // --- Deep variant parity: 24 layers × 512 hidden, seq=32 with 16-wide sliding ---
    // Mirrors the real Gemma 3 12B's sliding_window_pattern=6 (so ~every 6th layer does
    // global attention) at scaled-down dims.  Catches drift patterns that only appear
    // across many layers / real hidden-size, without requiring the full 12B download.
    bool deep_present = false;
    for (const auto& k : tsm_keys) {
        if (k.rfind("text_encoder_deep.", 0) == 0) {
            deep_present = true;
            break;
        }
    }
    if (!deep_present) {
        std::printf("\n[deep] no text_encoder_deep.* tensors found (run "
                    "`GEMMA_PARITY_VARIANT=deep dump_gemma.py` to enable); skipping\n");
        return all_pass ? 0 : 3;
    }

    std::printf("\n=== Gemma deep parity (24L × 512H, sliding=16, seq=32) ===\n");
    LLM::LLMRunner deep_runner(LLM::LLMArch::GEMMA3, backend, /*offload=*/false,
                               tsm, /*prefix=*/"text_encoder_deep", /*enable_vision=*/false);
    const auto& dp = deep_runner.params;
    std::printf("[deep config] layers=%ld hidden=%ld heads=%d kv_heads=%d head_dim=%d ff=%ld\n",
                dp.num_layers, dp.hidden_size, dp.num_heads, dp.num_kv_heads, dp.head_dim,
                dp.intermediate_size);

    deep_runner.alloc_params_buffer();
    std::map<std::string, ggml_tensor*> deep_params;
    deep_runner.get_param_tensors(deep_params, "text_encoder_deep");
    if (!loader.load_tensors(deep_params)) {
        std::fprintf(stderr, "fatal: deep load_tensors failed\n");
        return 1;
    }

    const int64_t Td = 32;
    const int64_t Hd = dp.hidden_size;
    auto deep_input_ids_f32 = load_raw_bin(ref_dir + "/tensors/deep_input_ids.bin", {Td, 1});
    sd::Tensor<int32_t> deep_input_ids({Td, 1});
    for (int64_t i = 0; i < Td; ++i) deep_input_ids.data()[i] = (int32_t)deep_input_ids_f32.data()[i];
    sd::Tensor<float> deep_empty_mask;

    // Override sliding window to match the deep variant's config (tiny: 4, deep: 16).
    deep_runner.params.sliding_window = 16;

    auto deep_stacked = deep_runner.compute_all_hidden_states(/*n_threads=*/1,
                                                              deep_input_ids,
                                                              deep_empty_mask);
    const int64_t deep_N_plus_1 = dp.num_layers + 1;
    GGML_ASSERT(deep_stacked.numel() == deep_N_plus_1 * Hd * Td * 1);

    std::printf("[deep output] stacked shape=[%ld,%ld,%ld,1] numel=%ld\n",
                deep_N_plus_1, Hd, Td, deep_stacked.numel());

    const float deep_tol_max_abs  = 5e-2f;  // 24 layers → ~4× baseline drift budget
    const float deep_tol_mean_abs = 1e-2f;
    bool deep_all_pass = true;
    std::printf("%-22s %11s %11s %11s\n", "tag", "max_abs", "mean_abs", "max_rel");

    std::vector<float> deep_layer_buf(Td * Hd);
    for (int l = 0; l < deep_N_plus_1; ++l) {
        const float* src = deep_stacked.data();
        for (int64_t t = 0; t < Td; ++t) {
            for (int64_t h = 0; h < Hd; ++h) {
                int64_t stacked_idx             = (t * Hd + h) * deep_N_plus_1 + l;
                deep_layer_buf[t * Hd + h]      = src[stacked_idx];
            }
        }
        std::string tag = (l == 0) ? "deep_hs_embed" : ("deep_hs_" + std::string(l < 10 ? "0" : "") + std::to_string(l));
        auto ref        = load_raw_bin(ref_dir + "/tensors/" + tag + ".bin", {Hd, Td, 1});
        auto s          = diff_fp32(deep_layer_buf.data(), ref.data(), (int64_t)deep_layer_buf.size());
        bool pass       = s.max_abs < deep_tol_max_abs && s.mean_abs < deep_tol_mean_abs;
        std::printf("  %-20s %.3e %.3e %.3e  %s\n",
                    tag.c_str(), s.max_abs, s.mean_abs, s.max_rel, pass ? "PASS" : "FAIL");
        deep_all_pass &= pass;
    }
    std::printf("\n%s (tol: max_abs<%.1e mean_abs<%.1e)\n",
                deep_all_pass ? "Gemma deep parity: PASS" : "Gemma deep parity: FAIL",
                deep_tol_max_abs, deep_tol_mean_abs);

    return (all_pass && deep_all_pass) ? 0 : 3;
}
