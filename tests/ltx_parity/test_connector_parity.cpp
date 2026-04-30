// LTX-2 text connector parity test (V1 / 19B).
//
// Loads /tmp/connector_ref/{state_dict.safetensors, tensors/*.bin} produced by
// dump_connector.py, runs:
//   1. CPU feature_extractor_normalize on the stacked input
//   2. LTX2ConnectorRunner::compute through each probe stage
// and diffs against the Python reference.

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
#include "ltx_connector.hpp"
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
    const std::string ref_dir    = "/tmp/connector_ref";
    const std::string state_path = ref_dir + "/state_dict.safetensors";

    // Tiny config (must match dump_connector.py).
    const int64_t B = 1;
    const int64_t T = 8;
    const int NUM_HEADS = 2;
    const int HEAD_DIM  = 32;
    const int64_t D = NUM_HEADS * HEAD_DIM;   // connector inner_dim = 64
    const int64_t L = 5;                      // stacked layers
    const int64_t FLAT_DIM = D * L;           // 320
    const int NUM_LAYERS = 2;
    const int NUM_REGISTERS = 4;
    const int64_t CAPTION_CHANNELS = D;       // 64
    const int64_t CAPTION_HIDDEN   = 128;
    const int64_t CAPTION_OUT      = 128;
    const float THETA = 10000.0f;
    const std::vector<int> MAX_POS = {1};

    // --- 1. Load state dict.
    ModelLoader loader;
    if (!loader.init_from_file(state_path)) {
        std::fprintf(stderr, "fatal: init_from_file failed for %s\n", state_path.c_str());
        return 1;
    }
    const auto& tsm = loader.get_tensor_storage_map();
    std::printf("[state_dict] loaded %zu tensors from %s\n", tsm.size(), state_path.c_str());

    // --- 2. Construct runner.
    ggml_backend_t backend = ggml_backend_cpu_init();
    LTXConnector::LTX2ConnectorRunner runner(
        backend, /*offload_params_to_cpu=*/false,
        FLAT_DIM, NUM_HEADS, HEAD_DIM, NUM_LAYERS, NUM_REGISTERS,
        CAPTION_CHANNELS, CAPTION_HIDDEN, CAPTION_OUT,
        THETA, MAX_POS, tsm, /*prefix=*/"");

    runner.alloc_params_buffer();
    std::map<std::string, ggml_tensor*> param_tensors;
    runner.get_param_tensors(param_tensors, "");
    std::printf("[load] %zu param tensors…\n", param_tensors.size());

    // Diagnose any missing tensors.
    int missing_shown = 0;
    std::set<std::string> tsm_keys;
    for (const auto& kv : tsm) tsm_keys.insert(kv.first);
    for (const auto& pt : param_tensors) {
        if (tsm_keys.find(pt.first) == tsm_keys.end()) {
            if (missing_shown < 5) {
                std::printf("[load] missing in file: %s\n", pt.first.c_str());
                missing_shown++;
            }
        }
    }

    if (!loader.load_tensors(param_tensors)) {
        std::fprintf(stderr, "fatal: load_tensors failed\n");
        return 1;
    }

    // --- 3. Load stacked input (ref layout: [B, T, D, L]).
    auto stacked_in = load_raw_bin(ref_dir + "/tensors/stacked_in.bin", {L, D, T, B});

    // --- 4. CPU-side feature extractor normalization.
    std::vector<int> seq_lens(B, static_cast<int>(T));  // all-ones mask
    sd::Tensor<float> normed({FLAT_DIM, T, B});
    LTXConnector::feature_extractor_normalize(
        stacked_in.data(), seq_lens.data(), normed.data(),
        static_cast<int>(B), static_cast<int>(T), static_cast<int>(D), static_cast<int>(L),
        "left", 1e-6f);

    // --- 5. Run each probe stage and diff.
    struct Probe {
        int stage;
        const char* name;
        std::vector<int64_t> shape;  // ne order (innermost first)
        float tol_max_abs;
        float tol_mean_abs;
    };
// Tolerances reflect: (1) fp16 K/V cast in ggml_ext_attention_ext (~1e-3 per
    // attention layer), (2) residual fp32 cos/sin divergence between torch and
    // libm at the tail of the freq grid (~6e-3 PE max diff → ~1e-3 per q/k
    // rotation). Two attention layers → ~2-3e-3 max_abs cap end-to-end.
    const Probe probes[] = {
        {0, "feat_ext_out",     {D, T, B},           1e-4f, 5e-5f},
        {1, "conn_block_0_out", {D, T, B},           3e-3f, 5e-4f},
        {2, "conn_block_1_out", {D, T, B},           4e-3f, 1e-3f},
        {3, "conn_final_out",   {D, T, B},           3e-3f, 5e-4f},
        {4, "caption_proj_out", {CAPTION_OUT, T, B}, 3e-3f, 1e-3f},
    };

    bool all_pass = true;
    std::printf("\n=== LTX-2 Connector parity ===\n");
    std::printf("%-20s %11s %11s %11s  %s\n", "tag", "max_abs", "mean_abs", "max_rel", "result");

    for (const auto& p : probes) {
        auto out = runner.compute(/*n_threads=*/1, normed, p.stage);
        auto ref = load_raw_bin(ref_dir + "/tensors/" + p.name + ".bin", p.shape);
        if (out.numel() != ref.numel()) {
            std::fprintf(stderr, "[%s] size mismatch got=%ld want=%ld\n",
                         p.name, out.numel(), ref.numel());
            return 1;
        }
        auto s    = diff_fp32(out.data(), ref.data(), out.numel());
        bool pass = s.max_abs < p.tol_max_abs && s.mean_abs < p.tol_mean_abs;
        std::printf("  %-18s %.3e %.3e %.3e  %s\n",
                    p.name, s.max_abs, s.mean_abs, s.max_rel, pass ? "PASS" : "FAIL");
        if (!pass && s.max_abs_idx >= 0) {
            int64_t i = s.max_abs_idx;
            std::printf("    max-diff @ idx=%ld: got=%+.6f want=%+.6f diff=%+.6f\n",
                        i, out.data()[i], ref.data()[i], out.data()[i] - ref.data()[i]);
        }
        all_pass &= pass;
    }

    std::printf("\n%s\n", all_pass ? "Connector parity: PASS" : "Connector parity: FAIL");

    // ---------- Padded variant: T_REAL < NUM_REGISTERS ----------
    // This section exercises the learnable-register concat path in
    // LTX2ConnectorRunner::build_graph that the primary run above skips (there
    // T=8 > NUM_REGISTERS=4).  The reference is dumped by
    // `CONNECTOR_VARIANT=padded dump_connector.py` with NUM_REGISTERS=8,
    // SEQ_LEN=8 and a left-padded attention_mask making only the last 3 tokens
    // real.  Python runs the full pipeline (feature_extractor → replace_padded
    // → connector); C++ feeds only the 3 real tokens (slide-to-front done in
    // the conditioner on the production path) and the runner's concat-with-
    // registers path must reconstruct the same 8-token sequence internally.
    const std::string padded_dir = "/tmp/connector_ref_padded";
    std::ifstream padded_check(padded_dir + "/state_dict.safetensors");
    if (!padded_check.is_open()) {
        std::printf("\n[padded] %s not found — skip.  Run "
                    "`CONNECTOR_VARIANT=padded dump_connector.py` to enable.\n",
                    padded_dir.c_str());
        return all_pass ? 0 : 3;
    }
    padded_check.close();

    std::printf("\n=== LTX-2 Connector parity (padded: T_real=3 < num_reg=8) ===\n");

    const int64_t PAD_T_REAL    = 3;
    const int64_t PAD_T_FULL    = 8;
    const int NUM_REGISTERS_PAD = 8;

    ModelLoader pad_loader;
    if (!pad_loader.init_from_file(padded_dir + "/state_dict.safetensors")) {
        std::fprintf(stderr, "fatal: padded init_from_file failed\n");
        return 1;
    }
    const auto& pad_tsm = pad_loader.get_tensor_storage_map();
    std::printf("[padded state_dict] loaded %zu tensors\n", pad_tsm.size());

    LTXConnector::LTX2ConnectorRunner pad_runner(
        backend, /*offload_params_to_cpu=*/false,
        FLAT_DIM, NUM_HEADS, HEAD_DIM, NUM_LAYERS, NUM_REGISTERS_PAD,
        CAPTION_CHANNELS, CAPTION_HIDDEN, CAPTION_OUT,
        THETA, MAX_POS, pad_tsm, /*prefix=*/"");
    pad_runner.alloc_params_buffer();

    std::map<std::string, ggml_tensor*> pad_params;
    pad_runner.get_param_tensors(pad_params, "");
    if (!pad_loader.load_tensors(pad_params)) {
        std::fprintf(stderr, "fatal: padded load_tensors failed\n");
        return 1;
    }

    // Load the full padded stacked input (padded positions at the START), then
    // slice to only the T_REAL real tokens at the tail — this is what the
    // production conditioner passes to the connector runner after sliding the
    // real rows to the front.
    auto pad_stacked_full = load_raw_bin(padded_dir + "/tensors/stacked_in.bin",
                                         {L, D, PAD_T_FULL, B});
    // Ref layout [B, T, D, L] → ggml ne [L, D, T, B].  Real tokens occupy
    // indices [PAD_T_FULL - PAD_T_REAL .. PAD_T_FULL) along axis T (ne[2]).
    sd::Tensor<float> pad_stacked_real({L, D, PAD_T_REAL, B});
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < PAD_T_REAL; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                for (int64_t l = 0; l < L; ++l) {
                    int64_t src = ((b * PAD_T_FULL + (PAD_T_FULL - PAD_T_REAL + t)) * D + d) * L + l;
                    int64_t dst = ((b * PAD_T_REAL + t) * D + d) * L + l;
                    pad_stacked_real.data()[dst] = pad_stacked_full.data()[src];
                }
            }
        }
    }

    // CPU normalize the real-only stacked input (no padding).
    std::vector<int> pad_seq_lens(B, static_cast<int>(PAD_T_REAL));
    sd::Tensor<float> pad_normed({FLAT_DIM, PAD_T_REAL, B});
    LTXConnector::feature_extractor_normalize(
        pad_stacked_real.data(), pad_seq_lens.data(), pad_normed.data(),
        static_cast<int>(B), static_cast<int>(PAD_T_REAL), static_cast<int>(D), static_cast<int>(L),
        "left", 1e-6f);

    // Connector should internally concat learnable_registers[T_real:num_reg]
    // → output shape at the final stage is [D, num_reg, B].
    bool pad_pass = true;
    const Probe pad_probes[] = {
        // Feature-extractor output is just the T_REAL real tokens (shape
        // [D, T_REAL, B]); Python's feat_ext_out covers T_FULL padded and we
        // only check the real-token tail.
        {3, "conn_final_out",   {D, PAD_T_FULL, B},           6e-3f, 2e-3f},
        {4, "caption_proj_out", {CAPTION_OUT, PAD_T_FULL, B}, 6e-3f, 2e-3f},
    };

    for (const auto& p : pad_probes) {
        auto out = pad_runner.compute(/*n_threads=*/1, pad_normed, p.stage);
        auto ref = load_raw_bin(padded_dir + "/tensors/" + p.name + ".bin", p.shape);
        if (out.numel() != ref.numel()) {
            std::fprintf(stderr, "[padded %s] size mismatch got=%ld want=%ld\n",
                         p.name, out.numel(), ref.numel());
            return 1;
        }
        auto s    = diff_fp32(out.data(), ref.data(), out.numel());
        bool pass = s.max_abs < p.tol_max_abs && s.mean_abs < p.tol_mean_abs;
        std::printf("  %-18s %.3e %.3e %.3e  %s\n",
                    p.name, s.max_abs, s.mean_abs, s.max_rel, pass ? "PASS" : "FAIL");
        if (!pass && s.max_abs_idx >= 0) {
            int64_t i = s.max_abs_idx;
            std::printf("    max-diff @ idx=%ld: got=%+.6f want=%+.6f diff=%+.6f\n",
                        i, out.data()[i], ref.data()[i], out.data()[i] - ref.data()[i]);
        }
        pad_pass &= pass;
    }

    std::printf("\n%s\n", pad_pass ? "Connector padded parity: PASS" : "Connector padded parity: FAIL");
    return (all_pass && pad_pass) ? 0 : 3;
}
