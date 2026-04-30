// Layer-0 CPU vs CUDA parity for Gemma 3.
//
// Loads the user's real Gemma GGUF twice — once with a CPU backend, once with
// a CUDA backend — runs compute_all_hidden_states on the same tokens with
// g_layer0_taps set, and diffs each intermediate. This lets us pinpoint which
// Gemma layer-0 op first diverges between CPU and CUDA without pulling in the
// DiT (which would push a 32 GB system into swap/OOM).
//
// Usage:
//   sd-gemma-cpu-vs-cuda <gemma.gguf> [cuda_device]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef SD_USE_CUDA
#include "ggml-cuda.h"
#endif

#include "llm.hpp"
#include "model.h"
#include "tensor.hpp"

namespace {

struct DiffStats {
    float   max_abs  = 0.f;
    float   mean_abs = 0.f;
    int64_t argmax   = -1;
};

DiffStats diff_f32(const float* a, const float* b, int64_t n) {
    DiffStats s;
    double   sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > s.max_abs) {
            s.max_abs = d;
            s.argmax  = i;
        }
        sum += d;
    }
    s.mean_abs = static_cast<float>(sum / (n > 0 ? n : 1));
    return s;
}

std::vector<uint8_t> fetch(ggml_tensor* t) {
    size_t n = ggml_nbytes(t);
    std::vector<uint8_t> out(n);
    ggml_backend_tensor_get(t, out.data(), 0, n);
    return out;
}

struct TapDump {
    std::string          name;
    std::vector<int64_t> ne;
    ggml_type            type;
    std::vector<uint8_t> data;
};

std::vector<TapDump> run_and_dump(const std::string& model_path,
                                   ggml_backend_t     backend,
                                   const std::vector<int32_t>& tokens) {
    ModelLoader loader;
    if (!loader.init_from_file(model_path, "text_encoders.llm.")) {
        std::fprintf(stderr, "fatal: init_from_file failed: %s\n", model_path.c_str());
        std::exit(1);
    }
    loader.convert_tensors_name();

    // SD_GEMMA_FORCE_TYPE=f16|bf16|f32|q8_0|q4_0 — on-load retype so both
    // backends take the same matmul path (avoids iq4_xs×q8_K-vs-q8_1 drift).
    if (const char* t = std::getenv("SD_GEMMA_FORCE_TYPE")) {
        std::string s = t;
        ggml_type tgt = GGML_TYPE_F16;
        if      (s == "f32")  tgt = GGML_TYPE_F32;
        else if (s == "bf16") tgt = GGML_TYPE_BF16;
        else if (s == "q8_0" || s == "q8") tgt = GGML_TYPE_Q8_0;
        else if (s == "q4_0") tgt = GGML_TYPE_Q4_0;
        std::printf("[retype] forcing weights to %s\n", ggml_type_name(tgt));
        loader.set_wtype_override(tgt);
    }

    // Rename text_encoders.llm.* -> text_encoder.* (matches LTX-2 flow).
    auto& tsm = loader.get_tensor_storage_map();
    {
        const std::string from = "text_encoders.llm.";
        const std::string to   = "text_encoder.";
        String2TensorStorage out;
        for (auto& kv : tsm) {
            std::string k = kv.first;
            if (k.rfind(from, 0) == 0) {
                k             = to + k.substr(from.size());
                kv.second.name = k;
            }
            out[k] = std::move(kv.second);
        }
        tsm.swap(out);
    }
    // Gemma sandwich-norm renames (mirrored from stable-diffusion.cpp init).
    auto rename_suffix = [&](const std::string& old_suffix, const std::string& new_suffix) {
        String2TensorStorage out;
        for (auto& kv : tsm) {
            std::string k = kv.first;
            size_t      p = k.rfind(old_suffix);
            if (p != std::string::npos && p + old_suffix.size() == k.size() &&
                k.find("text_encoder.model.layers.") != std::string::npos) {
                k             = k.substr(0, p) + new_suffix;
                kv.second.name = k;
            }
            out[k] = std::move(kv.second);
        }
        tsm.swap(out);
    };
    rename_suffix(".post_attention_layernorm.weight", ".pre_feedforward_layernorm.weight");
    rename_suffix(".post_attention_norm.weight",      ".post_attention_layernorm.weight");
    rename_suffix(".post_ffw_norm.weight",            ".post_feedforward_layernorm.weight");

    LLM::LLMRunner runner(LLM::LLMArch::GEMMA3, backend, /*offload=*/false,
                          tsm, /*prefix=*/"text_encoder", /*enable_vision=*/false);

    runner.alloc_params_buffer();
    std::map<std::string, ggml_tensor*> param_tensors;
    runner.get_param_tensors(param_tensors, "text_encoder");
    if (!loader.load_tensors(param_tensors)) {
        std::fprintf(stderr, "fatal: load_tensors failed\n");
        std::exit(1);
    }

    // Dump token_embd weight rows to compare storage and sanity.
    auto it = param_tensors.find("text_encoder.model.embed_tokens.weight");
    if (it != param_tensors.end()) {
        ggml_tensor* w = it->second;
        std::printf("[weight] embed_tokens.weight: type=%s ne=[%ld,%ld,%ld,%ld] nbytes=%zu\n",
                    ggml_type_name(w->type), (long)w->ne[0], (long)w->ne[1], (long)w->ne[2], (long)w->ne[3], ggml_nbytes(w));
        if (w->type == GGML_TYPE_F32) {
            int64_t hidden = w->ne[0];
            for (int64_t row_idx : {0, 1, 2, 100, 106, 262207}) {
                std::vector<float> row(hidden);
                ggml_backend_tensor_get(w, row.data(), (size_t)row_idx * hidden * sizeof(float), hidden * sizeof(float));
                double sum_abs = 0;
                for (float v : row) sum_abs += std::fabs(v);
                std::printf("[weight] row %6ld first 4: %+.4e %+.4e %+.4e %+.4e  mean_abs=%.3e\n",
                            (long)row_idx, row[0], row[1], row[2], row[3], sum_abs / hidden);
            }
        }
    }

    const int64_t T = static_cast<int64_t>(tokens.size());
    sd::Tensor<int32_t> input_ids({T, 1});
    for (int64_t i = 0; i < T; ++i) input_ids.data()[i] = tokens[i];
    sd::Tensor<float> empty_mask;

    std::vector<ggml_tensor*> taps;
    ::g_layer0_taps      = &taps;
    ::g_attn_layer0_taps = &taps;  // share so attention internals also get captured
    ::g_attn_tap_count   = 0;       // reset per-graph budget
    fprintf(stderr, "[test] taps vec @ %p, g_attn_layer0_taps @ %p, set to %p\n",
            (void*)&taps, (void*)&::g_attn_layer0_taps, (void*)::g_attn_layer0_taps);
    auto stacked = runner.compute_all_hidden_states(/*n_threads=*/4, input_ids, empty_mask);
    fprintf(stderr, "[test] after compute, taps.size()=%zu\n", taps.size());

    // Collect tap dumps immediately while compute buffer is still alive.
    std::vector<TapDump> tap_dumps;
    for (auto* t : taps) {
        const char* nm = ggml_get_name(t);
        std::fprintf(stderr, "[tap-collect] tensor=%p name='%s' buffer=%p type=%s\n",
                     (void*)t, nm ? nm : "(null)", (void*)t->buffer, ggml_type_name(t->type));
        if (!nm || std::strncmp(nm, "DBG:", 4) != 0) continue;
        if (!t->buffer) {
            std::fprintf(stderr, "[tap] %s: no buffer (allocator aliased)\n", nm);
            continue;
        }
        TapDump td;
        td.name = nm + 4;
        td.type = t->type;
        for (int i = 0; i < 4; ++i) td.ne.push_back(t->ne[i]);
        td.data = fetch(t);
        tap_dumps.push_back(std::move(td));
    }
    ::g_layer0_taps      = nullptr;
    ::g_attn_layer0_taps = nullptr;

    // Slice each layer out of the stacked tensor. stacked layout (innermost
    // first): ne=[N+1, H, T, B]. For layer l: value at (b,t,h,l).
    const int64_t L       = runner.params.num_layers + 1;
    const int64_t H       = runner.params.hidden_size;
    const int64_t Tdim    = T;
    const int64_t B       = 1;
    const int64_t per_layer = H * Tdim * B;

    std::vector<TapDump> dumps;
    dumps.reserve(L);
    for (int64_t l = 0; l < L; ++l) {
        TapDump d;
        d.name = (l == 0) ? "stacked_L00" : ("stacked_L" + std::to_string(l));
        d.type = GGML_TYPE_F32;
        d.ne   = {H, Tdim, B, 1};
        d.data.resize(per_layer * sizeof(float));
        float* out = reinterpret_cast<float*>(d.data.data());
        const float* src = stacked.data();
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < Tdim; ++t) {
                for (int64_t h = 0; h < H; ++h) {
                    int64_t idx_stacked = ((b * Tdim + t) * H + h) * L + l;
                    out[(b * Tdim + t) * H + h] = src[idx_stacked];
                }
            }
        }
        dumps.push_back(std::move(d));
    }
    // Append tap dumps so the caller can diff per-op.
    for (auto& td : tap_dumps) dumps.push_back(std::move(td));
    return dumps;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <gemma.gguf> [cuda_device]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    int cuda_device              = argc >= 3 ? std::atoi(argv[2]) : 0;

    // Default short prompt; SD_GEMMA_TEST_LEN env var pads to a target length
    // (defaults to 128 to match LTX-2's runtime padding). This forces the
    // matmul dispatch into the MMQ path where batch > 8.
    std::vector<int32_t> tokens = {2, 106, 108, 1055, 674, 25148, 110, 107};
    {
        int target_len = 128;
        if (const char* e = std::getenv("SD_GEMMA_TEST_LEN")) target_len = std::atoi(e);
        if (target_len > (int) tokens.size()) {
            tokens.resize(target_len, /*pad token id=*/0);
        }
    }

    std::printf("[run] CPU forward...\n");
    std::fflush(stdout);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    auto cpu_dumps = run_and_dump(model_path, cpu_backend, tokens);
    ggml_backend_free(cpu_backend);
    std::printf("[run] CPU done, %zu taps\n", cpu_dumps.size());

#ifdef SD_USE_CUDA
    std::printf("[run] CUDA (device %d) forward...\n", cuda_device);
    std::fflush(stdout);
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(cuda_device);
    if (!cuda_backend) {
        std::fprintf(stderr, "fatal: CUDA backend init failed for device %d\n", cuda_device);
        return 1;
    }
    auto cuda_dumps = run_and_dump(model_path, cuda_backend, tokens);
    ggml_backend_free(cuda_backend);
    std::printf("[run] CUDA done, %zu taps\n", cuda_dumps.size());
#else
    std::fprintf(stderr, "fatal: built without SD_USE_CUDA\n");
    return 1;
#endif

    // Optional: dump specific CPU-side taps to disk for the standalone
    // mul_mat parity test. SD_GEMMA_DUMP_DIR=/tmp/gemma_taps writes
    // <name>.bin for each tap. We also write a small <name>.shape text file.
    if (const char* dir = std::getenv("SD_GEMMA_DUMP_DIR")) {
        for (const auto& d : cpu_dumps) {
            std::string fn = std::string(dir) + "/" + d.name + ".bin";
            FILE* fp = std::fopen(fn.c_str(), "wb");
            if (fp) {
                std::fwrite(d.data.data(), 1, d.data.size(), fp);
                std::fclose(fp);
            }
            std::string fn2 = std::string(dir) + "/" + d.name + ".shape";
            FILE* fp2 = std::fopen(fn2.c_str(), "w");
            if (fp2) {
                std::fprintf(fp2, "%ld %ld %ld %ld %s\n",
                             (long) d.ne[0], (long) d.ne[1], (long) d.ne[2], (long) d.ne[3],
                             ggml_type_name(d.type));
                std::fclose(fp2);
            }
        }
        std::printf("[dump] wrote %zu CPU taps to %s/\n", cpu_dumps.size(), dir);
    }

    // Diff by name.
    std::map<std::string, const TapDump*> cpu_idx;
    for (const auto& d : cpu_dumps) cpu_idx[d.name] = &d;

    std::printf("\n%-22s %-5s %12s %12s %12s %6s\n",
                "tap", "type", "max_abs", "mean_abs", "cpu_mean_mag", "shape");
    int fail_count = 0;
    for (const auto& c : cuda_dumps) {
        std::fprintf(stderr, "[diff] examining tap '%s' type=%s ne=[%ld,%ld,%ld,%ld]\n",
                c.name.c_str(), ggml_type_name(c.type),
                (long)c.ne[0], (long)c.ne[1], (long)c.ne[2], (long)c.ne[3]);
        auto it = cpu_idx.find(c.name);
        if (it == cpu_idx.end()) {
            std::printf("  %-20s [missing on CPU side]\n", c.name.c_str());
            continue;
        }
        const TapDump* p = it->second;
        if (p->type != c.type || p->ne != c.ne) {
            std::printf("  %-20s type/shape mismatch\n", c.name.c_str());
            continue;
        }
        if (c.type != GGML_TYPE_F32) {
            // Cast to F32 for diffing if needed. For simplicity we only handle F32 here.
            std::printf("  %-20s type=%s skipped\n", c.name.c_str(), ggml_type_name(c.type));
            continue;
        }
        int64_t n = int64_t(p->data.size() / sizeof(float));
        auto s = diff_f32(
            reinterpret_cast<const float*>(p->data.data()),
            reinterpret_cast<const float*>(c.data.data()),
            n);
        double cpu_mag = 0.0;
        const float* cp = reinterpret_cast<const float*>(p->data.data());
        for (int64_t i = 0; i < n; ++i) cpu_mag += std::fabs(cp[i]);
        cpu_mag /= (n > 0 ? n : 1);
        bool fail = (s.max_abs > 1e-3f * (float)cpu_mag + 1e-4f);
        std::printf("  %-20s %-5s %12.3e %12.3e %12.3e  [%ld,%ld,%ld,%ld] %s\n",
                    c.name.c_str(), ggml_type_name(c.type),
                    s.max_abs, s.mean_abs, (double)cpu_mag,
                    (long)c.ne[0], (long)c.ne[1], (long)c.ne[2], (long)c.ne[3],
                    fail ? "FAIL" : "ok");
        if (fail) {
            fail_count++;
            // First-fail detail dump: first 8 values from each side.
            if (fail_count == 1) {
                const float* cp = reinterpret_cast<const float*>(p->data.data());
                const float* cu = reinterpret_cast<const float*>(c.data.data());
                std::printf("    first 8 floats: CPU vs CUDA\n");
                for (int64_t i = 0; i < 8 && i < n; ++i) {
                    std::printf("      [%ld] %+.6e  vs  %+.6e  (diff %+.3e)\n",
                                (long)i, cp[i], cu[i], cu[i] - cp[i]);
                }
                std::printf("    argmax element: CPU=%+.6e CUDA=%+.6e  idx=%ld\n",
                            cp[s.argmax], cu[s.argmax], (long)s.argmax);
            }
        }
    }
    std::printf("\n%d taps diverged (max_abs > 1e-3 × mean(|cpu|) + 1e-4).\n", fail_count);
    return fail_count == 0 ? 0 : 3;
}
