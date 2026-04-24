// Gemma-3-12B numerical validation.
//
// Loads Gemma-3-12B from HF safetensors (or GGUF), tokenises a prompt,
// runs the text-only transformer forward on a CUDA backend, and prints
// per-layer hidden-state statistics so we can diff against a HuggingFace
// reference.
//
// Usage:
//   gemma3-test <gemma-dir-or-safetensors> <tokenizer.model> "<prompt>"
//
// The first argument can be either a single safetensors path or a
// directory containing shard files (model-00001-of-00005.safetensors,
// etc.) — we glob "*.safetensors" in that directory.

#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "../../src/gemma3.hpp"
#include "../../src/ggml_extend.hpp"
#include "../../src/model.h"
#include "../../src/tokenizers/gemma3_tokenizer.h"

static bool path_is_directory(const std::string& p) {
    struct stat st;
    if (stat(p.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

static std::vector<std::string> list_safetensors(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = opendir(dir.c_str());
    if (!d) return out;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        std::string name = e->d_name;
        if (name.size() > 12 && name.substr(name.size() - 12) == ".safetensors") {
            out.push_back(dir + "/" + name);
        }
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

static void log_stats(const char* label, const sd::Tensor<float>& t) {
    if (t.empty()) {
        std::fprintf(stderr, "[stats] %s: EMPTY\n", label);
        return;
    }
    int64_t n = t.numel();
    const float* d = t.data();
    double mn = 1e30, mx = -1e30, sum = 0, sq = 0;
    size_t nan = 0;
    for (int64_t i = 0; i < n; ++i) {
        double v = d[i];
        if (std::isnan(v)) { nan++; continue; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        sq  += v * v;
    }
    double mean = (n - nan) ? sum / (n - nan) : 0;
    double var  = (n - nan) ? (sq / (n - nan)) - mean * mean : 0;
    double stdv = var > 0 ? std::sqrt(var) : 0;
    std::string shape;
    for (size_t i = 0; i < t.shape().size(); ++i) {
        if (i) shape += "x";
        shape += std::to_string(t.shape()[i]);
    }
    std::fprintf(stderr,
                 "[stats] %-22s shape=[%s] min=%+.4f max=%+.4f mean=%+.4f std=%.4f nan=%zu\n",
                 label, shape.c_str(), mn, mx, mean, stdv, nan);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <gemma_dir_or_safetensors> <tokenizer.model> \"<prompt>\"\n",
                     argv[0]);
        return 1;
    }
    std::string model_path = argv[1];
    std::string tok_path   = argv[2];
    std::string prompt     = argv[3];

    // Tokenise.
    Gemma3Tokenizer tok;
    std::string terr;
    if (!tok.load_from_spm(tok_path, &terr)) {
        std::fprintf(stderr, "tokenizer load failed: %s\n", terr.c_str());
        return 2;
    }
    auto ids = tok.encode(prompt, /*add_bos=*/true, /*add_eos=*/false);
    std::fprintf(stderr, "[tok] encoded %zu tokens: [", ids.size());
    for (size_t i = 0; i < ids.size() && i < 32; ++i) {
        std::fprintf(stderr, "%s%d", i ? "," : "", ids[i]);
    }
    if (ids.size() > 32) std::fprintf(stderr, ",...");
    std::fprintf(stderr, "]\n");

    // Build sd::Tensor<int32_t> input_ids [L, N=1].
    sd::Tensor<int32_t> input_ids(std::vector<int64_t>{(int64_t)ids.size(), 1});
    std::memcpy(input_ids.data(), ids.data(), ids.size() * sizeof(int32_t));

    // Model loader: accept a single file or a directory of shards.
    ModelLoader loader;
    std::vector<std::string> files;
    if (path_is_directory(model_path)) {
        files = list_safetensors(model_path);
    } else {
        files.push_back(model_path);
    }
    if (files.empty()) {
        std::fprintf(stderr, "no safetensors files at %s\n", model_path.c_str());
        return 3;
    }
    for (const auto& f : files) {
        std::fprintf(stderr, "[load] %s\n", f.c_str());
        if (!loader.init_from_file(f, /*prefix=*/"language_model.")) {
            std::fprintf(stderr, "init_from_file failed: %s\n", f.c_str());
            return 4;
        }
    }

    // Backend.
#ifdef SD_USE_CUDA
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "CUDA init failed; falling back to CPU\n");
    }
#else
    ggml_backend_t backend = nullptr;
#endif
    if (!backend) backend = ggml_backend_cpu_init();
    std::fprintf(stderr, "[be] %s\n", ggml_backend_name(backend));

    // Build runner. The tensor map sees language_model.model.* keys from
    // HF — our Runner prefix is "model." so together they resolve to the
    // expected names (language_model.model.embed_tokens.weight etc).
    GEMMA3::Gemma3Runner runner(backend, /*offload=*/false,
                                loader.get_tensor_storage_map(),
                                /*prefix=*/"model");
    if (!runner.alloc_params_buffer()) {
        std::fprintf(stderr, "alloc_params_buffer failed\n");
        return 5;
    }

    std::map<std::string, ggml_tensor*> tensors;
    runner.get_param_tensors(tensors, /*prefix=*/"language_model.model");
    std::fprintf(stderr, "[load] mapping %zu tensors\n", tensors.size());
    if (!loader.load_tensors(tensors, /*ignore=*/{}, /*n_threads=*/4)) {
        std::fprintf(stderr, "load_tensors failed\n");
        return 6;
    }

    // Probe every 4th layer so we can diff against HF's hidden_states.
    for (int probe : {0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47, 48}) {
        auto h = runner.compute_layer_hidden(4, input_ids, probe);
        char label[64];
        std::snprintf(label, sizeof(label), "layer[%d]", probe);
        log_stats(label, h);
    }
    return 0;
}
