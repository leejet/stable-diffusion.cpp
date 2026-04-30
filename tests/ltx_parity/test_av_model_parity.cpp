// LTX-2 AV MODEL parity test (the full LTXModel, not just one block).
//
// Loads weights+inputs+outputs dumped by tests/ltx_parity/dump_av_model.py and
// runs LTXModel::forward_av on the same inputs, then diffs against the python
// reference. Exercises:
//   - audio_patchify_proj + audio_adaln_single + audio_prompt_adaln_single
//   - 4 cross-modal AdaLN modules (av_ca_*_adaln_single)
//   - num_layers transformer blocks via forward_av
//   - both output heads (video proj_out + audio_proj_out)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ltx.hpp"
#include "model.h"
#include "tensor.hpp"

namespace {

const std::string REF_DIR = "/tmp/ltx_av_model_ref";

sd::Tensor<float> load_raw(const std::string& path, const std::vector<int64_t>& ne) {
    sd::Tensor<float> t(ne);
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { std::fprintf(stderr, "fatal: cannot open %s\n", path.c_str()); std::exit(2); }
    f.read(reinterpret_cast<char*>(t.data()),
           static_cast<std::streamsize>(t.numel() * sizeof(float)));
    if (!f.good()) { std::fprintf(stderr, "fatal: short read on %s\n", path.c_str()); std::exit(2); }
    return t;
}

struct DiffStats { double max_abs=0, mean_abs=0, max_rel=0; int64_t n=0; };

DiffStats diff_f32(const float* a, const float* b, int64_t n) {
    DiffStats s; s.n = n; double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double abs_err = std::fabs(double(a[i]) - double(b[i]));
        double rel_err = abs_err / (std::fabs(double(b[i])) + 1e-8);
        if (abs_err > s.max_abs) s.max_abs = abs_err;
        if (rel_err > s.max_rel) s.max_rel = rel_err;
        sum += abs_err;
    }
    s.mean_abs = sum / std::max<int64_t>(1, n);
    return s;
}

sd::Tensor<float> build_pe(const std::string& cos_path, const std::string& sin_path,
                            int64_t inner_dim, int64_t T) {
    sd::Tensor<float> pe({inner_dim, T, 2});
    auto cos = load_raw(cos_path, {inner_dim, T, 1});
    auto sin = load_raw(sin_path, {inner_dim, T, 1});
    std::memcpy(pe.data(),               cos.data(), cos.numel() * sizeof(float));
    std::memcpy(pe.data() + cos.numel(), sin.data(), sin.numel() * sizeof(float));
    return pe;
}

struct Manifest {
    std::map<std::string, std::string> weights_path;
    std::map<std::string, std::string> inputs_path;
    std::map<std::string, std::string> outputs_path;
    std::map<std::string, std::string> config_str;
};

Manifest read_manifest(const std::string& dir) {
    std::string cmd = "/home/ilintar/venv/bin/python -c '"
        "import json,sys\n"
        "m=json.load(open(\"" + dir + "/manifest.json\"))\n"
        "for k,v in m[\"config\"].items(): print(\"C:\"+k+\":\"+str(v))\n"
        "for sec in [\"weights\",\"inputs\",\"outputs\"]:\n"
        "    for n,d in m[sec].items():\n"
        "        print(sec[0].upper()+\":\"+n+\":\"+d[\"path\"])\n"
        "' 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) { std::fprintf(stderr, "fatal: popen failed\n"); std::exit(2); }
    Manifest m;
    char line[4096];
    while (std::fgets(line, sizeof(line), p)) {
        std::string s(line);
        if (!s.empty() && s.back() == '\n') s.pop_back();
        if (s.size() < 3 || s[1] != ':') continue;
        char tag = s[0];
        std::string rest = s.substr(2);
        size_t c = rest.find(':');
        if (c == std::string::npos) continue;
        std::string name = rest.substr(0, c);
        std::string val  = rest.substr(c + 1);
        if (tag == 'C')      m.config_str[name] = val;
        else if (tag == 'W') m.weights_path[name]  = dir + "/" + val;
        else if (tag == 'I') m.inputs_path[name]   = dir + "/" + val;
        else if (tag == 'O') m.outputs_path[name]  = dir + "/" + val;
    }
    pclose(p);
    return m;
}

int parse_dict(const std::map<std::string, std::string>& cfg, const std::string& key, const char* sub) {
    auto it = cfg.find(key);
    if (it == cfg.end()) { std::fprintf(stderr, "fatal: %s\n", key.c_str()); std::exit(2); }
    std::string s = it->second;
    std::string needle = std::string("'") + sub + "': ";
    size_t p = s.find(needle);
    if (p == std::string::npos) { std::fprintf(stderr, "fatal: %s.%s\n", key.c_str(), sub); std::exit(2); }
    p += needle.size();
    size_t q = s.find_first_of(",}", p);
    return std::stoi(s.substr(p, q - p));
}
bool parse_bool(const std::map<std::string, std::string>& cfg, const std::string& k) {
    auto it = cfg.find(k);
    if (it == cfg.end()) return false;
    return it->second == "True" || it->second == "true" || it->second == "1";
}

struct AVModelParityRunner : public GGMLRunner {
    LTX::LTXModel model;

    sd::Tensor<float> v_latent, a_latent;
    sd::Tensor<float> v_context, a_context;
    sd::Tensor<float> v_t_self, a_t_self;
    sd::Tensor<float> v_t_prompt_self, a_t_prompt_self;
    sd::Tensor<float> v_t_cross_ss, a_t_cross_ss;
    sd::Tensor<float> v_t_cross_gate, a_t_cross_gate;
    sd::Tensor<float> v_pe, a_pe, v_cpe, a_cpe;

    sd::Tensor<float> result_v, result_a;

    AVModelParityRunner(ggml_backend_t backend, LTX::LTXParams params)
        : GGMLRunner(backend, /*offload_params_to_cpu=*/false), model(params) {
        model.init(params_ctx, /*tensor_storage_map=*/{}, /*prefix=*/"");
    }

    std::string get_desc() override { return "AVModelParityRunner"; }

    ggml_cgraph* build_graph() {
        auto gf = new_graph_custom(LTX::LTX_GRAPH_SIZE);
        ggml_tensor* g_v_latent = make_input(v_latent);
        ggml_tensor* g_a_latent = make_input(a_latent);
        ggml_tensor* g_v_ctx    = make_input(v_context);
        ggml_tensor* g_a_ctx    = make_input(a_context);
        ggml_tensor* g_v_t      = make_input(v_t_self);
        ggml_tensor* g_a_t      = make_input(a_t_self);
        ggml_tensor* g_v_t_p    = v_t_prompt_self.empty() ? nullptr : make_input(v_t_prompt_self);
        ggml_tensor* g_a_t_p    = a_t_prompt_self.empty() ? nullptr : make_input(a_t_prompt_self);
        ggml_tensor* g_v_t_xss  = make_input(v_t_cross_ss);
        ggml_tensor* g_a_t_xss  = make_input(a_t_cross_ss);
        ggml_tensor* g_v_t_xg   = make_input(v_t_cross_gate);
        ggml_tensor* g_a_t_xg   = make_input(a_t_cross_gate);
        ggml_tensor* g_v_pe     = make_input(v_pe);
        ggml_tensor* g_a_pe     = make_input(a_pe);
        ggml_tensor* g_v_cpe    = make_input(v_cpe);
        ggml_tensor* g_a_cpe    = make_input(a_cpe);

        auto rctx = get_context();
        auto outs = model.forward_av(&rctx,
            g_v_latent, g_a_latent,
            g_v_t, g_a_t,
            g_v_t_p, g_a_t_p,
            g_v_t_xss, g_a_t_xss,
            g_v_t_xg, g_a_t_xg,
            g_v_ctx, g_a_ctx,
            g_v_pe, g_a_pe,
            g_v_cpe, g_a_cpe,
            nullptr, nullptr);
        ggml_set_name(outs.first,  "av_model_video_out");
        ggml_set_name(outs.second, "av_model_audio_out");
        ggml_build_forward_expand(gf, outs.first);
        ggml_build_forward_expand(gf, outs.second);
        cache("av_model_video_out", outs.first);
        cache("av_model_audio_out", outs.second);
        return gf;
    }

    bool compute_and_capture(int n_threads,
                             int64_t v_t, int64_t v_out_dim,
                             int64_t a_t, int64_t a_out_dim) {
        auto gg = [this]() { return build_graph(); };
        compute<float>(gg, n_threads, /*free_compute_buffer_immediately=*/false,
                       /*no_return=*/true);
        ggml_tensor* tv = get_cache_tensor_by_name("av_model_video_out");
        ggml_tensor* ta = get_cache_tensor_by_name("av_model_audio_out");
        if (!tv || !ta) { std::fprintf(stderr, "fatal: missing output tensors\n"); return false; }
        result_v = sd::Tensor<float>({v_out_dim, v_t, 1});
        result_a = sd::Tensor<float>({a_out_dim, a_t, 1});
        ggml_backend_tensor_get(tv, result_v.data(), 0, ggml_nbytes(tv));
        ggml_backend_tensor_get(ta, result_a.data(), 0, ggml_nbytes(ta));
        return true;
    }

    bool load_weight_into(const std::string& name, const std::string& path) {
        std::map<std::string, ggml_tensor*> all;
        model.get_param_tensors(all, /*prefix=*/"");
        auto it = all.find(name);
        if (it == all.end()) return false;
        ggml_tensor* t = it->second;
        std::vector<float> buf(ggml_nelements(t));
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        f.read(reinterpret_cast<char*>(buf.data()),
               static_cast<std::streamsize>(buf.size() * sizeof(float)));
        ggml_backend_tensor_set(t, buf.data(), 0, ggml_nbytes(t));
        return true;
    }
};

}  // namespace


int main() {
    Manifest m = read_manifest(REF_DIR);
    if (m.config_str.empty()) {
        std::fprintf(stderr, "fatal: empty manifest. Run dump_av_model.py first.\n");
        return 2;
    }

    LTX::LTXParams params;
    params.in_channels       = parse_dict(m.config_str, "video", "in_channels");
    params.out_channels      = parse_dict(m.config_str, "video", "out_channels");
    params.inner_dim         = parse_dict(m.config_str, "video", "dim");
    params.num_heads         = parse_dict(m.config_str, "video", "heads");
    params.head_dim          = parse_dict(m.config_str, "video", "d_head");
    params.cross_attention_dim = parse_dict(m.config_str, "video", "ctx_dim");
    params.num_layers        = std::stoi(m.config_str.at("num_layers"));
    params.cross_attention_adaln  = parse_bool(m.config_str, "cross_attention_adaln");
    params.apply_gated_attention  = parse_bool(m.config_str, "apply_gated_attention");
    params.norm_eps               = std::stof(m.config_str.at("norm_eps"));
    {
        const std::string& rs = m.config_str.at("rope_type");
        params.rope_type = (rs.find("split") != std::string::npos) ? LTX::RopeType::SPLIT
                                                                    : LTX::RopeType::INTERLEAVED;
    }

    params.has_audio_video           = true;
    params.audio_in_channels         = parse_dict(m.config_str, "audio", "in_channels");
    params.audio_out_channels        = parse_dict(m.config_str, "audio", "out_channels");
    params.audio_inner_dim           = parse_dict(m.config_str, "audio", "dim");
    params.audio_num_heads           = parse_dict(m.config_str, "audio", "heads");
    params.audio_head_dim            = parse_dict(m.config_str, "audio", "d_head");
    params.audio_cross_attention_dim = parse_dict(m.config_str, "audio", "ctx_dim");

    const int T_VIDEO = parse_dict(m.config_str, "video", "T");
    const int T_AUDIO = parse_dict(m.config_str, "audio", "T");
    const int S_VIDEO = parse_dict(m.config_str, "video", "S");
    const int S_AUDIO = parse_dict(m.config_str, "audio", "S");

    std::printf("=== LTX-2 AV MODEL parity ===\n");
    std::printf("video: dim=%lld heads=%d d_head=%d in=%lld out=%lld T=%d S=%d\n",
                (long long)params.inner_dim, params.num_heads, params.head_dim,
                (long long)params.in_channels, (long long)params.out_channels,
                T_VIDEO, S_VIDEO);
    std::printf("audio: dim=%lld heads=%d d_head=%d in=%lld out=%lld T=%d S=%d\n",
                (long long)params.audio_inner_dim, params.audio_num_heads, params.audio_head_dim,
                (long long)params.audio_in_channels, (long long)params.audio_out_channels,
                T_AUDIO, S_AUDIO);
    std::printf("flags: ca_adaln=%d gated=%d num_layers=%d\n",
                params.cross_attention_adaln, params.apply_gated_attention, params.num_layers);

    auto backend = ggml_backend_cpu_init();
    AVModelParityRunner runner(backend, params);
    runner.alloc_params_buffer();

    int loaded = 0, skipped = 0;
    for (const auto& kv : m.weights_path) {
        if (!runner.load_weight_into(kv.first, kv.second)) {
            std::fprintf(stderr, "WARN: no slot for weight '%s' — skipping\n", kv.first.c_str());
            skipped++;
        } else loaded++;
    }
    std::printf("loaded %d weights (skipped %d)\n", loaded, skipped);
    if (skipped > 0) {
        std::fprintf(stderr, "FAIL: some weights unmapped — implementation gap\n");
        ggml_backend_free(backend);
        return 1;
    }

    auto path = [&](const char* k) { return m.inputs_path.at(k); };
    runner.v_latent       = load_raw(path("video__latent"),  {params.in_channels,        T_VIDEO, 1});
    runner.a_latent       = load_raw(path("audio__latent"),  {params.audio_in_channels,  T_AUDIO, 1});
    runner.v_context      = load_raw(path("video__context"), {params.cross_attention_dim, S_VIDEO, 1});
    runner.a_context      = load_raw(path("audio__context"), {params.audio_cross_attention_dim, S_AUDIO, 1});
    runner.v_t_self       = load_raw(path("video__t_self"),       {1});
    runner.a_t_self       = load_raw(path("audio__t_self"),       {1});
    if (params.cross_attention_adaln) {
        runner.v_t_prompt_self = load_raw(path("video__t_prompt_self"), {1});
        runner.a_t_prompt_self = load_raw(path("audio__t_prompt_self"), {1});
    }
    runner.v_t_cross_ss   = load_raw(path("video__t_cross_ss"),   {1});
    runner.a_t_cross_ss   = load_raw(path("audio__t_cross_ss"),   {1});
    runner.v_t_cross_gate = load_raw(path("video__t_cross_gate"), {1});
    runner.a_t_cross_gate = load_raw(path("audio__t_cross_gate"), {1});

    runner.v_pe  = build_pe(path("video__pe_cos"), path("video__pe_sin"), params.inner_dim,        T_VIDEO);
    runner.a_pe  = build_pe(path("audio__pe_cos"), path("audio__pe_sin"), params.audio_inner_dim,  T_AUDIO);
    runner.v_cpe = build_pe(path("video__cross_pe_cos"), path("video__cross_pe_sin"),
                             params.audio_inner_dim, T_VIDEO);
    runner.a_cpe = build_pe(path("audio__cross_pe_cos"), path("audio__cross_pe_sin"),
                             params.audio_inner_dim, T_AUDIO);

    if (!runner.compute_and_capture(/*n_threads=*/1,
                                    T_VIDEO, params.out_channels,
                                    T_AUDIO, params.audio_out_channels)) {
        ggml_backend_free(backend);
        return 1;
    }

    auto v_ref = load_raw(m.outputs_path.at("video__x_out"), {params.out_channels,       T_VIDEO, 1});
    auto a_ref = load_raw(m.outputs_path.at("audio__x_out"), {params.audio_out_channels, T_AUDIO, 1});

    auto vs  = diff_f32(runner.result_v.data(), v_ref.data(), runner.result_v.numel());
    auto as_ = diff_f32(runner.result_a.data(), a_ref.data(), runner.result_a.numel());

    const double tol_abs = 1e-4;  // 2-block model accumulates more drift than 1-block
    bool pass = (vs.max_abs < tol_abs) && (as_.max_abs < tol_abs);

    std::printf("\nvideo  out: max_abs=%.3e  mean_abs=%.3e  max_rel=%.3e  (n=%lld)\n",
                vs.max_abs, vs.mean_abs, vs.max_rel, (long long)vs.n);
    std::printf("audio  out: max_abs=%.3e  mean_abs=%.3e  max_rel=%.3e  (n=%lld)\n",
                as_.max_abs, as_.mean_abs, as_.max_rel, (long long)as_.n);
    std::printf("\n%s (tol_abs=%.0e)\n", pass ? "PASS" : "FAIL", tol_abs);

    ggml_backend_free(backend);
    return pass ? 0 : 1;
}
