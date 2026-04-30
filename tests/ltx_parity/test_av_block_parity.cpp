// LTX-2 AV transformer block parity test.
//
// Loads weights + inputs + outputs dumped by tests/ltx_parity/dump_av_block.py,
// constructs an LTXTransformerBlock with the same flags (cross_attention_adaln,
// apply_gated_attention, audio dims), runs forward_av, and diffs against the
// python reference outputs.
//
// Tolerances: F32 CPU backend should match python torch.float32 to ~1e-5 abs /
// ~1e-4 rel. RoPE INTERLEAVED is well-tested in the existing LTX parity. The
// gated-attention sigmoid path adds a tiny amount of drift (per-head gate * 2)
// — still within tolerance for one block.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ltx.hpp"
#include "ltx_rope.hpp"
#include "model.h"
#include "tensor.hpp"

namespace {

// Tiny no-deps JSON-ish reader. dump_av_block.py emits a regular JSON object;
// we only need to look up nested fields like manifest["weights"][name]["shape"]
// and ["path"], plus manifest["config"][k]. To avoid pulling in a JSON library
// we shell out to /home/ilintar/venv/bin/python and read the manifest values
// directly via the load helpers below — the dump script's path is fixed.

const std::string REF_DIR = "/tmp/ltx_av_block_ref";

// Read raw f32 bytes from a file into an sd::Tensor with the given ggml ne.
sd::Tensor<float> load_raw(const std::string& path, const std::vector<int64_t>& ne) {
    sd::Tensor<float> t(ne);
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::fprintf(stderr, "fatal: cannot open %s\n", path.c_str());
        std::exit(2);
    }
    f.read(reinterpret_cast<char*>(t.data()),
           static_cast<std::streamsize>(t.numel() * sizeof(float)));
    if (!f.good()) {
        std::fprintf(stderr, "fatal: short read on %s\n", path.c_str());
        std::exit(2);
    }
    return t;
}

struct DiffStats {
    double max_abs    = 0.0;
    double mean_abs   = 0.0;
    double max_rel    = 0.0;
    int64_t n         = 0;
};

DiffStats diff_f32(const float* a, const float* b, int64_t n) {
    DiffStats s; s.n = n;
    double sum = 0.0;
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

// Build a [inner_dim, T, 2] PE tensor from a (cos, sin) pair where each input is
// [B=1, T, inner_dim] in raw f32 row-major. The output's ggml ne layout is
// (inner_dim fastest, T mid, slice index outer): slice 0 = cos bytes, slice 1 = sin.
sd::Tensor<float> build_pe_packed(const std::string& cos_path,
                                  const std::string& sin_path,
                                  int64_t inner_dim, int64_t T) {
    sd::Tensor<float> pe({inner_dim, T, 2});
    auto cos = load_raw(cos_path, {inner_dim, T, 1});
    auto sin = load_raw(sin_path, {inner_dim, T, 1});
    std::memcpy(pe.data(),                     cos.data(), cos.numel() * sizeof(float));
    std::memcpy(pe.data() + cos.numel(),       sin.data(), sin.numel() * sizeof(float));
    return pe;
}

// Custom GGMLRunner that owns a single LTXTransformerBlock with audio enabled
// and runs forward_av once with externally-supplied inputs.
struct AVBlockParityRunner : public GGMLRunner {
    LTX::LTXTransformerBlock block;

    sd::Tensor<float> vx, ax;
    sd::Tensor<float> v_ctx, a_ctx;
    sd::Tensor<float> v_mod, a_mod;
    sd::Tensor<float> v_pmod, a_pmod;        // prompt modulation
    sd::Tensor<float> v_pe, a_pe;
    sd::Tensor<float> v_cpe, a_cpe;
    sd::Tensor<float> v_css, a_css;          // cross_scale_shift_modulation
    sd::Tensor<float> v_cg,  a_cg;           // cross_gate_modulation

    sd::Tensor<float> result_v, result_a;    // captured after compute

    AVBlockParityRunner(ggml_backend_t backend,
                         int64_t v_dim, int v_h, int v_hd,
                         int64_t a_dim, int a_h, int a_hd,
                         int64_t v_ctx, int64_t a_ctx,
                         bool cross_attention_adaln, bool apply_gated_attention,
                         LTX::RopeType rope_type, float norm_eps)
        : GGMLRunner(backend, /*offload_params_to_cpu=*/false),
          block(v_dim, v_h, v_hd, v_ctx,
                cross_attention_adaln, apply_gated_attention, norm_eps,
                rope_type,
                a_dim, a_h, a_hd, a_ctx) {
        block.init(params_ctx, /*tensor_storage_map=*/{}, /*prefix=*/"");
    }

    std::string get_desc() override { return "AVBlockParityRunner"; }

    ggml_cgraph* build_graph() {
        auto gf = new_graph_custom(LTX::LTX_GRAPH_SIZE);

        ggml_tensor* g_vx     = make_input(vx);
        ggml_tensor* g_ax     = make_input(ax);
        ggml_tensor* g_vctx   = make_input(v_ctx);
        ggml_tensor* g_actx   = make_input(a_ctx);
        ggml_tensor* g_vmod   = make_input(v_mod);
        ggml_tensor* g_amod   = make_input(a_mod);
        ggml_tensor* g_vpmod  = v_pmod.empty() ? nullptr : make_input(v_pmod);
        ggml_tensor* g_apmod  = a_pmod.empty() ? nullptr : make_input(a_pmod);
        ggml_tensor* g_vpe    = make_input(v_pe);
        ggml_tensor* g_ape    = make_input(a_pe);
        ggml_tensor* g_vcpe   = make_input(v_cpe);
        ggml_tensor* g_acpe   = make_input(a_cpe);
        ggml_tensor* g_vcss   = make_input(v_css);
        ggml_tensor* g_acss   = make_input(a_css);
        ggml_tensor* g_vcg    = make_input(v_cg);
        ggml_tensor* g_acg    = make_input(a_cg);

        LTX::LTX2AVModalityArgs vargs;
        vargs.x = g_vx; vargs.context = g_vctx; vargs.modulation = g_vmod;
        vargs.pe = g_vpe; vargs.cross_pe = g_vcpe;
        vargs.prompt_modulation = g_vpmod;
        vargs.cross_scale_shift_modulation = g_vcss;
        vargs.cross_gate_modulation        = g_vcg;

        LTX::LTX2AVModalityArgs aargs;
        aargs.x = g_ax; aargs.context = g_actx; aargs.modulation = g_amod;
        aargs.pe = g_ape; aargs.cross_pe = g_acpe;
        aargs.prompt_modulation = g_apmod;
        aargs.cross_scale_shift_modulation = g_acss;
        aargs.cross_gate_modulation        = g_acg;

        auto rctx = get_context();
        auto outs = block.forward_av(&rctx, vargs, aargs);

        // Force both to live by adding to the graph; we'll fetch them by name.
        ggml_set_name(outs.first,  "av_video_out");
        ggml_set_name(outs.second, "av_audio_out");
        ggml_build_forward_expand(gf, outs.first);
        ggml_build_forward_expand(gf, outs.second);

        // Cache so we can fetch them post-compute via get_cache_tensor_by_name.
        cache("av_video_out", outs.first);
        cache("av_audio_out", outs.second);

        return gf;
    }

    bool compute_and_capture(int n_threads,
                             int64_t v_t, int64_t v_dim,
                             int64_t a_t, int64_t a_dim) {
        auto gg = [this]() { return build_graph(); };
        compute<float>(gg, n_threads, /*free_compute_buffer_immediately=*/false,
                       /*no_return=*/true);

        ggml_tensor* tv = get_cache_tensor_by_name("av_video_out");
        ggml_tensor* ta = get_cache_tensor_by_name("av_audio_out");
        if (tv == nullptr || ta == nullptr) {
            std::fprintf(stderr, "fatal: missing cached output tensor(s)\n");
            return false;
        }
        result_v = sd::Tensor<float>({v_dim, v_t, 1});
        result_a = sd::Tensor<float>({a_dim, a_t, 1});
        ggml_backend_tensor_get(tv, result_v.data(), 0, ggml_nbytes(tv));
        ggml_backend_tensor_get(ta, result_a.data(), 0, ggml_nbytes(ta));
        return true;
    }

    // Load a python-dumped weight by name into the corresponding block tensor.
    // Returns false if the name is unknown.
    bool load_weight_into(const std::string& name, const std::string& path) {
        std::map<std::string, ggml_tensor*> all;
        block.get_param_tensors(all, /*prefix=*/"");
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

// Read manifest.json by shelling to python (avoids a JSON dependency).
struct Manifest {
    std::map<std::string, std::vector<int64_t>> weights_shape;
    std::map<std::string, std::string>          weights_path;
    std::map<std::string, std::vector<int64_t>> inputs_shape;
    std::map<std::string, std::string>          inputs_path;
    std::map<std::string, std::vector<int64_t>> outputs_shape;
    std::map<std::string, std::string>          outputs_path;
    std::map<std::string, std::string>          config_str;
};

// Parse manifest.json via a one-shot python helper that prints simple lines.
// Format: SECTION:NAME:SHAPE,COMMA,SEP:PATH
Manifest read_manifest(const std::string& dir) {
    std::string cmd = "/home/ilintar/venv/bin/python -c '"
        "import json,sys\n"
        "m=json.load(open(\"" + dir + "/manifest.json\"))\n"
        "for k,v in m[\"config\"].items(): print(\"C:\"+k+\":\"+str(v))\n"
        "for sec in [\"weights\",\"inputs\",\"outputs\"]:\n"
        "    for n,d in m[sec].items():\n"
        "        sh=\",\".join(str(x) for x in d[\"shape\"])\n"
        "        print(sec[0].upper()+\":\"+n+\":\"+sh+\":\"+d[\"path\"])\n"
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
        size_t c1 = rest.find(':');
        if (c1 == std::string::npos) continue;
        std::string name = rest.substr(0, c1);
        std::string after = rest.substr(c1 + 1);
        if (tag == 'C') {
            m.config_str[name] = after;
            continue;
        }
        size_t c2 = after.find(':');
        if (c2 == std::string::npos) continue;
        std::string shape_str = after.substr(0, c2);
        std::string path = dir + "/" + after.substr(c2 + 1);
        std::vector<int64_t> shape;
        std::stringstream ss(shape_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty()) shape.push_back(std::stoll(tok));
        }
        if      (tag == 'W') { m.weights_shape[name] = shape; m.weights_path[name] = path; }
        else if (tag == 'I') { m.inputs_shape[name]  = shape; m.inputs_path[name]  = path; }
        else if (tag == 'O') { m.outputs_shape[name] = shape; m.outputs_path[name] = path; }
    }
    pclose(p);
    return m;
}

int parse_int(const std::map<std::string, std::string>& cfg, const std::string& k) {
    auto it = cfg.find(k);
    if (it == cfg.end()) { std::fprintf(stderr, "fatal: missing config %s\n", k.c_str()); std::exit(2); }
    return std::stoi(it->second);
}
bool parse_bool(const std::map<std::string, std::string>& cfg, const std::string& k) {
    auto it = cfg.find(k);
    if (it == cfg.end()) return false;
    return it->second == "True" || it->second == "true" || it->second == "1";
}

}  // namespace


int main() {
    Manifest m = read_manifest(REF_DIR);
    if (m.config_str.empty()) {
        std::fprintf(stderr, "fatal: empty manifest. Run dump_av_block.py first.\n");
        return 2;
    }

    // ---- Pull dims & flags from config (parsing a few python repr forms) ----
    auto parse_dict = [&](const std::string& key, const char* sub) -> int {
        // matches "{'dim': 128, ..." → grab the value after sub.
        auto it = m.config_str.find(key);
        if (it == m.config_str.end()) { std::fprintf(stderr, "fatal: %s\n", key.c_str()); std::exit(2); }
        std::string s = it->second;
        std::string needle = std::string("'") + sub + "': ";
        size_t p = s.find(needle);
        if (p == std::string::npos) { std::fprintf(stderr, "fatal: %s.%s\n", key.c_str(), sub); std::exit(2); }
        p += needle.size();
        size_t q = s.find_first_of(",}", p);
        return std::stoi(s.substr(p, q - p));
    };

    const int V_DIM   = parse_dict("video", "dim");
    const int V_H     = parse_dict("video", "heads");
    const int V_HD    = parse_dict("video", "d_head");
    const int V_CTX   = parse_dict("video", "ctx_dim");
    const int T_VIDEO = parse_dict("video", "T");
    const int S_VIDEO = parse_dict("video", "S");

    const int A_DIM   = parse_dict("audio", "dim");
    const int A_H     = parse_dict("audio", "heads");
    const int A_HD    = parse_dict("audio", "d_head");
    const int A_CTX   = parse_dict("audio", "ctx_dim");
    const int T_AUDIO = parse_dict("audio", "T");
    const int S_AUDIO = parse_dict("audio", "S");

    const bool cross_attention_adaln = parse_bool(m.config_str, "cross_attention_adaln");
    const bool apply_gated_attention = parse_bool(m.config_str, "apply_gated_attention");
    const float norm_eps             = std::stof(m.config_str.at("norm_eps"));
    const std::string rope_str       = m.config_str.at("rope_type");
    const LTX::RopeType rope_type =
        (rope_str.find("split") != std::string::npos) ? LTX::RopeType::SPLIT : LTX::RopeType::INTERLEAVED;

    std::printf("=== LTX-2 AV transformer block parity ===\n");
    std::printf("video: dim=%d heads=%d d_head=%d ctx=%d T=%d S=%d\n",
                V_DIM, V_H, V_HD, V_CTX, T_VIDEO, S_VIDEO);
    std::printf("audio: dim=%d heads=%d d_head=%d ctx=%d T=%d S=%d\n",
                A_DIM, A_H, A_HD, A_CTX, T_AUDIO, S_AUDIO);
    std::printf("flags: cross_attention_adaln=%d apply_gated_attention=%d rope=%s norm_eps=%g\n",
                cross_attention_adaln, apply_gated_attention, rope_str.c_str(), norm_eps);

    auto backend = ggml_backend_cpu_init();
    AVBlockParityRunner runner(backend,
                                V_DIM, V_H, V_HD,
                                A_DIM, A_H, A_HD,
                                V_CTX, A_CTX,
                                cross_attention_adaln, apply_gated_attention,
                                rope_type, norm_eps);
    runner.alloc_params_buffer();

    // ---- Load every weight by name ----
    int loaded = 0, skipped = 0;
    for (const auto& kv : m.weights_path) {
        if (!runner.load_weight_into(kv.first, kv.second)) {
            std::fprintf(stderr, "WARN: no slot for weight '%s' — skipping\n", kv.first.c_str());
            skipped++;
        } else {
            loaded++;
        }
    }
    std::printf("loaded %d weights (skipped %d)\n", loaded, skipped);
    if (skipped > 0) {
        std::fprintf(stderr, "FAIL: some weights were unmapped — implementation gap\n");
        ggml_backend_free(backend);
        return 1;
    }

    // ---- Load all inputs ----
    auto path_of = [&](const char* k) -> std::string {
        auto it = m.inputs_path.find(k);
        if (it == m.inputs_path.end()) return "";
        return it->second;
    };
    auto load_required = [&](const char* k, const std::vector<int64_t>& ne) -> sd::Tensor<float> {
        std::string p = path_of(k);
        if (p.empty()) {
            std::fprintf(stderr, "fatal: missing input %s\n", k); std::exit(2);
        }
        return load_raw(p, ne);
    };
    auto load_optional = [&](const char* k, const std::vector<int64_t>& ne) -> sd::Tensor<float> {
        std::string p = path_of(k);
        if (p.empty()) return sd::Tensor<float>();
        return load_raw(p, ne);
    };

    const int B = 1;
    const int num_main_mod = cross_attention_adaln ? 9 : 6;

    runner.vx    = load_required("video__x",       {V_DIM, T_VIDEO, B});
    runner.ax    = load_required("audio__x",       {A_DIM, T_AUDIO, B});
    runner.v_ctx = load_required("video__context", {V_CTX, S_VIDEO, B});
    runner.a_ctx = load_required("audio__context", {A_CTX, S_AUDIO, B});
    runner.v_mod = load_required("video__timesteps", {V_DIM, num_main_mod, B});
    runner.a_mod = load_required("audio__timesteps", {A_DIM, num_main_mod, B});

    if (cross_attention_adaln) {
        runner.v_pmod = load_required("video__prompt_timestep", {V_DIM, 2, B});
        runner.a_pmod = load_required("audio__prompt_timestep", {A_DIM, 2, B});
    }
    runner.v_pe   = build_pe_packed(path_of("video__pe_cos"), path_of("video__pe_sin"), V_DIM, T_VIDEO);
    runner.a_pe   = build_pe_packed(path_of("audio__pe_cos"), path_of("audio__pe_sin"), A_DIM, T_AUDIO);
    // Cross-modal RoPE: inner_dim_cross == audio.heads * audio.d_head == A_DIM (both modalities).
    runner.v_cpe  = build_pe_packed(path_of("video__cross_pe_cos"), path_of("video__cross_pe_sin"), A_DIM, T_VIDEO);
    runner.a_cpe  = build_pe_packed(path_of("audio__cross_pe_cos"), path_of("audio__cross_pe_sin"), A_DIM, T_AUDIO);
    runner.v_css  = load_required("video__cross_scale_shift_timestep", {V_DIM, 4, B});
    runner.a_css  = load_required("audio__cross_scale_shift_timestep", {A_DIM, 4, B});
    runner.v_cg   = load_required("video__cross_gate_timestep",        {V_DIM, 1, B});
    runner.a_cg   = load_required("audio__cross_gate_timestep",        {A_DIM, 1, B});

    // ---- Run ----
    if (!runner.compute_and_capture(/*n_threads=*/1, T_VIDEO, V_DIM, T_AUDIO, A_DIM)) {
        ggml_backend_free(backend);
        return 1;
    }

    // ---- Diff vs python outputs ----
    auto v_ref = load_raw(m.outputs_path.at("video__x_out"), {V_DIM, T_VIDEO, B});
    auto a_ref = load_raw(m.outputs_path.at("audio__x_out"), {A_DIM, T_AUDIO, B});

    auto vs = diff_f32(runner.result_v.data(), v_ref.data(), runner.result_v.numel());
    auto as_= diff_f32(runner.result_a.data(), a_ref.data(), runner.result_a.numel());

    const double tol_abs = 5e-5;  // generous for a 1-block forward at fp32
    bool pass = (vs.max_abs < tol_abs) && (as_.max_abs < tol_abs);

    std::printf("\nvideo  out: max_abs=%.3e  mean_abs=%.3e  max_rel=%.3e  (n=%lld)\n",
                vs.max_abs, vs.mean_abs, vs.max_rel, (long long)vs.n);
    std::printf("audio  out: max_abs=%.3e  mean_abs=%.3e  max_rel=%.3e  (n=%lld)\n",
                as_.max_abs, as_.mean_abs, as_.max_rel, (long long)as_.n);
    std::printf("\n%s (tol_abs=%.0e)\n", pass ? "PASS" : "FAIL", tol_abs);

    ggml_backend_free(backend);
    return pass ? 0 : 1;
}
