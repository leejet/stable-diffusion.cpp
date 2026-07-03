#include "runtime/auto_quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <sstream>

#include "core/util.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "model_loader.h"
#include "runtime/imatrix.h"

/* Adapted from llama.cpp's tools/auto-tensor-type (q3_pt branch). */

namespace {

constexpr char RB_SEP = '\x01';

std::string make_rb_key(const std::string& role, int bucket) {
    return role + RB_SEP + std::to_string(bucket);
}

std::string rb_role(const std::string& key) {
    auto p = key.find(RB_SEP);
    return (p == std::string::npos) ? key : key.substr(0, p);
}

int rb_bucket(const std::string& key) {
    auto p = key.find(RB_SEP);
    return (p == std::string::npos) ? 0 : std::stoi(key.substr(p + 1));
}

std::string rb_display(const std::string& key) {
    int b = rb_bucket(key);
    if (b < 0) {
        return rb_role(key) + "[G]";
    }
    return rb_role(key) + "[" + std::to_string(b) + "]";
}

// Bucket of a layer given its position within its role class.
int compute_bucket(int pos_in_class, int n_in_class, int n_buckets) {
    if (n_in_class <= 1 || n_buckets <= 1) {
        return 0;
    }
    int b = (int)(((long long)n_buckets * pos_in_class) / n_in_class);
    return std::min(std::max(b, 0), n_buckets - 1);
}

// "model.diffusion_model.layers.30.attention.qkv.weight"
//   -> role "model.diffusion_model.layers.#.attention.qkv.weight", layer 30
// Returns false (layer -1) for tensors without a block index.
bool split_block_name(const std::string& name, std::string& role, int& layer) {
    static const char* block_keywords[] = {"transformer_blocks.", "joint_blocks.", "double_blocks.",
                                           "single_blocks.", "blocks.", "block.", "layers."};
    for (const char* keyword : block_keywords) {
        size_t pos = name.find(keyword);
        if (pos == std::string::npos) {
            continue;
        }
        pos += strlen(keyword);
        size_t end = pos;
        while (end < name.size() && name[end] >= '0' && name[end] <= '9') {
            end++;
        }
        if (end > pos && (end == name.size() || name[end] == '.')) {
            layer = atoi(name.substr(pos, end - pos).c_str());
            role  = name.substr(0, pos) + "#" + name.substr(end);
            return true;
        }
    }
    role  = name;
    layer = -1;
    return false;
}

// remove backend decorations: CUDA0#model.diffusion_model...#0 -> model.diffusion_model...
std::string filter_tensor_name(const char* name) {
    std::string wname;
    const char* p = strchr(name, '#');
    if (p != NULL) {
        p             = p + 1;
        const char* q = strchr(p, '#');
        if (q != NULL) {
            wname = std::string(p, q - p);
        } else {
            wname = p;
        }
    } else {
        wname = name;
    }
    return wname;
}

// Relative squared L2 error per output row: ||ref - quant||^2 / ||ref||^2.
// Scale-free, so residual-stream writers (small-magnitude outputs) are not
// underweighted the way a softmax-KLD over raw matmul outputs would.
double relative_l2_row(const float* ref, const float* quant, int64_t n) {
    double num = 0;
    double den = 0;
    for (int64_t i = 0; i < n; i++) {
        double r = (double)ref[i];
        double d = r - (double)quant[i];
        num += d * d;
        den += r * r;
    }
    if (den <= 1e-20) {
        return num > 1e-20 ? 1.0 : 0.0;
    }
    return num / den;
}

double relative_l2(const float* ref, const float* quant, int64_t ne0, int64_t ne1) {
    double total       = 0;
    int64_t valid_rows = 0;
    for (int64_t row = 0; row < ne1; row++) {
        double e = relative_l2_row(ref + row * ne0, quant + row * ne0, ne0);
        if (std::isfinite(e)) {
            total += e;
            valid_rows++;
        }
    }
    return valid_rows > 0 ? total / valid_rows : 1e30;
}

double type_bpw(ggml_type type) {
    return 8.0 * (double)ggml_type_size(type) / (double)ggml_blck_size(type);
}

// result = weight^T * input on the given backend.
bool eval_mul_mat(ggml_type weight_type,
                  const void* weight_data,
                  int64_t weight_ne0,
                  int64_t weight_ne1,
                  const float* input_data,
                  int64_t input_ne0,
                  int64_t input_ne1,
                  std::vector<float>& result_data,
                  ggml_backend_t backend) {
    size_t weight_bytes = ggml_row_size(weight_type, weight_ne0) * weight_ne1;
    size_t input_bytes  = (size_t)input_ne0 * input_ne1 * sizeof(float);

    size_t ctx_size          = ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 4096;
    ggml_init_params params  = {ctx_size, NULL, /*no_alloc=*/true};
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    struct ggml_tensor* w      = ggml_new_tensor_2d(ctx, weight_type, weight_ne0, weight_ne1);
    struct ggml_tensor* x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_ne0, input_ne1);
    struct ggml_tensor* result = ggml_mul_mat(ctx, w, x);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(w, weight_data, 0, weight_bytes);
    ggml_backend_tensor_set(x, input_data, 0, input_bytes);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    result_data.resize((size_t)weight_ne1 * input_ne1);
    ggml_backend_tensor_get(result, result_data.data(), 0, ggml_nbytes(result));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

struct CostEntry {
    double err = 1e30;  // element-averaged relative-L2
    double bpw = 0;
    bool allowed = false;  // some tensor in the group cannot take this type
};

struct Assignment {
    std::map<std::string, ggml_type> rb_to_type;
    double total_err = 0;
    double total_bpw = 0;
    bool valid       = false;
};

bool parse_options(const std::string& options, AutoQuantConfig& cfg) {
    std::stringstream ss(options);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t eq = item.find('=');
        if (eq == std::string::npos) {
            LOG_WARN("auto-tensor-type: ignoring malformed option '%s'", item.c_str());
            continue;
        }
        std::string key = item.substr(0, eq);
        std::string val = item.substr(eq + 1);
        if (key == "out") {
            cfg.out_path = val;
        } else if (key == "bpw") {
            cfg.target_bpw = std::stof(val);
        } else if (key == "tol-high") {
            cfg.bpw_tol_high = std::stof(val);
        } else if (key == "tol-low") {
            cfg.bpw_tol_low = std::stof(val);
        } else if (key == "types") {
            cfg.quant_types.clear();
            std::stringstream ts(val);
            std::string tname;
            while (std::getline(ts, tname, '|')) {
                ggml_type found = GGML_TYPE_COUNT;
                for (int i = 0; i < GGML_TYPE_COUNT; i++) {
                    const auto* traits = ggml_get_type_traits((ggml_type)i);
                    if (traits->type_name != nullptr && tname == traits->type_name) {
                        found = (ggml_type)i;
                        break;
                    }
                }
                if (found == GGML_TYPE_COUNT) {
                    LOG_ERROR("auto-tensor-type: unknown quant type '%s'", tname.c_str());
                    return false;
                }
                cfg.quant_types.push_back(found);
            }
        } else if (key == "buckets") {
            cfg.n_layer_buckets = std::stoi(val);
        } else if (key == "reps") {
            cfg.n_reps_per_bucket = std::stoi(val);
        } else if (key == "samples") {
            cfg.n_samples_per_weight = std::stoi(val);
        } else if (key == "stride") {
            cfg.occurrence_stride = std::stoi(val);
        } else if (key == "max-tokens") {
            cfg.max_tokens = std::stoll(val);
        } else if (key == "min-elements") {
            cfg.min_elements = std::stoll(val);
        } else if (key == "threads") {
            cfg.n_threads = std::stoi(val);
        } else {
            LOG_WARN("auto-tensor-type: ignoring unknown option '%s'", key.c_str());
        }
    }
    if (cfg.out_path.empty()) {
        LOG_ERROR("auto-tensor-type: missing required option out=<path>");
        return false;
    }
    if (cfg.target_bpw <= 0.f) {
        LOG_ERROR("auto-tensor-type: missing/invalid bpw=<target>");
        return false;
    }
    if (cfg.quant_types.empty()) {
        // Default candidate ladder, ~2.5-8.5 bpw. The IQ types are the
        // interesting low end and are what the imatrix buys quality for;
        // IQ2_XXS/IQ2_XS/IQ1_S are not defaulted (near-unusable without a
        // strong imatrix) but can be requested via types=.
        for (ggml_type t : {GGML_TYPE_Q2_K, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
                            GGML_TYPE_Q3_K, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K,
                            GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {
            cfg.quant_types.push_back(t);
        }
    }
    return true;
}

std::string escape_regex(const std::string& s) {
    std::string out;
    out.reserve(s.size() * 2);
    for (char c : s) {
        if (strchr(".^$|()[]{}*+?\\", c) != nullptr) {
            out += '\\';
        }
        out += c;
    }
    return out;
}

}  // namespace

// ============================================================================
// Inventory + capture
// ============================================================================

bool AutoQuantCollector::build_inventory() {
    ModelLoader loader;
    if (!loader.init_from_file(model_path_, prefix_)) {
        LOG_ERROR("auto-tensor-type: failed to read model metadata from '%s'", model_path_.c_str());
        return false;
    }
    loader.convert_tensors_name();

    // Collect 2D weights; group block tensors into role classes.
    std::map<std::string, std::vector<int>> role_layers;
    for (const auto& [name, ts] : loader.get_tensor_storage_map()) {
        if (ts.n_dims != 2 || ts.ne[0] <= 1 || ts.ne[1] <= 1) {
            continue;
        }
        TensorEntry e;
        e.name       = name;
        e.ne0        = ts.ne[0];
        e.ne1        = ts.ne[1];
        e.n_elements = ts.ne[0] * ts.ne[1];
        split_block_name(name, e.role, e.layer);
        e.quantizable = e.n_elements >= cfg_.min_elements;
        tensor_index_[name] = tensors_.size();
        tensors_.push_back(e);
        if (e.quantizable && e.layer >= 0) {
            role_layers[e.role].push_back(e.layer);
        }
    }
    if (tensors_.empty()) {
        LOG_ERROR("auto-tensor-type: no 2D weights found under prefix '%s'", prefix_.c_str());
        return false;
    }

    // Assign buckets by position within the role class.
    for (auto& [role, layers] : role_layers) {
        std::sort(layers.begin(), layers.end());
        layers.erase(std::unique(layers.begin(), layers.end()), layers.end());
    }
    for (auto& e : tensors_) {
        if (!e.quantizable) {
            continue;
        }
        if (e.layer < 0) {
            e.bucket = -1;  // global tensor: its own measurement group
            continue;
        }
        const auto& layers = role_layers[e.role];
        int pos            = (int)(std::lower_bound(layers.begin(), layers.end(), e.layer) - layers.begin());
        e.bucket           = compute_bucket(pos, (int)layers.size(), cfg_.n_layer_buckets);
    }

    // Representative layers per (role, bucket): first + last + evenly spaced.
    for (const auto& [role, layers] : role_layers) {
        std::map<int, std::vector<int>> layers_by_bucket;
        for (size_t i = 0; i < layers.size(); i++) {
            layers_by_bucket[compute_bucket((int)i, (int)layers.size(), cfg_.n_layer_buckets)].push_back(layers[i]);
        }
        for (const auto& [bucket, bl] : layers_by_bucket) {
            std::vector<int> reps;
            reps.push_back(bl.front());
            if (bl.size() > 1) {
                reps.push_back(bl.back());
            }
            for (int k = 1; (int)reps.size() < cfg_.n_reps_per_bucket && k < (int)bl.size() - 1; k++) {
                int idx = (int)(((long long)k * bl.size()) / (cfg_.n_reps_per_bucket + 1));
                if (std::find(reps.begin(), reps.end(), bl[idx]) == reps.end()) {
                    reps.push_back(bl[idx]);
                }
            }
            for (const auto& e : tensors_) {
                if (e.quantizable && e.role == role && e.bucket == bucket &&
                    std::find(reps.begin(), reps.end(), e.layer) != reps.end()) {
                    target_names_.insert(e.name);
                }
            }
        }
    }
    // Global quantizable tensors are always measured directly.
    for (const auto& e : tensors_) {
        if (e.quantizable && e.layer < 0) {
            target_names_.insert(e.name);
        }
    }

    size_t n_quantizable = 0;
    for (const auto& e : tensors_) {
        n_quantizable += e.quantizable ? 1 : 0;
    }
    LOG_INFO("auto-tensor-type: %zu 2D weights (%zu quantizable), %zu role classes, %zu capture targets",
             tensors_.size(), n_quantizable, role_layers.size(), target_names_.size());
    return true;
}

bool AutoQuantCollector::enable(const std::string& diffusion_model_path,
                                const std::string& options,
                                bool chain_imatrix) {
    cfg_ = AutoQuantConfig();
    if (!parse_options(options, cfg_)) {
        return false;
    }
    model_path_    = diffusion_model_path;
    prefix_        = "model.diffusion_model.";
    chain_imatrix_ = chain_imatrix;
    tensors_.clear();
    tensor_index_.clear();
    target_names_.clear();
    occurrences_.clear();
    samples_taken_.clear();
    captures_.clear();
    pending_.clear();

    if (!build_inventory()) {
        return false;
    }
    active_ = true;

    std::string type_list;
    for (ggml_type t : cfg_.quant_types) {
        if (!type_list.empty()) {
            type_list += "|";
        }
        type_list += ggml_type_name(t);
    }
    LOG_INFO("auto-tensor-type: target %.3f bpw, candidates %s, %d buckets x %d reps, %d samples/weight",
             cfg_.target_bpw, type_list.c_str(), cfg_.n_layer_buckets, cfg_.n_reps_per_bucket,
             cfg_.n_samples_per_weight);
    return true;
}

bool AutoQuantCollector::want_capture(const std::string& weight_name) {
    if (target_names_.find(weight_name) == target_names_.end()) {
        return false;
    }
    int occurrence = occurrences_[weight_name]++;
    if (samples_taken_[weight_name] >= cfg_.n_samples_per_weight) {
        return false;
    }
    // Space the samples across occurrences so different denoise timesteps
    // (and cond/uncond passes) are represented.
    return occurrence % std::max(1, cfg_.occurrence_stride) == 0;
}

void AutoQuantCollector::do_capture(struct ggml_tensor* t, const std::string& weight_name) {
    const struct ggml_tensor* src0 = t->src[0];
    const struct ggml_tensor* src1 = t->src[1];

    auto it = tensor_index_.find(weight_name);
    if (it == tensor_index_.end()) {
        return;
    }
    const TensorEntry& entry = tensors_[it->second];

    // Only plain 2D activations (collapse higher batch dims by treating
    // ne1*ne2*ne3 as tokens; matmul is column-wise so this is exact).
    const int64_t in_ne0    = src1->ne[0];
    const int64_t in_tokens = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const int64_t out_ne0   = t->ne[0];
    if (in_ne0 != entry.ne0 || out_ne0 != entry.ne1) {
        return;  // unexpected shape (e.g. weight used transposed) — skip
    }
    if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(t)) {
        return;
    }

    const int64_t take   = std::min<int64_t>(in_tokens, cfg_.max_tokens);
    const int64_t stride = std::max<int64_t>(1, in_tokens / take);

    AutoQuantCapture cap;
    cap.weight_name = weight_name;
    cap.weight_ne0  = entry.ne0;
    cap.weight_ne1  = entry.ne1;
    cap.n_tokens    = take;
    cap.input.resize((size_t)take * in_ne0);
    cap.ref_output.resize((size_t)take * out_ne0);

    // Column-strided copy of the token subset (per-token columns are
    // contiguous in ggml layout).
    for (int64_t k = 0; k < take; k++) {
        const int64_t col = std::min(k * stride, in_tokens - 1);
        ggml_backend_tensor_get(const_cast<ggml_tensor*>(src1),
                                cap.input.data() + k * in_ne0,
                                (size_t)col * in_ne0 * sizeof(float),
                                (size_t)in_ne0 * sizeof(float));
        ggml_backend_tensor_get(t,
                                cap.ref_output.data() + k * out_ne0,
                                (size_t)col * out_ne0 * sizeof(float),
                                (size_t)out_ne0 * sizeof(float));
    }

    samples_taken_[weight_name]++;
    captures_[make_rb_key(entry.role, entry.bucket)].push_back(std::move(cap));
}

bool AutoQuantCollector::collect(struct ggml_tensor* t, bool ask) {
    if (!active_ || t == nullptr) {
        // Never return false on a collect pass — the runner treats that as an
        // abort request.
        return !ask;
    }

    bool imatrix_wants = false;
    if (chain_imatrix_ && ask) {
        // Forward strictly following the ask/collect protocol: only collect
        // for the imatrix if it asked for this tensor.
        imatrix_wants = get_imatrix_collector().collect_imatrix(t, true, nullptr);
    }

    bool mine = false;
    if (t->op == GGML_OP_MUL_MAT && t->src[0] != nullptr && t->src[1] != nullptr &&
        t->src[1]->type == GGML_TYPE_F32 && t->type == GGML_TYPE_F32) {
        std::string wname = filter_tensor_name(t->src[0]->name);
        if (ask) {
            mine = want_capture(wname);
        } else {
            auto pit = pending_.find(t);
            if (pit != pending_.end() && pit->second.second) {
                do_capture(t, wname);
            }
        }
    }

    if (ask) {
        if (imatrix_wants || mine) {
            pending_[t] = {imatrix_wants, mine};
        } else {
            pending_.erase(t);
        }
        return imatrix_wants || mine;
    }

    auto pit = pending_.find(t);
    if (pit != pending_.end()) {
        if (pit->second.first) {
            get_imatrix_collector().collect_imatrix(t, false, nullptr);
        }
        pending_.erase(pit);
    }
    return true;
}

// ============================================================================
// Analysis
// ============================================================================

bool AutoQuantCollector::finish() {
    if (!active_) {
        return false;
    }
    active_ = false;

    size_t n_captures = 0;
    for (const auto& [rb, caps] : captures_) {
        n_captures += caps.size();
    }
    if (n_captures == 0) {
        LOG_ERROR("auto-tensor-type: no activations captured — did the generation run?");
        return false;
    }
    LOG_INFO("auto-tensor-type: analyzing %zu captures across %zu (role, bucket) groups",
             n_captures, captures_.size());

    ModelLoader loader;
    if (!loader.init_from_file(model_path_, prefix_)) {
        return false;
    }
    loader.convert_tensors_name();
    loader.set_n_threads(cfg_.n_threads);
    const auto& storage_map = loader.get_tensor_storage_map();

    ggml_backend_t backend = ggml_backend_init_best();
    if (backend == nullptr) {
        LOG_ERROR("auto-tensor-type: failed to init a backend for evaluation");
        return false;
    }
    LOG_INFO("auto-tensor-type: evaluating on %s", ggml_backend_name(backend));

    for (ggml_type t : cfg_.quant_types) {
        ggml_quantize_init(t);
    }

    // ---- Cost matrix: (role, bucket) x qtype -> mean relative-L2 ----
    std::map<std::string, std::map<ggml_type, CostEntry>> cost_matrix;
    int group_idx = 0;
    for (auto& [rb, caps] : captures_) {
        group_idx++;
        // group captures by weight so each weight is read+quantized once
        std::map<std::string, std::vector<const AutoQuantCapture*>> by_weight;
        for (const auto& cap : caps) {
            by_weight[cap.weight_name].push_back(&cap);
        }
        LOG_INFO("auto-tensor-type: [%d/%zu] %s (%zu weights, %zu captures)",
                 group_idx, captures_.size(), rb_display(rb).c_str(), by_weight.size(), caps.size());

        std::map<ggml_type, double> err_sums;
        std::map<ggml_type, int> err_counts;

        for (const auto& [weight_name, wcaps] : by_weight) {
            auto sit = storage_map.find(weight_name);
            if (sit == storage_map.end()) {
                LOG_WARN("auto-tensor-type: '%s' not in model file, skipping", weight_name.c_str());
                continue;
            }
            std::vector<float> weight_f32;
            if (!loader.load_float_tensor(weight_name, weight_f32, cfg_.n_threads)) {
                LOG_WARN("auto-tensor-type: failed to load '%s', skipping", weight_name.c_str());
                continue;
            }
            const int64_t ne0 = wcaps.front()->weight_ne0;
            const int64_t ne1 = wcaps.front()->weight_ne1;
            if ((int64_t)weight_f32.size() != ne0 * ne1) {
                LOG_WARN("auto-tensor-type: '%s' size mismatch (%zu vs %lld)", weight_name.c_str(),
                         weight_f32.size(), (long long)(ne0 * ne1));
                continue;
            }

            std::vector<float> imatrix = get_imatrix_collector().get_values(weight_name);
            if ((int64_t)imatrix.size() != ne0) {
                if (!imatrix.empty()) {
                    LOG_WARN("auto-tensor-type: imatrix size mismatch for '%s' (%zu vs %lld); using uniform",
                             weight_name.c_str(), imatrix.size(), (long long)ne0);
                }
                imatrix.assign(ne0, 1.0f);
            }

            // Quantize once per type (parallel on CPU), then evaluate serially
            // on the backend.
            std::map<ggml_type, std::vector<uint8_t>> quantized;
            {
                std::mutex m;
                std::vector<std::future<void>> futures;
                for (ggml_type qt : cfg_.quant_types) {
                    if (!loader.tensor_should_be_converted(sit->second, qt)) {
                        continue;
                    }
                    futures.push_back(std::async(std::launch::async, [&, qt]() {
                        std::vector<uint8_t> data(ggml_row_size(qt, ne0) * ne1);
                        ggml_quantize_chunk(qt, weight_f32.data(), data.data(), 0, ne1, ne0, imatrix.data());
                        std::lock_guard<std::mutex> lock(m);
                        quantized[qt] = std::move(data);
                    }));
                    if ((int)futures.size() >= cfg_.n_threads) {
                        for (auto& f : futures) {
                            f.get();
                        }
                        futures.clear();
                    }
                }
                for (auto& f : futures) {
                    f.get();
                }
            }

            for (const auto* cap : wcaps) {
                for (const auto& [qt, qdata] : quantized) {
                    std::vector<float> quant_output;
                    if (!eval_mul_mat(qt, qdata.data(), ne0, ne1,
                                      cap->input.data(), ne0, cap->n_tokens,
                                      quant_output, backend)) {
                        continue;
                    }
                    double err = relative_l2(cap->ref_output.data(), quant_output.data(),
                                             cap->weight_ne1, cap->n_tokens);
                    if (std::isfinite(err)) {
                        err_sums[qt] += err;
                        err_counts[qt]++;
                    }
                }
            }
        }

        for (ggml_type qt : cfg_.quant_types) {
            CostEntry entry;
            entry.bpw = type_bpw(qt);
            if (err_counts.count(qt) != 0 && err_counts[qt] > 0) {
                entry.err     = err_sums[qt] / err_counts[qt];
                entry.allowed = true;
            }
            cost_matrix[rb][qt] = entry;
        }
        std::vector<AutoQuantCapture>().swap(caps);  // release capture memory
    }
    ggml_backend_free(backend);

    // ---- BPW accounting over the whole diffusion model ----
    // Quantizable groups are DP items; everything else keeps its source type
    // and counts as fixed overhead bits.
    std::map<std::string, int64_t> rb_elements;
    std::map<std::string, std::vector<int>> rb_layers;
    int64_t total_elements = 0;
    double fixed_bits      = 0;
    for (const auto& [name, ts] : storage_map) {
        if (name.compare(0, prefix_.size(), prefix_) != 0) {
            continue;
        }
        total_elements += ts.nelements();
        auto iit          = tensor_index_.find(name);
        bool in_dp        = false;
        if (iit != tensor_index_.end()) {
            const TensorEntry& e = tensors_[iit->second];
            std::string rb       = make_rb_key(e.role, e.bucket);
            if (e.quantizable && cost_matrix.count(rb) != 0) {
                rb_elements[rb] += e.n_elements;
                if (e.layer >= 0) {
                    rb_layers[rb].push_back(e.layer);
                }
                in_dp = true;
                // A type is only allowed for the group if every member tensor
                // can take it (block-size divisibility etc.).
                for (auto& [qt, ce] : cost_matrix[rb]) {
                    if (ce.allowed && !loader.tensor_should_be_converted(ts, qt)) {
                        ce.allowed = false;
                    }
                }
            }
        }
        if (!in_dp) {
            fixed_bits += (double)ts.nbytes() * 8.0;
        }
    }

    // ---- Element-weighted multi-choice knapsack DP ----
    struct DPChoice {
        ggml_type qt;
        int units;
        double err;
    };
    struct DPItem {
        std::string rb;
        int64_t n_elements;
        std::vector<DPChoice> choices;
    };

    constexpr int N_UNITS     = 16384;
    const double bits_per_unit = (double)total_elements / (double)N_UNITS;

    const double hi_bpw = cfg_.target_bpw + cfg_.bpw_tol_high;
    const double lo_bpw = std::max(0.0, (double)cfg_.target_bpw - (double)cfg_.bpw_tol_low);
    const int budget_hi = (int)std::ceil(hi_bpw * N_UNITS);
    const int budget_lo = (int)std::floor(lo_bpw * N_UNITS);
    const int fixed_units = (int)std::round(fixed_bits / bits_per_unit);
    const int overshoot   = (int)std::ceil(2.0 * N_UNITS);
    const int B           = budget_hi + overshoot + 1;

    std::vector<DPItem> items;
    for (const auto& [rb, row] : cost_matrix) {
        auto eit = rb_elements.find(rb);
        if (eit == rb_elements.end() || eit->second == 0) {
            continue;
        }
        DPItem item;
        item.rb         = rb;
        item.n_elements = eit->second;
        for (const auto& [qt, ce] : row) {
            if (!ce.allowed) {
                continue;
            }
            DPChoice c;
            c.qt    = qt;
            c.units = std::max(1, (int)std::round((double)item.n_elements * ce.bpw / bits_per_unit));
            c.err   = ce.err * (double)item.n_elements;
            item.choices.push_back(c);
        }
        if (!item.choices.empty()) {
            items.push_back(std::move(item));
        }
    }
    if (items.empty()) {
        LOG_ERROR("auto-tensor-type: empty cost matrix, nothing to optimize");
        return false;
    }

    const int n = (int)items.size();
    constexpr double INF = 1e300;
    std::vector<double> dp(B, INF), next_dp(B, INF);
    std::vector<std::vector<int>> choice_taken(n, std::vector<int>(B, -1));
    std::vector<std::vector<int>> prev_u(n, std::vector<int>(B, -1));

    if (fixed_units >= 0 && fixed_units < B) {
        dp[fixed_units] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        std::fill(next_dp.begin(), next_dp.end(), INF);
        for (int u = 0; u < B; u++) {
            if (dp[u] >= INF) {
                continue;
            }
            for (int ci = 0; ci < (int)items[i].choices.size(); ci++) {
                const auto& c = items[i].choices[ci];
                int nu        = u + c.units;
                if (nu >= B) {
                    continue;
                }
                double ne = dp[u] + c.err;
                if (ne < next_dp[nu]) {
                    next_dp[nu]         = ne;
                    choice_taken[i][nu] = ci;
                    prev_u[i][nu]       = u;
                }
            }
        }
        dp.swap(next_dp);
    }

    int best_u      = -1;
    double best_err = INF;
    for (int u = std::max(0, budget_lo); u <= budget_hi && u < B; u++) {
        if (dp[u] < best_err) {
            best_err = dp[u];
            best_u   = u;
        }
    }
    if (best_u < 0) {
        for (int u = 0; u <= budget_hi && u < B; u++) {
            if (dp[u] < best_err) {
                best_err = dp[u];
                best_u   = u;
            }
        }
    }
    if (best_u < 0) {
        for (int u = 0; u < B; u++) {
            if (dp[u] < INF) {
                best_u   = u;
                best_err = dp[u];
                LOG_WARN("auto-tensor-type: target %.3f bpw infeasible; best achievable is %.3f bpw",
                         cfg_.target_bpw, (double)u / N_UNITS);
                break;
            }
        }
    }
    if (best_u < 0) {
        LOG_ERROR("auto-tensor-type: no feasible assignment");
        return false;
    }

    Assignment assignment;
    assignment.valid = true;
    int cur_u        = best_u;
    for (int i = n - 1; i >= 0; i--) {
        int ci = choice_taken[i][cur_u];
        if (ci < 0) {
            LOG_ERROR("auto-tensor-type: DP reconstruction failed");
            return false;
        }
        assignment.rb_to_type[items[i].rb] = items[i].choices[ci].qt;
        cur_u                              = prev_u[i][cur_u];
    }
    assignment.total_bpw = (double)best_u / N_UNITS;
    assignment.total_err = best_err;

    // ---- Report + rules output ----
    LOG_INFO("auto-tensor-type: assignment at %.4f bpw (weighted rel-L2 %.4e):", assignment.total_bpw, best_err);
    for (const auto& item : items) {
        ggml_type qt   = assignment.rb_to_type[item.rb];
        const auto& ce = cost_matrix[item.rb][qt];
        LOG_INFO("  %-64s -> %-8s (%.2f bpw, err %.3e, %lld elem)",
                 rb_display(item.rb).c_str(), ggml_type_name(qt), ce.bpw, ce.err,
                 (long long)item.n_elements);
    }

    std::ofstream file(cfg_.out_path);
    if (!file) {
        LOG_ERROR("auto-tensor-type: cannot write '%s'", cfg_.out_path.c_str());
        return false;
    }
    std::string rules;
    for (const auto& [rb, qt] : assignment.rb_to_type) {
        std::string role = rb_role(rb);
        std::string pattern;
        size_t hash = role.find('#');
        if (hash == std::string::npos) {
            pattern = "^" + escape_regex(role) + "$";
        } else {
            auto lit = rb_layers.find(rb);
            std::string alternation = "\\d+";
            if (lit != rb_layers.end() && !lit->second.empty()) {
                std::vector<int> ls = lit->second;
                std::sort(ls.begin(), ls.end());
                ls.erase(std::unique(ls.begin(), ls.end()), ls.end());
                alternation = "(";
                for (size_t i = 0; i < ls.size(); i++) {
                    if (i) {
                        alternation += "|";
                    }
                    alternation += std::to_string(ls[i]);
                }
                alternation += ")";
            }
            pattern = "^" + escape_regex(role.substr(0, hash)) + alternation +
                      escape_regex(role.substr(hash + 1)) + "$";
        }
        if (!rules.empty()) {
            rules += ",";
        }
        rules += pattern + "=" + ggml_type_name(qt);
    }
    file << rules << "\n";
    file.close();
    LOG_INFO("auto-tensor-type: wrote rules to %s (pass as --tensor-type-rules \"$(cat %s)\" to -M convert)",
             cfg_.out_path.c_str(), cfg_.out_path.c_str());
    return true;
}

AutoQuantCollector& get_auto_quant_collector() {
    static AutoQuantCollector collector;
    return collector;
}

// ============================================================================
// C API
// ============================================================================

#include "stable-diffusion.h"

static bool auto_quant_eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    (void)user_data;
    return get_auto_quant_collector().collect(t, ask);
}

bool sd_enable_auto_tensor_type(const char* diffusion_model_path,
                                const char* options,
                                bool collect_imatrix_too) {
    if (diffusion_model_path == nullptr || options == nullptr) {
        return false;
    }
    if (!get_auto_quant_collector().enable(diffusion_model_path, options, collect_imatrix_too)) {
        return false;
    }
    sd_set_backend_eval_callback(auto_quant_eval_callback, nullptr);
    return true;
}

bool sd_finish_auto_tensor_type(void) {
    sd_set_backend_eval_callback(nullptr, nullptr);
    return get_auto_quant_collector().finish();
}
