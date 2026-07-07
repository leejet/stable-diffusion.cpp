#ifndef __SD_RUNTIME_AUTO_QUANT_H__
#define __SD_RUNTIME_AUTO_QUANT_H__

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml.h"

enum class CostMode { Mean,
                      Tail,
                      MaxMean };

enum class SigmaMode { Uniform,
                       Low };

struct AutoQuantConfig {
    std::string out_path;
    float target_bpw   = 4.0f;
    float bpw_tol_high = 0.05f;
    float bpw_tol_low  = 0.5f;

    std::vector<ggml_type> quant_types;

    CostMode cost_mode = CostMode::Mean;
    float tail_q       = 0.95f;
    float tail_lambda  = 1.0f;

    float role_floor_bpw    = 0.0f;
    std::string floor_roles = "attention|attn|adaln|ada_ln|modulation|mod.";

    SigmaMode sigma_mode = SigmaMode::Low;

    int top_k = 1;

    int n_layer_buckets   = 3;
    int n_reps_per_bucket = 2;

    int n_samples_per_weight = 2;
    int occurrence_stride    = 7;

    int64_t max_tokens = 256;

    int64_t min_elements = 40000;
    int n_threads        = 4;
};

struct AutoQuantCapture {
    std::string weight_name;
    int64_t weight_ne0 = 0;
    int64_t weight_ne1 = 0;
    int64_t n_tokens   = 0;
    std::vector<float> input;
    std::vector<float> ref_output;
};

class AutoQuantCollector {
private:
    AutoQuantConfig cfg_;
    std::string model_path_;
    std::string prefix_;
    bool active_        = false;
    bool chain_imatrix_ = false;

    struct TensorEntry {
        std::string name;
        std::string role;
        int layer   = -1;
        int bucket  = -1;
        int64_t ne0 = 0, ne1 = 0;
        int64_t n_elements = 0;
        bool quantizable   = false;
    };
    std::vector<TensorEntry> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;

    std::unordered_set<std::string> target_names_;
    std::unordered_map<std::string, int> occurrences_;
    std::unordered_map<std::string, int> samples_taken_;

    std::map<std::string, std::vector<AutoQuantCapture>> captures_;

    std::unordered_map<const ggml_tensor*, std::pair<bool, bool>> pending_;

    bool build_inventory();
    bool want_capture(const std::string& weight_name);
    void do_capture(struct ggml_tensor* t, const std::string& weight_name);

public:
    bool enable(const std::string& diffusion_model_path,
                const std::string& options,
                bool chain_imatrix);
    bool collect(struct ggml_tensor* t, bool ask);

    bool finish();
    bool is_active() const { return active_; }
};

AutoQuantCollector& get_auto_quant_collector();

#endif
