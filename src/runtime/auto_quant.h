#ifndef __SD_RUNTIME_AUTO_QUANT_H__
#define __SD_RUNTIME_AUTO_QUANT_H__

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml.h"

/* Port of llama.cpp's auto-tensor-type tool (q3_pt branch) to diffusion
 * models. During a normal generation run it captures MUL_MAT inputs and
 * reference outputs for representative weights of each (role, layer-bucket)
 * class of the diffusion model; afterwards it quantizes each captured weight
 * to every candidate type (imatrix-weighted, exactly as `-M convert` would),
 * re-runs the matmuls on the captured activations, measures the relative-L2
 * output error, and solves an element-weighted multi-choice knapsack for the
 * lowest-error type assignment within a bits-per-weight budget. The result is
 * written as a --tensor-type-rules string for `sd-cli -M convert`. */

// How the per-capture output error is reduced over output rows. LLM logits are
// fine with the mean, but diffusion glyph formation is punished by worst-case
// error, so Tail/MaxMean expose the catastrophic per-tensor trades that a mean
// hides. (Iteration 1.)
enum class CostMode { Mean, Tail, MaxMean };

// Which denoise occurrences of a weight are kept. Uniform spaces samples across
// the whole trajectory; Low keeps the most-recent (lowest-sigma / detail-
// formation) occurrences via a ring buffer. (Iteration 3.)
enum class SigmaMode { Uniform, Low };

struct AutoQuantConfig {
    std::string out_path;
    float target_bpw   = 4.0f;
    float bpw_tol_high = 0.05f;  // may exceed the target by this much
    float bpw_tol_low  = 0.5f;   // may undershoot by this much
    // Candidate quant types (default: q2_K..q6_K, q8_0, iq4_xs).
    std::vector<ggml_type> quant_types;

    // --- Iteration 1: tail-aware cost -------------------------------------
    CostMode cost_mode = CostMode::Mean;
    float tail_q       = 0.95f;  // quantile of per-row rel-L2 for Tail mode
    float tail_lambda  = 1.0f;   // weight of max term for MaxMean mode

    // --- Iteration 2: per-role bpw floor ----------------------------------
    // Drop candidate types below this bpw for roles matching floor_roles.
    // 0 disables. floor_roles is a '|'-separated list of case-insensitive
    // substrings matched against the role name.
    float role_floor_bpw    = 0.0f;
    std::string floor_roles = "attention|attn|adaln|ada_ln|modulation|mod.";

    // --- Iteration 3: low-sigma capture bias ------------------------------
    // Default: Low. A 2x2 (sigma x samples) factorial on Z-Image showed the
    // low-sigma ring buffer cuts output error ~15% (mean rel-L2 / MSE-to-full-
    // precision) vs Uniform, consistently across two backends (Vulkan, ROCm)
    // and independent of sample count. Set sigma=uniform to restore trajectory-
    // wide sampling. See docs/AUTO_TYPE_FINDINGS.md.
    SigmaMode sigma_mode = SigmaMode::Low;

    // --- Iteration 4: end-to-end candidate validation ---------------------
    // Emit the top-K distinct DP assignments (best -> out_path, the rest to
    // out_path.candN) plus a manifest, so an e2e driver can generate with each
    // and pick the one closest to the full-precision reference.
    int top_k = 1;
    // Layer buckets per role: early/middle/late layers of the same role can
    // get different types. Representative layers are sampled per bucket.
    int n_layer_buckets   = 3;
    int n_reps_per_bucket = 2;
    // Activation samples kept per representative weight, spaced by an
    // occurrence stride so different denoise timesteps are covered.
    int n_samples_per_weight = 2;
    int occurrence_stride    = 7;
    // Token columns kept per capture (strided subsample) to bound host RAM.
    int64_t max_tokens = 256;
    // Tensors smaller than this are not measured (they keep the source type).
    int64_t min_elements = 40000;
    int n_threads        = 4;
};

// One captured MUL_MAT: strided token subset of the input activations and of
// the reference output, for one representative weight.
struct AutoQuantCapture {
    std::string weight_name;
    int64_t weight_ne0 = 0;  // input features
    int64_t weight_ne1 = 0;  // output features
    int64_t n_tokens   = 0;  // captured token columns
    std::vector<float> input;       // [weight_ne0, n_tokens]
    std::vector<float> ref_output;  // [weight_ne1, n_tokens]
};

class AutoQuantCollector {
private:
    AutoQuantConfig cfg_;
    std::string model_path_;
    std::string prefix_;
    bool active_        = false;
    bool chain_imatrix_ = false;

    // Inventory (built from the model metadata at enable time)
    struct TensorEntry {
        std::string name;
        std::string role;  // name with the layer index replaced by '#'
        int layer  = -1;   // -1 for non-block tensors
        int bucket = -1;
        int64_t ne0 = 0, ne1 = 0;
        int64_t n_elements = 0;
        bool quantizable   = false;  // measured + assigned by the optimizer
    };
    std::vector<TensorEntry> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;

    // Capture state
    std::unordered_set<std::string> target_names_;
    std::unordered_map<std::string, int> occurrences_;
    std::unordered_map<std::string, int> samples_taken_;
    // weight_name -> captures. In Uniform mode this fills to n_samples_per_weight
    // and stops; in Low mode it is a ring buffer keeping the most-recent
    // n_samples_per_weight occurrences. Regrouped by (role,bucket) in finish().
    std::map<std::string, std::vector<AutoQuantCapture>> captures_;
    // per-node ask results (the runner calls ask/collect strictly in order)
    std::unordered_map<const ggml_tensor*, std::pair<bool, bool>> pending_;

    bool build_inventory();
    bool want_capture(const std::string& weight_name);
    void do_capture(struct ggml_tensor* t, const std::string& weight_name);

public:
    bool enable(const std::string& diffusion_model_path,
                const std::string& options,
                bool chain_imatrix);
    bool collect(struct ggml_tensor* t, bool ask);
    // Runs the analysis and writes the rules file. Returns false on failure.
    bool finish();
    bool is_active() const { return active_; }
};

AutoQuantCollector& get_auto_quant_collector();

#endif  // __SD_RUNTIME_AUTO_QUANT_H__
