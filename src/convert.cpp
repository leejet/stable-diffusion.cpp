#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <regex>
#include <thread>
#include <vector>

#include "model.h"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "stable-diffusion.h"
#include "util.h"

#include "ggml-cpu.h"
#include "gguf.h"

#ifndef SAFE_STR
#define SAFE_STR(s) ((s) ? (s) : "")
#endif

// Candidate types for RMSE sweep, ordered coarsest to finest.
// find_best_type_for_rmse stops at the ceiling type, so order matters.
static const ggml_type RMSE_CANDIDATES[] = {
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    GGML_TYPE_F16,
};
static const int N_RMSE_CANDIDATES = (int)(sizeof(RMSE_CANDIDATES) / sizeof(RMSE_CANDIDATES[0]));

// Returns ||orig - recon||_2 / ||orig||_2, i.e. relative RMSE.
static float compute_relative_rmse(const float* orig, const float* recon, int64_t n) {
    double sum_orig_sq = 0.0, sum_delta_sq = 0.0;
    for (int64_t i = 0; i < n; i++) {
        sum_orig_sq += (double)orig[i] * orig[i];
        double d = (double)orig[i] - (double)recon[i];
        sum_delta_sq += d * d;
    }
    if (sum_orig_sq == 0.0) return 0.0f;
    return (float)std::sqrt(sum_delta_sq / sum_orig_sq);
}

// Sweep RMSE_CANDIDATES from coarsest to finest up to ceiling_type.
// Returns the coarsest type whose relative RMSE <= threshold.
// If no candidate meets the threshold, returns ceiling_type.
static ggml_type find_best_type_for_rmse(const float* data,
                                          int64_t nrows,
                                          int64_t n_per_row,
                                          ggml_type ceiling_type,
                                          float threshold) {
    int64_t n = nrows * n_per_row;
    std::vector<uint8_t> quant_buf;
    std::vector<float> recon(n);
    std::vector<float> imatrix(n_per_row, 1.0f);

    for (int ci = 0; ci < N_RMSE_CANDIDATES; ci++) {
        ggml_type ctype = RMSE_CANDIDATES[ci];

        // Skip candidates coarser than the ceiling (shouldn't happen with ordered list,
        // but guard in case ceiling is before the end of the array).
        bool at_ceiling = (ctype == ceiling_type);

        if (ggml_is_quantized(ctype) && n_per_row % ggml_blck_size(ctype) != 0) {
            if (at_ceiling) return ceiling_type;
            continue;
        }

        size_t qsize;
        if (ctype == GGML_TYPE_F16) {
            qsize = (size_t)n * sizeof(ggml_fp16_t);
        } else {
            qsize = (size_t)nrows * ggml_row_size(ctype, n_per_row);
        }
        quant_buf.resize(qsize);

        if (ctype == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(data, (ggml_fp16_t*)quant_buf.data(), n);
            ggml_fp16_to_fp32_row((ggml_fp16_t*)quant_buf.data(), recon.data(), n);
        } else {
            const ggml_type_traits* traits = ggml_get_type_traits(ctype);
            if (traits->to_float == nullptr) {
                if (at_ceiling) return ceiling_type;
                continue;
            }
            ggml_quantize_chunk(ctype, data, quant_buf.data(), 0, nrows, n_per_row, imatrix.data());
            traits->to_float(quant_buf.data(), recon.data(), n);
        }

        float rmse = compute_relative_rmse(data, recon.data(), n);
        if (rmse <= threshold) return ctype;
        if (at_ceiling) return ceiling_type;
    }

    return ceiling_type;
}

// ─── Normal (non-RMSE) export path ────────────────────────────────────────────

static ggml_type get_export_tensor_type(ModelLoader& model_loader,
                                        const TensorStorage& tensor_storage,
                                        ggml_type type,
                                        const TensorTypeRules& tensor_type_rules) {
    const std::string& name = tensor_storage.name;
    ggml_type tensor_type   = tensor_storage.type;
    ggml_type dst_type      = type;

    for (const auto& rule : tensor_type_rules) {
        std::regex pattern(rule.first);
        if (std::regex_search(name, pattern)) {
            dst_type = rule.second;
            break;
        }
    }

    if (model_loader.tensor_should_be_converted(tensor_storage, dst_type)) {
        tensor_type = dst_type;
    }

    return tensor_type;
}

static bool load_tensors_for_export(ModelLoader& model_loader,
                                    ggml_context* ggml_ctx,
                                    ggml_type type,
                                    const TensorTypeRules& tensor_type_rules,
                                    std::vector<TensorWriteInfo>& tensors) {
    std::mutex tensor_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& ts, ggml_tensor** dst_tensor) -> bool {
        ggml_type tensor_type = get_export_tensor_type(model_loader, ts, type, tensor_type_rules);

        std::lock_guard<std::mutex> lock(tensor_mutex);
        ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, tensor_type, ts.n_dims, ts.ne);
        if (tensor == nullptr) {
            LOG_ERROR("ggml_new_tensor failed");
            return false;
        }
        ggml_set_name(tensor, ts.name.c_str());

        if (!tensor->data) {
            GGML_ASSERT(ggml_nelements(tensor) == 0);
            tensor->data = ggml_get_mem_buffer(ggml_ctx);
        }

        TensorWriteInfo wi;
        wi.tensor = tensor;
        wi.n_dims = ts.n_dims;
        for (int i = 0; i < ts.n_dims; ++i) wi.ne[i] = ts.ne[i];

        *dst_tensor = tensor;
        tensors.push_back(std::move(wi));
        return true;
    };

    bool success = model_loader.load_tensors(on_new_tensor_cb);
    LOG_INFO("load tensors done");
    return success;
}

// ─── RMSE export path (streaming two-pass, low RAM) ──────────────────────────
//
// Pass 1: enumerate active tensors (null-dst callback, no data loaded), then
//         for each tensor: load as f32 → RMSE sweep → record target type → free.
//         Peak RAM = f32 size of the single largest tensor.
//
// Pass 2: write GGUF header (only_meta), then for each tensor: load as f32 →
//         quantize → write bytes at the correct file offset → free.
//         Same tiny peak RAM.

static ggml_type pick_target_type(const TensorStorage& ts,
                                   ModelLoader& model_loader,
                                   const TensorTypeRules& tensor_type_rules,
                                   ggml_type ceiling_type,
                                   float rmse_threshold,
                                   const float* data,
                                   int64_t n_per_row,
                                   int64_t nrows) {
    for (const auto& rule : tensor_type_rules) {
        std::regex pattern(rule.first);
        if (std::regex_search(ts.name, pattern)) return rule.second;
    }

    if (ceiling_type != GGML_TYPE_COUNT &&
        model_loader.tensor_should_be_converted(ts, ceiling_type)) {
        if (data != nullptr) {
            return find_best_type_for_rmse(data, nrows, n_per_row, ceiling_type, rmse_threshold);
        }
        return ceiling_type;
    }
    return GGML_TYPE_F16;
}

static bool convert_rmse_streaming(ModelLoader& model_loader,
                                    ggml_type ceiling_type,
                                    const TensorTypeRules& tensor_type_rules,
                                    float rmse_threshold,
                                    const std::string& output_path,
                                    std::string* error) {
    // Step 1: collect active tensor list without loading any data.
    // load_tensors uses a thread pool, so the callback is called concurrently.
    std::vector<TensorStorage> active;
    {
        std::mutex active_mtx;
        auto cb = [&](const TensorStorage& ts, ggml_tensor** dst) -> bool {
            std::lock_guard<std::mutex> lk(active_mtx);
            active.push_back(ts);
            *dst = nullptr;
            return true;
        };
        if (!model_loader.load_tensors(cb)) {
            if (error) *error = "failed to enumerate tensors";
            return false;
        }
    }
    LOG_INFO("RMSE sweep: %zu active tensors", active.size());

    // Step 2: type sweep — one tensor in RAM at a time.
    std::vector<ggml_type> target_types(active.size(), GGML_TYPE_F16);
    std::vector<float> f32_buf;

    int n_threads  = std::max(1, (int)sd_get_num_physical_cores());
    std::atomic<size_t> next_idx{0};
    std::mutex type_write_mutex;
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    // Worker threads each load their own tensor independently (different file offsets).
    auto worker_fn = [&]() {
        std::vector<float> local_buf;
        size_t i;
        while ((i = next_idx.fetch_add(1, std::memory_order_relaxed)) < active.size()) {
            const TensorStorage& ts = active[i];
            int64_t n          = ts.nelements();
            int64_t n_per_row  = ts.ne[0];
            int64_t nrows      = n / std::max(n_per_row, (int64_t)1);

            ggml_type ttype;
            if (n == 0 || !model_loader.tensor_should_be_converted(ts, ceiling_type)) {
                // Skip RMSE sweep for non-weight tensors; pick_target_type handles rules/F16.
                ttype = pick_target_type(ts, model_loader, tensor_type_rules,
                                         ceiling_type, rmse_threshold, nullptr, n_per_row, nrows);
            } else {
                if (!model_loader.load_tensor_f32(ts, local_buf)) {
                    LOG_WARN("RMSE sweep: failed to load '%s', defaulting to f16", ts.name.c_str());
                    ttype = GGML_TYPE_F16;
                } else {
                    ttype = pick_target_type(ts, model_loader, tensor_type_rules,
                                             ceiling_type, rmse_threshold,
                                             local_buf.data(), n_per_row, nrows);
                }
            }
            std::lock_guard<std::mutex> lk(type_write_mutex);
            target_types[i] = ttype;
        }
    };

    for (int t = 0; t < n_threads; t++) workers.emplace_back(worker_fn);
    for (auto& w : workers) w.join();
    LOG_INFO("RMSE sweep: type selection done on %d threads", n_threads);

    // Step 3: build GGUF header (no_alloc — metadata only, no tensor data in RAM).
    size_t meta_mem = 1 * 1024 * 1024 + active.size() * ggml_tensor_overhead();
    ggml_context* meta_ctx = ggml_init({meta_mem, nullptr, true});
    if (!meta_ctx) {
        if (error) *error = "ggml_init failed for meta context";
        return false;
    }

    gguf_context* gguf_ctx = gguf_init_empty();
    if (!gguf_ctx) {
        ggml_free(meta_ctx);
        if (error) *error = "gguf_init_empty failed";
        return false;
    }

    std::map<ggml_type, int>    type_tensor_count;
    std::map<ggml_type, size_t> type_byte_size;
    size_t total_f16_bytes = 0;

    for (size_t i = 0; i < active.size(); i++) {
        const TensorStorage& ts = active[i];
        ggml_type ttype         = target_types[i];
        int64_t n               = ts.nelements();

        ggml_tensor* t = ggml_new_tensor(meta_ctx, ttype, ts.n_dims, ts.ne);
        if (!t) {
            gguf_free(gguf_ctx);
            ggml_free(meta_ctx);
            if (error) *error = "ggml_new_tensor failed for '" + ts.name + "'";
            return false;
        }
        ggml_set_name(t, ts.name.c_str());
        gguf_add_tensor(gguf_ctx, t);

        if (n > 0) {
            total_f16_bytes += (size_t)n * sizeof(ggml_fp16_t);
            size_t out_bytes;
            if (ttype == GGML_TYPE_F32) {
                out_bytes = (size_t)n * sizeof(float);
            } else if (ttype == GGML_TYPE_F16) {
                out_bytes = (size_t)n * sizeof(ggml_fp16_t);
            } else {
                int64_t n_per_row = ts.ne[0];
                int64_t nrows     = n / n_per_row;
                out_bytes         = (size_t)nrows * ggml_row_size(ttype, n_per_row);
            }
            type_tensor_count[ttype]++;
            type_byte_size[ttype] += out_bytes;
        }
    }

    // Print summary table.
    size_t total_out_bytes = 0;
    for (auto& [t, b] : type_byte_size) total_out_bytes += b;
    LOG_INFO("---- RMSE mixed-quant summary (threshold %.1f%%) ----", rmse_threshold * 100.0f);
    LOG_INFO("  %-12s  %8s  %10s  %6s", "type", "tensors", "size (MB)", "share");
    for (auto& [t, count] : type_tensor_count) {
        size_t mb  = type_byte_size[t] / (1024 * 1024);
        float  pct = total_out_bytes > 0
                         ? (float)type_byte_size[t] * 100.0f / (float)total_out_bytes
                         : 0.0f;
        LOG_INFO("  %-12s  %8d  %10zu  %5.1f%%", ggml_type_name(t), count, mb, pct);
    }
    float ratio = total_f16_bytes > 0 ? (float)total_out_bytes / (float)total_f16_bytes : 1.0f;
    LOG_INFO("  total output: %.0f MB  (%.1fx vs flat f16 / %.0f MB)",
             (float)total_out_bytes / (1024.0f * 1024.0f),
             ratio,
             (float)total_f16_bytes / (1024.0f * 1024.0f));
    LOG_INFO("----------------------------------------------------");

    // Step 4: write GGUF header (only_meta=true).
    LOG_INFO("writing GGUF to %s", output_path.c_str());
    FILE* f = fopen(output_path.c_str(), "wb");
    if (!f) {
        gguf_free(gguf_ctx);
        ggml_free(meta_ctx);
        if (error) *error = "failed to open output file '" + output_path + "'";
        return false;
    }
    if (!gguf_write_to_file_ptr(gguf_ctx, f, true)) {
        fclose(f);
        gguf_free(gguf_ctx);
        ggml_free(meta_ctx);
        if (error) *error = "gguf_write_to_file_ptr (header) failed";
        return false;
    }

    // Step 5: streaming quantize + write tensor data, one tensor at a time.
    // gguf_get_data_offset returns ctx->offset which is only valid when reading.
    // For newly created contexts use gguf_get_meta_size which matches what was just written.
    size_t data_start = gguf_get_meta_size(gguf_ctx);
    int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);

    bool write_ok = true;
    std::vector<uint8_t> quant_buf;

    for (int64_t gi = 0; gi < n_tensors && write_ok; gi++) {
        const TensorStorage& ts = active[(size_t)gi];
        ggml_type ttype         = target_types[(size_t)gi];
        int64_t n               = ts.nelements();

        // Seek to the exact offset GGUF expects for this tensor.
        // Gaps are filled with zeros by the OS; no manual padding needed.
        size_t expected = data_start + gguf_get_tensor_offset(gguf_ctx, gi);
        if (fseeko(f, (off_t)expected, SEEK_SET) != 0) {
            if (error) *error = "fseeko failed for tensor '" + ts.name + "'";
            write_ok = false;
            break;
        }

        if (n == 0) continue;

        if (!model_loader.load_tensor_f32(ts, f32_buf)) {
            if (error) *error = "failed to load tensor '" + ts.name + "' in write pass";
            write_ok = false;
            break;
        }

        const float* data = f32_buf.data();
        int64_t n_per_row = ts.ne[0];
        int64_t nrows     = n / n_per_row;

        size_t out_bytes;
        const void* write_ptr;

        if (ttype == GGML_TYPE_F32) {
            out_bytes = (size_t)n * sizeof(float);
            write_ptr = data;
        } else if (ttype == GGML_TYPE_F16) {
            out_bytes = (size_t)n * sizeof(ggml_fp16_t);
            quant_buf.resize(out_bytes);
            ggml_fp32_to_fp16_row(data, (ggml_fp16_t*)quant_buf.data(), n);
            write_ptr = quant_buf.data();
        } else {
            out_bytes = (size_t)nrows * ggml_row_size(ttype, n_per_row);
            quant_buf.resize(out_bytes);
            std::vector<float> imatrix(n_per_row, 1.0f);
            ggml_quantize_chunk(ttype, data, quant_buf.data(), 0, nrows, n_per_row, imatrix.data());
            write_ptr = quant_buf.data();
        }

        if (fwrite(write_ptr, 1, out_bytes, f) != out_bytes) {
            if (error) *error = "fwrite failed for tensor '" + ts.name + "'";
            write_ok = false;
            break;
        }
    }

    fclose(f);
    gguf_free(gguf_ctx);
    ggml_free(meta_ctx);
    return write_ok;
}

// ─── Public entry point ───────────────────────────────────────────────────────

bool convert(const sd_ctx_params_t* params,
             const char* output_path,
             bool convert_name,
             float rmse_threshold) {
    ModelLoader model_loader;

    if (strlen(SAFE_STR(params->model_path)) > 0) {
        if (!model_loader.init_from_file(params->model_path)) {
            LOG_ERROR("init model loader from file failed: '%s'", params->model_path);
            return false;
        }
    }

    if (strlen(SAFE_STR(params->diffusion_model_path)) > 0) {
        if (!model_loader.init_from_file(params->diffusion_model_path, "model.diffusion_model.")) {
            LOG_ERROR("init model loader from file failed: '%s'", params->diffusion_model_path);
            return false;
        }
    }

    bool is_unet = sd_version_is_unet(model_loader.get_sd_version());

    if (strlen(SAFE_STR(params->clip_l_path)) > 0) {
        std::string prefix = is_unet ? "cond_stage_model.transformer." : "text_encoders.clip_l.transformer.";
        if (!model_loader.init_from_file(params->clip_l_path, prefix)) {
            LOG_ERROR("init model loader from file failed: '%s'", params->clip_l_path);
            return false;
        }
    }

    if (strlen(SAFE_STR(params->clip_g_path)) > 0) {
        std::string prefix = is_unet ? "cond_stage_model.1.transformer." : "text_encoders.clip_g.transformer.";
        if (!model_loader.init_from_file(params->clip_g_path, prefix)) {
            LOG_ERROR("init model loader from file failed: '%s'", params->clip_g_path);
            return false;
        }
    }

    if (strlen(SAFE_STR(params->t5xxl_path)) > 0) {
        if (!model_loader.init_from_file(params->t5xxl_path, "text_encoders.t5xxl.transformer.")) {
            LOG_ERROR("init model loader from file failed: '%s'", params->t5xxl_path);
            return false;
        }
    }

    if (strlen(SAFE_STR(params->llm_path)) > 0) {
        if (!model_loader.init_from_file(params->llm_path, "text_encoders.llm.")) {
            LOG_ERROR("init model loader from file failed: '%s'", params->llm_path);
            return false;
        }
    }

    if (strlen(SAFE_STR(params->llm_vision_path)) > 0) {
        if (!model_loader.init_from_file(params->llm_vision_path, "text_encoders.llm.visual.")) {
            LOG_ERROR("init model loader from file failed: '%s'", params->llm_vision_path);
            return false;
        }
    }

    if (strlen(SAFE_STR(params->vae_path)) > 0) {
        if (!model_loader.init_from_file(params->vae_path, "vae.")) {
            LOG_ERROR("init model loader from file failed: '%s'", params->vae_path);
            return false;
        }
    }

    if (convert_name) {
        model_loader.convert_tensors_name();
    }

    // When --type is not given and RMSE mode is active, default ceiling to f16.
    ggml_type ceiling_type = (params->wtype != SD_TYPE_COUNT)
                                 ? (ggml_type)params->wtype
                                 : (rmse_threshold > 0.0f ? GGML_TYPE_F16 : GGML_TYPE_COUNT);

    bool output_is_safetensors = ends_with(output_path, ".safetensors");
    TensorTypeRules type_rules = parse_tensor_type_rules(SAFE_STR(params->tensor_type_rules));

    auto backend = ggml_backend_cpu_init();
    bool success = false;
    std::string error;

    if (rmse_threshold > 0.0f) {
        // ── RMSE path (streaming, low RAM) ────────────────────────────────────
        // Two-pass: type sweep then quantize+write, one tensor in RAM at a time.
        ggml_backend_free(backend);
        if (output_is_safetensors) {
            LOG_ERROR("RMSE streaming mode does not support safetensors output; use .gguf");
            return false;
        }
        success = convert_rmse_streaming(model_loader, ceiling_type, type_rules,
                                          rmse_threshold, output_path, &error);
    } else {
        // ── Normal path ────────────────────────────────────────────────────────
        size_t mem_size = 1 * 1024 * 1024;
        mem_size += model_loader.get_tensor_storage_map().size() * ggml_tensor_overhead();
        mem_size += model_loader.get_params_mem_size(backend, ceiling_type);

        ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});
        if (!ggml_ctx) {
            LOG_ERROR("ggml_init failed for converter");
            ggml_backend_free(backend);
            return false;
        }

        std::vector<TensorWriteInfo> tensors;
        success = load_tensors_for_export(model_loader, ggml_ctx, ceiling_type, type_rules, tensors);
        ggml_backend_free(backend);

        if (success) {
            if (output_is_safetensors) {
                success = write_safetensors_file(output_path, tensors, &error);
            } else {
                success = write_gguf_file(output_path, tensors, &error);
            }
        }

        ggml_free(ggml_ctx);
    }

    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }

    return success;
}
