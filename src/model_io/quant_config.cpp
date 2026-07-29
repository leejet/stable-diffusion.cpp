#include "model_io/quant_config.h"

#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include "core/util.h"
#include "json.hpp"

// Reads `count` raw bytes at `offset`. Sidecars are small (a few MB in total
// for a whole model), so a per-tensor read is cheap enough and avoids holding
// the whole file.
static bool read_raw(std::ifstream& file, uint64_t offset, size_t count, void* dst) {
    file.seekg((std::streamoff)offset, std::ios::beg);
    if (!file) {
        return false;
    }
    file.read((char*)dst, (std::streamsize)count);
    return (bool)file;
}

static bool read_f32_tensor(std::ifstream& file, const TensorStorage& ts, std::vector<float>& out) {
    if (ts.type != GGML_TYPE_F32) {
        return false;
    }
    out.resize((size_t)ts.nelements());
    return read_raw(file, ts.offset, out.size() * sizeof(float), out.data());
}

static std::string strip_suffix(const std::string& name, const std::string& suffix) {
    if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return name.substr(0, name.size() - suffix.size());
    }
    return {};
}

static int json_int(const nlohmann::json& obj, const char* key, int fallback) {
    if (obj.is_object()) {
        auto it = obj.find(key);
        if (it != obj.end() && it->is_number_integer()) {
            return it->get<int>();
        }
    }
    return fallback;
}

static std::string json_str(const nlohmann::json& obj, const char* key, const std::string& fallback) {
    if (obj.is_object()) {
        auto it = obj.find(key);
        if (it != obj.end() && it->is_string()) {
            return it->get<std::string>();
        }
    }
    return fallback;
}

struct LayerFormat {
    SDQuantPack pack      = SD_QUANT_PACK_NONE;
    int convrot_groupsize = 0;
};

// Parses ComfyUI's `_quantization_metadata` into per-layer decisions.
static bool parse_comfy_quant_metadata(const std::string& raw,
                                       std::unordered_map<std::string, LayerFormat>& out) {
    nlohmann::json root = nlohmann::json::parse(raw, nullptr, false);
    if (root.is_discarded() || !root.is_object()) {
        LOG_WARN("quantization metadata is not valid JSON; ignoring it");
        return false;
    }
    auto layers_it = root.find("layers");
    if (layers_it == root.end() || !layers_it->is_object()) {
        return false;
    }
    const nlohmann::json& defaults = root.value("params", nlohmann::json::object());

    for (const auto& item : layers_it->items()) {
        const nlohmann::json& conf = item.value();
        if (!conf.is_object()) {
            continue;
        }
        const nlohmann::json& params = conf.value("params", nlohmann::json::object());
        const std::string format     = json_str(conf, "format", "");

        LayerFormat lf;
        if (format == "convrot_w4a4") {
            // The stored payload width follows linear_dtype, which defaults to
            // int4; convrot_w4a4 checkpoints may still carry int8 weights.
            std::string linear_dtype = json_str(conf, "linear_dtype", json_str(params, "linear_dtype", "int4"));
            lf.pack                  = (linear_dtype == "int8") ? SD_QUANT_PACK_INT8 : SD_QUANT_PACK_INT4;
            lf.convrot_groupsize     = json_int(conf, "convrot_groupsize",
                                                json_int(params, "convrot_groupsize",
                                                         json_int(defaults, "convrot_groupsize", 256)));
        } else if (format == "int8_tensorwise") {
            lf.pack = SD_QUANT_PACK_INT8;
            // Rotation is optional for this format and off unless flagged.
            bool convrot = conf.value("convrot", params.value("convrot", false));
            if (convrot) {
                lf.convrot_groupsize = json_int(conf, "convrot_groupsize",
                                                json_int(params, "convrot_groupsize",
                                                         json_int(defaults, "convrot_groupsize", 256)));
            }
        } else {
            LOG_WARN("unsupported quantization format '%s' for layer '%s'; leaving it unclaimed",
                     format.c_str(), item.key().c_str());
            continue;
        }
        out[item.key()] = lf;
    }
    return !out.empty();
}

bool sd_apply_quant_metadata(const std::string& file_path,
                             const std::map<std::string, std::string>& metadata,
                             std::vector<TensorStorage>& tensor_storages) {
    // Index by name so sidecars can be located without a quadratic scan.
    std::unordered_map<std::string, size_t> by_name;
    by_name.reserve(tensor_storages.size() * 2);
    for (size_t i = 0; i < tensor_storages.size(); i++) {
        by_name[tensor_storages[i].name] = i;
    }

    std::unordered_map<std::string, LayerFormat> layer_formats;
    auto meta_it = metadata.find("_quantization_metadata");
    if (meta_it != metadata.end()) {
        parse_comfy_quant_metadata(meta_it->second, layer_formats);
    }

    // Discover bitsandbytes NF4 weights, which describe themselves through
    // sidecars rather than through file-level metadata.
    std::unordered_map<std::string, std::string> nf4_state_blob;  // weight name -> quant_state tensor
    for (const auto& ts : tensor_storages) {
        std::string base = strip_suffix(ts.name, ".quant_state.bitsandbytes__nf4");
        if (!base.empty()) {
            nf4_state_blob[base] = ts.name;
        }
    }

    if (layer_formats.empty() && nf4_state_blob.empty()) {
        return true;
    }

    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        LOG_ERROR("failed to reopen '%s' to read quantization sidecars", file_path.c_str());
        return false;
    }

    std::unordered_set<std::string> consumed;
    size_t claimed_int4 = 0, claimed_int8 = 0, claimed_nf4 = 0, rotated = 0;

    // ---- ComfyUI convrot / int8_tensorwise -------------------------------
    for (const auto& kv : layer_formats) {
        const std::string weight_name = kv.first + ".weight";
        const std::string scale_name  = kv.first + ".weight_scale";

        auto w_it = by_name.find(weight_name);
        if (w_it == by_name.end()) {
            continue;
        }
        TensorStorage& w = tensor_storages[w_it->second];
        if (w.type != GGML_TYPE_I8) {
            LOG_WARN("layer '%s' is marked quantized but its weight is %s; leaving it alone",
                     kv.first.c_str(), ggml_type_name(w.type));
            continue;
        }

        auto s_it = by_name.find(scale_name);
        if (s_it == by_name.end()) {
            LOG_ERROR("layer '%s' is marked %s but has no weight_scale", kv.first.c_str(),
                      sd_quant_pack_name(kv.second.pack));
            return false;
        }

        auto quant               = std::make_shared<SDQuantParams>();
        quant->pack              = kv.second.pack;
        quant->convrot_groupsize = kv.second.convrot_groupsize;
        if (!read_f32_tensor(file, tensor_storages[s_it->second], quant->scales)) {
            LOG_ERROR("failed to read weight_scale for layer '%s'", kv.first.c_str());
            return false;
        }

        // ne is already reversed into ggml order, so ne[0] is in_features (halved
        // on disk for int4) and ne[1] is out_features.
        if (quant->pack == SD_QUANT_PACK_INT4) {
            w.ne[0] *= 2;
        }
        const int64_t out_features = w.nelements() / w.ne[0];

        if ((int64_t)quant->scales.size() == 1) {
            // Tensor-wise scale: broadcast so the repack stays uniform.
            quant->scales.assign((size_t)out_features, quant->scales[0]);
        } else if ((int64_t)quant->scales.size() != out_features) {
            LOG_ERROR("layer '%s': weight_scale has %zu entries but %" PRId64 " output channels",
                      kv.first.c_str(), quant->scales.size(), out_features);
            return false;
        }

        if (w.ne[0] % 32 != 0) {
            LOG_ERROR("layer '%s': in_features %" PRId64 " is not a multiple of 32; cannot repack",
                      kv.first.c_str(), w.ne[0]);
            return false;
        }
        if (quant->convrot_groupsize > 0 && w.ne[0] % quant->convrot_groupsize != 0) {
            // ConvRot only rotates layers whose input divides the group size; a
            // marker that disagrees with the shape means the weight is unrotated.
            quant->convrot_groupsize = 0;
        }

        w.type  = sd_quant_pack_target_type(quant->pack);
        w.quant = quant;
        consumed.insert(scale_name);

        if (quant->pack == SD_QUANT_PACK_INT4) {
            claimed_int4++;
        } else {
            claimed_int8++;
        }
        if (quant->convrot_groupsize > 0) {
            rotated++;
        }
    }

    // ---- bitsandbytes NF4 ------------------------------------------------
    for (const auto& kv : nf4_state_blob) {
        const std::string& weight_name = kv.first;
        auto w_it                      = by_name.find(weight_name);
        auto blob_it                   = by_name.find(kv.second);
        if (w_it == by_name.end() || blob_it == by_name.end()) {
            continue;
        }
        TensorStorage& w = tensor_storages[w_it->second];
        if (w.type != GGML_TYPE_I8) {
            continue;
        }

        // The quant_state blob is UTF-8 JSON stored as bytes; it carries the
        // logical shape, which the packed weight tensor itself has lost.
        const TensorStorage& blob = tensor_storages[blob_it->second];
        std::string blob_text((size_t)blob.nelements(), '\0');
        if (!read_raw(file, blob.offset, blob_text.size(), &blob_text[0])) {
            LOG_ERROR("failed to read NF4 quant_state for '%s'", weight_name.c_str());
            return false;
        }
        nlohmann::json state = nlohmann::json::parse(blob_text, nullptr, false);
        if (state.is_discarded()) {
            LOG_ERROR("NF4 quant_state for '%s' is not valid JSON", weight_name.c_str());
            return false;
        }
        if (json_str(state, "quant_type", "nf4") != "nf4") {
            LOG_WARN("'%s' uses quant_type '%s'; only nf4 is supported", weight_name.c_str(),
                     json_str(state, "quant_type", "").c_str());
            continue;
        }

        auto quant        = std::make_shared<SDQuantParams>();
        quant->pack       = SD_QUANT_PACK_NF4;
        quant->block_size = json_int(state, "blocksize", 64);

        // Restore the logical 2-D shape in ggml order (ne[0] = in_features).
        auto shape_it = state.find("shape");
        if (shape_it != state.end() && shape_it->is_array() && shape_it->size() >= 2) {
            std::vector<int64_t> shape;
            for (const auto& d : *shape_it) {
                shape.push_back(d.get<int64_t>());
            }
            for (int i = 0; i < SD_MAX_DIMS; i++) {
                w.ne[i] = 1;
            }
            w.n_dims = (int)shape.size();
            for (size_t i = 0; i < shape.size(); i++) {
                w.ne[i] = shape[shape.size() - 1 - i];
            }
        } else {
            LOG_ERROR("NF4 quant_state for '%s' has no shape", weight_name.c_str());
            return false;
        }

        auto absmax_it = by_name.find(weight_name + ".absmax");
        if (absmax_it == by_name.end()) {
            LOG_ERROR("NF4 weight '%s' has no absmax", weight_name.c_str());
            return false;
        }
        const TensorStorage& absmax_ts = tensor_storages[absmax_it->second];

        if (absmax_ts.type == GGML_TYPE_F32) {
            if (!read_f32_tensor(file, absmax_ts, quant->scales)) {
                LOG_ERROR("failed to read NF4 absmax for '%s'", weight_name.c_str());
                return false;
            }
        } else {
            // Double-quantized absmax: itself int8-coded against nested_quant_map
            // with a nested absmax and a mean offset.
            auto n_absmax_it = by_name.find(weight_name + ".nested_absmax");
            auto n_map_it    = by_name.find(weight_name + ".nested_quant_map");
            if (n_absmax_it == by_name.end() || n_map_it == by_name.end()) {
                LOG_ERROR("NF4 weight '%s' has a double-quantized absmax but no nested state",
                          weight_name.c_str());
                return false;
            }
            std::vector<uint8_t> codes((size_t)absmax_ts.nelements());
            std::vector<float> nested_absmax, nested_map;
            if (!read_raw(file, absmax_ts.offset, codes.size(), codes.data()) ||
                !read_f32_tensor(file, tensor_storages[n_absmax_it->second], nested_absmax) ||
                !read_f32_tensor(file, tensor_storages[n_map_it->second], nested_map)) {
                LOG_ERROR("failed to read nested NF4 state for '%s'", weight_name.c_str());
                return false;
            }
            const int nested_blocksize = json_int(state, "nested_blocksize", 256);
            const float offset         = state.value("nested_offset", 0.0f);
            quant->scales.resize(codes.size());
            for (size_t i = 0; i < codes.size(); i++) {
                const float na   = nested_absmax.empty() ? 1.0f : nested_absmax[i / (size_t)nested_blocksize];
                quant->scales[i] = nested_map[codes[i]] * na + offset;
            }
            consumed.insert(weight_name + ".nested_absmax");
            consumed.insert(weight_name + ".nested_quant_map");
        }

        auto map_it = by_name.find(weight_name + ".quant_map");
        if (map_it != by_name.end()) {
            read_f32_tensor(file, tensor_storages[map_it->second], quant->codebook);
            consumed.insert(weight_name + ".quant_map");
        }
        if (quant->codebook.size() != 16) {
            quant->codebook.assign(sd_nf4_default_codebook(), sd_nf4_default_codebook() + 16);
        }

        w.type  = sd_quant_pack_target_type(quant->pack);
        w.quant = quant;
        consumed.insert(weight_name + ".absmax");
        consumed.insert(kv.second);
        consumed.insert(weight_name + ".bitsandbytes__nf4");
        claimed_nf4++;
    }

    // Drop consumed sidecars plus any byte tensor nothing claimed (metadata blobs,
    // leftover quant state), preserving the pre-existing behaviour of not
    // surfacing those to the model.
    const size_t before = tensor_storages.size();
    tensor_storages.erase(
        std::remove_if(tensor_storages.begin(), tensor_storages.end(),
                       [&](const TensorStorage& ts) {
                           if (consumed.count(ts.name) != 0) {
                               return true;
                           }
                           return ts.type == GGML_TYPE_I8 && !ts.is_packed_quant();
                       }),
        tensor_storages.end());

    if (claimed_int4 + claimed_int8 + claimed_nf4 > 0) {
        LOG_INFO("quantized weights: %zu int4, %zu int8, %zu nf4 (%zu convrot-rotated), %zu sidecars consumed",
                 claimed_int4, claimed_int8, claimed_nf4, rotated, before - tensor_storages.size());
    }
    return true;
}
