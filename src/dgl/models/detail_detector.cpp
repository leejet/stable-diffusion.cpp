#include "dgl/models/detail_detector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "core/image_geometry.h"
#include "core/util.h"
#include "json.hpp"
#include "model/detector/yolov8.h"
#include "model_loader.h"
#include "model_manager.h"

namespace sd::dgl::models {

    struct LetterboxInput {
        sd::Tensor<float> tensor;
        float scale = 1.f;
        int pad_x   = 0;
        int pad_y   = 0;
    };

    static std::string trim_copy(const std::string& value) {
        size_t first = value.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return "";
        }
        size_t last = value.find_last_not_of(" \t\r\n");
        return value.substr(first, last - first + 1);
    }

    static bool parse_int(const std::string& value, int* result) {
        try {
            size_t used = 0;
            int parsed  = std::stoi(trim_copy(value), &used);
            if (used != trim_copy(value).size()) {
                return false;
            }
            *result = parsed;
            return true;
        } catch (...) {
            return false;
        }
    }

    static float image_channel(const sd::Image& image, int x, int y, int channel) {
        if (image.data == nullptr || image.width == 0 || image.height == 0 || image.channel == 0) {
            return 0.f;
        }
        x                  = std::clamp(x, 0, static_cast<int>(image.width) - 1);
        y                  = std::clamp(y, 0, static_cast<int>(image.height) - 1);
        int source_channel = image.channel == 1 ? 0 : std::min(channel, static_cast<int>(image.channel) - 1);
        size_t index       = (static_cast<size_t>(y) * image.width + x) * image.channel + source_channel;
        return image.data[index] / 255.f;
    }

    static float bilinear_channel(const sd::Image& image, float x, float y, int channel) {
        int x0   = static_cast<int>(std::floor(x));
        int y0   = static_cast<int>(std::floor(y));
        int x1   = x0 + 1;
        int y1   = y0 + 1;
        float wx = x - x0;
        float wy = y - y0;
        float a  = image_channel(image, x0, y0, channel) * (1.f - wx) + image_channel(image, x1, y0, channel) * wx;
        float b  = image_channel(image, x0, y1, channel) * (1.f - wx) + image_channel(image, x1, y1, channel) * wx;
        return a * (1.f - wy) + b * wy;
    }

    static LetterboxInput make_letterbox_input(sd::Image image, int input_size) {
        LetterboxInput result;
        result.tensor      = sd::full<float>({input_size, input_size, 3, 1}, 114.f / 255.f);
        result.scale       = std::min(static_cast<float>(input_size) / image.width,
                                      static_cast<float>(input_size) / image.height);
        int resized_width  = std::max(1, static_cast<int>(std::round(image.width * result.scale)));
        int resized_height = std::max(1, static_cast<int>(std::round(image.height * result.scale)));
        result.pad_x       = (input_size - resized_width) / 2;
        result.pad_y       = (input_size - resized_height) / 2;

        for (int y = 0; y < resized_height; ++y) {
            float source_y = (y + 0.5f) / result.scale - 0.5f;
            for (int x = 0; x < resized_width; ++x) {
                float source_x = (x + 0.5f) / result.scale - 0.5f;
                for (int c = 0; c < 3; ++c) {
                    result.tensor.index(x + result.pad_x, y + result.pad_y, c, 0) = bilinear_channel(image, source_x, source_y, c);
                }
            }
        }
        return result;
    }

    static float logistic(float x) {
        if (x >= 0.f) {
            float z = std::exp(-x);
            return 1.f / (1.f + z);
        }
        float z = std::exp(x);
        return z / (1.f + z);
    }

    static float dfl_expectation(const float* values, int64_t anchor_count, int channel_offset, int reg_max, int64_t anchor) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < reg_max; ++i) {
            maximum = std::max(maximum, values[anchor + anchor_count * (channel_offset + i)]);
        }
        float denominator = 0.f;
        float numerator   = 0.f;
        for (int i = 0; i < reg_max; ++i) {
            float probability = std::exp(values[anchor + anchor_count * (channel_offset + i)] - maximum);
            denominator += probability;
            numerator += probability * i;
        }
        return denominator > 0.f ? numerator / denominator : 0.f;
    }

    static float box_iou(const DetailDetection& a, const DetailDetection& b) {
        float x1           = std::max(a.x1, b.x1);
        float y1           = std::max(a.y1, b.y1);
        float x2           = std::min(a.x2, b.x2);
        float y2           = std::min(a.y2, b.y2);
        float intersection = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
        float area_a       = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
        float area_b       = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
        float union_area   = area_a + area_b - intersection;
        return union_area > 0.f ? intersection / union_area : 0.f;
    }

    static std::vector<DetailDetection> decode_detections(const sd::Tensor<float>& raw,
                                                          const YOLOv8Config& config,
                                                          const LetterboxInput& letterbox,
                                                          sd::Image image,
                                                          const DetectionOptions& params) {
        std::vector<DetailDetection> candidates;
        if (raw.empty() || raw.dim() < 2) {
            return candidates;
        }
        int64_t anchor_count = raw.shape()[0];
        int64_t channels     = raw.shape()[1];
        if (channels != config.reg_max * 4 + config.num_classes) {
            LOG_ERROR("unexpected YOLOv8 output channels: %lld", static_cast<long long>(channels));
            return candidates;
        }

        const std::array<int, 3> strides = {8, 16, 32};
        std::array<int64_t, 4> offsets   = {0, 0, 0, 0};
        for (int i = 0; i < 3; ++i) {
            int grid       = params.input_size / strides[i];
            offsets[i + 1] = offsets[i] + static_cast<int64_t>(grid) * grid;
        }
        if (offsets[3] != anchor_count) {
            LOG_ERROR("unexpected YOLOv8 anchor count: %lld (expected %lld)",
                      static_cast<long long>(anchor_count),
                      static_cast<long long>(offsets[3]));
            return candidates;
        }

        const float* values = raw.data();
        for (int scale_index = 0; scale_index < 3; ++scale_index) {
            int stride = strides[scale_index];
            int grid   = params.input_size / stride;
            for (int64_t anchor = offsets[scale_index]; anchor < offsets[scale_index + 1]; ++anchor) {
                int64_t local = anchor - offsets[scale_index];
                int grid_x    = static_cast<int>(local % grid);
                int grid_y    = static_cast<int>(local / grid);

                float confidence = 0.f;
                int class_id     = 0;
                for (int c = 0; c < config.num_classes; ++c) {
                    float score = logistic(values[anchor + anchor_count * (config.reg_max * 4 + c)]);
                    if (score > confidence) {
                        confidence = score;
                        class_id   = c;
                    }
                }
                if (confidence < params.confidence) {
                    continue;
                }

                float left     = dfl_expectation(values, anchor_count, 0 * config.reg_max, config.reg_max, anchor);
                float top      = dfl_expectation(values, anchor_count, 1 * config.reg_max, config.reg_max, anchor);
                float right    = dfl_expectation(values, anchor_count, 2 * config.reg_max, config.reg_max, anchor);
                float bottom   = dfl_expectation(values, anchor_count, 3 * config.reg_max, config.reg_max, anchor);
                float center_x = (grid_x + 0.5f) * stride;
                float center_y = (grid_y + 0.5f) * stride;

                DetailDetection detection;
                detection.x1         = (center_x - left * stride - letterbox.pad_x) / letterbox.scale;
                detection.y1         = (center_y - top * stride - letterbox.pad_y) / letterbox.scale;
                detection.x2         = (center_x + right * stride - letterbox.pad_x) / letterbox.scale;
                detection.y2         = (center_y + bottom * stride - letterbox.pad_y) / letterbox.scale;
                detection.x1         = std::clamp(detection.x1, 0.f, static_cast<float>(image.width));
                detection.y1         = std::clamp(detection.y1, 0.f, static_cast<float>(image.height));
                detection.x2         = std::clamp(detection.x2, 0.f, static_cast<float>(image.width));
                detection.y2         = std::clamp(detection.y2, 0.f, static_cast<float>(image.height));
                detection.confidence = confidence;
                detection.class_id   = class_id;
                if (detection.x2 > detection.x1 && detection.y2 > detection.y1) {
                    candidates.push_back(detection);
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            return a.confidence > b.confidence;
        });
        std::vector<DetailDetection> selected;
        for (const auto& candidate : candidates) {
            bool suppressed = false;
            for (const auto& kept : selected) {
                if (candidate.class_id == kept.class_id && box_iou(candidate, kept) > params.nms_threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                selected.push_back(candidate);
                if (static_cast<int>(selected.size()) >= params.max_detections) {
                    break;
                }
            }
        }
        return selected;
    }

    static std::vector<std::string> parse_class_names(const std::string& value, int class_count) {
        std::vector<std::string> names(static_cast<size_t>(class_count));
        try {
            nlohmann::json parsed = nlohmann::json::parse(value);
            if (parsed.is_array()) {
                for (size_t i = 0; i < names.size() && i < parsed.size(); ++i) {
                    if (parsed[i].is_string()) {
                        names[i] = parsed[i].get<std::string>();
                    }
                }
            } else if (parsed.is_object()) {
                for (const auto& item : parsed.items()) {
                    int class_id = 0;
                    if (parse_int(item.key(), &class_id) && class_id >= 0 && class_id < class_count &&
                        item.value().is_string()) {
                        names[static_cast<size_t>(class_id)] = item.value().get<std::string>();
                    }
                }
            }
        } catch (const std::exception&) {
        }
        return names;
    }

    DetailDetector::DetailDetector(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends)
        : Model(std::move(manager)), backend_manager(std::move(backends)) {
        params_placement_ = [this](const std::string& name) {
            std::string error;
            if (!backend_manager.init(params_.backend.c_str(), name.c_str(), nullptr, &error))
                throw std::runtime_error(error);
            params_.params_backend = name;
            return true;
        };
    }

    DetailDetector::~DetailDetector() {
        if (detector) {
            detector->runner_end();
            set_weight_adapter(nullptr);
        }
        release_weights();
    }

    bool DetailDetector::init(DetailDetectorParams params) {
        params_ = std::move(params);
        ModelLoader loader;
        if (!loader.init_from_file(params_.path)) {
            LOG_ERROR("failed to load ADetailer detector: '%s'", params_.path.c_str());
            return false;
        }
        return build(std::move(loader));
    }

    bool DetailDetector::build_variant(const Model& source, ModelLoader loader) {
        params_ = static_cast<const DetailDetector&>(source).params_;
        return build(std::move(loader));
    }

    bool DetailDetector::build(ModelLoader loader) {
        std::string error;
        if (!backend_manager.init(params_.backend.c_str(), params_.params_backend.c_str(), nullptr, &error)) {
            LOG_ERROR("ADetailer backend config failed: %s", error.c_str());
            return false;
        }
        ggml_backend_t backend        = backend_manager.runtime_backend(SDBackendModule::DETECTOR);
        ggml_backend_t params_backend = backend_manager.params_backend(SDBackendModule::DETECTOR);
        if (backend == nullptr || params_backend == nullptr) {
            LOG_ERROR("failed to initialize detector backend");
            return false;
        }

        if (!model_weights->set_loader(std::move(loader)))
            return false;
        const auto& catalog = model_weights->loader();
        detector            = std::make_shared<YOLOv8Runner>(backend, catalog.get_tensor_storage_map(), model_weights);
        if (!detector || !detector->model || !detector->config.valid) {
            LOG_ERROR("unsupported YOLOv8 detection weights: '%s'", params_.path.c_str());
            return false;
        }

        class_names.assign(static_cast<size_t>(detector->config.num_classes), "");
        auto names_item = catalog.get_metadata().find("yolov8.names");
        if (names_item != catalog.get_metadata().end()) {
            class_names = parse_class_names(names_item->second, detector->config.num_classes);
        }

        std::map<std::string, ggml_tensor*> tensors;
        detector->get_param_tensors(tensors);
        if (!model_weights->register_param_tensors(ModelComponent::Detector,
                                                   std::move(tensors),
                                                   backend_manager.params_backend_is_disk(SDBackendModule::DETECTOR)
                                                       ? ModelManager::ResidencyMode::Disk
                                                       : ModelManager::ResidencyMode::ParamBackend,
                                                   backend,
                                                   params_backend) ||
            !model_weights->validate_registered_tensors()) {
            LOG_ERROR("failed to register YOLOv8 detector tensors");
            return false;
        }
        return load_weights_eagerly();
    }

    bool DetailDetector::attach_loras() {
        const auto mode = resolve_lora_mode(lora_mode(), model_weights->loader(), backend_manager, SDBackendModule::DETECTOR);
        return prepare_loras(VERSION_COUNT, backend_manager, SDBackendModule::DETECTOR,
                             n_threads, mode == LORA_APPLY_IMMEDIATELY);
    }

    void DetailDetector::set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) {
        detector->set_weight_adapter(adapter);
    }

    std::vector<DetailDetection> DetailDetector::predict(sd::Image image,
                                                         const DetectionOptions& options) {
        auto params = options;
        if (params.input_size <= 0 || params.input_size > 4096)
            throw std::runtime_error("input_size must be between 1 and 4096");
        params.input_size = int(sd::align_image_dimension(params.input_size, 32));
        if (params.input_size != options.input_size)
            LOG_VERBOSE("detector: aligning input_size from %d to %d", options.input_size, params.input_size);
        Model::ExecutionScope scope(*this);
        scope.require_ready();
        LetterboxInput input  = make_letterbox_input(image, params.input_size);
        int64_t start         = ggml_time_ms();
        sd::Tensor<float> raw = detector->compute(n_threads, input.tensor);
        detector->runner_end();
        if (raw.empty()) {
            LOG_ERROR("YOLOv8 detector inference failed");
            return {};
        }
        auto detections = decode_detections(raw, detector->config, input, image, params);
        LOG_INFO("ADetailer detected %zu object(s), taking %.2fs",
                 detections.size(),
                 (ggml_time_ms() - start) / 1000.f);
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& detection = detections[i];
            std::string object    = "class_" + std::to_string(detection.class_id);
            if (detection.class_id >= 0 && static_cast<size_t>(detection.class_id) < class_names.size() &&
                !class_names[static_cast<size_t>(detection.class_id)].empty()) {
                object = class_names[static_cast<size_t>(detection.class_id)];
            }
            LOG_INFO("ADetailer detection %zu: object=%s, class_id=%d, confidence=%.3f, bbox=[x1=%.1f, y1=%.1f, x2=%.1f, y2=%.1f]",
                     i + 1,
                     object.c_str(),
                     detection.class_id,
                     detection.confidence,
                     detection.x1,
                     detection.y1,
                     detection.x2,
                     detection.y2);
        }
        return detections;
    }

}  // namespace sd::dgl::models
