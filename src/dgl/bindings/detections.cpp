#include "dgl/bindings/detections.h"

#include <cmath>

#include "dgl/bindings/conversion.h"
#include "dgl/bindings/regions.h"
#include "dgl/models/detail_detector.h"
#include "dgl/ops/detailer.h"

namespace sd::dgl {
    struct DetectionView {
        const sd::Tensor<float>& xyxy;
        const sd::Tensor<float>& scores;
        const sd::Tensor<int64_t>& class_ids;
    };

    OutputSignature detections_signature() {
        return {{Kind::Dict}, {{"xyxy", {Kind::Tensor}}, {"scores", {Kind::Tensor}}, {"class_ids", {Kind::Tensor}}}};
    }

    static DetectionView read_detections(const Value& value) {
        if (value.type().kind != Kind::Dict)
            throw std::runtime_error("detections require a dict with xyxy, scores and class_ids");
        const auto& fields = value.fields;
        auto tensor        = [&](const char* name, sd_tensor_type_t dtype) -> const sd_tensor_t& {
            const auto found = fields.find(name);
            if (found == fields.end())
                throw std::runtime_error(std::string("detections are missing field '") + name + "'");
            if (found->second.type().kind != Kind::Tensor)
                throw std::runtime_error(std::string("detections.") + name + " must be a tensor");
            const auto& result = *found->second.as<sd_tensor_t>();
            if (result.type != dtype)
                throw std::runtime_error(std::string("detections.") + name + (dtype == SD_TENSOR_I64 ? " requires I64" : " requires F32"));
            return result;
        };
        const auto& xyxy      = tensor("xyxy", SD_TENSOR_F32);
        const auto& scores    = tensor("scores", SD_TENSOR_F32);
        const auto& class_ids = tensor("class_ids", SD_TENSOR_I64);
        if (xyxy.shape().size() != 2 || xyxy.shape()[0] != 4 || xyxy.shape()[1] > 4096)
            throw std::runtime_error("detections.xyxy requires F32 [4,N] with at most 4096 boxes");
        if (scores.shape() != std::vector<int64_t>{xyxy.shape()[1]} || class_ids.shape() != scores.shape())
            throw std::runtime_error("detections.scores and class_ids require shape [N] matching xyxy");
        static const sd::Tensor<float> empty_xyxy({4, 0}), empty_scores({0});
        static const sd::Tensor<int64_t> empty_ids({0});
        const bool populated = xyxy.shape()[1] != 0;
        DetectionView result{populated ? xyxy.as<float>() : empty_xyxy,
                             populated ? scores.as<float>() : empty_scores,
                             populated ? class_ids.as<int64_t>() : empty_ids};
        for (int64_t i = 0; i < xyxy.shape()[1]; ++i) {
            for (int j = 0; j < 4; ++j)
                if (!std::isfinite(result.xyxy[i * 4 + j]))
                    throw std::runtime_error("detection " + std::to_string(i) + " coordinates must be finite");
            if (!std::isfinite(result.scores[i]) || result.scores[i] < 0 || result.scores[i] > 1)
                throw std::runtime_error("detection " + std::to_string(i) + " score must be finite and in [0, 1]");
            if (result.class_ids[i] < 0)
                throw std::runtime_error("detection " + std::to_string(i) + " class_id must be nonnegative");
        }
        return result;
    }

    static Value detection_value(std::vector<float> xyxy, std::vector<float> scores, std::vector<int64_t> class_ids) {
        const auto count = int64_t(scores.size());
        return Value::dict({
            {"xyxy", Value::object(make_tensor(SD_TENSOR_F32, std::vector<int64_t>{4, count}, xyxy.data(), xyxy.size() * sizeof(float)))},
            {"scores", Value::object(make_tensor(SD_TENSOR_F32, std::vector<int64_t>{count}, scores.data(), scores.size() * sizeof(float)))},
            {"class_ids", Value::object(make_tensor(SD_TENSOR_I64, std::vector<int64_t>{count}, class_ids.data(), class_ids.size() * sizeof(int64_t)))},
        });
    }

    Value detect_objects(Values& args, RunState&) {
        const auto image = image_view(args.at("image"));
        models::DetectionOptions options;
        options.input_size     = int(get_int(args, "input_size"));
        options.confidence     = float(get_number(args, "confidence"));
        options.nms_threshold  = float(get_number(args, "nms"));
        options.max_detections = int(get_int(args, "max_detections"));
        const auto found       = object<models::DetailDetector>(args, "detector")->predict(image, options);
        std::vector<float> xyxy, scores;
        std::vector<int64_t> class_ids;
        xyxy.reserve(found.size() * 4);
        scores.reserve(found.size());
        class_ids.reserve(found.size());
        for (const auto& d : found) {
            xyxy.insert(xyxy.end(), {d.x1, d.y1, d.x2, d.y2});
            scores.push_back(d.confidence);
            class_ids.push_back(d.class_id);
        }
        auto result = detection_value(std::move(xyxy), std::move(scores), std::move(class_ids));
        read_detections(result);
        return result;
    }

    static std::vector<models::DetailDetection> detection_rows(const DetectionView& view, const sd::Image& image) {
        std::vector<models::DetailDetection> rows;
        for (int64_t i = 0; i < view.scores.numel(); ++i) {
            const auto* d = view.xyxy.data() + i * 4;
            if (d[0] < 0 || d[1] < 0 || d[2] < d[0] || d[3] < d[1] || d[2] > image.width || d[3] > image.height)
                throw std::runtime_error("detection " + std::to_string(i) + " lies outside the image or has reversed corners");
            rows.push_back({d[0], d[1], d[2], d[3], view.scores[i]});
        }
        return rows;
    }

    Value filter_detections(Values& args, RunState&) {
        using ops::DetectionSort;
        static const std::map<std::string, DetectionSort> orders{
            {"none", DetectionSort::None}, {"left_to_right", DetectionSort::LeftToRight}, {"center_to_edge", DetectionSort::CenterToEdge}, {"area", DetectionSort::Area}};
        ops::DetectionFilter filter;
        filter.top_k               = int(get_int(args, "top_k"));
        filter.top_k_by_confidence = get_string(args, "top_k_by") == "confidence";
        const auto min_ratio = get_number(args, "min_area_ratio"), max_ratio = get_number(args, "max_area_ratio");
        if (min_ratio > max_ratio)
            throw std::runtime_error("min_area_ratio must not exceed max_area_ratio");
        filter.min_area_ratio = float(min_ratio);
        filter.max_area_ratio = float(max_ratio);
        filter.sort_by        = orders.at(get_string(args, "sort_by"));
        const auto image      = image_view(args.at("image"));
        const auto view       = read_detections(args.at("detections"));
        auto rows             = detection_rows(view, image);
        const auto indices    = ops::filter_detection_indices(rows, image.width, image.height, filter);
        std::vector<float> xyxy, scores;
        std::vector<int64_t> class_ids;
        for (const auto index : indices) {
            const auto* box = view.xyxy.data() + index * 4;
            xyxy.insert(xyxy.end(), box, box + 4);
            scores.push_back(view.scores[index]);
            class_ids.push_back(view.class_ids[index]);
        }
        return detection_value(std::move(xyxy), std::move(scores), std::move(class_ids));
    }

    Value detection_box_masks(Values& args, RunState&) {
        const auto image = image_view(args.at("image"));
        const auto rows  = detection_rows(read_detections(args.at("detections")), image);
        if (uint64_t(image.width) * image.height * rows.size() > 256 * 1024 * 1024)
            throw std::runtime_error("detection masks exceed 256 MiB");
        std::vector<Value> masks;
        for (const auto& row : rows) {
            const auto mask = ops::box_mask(row, image.width, image.height);
            masks.push_back(Value::object(copy_image({uint32_t(mask.width), uint32_t(mask.height), 1, const_cast<uint8_t*>(mask.data.data())})));
        }
        return Value::list(std::move(masks), Kind::Tensor);
    }

    Value detections_to_regions(Values& args, RunState&) {
        const auto image      = image_view(args.at("image"));
        const auto detections = read_detections(args.at("detections"));
        auto boxes            = ops::boxes_from_detections(detections.xyxy, image.width, image.height);
        return regions_value(image.width, image.height, make_tensor(std::move(boxes)));
    }
}
