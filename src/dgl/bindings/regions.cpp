#include "dgl/bindings/regions.h"

#include "dgl/bindings/conversion.h"

namespace sd::dgl {
    OutputSignature regions_signature() {
        return {{Kind::Dict}, {{"width", {Kind::Integer}}, {"height", {Kind::Integer}}, {"boxes", {Kind::Tensor}}}};
    }

    static const sd::Tensor<int64_t>& region_box_tensor(const std::shared_ptr<sd_tensor_t>& boxes) {
        if (!boxes || boxes->type != SD_TENSOR_I64)
            throw std::runtime_error("region boxes require an I64 [4,N] tensor");
        if (boxes->shape().size() != 2 || boxes->shape()[0] != 4 || boxes->shape()[1] > 10000)
            throw std::runtime_error("region boxes require I64 [4,N] with at most 10000 boxes");
        // Public empty tensor handles retain shape; their internal storage does not.
        static const sd::Tensor<int64_t> empty({4, 0});
        return boxes->bytes ? boxes->as<int64_t>() : empty;
    }

    Value regions_value(int64_t width, int64_t height, std::shared_ptr<sd_tensor_t> boxes) {
        ops::validate_region_boxes(region_box_tensor(boxes), width, height);
        return Value::dict({{"width", Value::literal(width)}, {"height", Value::literal(height)}, {"boxes", Value::object(std::move(boxes))}});
    }

    static std::shared_ptr<sd_tensor_t> read_regions(const Value& value, const sd::Image& canvas) {
        if (value.type().kind != Kind::Dict)
            throw std::runtime_error("regions require a dict with width, height and boxes");
        const auto& fields = value.fields;
        for (const auto* name : {"width", "height", "boxes"}) {
            const auto found = fields.find(name);
            if (found == fields.end())
                throw std::runtime_error(std::string("regions are missing field '") + name + "'");
            const auto expected = std::string(name) == "boxes" ? Kind::Tensor : Kind::Integer;
            if (found->second.type().kind != expected)
                throw std::runtime_error(std::string("regions field '") + name + "' expects " + type_name({expected}));
        }
        const auto width = get_int(fields, "width"), height = get_int(fields, "height");
        auto boxes = fields.at("boxes").as<sd_tensor_t>();
        ops::validate_region_boxes(region_box_tensor(boxes), width, height);
        if (width != canvas.width || height != canvas.height)
            throw std::runtime_error("region canvas dimensions do not match the image");
        return boxes;
    }

    Value regions_from_boxes(Values& args, RunState&) {
        const auto& input = args.at("boxes");
        std::shared_ptr<sd_tensor_t> boxes;
        if (input.type().kind == Kind::Tensor) {
            boxes = input.as<sd_tensor_t>();
        } else if (input.type().kind == Kind::List) {
            if (input.items.size() % 4 || input.items.size() > 40000)
                throw std::runtime_error("boxes must contain x,y,width,height groups (at most 10000)");
            std::vector<int64_t> data;
            data.reserve(input.items.size());
            for (const auto& item : input.items) {
                if (item.type().kind != Kind::Integer)
                    throw std::runtime_error("region box list elements must be integers");
                data.push_back(item.scalar.get<int64_t>());
            }
            boxes = make_tensor(SD_TENSOR_I64, std::vector<int64_t>{4, int64_t(data.size() / 4)}, data.data(), data.size() * sizeof(int64_t));
        } else
            throw std::runtime_error("boxes require a flat list<int> or an I64 [4,N] tensor");
        return regions_value(get_int(args, "width"), get_int(args, "height"), std::move(boxes));
    }

    Value crop_regions(Values& args, RunState&) {
        const auto image = object<sd_tensor_t>(args, "image");
        auto boxes       = read_regions(args.at("regions"), image_view(*image));
        std::vector<Value> crops;
        for (auto crop : ops::crop_regions(*image, region_box_tensor(boxes)))
            crops.push_back(Value::object(std::move(crop)));
        return Value::list(std::move(crops), Kind::Tensor);
    }

    Value paste_regions(Values& args, RunState&) {
        const auto background = object<sd_tensor_t>(args, "background");
        auto boxes            = read_regions(args.at("regions"), image_view(*background));
        std::vector<std::shared_ptr<sd_tensor_t>> foregrounds;
        for (const auto& item : args.at("foregrounds").items)
            foregrounds.push_back(item.as<sd_tensor_t>());
        return Value::object(ops::paste_regions(*background, region_box_tensor(boxes), foregrounds));
    }
}
