#include "dgl/bindings/detailer.h"

#include "dgl/bindings/conversion.h"

#include "dgl/bindings/values.h"

#include "dgl/ops/detailer.h"
#include "dgl/ops/media.h"

namespace sd::dgl {

    Value plan_mask_crops(Values& args, RunState&) {
        const auto ratio = read_int_vector(args.at("aspect"));
        if (ratio.size() != 2 || ratio[0] <= 0 || ratio[1] <= 0 || ratio[0] > 16384 || ratio[1] > 16384)
            throw std::runtime_error("aspect requires two positive integers up to 16384");
        const auto aspect      = float(ratio[0]) / float(ratio[1]);
        const auto padding     = int(get_int(args, "padding"));
        const auto only_masked = get_bool(args, "only_masked");
        const auto& inputs     = args.at("masks").items;
        const auto canvas      = inputs.empty() ? sd::Image{} : image_view(inputs.front());
        std::vector<Value> regions;
        regions.reserve(inputs.size());
        for (const auto& input : inputs) {
            const auto mask = image_view(input);
            if (mask.channel != 1)
                throw std::runtime_error("mask regions require grayscale tensors");
            if (mask.width != canvas.width || mask.height != canvas.height)
                throw std::runtime_error("masks must match the same canvas dimensions");
            ops::DetailCrop crop;
            if (!ops::detail_crop_box(mask, aspect, padding, only_masked, crop))
                continue;
            regions.push_back(Value::dict({
                {"box", shape_value({crop.x1, crop.y1, crop.x2 - crop.x1, crop.y2 - crop.y1})},
                {"mask", input},
                {"index", Value::literal(int64_t(regions.size()))},
            }));
        }
        return Value::list(std::move(regions), Kind::Dict);
    }

}
