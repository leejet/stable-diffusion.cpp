#include "dgl/bindings/postprocess.h"

#include "dgl/bindings/conversion.h"

namespace sd::dgl {
    Value execute_canny(Values& args, RunState&) {
        auto source = image_view(args.at("image"));
        if (source.channel != 3 || get_number(args, "low_threshold") > get_number(args, "high_threshold"))
            throw std::runtime_error("image::canny requires RGB and low_threshold <= high_threshold");
        auto result = copy_image(source);
        if (!sd::preprocess_canny(image_view(*result), float(get_number(args, "high_threshold")), float(get_number(args, "low_threshold")), float(get_number(args, "weak")), float(get_number(args, "strong")), get_bool(args, "invert")))
            throw std::runtime_error("image::canny preprocessing failed");
        return Value::object(result);
    }

}
