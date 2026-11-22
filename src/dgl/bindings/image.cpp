#include "dgl/bindings/image.h"

#include "core/image_tensor.h"
#include "dgl/bindings/arguments.h"
#include "dgl/bindings/conversion.h"

namespace sd::dgl {
    Value execute_image_to_u8(Values& args, RunState&) {
        sd::Tensor<float> decoded;
        return image_to_u8(pixels(args.at("pixels"), decoded));
    }

    Value execute_image_to_f32(Values& args, RunState&) {
        const auto& image = args.at("image");
        sd::Tensor<float> decoded;
        const auto& input   = pixels(image, decoded);
        const auto channels = get_int(args, "channels");
        const auto current  = input.shape()[input.dim() == 5 ? 3 : 2];
        if (!channels || channels == current) {
            if (&input != &decoded)
                return image;
            return tensor_value(std::move(decoded));
        }
        if ((channels != 3 && channels != 4) || current == 2) {
            throw std::runtime_error("unsupported pixel channel conversion");
        }
        return tensor_value(sd::ensure_image_tensor_channels(input, int(channels)));
    }

    Value execute_load_image(Values& args, RunState&) {
        return Value::object(read_image_tensor(get_string(args, "path")));
    }
    Value execute_load_images(Values& args, RunState&) {
        std::vector<Value> images;
        for (auto image : ops::read_image_directory(std::filesystem::u8path(get_string(args, "path")), size_t(get_int(args, "max_images")), 0))
            images.push_back(Value::object(std::move(image)));
        return Value::list(std::move(images));
    }
    Value execute_save_image(Values& args, RunState& state) {
        ops::save_image(*object<sd_tensor_t>(args, "image"), get_string(args, "path"), state.metadata);
        return Value::void_value();
    }
}
