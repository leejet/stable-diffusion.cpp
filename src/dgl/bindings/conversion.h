#ifndef __SD_DGL_BINDINGS_CONVERSION_H__
#define __SD_DGL_BINDINGS_CONVERSION_H__

#include "core/file_access.h"
#include "dgl/executor.h"
#include "dgl/ops/audio.h"
#include "dgl/ops/image.h"
#include "dgl/ops/tensor.h"
#include "dgl/ops/video.h"
#include "dgl/tensor.h"
#include "stable-diffusion.h"

#include <filesystem>

namespace sd::dgl {
    using ops::allocate_image;
    using ops::audio_view;
    using ops::copy_audio;
    using ops::copy_image;
    using ops::crop_tensor;
    using ops::decode_base64;
    using ops::decode_pcm_audio;
    using ops::decode_video_jpeg;
    using ops::encode_pcm_audio;
    using ops::encode_video_jpeg;
    using ops::image_view;
    using ops::interpolate_tensor;
    using ops::read_image_tensor;
    using ops::read_wav;
    using ops::write_wav;

    const sd_tensor_t& i64_vector(const Value& value);
    std::vector<int64_t> read_int_vector(const Value& value);
    Value shape_value(const std::vector<int64_t>& shape);
    Value tensor_value(sd::Tensor<float> tensor);
    sd::Image image_view(const Value& value);

    template <class T>
    std::shared_ptr<T> object(const Values& args, const std::string& name) {
        return args.at(name).as<T>();
    }

}  // namespace sd::dgl

#endif  // __SD_DGL_BINDINGS_CONVERSION_H__
