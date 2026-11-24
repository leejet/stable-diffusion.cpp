#ifndef __SD_DGL_OPS_IMAGE_H__
#define __SD_DGL_OPS_IMAGE_H__

#include <filesystem>
#include "core/media.h"
#include "dgl/tensor.h"

namespace sd::dgl::ops {
    void copy_video_frame(sd::Tensor<float>& video, const sd::Tensor<float>& frame, size_t index);
    std::shared_ptr<sd_tensor_t> allocate_image(int width, int height, int channels);
    std::shared_ptr<sd_tensor_t> copy_image(const sd::Image& image);
    sd::Image image_view(const sd_tensor_t& tensor);
    std::shared_ptr<sd_tensor_t> read_image_tensor(const std::string& path, int requested_channels = 0);
    std::vector<std::shared_ptr<sd_tensor_t>> read_image_directory(const std::filesystem::path& path, size_t max_images, int channels);
    void save_image(const sd_tensor_t& input, const std::string& path, const std::string& metadata = {});
    std::vector<uint8_t> encode_image(const sd_tensor_t& input, const std::string& metadata = {});
    std::shared_ptr<sd_tensor_t> decode_video_jpeg(const std::vector<uint8_t>& bytes, int width, int height);
    std::vector<uint8_t> encode_video_jpeg(const sd::Image& image, int quality);
    void decode_base64(const std::string& text, uint8_t* bytes, size_t size);
    void validate_region_boxes(const sd::Tensor<int64_t>& boxes, int64_t width, int64_t height);
    sd::Tensor<int64_t> boxes_from_detections(const sd::Tensor<float>& detections, int64_t width, int64_t height);
    std::vector<std::shared_ptr<sd_tensor_t>> crop_regions(const sd_tensor_t& input, const sd::Tensor<int64_t>& boxes);
    std::shared_ptr<sd_tensor_t> paste_regions(const sd_tensor_t& background, const sd::Tensor<int64_t>& boxes, const std::vector<std::shared_ptr<sd_tensor_t>>& foregrounds);
    sd::Tensor<float> image_to_f32(const sd_tensor_t& tensor);
    std::shared_ptr<sd_tensor_t> image_to_u8(const sd::Tensor<float>& tensor);

}

#endif
