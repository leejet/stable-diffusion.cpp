#include "dgl/ops/image.h"
#include "api/metadata.h"
#include "core/file_access.h"
#include "core/ggml_tensor_utils.h"
#include "dgl/ops/tensor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "stb_image.h"
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include "stb_image_write.h"

namespace sd::dgl::ops {

    static void validate_image_shape(int64_t width, int64_t height, int64_t channels) {
        if (width <= 0 || height <= 0 || width > 16384 || height > 16384 ||
            width * height > 64 * 1024 * 1024 || (channels != 1 && channels != 3 && channels != 4))
            throw std::runtime_error("invalid image dimensions or channels (expected 1, 3 or 4)");
    }

    std::shared_ptr<sd_tensor_t> allocate_image(int width, int height, int channels) {
        validate_image_shape(width, height, channels);
        return make_tensor(SD_TENSOR_U8, std::vector<int64_t>{channels, width, height});
    }

    std::shared_ptr<sd_tensor_t> copy_image(const sd::Image& image) {
        validate_image_shape(image.width, image.height, image.channel);
        return make_tensor(SD_TENSOR_U8, std::vector<int64_t>{image.channel, image.width, image.height},
                           image.data, size_t(image.width) * image.height * image.channel);
    }

    sd::Image image_view(const sd_tensor_t& tensor) {
        const auto& shape = tensor.shape();
        if (tensor.type != SD_TENSOR_U8 || shape.size() != 3)
            throw std::runtime_error("image requires a U8 tensor with shape [channels, width, height]");
        validate_image_shape(shape[1], shape[2], shape[0]);
        return {uint32_t(shape[1]), uint32_t(shape[2]), uint32_t(shape[0]), const_cast<uint8_t*>(static_cast<const uint8_t*>(tensor.data()))};
    }

    std::shared_ptr<sd_tensor_t> decode_video_jpeg(const std::vector<uint8_t>& bytes, int width, int height) {
        int w = 0, h = 0, channels = 0;
        if (bytes.size() < 2 || bytes[0] != 0xff || bytes[1] != 0xd8 ||
            !stbi_info_from_memory(bytes.data(), int(bytes.size()), &w, &h, &channels) || w != width || h != height)
            throw std::runtime_error("invalid MJPEG frame dimensions");
        validate_image_shape(width, height, 3);
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(stbi_load_from_memory(bytes.data(), int(bytes.size()), &w, &h, &channels, 3), stbi_image_free);
        if (!pixels || w != width || h != height)
            throw std::runtime_error("cannot decode MJPEG frame");
        return copy_image({uint32_t(width), uint32_t(height), 3, pixels.get()});
    }

    std::vector<uint8_t> encode_video_jpeg(const sd::Image& image, int quality) {
        std::vector<uint8_t> bytes;
        auto append = [](void* context, void* data, int size) {
            auto& output = *static_cast<std::vector<uint8_t>*>(context);
            auto first   = static_cast<const uint8_t*>(data);
            output.insert(output.end(), first, first + size);
        };
        // Keep stb's subsampled JPEG path for compatibility with MJPEG players.
        if (!stbi_write_jpg_to_func(append, &bytes, image.width, image.height, 3, image.data, std::min(quality, 90)))
            throw std::runtime_error("cannot encode MJPEG frame");
        return bytes;
    }

    static const char* base64_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    void decode_base64(const std::string& text, uint8_t* bytes, size_t size) {
        if (text.size() != (size + 2) / 3 * 4)
            throw std::runtime_error("base64 length does not match tensor dimensions");
        size_t output = 0;
        for (size_t i = 0; i < text.size(); i += 4) {
            uint32_t n = 0;
            for (size_t j = 0; j < 4; ++j) {
                bool padding = i + 4 == text.size() && ((j == 3 && size % 3 != 0) || (j == 2 && size % 3 == 1));
                if (padding) {
                    if (text[i + j] != '=')
                        throw std::runtime_error("invalid base64 padding");
                    n <<= 6;
                } else {
                    const char* p = text[i + j] == '\0' ? nullptr : std::strchr(base64_alphabet, text[i + j]);
                    if (!p)
                        throw std::runtime_error("invalid base64 character");
                    n = (n << 6) | uint32_t(p - base64_alphabet);
                }
            }
            if (i + 4 == text.size() && ((size % 3 == 1 && (n & 0xffff)) || (size % 3 == 2 && (n & 0xff))))
                throw std::runtime_error("nonzero base64 padding bits");
            if (output < size)
                bytes[output++] = uint8_t(n >> 16);
            if (output < size)
                bytes[output++] = uint8_t(n >> 8);
            if (output < size)
                bytes[output++] = uint8_t(n);
        }
    }

    struct Region {
        int x, y, width, height;
    };

    static Region region_at(const sd::Tensor<int64_t>& boxes, int64_t index) {
        const auto* box = boxes.data() + index * 4;
        return {int(box[0]), int(box[1]), int(box[2]), int(box[3])};
    }

    void validate_region_boxes(const sd::Tensor<int64_t>& boxes, int64_t width, int64_t height) {
        if (width < 1 || width > 16384 || height < 1 || height > 16384)
            throw std::runtime_error("region canvas width and height must be in [1, 16384]");
        if (boxes.dim() != 2 || boxes.shape()[0] != 4 || boxes.shape()[1] > 10000)
            throw std::runtime_error("region boxes require I64 [4,N] with at most 10000 boxes");
        for (int64_t i = 0; i < boxes.shape()[1]; ++i) {
            const auto* box  = boxes.data() + i * 4;
            const auto label = "region box " + std::to_string(i);
            if (box[0] < 0 || box[1] < 0)
                throw std::runtime_error(label + " has negative coordinates");
            if (box[2] <= 0 || box[3] <= 0)
                throw std::runtime_error(label + " must have positive width and height");
            if (box[0] >= width || box[1] >= height || box[2] > width - box[0] || box[3] > height - box[1])
                throw std::runtime_error(label + " lies outside the canvas");
        }
    }

    sd::Tensor<int64_t> boxes_from_detections(const sd::Tensor<float>& xyxy, int64_t width, int64_t height) {
        if (width < 1 || width > 16384 || height < 1 || height > 16384)
            throw std::runtime_error("region canvas width and height must be in [1, 16384]");
        if (xyxy.dim() != 2 || xyxy.shape()[0] != 4 || xyxy.shape()[1] > 4096)
            throw std::runtime_error("detections.xyxy requires F32 [4,N] with at most 4096 boxes");
        std::vector<int64_t> boxes;
        boxes.reserve(size_t(xyxy.shape()[1]) * 4);
        for (int64_t i = 0; i < xyxy.shape()[1]; ++i) {
            const auto* d = xyxy.data() + i * 4;
            for (int j = 0; j < 4; ++j)
                if (!std::isfinite(d[j]))
                    throw std::runtime_error("detection " + std::to_string(i) + " coordinates must be finite");
            const auto x      = int64_t(std::clamp(std::floor(d[0]), 0.f, float(width)));
            const auto y      = int64_t(std::clamp(std::floor(d[1]), 0.f, float(height)));
            const auto right  = int64_t(std::clamp(std::ceil(d[2]), 0.f, float(width)));
            const auto bottom = int64_t(std::clamp(std::ceil(d[3]), 0.f, float(height)));
            if (right > x && bottom > y)
                boxes.insert(boxes.end(), {x, y, right - x, bottom - y});
        }
        const auto count = int64_t(boxes.size() / 4);
        return sd::Tensor<int64_t>({4, count}, std::move(boxes));
    }

    std::shared_ptr<sd_tensor_t> read_image_tensor(const std::string& path, int requested_channels) {
        require_file_access(path, SD_DGL_FILE_READ);
        int width, height, channels;
        if (!stbi_info(path.c_str(), &width, &height, &channels))
            throw std::runtime_error("cannot read image '" + path + "'");
        const int output_channels = requested_channels ? requested_channels : (channels == 2 ? 4 : channels);
        validate_image_shape(width, height, output_channels);
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(stbi_load(path.c_str(), &width, &height, &channels, output_channels), stbi_image_free);
        if (!pixels)
            throw std::runtime_error("image decode failed");
        return copy_image({uint32_t(width), uint32_t(height), uint32_t(output_channels), pixels.get()});
    }

    std::vector<std::shared_ptr<sd_tensor_t>> read_image_directory(const std::filesystem::path& path, size_t max_images, int channels) {
        require_file_access(path.u8string(), SD_DGL_FILE_READ);
        std::vector<std::filesystem::path> paths;
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            auto extension = entry.path().extension().u8string();
            std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
            if (entry.is_regular_file() && (extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp"))
                paths.push_back(entry.path());
        }
        std::sort(paths.begin(), paths.end());
        if (max_images && paths.size() > max_images)
            paths.resize(max_images);
        if (paths.empty())
            throw std::runtime_error("image directory has no supported images");
        std::vector<std::shared_ptr<sd_tensor_t>> images;
        size_t bytes = 0;
        for (const auto& file : paths) {
            auto image = read_image_tensor(file.u8string(), channels);
            if (image->bytes > 256 * 1024 * 1024 - bytes)
                throw std::runtime_error("image directory exceeds 256 MiB");
            bytes += image->bytes;
            images.push_back(std::move(image));
        }
        return images;
    }

    std::vector<std::shared_ptr<sd_tensor_t>> crop_regions(const sd_tensor_t& input, const sd::Tensor<int64_t>& boxes) {
        const auto image = image_view(input);
        validate_region_boxes(boxes, image.width, image.height);
        std::vector<std::shared_ptr<sd_tensor_t>> crops;
        crops.reserve(size_t(boxes.shape()[1]));
        for (int64_t i = 0; i < boxes.shape()[1]; ++i) {
            const auto r = region_at(boxes, i);
            crops.push_back(crop_tensor(input, {1, 2}, {r.x, r.y}, {r.width, r.height}));
        }
        return crops;
    }

    std::shared_ptr<sd_tensor_t> paste_regions(const sd_tensor_t& background, const sd::Tensor<int64_t>& boxes, const std::vector<std::shared_ptr<sd_tensor_t>>& items) {
        const auto source = image_view(background);
        validate_region_boxes(boxes, source.width, source.height);
        if (items.size() != size_t(boxes.shape()[1]))
            throw std::runtime_error("foreground count does not match region boxes");
        auto result      = copy_image(source);
        const auto image = image_view(*result);
        for (size_t i = 0; i < items.size(); ++i) {
            const auto r = region_at(boxes, int64_t(i));
            auto patch   = items[i];
            if (!patch || image_view(*patch).channel != image.channel)
                throw std::runtime_error("region foreground channels differ from the background");
            auto resized    = interpolate_tensor(*patch, {r.width, r.height}, {1, 2}, "bilinear", false);
            const auto view = image_view(*resized);
            for (int y = 0; y < r.height; ++y)
                std::memcpy(image.data + (size_t(y + r.y) * image.width + r.x) * image.channel,
                            view.data + size_t(y) * r.width * image.channel, size_t(r.width) * image.channel);
        }
        return result;
    }

    std::vector<uint8_t> encode_image(const sd_tensor_t& input, const std::string& metadata) {
        const auto image = image_view(input);
        std::vector<uint8_t> bytes;
        auto append = [](void* context, void* data, int size) {
            auto& output      = *static_cast<std::vector<uint8_t>*>(context);
            const auto* first = static_cast<const uint8_t*>(data);
            output.insert(output.end(), first, first + size);
        };
        if (!stbi_write_png_to_func(append, &bytes, image.width, image.height, image.channel, image.data, image.width * image.channel) ||
            !sd::metadata::embed_png_metadata(bytes, metadata))
            throw std::runtime_error("cannot encode image or generation metadata");
        return bytes;
    }

    void save_image(const sd_tensor_t& input, const std::string& path, const std::string& metadata) {
        if (path.size() < 4 || path.substr(path.size() - 4) != ".png")
            throw std::runtime_error("image::save requires a .png path");
        require_file_access(path, SD_DGL_FILE_WRITE);
        const auto bytes = encode_image(input, metadata);
        std::ofstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        file.close();
        if (!file)
            throw std::runtime_error("cannot save image '" + path + "'");
    }

    void copy_video_frame(sd::Tensor<float>& video, const sd::Tensor<float>& frame, size_t index) {
        const auto& shape      = video.shape();
        const auto plane_size  = size_t(shape[0]) * size_t(shape[1]);
        const auto frame_count = size_t(shape[2]);
        for (size_t channel = 0; channel < size_t(shape[3]); ++channel) {
            const auto offset = (channel * frame_count + index) * plane_size;
            std::copy_n(frame.data() + channel * plane_size, plane_size, video.data() + offset);
        }
    }
    sd::Tensor<float> image_to_f32(const sd_tensor_t& tensor) {
        const auto& s = tensor.shape();
        if (tensor.type != SD_TENSOR_U8 || (s.size() != 3 && s.size() != 4) || s[0] < 1 || s[0] > 4 ||
            s[1] < 1 || s[1] > 16384 || s[2] < 1 || s[2] > 16384 || (s.size() == 4 && s[3] < 1)) {
            throw std::runtime_error("expected a U8 tensor with channels, width, height and optional time");
        }
        auto* data        = const_cast<uint8_t*>(static_cast<const uint8_t*>(tensor.data()));
        const auto stride = s[0] * s[1] * s[2];
        auto frame        = [&](int64_t i) {
            return sd_image_to_tensor({uint32_t(s[1]), uint32_t(s[2]), uint32_t(s[0]), data + i * stride});
        };
        if (s.size() == 3) {
            return frame(0);
        }
        const std::vector<int64_t> shape{s[1], s[2], s[3], s[0], 1};
        if (tensor.bytes > std::vector<float>().max_size()) {
            throw std::runtime_error("video pixel shape exceeds addressable float storage");
        }
        sd::Tensor<float> output(shape);
        for (int64_t i = 0; i < s[3]; ++i) {
            copy_video_frame(output, frame(i), size_t(i));
        }
        return output;
    }
    std::shared_ptr<sd_tensor_t> image_to_u8(const sd::Tensor<float>& tensor) {
        if (tensor.empty() || (tensor.dim() != 4 && tensor.dim() != 5)) {
            throw std::runtime_error("expected nonempty float pixels with width, height, optional time, channels and batch");
        }
        const auto& s       = tensor.shape();
        const auto channels = s[tensor.dim() == 5 ? 3 : 2];
        const auto frames   = tensor.dim() == 5 ? s[2] : 1;
        std::vector<int64_t> shape{channels, s[0], s[1]};
        if (tensor.dim() == 5) {
            shape.push_back(frames);
        }
        auto pixels       = make_tensor(SD_TENSOR_U8, std::move(shape));
        auto* data        = static_cast<uint8_t*>(pixels->mutable_data());
        const auto stride = pixels->bytes / size_t(frames);
        for (int64_t i = 0; i < frames; ++i) {
            std::unique_ptr<uint8_t, decltype(&std::free)> frame(tensor_to_sd_image(tensor, int(i)).data, std::free);
            std::memcpy(data + stride * size_t(i), frame.get(), stride);
        }
        return pixels;
    }
}  // namespace sd::dgl::ops
