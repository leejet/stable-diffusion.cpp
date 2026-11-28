#include "photomaker_embedding.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>

#include "ggml.h"
#include "model_io/npy_io.h"

namespace sd::photomaker {

    static Tensor<float> load_legacy_embedding(const std::filesystem::path& path,
                                               std::vector<int64_t> shape,
                                               size_t bytes) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("cannot open PhotoMaker ID embedding");
        const std::streamoff file_size = file.tellg();
        file.seekg(0);

        int32_t header[3];
        if (!file.read(reinterpret_cast<char*>(header), sizeof(header)) ||
            header[0] < 2 || header[0] > 4 || header[1] < 0 || header[2] != GGML_TYPE_F32) {
            throw std::runtime_error("invalid legacy PhotoMaker ID embedding header");
        }
        int32_t dims[4]{1, 1, 1, 1};
        if (!file.read(reinterpret_cast<char*>(dims), header[0] * sizeof(int32_t)) ||
            dims[0] != shape[0] || dims[1] != shape[1] || dims[2] != 1 || dims[3] != 1) {
            throw std::runtime_error("legacy PhotoMaker ID embedding shape does not match the input images");
        }

        const auto payload = std::streamoff(sizeof(header) + header[0] * sizeof(int32_t)) + header[1];
        if (file_size < payload || uint64_t(file_size - payload) != bytes)
            throw std::runtime_error("invalid legacy PhotoMaker ID embedding payload size");
        file.seekg(payload);
        Tensor<float> result(std::move(shape));
        if (!file.read(reinterpret_cast<char*>(result.data()), bytes))
            throw std::runtime_error("cannot read legacy PhotoMaker ID embedding");
        return result;
    }

    Tensor<float> load_id_embedding(const std::string& path, int image_count) {
        constexpr size_t bytes_per_image = id_embedding_width * sizeof(float);
        if (image_count <= 0 || size_t(image_count) > std::numeric_limits<size_t>::max() / bytes_per_image)
            throw std::runtime_error("invalid PhotoMaker input image count");
        const auto bytes = size_t(image_count) * bytes_per_image;
        std::vector<int64_t> shape{id_embedding_width, image_count};
        const auto file_path = std::filesystem::u8path(path);
        auto extension       = file_path.extension().u8string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (extension != ".npy")
            return load_legacy_embedding(file_path, std::move(shape), bytes);

        npy::Reader reader(path, bytes);
        if (reader.type() != SD_TENSOR_F32 || reader.shape() != shape)
            throw std::runtime_error("PhotoMaker v2 expects a float32 NumPy array with shape (image_count, 512)");
        Tensor<float> result(std::move(shape));
        reader.read(result.data(), bytes);
        return result;
    }

}  // namespace sd::photomaker
