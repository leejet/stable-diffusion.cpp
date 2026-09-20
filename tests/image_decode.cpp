#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "common/common.h"
#include "common/media_io.h"

static std::string encode_base64(const std::vector<uint8_t>& bytes) {
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve((bytes.size() + 2) / 3 * 4);

    for (size_t i = 0; i < bytes.size(); i += 3) {
        const uint32_t value = (static_cast<uint32_t>(bytes[i]) << 16) |
                               (i + 1 < bytes.size() ? static_cast<uint32_t>(bytes[i + 1]) << 8 : 0) |
                               (i + 2 < bytes.size() ? static_cast<uint32_t>(bytes[i + 2]) : 0);
        result.push_back(alphabet[(value >> 18) & 0x3f]);
        result.push_back(alphabet[(value >> 12) & 0x3f]);
        result.push_back(i + 1 < bytes.size() ? alphabet[(value >> 6) & 0x3f] : '=');
        result.push_back(i + 2 < bytes.size() ? alphabet[value & 0x3f] : '=');
    }

    return result;
}

static bool expect_size(const char* name, const sd_image_t& image, uint32_t width, uint32_t height) {
    if (image.width == width && image.height == height) {
        return true;
    }

    std::cerr << name << " has size " << image.width << "x" << image.height << ", expected " << width << "x"
              << height << '\n';
    return false;
}

int main() {
    const uint8_t pixels[] = {
        0,
        32,
        64,
        16,
        48,
        80,
        32,
        64,
        96,
        48,
        80,
        112,
        64,
        96,
        128,
        80,
        112,
        144,
        96,
        128,
        160,
        112,
        144,
        176,
    };
    const std::vector<uint8_t> encoded = encode_image_to_vector(EncodedImageFormat::PNG, pixels, 4, 2, 3);
    if (encoded.empty()) {
        std::cerr << "failed to encode test image\n";
        return 1;
    }

    const std::string image = "data:image/png;base64," + encode_base64(encoded);

    int decoded_width  = 0;
    int decoded_height = 0;
    uint8_t* decoded   = load_image_from_memory(reinterpret_cast<const char*>(encoded.data()),
                                                static_cast<int>(encoded.size()),
                                                decoded_width,
                                                decoded_height,
                                                0,
                                                0,
                                                3);
    if (decoded == nullptr || decoded_width != 4 || decoded_height != 2) {
        std::cerr << "decode without expected dimensions did not preserve the source size\n";
        free(decoded);
        return 1;
    }
    free(decoded);

    SDGenerationParams params;
    const std::string json = "{\"width\":8,\"height\":8,\"init_image\":\"" + image +
                             "\",\"mask_image\":\"" + image + "\",\"ref_images\":[\"" + image +
                             "\"]}";
    if (!params.from_json_str(json)) {
        std::cerr << "failed to parse image generation parameters\n";
        return 1;
    }

    if (!expect_size("init_image", params.init_image.get(), 8, 8) ||
        !expect_size("mask_image", params.mask_image.get(), 8, 8) ||
        params.ref_images.size() != 1 || !expect_size("ref_images[0]", params.ref_images[0].get(), 4, 2)) {
        return 1;
    }

    return 0;
}
