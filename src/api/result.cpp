#include "result.h"
#include "api.h"

#include <cstdlib>
#include <cstring>
#include <utility>

void sd::free_audio(sd::Audio* audio) {
    if (audio == nullptr) {
        return;
    }
    free(audio->data);
    audio->data = nullptr;
    free(audio);
}

void sd::free_image(sd::Image* image) {
    if (image) {
        free(image->data);
        *image = {};
    }
}

void sd_result_free(sd_result_t* result) {
    delete result;
}

sd_result_kind_t sd_result_get_kind(const sd_result_t* result) {
    sd::api::clear_error();
    if (!result) {
        sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "result is required");
        return SD_RESULT_INVALID;
    }
    return result->kind;
}

size_t sd_result_get_image_count(const sd_result_t* result) {
    if (!sd_result_get_kind(result))
        return 0;
    return result->images.size();
}

const sd_tensor_t* sd_result_get_image(const sd_result_t* result, size_t index) {
    if (index >= sd_result_get_image_count(result)) {
        sd::api::error(result ? SD_STATUS_OUT_OF_RANGE : SD_STATUS_INVALID_ARGUMENT, nullptr, "invalid result or image index");
        return nullptr;
    }
    return result->images[index].get();
}

const char* sd_result_get_generation_info(const sd_result_t* result, size_t index) {
    if (!sd_result_get_image(result, index))
        return nullptr;
    return result->generation_infos[result->kind == SD_RESULT_VIDEO ? 0 : index].c_str();
}

const sd_audio_t* sd_result_get_audio(const sd_result_t* result) {
    if (!sd_result_get_kind(result))
        return nullptr;
    return result->audio.data ? &result->audio : nullptr;
}

int sd_result_get_fps(const sd_result_t* result) {
    if (!sd_result_get_kind(result))
        return 0;
    return result->fps;
}

const sd_video_t* sd_result_get_video(const sd_result_t* result) {
    const sd_video_t* output = nullptr;
    sd::api::call([&] {
        if (!result)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "result is required");
        if (result->kind != SD_RESULT_VIDEO)
            return sd::api::error(SD_STATUS_TYPE_MISMATCH, nullptr, "result is not a video");
        std::lock_guard<std::mutex> lock(result->video_mutex);
        if (!result->video_tensor) {
            if (result->images.empty())
                return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "video has no frames");
            const auto& first = *result->images.front().get();
            auto shape        = first.shape();
            shape.push_back(static_cast<int64_t>(result->images.size()));
            for (const auto& frame : result->images)
                if (frame->type != first.type || frame->shape() != first.shape())
                    return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "video frames must have identical shapes and types");
            auto packed       = sd::api::allocate_tensor(first.type, std::move(shape));
            auto* destination = static_cast<uint8_t*>(packed.get()->mutable_data());
            for (const auto& frame : result->images) {
                memcpy(destination, frame->data(), frame->bytes);
                destination += frame->bytes;
            }
            result->video_tensor = std::move(packed);
            result->video        = {result->video_tensor.get(), result->fps, result->audio};
        }
        output = &result->video;
        return SD_STATUS_OK;
    });
    return output;
}
