#include "dgl/ops/prompt.h"

#include <algorithm>
#include <filesystem>
#include <set>
#include "conditioning/conditioner.hpp"
#include "core/file_access.h"
#include "core/image_tensor.h"
#include "dgl/models/text_encoder.h"
#include "dgl/ops/image.h"
#include "dgl/ops/video.h"

namespace sd::dgl::ops {
    MiniMaxH3PresentationItem prepare_prompt_media(PromptMedia input) {
        const auto* video = std::get_if<std::shared_ptr<const sd_tensor_t>>(&input.media);
        if (input.stride < 1 || input.stride > 65536 || (input.stride != 1 && !video))
            throw std::runtime_error("media stride requires video frames and a positive frame interval");
        MiniMaxH3PresentationItem result;
        if (video) {
            if (!*video || input.fps_num <= 0 || input.fps_den <= 0)
                throw std::runtime_error("media video requires frames and a positive frame rate");
            const auto& tensor = **video;
            const auto& shape  = tensor.shape();
            size_t count       = 0;
            if (tensor.type == SD_TENSOR_U8) {
                validate_video_frames(tensor);
                count = size_t(shape[3]);
            } else if (tensor.type == SD_TENSOR_F32 && shape.size() == 5 &&
                       shape[0] > 0 && shape[0] <= 16384 && shape[1] > 0 && shape[1] <= 16384 &&
                       shape[2] > 0 && shape[3] == 3 && shape[4] == 1) {
                count = size_t(shape[2]);
            } else {
                throw std::runtime_error("media video requires U8 [3,width,height,frames] or F32 [width,height,frames,3,1]");
            }
            result.kind = MiniMaxH3PresentationKind::VIDEO;
            for (size_t i = 0; i < count; i += size_t(input.stride)) {
                if (tensor.type == SD_TENSOR_U8) {
                    result.frames.push_back(sd_image_to_tensor(video_frame_view(tensor, i)));
                } else {
                    Tensor<float> frame({shape[0], shape[1], 3, 1});
                    const auto plane = size_t(shape[0] * shape[1]);
                    for (size_t channel = 0; channel < 3; ++channel)
                        std::copy_n(tensor.as<float>().data() + (channel * count + i) * plane, plane, frame.data() + channel * plane);
                    result.frames.push_back(std::move(frame));
                }
                result.timestamps.push_back(float(i) * float(input.fps_den) / float(input.fps_num));
            }
        } else if (std::holds_alternative<AudioReadResult>(input.media)) {
            const auto& audio = std::get<AudioReadResult>(input.media);
            if (!audio.samples)
                throw std::runtime_error("media audio requires samples");
            audio_view(*audio.samples, audio.sample_rate);
            result.kind = MiniMaxH3PresentationKind::AUDIO;
        } else {
            result.kind = MiniMaxH3PresentationKind::IMAGE;
            auto image  = std::get<Tensor<float>>(std::move(input.media));
            if (image.dim() != 4 || image.empty() || image.shape()[3] != 1 || image.shape()[2] == 2)
                throw std::runtime_error("media image must be a single frame with 1, 3 or 4 channels");
            if (image.shape()[2] == 1)
                image = sd::ensure_image_tensor_channels(image, 3);
            result.frames.push_back(std::move(image));
        }
        return result;
    }

    SDCondition encode_prompt(models::TextEncoder& model, PromptEncodingOptions options) {
        if (options.raw && options.system)
            throw std::runtime_error("raw prompt encoding cannot be combined with system");
        if (options.raw && (options.media || options.image_bindings))
            throw std::runtime_error("raw prompt encoding accepts text only; omit media and image_bindings");
        std::vector<Tensor<float>> images;
        std::vector<MiniMaxH3PresentationItem> presentations;
        if (options.media && !options.media->empty()) {
            const bool images_only = std::all_of(options.media->begin(), options.media->end(), [](const PromptMedia& input) {
                return std::holds_alternative<Tensor<float>>(input.media);
            });
            if (images_only && options.media->size() > model.max_reference_images())
                throw std::runtime_error("this text encoder accepts at most " + std::to_string(model.max_reference_images()) + " media images");
            if (!images_only && !model.supports_multimodal_prompt())
                throw std::runtime_error("this text encoder does not support video or audio prompt media");
            for (size_t index = 0; index < options.media->size(); ++index) {
                try {
                    auto prepared = prepare_prompt_media(std::move((*options.media)[index]));
                    if (images_only)
                        images.push_back(std::move(prepared.frames.front()));
                    else
                        presentations.push_back(std::move(prepared));
                } catch (const std::exception& error) {
                    throw std::runtime_error("media[" + std::to_string(index) + "]: " + error.what());
                }
            }
        }
        models::Model::ExecutionScope scope(model);
        scope.require_ready();
        ConditionerParams params;
        params.text          = options.text;
        params.width         = options.width;
        params.height        = options.height;
        params.clip_skip     = options.clip_skip;
        params.raw           = options.raw;
        params.system_prompt = options.system;
        if (!images.empty())
            params.ref_images = &images;
        if (!presentations.empty())
            params.minimax_h3_references = &presentations;
        if (params.raw) {
            auto condition = model.get_learned_condition(params);
            if (condition.empty())
                throw std::runtime_error("prompt encoding failed");
            return condition;
        }
        auto prompt = sd::parse_prompt_input(params.text, true);
        std::set<std::string> embedding_paths;
        for (auto& part : prompt) {
            if (part.type != sd::PromptPart::Type::EMBEDDING)
                continue;
            if (!model.supports_textual_inversion())
                throw std::runtime_error("inline textual inversion requires an SD 1.x, SD 2.x or SDXL CLIP text encoder");
            sd::require_file_access(part.content, SD_DGL_FILE_READ_MODEL);
            part.content = std::filesystem::weakly_canonical(std::filesystem::u8path(part.content)).generic_u8string();
            ModelLoader::FileStamp stamp;
            if (!ModelLoader::read_file_stamp(part.content, stamp))
                throw std::runtime_error("cannot inspect textual inversion '" + part.content + "'");
            if (!params.textual_inversion)
                params.textual_inversion = std::make_shared<sd::TextualInversion>();
            if (embedding_paths.insert(part.content).second)
                params.textual_inversion->sources.push_back(std::move(stamp));
        }
        if (sd::prompt_has_images(prompt)) {
            if (!model.supports_prompt_images())
                throw std::runtime_error("inline prompt images require a compatible text encoder and vision weights");
            if (!sd::resolve_prompt_images(prompt, [&](const std::string& source) {
                    auto image = options.image_bindings ? options.image_bindings(source) : Tensor<float>{};
                    if (image.empty())
                        image = sd_image_to_tensor(image_view(*read_image_tensor(source, 3)));
                    if (image.dim() != 4)
                        throw std::runtime_error("inline prompt images must be single images");
                    if (image.shape()[2] == 3)
                        return image;
                    if (image.shape()[2] == 2)
                        image = sd::ops::slice(image, 2, 0, 1);
                    return sd::ensure_image_tensor_channels(image, 3);
                }))
                throw std::runtime_error("cannot resolve inline prompt images");
            params.prompt_input = &prompt;
        }
        if (params.textual_inversion)
            params.prompt_input = &prompt;
        if (!params.prompt_input)
            params.text = sd::prompt_input_text(prompt);
        auto condition = model.get_learned_condition(params);
        if (condition.empty())
            throw std::runtime_error("prompt encoding failed");
        return condition;
    }
}
