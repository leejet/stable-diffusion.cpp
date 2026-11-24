#ifndef __SD_EXAMPLES_CLI_MEDIA_INPUT_H__
#define __SD_EXAMPLES_CLI_MEDIA_INPUT_H__

#include "common/resource_owners.hpp"

struct VideoInput {
    SDTensorPtr frames;
    SDTensorPtr samples;
    uint32_t sample_rate = 0;
    int fps              = 24;

    sd_video_t view() const { return {frames.get(), fps, {sample_rate, sample_rate ? samples.get() : nullptr}}; }
};

bool load_images_from_dir(const std::string& path, std::vector<SDImageOwner>& images, int max_image_num = 0);
bool load_video_input(const std::string& path, int fps, size_t max_frames, bool audio, VideoInput& result);
bool load_reference_video_inputs(const std::vector<std::string>& paths, const std::vector<std::string>& audio_paths, std::vector<VideoInput>& result);
bool set_video_inputs(sd_opt_t* options, const std::vector<VideoInput>& references, const sd_tensor_t* control);

#endif  // __SD_EXAMPLES_CLI_MEDIA_INPUT_H__
