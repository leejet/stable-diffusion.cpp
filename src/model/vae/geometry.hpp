#ifndef __SD_MODEL_VAE_GEOMETRY_HPP__
#define __SD_MODEL_VAE_GEOMETRY_HPP__

#include "core/image_geometry.h"
#include "core/option_types.h"
#include "model.h"

namespace sd::vae {
    inline SDVersion version_for_format(sd_vae_format_t format, SDVersion fallback) {
        switch (format) {
            case SD_VAE_FORMAT_FLUX:
                return VERSION_FLUX;
            case SD_VAE_FORMAT_SD3:
                return VERSION_SD3;
            case SD_VAE_FORMAT_FLUX2:
                return VERSION_FLUX2;
            case SD_VAE_FORMAT_WAN:
                return VERSION_WAN2;
            default:
                return fallback;
        }
    }

    inline int spatial_scale(SDVersion version) {
        if (version == VERSION_LTXAV)
            return 32;
        if (version == VERSION_WAN2_2_TI2V || version == VERSION_QWEN_IMAGE_2_1 ||
            sd_version_is_hunyuan_video(version) || sd_version_is_mage_flow(version) ||
            sd_version_is_minimax_h3(version) || sd_version_uses_flux2_vae(version))
            return 16;
        if (version == VERSION_CHROMA_RADIANCE || version == VERSION_HIDREAM_O1 ||
            sd_version_is_minit2i(version) || sd_version_is_sensenova_u1(version) ||
            sd_version_is_z_image_l2p(version) || version == VERSION_IRIS)
            return 1;
        return 8;
    }

    inline int aligned_frames(SDVersion version, int frames) {
        if (frames <= 0)
            throw std::invalid_argument("video frame count must be positive");
        if (sd_version_is_minimax_h3(version))
            return int(sd::align_video_frame_count(frames, 17, 5));
        if (sd_version_is_ltxav(version))
            return int(sd::align_video_frame_count(frames, 8));
        if (sd_version_is_wan(version) || sd_version_is_lingbot_video(version) ||
            sd_version_is_hunyuan_video(version) || version == VERSION_SEEDVR2)
            return int(sd::align_video_frame_count(frames, 4));
        return frames;
    }
}

#endif  // __SD_MODEL_VAE_GEOMETRY_HPP__
