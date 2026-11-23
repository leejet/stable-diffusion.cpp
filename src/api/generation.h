#ifndef __SD_API_GENERATION_H__
#define __SD_API_GENERATION_H__

#include <atomic>
#include <map>
#include <mutex>
#include <string>

#include "api/generation_template.h"
#include "api/options.h"
#include "dgl/context.h"
#include "dgl/graph.h"
#include "model_definition.h"

namespace sd::api {
    struct GenerationContext;

    struct GenerationModel {
        std::shared_ptr<sd_opt_t> options;
        OptConfig config;
        SDVersion family                = VERSION_COUNT;
        prediction_t default_prediction = PREDICTION_COUNT;
        bool wan_image_condition        = false;
        bool wan_vision_condition       = false;
        bool wan_end_vision             = false;
        bool wan_vace                   = false;
        std::string embedded_audio_vae;
        ModelLoader::Dependencies probe_dependencies;

        GenerationModel(std::shared_ptr<sd_opt_t> source, Operation purpose);
        const ModelDefinition& definition(GenerationContext& context);
    };

    struct GenerationContext {
        GenerationModel model;
        sd_dgl_ctx_t execution;
        FileAccessPolicy file_access;
        std::recursive_mutex mutex;
        bool executing     = false;
        bool probing_model = false;
        std::atomic<sd_cancel_mode_t> cancellation{SD_CANCEL_RESET};
        dgl::RunEvents* active_run           = nullptr;
        sd_generation_progress_cb_t progress = nullptr;
        void* progress_data                  = nullptr;
        sd_prompt_image_cb_t prompt_image    = nullptr;
        void* prompt_image_data              = nullptr;
        int sample_pass                      = 0;

        GenerationContext(const sd_opt_t* source, Operation purpose);
        void check_cancelled() const;
    };

    bool supports_generation(const ModelDefinition& definition, bool video);

    class GenerationRun {
        GenerationContext& context_;

    public:
        explicit GenerationRun(GenerationContext& context);
        ~GenerationRun();
        GenerationRun(const GenerationRun&)            = delete;
        GenerationRun& operator=(const GenerationRun&) = delete;
    };

    struct GenerationProgram {
        std::string source;
        dgl::Values inputs;
    };

    dgl::Value execute_program(GenerationContext& context, const GenerationProgram& program, const dgl::Graph* compiled = nullptr);
    GenerationProgram image_program(GenerationContext& context, const ImageConfig& image, const sd_opt_t* explicit_options, int64_t seed);
    GenerationProgram video_program(GenerationContext& context, const VideoConfig& video, const sd_opt_t* explicit_options, int64_t seed);
    sd_status_t generate_images(GenerationContext& context, const sd_img_gen_opt_t* options, sd_result_t** result);
    sd_status_t generate_video(GenerationContext& context, const sd_vid_gen_opt_t* options, sd_result_t** result);
}

#endif  // __SD_API_GENERATION_H__
