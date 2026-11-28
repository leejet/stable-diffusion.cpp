#include "api/generation.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>

#include "api/generation_program.h"
#include "api/result.h"
#include "api/tensor.h"
#include "core/ggml_extend_backend.h"
#include "core/image_geometry.h"
#include "core/util.h"
#include "dgl/diagnostic.h"
#include "dgl/executor.h"
#include "dgl/models/component_sources.h"
#include "model/vae/geometry.hpp"
#include "name_conversion.h"

namespace sd::api {
    GenerationModel::GenerationModel(std::shared_ptr<sd_opt_t> source, Operation purpose)
        : options(std::move(source)) {
        const auto status = resolve(options.get(), purpose, config);
        if (status != SD_STATUS_OK)
            throw dgl::Error(status, sd_get_last_error());
    }

    GenerationContext::GenerationContext(const sd_opt_t* source, Operation purpose)
        : model(copy_options(source), purpose) {
        const auto& settings = model.config.context;
        execution.set("n_threads", std::to_string(settings.n_threads));
        if (settings.linear_type)
            execution.set("linear_type", settings.linear_type);
        execution.set("conditioning_cache_size", std::to_string(settings.conditioning_cache_size));
        for (const auto& [key, enabled] : std::map<std::string, bool>{
                 {"auto_fit", settings.auto_fit}, {"batched_cfg", settings.batched_cfg}, {"enable_mmap", settings.enable_mmap}, {"eager_load", settings.eager_load}, {"disable_prefetch", settings.disable_prefetch}, {"disable_segmented_compute", settings.disable_segmented_compute}})
            execution.set(key, enabled ? "true" : "false");
        if (settings.max_vram && *settings.max_vram)
            execution.set("max_vram", settings.max_vram);
        if (settings.max_ram && *settings.max_ram)
            execution.set("max_ram", settings.max_ram);
        if (settings.rpc_servers && *settings.rpc_servers && !add_rpc_devices(settings.rpc_servers))
            throw dgl::Error(SD_STATUS_LOAD_FAILED, "cannot register RPC devices");
    }

    static bool contains_audio_vae(const ModelLoader& metadata, SDVersion version) {
        for (const auto& name : metadata.get_tensor_names()) {
            const auto canonical = convert_tensor_name(name, version);
            if (canonical.rfind("audio_vae.", 0) == 0 || canonical.rfind("vocoder.", 0) == 0)
                return true;
        }
        return false;
    }

    const ModelDefinition& GenerationModel::definition(GenerationContext& context) {
        std::lock_guard<std::recursive_mutex> lock(context.mutex);
        if (context.probing_model)
            throw dgl::Error(SD_STATUS_BUSY, "model probe is in progress");
        context.probing_model = true;
        struct ProbeGuard {
            bool& probing;
            ~ProbeGuard() { probing = false; }
        } probe{context.probing_model};
        context.check_cancelled();
        const auto& settings = config.context;
        const auto* path     = settings.diffusion_model_path && *settings.diffusion_model_path ? settings.diffusion_model_path : settings.model_path;
        if (!path || !*path)
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "model_path or diffusion_model_path is required");
        require_file_access(path, SD_DGL_FILE_READ_MODEL);
        if ((sd_version_is_minimax_h3(family) || sd_version_is_ltxav(family)) && settings.vae_path && *settings.vae_path)
            require_file_access(settings.vae_path, SD_DGL_FILE_READ_MODEL);
        bool changed = family == VERSION_COUNT;
        if (!changed && !ModelLoader::dependencies_changed(probe_dependencies, changed))
            throw dgl::Error(SD_STATUS_LOAD_FAILED, "cannot inspect model dependencies");
        if (changed) {
            try {
                ModelLoader::Dependencies dependencies;
                ModelLoader metadata;
                auto detected = dgl::models::resolve_family(VERSION_COUNT, path, &dependencies, &metadata);
                ModelLoader vae_metadata;
                bool inspected_vae = false;
                if (detected == VERSION_MINIMAX_H3 && settings.vae_path && *settings.vae_path && std::string(settings.vae_path) != path) {
                    require_file_access(settings.vae_path, SD_DGL_FILE_READ_MODEL);
                    if (!vae_metadata.init_from_file(settings.vae_path))
                        throw dgl::Error(SD_STATUS_LOAD_FAILED, "cannot inspect video VAE metadata");
                    inspected_vae      = true;
                    const auto sources = vae_metadata.dependencies();
                    dependencies.insert(dependencies.end(), sources.begin(), sources.end());
                    for (const auto& [name, tensor] : vae_metadata.get_tensor_storage_map())
                        if (ends_with(name, "decoder.proj_out.weight") && tensor.ne[1] == 768)
                            detected = VERSION_P_H3_IMAGE;
                }
                const auto* definition = find_model_definition(detected);
                if (!definition)
                    throw dgl::Error(SD_STATUS_UNSUPPORTED, "unsupported model family");
                default_prediction   = definition->sampling.prediction_from_weights ? definition->sampling.prediction_from_weights(metadata.get_tensor_storage_map()) : definition->sampling.default_prediction;
                wan_image_condition  = detected == VERSION_WAN2_2_I2V;
                wan_vision_condition = wan_end_vision = wan_vace = false;
                if (sd_version_is_wan(detected))
                    for (const auto& [name, tensor] : metadata.get_tensor_storage_map()) {
                        wan_vision_condition |= name.find("img_emb.") != std::string::npos;
                        wan_end_vision |= name.find("img_emb.emb_pos") != std::string::npos;
                        wan_vace |= name.find("vace_blocks.") != std::string::npos;
                    }
                wan_image_condition |= wan_vision_condition;
                embedded_audio_vae.clear();
                if (sd_version_is_ltxav(detected) || sd_version_is_minimax_h3(detected)) {
                    if (contains_audio_vae(metadata, detected)) {
                        embedded_audio_vae = path;
                    } else if (settings.vae_path && *settings.vae_path && std::string(settings.vae_path) != path &&
                               (!settings.audio_vae_path || !*settings.audio_vae_path)) {
                        require_file_access(settings.vae_path, SD_DGL_FILE_READ_MODEL);
                        if (!inspected_vae && !vae_metadata.init_from_file(settings.vae_path))
                            throw dgl::Error(SD_STATUS_LOAD_FAILED, "cannot inspect video VAE metadata");
                        if (!inspected_vae) {
                            const auto sources = vae_metadata.dependencies();
                            dependencies.insert(dependencies.end(), sources.begin(), sources.end());
                        }
                        if (contains_audio_vae(vae_metadata, detected))
                            embedded_audio_vae = settings.vae_path;
                    }
                }
                context.check_cancelled();
                family             = detected;
                probe_dependencies = std::move(dependencies);
                LOG_INFO("Detected model family: %s (%s)", sd_version_name(family), std::filesystem::u8path(path).filename().u8string().c_str());
            } catch (const dgl::Error&) {
                family = VERSION_COUNT;
                throw;
            } catch (const std::exception& error) {
                family = VERSION_COUNT;
                throw dgl::Error(SD_STATUS_LOAD_FAILED, error.what());
            }
        }
        const auto* definition = find_model_definition(family);
        if (!definition)
            throw dgl::Error(SD_STATUS_UNSUPPORTED, "unsupported model family");
        return *definition;
    }

    bool supports_generation(const ModelDefinition& definition, bool video) {
        return definition.traits.supports_video_generation == video && definition.version != VERSION_ESRGAN &&
               definition.version != VERSION_SEEDVR2 && definition.version != VERSION_RELAY_VSR;
    }

    void GenerationContext::check_cancelled() const {
        if (cancellation.load(std::memory_order_relaxed) == SD_CANCEL_ALL)
            throw dgl::Error(SD_STATUS_CANCELLED, "generation cancelled");
    }

    GenerationRun::GenerationRun(GenerationContext& context)
        : context_(context) {
        std::unique_lock<std::recursive_mutex> lock(context_.mutex, std::try_to_lock);
        if (!lock.owns_lock() || context_.executing || context_.probing_model)
            throw dgl::Error(SD_STATUS_BUSY, "context is busy");
        std::unique_lock<std::mutex> execution_lock(context_.execution.mutex, std::try_to_lock);
        if (!execution_lock.owns_lock() || context_.execution.active_run)
            throw dgl::Error(SD_STATUS_BUSY, "DGL context is busy");
        context_.execution.generating = true;
        context_.executing            = true;
    }

    GenerationRun::~GenerationRun() {
        std::lock_guard<std::recursive_mutex> lock(context_.mutex);
        std::lock_guard<std::mutex> execution_lock(context_.execution.mutex);
        context_.execution.generating = false;
        context_.cancellation.store(SD_CANCEL_RESET);
        context_.executing = false;
    }

    static void generation_progress(const sd_generation_progress_t* info, void* data) {
        auto& context   = *static_cast<GenerationContext*>(data);
        auto translated = *info;
        translated.sample_pass += context.sample_pass;
        if (context.progress)
            context.progress(&translated, context.progress_data);
    }

    dgl::Value execute_program(GenerationContext& context, const GenerationProgram& program, const dgl::Graph* compiled) {
        context.check_cancelled();
        FileAccessScope access(context.file_access);
        dgl::Graph graph;
        if (!compiled) {
            graph    = dgl::compile(program.source, dgl::operations(), !program.inputs.empty());
            compiled = &graph;
        }
        context.check_cancelled();
        dgl::RunCallbacks callbacks;
        callbacks.progress      = generation_progress;
        callbacks.progress_data = &context;
        dgl::RunEvents events(callbacks, true);
        struct ActiveRun {
            GenerationContext& context;
            dgl::RunEvents& events;
            ~ActiveRun() {
                std::lock_guard<std::recursive_mutex> lock(context.mutex);
                std::lock_guard<std::mutex> execution_lock(context.execution.mutex);
                context.sample_pass += events.preview_info().sample_pass;
                context.active_run           = nullptr;
                context.execution.active_run = nullptr;
            }
        } active{context, events};
        {
            std::lock_guard<std::recursive_mutex> lock(context.mutex);
            std::lock_guard<std::mutex> execution_lock(context.execution.mutex);
            context.active_run           = &events;
            context.execution.active_run = &events;
            if (context.cancellation.load() == SD_CANCEL_ALL)
                events.cancel();
        }
        dgl::RunEventsScope scope(events);
        dgl::RunState state{context.execution, events};
        LOG_DEBUG("Executing DGL program:\n%s", compiled->source.c_str());
        return dgl::run(*compiled, state, program.inputs);
    }

    static int64_t resolved_seed(int64_t requested) {
        if (requested >= 0)
            return requested;
        std::random_device entropy;
        return int64_t((uint64_t(entropy()) << 32 | entropy()) & INT64_MAX);
    }

    sd_status_t generate_images(GenerationContext& context, const sd_img_gen_opt_t* options, sd_result_t** out) {
        GenerationRun request(context);
        FileAccessScope access(context.file_access);
        auto snapshot = copy_options(options);
        OptConfig config;
        const auto status = resolve(snapshot.get(), Image, config);
        if (status != SD_STATUS_OK)
            return status;
        context.check_cancelled();
        const auto seed = resolved_seed(config.image.seed);
        auto result     = std::make_unique<sd_result_t>();
        for (int i = 0; i < config.image.batch_count; ++i) {
            if (context.cancellation.load() == SD_CANCEL_NEW_LATENTS)
                break;
            auto program = image_program(context, config.image, snapshot.get(), int64_t((uint64_t(seed) + uint64_t(i)) & INT64_MAX));
            auto value   = execute_program(context, program);
            append_program_image(*result, value, program.source);
        }
        if (result->images.empty())
            throw dgl::Error(SD_STATUS_CANCELLED, "generation cancelled before the next image");
        context.check_cancelled();
        *out = result.release();
        return SD_STATUS_OK;
    }
    sd_status_t generate_video(GenerationContext& context, const sd_vid_gen_opt_t* options, sd_result_t** out) {
        GenerationRun request(context);
        FileAccessScope access(context.file_access);
        auto snapshot = copy_options(options);
        OptConfig config;
        const auto status = resolve(snapshot.get(), Video, config);
        if (status != SD_STATUS_OK)
            return status;
        auto program = video_program(context, config.video, snapshot.get(), resolved_seed(config.video.seed));
        auto value   = execute_program(context, program);
        auto result  = std::make_unique<sd_result_t>();
        set_program_video(*result, value, program.source);
        context.check_cancelled();
        *out = result.release();
        return SD_STATUS_OK;
    }

}
