#ifndef __DIFFUSION_MODEL_H__
#define __DIFFUSION_MODEL_H__

#include "flux.hpp"
#include "mmdit.hpp"
#include "unet.hpp"
#include "wan.hpp"

struct DiffusionModel {
    virtual std::string get_desc()                                                      = 0;
    virtual void compute(int n_threads,
                         struct ggml_tensor* x,
                         struct ggml_tensor* timesteps,
                         struct ggml_tensor* context,
                         struct ggml_tensor* c_concat,
                         struct ggml_tensor* y,
                         struct ggml_tensor* guidance,
                         std::vector<ggml_tensor*> ref_latents     = {},
                         bool increase_ref_index                   = false,
                         int num_video_frames                      = -1,
                         std::vector<struct ggml_tensor*> controls = {},
                         float control_strength                    = 0.f,
                         struct ggml_tensor** output               = NULL,
                         struct ggml_context* output_ctx           = NULL,
                         std::vector<int> skip_layers              = std::vector<int>())             = 0;
    virtual void alloc_params_buffer()                                                  = 0;
    virtual void free_params_buffer()                                                   = 0;
    virtual void free_compute_buffer()                                                  = 0;
    virtual void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) = 0;
    virtual size_t get_params_buffer_size()                                             = 0;
    virtual int64_t get_adm_in_channels()                                               = 0;
};

struct UNetModel : public DiffusionModel {
    UNetModelRunner unet;

    UNetModel(ggml_backend_t backend,
              bool offload_params_to_cpu,
              const String2GGMLType& tensor_types = {},
              SDVersion version                   = VERSION_SD1,
              bool flash_attn                     = false)
        : unet(backend, offload_params_to_cpu, tensor_types, "model.diffusion_model", version, flash_attn) {
    }

    std::string get_desc() {
        return unet.get_desc();
    }

    void alloc_params_buffer() {
        unet.alloc_params_buffer();
    }

    void free_params_buffer() {
        unet.free_params_buffer();
    }

    void free_compute_buffer() {
        unet.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        unet.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() {
        return unet.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        return unet.unet.adm_in_channels;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* c_concat,
                 struct ggml_tensor* y,
                 struct ggml_tensor* guidance,
                 std::vector<ggml_tensor*> ref_latents     = {},
                 bool increase_ref_index                   = false,
                 int num_video_frames                      = -1,
                 std::vector<struct ggml_tensor*> controls = {},
                 float control_strength                    = 0.f,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL,
                 std::vector<int> skip_layers              = std::vector<int>()) {
        (void)skip_layers;  // SLG doesn't work with UNet models
        return unet.compute(n_threads, x, timesteps, context, c_concat, y, num_video_frames, controls, control_strength, output, output_ctx);
    }
};

struct MMDiTModel : public DiffusionModel {
    MMDiTRunner mmdit;

    MMDiTModel(ggml_backend_t backend,
               bool offload_params_to_cpu,
               bool flash_attn                     = false,
               const String2GGMLType& tensor_types = {})
        : mmdit(backend, offload_params_to_cpu, flash_attn, tensor_types, "model.diffusion_model") {
    }

    std::string get_desc() {
        return mmdit.get_desc();
    }

    void alloc_params_buffer() {
        mmdit.alloc_params_buffer();
    }

    void free_params_buffer() {
        mmdit.free_params_buffer();
    }

    void free_compute_buffer() {
        mmdit.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        mmdit.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() {
        return mmdit.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        return 768 + 1280;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* c_concat,
                 struct ggml_tensor* y,
                 struct ggml_tensor* guidance,
                 std::vector<ggml_tensor*> ref_latents     = {},
                 bool increase_ref_index                   = false,
                 int num_video_frames                      = -1,
                 std::vector<struct ggml_tensor*> controls = {},
                 float control_strength                    = 0.f,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL,
                 std::vector<int> skip_layers              = std::vector<int>()) {
        return mmdit.compute(n_threads, x, timesteps, context, y, output, output_ctx, skip_layers);
    }
};

struct FluxModel : public DiffusionModel {
    Flux::FluxRunner flux;

    FluxModel(ggml_backend_t backend,
              bool offload_params_to_cpu,
              const String2GGMLType& tensor_types = {},
              SDVersion version                   = VERSION_FLUX,
              bool flash_attn                     = false,
              bool use_mask                       = false)
        : flux(backend, offload_params_to_cpu, tensor_types, "model.diffusion_model", version, flash_attn, use_mask) {
    }

    std::string get_desc() {
        return flux.get_desc();
    }

    void alloc_params_buffer() {
        flux.alloc_params_buffer();
    }

    void free_params_buffer() {
        flux.free_params_buffer();
    }

    void free_compute_buffer() {
        flux.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        flux.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() {
        return flux.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        return 768;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* c_concat,
                 struct ggml_tensor* y,
                 struct ggml_tensor* guidance,
                 std::vector<ggml_tensor*> ref_latents     = {},
                 bool increase_ref_index                   = false,
                 int num_video_frames                      = -1,
                 std::vector<struct ggml_tensor*> controls = {},
                 float control_strength                    = 0.f,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL,
                 std::vector<int> skip_layers              = std::vector<int>()) {
        return flux.compute(n_threads, x, timesteps, context, c_concat, y, guidance, ref_latents, increase_ref_index, output, output_ctx, skip_layers);
    }
};

struct WanModel : public DiffusionModel {
    std::string prefix;
    WAN::WanRunner wan;

    WanModel(ggml_backend_t backend,
             bool offload_params_to_cpu,
             const String2GGMLType& tensor_types = {},
             const std::string prefix            = "model.diffusion_model",
             SDVersion version                   = VERSION_WAN2,
             bool flash_attn                     = false)
        : prefix(prefix), wan(backend, offload_params_to_cpu, tensor_types, prefix, version, flash_attn) {
    }

    std::string get_desc() {
        return wan.get_desc();
    }

    void alloc_params_buffer() {
        wan.alloc_params_buffer();
    }

    void free_params_buffer() {
        wan.free_params_buffer();
    }

    void free_compute_buffer() {
        wan.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        wan.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() {
        return wan.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        return 768;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* c_concat,
                 struct ggml_tensor* y,
                 struct ggml_tensor* guidance,
                 std::vector<ggml_tensor*> ref_latents     = {},
                 bool increase_ref_index                   = false,
                 int num_video_frames                      = -1,
                 std::vector<struct ggml_tensor*> controls = {},
                 float control_strength                    = 0.f,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL,
                 std::vector<int> skip_layers              = std::vector<int>()) {
        return wan.compute(n_threads, x, timesteps, context, y, c_concat, NULL, output, output_ctx);
    }
};

#endif
