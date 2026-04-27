#ifndef __DIFFUSION_MODEL_H__
#define __DIFFUSION_MODEL_H__

#include <optional>
#include "anima.hpp"
#include "ernie_image.hpp"
#include "flux.hpp"
#include "mmdit.hpp"
#include "qwen_image.hpp"
#include "tensor_ggml.hpp"
#include "unet.hpp"
#include "wan.hpp"
#include "z_image.hpp"

struct DiffusionParams {
    const sd::Tensor<float>* x                        = nullptr;
    const sd::Tensor<float>* timesteps                = nullptr;
    const sd::Tensor<float>* context                  = nullptr;
    const sd::Tensor<float>* c_concat                 = nullptr;
    const sd::Tensor<float>* y                        = nullptr;
    const sd::Tensor<int32_t>* t5_ids                 = nullptr;
    const sd::Tensor<float>* t5_weights               = nullptr;
    const sd::Tensor<float>* guidance                 = nullptr;
    const std::vector<sd::Tensor<float>>* ref_latents = nullptr;
    bool increase_ref_index                           = false;
    int num_video_frames                              = -1;
    const std::vector<sd::Tensor<float>>* controls    = nullptr;
    float control_strength                            = 0.f;
    const sd::Tensor<float>* vace_context             = nullptr;
    float vace_strength                               = 1.f;
    const std::vector<int>* skip_layers               = nullptr;
};

template <typename T>
static inline const sd::Tensor<T>& tensor_or_empty(const sd::Tensor<T>* tensor) {
    static const sd::Tensor<T> kEmpty;
    return tensor != nullptr ? *tensor : kEmpty;
}

// Helper to convert sd::Tensor<T> pointers to temporary ggml_tensor* for streaming code paths.
// The returned ggml_tensors live in the provided ggml_context and point to the sd::Tensor's data.
struct StreamingParamConverter {
    ggml_context* ctx = nullptr;

    StreamingParamConverter() {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 16 * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
    }

    ~StreamingParamConverter() {
        if (ctx) ggml_free(ctx);
    }

    template <typename T>
    ggml_tensor* convert(const sd::Tensor<T>* tensor) {
        if (tensor == nullptr || tensor->numel() == 0) return nullptr;
        ggml_tensor* t = sd::make_ggml_tensor(ctx, *tensor, false);
        t->data = const_cast<void*>(static_cast<const void*>(tensor->data()));
        return t;
    }

    std::vector<ggml_tensor*> convert_vec(const std::vector<sd::Tensor<float>>* tensors) {
        std::vector<ggml_tensor*> result;
        if (tensors == nullptr) return result;
        for (const auto& t : *tensors) {
            sd::Tensor<float> tmp_ref = t;  // non-const copy for convert
            result.push_back(convert(&tmp_ref));
        }
        return result;
    }
};

struct DiffusionModel {
    virtual std::string get_desc()                                               = 0;
    virtual sd::Tensor<float> compute(int n_threads,
                                      const DiffusionParams& diffusion_params)   = 0;
    virtual void alloc_params_buffer()                                           = 0;
    virtual void free_params_buffer()                                            = 0;
    virtual void free_compute_buffer()                                           = 0;
    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) = 0;
    virtual size_t get_params_buffer_size()                                      = 0;
    virtual void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter){};
    virtual int64_t get_adm_in_channels()                            = 0;
    virtual void set_flash_attention_enabled(bool enabled)           = 0;
    virtual void set_circular_axes(bool circular_x, bool circular_y) = 0;

    // Dynamic tensor offloading interface
    virtual bool is_params_on_gpu() const { return false; }
    virtual bool move_params_to_cpu() { return false; }
    virtual bool move_params_to_gpu() { return false; }
    virtual size_t get_params_vram_size() const { return 0; }

    // Layer streaming interface (for granular tensor offloading)
    virtual bool supports_layer_streaming() const { return false; }
    virtual void enable_layer_streaming(int prefetch_layers = 1, size_t min_free_vram = 512 * 1024 * 1024) {
        (void)prefetch_layers;
        (void)min_free_vram;
    }
    virtual void disable_layer_streaming() {}
    virtual bool is_layer_streaming_enabled() const { return false; }
    virtual bool compute_streaming(int n_threads,
                                   DiffusionParams diffusion_params,
                                   struct ggml_tensor** output     = nullptr,
                                   struct ggml_context* output_ctx = nullptr) {
        // Default: fall back to regular compute, copy result to output
        auto result = compute(n_threads, diffusion_params);
        if (output != nullptr && result.numel() > 0) {
            if (*output == nullptr && output_ctx != nullptr) {
                auto shape = result.shape();
                int n_dims = std::min(static_cast<int>(shape.size()), GGML_MAX_DIMS);
                std::array<int64_t, GGML_MAX_DIMS> ne = {1, 1, 1, 1};
                for (int i = 0; i < n_dims; i++) ne[i] = shape[i];
                *output = ggml_new_tensor(output_ctx, GGML_TYPE_F32, n_dims, ne.data());
            }
            if (*output != nullptr) {
                memcpy((*output)->data, result.data(), result.numel() * sizeof(float));
            }
        }
        return result.numel() > 0;
    }
    // Offload all streaming layers to CPU (free GPU memory after diffusion)
    virtual void offload_streaming_layers() {}
};

struct UNetModel : public DiffusionModel {
    UNetModelRunner unet;

    UNetModel(ggml_backend_t backend,
              bool offload_params_to_cpu,
              const String2TensorStorage& tensor_storage_map = {},
              SDVersion version                              = VERSION_SD1)
        : unet(backend, offload_params_to_cpu, tensor_storage_map, "model.diffusion_model", version) {
    }

    std::string get_desc() override {
        return unet.get_desc();
    }

    void alloc_params_buffer() override {
        unet.alloc_params_buffer();
    }

    void free_params_buffer() override {
        unet.free_params_buffer();
    }

    void free_compute_buffer() override {
        unet.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        unet.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return unet.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        unet.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return unet.unet.adm_in_channels;
    }

    void set_flash_attention_enabled(bool enabled) {
        unet.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        unet.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<sd::Tensor<float>> empty_controls;
        return unet.compute(n_threads,
                            *diffusion_params.x,
                            *diffusion_params.timesteps,
                            tensor_or_empty(diffusion_params.context),
                            tensor_or_empty(diffusion_params.c_concat),
                            tensor_or_empty(diffusion_params.y),
                            diffusion_params.num_video_frames,
                            diffusion_params.controls ? *diffusion_params.controls : empty_controls,
                            diffusion_params.control_strength);
    }

    // Dynamic tensor offloading
    bool is_params_on_gpu() const override { return unet.is_params_on_gpu(); }
    bool move_params_to_cpu() override { return unet.move_params_to_cpu(); }
    bool move_params_to_gpu() override { return unet.move_params_to_gpu(); }
    size_t get_params_vram_size() const override { return unet.get_params_vram_size(); }

    // Layer streaming (coarse-stage for UNet due to skip connections)
    bool supports_layer_streaming() const override { return true; }

    void enable_layer_streaming(int prefetch_layers, size_t min_free_vram) override {
        LayerStreaming::StreamingConfig config;
        config.prefetch_layers = prefetch_layers;
        config.min_free_vram = min_free_vram;
        unet.enable_layer_streaming(config);
    }

    void disable_layer_streaming() override {
        unet.disable_layer_streaming();
    }

    bool is_layer_streaming_enabled() const override {
        return unet.is_streaming_enabled();
    }

    void offload_streaming_layers() override {
        unet.offload_streaming_layers();
    }

    bool compute_streaming(int n_threads,
                           DiffusionParams diffusion_params,
                           struct ggml_tensor** output     = nullptr,
                           struct ggml_context* output_ctx = nullptr) override {
        StreamingParamConverter cvt;
        auto controls_vec = cvt.convert_vec(diffusion_params.controls);
        return unet.compute_streaming(n_threads,
                                      cvt.convert(diffusion_params.x),
                                      cvt.convert(diffusion_params.timesteps),
                                      cvt.convert(diffusion_params.context),
                                      cvt.convert(diffusion_params.c_concat),
                                      cvt.convert(diffusion_params.y),
                                      diffusion_params.num_video_frames,
                                      controls_vec,
                                      diffusion_params.control_strength,
                                      output,
                                      output_ctx);
    }
};

struct MMDiTModel : public DiffusionModel {
    MMDiTRunner mmdit;

    MMDiTModel(ggml_backend_t backend,
               bool offload_params_to_cpu,
               const String2TensorStorage& tensor_storage_map = {})
        : mmdit(backend, offload_params_to_cpu, tensor_storage_map, "model.diffusion_model") {
    }

    std::string get_desc() override {
        return mmdit.get_desc();
    }

    void alloc_params_buffer() override {
        mmdit.alloc_params_buffer();
    }

    void free_params_buffer() override {
        mmdit.free_params_buffer();
    }

    void free_compute_buffer() override {
        mmdit.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        mmdit.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return mmdit.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        mmdit.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768 + 1280;
    }

    void set_flash_attention_enabled(bool enabled) {
        mmdit.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        mmdit.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<int> empty_skip_layers;
        return mmdit.compute(n_threads,
                             *diffusion_params.x,
                             *diffusion_params.timesteps,
                             tensor_or_empty(diffusion_params.context),
                             tensor_or_empty(diffusion_params.y),
                             diffusion_params.skip_layers ? *diffusion_params.skip_layers : empty_skip_layers);
    }

    // Dynamic tensor offloading
    bool is_params_on_gpu() const override { return mmdit.is_params_on_gpu(); }
    bool move_params_to_cpu() override { return mmdit.move_params_to_cpu(); }
    bool move_params_to_gpu() override { return mmdit.move_params_to_gpu(); }
    size_t get_params_vram_size() const override { return mmdit.get_params_vram_size(); }

    // Layer streaming (granular tensor offloading)
    bool supports_layer_streaming() const override { return true; }

    void enable_layer_streaming(int prefetch_layers, size_t min_free_vram) override {
        LayerStreaming::StreamingConfig config;
        config.prefetch_layers = prefetch_layers;
        config.min_free_vram = min_free_vram;
        mmdit.enable_layer_streaming(config);
    }

    void disable_layer_streaming() override {
        mmdit.disable_layer_streaming();
    }

    bool is_layer_streaming_enabled() const override {
        return mmdit.is_streaming_enabled();
    }

    void offload_streaming_layers() override {
        mmdit.offload_streaming_layers();
    }

    bool compute_streaming(int n_threads,
                           DiffusionParams diffusion_params,
                           struct ggml_tensor** output     = nullptr,
                           struct ggml_context* output_ctx = nullptr) override {
        StreamingParamConverter cvt;
        auto skip = diffusion_params.skip_layers ? *diffusion_params.skip_layers : std::vector<int>();
        return mmdit.compute_streaming(n_threads,
                                       cvt.convert(diffusion_params.x),
                                       cvt.convert(diffusion_params.timesteps),
                                       cvt.convert(diffusion_params.context),
                                       cvt.convert(diffusion_params.y),
                                       output,
                                       output_ctx,
                                       skip);
    }
};

struct FluxModel : public DiffusionModel {
    Flux::FluxRunner flux;

    FluxModel(ggml_backend_t backend,
              bool offload_params_to_cpu,
              const String2TensorStorage& tensor_storage_map = {},
              SDVersion version                              = VERSION_FLUX,
              bool use_mask                                  = false)
        : flux(backend, offload_params_to_cpu, tensor_storage_map, "model.diffusion_model", version, use_mask) {
    }

    std::string get_desc() override {
        return flux.get_desc();
    }

    void alloc_params_buffer() override {
        flux.alloc_params_buffer();
    }

    void free_params_buffer() override {
        flux.free_params_buffer();
    }

    void free_compute_buffer() override {
        flux.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        flux.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() override {
        return flux.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        flux.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        flux.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        flux.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<sd::Tensor<float>> empty_ref_latents;
        static const std::vector<int> empty_skip_layers;
        return flux.compute(n_threads,
                            *diffusion_params.x,
                            *diffusion_params.timesteps,
                            tensor_or_empty(diffusion_params.context),
                            tensor_or_empty(diffusion_params.c_concat),
                            tensor_or_empty(diffusion_params.y),
                            tensor_or_empty(diffusion_params.guidance),
                            diffusion_params.ref_latents ? *diffusion_params.ref_latents : empty_ref_latents,
                            diffusion_params.increase_ref_index,
                            diffusion_params.skip_layers ? *diffusion_params.skip_layers : empty_skip_layers);
    }

    // Dynamic tensor offloading
    bool is_params_on_gpu() const override { return flux.is_params_on_gpu(); }
    bool move_params_to_cpu() override { return flux.move_params_to_cpu(); }
    bool move_params_to_gpu() override { return flux.move_params_to_gpu(); }
    size_t get_params_vram_size() const override { return flux.get_params_vram_size(); }

    // Layer streaming (granular tensor offloading)
    bool supports_layer_streaming() const override { return true; }

    void enable_layer_streaming(int prefetch_layers, size_t min_free_vram) override {
        LayerStreaming::StreamingConfig config;
        config.prefetch_layers = prefetch_layers;
        config.min_free_vram = min_free_vram;
        flux.enable_layer_streaming(config);
    }

    void disable_layer_streaming() override {
        flux.disable_layer_streaming();
    }

    bool is_layer_streaming_enabled() const override {
        return flux.is_streaming_enabled();
    }

    void offload_streaming_layers() override {
        flux.offload_streaming_layers();
    }

    bool compute_streaming(int n_threads,
                           DiffusionParams diffusion_params,
                           struct ggml_tensor** output     = nullptr,
                           struct ggml_context* output_ctx = nullptr) override {
        StreamingParamConverter cvt;
        auto ref_vec  = cvt.convert_vec(diffusion_params.ref_latents);
        auto skip     = diffusion_params.skip_layers ? *diffusion_params.skip_layers : std::vector<int>();
        return flux.compute_streaming(n_threads,
                                      cvt.convert(diffusion_params.x),
                                      cvt.convert(diffusion_params.timesteps),
                                      cvt.convert(diffusion_params.context),
                                      cvt.convert(diffusion_params.c_concat),
                                      cvt.convert(diffusion_params.y),
                                      cvt.convert(diffusion_params.guidance),
                                      ref_vec,
                                      diffusion_params.increase_ref_index,
                                      output,
                                      output_ctx,
                                      skip);
    }
};

struct AnimaModel : public DiffusionModel {
    std::string prefix;
    Anima::AnimaRunner anima;

    AnimaModel(ggml_backend_t backend,
               bool offload_params_to_cpu,
               const String2TensorStorage& tensor_storage_map = {},
               const std::string prefix                       = "model.diffusion_model")
        : prefix(prefix), anima(backend, offload_params_to_cpu, tensor_storage_map, prefix) {
    }

    std::string get_desc() override {
        return anima.get_desc();
    }

    void alloc_params_buffer() override {
        anima.alloc_params_buffer();
    }

    void free_params_buffer() override {
        anima.free_params_buffer();
    }

    void free_compute_buffer() override {
        anima.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        anima.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return anima.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        anima.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        anima.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        anima.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        return anima.compute(n_threads,
                             *diffusion_params.x,
                             *diffusion_params.timesteps,
                             tensor_or_empty(diffusion_params.context),
                             tensor_or_empty(diffusion_params.t5_ids),
                             tensor_or_empty(diffusion_params.t5_weights));
    }

    bool supports_layer_streaming() const override { return true; }

    void enable_layer_streaming(int prefetch_layers, size_t min_free_vram) override {
        LayerStreaming::StreamingConfig config;
        config.prefetch_layers = prefetch_layers;
        config.min_free_vram = min_free_vram;
        anima.enable_layer_streaming(config);
    }

    void disable_layer_streaming() override {
        anima.disable_layer_streaming();
    }

    bool is_layer_streaming_enabled() const override {
        return anima.is_streaming_enabled();
    }

    void offload_streaming_layers() override {
        anima.offload_streaming_layers();
    }

    bool compute_streaming(int n_threads,
                           DiffusionParams diffusion_params,
                           struct ggml_tensor** output     = nullptr,
                           struct ggml_context* output_ctx = nullptr) override {
        StreamingParamConverter cvt;
        return anima.compute_streaming(n_threads,
                                       cvt.convert(diffusion_params.x),
                                       cvt.convert(diffusion_params.timesteps),
                                       cvt.convert(diffusion_params.context),
                                       cvt.convert(diffusion_params.t5_ids),
                                       cvt.convert(diffusion_params.t5_weights),
                                       output,
                                       output_ctx);
    }
};

struct WanModel : public DiffusionModel {
    std::string prefix;
    WAN::WanRunner wan;

    WanModel(ggml_backend_t backend,
             bool offload_params_to_cpu,
             const String2TensorStorage& tensor_storage_map = {},
             const std::string prefix                       = "model.diffusion_model",
             SDVersion version                              = VERSION_WAN2)
        : prefix(prefix), wan(backend, offload_params_to_cpu, tensor_storage_map, prefix, version) {
    }

    std::string get_desc() override {
        return wan.get_desc();
    }

    void alloc_params_buffer() override {
        wan.alloc_params_buffer();
    }

    void free_params_buffer() override {
        wan.free_params_buffer();
    }

    void free_compute_buffer() override {
        wan.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        wan.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return wan.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        wan.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        wan.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        wan.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        return wan.compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context),
                           tensor_or_empty(diffusion_params.y),
                           tensor_or_empty(diffusion_params.c_concat),
                           sd::Tensor<float>(),
                           tensor_or_empty(diffusion_params.vace_context),
                           diffusion_params.vace_strength);
    }

    // Dynamic tensor offloading
    bool is_params_on_gpu() const override { return wan.is_params_on_gpu(); }
    bool move_params_to_cpu() override { return wan.move_params_to_cpu(); }
    bool move_params_to_gpu() override { return wan.move_params_to_gpu(); }
    size_t get_params_vram_size() const override { return wan.get_params_vram_size(); }

    // Layer streaming (granular tensor offloading)
    bool supports_layer_streaming() const override { return true; }

    void enable_layer_streaming(int prefetch_layers, size_t min_free_vram) override {
        LayerStreaming::StreamingConfig config;
        config.prefetch_layers = prefetch_layers;
        config.min_free_vram = min_free_vram;
        wan.enable_layer_streaming(config);
    }

    void disable_layer_streaming() override {
        wan.disable_layer_streaming();
    }

    bool is_layer_streaming_enabled() const override {
        return wan.is_streaming_enabled();
    }

    void offload_streaming_layers() override {
        wan.offload_streaming_layers();
    }

    bool compute_streaming(int n_threads,
                           DiffusionParams diffusion_params,
                           struct ggml_tensor** output     = nullptr,
                           struct ggml_context* output_ctx = nullptr) override {
        StreamingParamConverter cvt;
        return wan.compute_streaming(n_threads,
                                     cvt.convert(diffusion_params.x),
                                     cvt.convert(diffusion_params.timesteps),
                                     cvt.convert(diffusion_params.context),
                                     cvt.convert(diffusion_params.y),
                                     cvt.convert(diffusion_params.c_concat),
                                     nullptr,
                                     cvt.convert(diffusion_params.vace_context),
                                     diffusion_params.vace_strength,
                                     output,
                                     output_ctx);
    }
};

struct QwenImageModel : public DiffusionModel {
    std::string prefix;
    Qwen::QwenImageRunner qwen_image;

    QwenImageModel(ggml_backend_t backend,
                   bool offload_params_to_cpu,
                   const String2TensorStorage& tensor_storage_map = {},
                   const std::string prefix                       = "model.diffusion_model",
                   SDVersion version                              = VERSION_QWEN_IMAGE,
                   bool zero_cond_t                               = false)
        : prefix(prefix), qwen_image(backend, offload_params_to_cpu, tensor_storage_map, prefix, version, zero_cond_t) {
    }

    std::string get_desc() override {
        return qwen_image.get_desc();
    }

    void alloc_params_buffer() override {
        qwen_image.alloc_params_buffer();
    }

    void free_params_buffer() override {
        qwen_image.free_params_buffer();
    }

    void free_compute_buffer() override {
        qwen_image.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        qwen_image.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return qwen_image.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        qwen_image.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        qwen_image.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        qwen_image.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<sd::Tensor<float>> empty_ref_latents;
        return qwen_image.compute(n_threads,
                                  *diffusion_params.x,
                                  *diffusion_params.timesteps,
                                  tensor_or_empty(diffusion_params.context),
                                  diffusion_params.ref_latents ? *diffusion_params.ref_latents : empty_ref_latents,
                                  true);
    }

    // Dynamic tensor offloading
    bool is_params_on_gpu() const override { return qwen_image.is_params_on_gpu(); }
    bool move_params_to_cpu() override { return qwen_image.move_params_to_cpu(); }
    bool move_params_to_gpu() override { return qwen_image.move_params_to_gpu(); }
    size_t get_params_vram_size() const override { return qwen_image.get_params_vram_size(); }

    // Layer streaming (granular tensor offloading)
    bool supports_layer_streaming() const override { return true; }

    void enable_layer_streaming(int prefetch_layers, size_t min_free_vram) override {
        LayerStreaming::StreamingConfig config;
        config.prefetch_layers = prefetch_layers;
        config.min_free_vram = min_free_vram;
        qwen_image.enable_layer_streaming(config);
    }

    void disable_layer_streaming() override {
        qwen_image.disable_layer_streaming();
    }

    bool is_layer_streaming_enabled() const override {
        return qwen_image.is_streaming_enabled();
    }

    bool compute_streaming(int n_threads,
                           DiffusionParams diffusion_params,
                           struct ggml_tensor** output     = nullptr,
                           struct ggml_context* output_ctx = nullptr) override {
        StreamingParamConverter cvt;
        auto ref_vec = cvt.convert_vec(diffusion_params.ref_latents);
        return qwen_image.compute_streaming(n_threads,
                                            cvt.convert(diffusion_params.x),
                                            cvt.convert(diffusion_params.timesteps),
                                            cvt.convert(diffusion_params.context),
                                            ref_vec,
                                            true,  // increase_ref_index
                                            output,
                                            output_ctx);
    }

    void offload_streaming_layers() override {
        qwen_image.offload_streaming_layers();
    }
};

struct ZImageModel : public DiffusionModel {
    std::string prefix;
    ZImage::ZImageRunner z_image;

    ZImageModel(ggml_backend_t backend,
                bool offload_params_to_cpu,
                const String2TensorStorage& tensor_storage_map = {},
                const std::string prefix                       = "model.diffusion_model",
                SDVersion version                              = VERSION_Z_IMAGE)
        : prefix(prefix), z_image(backend, offload_params_to_cpu, tensor_storage_map, prefix, version) {
    }

    std::string get_desc() override {
        return z_image.get_desc();
    }

    void alloc_params_buffer() override {
        z_image.alloc_params_buffer();
    }

    void free_params_buffer() override {
        z_image.free_params_buffer();
    }

    void free_compute_buffer() override {
        z_image.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        z_image.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return z_image.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        z_image.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        z_image.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        z_image.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        static const std::vector<sd::Tensor<float>> empty_ref_latents;
        return z_image.compute(n_threads,
                               *diffusion_params.x,
                               *diffusion_params.timesteps,
                               tensor_or_empty(diffusion_params.context),
                               diffusion_params.ref_latents ? *diffusion_params.ref_latents : empty_ref_latents,
                               true);
    }

    // Dynamic tensor offloading
    bool is_params_on_gpu() const override { return z_image.is_params_on_gpu(); }
    bool move_params_to_cpu() override { return z_image.move_params_to_cpu(); }
    bool move_params_to_gpu() override { return z_image.move_params_to_gpu(); }
    size_t get_params_vram_size() const override { return z_image.get_params_vram_size(); }

    // Layer streaming (granular tensor offloading)
    bool supports_layer_streaming() const override { return true; }

    void enable_layer_streaming(int prefetch_layers, size_t min_free_vram) override {
        LayerStreaming::StreamingConfig config;
        config.prefetch_layers = prefetch_layers;
        config.min_free_vram = min_free_vram;
        z_image.enable_layer_streaming(config);
    }

    void disable_layer_streaming() override {
        z_image.disable_layer_streaming();
    }

    bool is_layer_streaming_enabled() const override {
        return z_image.is_streaming_enabled();
    }

    void offload_streaming_layers() override {
        z_image.offload_streaming_layers();
    }

    bool compute_streaming(int n_threads,
                           DiffusionParams diffusion_params,
                           struct ggml_tensor** output     = nullptr,
                           struct ggml_context* output_ctx = nullptr) override {
        StreamingParamConverter cvt;
        auto ref_vec = cvt.convert_vec(diffusion_params.ref_latents);
        return z_image.compute_streaming(n_threads,
                                         cvt.convert(diffusion_params.x),
                                         cvt.convert(diffusion_params.timesteps),
                                         cvt.convert(diffusion_params.context),
                                         ref_vec,
                                         true,  // increase_ref_index
                                         output,
                                         output_ctx);
    }
};

struct ErnieImageModel : public DiffusionModel {
    std::string prefix;
    ErnieImage::ErnieImageRunner ernie_image;

    ErnieImageModel(ggml_backend_t backend,
                    bool offload_params_to_cpu,
                    const String2TensorStorage& tensor_storage_map = {},
                    const std::string prefix                       = "model.diffusion_model")
        : prefix(prefix), ernie_image(backend, offload_params_to_cpu, tensor_storage_map, prefix) {
    }

    std::string get_desc() override {
        return ernie_image.get_desc();
    }

    void alloc_params_buffer() override {
        ernie_image.alloc_params_buffer();
    }

    void free_params_buffer() override {
        ernie_image.free_params_buffer();
    }

    void free_compute_buffer() override {
        ernie_image.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        ernie_image.get_param_tensors(tensors, prefix);
    }

    size_t get_params_buffer_size() override {
        return ernie_image.get_params_buffer_size();
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        ernie_image.set_weight_adapter(adapter);
    }

    int64_t get_adm_in_channels() override {
        return 768;
    }

    void set_flash_attention_enabled(bool enabled) {
        ernie_image.set_flash_attention_enabled(enabled);
    }

    void set_circular_axes(bool circular_x, bool circular_y) override {
        ernie_image.set_circular_axes(circular_x, circular_y);
    }

    sd::Tensor<float> compute(int n_threads,
                              const DiffusionParams& diffusion_params) override {
        GGML_ASSERT(diffusion_params.x != nullptr);
        GGML_ASSERT(diffusion_params.timesteps != nullptr);
        return ernie_image.compute(n_threads,
                                   *diffusion_params.x,
                                   *diffusion_params.timesteps,
                                   tensor_or_empty(diffusion_params.context));
    }
};

#endif
