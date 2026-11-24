#include "dgl/bindings/relay_vsr.h"

#include "dgl/bindings/arguments.h"
#include "dgl/bindings/values.h"
#include "dgl/models/diffusion_model.h"
#include "dgl/models/dit.h"
#include "dgl/models/vae.h"
#include "dgl/ops/relay_vsr.h"

namespace sd::dgl {
    template <typename T>
    static T& relay_component(T& engine) {
        if (engine.version() != VERSION_RELAY_VSR)
            throw std::runtime_error("expected a RelayVSR component");
        return engine;
    }

    Value positive_embedding(Values& args, RunState&) {
        auto& engine = relay_component(*object<models::DiffusionModel>(args, "dit")->dit);
        Model::ExecutionScope scope(engine);
        scope.require_ready();
        return tensor_value(sd::dgl::ops::relay_vsr_positive_embedding(*engine.diffusion_model, *engine.model_weights));
    }

    Value relay_latent_shape(Values& args, RunState&) {
        auto& engine = relay_component(*object<models::DiffusionModel>(args, "dit")->dit);
        Model::ExecutionScope scope(engine);
        scope.require_ready();
        const auto frames = object<sd_tensor_t>(args, "keyframes");
        const auto& shape = frames->shape();
        if (frames->type != SD_TENSOR_U8 || shape.size() != 4 || shape[0] != 3 || !frames->bytes || shape[1] > 16384 || shape[2] > 16384)
            throw std::runtime_error("RelayVSR keyframes require nonempty U8 [3, width, height, time]");
        return shape_value(sd::dgl::ops::relay_vsr_latent_shape(*engine.diffusion_model, shape[1], shape[2], shape[3]));
    }

    Value relay_decode(Values& args, RunState&) {
        auto& engine = relay_component(*object<VAE>(args, "flash"));
        Model::ExecutionScope scope(engine);
        scope.require_ready();
        const auto& frames = args.at("frames");
        const auto* raw    = frames.get_if<sd_tensor_t>();
        if (!raw || raw->shape().size() != 4 || !raw->bytes || raw->shape()[1] > 16384 || raw->shape()[2] > 16384)
            throw std::runtime_error("RelayVSR frames require nonempty U8 [3, width, height, time]");
        auto output = sd::dgl::ops::relay_vsr_decode(
            *engine.first_stage_model, engine.n_threads, float_tensor(args, "latent"),
            read_tensor<uint8_t>(frames), read_tensor<int64_t>(args.at("keyframe_indices")),
            float(get_number(args, "detail")));
        return Value::object(make_tensor(std::move(output)));
    }

}  // namespace sd::dgl
