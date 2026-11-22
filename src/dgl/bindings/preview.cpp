#include "dgl/bindings/preview.h"

#include "dgl/bindings/arguments.h"
#include "dgl/models/vae.h"
#include "dgl/ops/preview.h"

namespace sd::dgl {
    Value make_preview(Values& args, RunState&) {
        auto decoder                                    = args.count("decoder") ? object<models::VAE>(args, "decoder") : std::shared_ptr<models::VAE>{};
        std::shared_ptr<ops::SamplingObserver> observer = std::make_shared<ops::PreviewObserver>(
            std::move(decoder), get_bool(args, "circular_x"), get_bool(args, "circular_y"));
        return Value::object(std::move(observer));
    }
}  // namespace sd::dgl
