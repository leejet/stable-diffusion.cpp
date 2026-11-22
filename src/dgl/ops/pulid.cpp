#include "dgl/ops/pulid.h"

#include "dgl/models/fwd.h"
#include "dgl/models/pulid.h"
#include "model/adapter/pulid.hpp"

namespace sd::dgl::ops {
    PuLIDAttention::PuLIDAttention(const std::vector<const PuLIDBinding*>& bindings, PuLIDCache& cache) {
        adapters_.reserve(bindings.size());
        for (const auto* binding : bindings) {
            auto& projections = cache[binding->model.get()];
            auto entry        = projections.find(binding->embedding.get());
            if (entry == projections.end()) {
                auto kv = binding->model->compute_kv(*binding->embedding);
                if (kv.empty())
                    throw std::runtime_error("PuLID key/value projection failed");
                entry = projections.emplace(binding->embedding.get(), std::move(kv)).first;
            }
            const auto existing = std::find_if(adapters_.begin(), adapters_.end(), [&](const auto& adapter) {
                return adapter.kv == &entry->second;
            });
            if (existing == adapters_.end())
                adapters_.push_back({binding->model.get(), &entry->second, binding->strength});
            else
                existing->strength += binding->strength;
        }
    }

    bool PuLIDAttention::applies(bool single_stream, int block) const {
        return !adapters_.empty() && block % (single_stream ? PuLIDRunner::single_interval : PuLIDRunner::double_interval) == 0;
    }

    ggml_tensor* PuLIDAttention::apply(GGMLRunnerContext& ctx, bool single_stream, int block, ggml_tensor* image) const {
        const size_t layer = single_stream ? PuLIDRunner::double_layers + block / PuLIDRunner::single_interval
                                           : block / PuLIDRunner::double_interval;
        auto* output       = image;
        for (const auto& adapter : adapters_) {
            auto* delta = adapter.model->attend(ctx, layer, adapter.kv->at(layer), image);
            output      = ggml_add(ctx.ggml_ctx, output, ggml_scale(ctx.ggml_ctx, delta, adapter.strength));
        }
        return output;
    }

    size_t PuLIDAttention::extra_graph_size() const {
        size_t size = 0;
        for (const auto& adapter : adapters_)
            size += adapter.model->extra_graph_size();
        return size;
    }
}
