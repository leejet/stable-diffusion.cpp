#include "conditioning/textual_inversion.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "core/util.h"

namespace sd {
    bool read_textual_inversion(ModelLoader& loader, ggml_context* ctx, int64_t hidden_size, int64_t hidden_size2, ggml_type embd_type, ggml_type embd2_type, ggml_tensor*& embd, ggml_tensor*& embd2) {
        auto on_load = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) {
            if (tensor_storage.ne[0] == hidden_size) {
                embd = ggml_new_tensor_2d(ctx, embd_type, hidden_size, tensor_storage.n_dims > 1 ? tensor_storage.ne[1] : 1);
                if (embd == nullptr) {
                    return false;
                }
                *dst_tensor = embd;
            } else if (hidden_size2 > 0) {
                if (tensor_storage.ne[0] == hidden_size2) {
                    embd2 = ggml_new_tensor_2d(ctx, embd2_type, hidden_size2, tensor_storage.n_dims > 1 ? tensor_storage.ne[1] : 1);
                    if (embd2 == nullptr) {
                        return false;
                    }
                    *dst_tensor = embd2;
                } else {
                    LOG_VERBOSE("embedding skipped, wrong hidden size, got %" PRId64 ", expected %" PRId64 " or %" PRId64, tensor_storage.ne[0], hidden_size, hidden_size2);
                }
            } else {
                LOG_VERBOSE("embedding skipped, wrong hidden size, got %" PRId64 ", expected %" PRId64, tensor_storage.ne[0], hidden_size);
            }
            return true;
        };
        loader.set_n_threads(1);
        return loader.load_tensors(on_load);
    }

    void TextualInversion::load(int64_t hidden_size, int64_t hidden_size2) {
        std::vector<float> values, values2;
        int count = 0;
        dependencies.clear();
        token_ranges.clear();
        for (const auto& source : sources) {
            ModelLoader loader;
            if (!loader.init_from_file_and_convert_name(source.path))
                throw std::runtime_error("cannot read textual inversion '" + source.path + "'");
            ggml_init_params params{100 * 1024 * 1024, nullptr, false};
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
            if (!ctx)
                throw std::runtime_error("cannot allocate textual inversion buffers");
            ggml_tensor* embd  = nullptr;
            ggml_tensor* embd2 = nullptr;
            if (!read_textual_inversion(loader, ctx.get(), hidden_size, hidden_size2,
                                        GGML_TYPE_F32, GGML_TYPE_F32, embd, embd2) ||
                (!embd && !embd2))
                throw std::runtime_error("textual inversion '" + source.path + "' has no usable tensor for this CLIP encoder");
            const int64_t rows  = embd ? embd->ne[1] : 0;
            const int64_t rows2 = embd2 ? embd2->ne[1] : 0;
            const int64_t total = std::max(rows, rows2);
            if (hidden_size2 && rows != rows2)
                LOG_WARN("textual inversion '%s' has unequal CLIP rows, zero-padding", source.path.c_str());
            values.resize((count + total) * hidden_size);
            if (embd)
                memcpy(values.data() + count * hidden_size, embd->data, rows * hidden_size * sizeof(float));
            if (hidden_size2) {
                values2.resize((count + total) * hidden_size2);
                if (embd2)
                    memcpy(values2.data() + count * hidden_size2, embd2->data, rows2 * hidden_size2 * sizeof(float));
            }
            token_ranges.emplace(source.path, std::pair<int, int>{count, count + total});
            count += int(total);
            auto files = loader.dependencies();
            dependencies.insert(dependencies.end(), files.begin(), files.end());
        }
        first = Tensor<float>({hidden_size, count}, std::move(values));
        if (hidden_size2)
            second = Tensor<float>({hidden_size2, count}, std::move(values2));
    }
}
