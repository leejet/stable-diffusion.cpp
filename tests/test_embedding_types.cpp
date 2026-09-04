#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>

#include "core/ggml_extend.hpp"

static bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

int main() {
    bool ok = true;

    constexpr std::array<ggml_type, 5> k_quants = {
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
    };
    for (ggml_type type : k_quants) {
        ok = expect(support_get_rows(type),
                    std::string("K-quant should support GET_ROWS: ") + ggml_type_name(type)) &&
             ok;
    }

    constexpr std::array<ggml_type, 6> existing_types = {
        GGML_TYPE_F16,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q4_0,
    };
    for (ggml_type type : existing_types) {
        ok = expect(support_get_rows(type),
                    std::string("existing GET_ROWS type regressed: ") + ggml_type_name(type)) &&
             ok;
    }

    // Exercise the allocation path that was regressing: a K-quant storage
    // entry must remain quantized rather than silently becoming F32.
    for (ggml_type type : k_quants) {
        ggml_init_params init_params = {};
        init_params.mem_size         = 16 * ggml_tensor_overhead();
        init_params.mem_buffer       = nullptr;
        init_params.no_alloc         = true;
        ggml_context* ctx            = ggml_init(init_params);
        ok                           = expect(ctx != nullptr, "failed to create no-alloc GGML context") && ok;
        if (ctx == nullptr) {
            continue;
        }

        int64_t shape[2] = {256, 4};
        String2TensorStorage storage;
        storage.emplace("weight", TensorStorage("weight", type, shape, 2, 0));

        Embedding embedding(/*num_embeddings=*/shape[1], /*embedding_dim=*/shape[0]);
        embedding.init(ctx, storage);
        std::map<std::string, ggml_tensor*> tensors;
        embedding.get_param_tensors(tensors);
        const auto it = tensors.find("weight");
        ok            = expect(it != tensors.end(), "embedding weight was not registered") && ok;
        if (it != tensors.end()) {
            ok = expect(it->second->type == type,
                        std::string("embedding allocation changed ") + ggml_type_name(type) + " to " +
                            ggml_type_name(it->second->type)) &&
                 ok;
            ok = expect(ggml_nbytes(it->second) < static_cast<size_t>(shape[0] * shape[1] * sizeof(float)),
                        std::string("K-quant embedding should use less memory than F32: ") + ggml_type_name(type)) &&
                 ok;
        }
        ggml_free(ctx);
    }

    return ok ? 0 : 1;
}
