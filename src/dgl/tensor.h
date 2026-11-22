#ifndef __SD_DGL_TENSOR_H__
#define __SD_DGL_TENSOR_H__

#include <memory>
#include <utility>

#include "api/tensor.h"

namespace sd::dgl {
    inline constexpr size_t tensor_byte_limit = size_t(1024) * 1024 * 1024;
    using sd::api::tensor_dtype;

    template <class F>
    decltype(auto) with_dtype(sd_tensor_type_t dtype, F function) {
        switch (dtype) {
            case SD_TENSOR_U8:
                return function(uint8_t{});
            case SD_TENSOR_I32:
                return function(int32_t{});
            case SD_TENSOR_I64:
                return function(int64_t{});
            case SD_TENSOR_F32:
                return function(float{});
            case SD_TENSOR_F64:
                return function(double{});
            default:
                throw std::runtime_error("invalid tensor dtype");
        }
    }

    template <class... Args>
    std::shared_ptr<sd_tensor_t> make_tensor(Args&&... args) {
        return {new sd_tensor_t(std::forward<Args>(args)...), sd_tensor_free};
    }
}

#endif  // __SD_DGL_TENSOR_H__
