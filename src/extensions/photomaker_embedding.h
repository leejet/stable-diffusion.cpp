#ifndef __SD_EXTENSIONS_PHOTOMAKER_EMBEDDING_H__
#define __SD_EXTENSIONS_PHOTOMAKER_EMBEDDING_H__

#include "core/tensor.hpp"

namespace sd::photomaker {

    inline constexpr int64_t id_embedding_width = 512;

    Tensor<float> load_id_embedding(const std::string& path, int image_count);

}  // namespace sd::photomaker

#endif  // __SD_EXTENSIONS_PHOTOMAKER_EMBEDDING_H__
