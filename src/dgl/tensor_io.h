#ifndef __SD_DGL_TENSOR_IO_H__
#define __SD_DGL_TENSOR_IO_H__

#include <string>

#include "tensor.h"

namespace sd::dgl {

    std::shared_ptr<sd_tensor_t> load_npy(const std::string& path, size_t max_bytes);
    void save_npy_atomic(const sd_tensor_t& tensor, const std::string& path);

}  // namespace sd::dgl

#endif  // __SD_DGL_TENSOR_IO_H__
