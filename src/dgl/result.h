#ifndef __SD_DGL_RESULT_H__
#define __SD_DGL_RESULT_H__

#include <mutex>

#include "stable-diffusion.h"
#include "value.h"

struct sd_dgl_value_t {
    sd_dgl_value_kind_t kind;
    std::string name;
    nlohmann::json scalar;
    std::shared_ptr<sd_tensor_t> tensor;
    std::vector<sd_dgl_value_t> children;
    std::shared_ptr<const sd::dgl::Bytes> bytes;
};

struct sd_dgl_result_t {
    using Tensors = std::map<std::string, std::shared_ptr<sd_tensor_t>, std::less<>>;

    std::string type;
    sd_dgl_value_t root;

    explicit sd_dgl_result_t(sd::dgl::Value value);
    const Tensors& flat_tensors() const;

private:
    mutable std::once_flag flat_init_;
    mutable Tensors flat_;
};

#endif  // __SD_DGL_RESULT_H__
