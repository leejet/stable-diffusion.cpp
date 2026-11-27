#ifndef __SD_MODEL_IO_NPY_IO_H__
#define __SD_MODEL_IO_NPY_IO_H__

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "stable-diffusion.h"

namespace sd::npy {

    class Reader {
        std::ifstream file_;
        std::vector<int64_t> shape_;
        sd_tensor_type_t type_;
        size_t bytes_;
        size_t width_;
        bool fortran_order_;
        bool swap_bytes_;

    public:
        Reader(const std::string& path, size_t max_bytes);

        const std::vector<int64_t>& shape() const { return shape_; }
        sd_tensor_type_t type() const { return type_; }
        void read(void* buffer, size_t bytes);
    };

    void write(const std::filesystem::path& path, sd_tensor_type_t type, const std::vector<int64_t>& shape, const void* data, size_t bytes);

}  // namespace sd::npy

#endif  // __SD_MODEL_IO_NPY_IO_H__
