#ifndef __SD_MODEL_IO_SAFETENSORS_IO_H__
#define __SD_MODEL_IO_SAFETENSORS_IO_H__

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "tensor_storage.h"

bool is_safetensors_file(const std::string& file_path);
bool read_safetensors_file(const std::string& file_path,
                           std::vector<TensorStorage>& tensor_storages,
                           std::string* error = nullptr);
bool write_safetensors_file(const std::string& file_path,
                            const std::vector<TensorWriteInfo>& tensors,
                            std::string* error = nullptr);

class SafetensorsStreamingWriter {
public:
    SafetensorsStreamingWriter() = default;
    ~SafetensorsStreamingWriter();

    bool open(const std::string& file_path,
              const std::vector<TensorWritePlan>& tensors,
              int n_writers,
              std::string* error = nullptr);
    bool write_tensor(size_t tensor_index,
                      const uint8_t* data,
                      size_t size,
                      int writer_index,
                      std::string* error = nullptr);
    void close();

private:
    std::string file_path_;
    std::vector<TensorWritePlan> tensors_;
    std::vector<uint64_t> tensor_offsets_;
    std::vector<FILE*> files_;
    uint64_t data_start_ = 0;
};

#endif  // __SD_MODEL_IO_SAFETENSORS_IO_H__
