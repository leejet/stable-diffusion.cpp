#ifndef __SD_MODEL_IO_GGUF_IO_H__
#define __SD_MODEL_IO_GGUF_IO_H__

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "tensor_storage.h"

struct ggml_context;
struct gguf_context;

bool is_gguf_file(const std::string& file_path);
bool read_gguf_file(const std::string& file_path,
                    std::vector<TensorStorage>& tensor_storages,
                    std::string* error = nullptr);
bool write_gguf_file(const std::string& file_path,
                     const std::vector<TensorWriteInfo>& tensors,
                     std::string* error = nullptr);

class GGUFStreamingWriter {
public:
    GGUFStreamingWriter() = default;
    ~GGUFStreamingWriter();

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
    ggml_context* meta_ctx_ = nullptr;
    gguf_context* gguf_ctx_ = nullptr;
};

#endif  // __SD_MODEL_IO_GGUF_IO_H__
