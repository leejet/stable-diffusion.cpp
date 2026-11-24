#include "tensor_io.h"

#include <atomic>
#include <chrono>
#include <filesystem>

#include "core/file_access.h"
#include "model_io/npy_io.h"

namespace sd::dgl {

    std::shared_ptr<sd_tensor_t> load_npy(const std::string& path, size_t max_bytes) {
        require_file_access(path, SD_DGL_FILE_READ);
        npy::Reader reader(path, max_bytes);
        auto tensor = make_tensor(reader.type(), reader.shape());
        reader.read(tensor->mutable_data(), tensor->bytes);
        return tensor;
    }

    class TemporaryNpyFile {
        std::filesystem::path directory_;
        std::filesystem::path path_;

    public:
        explicit TemporaryNpyFile(const std::filesystem::path& parent) {
            static std::atomic<uint64_t> sequence{0};
            const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
            const auto prefix    = ".dgl-tensor-" + std::to_string(timestamp) + "-";
            do {
                directory_ = parent / (prefix + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
                path_      = directory_ / "tensor.npy";
            } while (!std::filesystem::create_directory(directory_));
        }

        TemporaryNpyFile(const TemporaryNpyFile&)            = delete;
        TemporaryNpyFile& operator=(const TemporaryNpyFile&) = delete;

        const std::filesystem::path& path() const { return path_; }

        ~TemporaryNpyFile() {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
            std::filesystem::remove(directory_, ignored);
        }
    };

    void save_npy_atomic(const sd_tensor_t& tensor, const std::string& path) {
        if (path.empty())
            throw std::runtime_error("tensor output path is required");
        require_file_access(path, SD_DGL_FILE_WRITE);
        const auto destination = std::filesystem::u8path(path);
        const TemporaryNpyFile temporary(destination.parent_path());
        npy::write(temporary.path(), tensor.type, tensor.shape(), tensor.data(), tensor.bytes);
        std::filesystem::rename(temporary.path(), destination);
    }

}  // namespace sd::dgl
