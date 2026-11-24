#ifndef __SD_CORE_FILE_ACCESS_H__
#define __SD_CORE_FILE_ACCESS_H__

#include <stdexcept>
#include <string>

#include "stable-diffusion.h"

namespace sd {
    struct FileAccessPolicy {
        sd_dgl_file_access_cb_t callback = nullptr;
        void* userdata                   = nullptr;
    };

    inline thread_local FileAccessPolicy active_file_access;

    class FileAccessScope {
        FileAccessPolicy previous_;

    public:
        explicit FileAccessScope(FileAccessPolicy policy)
            : previous_(active_file_access) { active_file_access = policy; }
        ~FileAccessScope() { active_file_access = previous_; }
        FileAccessScope(const FileAccessScope&)            = delete;
        FileAccessScope& operator=(const FileAccessScope&) = delete;
    };

    class FileAccessDenied : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    inline void require_file_access(const std::string& path, sd_dgl_file_access_t access) {
        const auto& policy = active_file_access;
        if (policy.callback &&
            (path.find('\0') != std::string::npos || !policy.callback(path.c_str(), access, policy.userdata))) {
            const char* action = access == SD_DGL_FILE_WRITE ? "write" : access == SD_DGL_FILE_READ_MODEL ? "model read"
                                                                                                          : "read";
            throw FileAccessDenied(std::string("file ") + action + " denied for '" + path + "'");
        }
    }
}

#endif  // __SD_CORE_FILE_ACCESS_H__
