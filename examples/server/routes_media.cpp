#include "routes.h"

#include <array>
#include <limits>
#include <string_view>
#include <utility>

#ifdef _WIN32
#include <winternl.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#endif

namespace media_view {
    namespace fs = std::filesystem;

#ifdef _WIN32
    class FileHandle {
        HANDLE handle_;

    public:
        explicit FileHandle(HANDLE handle = INVALID_HANDLE_VALUE) : handle_(handle) {}
        ~FileHandle() {
            if (*this)
                CloseHandle(handle_);
        }
        FileHandle(FileHandle&& other) noexcept : handle_(std::exchange(other.handle_, INVALID_HANDLE_VALUE)) {}
        FileHandle& operator=(FileHandle&& other) noexcept {
            std::swap(handle_, other.handle_);
            return *this;
        }
        explicit operator bool() const { return handle_ != INVALID_HANDLE_VALUE; }
        HANDLE get() const { return handle_; }
    };

    static FileHandle open_directory(const fs::path& path) {
        return FileHandle(CreateFileW(path.c_str(), FILE_TRAVERSE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                      OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr));
    }

    static FileHandle open_child(const FileHandle& parent, const fs::path& part, bool directory) {
        static const auto open_file = reinterpret_cast<decltype(&NtOpenFile)>(
            GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtOpenFile"));
        if (!open_file)
            return FileHandle();
        auto name        = part.native();
        const auto bytes = static_cast<USHORT>(name.size() * sizeof(wchar_t));
        UNICODE_STRING relative{bytes, bytes, name.data()};
        OBJECT_ATTRIBUTES object;
        InitializeObjectAttributes(&object, &relative, OBJ_CASE_INSENSITIVE, parent.get(), nullptr);
        IO_STATUS_BLOCK status{};
        HANDLE handle     = INVALID_HANDLE_VALUE;
        const auto access = (directory ? FILE_TRAVERSE : FILE_READ_DATA) | FILE_READ_ATTRIBUTES | SYNCHRONIZE;
        // A handle-relative open keeps every component under the directory already checked.
        if (open_file(&handle, access, &object, &status, FILE_SHARE_READ | FILE_SHARE_WRITE,
                      FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT) < 0)
            return FileHandle();
        FileHandle file(handle);
        BY_HANDLE_FILE_INFORMATION info{};
        if (GetFileType(handle) != FILE_TYPE_DISK || !GetFileInformationByHandle(handle, &info) ||
            (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) ||
            static_cast<bool>(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != directory ||
            (!directory && info.nNumberOfLinks != 1))
            return FileHandle();
        return file;
    }

    static uint64_t file_size(const FileHandle& file) {
        LARGE_INTEGER size{};
        return GetFileSizeEx(file.get(), &size) && size.QuadPart > 0 ? static_cast<uint64_t>(size.QuadPart) : 0;
    }

    static size_t read_at(const FileHandle& file, size_t offset, char* buffer, size_t length) {
        LARGE_INTEGER position{};
        position.QuadPart = static_cast<LONGLONG>(offset);
        DWORD count       = 0;
        if (!SetFilePointerEx(file.get(), position, nullptr, FILE_BEGIN) ||
            !ReadFile(file.get(), buffer, static_cast<DWORD>(length), &count, nullptr))
            return 0;
        return count;
    }
#else
    class FileHandle {
        int fd_;

    public:
        explicit FileHandle(int fd = -1) : fd_(fd) {}
        ~FileHandle() {
            if (*this)
                close(fd_);
        }
        FileHandle(FileHandle&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
        FileHandle& operator=(FileHandle&& other) noexcept {
            std::swap(fd_, other.fd_);
            return *this;
        }
        explicit operator bool() const { return fd_ >= 0; }
        int get() const { return fd_; }
    };

    static FileHandle open_directory(const fs::path& path) {
        return FileHandle(open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    }

    static FileHandle open_child(const FileHandle& parent, const fs::path& part, bool directory) {
        const auto flags = O_RDONLY | O_NOFOLLOW | O_CLOEXEC | (directory ? O_DIRECTORY : O_NONBLOCK);
        return FileHandle(openat(parent.get(), part.c_str(), flags));
    }

    static uint64_t file_size(const FileHandle& file) {
        struct stat info{};
        if (fstat(file.get(), &info) != 0 || !S_ISREG(info.st_mode) || info.st_nlink != 1 || info.st_size <= 0)
            return 0;
        return static_cast<uint64_t>(info.st_size);
    }

    static size_t read_at(const FileHandle& file, size_t offset, char* buffer, size_t length) {
        ssize_t count;
        do {
            count = pread(file.get(), buffer, length, static_cast<off_t>(offset));
        } while (count < 0 && errno == EINTR);
        return count > 0 ? static_cast<size_t>(count) : 0;
    }
#endif

    static bool valid_component(std::string_view part) {
        return !part.empty() && part.size() <= 1020 && part != "." && part != ".." &&
               part.back() != '.' && part.back() != ' ' && part.find_first_of("\\:*?\"<>|") == std::string_view::npos &&
               std::none_of(part.begin(), part.end(), [](unsigned char c) { return c < 32 || c == 127; });
    }

    static FileHandle open_output_root(const fs::path& path) {
        auto root           = path.root_path();
        const auto relative = path.relative_path();
        auto part           = relative.begin();
#ifdef _WIN32
        // A UNC path needs the share name before it can be opened as a directory.
        if (path.root_name().native().compare(0, 2, L"\\\\") == 0 && part != relative.end()) {
            root /= *part;
            ++part;
        }
#endif
        auto directory = open_directory(root);
        for (; directory && part != relative.end(); ++part)
            directory = open_child(directory, *part, true);
        return directory;
    }

    static FileHandle open_media(const fs::path& output_root, const std::string& root, std::string_view path) {
        FileHandle parent;
        if (root == "output") {
            parent = open_output_root(output_root);
        } else if (root == "input" || root == "assets") {
            const auto cwd = open_directory(".");
            if (cwd)
                parent = open_child(cwd, root, true);
        }
        while (parent) {
            const auto slash = path.find('/');
            const auto part  = path.substr(0, slash);
            if (!valid_component(part))
                break;
            const bool directory = slash != std::string_view::npos;
            auto child           = open_child(parent, fs::u8path(part), directory);
            if (!directory)
                return child;
            parent = std::move(child);
            path.remove_prefix(slash + 1);
        }
        return FileHandle();
    }

    static size_t ebml_vint_width(std::string_view bytes, size_t max_width) {
        if (bytes.empty())
            return 0;
        const auto first = static_cast<unsigned char>(bytes.front());
        for (size_t width = 1; width <= std::min(max_width, bytes.size()); ++width) {
            if (first & (0x80u >> (width - 1)))
                return width;
        }
        return 0;
    }

    static bool read_ebml_size(std::string_view& bytes, uint64_t& size) {
        const auto width = ebml_vint_width(bytes, 8);
        if (width == 0)
            return false;
        size = static_cast<unsigned char>(bytes.front()) & (0xffu >> width);
        for (size_t i = 1; i < width; ++i)
            size = (size << 8) | static_cast<unsigned char>(bytes[i]);
        bytes.remove_prefix(width);
        return size != (uint64_t{1} << (7 * width)) - 1;
    }

    static bool has_webm_doctype(std::string_view header, uint64_t file_size) {
        constexpr std::string_view ebml_id = "\x1a\x45\xdf\xa3";
        if (header.substr(0, ebml_id.size()) != ebml_id)
            return false;
        const auto probe_size = header.size();
        header.remove_prefix(ebml_id.size());
        uint64_t size;
        if (!read_ebml_size(header, size))
            return false;
        const auto prefix_size = probe_size - header.size();
        if (prefix_size > file_size || size > file_size - prefix_size)
            return false;
        header = header.substr(0, static_cast<size_t>(size));
        while (!header.empty()) {
            const auto width = ebml_vint_width(header, 4);
            if (width == 0)
                return false;
            const bool is_doctype = header.substr(0, width) == "\x42\x82";
            header.remove_prefix(width);
            if (!read_ebml_size(header, size) || size > header.size())
                return false;
            if (is_doctype)
                return header.substr(0, static_cast<size_t>(size)) == "webm";
            header.remove_prefix(static_cast<size_t>(size));
        }
        return false;
    }

    static const char* media_type(std::string extension, std::string_view header, uint64_t size) {
        for (auto& ch : extension) {
            if (ch >= 'A' && ch <= 'Z')
                ch += 'a' - 'A';
        }
        if (extension == ".png" && header.substr(0, 8) == "\x89PNG\r\n\x1a\n")
            return "image/png";
        if ((extension == ".jpg" || extension == ".jpeg") && header.substr(0, 3) == "\xff\xd8\xff")
            return "image/jpeg";
        if (extension == ".gif" && (header.substr(0, 6) == "GIF87a" || header.substr(0, 6) == "GIF89a"))
            return "image/gif";
        if (extension == ".webp" && header.size() >= 16 && header.substr(0, 4) == "RIFF" && header.substr(8, 7) == "WEBPVP8" &&
            (header[15] == ' ' || header[15] == 'L' || header[15] == 'X'))
            return "image/webp";
        if (extension == ".webm" && has_webm_doctype(header, size))
            return "video/webm";
        if (extension == ".avi" && header.size() >= 12 && header.substr(0, 4) == "RIFF" && header.substr(8, 4) == "AVI ")
            return "video/x-msvideo";
        if (extension == ".mp4" && header.size() >= 16 && header.substr(4, 4) == "ftyp") {
            uint32_t box_size = 0;
            for (size_t i = 0; i < 4; ++i)
                box_size = (box_size << 8) | static_cast<unsigned char>(header[i]);
            if (box_size >= 16 && box_size <= size)
                return "video/mp4";
        }
        return nullptr;
    }

    static bool set_media_content(const fs::path& output_root,
                                  const std::string& root,
                                  const std::string& path,
                                  httplib::Response& res) {
        auto file = open_media(output_root, root, path);
        if (!file)
            return false;
        const auto size = file_size(file);
        if (size == 0 || size > std::numeric_limits<size_t>::max())
            return false;
        std::array<char, 256> header;
        const auto count     = read_at(file, 0, header.data(), header.size());
        const auto extension = path.find_last_of('.');
        if (extension == std::string::npos)
            return false;
        const auto* type = media_type(path.substr(extension), {header.data(), count}, size);
        if (!type)
            return false;
        res.set_header("Accept-Ranges", "bytes");
        res.set_content_provider(static_cast<size_t>(size), type,
                                 [file = std::make_shared<FileHandle>(std::move(file))](size_t offset, size_t length, httplib::DataSink& sink) {
                                     std::array<char, 64 * 1024> buffer;
                                     const auto count = read_at(*file, offset, buffer.data(), std::min(length, buffer.size()));
                                     return count != 0 && sink.write(buffer.data(), count);
                                 });
        return true;
    }
}

void register_media_view_endpoint(httplib::Server& svr, bool enabled, const std::filesystem::path& output_root) {
    svr.Get("/sdcpp/v1/view", [enabled, output_root](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Cache-Control", "no-store");
        res.set_header("X-Content-Type-Options", "nosniff");
        if (!enabled) {
            res.status = 403;
            return;
        }
        try {
            if (req.get_param_value_count("root") == 1 && req.get_param_value_count("path") == 1 &&
                media_view::set_media_content(output_root, req.get_param_value("root"), req.get_param_value("path"), res))
                return;
        } catch (const std::system_error&) {
            // Invalid path encodings are treated like unavailable files.
        }
        res.status = 404;
    });
}
