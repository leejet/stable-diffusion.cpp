#include "util.h"

#include <stdarg.h>
#include <codecvt>
#include <fstream>
#include <locale>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#if !defined(_WIN32)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include "ggml/ggml.h"
#include "stable-diffusion.h"

bool ends_with(const std::string& str, const std::string& ending) {
    if (str.length() >= ending.length()) {
        return (str.compare(str.length() - ending.length(), ending.length(), ending) == 0);
    } else {
        return false;
    }
}

bool starts_with(const std::string& str, const std::string& start) {
    if (str.find(start) == 0) {
        return true;
    }
    return false;
}

void replace_all_chars(std::string& str, char target, char replacement) {
    for (size_t i = 0; i < str.length(); ++i) {
        if (str[i] == target) {
            str[i] = replacement;
        }
    }
}

std::string format(const char* fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

#ifdef _WIN32  // code for windows
#include <windows.h>

bool file_exists(const std::string& filename) {
    DWORD attributes = GetFileAttributesA(filename.c_str());
    return (attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY));
}

bool is_directory(const std::string& path) {
    DWORD attributes = GetFileAttributesA(path.c_str());
    return (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY));
}

#else  // Unix
#include <dirent.h>
#include <sys/stat.h>

bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

bool is_directory(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

#endif

// get_num_physical_cores is copy from
// https://github.com/ggerganov/llama.cpp/blob/master/examples/common.cpp
// LICENSE: https://github.com/ggerganov/llama.cpp/blob/master/LICENSE
int32_t get_num_physical_cores() {
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu = 0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu" + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break;  // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (siblings.size() > 0) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32)
    // TODO: Implement
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

std::u32string utf8_to_utf32(const std::string& utf8_str) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.from_bytes(utf8_str);
}

std::string utf32_to_utf8(const std::u32string& utf32_str) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.to_bytes(utf32_str);
}

std::u32string unicode_value_to_utf32(int unicode_value) {
    std::u32string utf32_string = {static_cast<char32_t>(unicode_value)};
    return utf32_string;
}

std::string sd_basename(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    pos = path.find_last_of('\\');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

std::string path_join(const std::string& p1, const std::string& p2) {
    if (p1.empty()) {
        return p2;
    }

    if (p2.empty()) {
        return p1;
    }

    if (p1[p1.length() - 1] == '/' || p1[p1.length() - 1] == '\\') {
        return p1 + p2;
    }

    return p1 + "/" + p2;
}

void pretty_progress(int step, int steps, float time) {
    std::string progress = "  |";
    int max_progress     = 50;
    int32_t current      = (int32_t)(step * 1.f * max_progress / steps);
    for (int i = 0; i < 50; i++) {
        if (i > current) {
            progress += " ";
        } else if (i == current && i != max_progress - 1) {
            progress += ">";
        } else {
            progress += "=";
        }
    }
    progress += "|";
    printf(time > 1.0f ? "\r%s %i/%i - %.2fs/it" : "\r%s %i/%i - %.2fit/s",
           progress.c_str(), step, steps,
           time > 1.0f || time == 0 ? time : (1.0f / time));
    fflush(stdout);  // for linux
    if (step == steps) {
        printf("\n");
    }
}

static sd_log_cb_t sd_log_cb = NULL;
void* sd_log_cb_data         = NULL;

#define LOG_BUFFER_SIZE 1024

void log_printf(sd_log_level_t level, const char* file, int line, const char* format, ...) {
    va_list args;
    va_start(args, format);

    const char* level_str = "DEBUG";
    if (level == SD_LOG_INFO) {
        level_str = "INFO ";
    } else if (level == SD_LOG_WARN) {
        level_str = "WARN ";
    } else if (level == SD_LOG_ERROR) {
        level_str = "ERROR";
    }

    static char log_buffer[LOG_BUFFER_SIZE];

    int written = snprintf(log_buffer, LOG_BUFFER_SIZE, "[%s] %s:%-4d - ", level_str, sd_basename(file).c_str(), line);

    if (written >= 0 && written < LOG_BUFFER_SIZE) {
        vsnprintf(log_buffer + written, LOG_BUFFER_SIZE - written, format, args);
        strncat(log_buffer, "\n", LOG_BUFFER_SIZE - strlen(log_buffer) - 1);
    }

    if (sd_log_cb) {
        sd_log_cb(level, log_buffer, sd_log_cb_data);
    }

    va_end(args);
}

void sd_set_log_callback(sd_log_cb_t cb, void* data) {
    sd_log_cb      = cb;
    sd_log_cb_data = data;
}

const char* sd_get_system_info() {
    static char buffer[1024];
    std::stringstream ss;
    ss << "System Info: \n";
    ss << "    BLAS = " << ggml_cpu_has_blas() << std::endl;
    ss << "    SSE3 = " << ggml_cpu_has_sse3() << std::endl;
    ss << "    AVX = " << ggml_cpu_has_avx() << std::endl;
    ss << "    AVX2 = " << ggml_cpu_has_avx2() << std::endl;
    ss << "    AVX512 = " << ggml_cpu_has_avx512() << std::endl;
    ss << "    AVX512_VBMI = " << ggml_cpu_has_avx512_vbmi() << std::endl;
    ss << "    AVX512_VNNI = " << ggml_cpu_has_avx512_vnni() << std::endl;
    ss << "    FMA = " << ggml_cpu_has_fma() << std::endl;
    ss << "    NEON = " << ggml_cpu_has_neon() << std::endl;
    ss << "    ARM_FMA = " << ggml_cpu_has_arm_fma() << std::endl;
    ss << "    F16C = " << ggml_cpu_has_f16c() << std::endl;
    ss << "    FP16_VA = " << ggml_cpu_has_fp16_va() << std::endl;
    ss << "    WASM_SIMD = " << ggml_cpu_has_wasm_simd() << std::endl;
    ss << "    VSX = " << ggml_cpu_has_vsx() << std::endl;
    snprintf(buffer, sizeof(buffer), "%s", ss.str().c_str());
    return buffer;
}

const char* sd_type_name(enum sd_type_t type) {
    return ggml_type_name((ggml_type)type);
}
