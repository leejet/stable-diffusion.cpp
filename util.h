#ifndef __UTIL_H__
#define __UTIL_H__

#include <cstdint>
#include <functional>
#include <string>

bool ends_with(const std::string& str, const std::string& ending);
bool starts_with(const std::string& str, const std::string& start);

std::string format(const char* fmt, ...);

void replace_all_chars(std::string& str, char target, char replacement);

bool file_exists(const std::string& filename);
bool is_directory(const std::string& path);

std::u32string utf8_to_utf32(const std::string& utf8_str);
std::string utf32_to_utf8(const std::u32string& utf32_str);
std::u32string unicode_value_to_utf32(int unicode_value);

std::string basename(const std::string& path);

std::string path_join(const std::string& p1, const std::string& p2);

int32_t get_num_physical_cores();

enum SDLogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

void set_sd_log_level(SDLogLevel level);

void log_printf(SDLogLevel level, bool enable_log_tag, const char* file, int line, const char* format, ...);

typedef std::function<void(SDLogLevel level, const char* text)> sd_logger_function_t;

void set_sd_logger(const sd_logger_function_t& sd_logger_function);

#define LOG_DEFAULT(format, ...) log_printf(SDLogLevel::INFO, false, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_DEBUG(format, ...) log_printf(SDLogLevel::DEBUG, true, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) log_printf(SDLogLevel::INFO, true, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) log_printf(SDLogLevel::WARN, true, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) log_printf(SDLogLevel::ERROR, true, __FILE__, __LINE__, format, ##__VA_ARGS__)
#endif  // __UTIL_H__
