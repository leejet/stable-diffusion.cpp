#ifndef __UTIL_H__
#define __UTIL_H__

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

void log_printf(SDLogLevel level, const char* file, int line, const char* format, ...);

#define LOG_DEBUG(format, ...) log_printf(SDLogLevel::DEBUG, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) log_printf(SDLogLevel::INFO, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) log_printf(SDLogLevel::WARN, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) log_printf(SDLogLevel::ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__)
#endif  // __UTIL_H__