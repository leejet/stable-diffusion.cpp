#ifndef __SD_CORE_LAST_ERROR_H__
#define __SD_CORE_LAST_ERROR_H__

#include <string>

#include "core/util.h"
#include "stable-diffusion.h"

// Why the latest API call of a thread failed (sd_get_last_error_code). The code is recorded where
// the cause is known; the text comes from the error log line written there.
namespace sd::last_error {

    // Forgets the error of the previous call made by this thread.
    void clear();

    // Remembers the text of an error-level log line. Called by the logger, from any thread.
    void note_log(const std::string& message);

    // Records the failure of the current call, with the latest error log line as its text.
    // Only the first failure after clear() is kept: what follows is usually its consequence.
    void mark(sd_error_code_t code);
    void mark(sd_error_code_t code, const std::string& message);

    // Records the failure of worker threads that this thread has joined: the code they recorded
    // since clear(), or fallback when they recorded none.
    void mark_from_workers(sd_error_code_t fallback);

    // For a call that is about to return a failure: keeps what was recorded, or records
    // fallback with the first error log line written since clear().
    void ensure(sd_error_code_t fallback);

    // The code of the latest failure of this thread and, when message is not null, its text
    // (truncated to message_size - 1 bytes and terminated).
    sd_error_code_t get(char* message, size_t message_size);

}  // namespace sd::last_error

#define LOG_ERROR_CODE(code, format, ...) \
    do {                                  \
        LOG_ERROR(format, ##__VA_ARGS__); \
        ::sd::last_error::mark(code);     \
    } while (0)

#endif  // __SD_CORE_LAST_ERROR_H__
