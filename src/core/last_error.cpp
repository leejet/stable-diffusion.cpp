#include "core/last_error.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace sd::last_error {

    namespace {

        constexpr size_t kMaxMessageBytes = 480;

        struct ThreadState {
            sd_error_code_t code = SD_ERROR_NONE;
            std::string message;
            std::string first_logged;
            std::string latest_logged;
            uint64_t board_sequence = 0;
        };

        ThreadState& thread_state() {
            thread_local ThreadState state;
            return state;
        }

        // The model loader reads tensors on worker threads, and what they report is not visible
        // to the thread that reports the failure of the call. The board carries the latest error
        // text and the latest recorded code across threads.
        struct Board {
            std::mutex mutex;
            uint64_t sequence = 0;
            std::string logged;
            uint64_t logged_sequence = 0;
            sd_error_code_t code     = SD_ERROR_NONE;
            std::string code_message;
            uint64_t code_sequence = 0;
        };

        Board& board() {
            static Board instance;
            return instance;
        }

        std::string bounded(const std::string& text) {
            if (text.size() <= kMaxMessageBytes) {
                return text;
            }
            return text.substr(0, kMaxMessageBytes - 3) + "...";
        }

        std::string latest_error_text(const ThreadState& state) {
            if (!state.latest_logged.empty()) {
                return state.latest_logged;
            }
            std::lock_guard<std::mutex> lock(board().mutex);
            return board().logged_sequence > state.board_sequence ? board().logged : std::string();
        }

    }  // namespace

    void clear() {
        ThreadState& state = thread_state();
        state.code         = SD_ERROR_NONE;
        state.message.clear();
        state.first_logged.clear();
        state.latest_logged.clear();
        std::lock_guard<std::mutex> lock(board().mutex);
        state.board_sequence = board().sequence;
    }

    void note_log(const std::string& message) {
        const std::string text = bounded(message);
        ThreadState& state     = thread_state();
        if (state.first_logged.empty()) {
            state.first_logged = text;
        }
        state.latest_logged = text;
        std::lock_guard<std::mutex> lock(board().mutex);
        board().logged          = text;
        board().logged_sequence = ++board().sequence;
    }

    void mark(sd_error_code_t code) {
        mark(code, latest_error_text(thread_state()));
    }

    void mark(sd_error_code_t code, const std::string& message) {
        ThreadState& state     = thread_state();
        const std::string text = bounded(message);
        {
            std::lock_guard<std::mutex> lock(board().mutex);
            board().code          = code;
            board().code_message  = text;
            board().code_sequence = ++board().sequence;
        }
        if (state.code != SD_ERROR_NONE) {
            return;
        }
        state.code    = code;
        state.message = text;
    }

    void mark_from_workers(sd_error_code_t fallback) {
        ThreadState& state   = thread_state();
        sd_error_code_t code = fallback;
        std::string text;
        {
            std::lock_guard<std::mutex> lock(board().mutex);
            if (board().code != SD_ERROR_NONE && board().code_sequence > state.board_sequence) {
                code = board().code;
                text = board().code_message;
            } else if (board().logged_sequence > state.board_sequence) {
                text = board().logged;
            }
        }
        mark(code, text.empty() ? latest_error_text(state) : text);
    }

    void ensure(sd_error_code_t fallback) {
        ThreadState& state = thread_state();
        if (state.code != SD_ERROR_NONE) {
            return;
        }
        std::string text = !state.first_logged.empty() ? state.first_logged : latest_error_text(state);
        state.code       = fallback;
        state.message    = text.empty() ? "the operation failed" : text;
    }

    sd_error_code_t get(char* message, size_t message_size) {
        const ThreadState& state = thread_state();
        if (message != nullptr && message_size > 0) {
            const size_t count = std::min(state.message.size(), message_size - 1);
            memcpy(message, state.message.data(), count);
            message[count] = '\0';
        }
        return state.code;
    }

}  // namespace sd::last_error

enum sd_error_code_t sd_get_last_error_code(char* message, size_t message_size) {
    return sd::last_error::get(message, message_size);
}

const char* sd_error_code_name(enum sd_error_code_t code) {
    switch (code) {
        case SD_ERROR_NONE:
            return "none";
        case SD_ERROR_INVALID_ARGUMENT:
            return "invalid_argument";
        case SD_ERROR_MODEL_FILE:
            return "model_file";
        case SD_ERROR_OUT_OF_MEMORY:
            return "out_of_memory";
        case SD_ERROR_BACKEND:
            return "backend";
        case SD_ERROR_BUSY:
            return "busy";
        case SD_ERROR_CANCELLED:
            return "cancelled";
        case SD_ERROR_UNKNOWN:
        default:
            return "unknown";
    }
}
