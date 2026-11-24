#ifndef __SD_CORE_PROGRESS_H__
#define __SD_CORE_PROGRESS_H__

#include <cstdint>

void pretty_progress(int step, int steps, float time);
void pretty_bytes_progress(int step, int steps, uint64_t bytes_processed, float elapsed_seconds);

namespace sd {
    class ByteProgressScope {
        using Callback = void (*)(int completed, int total, void* data);
        Callback previous_callback_;
        void* previous_data_;

    public:
        ByteProgressScope(Callback callback, void* data);
        ~ByteProgressScope();
        ByteProgressScope(const ByteProgressScope&)            = delete;
        ByteProgressScope& operator=(const ByteProgressScope&) = delete;
    };

    class ByteProgressConsoleScope {
        bool previous_;

    public:
        explicit ByteProgressConsoleScope(bool enabled = true);
        ~ByteProgressConsoleScope();
        ByteProgressConsoleScope(const ByteProgressConsoleScope&)            = delete;
        ByteProgressConsoleScope& operator=(const ByteProgressConsoleScope&) = delete;
    };
}

#endif
