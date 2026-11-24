// Robustness test for the public C API (device enumeration, param defaults, error reporting,
// system memory).
// Build (from the repository root, after building the static library):
//   clang++ -std=c++17 -g -fsanitize=address,undefined -Iinclude test/api_robustness.cpp \
//       build/libstable-diffusion.a build/ggml/src/libggml.a build/ggml/src/libggml-cpu.a \
//       build/ggml/src/libggml-base.a <platform libs> -o api_robustness
// Run: ./api_robustness ; exit code 0 = all checks passed.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <vector>

#include "stable-diffusion.h"

static int failures = 0;
#define CHECK(cond, msg)                  \
    do {                                  \
        if (!(cond)) {                    \
            printf("FAIL: %s\n", msg);    \
            failures++;                   \
        } else {                          \
            printf("ok:   %s\n", msg);    \
        }                                 \
    } while (0)

int main() {
    CHECK(!sd_get_device_info(0, nullptr), "info(null out) rejected");
    sd_device_info_t info;
    CHECK(!sd_get_device_info((size_t)-1, &info), "info(SIZE_MAX index) rejected");
    CHECK(!sd_get_device_info(sd_get_device_count(), &info), "info(index == count) rejected");

    const size_t n = sd_get_device_count();
    CHECK(n >= 1, "at least one device (CPU)");
    bool has_cpu = false, terminated = true;
    for (size_t i = 0; i < n; i++) {
        memset(&info, 0xAB, sizeof(info));
        CHECK(sd_get_device_info(i, &info), "info(i) valid");
        terminated = terminated && memchr(info.name, 0, sizeof(info.name)) &&
                     memchr(info.description, 0, sizeof(info.description));
        has_cpu = has_cpu || info.type == SD_DEVICE_TYPE_CPU;
        CHECK(info.free_memory_bytes <= info.total_memory_bytes || info.total_memory_bytes == 0,
              "free <= total");
    }
    CHECK(terminated, "strings always null-terminated");
    CHECK(has_cpu, "CPU device always present (fallback base)");
    CHECK(strcmp(sd_device_type_name((sd_device_type_t)9999), "unknown") == 0,
          "type_name(invalid) = unknown");

    const size_t need = sd_list_devices(nullptr, 0);
    char tiny[4];
    CHECK(sd_list_devices(tiny, sizeof(tiny)) == need && tiny[3] == '\0',
          "list_devices tiny buffer: no overflow");

    std::vector<std::thread> threads;
    std::vector<int> bad(8, 0);
    for (int t = 0; t < 8; t++) {
        threads.emplace_back([&, t] {
            for (int k = 0; k < 2000; k++) {
                sd_device_info_t x;
                const size_t c = sd_get_device_count();
                for (size_t i = 0; i < c; i++) {
                    if (!sd_get_device_info(i, &x)) {
                        bad[t]++;
                    }
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    int total_bad = 0;
    for (int b : bad) {
        total_bad += b;
    }
    CHECK(total_bad == 0, "8-thread enumeration stress without errors");

    sd_ctx_opt_t* p = sd_ctx_opt_create();
    CHECK(p != nullptr && sd_opt_set_int(p, "memory_guard", 0) == SD_STATUS_OK &&
              sd_opt_set_bool(p, "disable_backend_fallback", false) == SD_STATUS_OK &&
              sd_opt_set_string(p, "multi_gpu", "auto") == SD_STATUS_OK,
          "the context options know memory_guard, disable_backend_fallback and multi_gpu");
    CHECK(sd_opt_set_int(p, "memory_guard", 101) != SD_STATUS_OK, "memory_guard refuses a percent above 100");
    sd_opt_free(p);

    sd_system_memory_t memory;
    CHECK(!sd_get_system_memory(nullptr), "system_memory(null out) rejected");
    memset(&memory, 0xAB, sizeof(memory));
    const bool have_memory = sd_get_system_memory(&memory);
#if defined(__linux__) || defined(_WIN32) || defined(__APPLE__)
    CHECK(have_memory && memory.total_bytes > 0 && memory.available_bytes > 0, "system memory is reported");
#endif
    CHECK(!have_memory || memory.total_bytes == 0 || memory.available_bytes <= memory.total_bytes,
          "available RAM never exceeds the total");
    CHECK(!have_memory || memory.total_bytes == 0 || memory.process_bytes <= memory.total_bytes,
          "process RAM never exceeds the total");
    for (size_t i = 0; i < n; i++) {
        if (sd_get_device_info(i, &info) && info.type == SD_DEVICE_TYPE_CPU && have_memory && memory.available_bytes > 0) {
            const uint64_t spread = info.free_memory_bytes > memory.available_bytes
                                        ? info.free_memory_bytes - memory.available_bytes
                                        : memory.available_bytes - info.free_memory_bytes;
            CHECK(spread < (uint64_t)1 << 30, "CPU device reports the available RAM as free memory");
        }
    }

    char message[256];
    CHECK(sd_get_last_error_code(nullptr, 0) == SD_ERROR_NONE, "no error before any failure; null buffer accepted");
    sd_ctx_t* ctx = reinterpret_cast<sd_ctx_t*>(0x1);
    CHECK(sd_ctx_create(nullptr, &ctx) != SD_STATUS_OK && ctx == nullptr, "sd_ctx_create(null) fails");
    CHECK(sd_get_last_error_code(message, sizeof(message)) == SD_ERROR_INVALID_ARGUMENT && message[0] != '\0',
          "null parameters are an invalid argument, with a message");
    char tiny_message[8];
    memset(tiny_message, 'x', sizeof(tiny_message));
    CHECK(sd_get_last_error_code(tiny_message, sizeof(tiny_message)) == SD_ERROR_INVALID_ARGUMENT &&
              memchr(tiny_message, 0, sizeof(tiny_message)) != nullptr,
          "a small message buffer is truncated and terminated");
    CHECK(sd_get_last_error_code(message, 0) == SD_ERROR_INVALID_ARGUMENT, "a zero-size buffer is accepted");

    sd_ctx_opt_t* missing = sd_ctx_opt_create();
    sd_opt_set_string(missing, "model_path", "/nonexistent/diffusione/model.safetensors");
    CHECK(sd_ctx_create(missing, &ctx) != SD_STATUS_OK && ctx == nullptr, "sd_ctx_create(missing model) fails");
    sd_opt_free(missing);
    CHECK(sd_get_last_error_code(message, sizeof(message)) == SD_ERROR_MODEL_FILE && strstr(message, "nonexistent") != nullptr,
          "a missing model is a model file error that names the file");

    sd_result_t* result = reinterpret_cast<sd_result_t*>(0x1);
    CHECK(sd_generate_image(nullptr, nullptr, &result) != SD_STATUS_OK && result == nullptr,
          "sd_generate_image(null) fails and clears the result");
    CHECK(sd_get_last_error_code(message, sizeof(message)) == SD_ERROR_INVALID_ARGUMENT, "sd_generate_image(null) is an invalid argument");
    CHECK(sd_generate_image(nullptr, nullptr, nullptr) != SD_STATUS_OK &&
              sd_get_last_error_code(message, sizeof(message)) == SD_ERROR_INVALID_ARGUMENT,
          "sd_generate_image without a place for the result is an invalid argument");

    std::thread other([&] {
        CHECK(sd_get_last_error_code(nullptr, 0) == SD_ERROR_NONE, "the error state belongs to the thread that made the call");
    });
    other.join();

    CHECK(strcmp(sd_error_code_name(SD_ERROR_OUT_OF_MEMORY), "out_of_memory") == 0 &&
              strcmp(sd_error_code_name(SD_ERROR_MODEL_FILE), "model_file") == 0 &&
              strcmp(sd_error_code_name(SD_ERROR_BACKEND), "backend") == 0 &&
              strcmp(sd_error_code_name(SD_ERROR_BUSY), "busy") == 0 &&
              strcmp(sd_error_code_name(SD_ERROR_CANCELLED), "cancelled") == 0 &&
              strcmp(sd_error_code_name(SD_ERROR_NONE), "none") == 0 &&
              strcmp(sd_error_code_name((sd_error_code_t)9999), "unknown") == 0,
          "error code names");

    printf("\n%s (%d failures)\n", failures ? "TEST FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
