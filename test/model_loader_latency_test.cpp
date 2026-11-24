// Test of the wait at the end of ModelLoader::load_tensors (src/model_loader.cpp): a load whose tensors are
// quick to read must return as soon as the workers are done, not after a fixed progress interval. Streamed
// weights are loaded segment by segment at every sampling step, so any fixed wait is paid dozens of times.
// The test writes a small safetensors file and loads it into host tensors: no GPU and no model are needed,
// but the compiled library is (ModelLoader is internal to it).
// Build (from the repository root, after building the library into build/):
//   c++ -std=c++17 -DGGML_MAX_NAME=160 -Isrc -Isrc/core -Iinclude -Ithirdparty -Iggml/include \
//       test/model_loader_latency_test.cpp -Lbuild/bin -lstable-diffusion -Wl,-rpath,build/bin -pthread -o model_loader_latency_test
// Run: ./model_loader_latency_test ; exit code 0 = all checks passed.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "ggml.h"
#include "model_loader.h"

namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(cond, msg)               \
    do {                               \
        if (!(cond)) {                 \
            printf("FAIL: %s\n", msg); \
            failures++;                \
        } else {                       \
            printf("ok:   %s\n", msg); \
        }                              \
    } while (0)

static constexpr int SIDE     = 32;
static constexpr int ELEMENTS = SIDE * SIDE;

static std::string tensor_name(int index) {
    return "block." + std::to_string(index) + ".weight";
}

static float expected_value(int index, int element) {
    return static_cast<float>(index * 1000 + element % 997);
}

static std::string write_safetensors(const fs::path& path, int count) {
    std::string header = "{";
    for (int i = 0; i < count; ++i) {
        const size_t begin = static_cast<size_t>(i) * ELEMENTS * sizeof(float);
        const size_t end   = begin + ELEMENTS * sizeof(float);
        if (i > 0) {
            header += ",";
        }
        header += "\"" + tensor_name(i) + "\":{\"dtype\":\"F32\",\"shape\":[" + std::to_string(SIDE) + "," +
                  std::to_string(SIDE) + "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
    }
    header += "}";
    while (header.size() % 8 != 0) {
        header += " ";
    }
    std::ofstream out(path, std::ios::binary);
    const uint64_t header_size = header.size();
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    std::vector<float> data(ELEMENTS);
    for (int i = 0; i < count; ++i) {
        for (int e = 0; e < ELEMENTS; ++e) {
            data[e] = expected_value(i, e);
        }
        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    }
    return path.string();
}

struct Destination {
    ggml_context* ctx = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    explicit Destination(int count) {
        ggml_init_params params{static_cast<size_t>(count) * (ELEMENTS * sizeof(float) + 1024) + 4096, nullptr, false};
        ctx = ggml_init(params);
        for (int i = 0; i < count; ++i) {
            ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, SIDE, SIDE);
            ggml_set_name(tensor, tensor_name(i).c_str());
            tensors[tensor_name(i)] = tensor;
        }
    }
    ~Destination() {
        ggml_free(ctx);
    }
    void clear() {
        for (auto& entry : tensors) {
            std::memset(entry.second->data, 0, ggml_nbytes(entry.second));
        }
    }
    bool matches(int count) const {
        for (int i = 0; i < count; ++i) {
            const float* values = static_cast<const float*>(tensors.at(tensor_name(i))->data);
            for (int e = 0; e < ELEMENTS; ++e) {
                if (values[e] != expected_value(i, e)) {
                    return false;
                }
            }
        }
        return true;
    }
};

static double timed_load_ms(ModelLoader& loader, Destination& destination, bool& ok) {
    auto callback = [&](const TensorStorage& storage, ggml_tensor** target) -> bool {
        auto found = destination.tensors.find(storage.name);
        if (found == destination.tensors.end()) {
            return false;
        }
        *target = found->second;
        return true;
    };
    const auto start = std::chrono::steady_clock::now();
    ok               = loader.load_tensors(callback, false, nullptr, false);
    const auto stop  = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

static double fastest_load_ms(const fs::path& file, int count, bool& all_ok, bool& data_ok) {
    ModelLoader loader;
    loader.set_n_threads(4);
    all_ok = loader.init_from_file(file.string());
    if (!all_ok) {
        return -1;
    }
    Destination destination(count);
    double best = 1e9;
    data_ok     = true;
    for (int round = 0; round < 10; ++round) {
        destination.clear();
        bool ok         = false;
        const double ms = timed_load_ms(loader, destination, ok);
        all_ok          = all_ok && ok;
        data_ok         = data_ok && destination.matches(count);
        best            = std::min(best, ms);
    }
    return best;
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "model_loader_latency_test";
    fs::create_directories(dir);

    for (int count : {16, 3}) {
        const fs::path file = dir / ("tensors_" + std::to_string(count) + ".safetensors");
        write_safetensors(file, count);
        bool all_ok = false, data_ok = false;
        const double best = fastest_load_ms(file, count, all_ok, data_ok);
        printf("      %d tensors: fastest of 10 loads %.1f ms\n", count, best);
        CHECK(all_ok, count == 16 ? "16 tensors: every load succeeds" : "3 tensors: every load succeeds");
        CHECK(data_ok, count == 16 ? "16 tensors: the data read is exact" : "3 tensors: the data read is exact");
        CHECK(best >= 0 && best < 100.0,
              count == 16 ? "16 tensors: a load that is already quick returns in under 100 ms, not after a progress tick"
                          : "3 tensors: a load that is already quick returns in under 100 ms");
    }

    {
        const fs::path file = dir / "tensors_failure.safetensors";
        write_safetensors(file, 16);
        ModelLoader loader;
        loader.set_n_threads(4);
        CHECK(loader.init_from_file(file.string()), "failure case: the file is read");
        auto refuse      = [](const TensorStorage&, ggml_tensor**) -> bool { return false; };
        const auto start = std::chrono::steady_clock::now();
        const bool ok    = loader.load_tensors(refuse, false, nullptr, false);
        const double ms  = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        printf("      refused load took %.1f ms\n", ms);
        CHECK(!ok, "failure case: a refused tensor makes the load fail");
        CHECK(ms < 100.0, "failure case: the failure is reported without waiting for a progress tick");
    }

    fs::remove_all(dir);
    if (failures == 0) {
        printf("all checks passed\n");
    } else {
        printf("%d check(s) failed\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
