// Test of the memory guard inside the execution engine (src/core/execution_engine.cpp): ensure_capacity frees
// weight blocks until a request both fits and keeps the usage under the guard threshold, and still runs a request
// that fits when the threshold cannot be reached. The residency manager is a scripted fake, so no model is needed.
// Build (from the repository root, after building the static library into build/):
//   c++ -std=c++17 -DGGML_MAX_NAME=160 -Isrc -Isrc/core -Iinclude -Ithirdparty -Iggml/include \
//       test/memory_guard_engine_test.cpp build/libstable-diffusion.a build/ggml/src/libggml.a \
//       build/ggml/src/libggml-cpu.a build/ggml/src/libggml-base.a -ldl -lm -fopenmp -pthread \
//       -o memory_guard_engine_test
// Run: ./memory_guard_engine_test ; exit code 0 = all checks passed.
#include <cstdio>
#include <vector>

#include "core/execution_engine.h"
#include "device_residency_manager.h"
#include "ggml-cpu.h"

static int failures = 0;

static void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "ok:  " : "FAIL:", what);
    failures += condition ? 0 : 1;
}

// Every eviction frees one block. The capacity fits once fits_after blocks are gone and the guard is satisfied
// once guard_after blocks are gone (never when guard_after is negative).
struct ScriptedManager : DeviceResidencyManager {
    int blocks            = 4;
    int evicted           = 0;
    int fits_after        = 0;
    int guard_after       = 0;
    int unreachable_calls = 0;

    bool segmented_compute_enabled() const override { return true; }
    bool prefetch_enabled() const override { return false; }
    void register_runtime_owner(uintptr_t) override {}
    void remove_runtime_owner(uintptr_t) override {}
    bool fits_compute_backend_capacity(const DeviceMemoryRequest&, const std::vector<ggml_tensor*>&, bool) const override {
        return true;
    }
    bool assign_compute_backend(const std::vector<ggml_tensor*>&, ggml_backend_t) override { return true; }
    bool prepare_params(const std::vector<ggml_tensor*>&) override { return true; }
    void release_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
    void evict_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
    WeightResidencyInfo inspect_compute_backend_params(const std::vector<ggml_tensor*>&) const override { return {}; }
    void update_runtime_residency(uintptr_t, ggml_backend_t, size_t) override {}
    bool ensure_compute_backend_capacity(const DeviceMemoryRequest& request,
                                         const std::vector<ggml_tensor*>& required_params,
                                         const std::vector<std::vector<ggml_tensor*>>& preferred_eviction_order,
                                         const std::vector<ggml_tensor*>& protected_params) override {
        return execution_engine()->ensure_capacity(*this, request, required_params, preferred_eviction_order, protected_params);
    }
    DeviceMemoryCapacity inspect_compute_backend_capacity(const DeviceMemoryRequest&,
                                                          const std::vector<ggml_tensor*>&,
                                                          bool) const override {
        DeviceMemoryCapacity capacity;
        capacity.required_device_bytes  = 100;
        capacity.available_device_bytes = evicted >= fits_after ? 100 : 50;
        return capacity;
    }
    std::vector<WeightResidencyBlock> evictable_compute_backend_blocks(ggml_backend_t backend,
                                                                       const std::vector<ggml_tensor*>&) const override {
        std::vector<WeightResidencyBlock> result;
        for (int i = evicted; i < blocks; ++i) {
            WeightResidencyBlock block;
            block.backend        = backend;
            block.bytes          = 10;
            block.last_use_epoch = static_cast<uint64_t>(i);
            result.push_back(block);
        }
        return result;
    }
    bool evict_compute_backend_block(const WeightResidencyBlock&) override {
        ++evicted;
        return true;
    }
    bool memory_guard_ok(const DeviceMemoryRequest&, const std::vector<ggml_tensor*>&) override {
        return guard_after >= 0 && evicted >= guard_after;
    }
    void memory_guard_unreachable(const DeviceMemoryRequest&, const std::vector<ggml_tensor*>&) override {
        ++unreachable_calls;
    }
    WeightPrefetchResult prefetch_params(uintptr_t, const std::vector<ggml_tensor*>&) override {
        return WeightPrefetchResult::Unsupported;
    }
    bool activate_prefetched_params(uintptr_t, const std::vector<ggml_tensor*>&) override { return true; }
    void clear_prefetched_params(uintptr_t) override {}
};

static bool run(ScriptedManager& manager, ggml_backend_t backend) {
    DeviceMemoryRequest request;
    request.compute_backend = backend;
    request.owner_id        = 1;
    return manager.ensure_compute_backend_capacity(request, {}, {}, {});
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        std::printf("FAIL: no CPU backend\n");
        return 1;
    }
    {
        ScriptedManager manager;
        const bool ok = run(manager, backend);
        check(ok && manager.evicted == 0 && manager.unreachable_calls == 0,
              "a request that fits under the guard threshold frees nothing");
    }
    {
        ScriptedManager manager;
        manager.guard_after = 2;
        const bool ok       = run(manager, backend);
        check(ok && manager.evicted == 2 && manager.unreachable_calls == 0,
              "above the guard threshold, blocks are freed until the usage is under it");
    }
    {
        ScriptedManager manager;
        manager.fits_after  = 1;
        manager.guard_after = 3;
        const bool ok       = run(manager, backend);
        check(ok && manager.evicted == 3 && manager.unreachable_calls == 0,
              "freeing goes on after the request fits, until the guard is satisfied too");
    }
    {
        ScriptedManager manager;
        manager.guard_after = -1;
        const bool ok       = run(manager, backend);
        check(ok && manager.evicted == manager.blocks && manager.unreachable_calls == 1,
              "when the guard target is out of reach, a request that fits still runs and the guard is told once");
    }
    {
        ScriptedManager manager;
        manager.fits_after = 99;
        const bool ok      = run(manager, backend);
        check(!ok && manager.unreachable_calls == 0,
              "a request that cannot fit fails as before, without a guard warning");
    }
    ggml_backend_free(backend);
    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED", failures);
    return failures == 0 ? 0 : 1;
}
