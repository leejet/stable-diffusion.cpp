// Test of how the layer split fills the devices (src/core/layer_split_partition.*, --layer-split-fill):
// proportional ranges, the first device filled first, the choice auto makes for the kinds of devices, and the
// placement the planner and the split share. No GPU and no model are needed.
// Build (from the repository root, after building the static library into build/):
//   c++ -std=c++17 -DGGML_MAX_NAME=160 -Isrc -Isrc/core -Iinclude -Ithirdparty -Iggml/include \
//       test/layer_split_fill_test.cpp build/libstable-diffusion.a build/ggml/src/libggml.a \
//       build/ggml/src/libggml-cpu.a build/ggml/src/libggml-base.a -ldl -lm -fopenmp -pthread \
//       -o layer_split_fill_test
// Run: ./layer_split_fill_test ; exit code 0 = all checks passed.
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "core/layer_split_partition.h"

static int failures = 0;

static void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "ok:  " : "FAIL:", what);
    failures += condition ? 0 : 1;
}

static constexpr int64_t GiB = 1024ll * 1024 * 1024;

static std::vector<int64_t> pieces(int64_t total, int count) {
    std::vector<int64_t> result(count, total / count);
    result.back() += total % count;
    return result;
}

static std::vector<int64_t> place(sd::LayerSplitFill fill, const std::vector<int64_t>& segments, const std::vector<int64_t>& capacities) {
    std::vector<int64_t> bytes;
    if (!sd::layer_split_place(segments, sd::layer_split_fill_limits(fill, segments, capacities), bytes)) {
        return {};
    }
    return bytes;
}

static bool near(int64_t value, int64_t expected, int64_t tolerance) {
    return value >= expected - tolerance && value <= expected + tolerance;
}

int main() {
    using sd::LayerSplitFill;

    LayerSplitFill fill = LayerSplitFill::FILL_FIRST;
    check(sd::parse_layer_split_fill("", fill) && fill == LayerSplitFill::AUTO, "an empty value means auto");
    check(sd::parse_layer_split_fill("Proportional", fill) && fill == LayerSplitFill::PROPORTIONAL,
          "proportional is accepted in any case");
    check(sd::parse_layer_split_fill("fill-first", fill) && fill == LayerSplitFill::FILL_FIRST, "fill-first is accepted");
    check(!sd::parse_layer_split_fill("sideways", fill), "anything else is refused");
    check(std::string(sd::layer_split_fill_name(LayerSplitFill::FILL_FIRST)) == "fill-first", "the names round-trip");

    const std::vector<int64_t> segments = pieces(8 * GiB, 64);
    {
        const auto bytes = place(LayerSplitFill::FILL_FIRST, segments, {10 * GiB, 10 * GiB});
        check(bytes.size() == 2 && bytes[0] == 8 * GiB && bytes[1] == 0, "fill-first keeps everything on the first device while it fits");
    }
    {
        const auto bytes = place(LayerSplitFill::PROPORTIONAL, segments, {10 * GiB, 10 * GiB});
        check(bytes.size() == 2 && near(bytes[0], 4 * GiB, segments[0]) && near(bytes[1], 4 * GiB, segments[0]),
              "proportional halves the weights between two equal devices");
    }
    {
        const auto bytes = place(LayerSplitFill::PROPORTIONAL, segments, {12 * GiB, 4 * GiB});
        check(bytes.size() == 2 && near(bytes[0], 6 * GiB, segments[0]) && near(bytes[1], 2 * GiB, segments[0]),
              "proportional follows the capacities, 3 to 1");
    }
    {
        const auto fill_first   = place(LayerSplitFill::FILL_FIRST, segments, {6 * GiB, 2 * GiB + segments[0]});
        const auto proportional = place(LayerSplitFill::PROPORTIONAL, segments, {6 * GiB, 2 * GiB + segments[0]});
        check(!fill_first.empty() && fill_first == proportional,
              "when the weights barely fit, both fill the devices the same way");
    }
    {
        const auto bytes = place(LayerSplitFill::PROPORTIONAL, segments, {3 * GiB, 3 * GiB});
        check(bytes.empty(), "weights larger than all the devices together are not placed");
    }
    for (LayerSplitFill mode : {LayerSplitFill::FILL_FIRST, LayerSplitFill::PROPORTIONAL}) {
        const auto bytes = place(mode, segments, {7 * GiB, 5 * GiB, 3 * GiB});
        check(bytes.size() == 3 && std::accumulate(bytes.begin(), bytes.end(), int64_t(0)) == 8 * GiB &&
                  bytes[0] <= 7 * GiB && bytes[1] <= 5 * GiB && bytes[2] <= 3 * GiB,
              "every byte is placed and no device gets more than its capacity");
    }

    {
        const std::vector<int64_t> capacities = {10 * GiB, 10 * GiB};
        check(sd::layer_split_apply_shares(capacities, {1.0, 1.0}) == capacities, "healthy devices offer all of their capacity");
        const auto strained = sd::layer_split_apply_shares(capacities, {0.70, 1.0});
        check(strained[0] == 7 * GiB && strained[1] == 10 * GiB, "a device under strain at 70% offers 70% of its capacity");

        const auto fill_first = place(LayerSplitFill::FILL_FIRST, segments, strained);
        check(fill_first.size() == 2 && near(fill_first[0], 7 * GiB, segments[0]) && near(fill_first[1], GiB, segments[0]),
              "fill-first: the next device takes what the device under strain gives up");
        const auto healthy      = place(LayerSplitFill::PROPORTIONAL, segments, capacities);
        const auto proportional = place(LayerSplitFill::PROPORTIONAL, segments, strained);
        check(proportional.size() == 2 && near(proportional[0], 8 * GiB * 7 / 17, segments[0]) &&
                  near(proportional[1], 8 * GiB * 10 / 17, segments[0]),
              "proportional: what it gives up is spread in proportion to the capacities");
        check(!healthy.empty() && !proportional.empty() && proportional[0] < healthy[0],
              "so the device under strain holds less than when it was healthy");

        const auto excluded = place(LayerSplitFill::PROPORTIONAL, segments, sd::layer_split_apply_shares(capacities, {0.0, 1.0}));
        check(excluded.size() == 2 && excluded[0] == 0 && excluded[1] == 8 * GiB, "an excluded device takes nothing");
        check(place(LayerSplitFill::FILL_FIRST, segments, sd::layer_split_apply_shares({10 * GiB, 5 * GiB}, {0.0, 1.0})).empty(),
              "when the others cannot hold what an excluded device leaves, nothing is placed");
        check(place(LayerSplitFill::PROPORTIONAL, segments, sd::layer_split_apply_shares(capacities, {1.0, 1.0})) == healthy,
              "when the device recovers, its share comes back");

        const auto unknown = sd::layer_split_apply_shares({sd::layer_split_unknown_capacity, 10 * GiB}, {0.85, 1.0});
        check(unknown[0] == sd::layer_split_unknown_capacity, "a capacity the device does not report stays unknown");
        check(sd::layer_split_apply_shares({sd::layer_split_unknown_capacity}, {0.0})[0] == 0, "unless the device is excluded");
    }

    const auto gpu  = GGML_BACKEND_DEVICE_TYPE_GPU;
    const auto igpu = GGML_BACKEND_DEVICE_TYPE_IGPU;
    const auto cpu  = GGML_BACKEND_DEVICE_TYPE_CPU;
    check(sd::resolve_layer_split_fill(LayerSplitFill::AUTO, {gpu, gpu}) == LayerSplitFill::PROPORTIONAL,
          "auto spreads the weights over discrete GPUs");
    check(sd::resolve_layer_split_fill(LayerSplitFill::AUTO, {gpu, igpu}) == LayerSplitFill::FILL_FIRST,
          "auto fills the discrete GPU first when an integrated GPU takes part");
    check(sd::resolve_layer_split_fill(LayerSplitFill::AUTO, {gpu, cpu}) == LayerSplitFill::FILL_FIRST,
          "and when the CPU takes part");
    check(sd::resolve_layer_split_fill(LayerSplitFill::PROPORTIONAL, {gpu, igpu}) == LayerSplitFill::PROPORTIONAL &&
              sd::resolve_layer_split_fill(LayerSplitFill::FILL_FIRST, {gpu, gpu}) == LayerSplitFill::FILL_FIRST,
          "a choice made by the user is kept");

    check(sd::layer_split_fill() == LayerSplitFill::AUTO, "auto until a context chooses");
    sd::set_layer_split_fill(LayerSplitFill::FILL_FIRST);
    check(sd::layer_split_fill() == LayerSplitFill::FILL_FIRST, "the choice of the context is kept");
    sd::set_layer_split_fill(LayerSplitFill::AUTO);

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED", failures);
    return failures == 0 ? 0 : 1;
}
