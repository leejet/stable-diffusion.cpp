// Test of the multi-GPU placement of --auto-fit (src/core/backend_fit.cpp) with simulated devices, so no GPU
// is needed: which module goes to which GPU for --multi-gpu auto|on|off, when the diffusion model is split,
// and that a plan never promises more memory than a device has.
// The planner functions are internal to the library, so the test includes the source file itself.
// Build (from the repository root, after building the library into build/):
//   c++ -std=c++17 -DGGML_MAX_NAME=160 -Isrc -Isrc/core -Iinclude -Ithirdparty -Iggml/include \
//       test/backend_fit_plan_test.cpp -Lbuild/bin -lstable-diffusion -Wl,-rpath,build/bin -pthread -o backend_fit_plan_test
// Run: ./backend_fit_plan_test ; exit code 0 = all checks passed.
#include "../src/core/backend_fit.cpp"

#include <cstdio>
#include <random>

using namespace sd::backend_fit;

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

static constexpr int64_t GiB = 1024 * MiB;

static Device make_gpu(const std::string& name, const char* registry, double free_gib, bool discrete = true) {
    Device device;
    device.name         = name;
    device.description  = name;
    device.registry     = registry;
    device.discrete     = discrete;
    device.free_bytes   = (int64_t)(free_gib * GiB);
    device.budget_bytes = std::max<int64_t>(device.free_bytes - 512 * MiB, 0);
    return device;
}

static Device with_id(Device device, const char* device_id) {
    device.device_id = device_id;
    return device;
}

// staging is the size of the largest tensor of a module, which the compute device must be able to hold at once.
static std::vector<Component> make_components(double dit_gib, double te_gib, double vae_gib, int64_t te_staging = 256 * MiB, int64_t vae_staging = 128 * MiB) {
    return {
        {ComponentKind::DIT, "DiT", (int64_t)(dit_gib * GiB), 2048 * MiB, 256 * MiB},
        {ComponentKind::CONDITIONER, "Conditioner", (int64_t)(te_gib * GiB), 2048 * MiB, te_staging},
        {ComponentKind::VAE, "VAE", (int64_t)(vae_gib * GiB), 1024 * MiB, vae_staging},
    };
}

struct Outcome {
    bool changed = false;
    std::vector<Runtime> start;
    std::vector<Runtime> runtimes;
    std::vector<std::string> notes;
    Plan before;
    Plan plan;
};

// Starts the way derive_backend_specs does unless told otherwise: every module on the device with the largest budget.
// The --layer-split-fill the planner is given; scenarios written for one way of filling set it.
static sd::LayerSplitFill planned_fill = sd::LayerSplitFill::AUTO;

static Outcome run(const std::vector<Component>& components,
                   const std::vector<Device>& devices,
                   MultiGpuMode mode,
                   int64_t ram_gib                         = 0,
                   std::vector<bool> explicit_runtime      = {},
                   std::vector<bool> allow_split           = {},
                   const std::vector<Runtime>& start_given = {}) {
    Outcome out;
    if (start_given.empty()) {
        const size_t main_device = select_main_device(devices);
        out.start                = std::vector<Runtime>(components.size(), Runtime{devices[main_device].name, {main_device}});
    } else {
        out.start = start_given;
    }
    explicit_runtime.resize(components.size(), false);
    allow_split.resize(components.size(), true);
    out.runtimes = out.start;
    out.before   = compute_plan(components, devices, ram_gib * GiB, out.start);
    // As auto-fit plans: with --layer-split-fill auto, the fill that keeps more modules on the GPUs.
    sd::set_layer_split_fill(planned_fill);
    out.changed  = plan_multi_gpu_runtimes_with_fill(components, devices, ram_gib * GiB, mode, explicit_runtime, allow_split, out.runtimes,
                                                     out.notes);
    out.plan     = compute_plan(components, devices, ram_gib * GiB, out.runtimes);
    return out;
}

static std::string names(const Outcome& out) {
    std::string text;
    for (const Runtime& runtime : out.runtimes) {
        text += (text.empty() ? "" : " | ") + runtime.name;
    }
    return text;
}

static bool same_names(const Outcome& out, const char* a, const char* b, const char* c) {
    return out.runtimes.size() == 3 && out.runtimes[0].name == a && out.runtimes[1].name == b && out.runtimes[2].name == c;
}

static bool unchanged(const Outcome& out) {
    if (out.changed || out.runtimes.size() != out.start.size()) {
        return false;
    }
    for (size_t i = 0; i < out.runtimes.size(); ++i) {
        if (out.runtimes[i].name != out.start[i].name || out.runtimes[i].devices != out.start[i].devices) {
            return false;
        }
    }
    return true;
}

static bool resident(const Plan& plan, size_t ci) {
    return plan.decisions[ci].params_location == ParamsLocation::MAIN_GPU;
}

static bool mentions(const Outcome& out, const char* text) {
    for (const std::string& note : out.notes) {
        if (note.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// What the plan puts on each device: the weights of the resident modules, split ones by their shares.
static std::vector<int64_t> weights_per_device(const std::vector<Component>& components, const std::vector<Device>& devices, const Outcome& out) {
    std::vector<int64_t> used(devices.size(), 0);
    for (size_t ci = 0; ci < components.size(); ++ci) {
        if (components[ci].params_bytes == 0 || !resident(out.plan, ci)) {
            continue;
        }
        const auto& shares = out.plan.decisions[ci].shares;
        for (size_t i = 0; i < out.plan.runtimes[ci].devices.size(); ++i) {
            used[out.plan.runtimes[ci].devices[i]] += shares.empty() ? components[ci].params_bytes : shares[i];
        }
    }
    return used;
}

// The largest compute reserve among the resident modules that run on the device.
static int64_t reserve_per_device(const std::vector<Component>& components, size_t device, const Outcome& out) {
    int64_t reserve = 0;
    for (size_t ci = 0; ci < components.size(); ++ci) {
        const auto& used_devices = out.plan.runtimes[ci].devices;
        if (components[ci].params_bytes > 0 && resident(out.plan, ci) &&
            std::find(used_devices.begin(), used_devices.end(), device) != used_devices.end()) {
            reserve = std::max(reserve, components[ci].reserve_bytes);
        }
    }
    return reserve;
}

static void check_random_machines() {
    std::mt19937 rng(12345);
    auto pick = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); };

    int plans = 0, moved = 0, splits = 0;
    int over_budget = 0, bad_shares = 0, not_resident = 0, evicted = 0, legacy_changed = 0, explicit_changed = 0, bad_group = 0;
    std::string first_problem;

    for (int round = 0; round < 6000; ++round) {
        std::vector<Device> devices;
        const int gpu_count = 1 + (int)(rng() % 4);
        for (int i = 0; i < gpu_count; ++i) {
            const bool vulkan     = rng() % 4 == 0;
            const bool integrated = rng() % 6 == 0;
            devices.push_back(make_gpu((vulkan ? "Vulkan" : "CUDA") + std::to_string(i), vulkan ? "Vulkan" : "CUDA", pick(1, 48), !integrated));
        }
        const std::vector<Component> components = make_components(pick(0.5, 70), rng() % 5 == 0 ? 0 : pick(0.2, 20), pick(0.1, 3),
                                                                  (int64_t)pick(64, 1500) * MiB, (int64_t)pick(64, 1500) * MiB);
        const int64_t ram_gib                   = (int64_t)pick(0, 64);

        const size_t main_device = select_main_device(devices);
        std::vector<Runtime> start(components.size(), Runtime{devices[main_device].name, {main_device}});
        std::vector<bool> explicit_runtime(components.size(), false);
        std::vector<bool> allow_split(components.size(), true);
        for (size_t ci = 0; ci < components.size(); ++ci) {
            if (rng() % 7 == 0) {
                const size_t di      = rng() % devices.size();
                start[ci]            = Runtime{devices[di].name, {di}};
                explicit_runtime[ci] = true;
            }
            allow_split[ci] = rng() % 5 != 0;
        }

        for (MultiGpuMode mode : {MultiGpuMode::AUTO, MultiGpuMode::ON, MultiGpuMode::OFF}) {
            const Outcome out = run(components, devices, mode, ram_gib, explicit_runtime, allow_split, start);
            ++plans;
            moved += out.changed ? 1 : 0;
            auto problem = [&](const char* what) {
                if (first_problem.empty()) {
                    first_problem = std::string(what) + ": " + names(out) + " (round " + std::to_string(round) + ")";
                }
            };

            const std::vector<int64_t> used = weights_per_device(components, devices, out);
            for (size_t di = 0; di < devices.size(); ++di) {
                if (used[di] + reserve_per_device(components, di, out) > devices[di].budget_bytes) {
                    ++over_budget;
                    problem("over budget");
                }
            }

            bool higher_changed = false;
            for (size_t ci = 0; ci < components.size(); ++ci) {
                const Runtime& runtime = out.runtimes[ci];
                const bool is_split    = runtime.devices.size() > 1 && runtime.estimate_shares;
                if (is_split && resident(out.plan, ci)) {
                    ++splits;
                    int64_t sum = 0;
                    for (int64_t share : out.plan.decisions[ci].shares) {
                        sum += share;
                    }
                    if (out.plan.decisions[ci].shares.size() != runtime.devices.size() || sum != components[ci].params_bytes) {
                        ++bad_shares;
                        problem("shares do not add up");
                    }
                }
                if (is_split) {
                    bool same_group = components[ci].kind != ComponentKind::VAE && allow_split[ci] && !explicit_runtime[ci];
                    for (size_t di : runtime.devices) {
                        same_group = same_group && devices[di].discrete && devices[di].registry == devices[runtime.devices.front()].registry;
                    }
                    if (!same_group) {
                        ++bad_group;
                        problem("split over GPUs that cannot share a module");
                    }
                }
                const bool changed_runtime = runtime.name != out.start[ci].name;
                if (changed_runtime && !resident(out.plan, ci)) {
                    ++not_resident;
                    problem("a moved module is not resident");
                }
                if (explicit_runtime[ci] && changed_runtime) {
                    ++explicit_changed;
                    problem("a module assigned by the user was moved");
                }
                if (resident(out.before, ci) && !resident(out.plan, ci) && (ci == 0 || !higher_changed)) {
                    ++evicted;
                    problem("a resident module was pushed out");
                }
                higher_changed = higher_changed || changed_runtime;
            }

            size_t discrete_best = 0;
            for (const Device& device : devices) {
                discrete_best += device.discrete && device.budget_bytes > 0 ? 1 : 0;
            }
            bool all_resident = true;
            for (size_t ci = 0; ci < components.size(); ++ci) {
                all_resident = all_resident && (components[ci].params_bytes == 0 || resident(out.before, ci));
            }
            if ((mode == MultiGpuMode::OFF || discrete_best < 2 || (mode == MultiGpuMode::AUTO && all_resident)) && !unchanged(out)) {
                ++legacy_changed;
                problem("a case that must keep the old plan changed");
            }
        }
    }

    printf("      %d plans, %d with a move, %d splits\n", plans, moved, splits);
    if (!first_problem.empty()) {
        printf("      first problem: %s\n", first_problem.c_str());
    }
    CHECK(moved > 0 && splits > 0, "random machines: the cases cover moves and splits");
    CHECK(over_budget == 0, "random machines: no device is given more weights than its budget less the compute reserve");
    CHECK(bad_shares == 0, "random machines: the shares of every split add up to the weights");
    CHECK(bad_group == 0, "random machines: only discrete GPUs of one backend, and only a splittable module, are ever split");
    CHECK(not_resident == 0, "random machines: a module that was moved ends up with its weights resident");
    CHECK(evicted == 0, "random machines: a module is never pushed out of its GPU by a change that does not concern a module of higher priority");
    CHECK(explicit_changed == 0, "random machines: a module assigned by the user is never moved");
    CHECK(legacy_changed == 0, "random machines: off, a single GPU, and a plan that already fits change nothing");
}

// Machines with several backends installed side by side: the same card is listed by CUDA and by Vulkan, and an
// integrated GPU reports the shared system RAM as its own memory.
static void check_mixed_backends() {
    const char* const card_id = "0000:01:00.0";
    const char* const igpu_id = "0000:05:00.0";

    {
        const std::vector<Device> card_and_igpu = {make_gpu("Vulkan0", "Vulkan", 5), make_gpu("Vulkan1", "Vulkan", 7.5, false)};
        CHECK(select_main_device(card_and_igpu) == 0, "a graphics card is the main device even when an integrated GPU reports more memory");
        const std::vector<Device> card_full = {make_gpu("Vulkan0", "Vulkan", 0.3), make_gpu("Vulkan1", "Vulkan", 7.5, false)};
        CHECK(select_main_device(card_full) == 1, "an integrated GPU is the main device when no graphics card has a budget");
        CHECK(select_main_device({make_gpu("Vulkan0", "Vulkan", 7.5, false)}) == 0, "and when it is the only GPU");
        CHECK(select_main_device({}) == SIZE_MAX && select_main_device({make_gpu("CUDA0", "CUDA", 0.2)}) == SIZE_MAX,
              "without a GPU, or without one that has a budget, there is no main device");
        const std::vector<Device> cards = {make_gpu("CUDA0", "CUDA", 8), make_gpu("CUDA1", "CUDA", 12), make_gpu("Vulkan0", "Vulkan", 30, false)};
        CHECK(select_main_device(cards) == 1, "among graphics cards the largest budget still wins");
    }

    {
        std::vector<Device> listed = {with_id(make_gpu("CUDA0", "CUDA", 5), card_id),
                                      with_id(make_gpu("Vulkan0", "Vulkan", 5), card_id),
                                      with_id(make_gpu("Vulkan1", "Vulkan", 7.5, false), igpu_id)};
        mark_shadowed_devices(listed);
        CHECK(listed[0].shadowed_by.empty() && listed[1].shadowed_by == "CUDA0" && listed[2].shadowed_by.empty(),
              "a card listed by CUDA and by Vulkan is one GPU: the Vulkan listing is shadowed by the CUDA one");
        CHECK(has_own_memory(listed[0]) && !has_own_memory(listed[1]) && !has_own_memory(listed[2]),
              "only the CUDA listing counts as memory of its own: not its Vulkan twin, not the integrated GPU");
        CHECK(device_note(listed[1]) == " (same GPU as CUDA0, not used automatically)" && device_note(listed[2]) == " (shares the system memory)" &&
                  device_note(listed[0]).empty(),
              "the plan says which listing is not used and why");

        std::vector<Device> reversed = {listed[2], listed[1], listed[0]};
        for (Device& device : reversed) {
            device.shadowed_by.clear();
        }
        mark_shadowed_devices(reversed);
        CHECK(reversed[1].shadowed_by == "CUDA0" && reversed[2].shadowed_by.empty(), "the backend keeps the card whatever the order of the listing");

        std::vector<Device> chain = {with_id(make_gpu("Vulkan0", "Vulkan", 8), card_id), with_id(make_gpu("OpenCL0", "OpenCL", 8), card_id),
                                     with_id(make_gpu("CUDA0", "CUDA", 8), card_id), make_gpu("Vulkan1", "Vulkan", 8)};
        mark_shadowed_devices(chain);
        CHECK(chain[0].shadowed_by == "CUDA0" && chain[1].shadowed_by == "CUDA0" && chain[2].shadowed_by.empty(),
              "a card listed by three backends is kept by the vendor's own");
        CHECK(chain[3].shadowed_by.empty(), "and a listing without a PCI id is never matched with another");

        std::vector<Device> same_backend = {with_id(make_gpu("Vulkan0", "Vulkan", 8), card_id), with_id(make_gpu("Vulkan1", "Vulkan", 8), card_id)};
        mark_shadowed_devices(same_backend);
        CHECK(same_backend[0].shadowed_by.empty() && same_backend[1].shadowed_by == "Vulkan0", "two listings by the same backend: the first one keeps the card");

        std::vector<Device> unknown = {make_gpu("CUDA0", "CUDA", 12), make_gpu("Vulkan0", "Vulkan", 12)};
        mark_shadowed_devices(unknown);
        CHECK(unknown[0].shadowed_by.empty() && unknown[1].shadowed_by.empty(), "devices that report no PCI id are left as they are");
    }

    {
        // A laptop with a graphics card (5 GiB free) next to an integrated GPU (7.5 GiB of shared RAM), CUDA and
        // Vulkan both installed, Pony-sized weights, the text encoder on the CPU. The integrated GPU used to be the
        // main device and took the whole model, which then ran several times slower than on the card.
        std::vector<Device> devices = {with_id(make_gpu("CUDA0", "CUDA", 5), card_id),
                                       with_id(make_gpu("Vulkan0", "Vulkan", 5), card_id),
                                       with_id(make_gpu("Vulkan1", "Vulkan", 7.5, false), igpu_id)};
        mark_shadowed_devices(devices);
        const std::vector<Component> components = make_components(4.8, 1.5, 0.15);
        const size_t main_device                = select_main_device(devices);
        CHECK(main_device == 0, "laptop with two backends: the main device is the card, seen through CUDA");

        const Runtime on_main{devices[main_device].name, {main_device}};
        const Outcome out = run(components, devices, MultiGpuMode::AUTO, 3, {false, true, false}, {}, {on_main, Runtime{"CPU", {}}, on_main});
        CHECK(unchanged(out) && out.notes.empty(), "and the planner finds nothing to move, and nothing to say about other backends");
        CHECK(out.plan.decisions[0].params_location == ParamsLocation::DISK,
              "the diffusion weights, too big for the card and for the RAM budget, wait on disk");
        CHECK(out.plan.decisions[1].params_location == ParamsLocation::CPU && out.plan.decisions[2].params_location == ParamsLocation::CPU,
              "the text encoder and the VAE wait in RAM");
        bool parked_on_gpu = false;
        for (const Decision& decision : out.plan.decisions) {
            parked_on_gpu = parked_on_gpu || decision.params_location == ParamsLocation::OTHER_GPU;
        }
        CHECK(!parked_on_gpu, "no weights are parked in the memory of the integrated GPU, which is the RAM that was found too small");
    }

    {
        const std::vector<Device> card_and_igpu = {make_gpu("CUDA0", "CUDA", 12), make_gpu("Vulkan0", "Vulkan", 30, false)};
        const std::vector<Runtime> on_card(3, Runtime{"CUDA0", {0}});
        const Plan on_disk = compute_plan(make_components(20, 2, 0.3), card_and_igpu, 0, on_card);
        CHECK(on_disk.decisions[0].params_location == ParamsLocation::DISK,
              "weights that fit neither the card nor RAM wait on disk, not in an integrated GPU's shared memory");
        const Plan in_ram = compute_plan(make_components(20, 2, 0.3), card_and_igpu, 64 * GiB, on_card);
        CHECK(in_ram.decisions[0].params_location == ParamsLocation::CPU, "and wait in RAM when there is room");

        const std::vector<Device> two_cards = {make_gpu("CUDA0", "CUDA", 12), make_gpu("CUDA1", "CUDA", 12)};
        const Plan other_card               = compute_plan(make_components(11, 0, 0), two_cards, 0, on_card);
        CHECK(other_card.decisions[0].params_location == ParamsLocation::OTHER_GPU && other_card.decisions[0].params_device == 1,
              "weights still wait in the memory of a second graphics card when the first cannot keep them resident");
    }

    {
        std::vector<Device> card = {with_id(make_gpu("CUDA0", "CUDA", 8), card_id), with_id(make_gpu("Vulkan0", "Vulkan", 8), card_id)};
        mark_shadowed_devices(card);
        const std::vector<Runtime> on_vulkan(3, Runtime{"Vulkan0", {1}});
        const Plan plan = compute_plan(make_components(6.5, 0, 0), card, 0, on_vulkan);
        CHECK(plan.decisions[0].params_location == ParamsLocation::DISK,
              "a module the user put on the Vulkan listing does not park its weights in the same card's CUDA memory");
    }

    {
        // Two cards, each listed by CUDA and by Vulkan; Vulkan reports slightly more memory than CUDA does.
        std::vector<Device> rig = {with_id(make_gpu("Vulkan1", "Vulkan", 12.02), "0000:02:00.0"), with_id(make_gpu("CUDA0", "CUDA", 12), "0000:01:00.0"),
                                   with_id(make_gpu("Vulkan0", "Vulkan", 12.02), "0000:01:00.0"), with_id(make_gpu("CUDA1", "CUDA", 12), "0000:02:00.0")};
        mark_shadowed_devices(rig);
        CHECK(rig[select_main_device(rig)].name == "CUDA0", "a Vulkan listing that reports a little more memory than CUDA does not become the main device");
        const Outcome out = run(make_components(16, 2, 0.3), rig, MultiGpuMode::AUTO);
        CHECK(out.runtimes[0].name == "CUDA0&CUDA1" && resident(out.plan, 0) && !mentions(out, "different backends"),
              "two cards, each listed twice: the diffusion model is split across the two CUDA devices, and the Vulkan twins are not mentioned");
    }
}

static bool has(const std::string& text, const char* part) {
    return text.find(part) != std::string::npos;
}

// The two warnings derive_backend_specs logs after the plan: a module run on a listing of a GPU that another backend
// already provides, and weights that stream from disk.
static void check_plan_warnings() {
    const char* const card_id = "0000:01:00.0";
    const Runtime on_cuda{"CUDA0", {0}};
    const Runtime on_vulkan{"Vulkan0", {1}};

    {
        std::vector<Device> card = {with_id(make_gpu("CUDA0", "CUDA", 8), card_id), with_id(make_gpu("Vulkan0", "Vulkan", 8), card_id)};
        mark_shadowed_devices(card);
        const std::vector<Component> components = make_components(4.8, 1.5, 0.15);

        const Plan forced                       = compute_plan(components, card, 16 * GiB, std::vector<Runtime>(3, on_vulkan));
        const std::vector<std::string> warnings = shadowed_runtime_warnings(forced, card);
        CHECK(warnings.size() == 1 && has(warnings[0], "Vulkan0 is the same GPU as CUDA0") && has(warnings[0], "name CUDA0 in --backend"),
              "--backend vulkan0 on a card that CUDA lists as CUDA0: one warning that names both and says what to pass instead");

        const std::vector<Runtime> mixed = {on_cuda, on_vulkan, on_vulkan};
        CHECK(shadowed_runtime_warnings(compute_plan(components, card, 16 * GiB, mixed), card).size() == 1,
              "and still one warning when several modules run on that listing");
        CHECK(shadowed_runtime_warnings(compute_plan(components, card, 16 * GiB, std::vector<Runtime>(3, on_cuda)), card).empty(),
              "nothing is said when the modules run on the listing the automatic choice makes");

        std::vector<Device> unknown = {make_gpu("CUDA0", "CUDA", 8), make_gpu("Vulkan0", "Vulkan", 8)};
        mark_shadowed_devices(unknown);
        CHECK(shadowed_runtime_warnings(compute_plan(components, unknown, 16 * GiB, std::vector<Runtime>(3, on_vulkan)), unknown).empty(),
              "listings that report no PCI id cannot be told apart from two cards, so nothing is claimed about them");
    }

    {
        std::vector<Device> rig = {with_id(make_gpu("CUDA0", "CUDA", 12), "0000:01:00.0"), with_id(make_gpu("CUDA1", "CUDA", 12), "0000:02:00.0"),
                                   with_id(make_gpu("Vulkan0", "Vulkan", 12), "0000:01:00.0"), with_id(make_gpu("Vulkan1", "Vulkan", 12), "0000:02:00.0")};
        mark_shadowed_devices(rig);
        const std::vector<Component> components = make_components(16, 2, 0.3);

        const std::vector<Runtime> on_cuda_pair(3, Runtime{"CUDA0&CUDA1", {0, 1}});
        CHECK(shadowed_runtime_warnings(compute_plan(components, rig, 0, on_cuda_pair), rig).empty(), "two cards each listed twice: the CUDA pair is not warned about");

        const std::vector<Runtime> on_vulkan_pair(3, Runtime{"Vulkan0&Vulkan1", {2, 3}});
        const std::vector<std::string> warnings = shadowed_runtime_warnings(compute_plan(components, rig, 0, on_vulkan_pair), rig);
        CHECK(warnings.size() == 2 && has(warnings[0], "Vulkan0 is the same GPU as CUDA0") && has(warnings[1], "Vulkan1 is the same GPU as CUDA1"),
              "and the Vulkan pair gets one warning per card");
    }

    {
        std::vector<Device> twice = {with_id(make_gpu("Vulkan0", "Vulkan", 8), card_id), with_id(make_gpu("Vulkan1", "Vulkan", 8), card_id)};
        mark_shadowed_devices(twice);
        const std::vector<std::string> warnings =
            shadowed_runtime_warnings(compute_plan(make_components(4.8, 1.5, 0.15), twice, 0, std::vector<Runtime>(3, Runtime{"Vulkan1", {1}})), twice);
        CHECK(warnings.size() == 1 && has(warnings[0], "Vulkan1 is the same GPU as Vulkan0") && !has(warnings[0], "generic graphics API"),
              "two listings of the same backend: the warning names the owner but does not claim the other backend is slower");
    }

    {
        // Pony-sized weights on a 5 GiB card with 3 GiB of RAM budget: the diffusion model fits neither.
        const std::vector<Device> card          = {make_gpu("CUDA0", "CUDA", 5)};
        const std::vector<Component> components = make_components(4.8, 1.5, 0.15);
        const std::vector<Runtime> start        = {on_cuda, Runtime{"CPU", {}}, on_cuda};
        const Plan plan                         = compute_plan(components, card, 3 * GiB, start);
        CHECK(plan.decisions[0].params_location == ParamsLocation::DISK, "setup: the diffusion weights are left on disk");

        const std::string warning = disk_streaming_warning(plan, components, 4 * GiB, 3 * GiB);
        CHECK(has(warning, "weights of DiT (") && !has(warning, "Conditioner") && !has(warning, "VAE") &&
                  has(warning, (std::to_string(components[0].params_bytes / MiB) + " MiB").c_str()),
              "weights on disk: the warning names the module that streams and how much it is");
        CHECK(has(warning, "the diffusion model at every sampling step") && has(warning, "free RAM is 4096 MiB") && has(warning, "budget for weights 3072 MiB") &&
                  has(warning, "--type"),
              "it says that the diffusion model is read at every step, how much RAM there is, and what helps");
        CHECK(disk_streaming_warning(plan, components, -1, 0).empty(), "an unknown amount of RAM gives no warning, since print_plan already says it is unknown");

        const Plan resident_plan = compute_plan(components, card, 64 * GiB, start);
        CHECK(resident_plan.decisions[0].params_location == ParamsLocation::CPU && disk_streaming_warning(resident_plan, components, 64 * GiB, 60 * GiB).empty(),
              "nothing is said when the weights wait in RAM");

        Plan vae_only;
        vae_only.decisions.assign(3, Decision{ParamsLocation::CPU});
        vae_only.decisions[2].params_location = ParamsLocation::DISK;
        const std::string vae_warning         = disk_streaming_warning(vae_only, components, 4 * GiB, 1 * GiB);
        CHECK(has(vae_warning, "weights of VAE (") && !has(vae_warning, "sampling step"), "a module other than the diffusion model is not said to be read at every sampling step");
    }
}

// Random machines with a card listed by one or two backends and sometimes an integrated GPU, in a random order:
// whatever the plan, nothing is placed on a listing that is not the GPU's own.
static void check_random_mixed_backends() {
    std::mt19937 rng(2024);
    auto pick = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); };

    int plans = 0, with_twins = 0, with_igpu = 0, parked = 0;
    int shadowed_main = 0, igpu_main = 0, bad_group = 0, on_shadowed = 0, bad_parking = 0, over_budget = 0;
    std::string first_problem;

    for (int round = 0; round < 4000; ++round) {
        std::vector<Device> devices;
        const int cards = 1 + (int)(rng() % 3);
        for (int i = 0; i < cards; ++i) {
            const std::string id = "0000:0" + std::to_string(i + 1) + ":00.0";
            const double memory  = pick(2, 24);
            const bool native    = rng() % 3 != 0;
            const bool twin      = !native || rng() % 2 == 0;
            if (native) {
                devices.push_back(with_id(make_gpu("CUDA" + std::to_string(i), "CUDA", memory), id.c_str()));
            }
            if (twin) {
                devices.push_back(with_id(make_gpu("Vulkan" + std::to_string(i), "Vulkan", memory + pick(-0.05, 0.05)), id.c_str()));
            }
            with_twins += native && twin ? 1 : 0;
        }
        if (rng() % 2 == 0) {
            devices.push_back(with_id(make_gpu("Vulkan" + std::to_string(cards), "Vulkan", pick(2, 40), false), "0000:0f:00.0"));
            ++with_igpu;
        }
        std::shuffle(devices.begin(), devices.end(), rng);
        mark_shadowed_devices(devices);

        const size_t main_device = select_main_device(devices);
        auto problem             = [&](const char* what) {
            if (first_problem.empty()) {
                first_problem = std::string(what) + " (round " + std::to_string(round) + ")";
            }
        };
        if (main_device == SIZE_MAX || !devices[main_device].shadowed_by.empty()) {
            ++shadowed_main;
            problem("the main device is missing or shadowed");
            continue;
        }
        if (!devices[main_device].discrete) {
            ++igpu_main;
            problem("an integrated GPU is the main device although a graphics card has a budget");
        }
        const std::vector<size_t> group = multi_gpu_group(devices);
        for (size_t di : group) {
            if (!has_own_memory(devices[di]) || devices[di].registry != devices[group.front()].registry) {
                ++bad_group;
                problem("the multi-GPU group holds a shadowed or integrated device, or mixes backends");
            }
        }

        const std::vector<Component> components = make_components(pick(0.5, 40), rng() % 5 == 0 ? 0 : pick(0.2, 12), pick(0.1, 2));
        const int64_t ram_gib                   = (int64_t)pick(0, 32);
        for (MultiGpuMode mode : {MultiGpuMode::AUTO, MultiGpuMode::ON, MultiGpuMode::OFF}) {
            const Outcome out = run(components, devices, mode, ram_gib);
            ++plans;
            const std::vector<int64_t> used = weights_per_device(components, devices, out);
            for (size_t di = 0; di < devices.size(); ++di) {
                if (used[di] + reserve_per_device(components, di, out) > devices[di].budget_bytes) {
                    ++over_budget;
                    problem("a device is given more than its budget");
                }
            }
            for (size_t ci = 0; ci < components.size(); ++ci) {
                for (size_t di : out.plan.runtimes[ci].devices) {
                    if (!devices[di].shadowed_by.empty()) {
                        ++on_shadowed;
                        problem("a module was placed on a listing of a card that another backend lists");
                    }
                }
                const Decision& decision = out.plan.decisions[ci];
                if (decision.params_location != ParamsLocation::OTHER_GPU) {
                    continue;
                }
                ++parked;
                const Device& target = devices[decision.params_device];
                bool computes_there  = false;
                for (size_t di : out.plan.runtimes[ci].devices) {
                    computes_there = computes_there || di == decision.params_device || devices[di].device_id == target.device_id;
                }
                if (!has_own_memory(target) || computes_there) {
                    ++bad_parking;
                    problem("weights were parked on an integrated GPU, a shadowed listing, or the card the module computes on");
                }
            }
        }
    }

    printf("      %d plans, %d cards listed by two backends, %d integrated GPUs, %d weights parked on another GPU\n", plans, with_twins, with_igpu, parked);
    if (!first_problem.empty()) {
        printf("      first problem: %s\n", first_problem.c_str());
    }
    CHECK(with_twins > 0 && with_igpu > 0 && parked > 0, "mixed backends: the cases cover twin listings, integrated GPUs and weights parked on another GPU");
    CHECK(shadowed_main == 0 && igpu_main == 0, "mixed backends: the main device is always a graphics card listed once, never an integrated GPU while a card has a budget");
    CHECK(bad_group == 0, "mixed backends: the multi-GPU group holds only graphics cards of one backend, each listed once");
    CHECK(on_shadowed == 0, "mixed backends: no module is ever placed on the second listing of a card");
    CHECK(bad_parking == 0, "mixed backends: weights are only parked in the memory of another graphics card");
    CHECK(over_budget == 0, "mixed backends: no device is given more weights than its budget less the compute reserve");
}

int main() {
    MultiGpuMode mode = MultiGpuMode::OFF;
    CHECK(parse_multi_gpu_mode("", mode) && mode == MultiGpuMode::AUTO, "an empty --multi-gpu means auto");
    CHECK(parse_multi_gpu_mode("auto", mode) && mode == MultiGpuMode::AUTO && parse_multi_gpu_mode("ON", mode) && mode == MultiGpuMode::ON &&
              parse_multi_gpu_mode(" Off ", mode) && mode == MultiGpuMode::OFF,
          "--multi-gpu accepts auto, on and off in any case");
    CHECK(!parse_multi_gpu_mode("maybe", mode) && !parse_multi_gpu_mode("1", mode), "--multi-gpu refuses anything else");

    {
        std::string spec = "te=cpu,vae=CUDA0";
        set_assignment(spec, "vae", "CUDA1");
        set_assignment(spec, "diffusion", "CUDA0&CUDA1");
        CHECK(spec == "te=cpu,vae=CUDA1,diffusion=CUDA0&CUDA1", "set_assignment replaces a module and appends a missing one");
    }

    {
        using sd::LayerSplitFill;
        std::vector<int64_t> shares;
        // The estimate cuts the weights into 256 even pieces, so a share can stop one piece short of a limit.
        const int64_t piece = 10 * GiB / 256;
        CHECK(estimate_split_shares(10 * GiB, {8 * GiB, 8 * GiB}, LayerSplitFill::FILL_FIRST, shares) &&
                  shares[0] > 8 * GiB - piece && shares[0] <= 8 * GiB && shares[0] + shares[1] == 10 * GiB,
              "fill-first fills the first device before the next");
        CHECK(estimate_split_shares(10 * GiB, {8 * GiB, 8 * GiB}, LayerSplitFill::PROPORTIONAL, shares) &&
                  shares[0] > 5 * GiB - piece && shares[0] < 5 * GiB + piece && shares[0] + shares[1] == 10 * GiB,
              "proportional gives two equal devices half each");
        CHECK(!estimate_split_shares(16 * GiB, {8 * GiB, 8 * GiB}, LayerSplitFill::PROPORTIONAL, shares),
              "a split needs room for the blocks that do not cut evenly");
        CHECK(estimate_split_shares(14 * GiB, {8 * GiB, 8 * GiB}, LayerSplitFill::FILL_FIRST, shares) && shares[0] + shares[1] == 14 * GiB,
              "the shares add up to the size of the weights");
    }

    const std::vector<Device> one_gpu  = {make_gpu("CUDA0", "CUDA", 12)};
    const std::vector<Device> two_gpus = {make_gpu("CUDA0", "CUDA", 12), make_gpu("CUDA1", "CUDA", 12)};

    {
        const std::vector<Runtime> on_cuda0(3, Runtime{"CUDA0", {0}});
        const Plan fits = compute_plan(make_components(6, 2, 0.3), one_gpu, 0, on_cuda0);
        CHECK(resident(fits, 0) && resident(fits, 1) && resident(fits, 2), "unchanged planner: weights that fit stay on the GPU");
        const Plan in_ram = compute_plan(make_components(16, 2, 0.3), one_gpu, 64 * GiB, on_cuda0);
        CHECK(in_ram.decisions[0].params_location == ParamsLocation::CPU && in_ram.decisions[0].shares.empty(),
              "unchanged planner: weights that do not fit wait in RAM");
        const Plan on_disk = compute_plan(make_components(16, 2, 0.3), one_gpu, 0, on_cuda0);
        CHECK(on_disk.decisions[0].params_location == ParamsLocation::DISK, "unchanged planner: without RAM they stay on disk");
        const std::vector<Runtime> listed(3, Runtime{"CUDA0&CUDA1", {0, 1}});
        const Plan both = compute_plan(make_components(9, 0, 0), two_gpus, 0, listed);
        CHECK(resident(both, 0) && both.decisions[0].shares.empty(), "unchanged planner: a device list given by the user counts the full size on each device");
        const Plan too_big = compute_plan(make_components(12, 0, 0), two_gpus, 0, listed);
        CHECK(!resident(too_big, 0), "and so a list is refused when the weights do not fit on every device, even if they would fit split");
    }

    for (MultiGpuMode m : {MultiGpuMode::AUTO, MultiGpuMode::ON, MultiGpuMode::OFF}) {
        const Outcome out = run(make_components(16, 2, 0.3), one_gpu, m);
        CHECK(unchanged(out) && out.notes.empty(), "one GPU: the plan is the same as before in every mode");
    }
    {
        const Outcome out = run(make_components(6, 2, 0.3), {}, MultiGpuMode::AUTO, 0, {}, {}, std::vector<Runtime>(3, Runtime{"CPU", {}}));
        CHECK(unchanged(out), "no GPU: nothing to place");
    }

    {
        const Outcome out = run(make_components(6, 2, 0.3), two_gpus, MultiGpuMode::AUTO);
        CHECK(unchanged(out) && same_names(out, "CUDA0", "CUDA0", "CUDA0") && resident(out.plan, 0) && resident(out.plan, 1) && resident(out.plan, 2),
              "two GPUs, everything fits on one: auto leaves it there");
        CHECK(mentions(out, "2 GPUs found (CUDA0, CUDA1)") && mentions(out, "everything fits on CUDA0, so CUDA1 will not be used"),
              "and says why the second GPU stays idle");
    }
    {
        const Outcome out = run(make_components(6, 2, 0.3), two_gpus, MultiGpuMode::OFF);
        CHECK(unchanged(out) && out.notes.empty(), "--multi-gpu off never touches the plan");
    }
    {
        const Outcome out = run(make_components(6, 2, 0.3), two_gpus, MultiGpuMode::ON);
        CHECK(out.changed && same_names(out, "CUDA0", "CUDA1", "CUDA1") && resident(out.plan, 0) && resident(out.plan, 1) && resident(out.plan, 2),
              "--multi-gpu on moves the text encoder and the VAE to the second GPU and keeps the diffusion model on the first");
        CHECK(mentions(out, "te runs on CUDA1") && mentions(out, "vae runs on CUDA1"), "and reports both moves");
    }

    {
        const std::vector<Component> components = make_components(16, 2, 0.3);
        planned_fill                            = sd::LayerSplitFill::FILL_FIRST;
        const Outcome out                       = run(components, two_gpus, MultiGpuMode::AUTO);
        planned_fill                            = sd::LayerSplitFill::AUTO;
        CHECK(!resident(out.before, 0), "the diffusion model alone does not fit on one 12 GiB GPU");
        CHECK(out.changed && out.runtimes[0].name == "CUDA0&CUDA1" && out.runtimes[0].devices.size() == 2 && out.runtimes[0].estimate_shares,
              "16 GiB of diffusion weights are split across both GPUs");
        CHECK(resident(out.plan, 0) && out.plan.decisions[0].shares.size() == 2 &&
                  out.plan.decisions[0].shares[0] + out.plan.decisions[0].shares[1] == 16 * GiB &&
                  out.plan.decisions[0].shares[0] > out.plan.decisions[0].shares[1],
              "the shares add up to the weights, the first GPU taking the most");
        CHECK(same_names(out, "CUDA0&CUDA1", "CUDA1", "CUDA1") && resident(out.plan, 1) && resident(out.plan, 2),
              "the text encoder and the VAE use what is left on the second GPU");
        CHECK(mentions(out, "diffusion is split across CUDA0&CUDA1"), "and the split is reported");
        const std::vector<int64_t> used = weights_per_device(components, two_gpus, out);
        CHECK(used[0] + 2 * GiB <= two_gpus[0].budget_bytes && used[1] + 2 * GiB <= two_gpus[1].budget_bytes,
              "no GPU is given more than its budget less the compute reserve");
    }
    {
        const std::vector<Component> components = make_components(16, 2, 0.3);
        const Outcome out                       = run(components, two_gpus, MultiGpuMode::AUTO);
        const auto& shares                      = out.plan.decisions[0].shares;
        CHECK(out.runtimes[0].name == "CUDA0&CUDA1" && resident(out.plan, 0) && resident(out.plan, 1) && resident(out.plan, 2) &&
                  shares.size() == 2 && shares[0] + shares[1] == 16 * GiB && shares[0] < shares[1] + GiB && shares[1] < shares[0] + GiB,
              "with --layer-split-fill auto every module stays on the GPUs either way, so the split is proportional");
        CHECK(mentions(out, "in proportion to their memory"), "and the choice is reported");
        const std::vector<int64_t> used = weights_per_device(components, two_gpus, out);
        CHECK(used[0] + 2 * GiB <= two_gpus[0].budget_bytes && used[1] + 2 * GiB <= two_gpus[1].budget_bytes,
              "the proportional plan too stays within the budgets");
        const Outcome on = run(components, two_gpus, MultiGpuMode::ON);
        CHECK(on.runtimes[0].name == out.runtimes[0].name && on.runtimes[1].name == out.runtimes[1].name &&
                  on.runtimes[2].name == out.runtimes[2].name,
              "on and auto agree when the diffusion model needs both GPUs");
    }
    {
        auto on_gpus = [](const Outcome& outcome) {
            int count = 0;
            for (size_t ci = 0; ci < 3; ++ci) {
                count += resident(outcome.plan, ci) ? 1 : 0;
            }
            return count;
        };
        planned_fill        = sd::LayerSplitFill::PROPORTIONAL;
        const Outcome even  = run(make_components(16, 2.5, 0.3), two_gpus, MultiGpuMode::AUTO);
        planned_fill        = sd::LayerSplitFill::AUTO;
        const Outcome chose = run(make_components(16, 2.5, 0.3), two_gpus, MultiGpuMode::AUTO);
        CHECK(on_gpus(even) == 2 && on_gpus(chose) == 3 && mentions(chose, "first one first"),
              "when a proportional split would push a module off the GPUs, auto fills the first GPU first and keeps all three on");
    }
    {
        const Outcome out = run(make_components(16, 2, 0.3), two_gpus, MultiGpuMode::OFF, 64);
        CHECK(unchanged(out) && out.plan.decisions[0].params_location == ParamsLocation::CPU,
              "off keeps the old plan: the weights wait in RAM and stream to the first GPU");
    }
    {
        const Outcome out = run(make_components(16, 2, 0.3), two_gpus, MultiGpuMode::AUTO, 64, {}, {false, true, true});
        CHECK(out.runtimes[0].name == "CUDA0" && !resident(out.plan, 0), "a module whose --split-mode is row is not split by auto-fit");
    }
    {
        const Outcome out = run(make_components(40, 2, 0.3), two_gpus, MultiGpuMode::AUTO, 64);
        CHECK(out.runtimes[0].name == "CUDA0" && !resident(out.plan, 0), "weights too big for both GPUs together are not split");
    }
    {
        const std::vector<Device> small_second = {make_gpu("CUDA0", "CUDA", 12), make_gpu("CUDA1", "CUDA", 3)};
        const Outcome out                      = run(make_components(16, 2, 0.3), small_second, MultiGpuMode::AUTO, 64);
        CHECK(out.runtimes[0].name == "CUDA0" && !resident(out.plan, 0), "a second GPU too small to add anything does not make a split");
    }
    {
        const std::vector<Device> three_gpus = {make_gpu("CUDA0", "CUDA", 12), make_gpu("CUDA1", "CUDA", 12), make_gpu("CUDA2", "CUDA", 12)};
        const Outcome out                    = run(make_components(24, 2, 0.3), three_gpus, MultiGpuMode::AUTO);
        CHECK(out.runtimes[0].name == "CUDA0&CUDA1&CUDA2" && resident(out.plan, 0) && out.plan.decisions[0].shares.size() == 3,
              "three GPUs: the diffusion model is split across all three when two are not enough");
    }
    {
        const std::vector<Device> uneven = {make_gpu("CUDA0", "CUDA", 8), make_gpu("CUDA1", "CUDA", 24)};
        const Outcome out                = run(make_components(16, 2, 0.3), uneven, MultiGpuMode::AUTO);
        CHECK(unchanged(out) && same_names(out, "CUDA1", "CUDA1", "CUDA1"), "uneven GPUs: everything stays on the larger one when it fits");
        const Outcome on = run(make_components(16, 2, 0.3), uneven, MultiGpuMode::ON);
        CHECK(same_names(on, "CUDA1", "CUDA0", "CUDA0"), "and --multi-gpu on moves the text encoder and the VAE to the smaller one");
    }
    {
        const std::vector<Device> mixed = {make_gpu("CUDA0", "CUDA", 12), make_gpu("Vulkan0", "Vulkan", 12)};
        const Outcome out               = run(make_components(16, 2, 0.3), mixed, MultiGpuMode::AUTO, 64);
        CHECK(unchanged(out) && mentions(out, "different backends"), "GPUs of different backends are not combined, and the user is told");
    }
    {
        const std::vector<Device> with_igpu = {make_gpu("CUDA0", "CUDA", 8), make_gpu("CUDA1", "CUDA", 8), make_gpu("Vulkan0", "Vulkan", 30, false)};
        const std::vector<Runtime> on_igpu(3, Runtime{"Vulkan0", {2}});
        const Outcome out = run(make_components(6, 2, 0.3), with_igpu, MultiGpuMode::AUTO, 0, {}, {}, on_igpu);
        CHECK(unchanged(out) && same_names(out, "Vulkan0", "Vulkan0", "Vulkan0"), "an integrated GPU keeps the modules it hosts when they fit");
        CHECK(mentions(out, "everything fits on Vulkan0, so CUDA0, CUDA1 will not be used"), "and the note names the device that hosts them");
        const Outcome big = run(make_components(40, 2, 0.3), with_igpu, MultiGpuMode::AUTO, 64, {}, {}, on_igpu);
        CHECK(big.runtimes[0].name == "Vulkan0" && !resident(big.plan, 0), "and is never part of a split");
    }
    {
        const std::vector<Runtime> start = {Runtime{"CUDA1", {1}}, Runtime{"CUDA0", {0}}, Runtime{"CUDA0", {0}}};
        const Outcome out                = run(make_components(6, 2, 0.3), two_gpus, MultiGpuMode::ON, 0, {true, false, false}, {}, start);
        CHECK(same_names(out, "CUDA1", "CUDA0", "CUDA0") && unchanged(out),
              "--multi-gpu on keeps the text encoder and the VAE away from a diffusion model the user put on another GPU");
        const Outcome same_gpu = run(make_components(6, 2, 0.3), two_gpus, MultiGpuMode::ON, 0, {true, false, false},
                                     {}, {Runtime{"CUDA0", {0}}, Runtime{"CUDA0", {0}}, Runtime{"CUDA0", {0}}});
        CHECK(same_names(same_gpu, "CUDA0", "CUDA1", "CUDA1"), "and moves them off the GPU the user gave to the diffusion model");
    }
    {
        const Outcome out = run(make_components(16, 2, 0.3), two_gpus, MultiGpuMode::ON, 0, {true, true, true});
        CHECK(unchanged(out), "when every module is assigned by the user there is nothing to plan");
    }
    {
        const Outcome out = run(make_components(16, 0, 0.3), two_gpus, MultiGpuMode::AUTO);
        CHECK(out.runtimes[0].name == "CUDA0&CUDA1" && resident(out.plan, 0) && resident(out.plan, 2),
              "a model without a text encoder still splits and places the VAE");
    }
    {
        // The split just fits while the text encoder and the VAE are on the first GPU. Neither can follow onto the
        // second one without pushing the diffusion model off the GPUs, so they stay where they are.
        const std::vector<Component> components = {
            {ComponentKind::DIT, "DiT", 17150 * MiB, 2048 * MiB, 256 * MiB},
            {ComponentKind::CONDITIONER, "Conditioner", 2048 * MiB, 2048 * MiB, 256 * MiB},
            {ComponentKind::VAE, "VAE", 300 * MiB, 1024 * MiB, 1200 * MiB},
        };
        const Outcome out = run(components, two_gpus, MultiGpuMode::AUTO, 64);
        CHECK(same_names(out, "CUDA0&CUDA1", "CUDA0", "CUDA0") && resident(out.plan, 0) && !resident(out.plan, 1) && !resident(out.plan, 2),
              "the split diffusion model keeps its place when the text encoder and the VAE cannot move next to it");
    }

    check_mixed_backends();
    check_plan_warnings();
    check_random_machines();
    check_random_mixed_backends();

    printf("\n%s (%d failures)\n", failures ? "TEST FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
