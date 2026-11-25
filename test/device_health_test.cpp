// Test of the registry of device health (src/core/device_health.*): levels that rise one step per event, events
// of the engine that come too close together, the way back after each quiet period, the signals of outside
// watchers (sources, durations, critical), the same card seen by two backends, what the levels offer (the share of
// a device and the threads of the CPU) and the values, changed at run time and through the C API. A fake clock
// replaces waiting; no GPU and no model are needed.
// Build (from the repository root, after building the static library into build/):
//   c++ -std=c++17 -DGGML_MAX_NAME=160 -Isrc -Isrc/core -Iinclude -Ithirdparty -Iggml/include \
//       test/device_health_test.cpp build/libstable-diffusion.a build/ggml/src/libggml.a \
//       build/ggml/src/libggml-cpu.a build/ggml/src/libggml-base.a -ldl -lm -fopenmp -pthread \
//       -o device_health_test
// Run: ./device_health_test ; exit code 0 = all checks passed.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "core/device_health.h"
#include "stable-diffusion.h"

using sd::DeviceHealth;

static int failures = 0;

static void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "ok:  " : "FAIL:", what);
    failures += condition ? 0 : 1;
}

static DeviceHealth::Clock::time_point fake_now = DeviceHealth::Clock::time_point(std::chrono::hours(100));

static void advance(int seconds) {
    fake_now += std::chrono::seconds(seconds);
}

// An NVIDIA card seen by CUDA and by Vulkan, an integrated GPU, the CPU, the RAM and the disk.
static std::vector<DeviceHealth::Component> machine() {
    return {{"CUDA0", "0000:01:00.0"}, {"Vulkan0", "0000:01:00.0"}, {"Vulkan1", "0000:05:00.0"}, {"CPU", ""}, {"RAM", ""}, {"disk", ""}};
}

static void prepare(DeviceHealth& health, DeviceHealth::Settings settings = DeviceHealth::Settings()) {
    health.set_clock([] { return fake_now; });
    health.set_component_lister(machine);
    health.set_settings(settings);
    health.reset();
}

static DeviceHealth::Signal signal(const char* source, int level, int seconds, const char* reason = "too hot") {
    DeviceHealth::Signal result;
    result.source   = source;
    result.level    = level;
    result.reason   = reason;
    result.duration = std::chrono::seconds(seconds);
    return result;
}

static DeviceHealth::Signal critical_signal(const char* source, int seconds) {
    DeviceHealth::Signal result = signal(source, 0, seconds, "emergency stop");
    result.critical             = true;
    return result;
}

static const DeviceHealth::State* find(const std::vector<DeviceHealth::State>& states, const char* name) {
    for (const auto& state : states) {
        if (state.name == name) {
            return &state;
        }
    }
    return nullptr;
}

static bool same(double a, double b) {
    return std::fabs(a - b) < 1e-9;
}

static void test_starts_healthy() {
    DeviceHealth health;
    prepare(health);
    check(health.level("CUDA0") == 0 && same(health.share("CUDA0"), 1.0) && !health.excluded("CUDA0"),
          "a device starts healthy and offers all of its capacity");
    check(health.cpu_threads(8) == 8, "the CPU starts with all of its threads");
    check(!health.any_critical(), "nothing asks to stop");
    const auto states = health.snapshot();
    check(states.size() == 6, "the snapshot lists every component of the machine");
    const auto* ram = find(states, "RAM");
    check(ram != nullptr && ram->level == 0 && ram->share_percent == 100 && !ram->excluded && ram->reason.empty(),
          "a healthy component has level 0, all of its share and no reason");
}

static void test_engine_events_rise_one_step_each() {
    DeviceHealth health;
    prepare(health);
    const auto start = health.revision();
    health.note_strain("CUDA0", "compute failed");
    check(health.level("CUDA0") == 1 && same(health.share("CUDA0"), 0.85), "an event of the engine raises its device one level, to 85%");
    check(health.revision() != start, "a new level changes the revision");
    advance(5);
    health.note_strain("CUDA0", "compute failed");
    check(health.level("CUDA0") == 1, "an event closer than the spacing to the previous one counts once");
    advance(10);
    health.note_strain("CUDA0", "compute failed");
    check(health.level("CUDA0") == 2 && same(health.share("CUDA0"), 0.70), "a later event raises it again, to 70%");
    advance(10);
    health.note_strain("CUDA0", "compute failed");
    check(health.level("CUDA0") == 3 && same(health.share("CUDA0"), 0.55), "the third level offers 55%");
    advance(10);
    health.note_strain("CUDA0", "compute failed");
    check(health.level("CUDA0") == 4 && health.excluded("CUDA0") && same(health.share("CUDA0"), 0.0),
          "the event after the last level of strain excludes the device");
    advance(10);
    health.note_strain("CUDA0", "compute failed");
    check(health.level("CUDA0") == 4, "an excluded device stays excluded");
    check(health.level("Vulkan0") == 0, "an event of the engine stays with its backend, even on the same card");
}

static void test_quiet_periods_bring_the_levels_back() {
    DeviceHealth health;
    prepare(health);
    health.note_strain("Vulkan1", "compute failed");
    advance(10);
    health.note_strain("Vulkan1", "compute failed");
    advance(299);
    check(health.level("Vulkan1") == 2, "the level holds during the quiet period");
    const auto before = health.revision();
    advance(1);
    check(health.level("Vulkan1") == 1, "after 300 quiet seconds it comes down one level");
    check(health.revision() != before, "coming down changes the revision");
    advance(300);
    check(health.level("Vulkan1") == 0, "and one more after the next quiet period");

    health.note_strain("Vulkan1", "compute failed");
    advance(10);
    health.note_strain("Vulkan1", "compute failed");
    advance(600);
    check(health.level("Vulkan1") == 0, "two quiet periods bring it down two levels even when nobody asked in between");

    health.note_strain("Vulkan1", "compute failed");
    advance(10);
    health.note_strain("Vulkan1", "compute failed");
    advance(200);
    health.note_strain("Vulkan1", "memory guard cannot get under its threshold");
    advance(200);
    check(health.level("Vulkan1") == 3, "an event restarts the quiet period");
    advance(100);
    check(health.level("Vulkan1") == 2, "which then runs from that event");

    DeviceHealth other;
    prepare(other);
    other.note_strain("CUDA0", "compute failed");
    advance(5);
    other.note_strain("CUDA0", "compute failed");
    advance(296);
    check(other.level("CUDA0") == 1, "an event that counts once still restarts the quiet period");
    advance(4);
    check(other.level("CUDA0") == 0, "from its own time");
}

static void test_signals_hold_a_level_while_they_last() {
    DeviceHealth health;
    prepare(health);
    check(health.report("Vulkan1", signal("termico", 2, 60)), "a signal on a known device is accepted");
    check(health.level("Vulkan1") == 2, "a signal sets its level at once");
    for (int i = 0; i < 3; ++i) {
        advance(30);
        health.report("Vulkan1", signal("termico", 2, 60));
    }
    check(health.level("Vulkan1") == 2, "a watcher that repeats its signal keeps the level");
    advance(60);
    check(health.level("Vulkan1") == 2, "when the signal expires the level does not drop at once");
    advance(299);
    check(health.level("Vulkan1") == 2, "the quiet period counts from the end of the signal");
    advance(1);
    check(health.level("Vulkan1") == 1, "and then the level comes down one step");
    advance(300);
    check(health.level("Vulkan1") == 0, "and one more");

    check(health.report("Vulkan1", signal("termico", 2, 1000)), "a long signal is accepted");
    advance(600);
    check(health.level("Vulkan1") == 2, "a signal holds its level for as long as it lasts");
    advance(400);
    check(health.level("Vulkan1") == 2, "even right after it expires");
    advance(300);
    check(health.level("Vulkan1") == 1, "the way back starts when it ends");

    DeviceHealth other;
    prepare(other);
    for (int i = 0; i < 3; ++i) {
        other.note_strain("Vulkan1", "compute failed");
        advance(10);
    }
    other.report("Vulkan1", signal("termico", 1, 60));
    check(other.level("Vulkan1") == 3, "a signal lower than the current level does not lower it");
}

static void test_each_source_holds_one_signal() {
    DeviceHealth health;
    prepare(health);
    health.report("Vulkan1", signal("termico", 3, 600));
    health.report("Vulkan1", signal("agente", 1, 3600));
    check(health.level("Vulkan1") == 3, "two sources: the higher signal sets the level");
    advance(400);
    health.report("Vulkan1", signal("termico", 0, 60));
    check(health.level("Vulkan1") == 3, "a withdrawn signal leaves the level where it is");
    advance(299);
    check(health.level("Vulkan1") == 3, "the way back counts from the withdrawal");
    advance(1);
    check(health.level("Vulkan1") == 2, "and then comes down one step");
    advance(300);
    check(health.level("Vulkan1") == 1, "down to the level the other source still holds");
    advance(1200);
    check(health.level("Vulkan1") == 1, "which it keeps while that signal lasts");
    health.report("Vulkan1", signal("agente", 0, 60));
    advance(300);
    check(health.level("Vulkan1") == 0, "when it is withdrawn too, the device comes back");

    health.report("Vulkan1", signal("termico", 2, 600));
    advance(400);
    health.report("Vulkan1", signal("termico", 1, 600));
    check(health.level("Vulkan1") == 2, "a new signal of the same source replaces the old one without dropping the level at once");
    advance(299);
    check(health.level("Vulkan1") == 2, "the way back counts from the replacement");
    advance(1);
    check(health.level("Vulkan1") == 1, "down to the level of the new signal");
}

static void test_critical_signals() {
    DeviceHealth health;
    prepare(health);
    check(health.report("CPU", critical_signal("guardiano", 30)), "a critical signal is accepted");
    check(health.any_critical() && health.excluded("CPU"), "a critical signal asks to stop and excludes its component");
    const auto states = health.snapshot();
    const auto* cpu   = find(states, "CPU");
    check(cpu != nullptr && cpu->critical && cpu->reason.find("emergency stop") != std::string::npos,
          "the snapshot shows the critical component and why");
    advance(30);
    check(!health.any_critical(), "the critical signal ends when it expires");
    check(health.excluded("CPU"), "its component stays excluded for a quiet period");
    advance(300);
    check(health.level("CPU") == 3, "and then comes back one level at a time");
}

static void test_signals_reach_the_same_card() {
    DeviceHealth health;
    prepare(health);
    health.report("CUDA0", signal("termico", 2, 60));
    check(health.level("Vulkan0") == 2, "a signal reaches the same card seen by another backend");
    check(health.level("Vulkan1") == 0, "but not another card");
    check(health.report("vulkan1", signal("termico", 1, 60)) && health.level("Vulkan1") == 1, "device names are matched in any case");
}

static void test_invalid_signals_are_refused() {
    DeviceHealth health;
    prepare(health);
    check(!health.report("GPU9", signal("termico", 1, 60)), "a device the machine does not have is refused");
    check(!health.report("CUDA0", signal("termico", -1, 60)), "a negative level is refused");
    check(!health.report("CUDA0", signal("termico", 5, 60)), "a level above exclusion is refused");
    check(!health.report("CUDA0", signal("termico", 1, 0)), "a signal without a duration is refused");
    check(!health.report("CUDA0", signal("", 1, 60)), "a signal without a source is refused");
    check(health.level("CUDA0") == 0, "refused signals change nothing");
    for (int i = 0; i < 8; ++i) {
        health.report("RAM", signal(("watcher" + std::to_string(i)).c_str(), 1, 60));
    }
    check(!health.report("RAM", signal("one-too-many", 1, 60)), "a component keeps at most eight sources");
    check(health.report("RAM", signal("watcher0", 2, 60)), "a known source can still update its signal");
    health.report("disk", signal("termico", 1, 60, std::string(1000, 'x').c_str()));
    const auto states = health.snapshot();
    const auto* disk  = find(states, "disk");
    check(disk != nullptr && disk->reason.size() <= 200, "a long reason is cut");
}

static void test_resume() {
    DeviceHealth health;
    prepare(health);
    health.report("CUDA0", critical_signal("guardiano", 3600));
    for (int i = 0; i < 4; ++i) {
        health.note_strain("Vulkan1", "compute failed");
        advance(10);
    }
    const auto before = health.revision();
    health.resume();
    check(!health.any_critical(), "resume withdraws the critical signals");
    check(health.level("CUDA0") == 3 && health.level("Vulkan1") == 3, "and brings the excluded devices back to the last level of strain");
    check(health.revision() != before, "which changes the revision");
}

static void test_cpu_threads() {
    DeviceHealth health;
    prepare(health);
    health.note_strain("CPU", "too hot");
    check(health.cpu_threads(8) == 6, "a CPU under strain gives up a quarter of its threads per level: 8 to 6");
    advance(10);
    health.note_strain("CPU", "too hot");
    check(health.cpu_threads(8) == 4, "then 4");
    advance(10);
    health.note_strain("CPU", "too hot");
    check(health.cpu_threads(8) == 2, "then 2");
    check(health.cpu_threads(1) == 1 && health.cpu_threads(3) == 1, "never fewer than one thread");
    advance(10);
    health.note_strain("CPU", "too hot");
    check(health.cpu_threads(8) == 2, "an excluded CPU keeps the threads of the last level");
}

static void test_settings() {
    DeviceHealth health;
    prepare(health);
    DeviceHealth::Settings settings;
    settings.strain_levels = 0;
    check(!health.set_settings(settings), "at least one level of strain");
    settings              = DeviceHealth::Settings();
    settings.step_percent = 40;
    check(!health.set_settings(settings), "steps that would leave nothing before exclusion are refused");
    settings.step_percent = 1431655766;
    check(!health.set_settings(settings), "even when the step times the levels does not fit in an int");
    settings                         = DeviceHealth::Settings();
    settings.cpu_thread_step_percent = 1431655766;
    check(!health.set_settings(settings), "and so are such steps for the CPU threads");
    settings              = DeviceHealth::Settings();
    settings.quiet_period = std::chrono::seconds(0);
    check(!health.set_settings(settings), "a quiet period of zero is refused");
    settings.quiet_period = std::chrono::seconds(24 * 60 * 60 + 1);
    check(!health.set_settings(settings), "a quiet period longer than a day is refused");
    settings.quiet_period = std::chrono::seconds(24 * 60 * 60);
    check(health.set_settings(settings), "a day is accepted");
    settings               = DeviceHealth::Settings();
    settings.event_spacing = std::chrono::seconds(-1);
    check(!health.set_settings(settings), "a negative spacing of events is refused");
    settings.event_spacing = std::chrono::seconds(24 * 60 * 60 + 1);
    check(!health.set_settings(settings), "a spacing of events longer than a day is refused too");
    settings              = DeviceHealth::Settings();
    settings.step_percent = 20;
    check(health.set_settings(settings), "sensible settings are accepted");
    health.note_strain("CUDA0", "compute failed");
    check(same(health.share("CUDA0"), 0.80), "the step applies to the share");

    DeviceHealth off;
    DeviceHealth::Settings disabled;
    disabled.enabled = false;
    prepare(off, disabled);
    const auto revision = off.revision();
    off.note_strain("CUDA0", "compute failed");
    check(!off.report("CUDA0", signal("termico", 2, 60)), "a disabled registry refuses signals");
    check(off.level("CUDA0") == 0 && same(off.share("CUDA0"), 1.0) && off.revision() == revision,
          "and changes nothing for the events of the engine");
}

// New settings while components are under strain or excluded: each one stays in its state, and the same settings
// written again change nothing, so the models are not placed again for nothing.
static void test_settings_change_at_run_time() {
    DeviceHealth health;
    prepare(health);
    for (int i = 0; i < 4; ++i) {
        health.note_strain("CUDA0", "compute failed");
        advance(10);
    }
    for (int i = 0; i < 3; ++i) {
        health.note_strain("Vulkan1", "compute failed");
        advance(10);
    }
    health.report("CPU", signal("termico", 3, 600));
    const auto before = health.revision();
    check(health.set_settings(DeviceHealth::Settings()) && health.revision() == before,
          "the same settings written again change nothing, not even the revision");

    DeviceHealth::Settings two;
    two.strain_levels = 2;
    check(health.set_settings(two) && health.revision() != before, "fewer levels are accepted and change the revision");
    check(health.excluded("CUDA0"), "an excluded device stays excluded with fewer levels");
    check(health.level("Vulkan1") == 2 && !health.excluded("Vulkan1"),
          "a device on the last level of strain moves to the new last one, not to exclusion");
    check(health.level("CPU") == 2 && !health.excluded("CPU"), "and so does the level a signal holds");

    DeviceHealth::Settings five;
    five.strain_levels           = 5;
    five.step_percent            = 10;
    five.cpu_thread_step_percent = 15;
    check(health.set_settings(five), "more levels are accepted");
    check(health.excluded("CUDA0") && health.level("CUDA0") == 6, "an excluded device stays excluded with more levels");
    check(health.level("Vulkan1") == 2 && same(health.share("Vulkan1"), 0.80), "a device under strain keeps its level, with the new step");
    advance(400);
    check(health.level("CPU") == 2, "a signal keeps holding its level after the change, past a quiet period");

    DeviceHealth held;
    prepare(held);
    held.report("Vulkan1", signal("termico", 4, 3600));
    check(held.set_settings(five) && held.excluded("Vulkan1"), "an exclusion held by a signal stays an exclusion with more levels");
    advance(400);
    check(held.excluded("Vulkan1"), "and the signal keeps holding it past a quiet period");
}

static void test_snapshot_and_revision() {
    DeviceHealth health;
    prepare(health);
    health.note_strain("RPC0[host:50052]", "compute failed");
    auto states     = health.snapshot();
    const auto* rpc = find(states, "RPC0[host:50052]");
    check(rpc != nullptr && rpc->level == 1, "an event on a device the lister does not know is kept");
    health.note_strain("CUDA0", "compute failed");
    advance(100);
    states           = health.snapshot();
    const auto* cuda = find(states, "CUDA0");
    check(cuda != nullptr && cuda->level == 1 && cuda->share_percent == 85 && cuda->reason == "compute failed",
          "the snapshot gives the level, the share and the reason");
    check(cuda != nullptr && cuda->next_step_in == std::chrono::seconds(200), "and how long until the next step down");

    health.report("Vulkan1", signal("termico", 2, 60));
    const auto revision = health.revision();
    advance(10);
    health.report("Vulkan1", signal("termico", 2, 60));
    check(health.revision() == revision, "a repeated signal at the same level leaves the revision as it is");
    advance(20);
    states              = health.snapshot();
    const auto* vulkan1 = find(states, "Vulkan1");
    check(vulkan1 != nullptr && vulkan1->held_for == std::chrono::seconds(40) && vulkan1->next_step_in == std::chrono::seconds(0),
          "a level held by a signal tells how long the signal still lasts, and no step down is due");
}

// The C functions a host (the server) reads and resumes the registry with.
static void test_c_api() {
    DeviceHealth& health = sd::device_health();
    prepare(health);
    check(sd_report_device_pressure("Vulkan1", "termico", 2, false, "86 C", 60), "the C API reports a signal");
    check(sd_report_device_pressure("CPU", "guardiano", 0, true, "emergency stop", 30), "and a critical one");
    check(!sd_report_device_pressure(nullptr, "termico", 1, false, "x", 60) && !sd_report_device_pressure("CPU", nullptr, 1, false, "x", 60) &&
              !sd_report_device_pressure("GPU9", "termico", 1, false, "x", 60) && !sd_report_device_pressure("CPU", "termico", 1, false, nullptr, 0),
          "and refuses a missing component or source, an unknown device and a signal without a duration");
    const size_t count = sd_get_device_health(nullptr, 0);
    check(count == 6, "the C API counts the components");
    std::vector<sd_device_health_t> entries(count + 2);
    check(sd_get_device_health(entries.data(), 2) == count, "and returns the count even when the array is shorter");
    sd_get_device_health(entries.data(), entries.size());
    const sd_device_health_t* vulkan1 = nullptr;
    const sd_device_health_t* cpu     = nullptr;
    for (size_t i = 0; i < count; ++i) {
        if (std::string(entries[i].name) == "Vulkan1") {
            vulkan1 = &entries[i];
        } else if (std::string(entries[i].name) == "CPU") {
            cpu = &entries[i];
        }
    }
    check(vulkan1 != nullptr && vulkan1->level == 2 && vulkan1->strain_levels == 3 && vulkan1->share_percent == 70 &&
              !vulkan1->excluded && !vulkan1->critical && std::string(vulkan1->reason) == "termico: 86 C" && vulkan1->held_ms == 60000,
          "an entry carries the level, the share, the reason and how long its signal holds");
    check(cpu != nullptr && cpu->excluded && cpu->critical && cpu->share_percent == 0, "a critical component is excluded");
    sd_resume_device_health();
    check(!health.any_critical() && health.level("CPU") == 3, "the C API resumes the registry");
    health.set_clock(nullptr);
    health.set_component_lister(nullptr);
    health.reset();
}

// The C functions a host reads and changes the values with: read them, change some, write them back.
static void test_c_api_settings() {
    DeviceHealth& health = sd::device_health();
    prepare(health);
    sd_device_health_settings_t settings = {};
    sd_get_device_health_settings(&settings);
    check(settings.strain_levels == 3 && settings.step_percent == 15 && settings.cpu_thread_step_percent == 25 &&
              settings.quiet_period_seconds == 300 && settings.event_spacing_seconds == 10,
          "the C API reads the values: 3 levels, 15% and 25% per level, 300 s, 10 s");
    settings.strain_levels           = 2;
    settings.step_percent            = 20;
    settings.cpu_thread_step_percent = 30;
    settings.quiet_period_seconds    = 60;
    settings.event_spacing_seconds   = 5;
    check(sd_set_device_health_settings(&settings), "and writes new ones");
    health.note_strain("CUDA0", "compute failed");
    check(same(health.share("CUDA0"), 0.80), "which the registry uses: the step");
    advance(59);
    check(health.level("CUDA0") == 1, "and the quiet period");
    advance(1);
    check(health.level("CUDA0") == 0, "to the second");
    health.note_strain("CPU", "too hot");
    check(health.cpu_threads(10) == 7, "the CPU step: 10 threads to 7");
    advance(4);
    health.note_strain("CPU", "too hot");
    check(health.level("CPU") == 1, "and the spacing: an event 4 s later counts once");
    advance(1);
    health.note_strain("CPU", "too hot");
    check(health.level("CPU") == 2, "one 5 s later counts again");
    sd_device_health_settings_t read_back = {};
    sd_get_device_health_settings(&read_back);
    check(read_back.strain_levels == 2 && read_back.step_percent == 20 && read_back.cpu_thread_step_percent == 30 &&
              read_back.quiet_period_seconds == 60 && read_back.event_spacing_seconds == 5,
          "the values read back are the ones written");
    const size_t count = sd_get_device_health(nullptr, 0);
    std::vector<sd_device_health_t> entries(count);
    sd_get_device_health(entries.data(), entries.size());
    check(count > 0 && entries[0].strain_levels == 2, "the entries carry the new number of levels");

    sd_device_health_settings_t wrong = read_back;
    wrong.step_percent                = 50;
    check(!sd_set_device_health_settings(&wrong), "values that make no sense are refused: 2 levels of 50% leave nothing");
    wrong                      = read_back;
    wrong.quiet_period_seconds = 24 * 60 * 60 + 1;
    check(!sd_set_device_health_settings(&wrong), "as is a quiet period longer than a day");
    sd_get_device_health_settings(&read_back);
    check(read_back.step_percent == 20 && read_back.quiet_period_seconds == 60, "and change nothing");
    check(!sd_set_device_health_settings(nullptr), "a missing pointer is refused");
    sd_get_device_health_settings(nullptr);
    check(health.settings().strain_levels == 2, "and reading into none changes nothing either");

    health.set_settings(DeviceHealth::Settings());
    health.set_clock(nullptr);
    health.set_component_lister(nullptr);
    health.reset();
}

static void test_backend_names() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check(cpu != nullptr && sd::device_health_name(cpu) == "CPU", "a backend is named by its device");
    DeviceHealth health;
    prepare(health);
    for (int i = 0; i < 4; ++i) {
        health.note_strain(sd::device_health_name(cpu), "compute failed");
        advance(10);
    }
    check(health.excluded(cpu), "and its device can be asked about through the backend");
    if (cpu != nullptr) {
        ggml_backend_free(cpu);
    }
    check(&sd::device_health() == &sd::device_health(), "the process has one registry");
}

// What the engine reports, through the registry of the process.
static void test_engine_hooks() {
    DeviceHealth& health = sd::device_health();
    prepare(health);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    sd::note_compute_status(cpu, GGML_STATUS_ALLOC_FAILED);
    check(health.level("CPU") == 0, "running out of memory is not a matter of health: the ladder and the guard handle it");
    sd::note_compute_status(cpu, GGML_STATUS_ABORTED);
    sd::note_compute_status(cpu, GGML_STATUS_SUCCESS);
    check(health.level("CPU") == 0, "nor is a compute that was stopped on request, or one that worked");
    sd::note_compute_status(cpu, GGML_STATUS_FAILED);
    check(health.level("CPU") == 1, "a compute that failed raises its device");
    advance(10);
    sd::note_backend_start_failure(ggml_backend_get_device(cpu));
    check(health.level("CPU") == 2, "so does a backend that does not start");
    auto states     = health.snapshot();
    const auto* dev = find(states, "CPU");
    check(dev != nullptr && dev->reason.find("did not start") != std::string::npos, "with the reason of the latest event");
    sd::note_memory_guard_unreachable(cpu, 90);
    check(health.level("RAM") == 1 && health.level("CPU") == 2, "the memory guard of the CPU speaks for the RAM");
    check(sd::health_scaled_bytes(cpu, 1000) == 700, "a budget on a device is scaled by its share");
    check(same(health.ram_share(), 0.85), "the RAM offers its share");
    for (int i = 0; i < 3; ++i) {
        advance(10);
        sd::note_memory_guard_unreachable(cpu, 90);
    }
    check(health.level("RAM") == 4 && same(health.ram_share(), 0.55),
          "but nothing runs without RAM: past the last level of strain it keeps the smallest share");
    sd::note_disk_read_failure("model.safetensors");
    states           = health.snapshot();
    const auto* disk = find(states, "disk");
    check(disk != nullptr && disk->level == 1 && disk->reason.find("model.safetensors") != std::string::npos,
          "a failed read raises the disk and names the file");
    if (cpu != nullptr) {
        ggml_backend_free(cpu);
    }
    health.set_clock(nullptr);
    health.set_component_lister(nullptr);
    health.reset();
}

int main() {
    test_starts_healthy();
    test_engine_events_rise_one_step_each();
    test_quiet_periods_bring_the_levels_back();
    test_signals_hold_a_level_while_they_last();
    test_each_source_holds_one_signal();
    test_critical_signals();
    test_signals_reach_the_same_card();
    test_invalid_signals_are_refused();
    test_resume();
    test_cpu_threads();
    test_settings();
    test_settings_change_at_run_time();
    test_snapshot_and_revision();
    test_backend_names();
    test_engine_hooks();
    test_c_api();
    test_c_api_settings();

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED", failures);
    return failures == 0 ? 0 : 1;
}
