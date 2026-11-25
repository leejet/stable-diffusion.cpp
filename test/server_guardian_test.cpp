// Test of the guardian of the server (examples/server/guardian.*): when it stops all work (a component reported as
// critical, or no compute device left because every GPU and the CPU are excluded by device health), when it
// resumes, how long it tells clients to wait, that the halt starts once, the operator's resume, what
// /sdcpp/v1/health shows, and the watcher that stops the work when no request comes. The registry of device health
// is replaced by scripted components, so no model or device is needed.
// Build (from the repository root, after building the library into build/); the sources go in one command:
//   c++ -std=c++17 -Iexamples -Iexamples/server -Iinclude -Ithirdparty -Isrc test/server_guardian_test.cpp
//       examples/server/guardian.cpp examples/common/log.cpp
//       -Lbuild/bin -lstable-diffusion -Wl,-rpath,build/bin -pthread -o server_guardian_test
// Run: ./server_guardian_test ; exit code 0 = all checks passed.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"

#include "guardian.h"

static int failures = 0;

static void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "ok:  " : "FAIL:", what);
    failures += condition ? 0 : 1;
}

static sd_device_health_t component(const char* name, int level, const char* reason = "") {
    sd_device_health_t entry{};
    std::snprintf(entry.name, sizeof(entry.name), "%s", name);
    std::snprintf(entry.reason, sizeof(entry.reason), "%s", reason);
    entry.level         = level;
    entry.strain_levels = 3;
    entry.excluded      = level > 3;
    entry.share_percent = level > 3 ? 0 : 100 - 15 * level;
    return entry;
}

struct Machine {
    std::vector<sd_device_health_t> components = {component("Vulkan0", 0), component("CPU", 0), component("RAM", 0), component("disk", 0)};
    int resumes                                = 0;

    Guardian::Hooks hooks() {
        Guardian::Hooks hooks;
        hooks.read_health   = [this] { return components; };
        hooks.resume_health = [this] { ++resumes; };
        return hooks;
    }
    sd_device_health_t& find(const char* name) {
        for (auto& entry : components) {
            if (std::strcmp(entry.name, name) == 0) {
                return entry;
            }
        }
        return components.front();
    }
};

static void test_healthy_and_disabled() {
    Machine machine;
    Guardian guardian(true, machine.hooks());
    check(!guardian.evaluate().halted, "a healthy machine is not halted");
    machine.find("Vulkan0") = component("Vulkan0", 2, "termico: 84 C");
    check(!guardian.evaluate().halted, "a device under strain does not halt the server: it only gives up work");

    Machine other;
    other.find("CPU")          = component("CPU", 4, "guardiano: stop");
    other.find("CPU").critical = true;
    Guardian off(false, other.hooks());
    check(!off.enabled() && !off.evaluate().halted, "a guardian that is off never halts the server");
}

static void test_critical_signal() {
    Machine machine;
    Guardian guardian(true, machine.hooks());
    machine.find("Vulkan0")          = component("Vulkan0", 4, "termico: 95 C");
    machine.find("Vulkan0").critical = true;
    machine.find("Vulkan0").held_ms  = 42500;
    bool started                     = false;
    const auto state                 = guardian.evaluate(&started);
    check(state.halted && started, "a critical component halts the server, and this evaluation is the one that starts the halt");
    check(state.reason.find("Vulkan0") != std::string::npos && state.reason.find("95 C") != std::string::npos,
          "the reason names the component and why");
    check(state.retry_after_seconds == 43, "clients are told to come back when the critical signal ends");
    bool again = true;
    check(guardian.evaluate(&again).halted && !again, "the halt starts once");
    machine.find("Vulkan0").held_ms = 2000;
    check(guardian.evaluate().retry_after_seconds == 10, "and never sooner than ten seconds");
    machine.find("Vulkan0").critical = false;
    machine.find("Vulkan0").held_ms  = 0;
    check(!guardian.evaluate().halted, "when the critical signal ends, the server resumes, even if the device is still excluded");
}

// A refused request reads the registry too, so it may be the first to see a halt; the start of the halt must still
// reach whoever stops the work, once.
static void test_halt_seen_first_by_a_request() {
    Machine machine;
    Guardian guardian(true, machine.hooks());
    machine.find("Vulkan0")          = component("Vulkan0", 4, "termico: 95 C");
    machine.find("Vulkan0").critical = true;
    httplib::Response res;
    check(refuse_while_halted(&guardian, res), "a request is the first to see the halt");
    bool started = false;
    check(guardian.evaluate(&started).halted && started, "the start of the halt still reaches the first caller that asks for it");
    bool again = true;
    check(guardian.evaluate(&again).halted && !again, "only once");

    machine.find("Vulkan0").critical = false;
    check(!guardian.evaluate().halted, "the cause ends");
    machine.find("Vulkan0").critical = true;
    check(guardian.evaluate().halted, "a second halt, seen by a request");
    machine.find("Vulkan0").critical = false;
    check(!guardian.evaluate().halted, "ends before anyone asked for its start");
    bool stale = true;
    guardian.evaluate(&stale);
    check(!stale, "a halt that has ended leaves nothing to stop");
}

static void test_no_compute_device_left() {
    Machine machine;
    Guardian guardian(true, machine.hooks());
    machine.find("Vulkan0") = component("Vulkan0", 4, "compute failed");
    check(!guardian.evaluate().halted, "with the GPU excluded the CPU is still there");
    machine.find("RAM")  = component("RAM", 4, "memory guard");
    machine.find("disk") = component("disk", 4, "read failed");
    check(!guardian.evaluate().halted, "the RAM and the disk are not compute devices");
    machine.find("CPU")                  = component("CPU", 4, "compute failed");
    machine.find("Vulkan0").next_step_ms = 120000;
    machine.find("CPU").next_step_ms     = 60000;
    bool started                         = false;
    const auto state                     = guardian.evaluate(&started);
    check(state.halted && started && state.reason.find("no compute device") != std::string::npos,
          "with every GPU and the CPU excluded the server halts");
    check(state.retry_after_seconds == 60, "and tells clients to come back when the first device returns");
    machine.find("CPU") = component("CPU", 3, "compute failed");
    check(!guardian.evaluate().halted, "it resumes as soon as one compute device is back");
}

static void test_resume_and_json() {
    Machine machine;
    Guardian guardian(true, machine.hooks());
    machine.find("CPU")          = component("CPU", 4, "guardiano: stop");
    machine.find("CPU").critical = true;
    check(guardian.evaluate().halted, "halted by a critical CPU");
    auto health = guardian.health_json();
    check(health["guardian"]["halted"] == true && health["guardian"]["enabled"] == true &&
              health["guardian"]["reason"].get<std::string>().find("CPU") != std::string::npos && health["guardian"]["since"].get<int64_t>() > 0,
          "the health shows the halt, why and since when");
    check(health["devices"].is_array() && health["devices"].size() == 4 && health["devices"][1]["name"] == "CPU" &&
              health["devices"][1]["excluded"] == true && health["devices"][1]["critical"] == true && health["devices"][1]["share_percent"] == 0,
          "and every component with its level and share");
    machine.find("CPU") = component("CPU", 3, "resumed by an operator");
    const auto state    = guardian.resume();
    check(machine.resumes == 1 && !state.halted, "the operator's resume reaches the registry and the server resumes");
    health = guardian.health_json();
    check(health["guardian"]["halted"] == false && health["guardian"]["reason"] == "", "and the health shows it");
}

static void test_refusal() {
    Machine machine;
    Guardian guardian(true, machine.hooks());
    httplib::Response res;
    check(!refuse_while_halted(&guardian, res), "a request goes through while the server is not halted");
    check(!refuse_while_halted(nullptr, res), "and when there is no guardian");
    machine.find("Vulkan0")          = component("Vulkan0", 4, "termico: 95 C");
    machine.find("Vulkan0").critical = true;
    machine.find("Vulkan0").held_ms  = 30000;
    check(refuse_while_halted(&guardian, res), "a request is refused while the server is halted");
    const auto body = nlohmann::json::parse(res.body, nullptr, false);
    check(res.status == 503 && res.get_header_value("Retry-After") == "30" && body.is_object() && body["code"] == "halted" &&
              body["error"].get<std::string>().find("95 C") != std::string::npos,
          "with 503, the code halted, the reason and Retry-After");
}

// The body of POST /sdcpp/v1/devices/{device}/pressure.
static void test_pressure_report() {
    auto parse = [](const char* text, PressureReport& report, std::string& error) {
        report = PressureReport();
        error.clear();
        return parse_pressure_report(nlohmann::json::parse(text), 3, report, error);
    };
    PressureReport report;
    std::string error;
    check(parse(R"({"level": 2})", report, error) && report.level == 2 && !report.critical && report.source == "http" &&
              report.duration_seconds == 60 && report.reason.empty(),
          "a level of strain, with the defaults: source http, one minute, no reason");
    check(parse(R"({"level": 0, "source": "termico"})", report, error) && report.level == 0 && report.source == "termico",
          "level 0 withdraws the signal of a source");
    check(parse(R"({"level": "excluded"})", report, error) && report.level == 4 && !report.critical, "excluded is the level after the last strain");
    check(parse(R"({"level": 4})", report, error) && report.level == 4 && !report.critical, "and so is that number");
    check(parse(R"({"level": "critical", "reason": "95 C", "duration_seconds": 30, "source": "termico.cpu-1"})", report, error) &&
              report.level == 4 && report.critical && report.reason == "95 C" && report.duration_seconds == 30 && report.source == "termico.cpu-1",
          "critical excludes the device and asks to stop all work, for the time given");
    check(parse(R"({"level": 1, "duration_seconds": 86400})", report, error) && report.duration_seconds == 86400, "a signal lasts a day at most");

    const char* invalid[] = {
        R"([])",
        R"({})",
        R"({"level": 5})",
        R"({"level": -1})",
        R"({"level": 2.5})",
        R"({"level": "hot"})",
        R"({"level": true})",
        R"({"level": 1, "duration_seconds": 0})",
        R"({"level": 1, "duration_seconds": 86401})",
        R"({"level": 1, "duration_seconds": "60"})",
        R"({"level": 1, "source": ""})",
        R"({"level": 1, "source": "a b"})",
        R"({"level": 1, "source": "123456789012345678901234567890123"})",
        R"({"level": 1, "reason": 5})",
        R"({"level": 1, "reason": "two\nlines"})",
        R"({"level": 1, "duration": 30})",
    };
    for (const char* text : invalid) {
        const bool accepted = parse(text, report, error);
        check(!accepted && !error.empty(), (std::string("refused, with the reason: ") + text + (error.empty() ? "" : "  -> " + error)).c_str());
    }
}

static bool wait_until(const std::function<bool()>& condition, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!condition()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
}

// The work must stop even when no request comes: the watcher reads the registry on a thread of its own.
static void test_watcher() {
    std::mutex mutex;
    bool critical = false;
    std::atomic<int> reads{0};
    Guardian::Hooks hooks;
    hooks.read_health = [&] {
        std::lock_guard<std::mutex> lock(mutex);
        ++reads;
        std::vector<sd_device_health_t> components = {component("Vulkan0", critical ? 4 : 0, critical ? "termico: 95 C" : ""), component("CPU", 0)};
        components[0].critical                     = critical;
        return components;
    };
    auto set_critical = [&](bool value) {
        std::lock_guard<std::mutex> lock(mutex);
        critical = value;
    };
    std::atomic<int> halts{0};
    std::string reason;
    auto on_halt = [&](const Guardian::State& state) {
        std::lock_guard<std::mutex> lock(mutex);
        reason = state.reason;
        ++halts;
    };

    {
        Guardian guardian(true, hooks);
        guardian.watch(on_halt, std::chrono::milliseconds(20));
        set_critical(true);
        check(wait_until([&] { return halts == 1; }, std::chrono::seconds(5)), "the watcher sees a halt by itself and stops the work");
        {
            std::lock_guard<std::mutex> lock(mutex);
            check(reason.find("95 C") != std::string::npos, "it knows why the server halted");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        check(halts == 1, "it stops the work once per halt");
        set_critical(false);
        check(wait_until([&] { return !guardian.evaluate().halted; }, std::chrono::seconds(5)), "the cause ends");
        set_critical(true);
        check(wait_until([&] { return halts == 2; }, std::chrono::seconds(5)), "a second halt stops the work again");
    }

    set_critical(false);
    const int before = halts;
    {
        auto guardian = std::make_unique<Guardian>(true, hooks);
        reads         = 0;
        guardian->watch(on_halt, std::chrono::hours(1));
        check(wait_until([&] { return reads >= 1; }, std::chrono::seconds(5)), "the watcher reads the registry when it starts");
        set_critical(true);
        guardian->check_now();
        check(wait_until([&] { return halts == before + 1; }, std::chrono::seconds(5)),
              "check_now makes it read the registry at once, without waiting for the period");
        const auto start = std::chrono::steady_clock::now();
        guardian.reset();
        check(std::chrono::steady_clock::now() - start < std::chrono::seconds(2), "the guardian ends its watcher at once when it ends");
    }

    {
        Guardian off(false, hooks);
        const int reads_before = reads;
        off.watch(on_halt, std::chrono::milliseconds(20));
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        check(halts == before + 1 && reads == reads_before, "a guardian that is off does not watch");
    }

    {
        std::atomic<bool> thrown{false};
        Guardian::Hooks failing = hooks;
        failing.read_health     = [&] {
            if (!thrown.exchange(true)) {
                throw std::runtime_error("the registry cannot be read");
            }
            return hooks.read_health();
        };
        Guardian guardian(true, failing);
        guardian.watch(on_halt, std::chrono::milliseconds(20));
        check(wait_until([&] { return halts == before + 2; }, std::chrono::seconds(5)) && thrown,
              "a failure to read the registry does not end the watcher");
    }

    {
        set_critical(false);
        Guardian guardian(true, hooks);
        guardian.watch(on_halt, std::chrono::milliseconds(20));
        const int handled = halts;
        guardian.stop_watching();
        set_critical(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        check(halts == handled, "after stop_watching the watcher handles no halt, so the server can end it before what on_halt uses");
        guardian.stop_watching();
    }
}

int main() {
    test_healthy_and_disabled();
    test_critical_signal();
    test_halt_seen_first_by_a_request();
    test_no_compute_device_left();
    test_resume_and_json();
    test_refusal();
    test_pressure_report();
    test_watcher();
    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED", failures);
    return failures == 0 ? 0 : 1;
}
