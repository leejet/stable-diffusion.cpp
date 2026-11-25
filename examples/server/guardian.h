#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <json.hpp>
#include "stable-diffusion.h"

namespace httplib {
    struct Response;
}

// Stops all the work of the server when the machine cannot take it: a watcher reports a component as critical, or
// no compute device is left because device health excludes every GPU and the CPU. While it is halted, the
// generation in progress is stopped, new requests are refused with 503 and the queued jobs wait; it resumes on its
// own when the cause ends, or when an operator says the machine was checked. It only reads the registry of device
// health, so asking it never waits for a compute device.
//
// It is a layer on top of the software, not a safety system: a stop that protects people or plants belongs to
// independent, certified hardware.
class Guardian {
public:
    using Clock = std::chrono::steady_clock;

    // Replaceable so that the logic can be tested without a registry. An empty function selects the real one.
    struct Hooks {
        // The components and how they are doing. Default: sd_get_device_health().
        std::function<std::vector<sd_device_health_t>()> read_health;
        // Default: sd_resume_device_health().
        std::function<void()> resume_health;
    };

    struct State {
        bool halted = false;
        std::string reason;
        // How long a client should wait before trying again; 0 when the server is not halted.
        int retry_after_seconds = 0;
    };

    explicit Guardian(bool enabled, const Hooks& hooks = Hooks());
    Guardian(const Guardian&)            = delete;
    Guardian& operator=(const Guardian&) = delete;
    ~Guardian();

    // Reads the registry every period on a thread of its own and calls on_halt there, once per halt, so that the
    // work in progress stops even when no request comes. The thread ends with the guardian. Nothing happens when
    // the guardian is off or already watches.
    void watch(std::function<void(const State&)> on_halt, std::chrono::milliseconds period = std::chrono::seconds(1));
    // Makes the watcher read the registry now, for example right after a report.
    void check_now();
    // Ends the watcher and waits for it; call it before what on_halt uses goes away. The destructor calls it too.
    void stop_watching();

    // Reads the registry and updates the state. started_halt is set, once per halt, for the first caller that passes
    // it, so that this caller stops the generation in progress once; a refused request reads the registry too but
    // passes nothing, so it does not use up the start of the halt.
    State evaluate(bool* started_halt = nullptr);
    // An operator says the machine was checked; returns the state after it.
    State resume();
    bool enabled() const {
        return enabled_;
    }
    // {"guardian": {...}, "devices": [...]}, fresh, for /sdcpp/v1/health.
    nlohmann::json health_json();

private:
    const bool enabled_;
    const Hooks hooks_;
    std::mutex mutex_;
    bool halted_       = false;
    bool stop_pending_ = false;  // the start of the halt has not reached a caller that stops the work yet
    int64_t since_     = 0;      // unix seconds of the start of the halt
    int retry_after_   = 0;
    std::string reason_;
    std::vector<sd_device_health_t> components_;  // as the last evaluation read them

    std::mutex watch_mutex_;
    std::condition_variable watch_wake_;
    bool watch_stop_  = false;
    bool watch_check_ = false;
    std::thread watcher_;
};

// Answers 503 with the code halted, the reason and Retry-After when the guardian has halted the server; false, and
// res untouched, otherwise or without a guardian.
bool refuse_while_halted(Guardian* guardian, httplib::Response& res);

// The components of device health as the registry lists them now, and one of them as /sdcpp/v1/health shows it.
std::vector<sd_device_health_t> read_device_health();
nlohmann::json device_health_json(const sd_device_health_t& component);

// What an outside watcher reports with POST /sdcpp/v1/devices/{device}/pressure, ready for
// sd_report_device_pressure().
struct PressureReport {
    std::string source = "http";
    int level          = 0;
    bool critical      = false;
    std::string reason;
    int duration_seconds = 60;
};

// Reads the body of that request for a device with strain_levels levels of strain; false, with the reason in
// error, when it is not valid.
bool parse_pressure_report(const nlohmann::json& body, int strain_levels, PressureReport& report, std::string& error);
