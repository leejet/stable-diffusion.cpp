// httplib.h must come before any header that reaches <windows.h> (common/log.h does): windows.h pulls in winsock.h,
// which clashes with the winsock2.h included by httplib.h.
#include "httplib.h"

#include "guardian.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <exception>
#include <limits>
#include <utility>

#include "common/log.h"

static bool is_compute_device(const sd_device_health_t& component) {
    std::string name = component.name;
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return name != "ram" && name != "disk";
}

std::vector<sd_device_health_t> read_device_health() {
    std::vector<sd_device_health_t> components(sd_get_device_health(nullptr, 0));
    const size_t count = sd_get_device_health(components.data(), components.size());
    components.resize(std::min(count, components.size()));
    return components;
}

nlohmann::json device_health_json(const sd_device_health_t& component) {
    return {{"name", component.name},
            {"level", component.level},
            {"strain_levels", component.strain_levels},
            {"share_percent", component.share_percent},
            {"excluded", component.excluded},
            {"critical", component.critical},
            {"reason", component.reason},
            {"held_ms", component.held_ms},
            {"next_step_ms", component.next_step_ms}};
}

static std::string json_text(const nlohmann::json& value) {
    return value.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

Guardian::Guardian(bool enabled, const Hooks& hooks)
    : enabled_(enabled), hooks_(hooks) {}

Guardian::~Guardian() {
    stop_watching();
}

void Guardian::watch(std::function<void(const State&)> on_halt, std::chrono::milliseconds period) {
    if (!enabled_ || watcher_.joinable()) {
        return;
    }
    watcher_ = std::thread([this, on_halt = std::move(on_halt), period]() {
        std::unique_lock<std::mutex> lock(watch_mutex_);
        while (!watch_stop_) {
            lock.unlock();
            // An exception must not end the thread that stops the work, nor the server with it.
            try {
                bool started      = false;
                const State state = evaluate(&started);
                if (started && on_halt) {
                    on_halt(state);
                }
            } catch (const std::exception& error) {
                LOG_ERROR("guardian: %s", error.what());
            }
            lock.lock();
            watch_wake_.wait_for(lock, period, [this] { return watch_stop_ || watch_check_; });
            watch_check_ = false;
        }
    });
}

void Guardian::stop_watching() {
    {
        std::lock_guard<std::mutex> lock(watch_mutex_);
        watch_stop_ = true;
    }
    watch_wake_.notify_all();
    if (watcher_.joinable()) {
        watcher_.join();
    }
}

void Guardian::check_now() {
    {
        std::lock_guard<std::mutex> lock(watch_mutex_);
        watch_check_ = true;
    }
    watch_wake_.notify_all();
}

Guardian::State Guardian::evaluate(bool* started_halt) {
    if (started_halt != nullptr) {
        *started_halt = false;
    }
    std::vector<sd_device_health_t> components = hooks_.read_health ? hooks_.read_health() : read_device_health();

    std::string reason;
    int64_t wait_ms = 0;
    for (const auto& component : components) {
        if (component.critical) {
            reason += (reason.empty() ? "" : "; ") + std::string(component.name) + " is critical" +
                      (component.reason[0] != '\0' ? ": " + std::string(component.reason) : std::string());
            wait_ms = std::max(wait_ms, component.held_ms);
        }
    }
    if (reason.empty()) {
        bool any_compute  = false;
        bool all_excluded = true;
        int64_t soonest   = std::numeric_limits<int64_t>::max();
        for (const auto& component : components) {
            if (!is_compute_device(component)) {
                continue;
            }
            any_compute = true;
            if (!component.excluded) {
                all_excluded = false;
                break;
            }
            soonest = std::min(soonest, component.held_ms > 0 ? component.held_ms : component.next_step_ms);
        }
        if (any_compute && all_excluded) {
            reason  = "no compute device left: device health excludes every GPU and the CPU";
            wait_ms = soonest == std::numeric_limits<int64_t>::max() ? 0 : soonest;
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    components_       = std::move(components);
    const bool halted = enabled_ && !reason.empty();
    if (halted && !halted_) {
        since_        = static_cast<int64_t>(std::time(nullptr));
        stop_pending_ = true;
        LOG_WARN("guardian: the server stops all work: %s", reason.c_str());
    } else if (!halted && halted_) {
        stop_pending_ = false;
        LOG_INFO("guardian: the cause has ended, the server resumes work");
    }
    if (started_halt != nullptr && stop_pending_) {
        *started_halt = true;
        stop_pending_ = false;
    }
    halted_      = halted;
    reason_      = halted ? reason : std::string();
    since_       = halted ? since_ : 0;
    retry_after_ = halted ? std::max(10, static_cast<int>((wait_ms + 999) / 1000)) : 0;

    State state;
    state.halted              = halted_;
    state.reason              = reason_;
    state.retry_after_seconds = retry_after_;
    return state;
}

Guardian::State Guardian::resume() {
    if (hooks_.resume_health) {
        hooks_.resume_health();
    } else {
        sd_resume_device_health();
    }
    LOG_INFO("guardian: an operator says the machine was checked");
    return evaluate();
}

nlohmann::json Guardian::health_json() {
    evaluate();
    std::lock_guard<std::mutex> lock(mutex_);
    nlohmann::json devices = nlohmann::json::array();
    for (const auto& component : components_) {
        devices.push_back(device_health_json(component));
    }
    nlohmann::json guardian = {{"enabled", enabled_},
                               {"halted", halted_},
                               {"reason", reason_},
                               {"since", since_},
                               {"retry_after_seconds", retry_after_}};
    return {{"guardian", std::move(guardian)}, {"devices", std::move(devices)}};
}

static bool valid_source_name(const std::string& name) {
    return !name.empty() && name.size() <= 32 && std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return std::isalnum(c) != 0 || c == '.' || c == '_' || c == '-';
    });
}

bool parse_pressure_report(const nlohmann::json& body, int strain_levels, PressureReport& report, std::string& error) {
    report = PressureReport();
    if (!body.is_object()) {
        error = "the body must be a JSON object";
        return false;
    }
    for (const auto& field : body.items()) {
        if (field.key() != "level" && field.key() != "source" && field.key() != "reason" && field.key() != "duration_seconds") {
            error = "unknown field '" + field.key() + "': the fields are level, source, reason and duration_seconds";
            return false;
        }
    }

    const int top                = strain_levels + 1;
    const std::string level_rule = "level must be 0 (withdraw the signal of this source) to " + std::to_string(top) +
                                   ", \"excluded\" or \"critical\"";
    const auto level = body.find("level");
    if (level == body.end()) {
        error = level_rule;
        return false;
    }
    if (level->is_number_integer()) {
        const int64_t value = level->get<int64_t>();
        if (value < 0 || value > top) {
            error = level_rule;
            return false;
        }
        report.level = static_cast<int>(value);
    } else if (level->is_string() && *level == "excluded") {
        report.level = top;
    } else if (level->is_string() && *level == "critical") {
        report.level    = top;
        report.critical = true;
    } else {
        error = level_rule;
        return false;
    }

    if (const auto source = body.find("source"); source != body.end()) {
        if (!source->is_string() || !valid_source_name(source->get<std::string>())) {
            error = "source must be 1 to 32 letters, digits, '.', '_' or '-'";
            return false;
        }
        report.source = source->get<std::string>();
    }
    if (const auto reason = body.find("reason"); reason != body.end()) {
        const bool one_line = reason->is_string() && std::none_of(reason->get_ref<const std::string&>().begin(),
                                                                  reason->get_ref<const std::string&>().end(),
                                                                  [](unsigned char c) { return c < 0x20 || c == 0x7f; });
        if (!one_line) {
            error = "reason must be one line of text";
            return false;
        }
        report.reason = reason->get<std::string>();
    }
    if (const auto duration = body.find("duration_seconds"); duration != body.end()) {
        if (!duration->is_number_integer() || duration->get<int64_t>() < 1 || duration->get<int64_t>() > 86400) {
            error = "duration_seconds must be a whole number of seconds from 1 to 86400";
            return false;
        }
        report.duration_seconds = static_cast<int>(duration->get<int64_t>());
    }
    return true;
}

bool refuse_while_halted(Guardian* guardian, httplib::Response& res) {
    if (guardian == nullptr) {
        return false;
    }
    const auto state = guardian->evaluate();
    if (!state.halted) {
        return false;
    }
    res.status                  = 503;
    nlohmann::json body         = {{"error", "the server has stopped all work: " + state.reason}, {"code", "halted"}};
    body["retry_after_seconds"] = state.retry_after_seconds;
    res.set_header("Retry-After", std::to_string(state.retry_after_seconds));
    res.set_content(json_text(body), "application/json");
    return true;
}
