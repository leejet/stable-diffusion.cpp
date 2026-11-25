#include "core/device_health.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <utility>

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "stable-diffusion.h"

namespace sd {

    // Limits on what an outside watcher can make the registry keep: signals are data from another program.
    static constexpr size_t max_signal_sources = 8;
    static constexpr size_t max_source_length  = 32;
    static constexpr size_t max_reason_length  = 160;
    static constexpr std::chrono::seconds max_signal_duration{24 * 60 * 60};
    static constexpr std::chrono::seconds max_settings_wait{24 * 60 * 60};

    static std::string lowercase(std::string text) {
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return text;
    }

    static bool same_name(const std::string& a, const std::string& b) {
        return a.size() == b.size() && lowercase(a) == lowercase(b);
    }

    // Cuts text to at most limit bytes without splitting a UTF-8 sequence.
    static std::string clipped(const std::string& text, size_t limit) {
        if (text.size() <= limit) {
            return text;
        }
        size_t end = limit;
        while (end > 0 && (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) {
            end--;
        }
        return text.substr(0, end);
    }

    static std::vector<DeviceHealth::Component> machine_components() {
        sd_backend_resolve_name("");
        std::vector<DeviceHealth::Component> components;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            DeviceHealth::Component component;
            component.name = ggml_backend_dev_name(dev);
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(dev, &props);
            if (props.device_id != nullptr) {
                component.hardware_id = props.device_id;
            }
            components.push_back(std::move(component));
        }
        components.push_back({"RAM", ""});
        components.push_back({"disk", ""});
        return components;
    }

    DeviceHealth::DeviceHealth()
        : lister_(machine_components), clock_([] { return Clock::now(); }) {}

    void DeviceHealth::set_component_lister(std::function<std::vector<Component>()> lister) {
        std::lock_guard<std::mutex> lock(mutex_);
        lister_ = lister ? std::move(lister) : machine_components;
        listed_ = false;
    }

    void DeviceHealth::set_clock(std::function<Clock::time_point()> now) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (now) {
            clock_ = std::move(now);
        } else {
            clock_ = [] { return Clock::now(); };
        }
    }

    bool DeviceHealth::set_settings(const Settings& settings) {
        // Each step is checked alone first, so the products below cannot overflow.
        const bool valid = settings.strain_levels >= 1 && settings.strain_levels <= 10 &&
                           settings.step_percent >= 1 && settings.step_percent < 100 &&
                           settings.step_percent * settings.strain_levels < 100 &&
                           settings.cpu_thread_step_percent >= 0 && settings.cpu_thread_step_percent < 100 &&
                           settings.cpu_thread_step_percent * settings.strain_levels < 100 &&
                           settings.quiet_period >= std::chrono::seconds(1) && settings.quiet_period <= max_settings_wait &&
                           settings.event_spacing >= std::chrono::seconds(0) && settings.event_spacing <= max_settings_wait;
        if (!valid) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        // A new revision makes the next generation place its models again.
        if (settings.enabled == settings_.enabled && settings.strain_levels == settings_.strain_levels &&
            settings.step_percent == settings_.step_percent && settings.cpu_thread_step_percent == settings_.cpu_thread_step_percent &&
            settings.quiet_period == settings_.quiet_period && settings.event_spacing == settings_.event_spacing) {
            return true;
        }
        const int old_top = top_locked();
        settings_         = settings;
        if (!settings_.enabled) {
            entries_.clear();
            listed_ = false;
        }
        // Every component keeps its state: an excluded one stays excluded, one under strain stays under strain, at
        // most on the new last level.
        const int top = top_locked();
        auto moved    = [&](int level) { return level >= old_top ? top : std::min(level, settings_.strain_levels); };
        for (auto& entry : entries_) {
            entry.level = moved(entry.level);
            for (auto& held : entry.signals) {
                held.level = moved(held.level);
            }
        }
        revision_++;
        return true;
    }

    DeviceHealth::Settings DeviceHealth::settings() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return settings_;
    }

    void DeviceHealth::reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
        listed_ = false;
        revision_++;
    }

    DeviceHealth::Clock::time_point DeviceHealth::now_locked() const {
        return clock_();
    }

    int DeviceHealth::top_locked() const {
        return settings_.strain_levels + 1;
    }

    int DeviceHealth::effective(const Held& held) const {
        return held.critical ? top_locked() : held.level;
    }

    void DeviceHealth::list_locked(bool refresh) {
        if (listed_ && !refresh) {
            return;
        }
        const auto now = now_locked();
        for (const auto& component : lister_()) {
            if (component.name.empty()) {
                continue;
            }
            if (Entry* entry = find_locked(component.name)) {
                entry->name        = component.name;
                entry->hardware_id = component.hardware_id;
                continue;
            }
            Entry entry;
            entry.name        = component.name;
            entry.hardware_id = component.hardware_id;
            entry.since       = now;
            entry.quiet_from  = now;
            entries_.push_back(std::move(entry));
        }
        listed_ = true;
    }

    DeviceHealth::Entry* DeviceHealth::find_locked(const std::string& name) {
        for (auto& entry : entries_) {
            if (same_name(entry.name, name)) {
                return &entry;
            }
        }
        return nullptr;
    }

    DeviceHealth::Entry& DeviceHealth::entry_locked(const std::string& name) {
        list_locked(false);
        if (Entry* entry = find_locked(name)) {
            return *entry;
        }
        // An event of the engine names a device it is using, even one the lister did not know (an RPC server
        // added later).
        Entry entry;
        entry.name       = name;
        entry.since      = now_locked();
        entry.quiet_from = entry.since;
        entries_.push_back(std::move(entry));
        return entries_.back();
    }

    void DeviceHealth::set_level_locked(Entry& entry, int level, const std::string& reason, Clock::time_point when) {
        const int previous = entry.level;
        entry.reason       = reason;
        if (level == previous) {
            return;
        }
        entry.level = level;
        entry.since = when;
        revision_++;
        const int top = top_locked();
        if (level == top) {
            LOG_WARN("device health: %s excluded (%s)", entry.name.c_str(), reason.c_str());
        } else if (level > previous) {
            LOG_WARN("device health: %s under strain, level %d of %d (%s); it offers %d%% of its capacity",
                     entry.name.c_str(), level, settings_.strain_levels, reason.c_str(), share_percent_locked(level));
        } else if (level == 0) {
            LOG_INFO("device health: %s is healthy again", entry.name.c_str());
        } else {
            LOG_INFO("device health: %s back to level %d of %d; it offers %d%% of its capacity",
                     entry.name.c_str(), level, settings_.strain_levels, share_percent_locked(level));
        }
    }

    void DeviceHealth::refresh_locked(Entry& entry, Clock::time_point now) {
        while (entry.level > 0) {
            // The quiet period runs from the last event, or from the end of the latest signal at this level or
            // above when that comes later; a signal that still holds puts the start in the future.
            Clock::time_point start = entry.quiet_from;
            for (const auto& held : entry.signals) {
                if (effective(held) >= entry.level) {
                    start = std::max(start, held.expires);
                }
            }
            if (now - start < settings_.quiet_period) {
                break;
            }
            const auto when  = start + settings_.quiet_period;
            entry.quiet_from = when;
            set_level_locked(entry, entry.level - 1, entry.level - 1 == 0 ? std::string() : entry.reason, when);
        }
        // A signal that ended before the quiet period started can no longer hold anything.
        entry.signals.erase(std::remove_if(entry.signals.begin(), entry.signals.end(), [&](const Held& held) {
                                return held.expires <= now && (entry.level == 0 || held.expires <= entry.quiet_from);
                            }),
                            entry.signals.end());
    }

    void DeviceHealth::refresh_all_locked(Clock::time_point now) {
        for (auto& entry : entries_) {
            refresh_locked(entry, now);
        }
    }

    int DeviceHealth::share_percent_locked(int level) const {
        if (level <= 0) {
            return 100;
        }
        if (level >= top_locked()) {
            return 0;
        }
        return 100 - settings_.step_percent * level;
    }

    std::chrono::seconds DeviceHealth::next_step_in_locked(const Entry& entry, Clock::time_point now) const {
        if (entry.level == 0) {
            return std::chrono::seconds(0);
        }
        Clock::time_point start = entry.quiet_from;
        for (const auto& held : entry.signals) {
            if (effective(held) >= entry.level) {
                if (held.expires > now) {
                    return std::chrono::seconds(0);
                }
                start = std::max(start, held.expires);
            }
        }
        const auto remaining = start + settings_.quiet_period - now;
        return std::chrono::ceil<std::chrono::seconds>(std::max(remaining, Clock::duration::zero()));
    }

    void DeviceHealth::note_strain(const std::string& component, const std::string& reason) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!settings_.enabled || component.empty()) {
            return;
        }
        const auto now = now_locked();
        Entry& entry   = entry_locked(component);
        refresh_locked(entry, now);
        entry.quiet_from = std::max(entry.quiet_from, now);
        if (entry.counted_event && now - entry.last_event < settings_.event_spacing) {
            return;
        }
        entry.counted_event = true;
        entry.last_event    = now;
        set_level_locked(entry, std::min(entry.level + 1, top_locked()), clipped(reason, max_reason_length), now);
    }

    bool DeviceHealth::report(const std::string& component, const Signal& signal) {
        std::lock_guard<std::mutex> lock(mutex_);
        const int top = top_locked();
        if (!settings_.enabled || signal.source.empty() || signal.level < 0 || signal.level > top ||
            signal.duration < std::chrono::seconds(1)) {
            return false;
        }
        list_locked(false);
        Entry* named = find_locked(component);
        if (named == nullptr) {
            list_locked(true);
            named = find_locked(component);
        }
        if (named == nullptr) {
            return false;
        }
        std::vector<Entry*> targets;
        for (auto& entry : entries_) {
            if (&entry == named || (!named->hardware_id.empty() && entry.hardware_id == named->hardware_id)) {
                targets.push_back(&entry);
            }
        }
        const std::string source = clipped(signal.source, max_source_length);
        const bool withdraw      = signal.level == 0 && !signal.critical;
        auto from_source         = [&](const Held& held) { return held.source == source; };
        for (Entry* entry : targets) {
            const bool known = std::any_of(entry->signals.begin(), entry->signals.end(), from_source);
            if (!withdraw && !known && entry->signals.size() >= max_signal_sources) {
                return false;
            }
        }

        const auto now      = now_locked();
        const auto duration = std::min(signal.duration, max_signal_duration);
        const std::string reason =
            source + ": " + clipped(signal.reason.empty() ? std::string(signal.critical ? "critical" : "pressure") : signal.reason, max_reason_length);
        for (Entry* entry : targets) {
            refresh_locked(*entry, now);
            auto old             = std::find_if(entry->signals.begin(), entry->signals.end(), from_source);
            const bool old_holds = old != entry->signals.end() && old->expires > now && effective(*old) >= entry->level;
            if (withdraw) {
                if (old != entry->signals.end()) {
                    entry->signals.erase(old);
                }
                if (old_holds) {
                    entry->quiet_from = std::max(entry->quiet_from, now);
                }
                continue;
            }
            Held held;
            held.source   = source;
            held.level    = signal.level;
            held.critical = signal.critical;
            held.reason   = reason;
            held.expires  = now + duration;
            if (old != entry->signals.end()) {
                *old = held;
            } else {
                entry->signals.push_back(held);
            }
            const int level = effective(held);
            if (level >= entry->level || old_holds) {
                entry->quiet_from = std::max(entry->quiet_from, now);
            }
            if (level >= entry->level) {
                set_level_locked(*entry, level, reason, now);
            }
        }
        return true;
    }

    void DeviceHealth::resume() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!settings_.enabled) {
            return;
        }
        const auto now = now_locked();
        const int top  = top_locked();
        for (auto& entry : entries_) {
            refresh_locked(entry, now);
            for (auto it = entry.signals.begin(); it != entry.signals.end();) {
                if (it->critical) {
                    it = entry.signals.erase(it);
                    continue;
                }
                it->level = std::min(it->level, settings_.strain_levels);
                ++it;
            }
            if (entry.level == top) {
                entry.quiet_from = std::max(entry.quiet_from, now);
                set_level_locked(entry, settings_.strain_levels, "resumed by an operator", now);
            }
        }
    }

    int DeviceHealth::level(const std::string& component) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!settings_.enabled) {
            return 0;
        }
        list_locked(false);
        Entry* entry = find_locked(component);
        if (entry == nullptr) {
            return 0;
        }
        refresh_locked(*entry, now_locked());
        return entry->level;
    }

    bool DeviceHealth::excluded(const std::string& component) {
        const int current = level(component);
        std::lock_guard<std::mutex> lock(mutex_);
        return settings_.enabled && current >= top_locked();
    }

    bool DeviceHealth::excluded(ggml_backend_t backend) {
        return backend != nullptr && excluded(device_health_name(backend));
    }

    double DeviceHealth::share(const std::string& component) {
        const int current = level(component);
        std::lock_guard<std::mutex> lock(mutex_);
        return share_percent_locked(current) / 100.0;
    }

    double DeviceHealth::ram_share() {
        const int current = level("RAM");
        std::lock_guard<std::mutex> lock(mutex_);
        return share_percent_locked(std::min(current, settings_.strain_levels)) / 100.0;
    }

    int DeviceHealth::cpu_threads(int requested) {
        if (requested <= 1) {
            return requested;
        }
        const int current = level("CPU");
        std::lock_guard<std::mutex> lock(mutex_);
        const int steps = std::min(current, settings_.strain_levels);
        if (steps <= 0) {
            return requested;
        }
        const int percent = 100 - settings_.cpu_thread_step_percent * steps;
        return std::max(1, (requested * percent + 50) / 100);
    }

    bool DeviceHealth::any_critical() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!settings_.enabled) {
            return false;
        }
        const auto now = now_locked();
        refresh_all_locked(now);
        for (const auto& entry : entries_) {
            for (const auto& held : entry.signals) {
                if (held.critical && held.expires > now) {
                    return true;
                }
            }
        }
        return false;
    }

    uint64_t DeviceHealth::revision() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (settings_.enabled) {
            refresh_all_locked(now_locked());
        }
        return revision_;
    }

    std::vector<DeviceHealth::State> DeviceHealth::snapshot() {
        std::lock_guard<std::mutex> lock(mutex_);
        list_locked(false);
        const auto now = now_locked();
        const int top  = top_locked();
        refresh_all_locked(now);
        std::vector<State> states;
        states.reserve(entries_.size());
        for (const auto& entry : entries_) {
            State state;
            state.name          = entry.name;
            state.level         = entry.level;
            state.share_percent = share_percent_locked(entry.level);
            state.excluded      = entry.level >= top;
            state.critical      = std::any_of(entry.signals.begin(), entry.signals.end(), [&](const Held& held) {
                return held.critical && held.expires > now;
            });
            state.reason        = entry.reason;
            state.since         = entry.since;
            state.next_step_in  = next_step_in_locked(entry, now);
            for (const auto& held : entry.signals) {
                if (entry.level > 0 && effective(held) >= entry.level && held.expires > now) {
                    state.held_for = std::max(state.held_for, std::chrono::ceil<std::chrono::seconds>(held.expires - now));
                }
            }
            states.push_back(std::move(state));
        }
        return states;
    }

    DeviceHealth& device_health() {
        static DeviceHealth health;
        return health;
    }

    std::string device_health_name(ggml_backend_t backend) {
        if (backend == nullptr) {
            return "";
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        const char* name       = dev != nullptr ? ggml_backend_dev_name(dev) : ggml_backend_name(backend);
        return name != nullptr ? name : "";
    }

    size_t health_scaled_bytes(ggml_backend_t backend, size_t bytes) {
        if (backend == nullptr) {
            return bytes;
        }
        const double share = device_health().share(device_health_name(backend));
        return share >= 1.0 ? bytes : static_cast<size_t>(static_cast<double>(bytes) * share);
    }

    void note_compute_status(ggml_backend_t backend, ggml_status status) {
        if (backend != nullptr && status == GGML_STATUS_FAILED) {
            device_health().note_strain(device_health_name(backend), "compute failed");
        }
    }

    void note_backend_start_failure(ggml_backend_dev_t device) {
        if (device != nullptr) {
            device_health().note_strain(ggml_backend_dev_name(device), "the backend did not start");
        }
    }

    void note_memory_guard_unreachable(ggml_backend_t backend, int threshold_percent) {
        if (backend == nullptr) {
            return;
        }
        const std::string component = sd_backend_is_cpu(backend) ? std::string("RAM") : device_health_name(backend);
        device_health().note_strain(component, "the memory guard cannot get usage under " + std::to_string(threshold_percent) + "%");
    }

    void note_disk_read_failure(const std::string& path) {
        // The file name only: the reason is shown by the server's health endpoint.
        device_health().note_strain("disk", "a read of " + std::filesystem::u8path(path).filename().u8string() + " failed");
    }

}  // namespace sd

size_t sd_get_device_health(sd_device_health_t* health, size_t capacity) {
    auto& registry      = sd::device_health();
    const auto states   = registry.snapshot();
    const auto settings = registry.settings();
    for (size_t i = 0; health != nullptr && i < states.size() && i < capacity; ++i) {
        const auto& state       = states[i];
        sd_device_health_t& out = health[i];
        out                     = sd_device_health_t{};
        snprintf(out.name, sizeof(out.name), "%s", state.name.c_str());
        out.level         = state.level;
        out.strain_levels = settings.strain_levels;
        out.share_percent = state.share_percent;
        out.excluded      = state.excluded;
        out.critical      = state.critical;
        snprintf(out.reason, sizeof(out.reason), "%s", state.reason.c_str());
        out.held_ms      = std::chrono::duration_cast<std::chrono::milliseconds>(state.held_for).count();
        out.next_step_ms = std::chrono::duration_cast<std::chrono::milliseconds>(state.next_step_in).count();
    }
    return states.size();
}

bool sd_report_device_pressure(const char* component,
                               const char* source,
                               int level,
                               bool critical,
                               const char* reason,
                               int duration_seconds) {
    if (component == nullptr || source == nullptr) {
        return false;
    }
    sd::DeviceHealth::Signal signal;
    signal.source   = source;
    signal.level    = level;
    signal.critical = critical;
    signal.reason   = reason != nullptr ? reason : "";
    signal.duration = std::chrono::seconds(duration_seconds);
    return sd::device_health().report(component, signal);
}

void sd_resume_device_health(void) {
    sd::device_health().resume();
}

void sd_get_device_health_settings(sd_device_health_settings_t* settings) {
    if (settings == nullptr) {
        return;
    }
    const auto current                = sd::device_health().settings();
    settings->strain_levels           = current.strain_levels;
    settings->step_percent            = current.step_percent;
    settings->cpu_thread_step_percent = current.cpu_thread_step_percent;
    settings->quiet_period_seconds    = static_cast<int>(current.quiet_period.count());
    settings->event_spacing_seconds   = static_cast<int>(current.event_spacing.count());
}

bool sd_set_device_health_settings(const sd_device_health_settings_t* settings) {
    if (settings == nullptr) {
        return false;
    }
    auto next                    = sd::device_health().settings();
    next.strain_levels           = settings->strain_levels;
    next.step_percent            = settings->step_percent;
    next.cpu_thread_step_percent = settings->cpu_thread_step_percent;
    next.quiet_period            = std::chrono::seconds(settings->quiet_period_seconds);
    next.event_spacing           = std::chrono::seconds(settings->event_spacing_seconds);
    return sd::device_health().set_settings(next);
}
