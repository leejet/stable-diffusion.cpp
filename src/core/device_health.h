#ifndef __SD_CORE_DEVICE_HEALTH_H__
#define __SD_CORE_DEVICE_HEALTH_H__

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "ggml-backend.h"

namespace sd {

    // How the compute devices, the RAM and the disk of the process are doing, from what the engine sees and what
    // outside watchers report. A component under strain offers less of its capacity, one step per level, and the
    // work it gives up goes to the others; an excluded device offers nothing and the engine never computes on it.
    // A level rises at once, one step per event, and comes back one step after each quiet period, never below a
    // signal that still holds. There is no thread: the clock is read whenever the registry is asked.
    class DeviceHealth {
    public:
        using Clock = std::chrono::steady_clock;

        struct Settings {
            bool enabled                = true;
            int strain_levels           = 3;   // levels of strain before exclusion
            int step_percent            = 15;  // capacity a component gives up at each level
            int cpu_thread_step_percent = 25;  // compute threads the CPU gives up at each level
            std::chrono::seconds quiet_period{300};
            // Events of the engine on one component closer than this count once: a failure shows up in more
            // than one place.
            std::chrono::seconds event_spacing{10};
        };

        // A device as the backends list it, "RAM" or "disk". The listings of one card through two backends (CUDA0
        // and Vulkan0) share a non-empty hardware id.
        struct Component {
            std::string name;
            std::string hardware_id;
        };

        // What an outside watcher reports. Each source holds at most one signal per component; a new one replaces
        // it, and level 0 withdraws it. A signal reaches every listing of the same hardware.
        struct Signal {
            std::string source;
            int level     = 0;      // 1..strain_levels, or strain_levels + 1 to exclude
            bool critical = false;  // asks to stop all work; excludes the component while it holds
            std::string reason;
            std::chrono::seconds duration{60};
        };

        struct State {
            std::string name;
            int level         = 0;  // 0 healthy, 1..strain_levels under strain, strain_levels + 1 excluded
            int share_percent = 100;
            bool excluded     = false;
            bool critical     = false;
            std::string reason;
            Clock::time_point since;  // when the level last changed
            // Until the level comes down one step when nothing else happens; zero when it is healthy or held by
            // a signal that has not ended yet.
            std::chrono::seconds next_step_in{0};
            // How long the signals that hold this level still last; zero when none does.
            std::chrono::seconds held_for{0};
        };

        DeviceHealth();

        // Replaceable so that the rules can be tested without devices and without waiting. Defaults: the devices
        // of the ggml backends with RAM and disk, and the steady clock.
        void set_component_lister(std::function<std::vector<Component>()> lister);
        void set_clock(std::function<Clock::time_point()> now);
        // False, and nothing changes, when the values make no sense.
        bool set_settings(const Settings& settings);
        Settings settings() const;
        // Forgets every level and every signal.
        void reset();

        // An event of the engine on one component: a compute that failed, a backend that did not start, the
        // memory guard that cannot get under its threshold, a read of the disk that failed. One level up.
        void note_strain(const std::string& component, const std::string& reason);
        // False when the component is unknown or the signal is not valid; the registry is then unchanged.
        bool report(const std::string& component, const Signal& signal);
        // An operator says the machine was checked: the critical signals are withdrawn and the excluded components
        // come back to the last level of strain.
        void resume();

        int level(const std::string& component);
        bool excluded(const std::string& component);
        bool excluded(ggml_backend_t backend);
        // What the component offers of its capacity: 1 when healthy, 0 when excluded.
        double share(const std::string& component);
        // What the RAM offers: as share("RAM"), but never less than at the last level of strain, because nothing
        // runs without RAM.
        double ram_share();
        // The compute threads the CPU takes out of requested, at least 1.
        int cpu_threads(int requested);
        bool any_critical();
        // Changes whenever a level changes, so that a placement made before knows it is out of date.
        uint64_t revision();
        std::vector<State> snapshot();

    private:
        struct Held {
            std::string source;
            int level     = 0;
            bool critical = false;
            std::string reason;
            Clock::time_point expires;
        };

        struct Entry {
            std::string name;
            std::string hardware_id;
            int level = 0;
            std::string reason;
            Clock::time_point since;
            // The quiet period that brings the level down runs from here, unless a signal holds the level longer.
            Clock::time_point quiet_from;
            bool counted_event = false;
            Clock::time_point last_event;
            std::vector<Held> signals;
        };

        Clock::time_point now_locked() const;
        int top_locked() const;
        int effective(const Held& held) const;
        void list_locked(bool refresh);
        Entry* find_locked(const std::string& name);
        Entry& entry_locked(const std::string& name);
        void refresh_locked(Entry& entry, Clock::time_point now);
        void refresh_all_locked(Clock::time_point now);
        void set_level_locked(Entry& entry, int level, const std::string& reason, Clock::time_point now);
        int share_percent_locked(int level) const;
        std::chrono::seconds next_step_in_locked(const Entry& entry, Clock::time_point now) const;

        mutable std::mutex mutex_;
        Settings settings_;
        std::function<std::vector<Component>()> lister_;
        std::function<Clock::time_point()> clock_;
        std::vector<Entry> entries_;
        bool listed_       = false;
        uint64_t revision_ = 0;
    };

    // The registry of the process: the devices are shared by every context.
    DeviceHealth& device_health();

    // The name the registry uses for the device a backend runs on.
    std::string device_health_name(ggml_backend_t backend);
    // A budget of memory on the device of a backend, scaled by the share the device offers.
    size_t health_scaled_bytes(ggml_backend_t backend, size_t bytes);

    // The events of the engine, as the registry of the process counts them. A compute that failed counts; one that
    // ran out of memory or was stopped on request does not: the profile ladder and the memory guard deal with
    // memory, and a stop is not a fault.
    void note_compute_status(ggml_backend_t backend, ggml_status status);
    void note_backend_start_failure(ggml_backend_dev_t device);
    // The guard of the CPU watches the system RAM, so its pressure is the RAM's.
    void note_memory_guard_unreachable(ggml_backend_t backend, int threshold_percent);
    void note_disk_read_failure(const std::string& path);

}  // namespace sd

#endif  // __SD_CORE_DEVICE_HEALTH_H__
