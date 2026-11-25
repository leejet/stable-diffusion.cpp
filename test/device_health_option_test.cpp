// Test of the --device-health option of sd-cli and sd-server (examples/common): the key=value list, the lists it
// refuses, and the values it writes into the device health of the process through the C API.
// Build (from the repository root, after building the static library into build/):
//   cc -c thirdparty/zip.c -o zip.o
//   c++ -std=c++17 -Iexamples -Iinclude -Ithirdparty -Isrc test/device_health_option_test.cpp \
//       examples/common/common.cpp examples/common/log.cpp examples/common/media_io.cpp \
//       examples/common/media_metadata.cpp examples/common/profile.cpp zip.o build/libstable-diffusion.a \
//       build/ggml/src/libggml.a build/ggml/src/libggml-cpu.a build/ggml/src/libggml-base.a \
//       -ldl -lm -fopenmp -pthread -o device_health_option_test
// Run: ./device_health_option_test ; exit code 0 = all checks passed.
#include <cstdio>
#include <string>
#include <vector>

#include "common/common.h"
#include "stable-diffusion.h"

static int failures = 0;

static void check(bool condition, const std::string& what) {
    std::printf("%s %s\n", condition ? "ok:  " : "FAIL:", what.c_str());
    failures += condition ? 0 : 1;
}

static sd_device_health_settings_t defaults() {
    sd_device_health_settings_t settings = {};
    settings.strain_levels               = 3;
    settings.step_percent                = 15;
    settings.cpu_thread_step_percent     = 25;
    settings.quiet_period_seconds        = 300;
    settings.event_spacing_seconds       = 10;
    return settings;
}

static bool same_values(const sd_device_health_settings_t& a, const sd_device_health_settings_t& b) {
    return a.strain_levels == b.strain_levels && a.step_percent == b.step_percent &&
           a.cpu_thread_step_percent == b.cpu_thread_step_percent && a.quiet_period_seconds == b.quiet_period_seconds &&
           a.event_spacing_seconds == b.event_spacing_seconds;
}

static void test_parse() {
    std::string error;
    sd_device_health_settings_t settings = defaults();
    check(parse_device_health_settings("levels=2,step=20", settings, error) && settings.strain_levels == 2 &&
              settings.step_percent == 20 && settings.cpu_thread_step_percent == 25 && settings.quiet_period_seconds == 300 &&
              settings.event_spacing_seconds == 10,
          "the keys given change their values, the others stay");
    settings = defaults();
    check(parse_device_health_settings("quiet=600", settings, error) && settings.quiet_period_seconds == 600 &&
              settings.strain_levels == 3 && settings.event_spacing_seconds == 10,
          "one key alone changes one value");
    settings = defaults();
    check(parse_device_health_settings(" cpu-step = 20 , spacing=0 ", settings, error) && settings.cpu_thread_step_percent == 20 &&
              settings.event_spacing_seconds == 0,
          "spaces around keys and values are allowed");
    settings = defaults();
    check(parse_device_health_settings("spacing=-1", settings, error) && settings.event_spacing_seconds == -1,
          "a negative number is read: the values are judged when they are written");

    struct Refused {
        const char* list;
        const char* what;
    };
    const Refused refused[] = {
        {"", "an empty list"},
        {"levels", "a key without a value"},
        {"levels=", "an empty value"},
        {"=2", "a value without a key"},
        {"levels=two", "a value that is not a number"},
        {"levels=2x", "a number followed by letters"},
        {"levels=2.5", "a number that is not whole"},
        {"levels=99999999999", "a number too big for an int"},
        {"levels=2,levels=3", "a key given twice"},
        {"speed=3", "an unknown key"},
        {"Levels=2", "a key in other letters"},
        {"levels=2,,step=20", "an empty item"},
        {"levels=2,", "a comma at the end"},
    };
    for (const auto& item : refused) {
        settings = defaults();
        error.clear();
        const bool parsed = parse_device_health_settings(item.list, settings, error);
        check(!parsed && !error.empty() && same_values(settings, defaults()),
              std::string("refused, with a reason and nothing changed: ") + item.what + " ('" + item.list + "')");
    }
    settings = defaults();
    parse_device_health_settings("speed=3", settings, error);
    check(error.find("levels") != std::string::npos && error.find("cpu-step") != std::string::npos &&
              error.find("spacing") != std::string::npos,
          "an unknown key is answered with the keys there are");
}

static void test_option_and_values() {
    sd_device_health_settings_t saved = {};
    sd_get_device_health_settings(&saved);
    check(same_values(saved, defaults()), "the process starts with the default values");

    SDContextParams params;
    const std::vector<std::string> words = {"test", "--device-health", "levels=2", "--device-health", "step=20"};
    std::vector<const char*> argv;
    for (const auto& word : words) {
        argv.push_back(word.c_str());
    }
    const bool parsed = parse_options(static_cast<int>(argv.size()), argv.data(), {params.get_options()});
    check(parsed && params.device_health == "levels=2,step=20", "the option is read, and given twice its lists are joined");
    check(params.apply_device_health_option(), "the values are written");
    sd_device_health_settings_t now = {};
    sd_get_device_health_settings(&now);
    sd_device_health_settings_t wanted = defaults();
    wanted.strain_levels               = 2;
    wanted.step_percent                = 20;
    check(same_values(now, wanted), "into the device health of the process, the others unchanged");

    SDContextParams wrong;
    wrong.device_health = "levels=8";
    check(!wrong.apply_device_health_option(), "values against the rules are refused: 8 levels of 20% leave nothing");
    wrong.device_health = "spacing=-1";
    check(!wrong.apply_device_health_option(), "and so is a negative spacing");
    wrong.device_health = "quiet=86401";
    check(!wrong.apply_device_health_option(), "and a quiet period longer than a day");
    wrong.device_health = "speed=3";
    check(!wrong.apply_device_health_option(), "and a list that cannot be read");
    sd_get_device_health_settings(&now);
    check(same_values(now, wanted), "and they change nothing");

    SDContextParams none;
    check(none.apply_device_health_option(), "without the option there is nothing to do");
    sd_get_device_health_settings(&now);
    check(same_values(now, wanted), "and nothing changes");

    check(params.to_string().find("device_health: \"levels=2,step=20\"") != std::string::npos,
          "the parameters printed at start name the option");
    sd_set_device_health_settings(&saved);
}

int main() {
    test_parse();
    test_option_and_values();
    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED", failures);
    return failures == 0 ? 0 : 1;
}
