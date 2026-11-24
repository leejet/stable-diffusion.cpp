#include "dgl.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>

#include "common/common.h"
#include "stable-diffusion.h"

ArgOptions DGLModeParams::get_options() {
    ArgOptions options;
    options.string_options = {
        {"", "--script", "path to the .dgl program", 0, &script_path},
        {"", "--write-builtins", "write the DGL built-in catalog to a file", 0, &write_builtins_path},
        {"", "--check-builtins", "verify a file against the DGL built-in catalog", 0, &check_builtins_path},
        {"", "--max-ram", "limit DGL context weight residency in GiB", 0, &max_ram},
    };
    options.bool_options = {
        {"", "--check", "check the DGL script without loading models", true, &check_only},
        {"", "--list-builtins", "print the DGL built-in catalog", true, &list_builtins},
    };
    options.manual_options = {
        {"", "--run-once", "on|off (default: on). Release DGL models after their last use when on",
         [this](int argc, const char** argv, int index) {
             const std::string_view value = index + 1 < argc ? argv[index + 1] : "";
             if (value != "on" && value != "off") {
                 LOG_ERROR("--run-once requires on or off");
                 return -1;
             }
             run_once = value == "on";
             return 1;
         }},
        {"", "--tensor", "save a named result tensor as .npy (NAME FILE, repeatable)",
         [this](int argc, const char** argv, int index) {
             if (index + 2 >= argc)
                 return -1;
             tensor_paths.emplace_back(argv_to_utf8(index + 1, argv), argv_to_utf8(index + 2, argv));
             return 2;
         }},
    };
    return options;
}

bool DGLModeParams::has_options() const {
    return !script_path.empty() || !write_builtins_path.empty() || !check_builtins_path.empty() ||
           !tensor_paths.empty() || !max_ram.empty() || check_only || run_once.has_value() || list_builtins;
}

bool DGLModeParams::validate() const {
    const int actions = int(!script_path.empty()) + int(!write_builtins_path.empty()) + int(!check_builtins_path.empty()) + int(list_builtins);
    if (actions != 1) {
        LOG_ERROR("DGL mode requires exactly one of --script, --list-builtins, --write-builtins and --check-builtins");
        return false;
    }
    if (script_path.empty() && (check_only || run_once.has_value() || !tensor_paths.empty() || !max_ram.empty())) {
        LOG_ERROR("--check, --run-once, --tensor and --max-ram require --script");
        return false;
    }
    return true;
}

static void check_dgl(bool success) {
    if (!success)
        throw std::runtime_error(sd_dgl_error());
}

int DGLModeParams::run(const SDContextParams& options, bool embed_metadata) const {
    try {
        if (script_path.empty()) {
            const auto* catalog = sd_dgl_builtin_catalog();
            check_dgl(catalog != nullptr);
            if (list_builtins) {
                std::cout << catalog;
            } else if (!write_builtins_path.empty()) {
                std::ofstream file(std::filesystem::u8path(write_builtins_path), std::ios::binary);
                file << catalog;
                file.close();
                if (!file)
                    throw std::runtime_error("cannot write the DGL built-in catalog");
            } else {
                std::ifstream file(std::filesystem::u8path(check_builtins_path));
                const std::string existing{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
                if (!file || existing != catalog)
                    throw std::runtime_error("DGL built-in catalog is out of date. Run the dgl-update-builtins CMake target.");
                std::cout << "DGL built-in catalog is current.\n";
            }
            return 0;
        }

        const auto path = std::filesystem::u8path(script_path);
        if (path.extension() != ".dgl")
            throw std::runtime_error("expected a .dgl file");
        std::ifstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("cannot open program file");
        const std::string source{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        if (check_only) {
            check_dgl(sd_dgl_check(source.c_str(), false) == SD_STATUS_OK);
            std::cout << "DGL syntax and types are valid. Models were not loaded.\n";
            return 0;
        }
        if (!options.register_rpc_devices())
            return 1;
        auto* raw_context = run_once.value_or(true) ? sd_dgl_ctx_create_once() : sd_dgl_ctx_create();
        check_dgl(raw_context != nullptr);
        std::unique_ptr<sd_dgl_ctx_t, decltype(&sd_dgl_ctx_free)> context(raw_context, sd_dgl_ctx_free);
        check_dgl(options.configure_dgl(context.get()));
        check_dgl(sd_dgl_ctx_set(context.get(), "embed_metadata", embed_metadata ? "on" : "off"));
        if (!max_ram.empty())
            check_dgl(sd_dgl_ctx_set(context.get(), "max_ram", max_ram.c_str()));
        sd_dgl_result_t* raw_result = nullptr;
        check_dgl(sd_dgl_run(context.get(), source.c_str(), nullptr, 0, &raw_result) == SD_STATUS_OK);
        std::unique_ptr<sd_dgl_result_t, decltype(&sd_dgl_result_free)> result(raw_result, sd_dgl_result_free);
        for (const auto& [name, output_path] : tensor_paths) {
            auto tensor = sd_dgl_result_tensor(result.get(), name.c_str());
            if (!tensor)
                throw std::runtime_error("unknown output tensor '" + name + "'");
            check_dgl(sd_tensor_save(tensor, output_path.c_str()));
        }
        const std::string type = sd_dgl_result_type(result.get());
        if (type == "string") {
            const auto* tensor = sd_dgl_result_tensor(result.get(), "result");
            const auto* data   = static_cast<const char*>(sd_tensor_data(tensor));
            const auto size    = sd_tensor_byte_size(tensor);
            if (size) {
                std::cout.write(data, size);
                if (data[size - 1] != '\n')
                    std::cout << '\n';
            }
            return 0;
        }
        const size_t count = sd_dgl_result_count(result.get());
        LOG_VERBOSE("Result: %s, tensors: %zu", type.c_str(), count);
        for (size_t i = 0; i < count; ++i) {
            LOG_VERBOSE("  %s", sd_dgl_result_name(result.get(), i));
        }
        return 0;
    } catch (const std::exception& error) {
        LOG_ERROR("%s", error.what());
        return 1;
    }
}
