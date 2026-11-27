#ifndef __EXAMPLES_CLI_DGL_H__
#define __EXAMPLES_CLI_DGL_H__

#include <optional>
#include <string>
#include <utility>
#include <vector>

struct ArgOptions;
struct SDContextParams;

struct DGLModeParams {
    std::string script_path;
    std::string write_builtins_path;
    std::string check_builtins_path;
    std::string max_ram;
    std::vector<std::pair<std::string, std::string>> tensor_paths;
    bool check_only = false;
    std::optional<bool> run_once;
    bool list_builtins = false;

    ArgOptions get_options();
    bool has_options() const;
    bool validate() const;
    int run(const SDContextParams& options, bool embed_metadata) const;
};

#endif  // __EXAMPLES_CLI_DGL_H__
