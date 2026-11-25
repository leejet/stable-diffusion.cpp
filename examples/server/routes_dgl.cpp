#include "routes.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string_view>

#include "async_jobs.h"

namespace {
    namespace fs   = std::filesystem;
    using Request  = httplib::Request;
    using Response = httplib::Response;

    struct DglWorkspace {
        fs::path directory;
        std::mutex mutex;

        explicit DglWorkspace(const std::string& path) : directory(fs::u8path(path)) {}
    };

    void send_json(Response& res, int status, const json& body) {
        res.status = status;
        res.set_content(body.dump(), "application/json");
    }

    void send_error(Response& res, int status, const std::string& message) {
        res.status = status;
        res.set_content(json({{"error", message}}).dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    }

    httplib::Server::Handler with_dgl_errors(httplib::Server::Handler handler) {
        return [handler = std::move(handler)](const Request& req, Response& res) {
            res.set_header("Cache-Control", "no-store");
            if (req.method == "POST") {
                const auto content_type = req.get_header_value("Content-Type");
                if (content_type.substr(0, content_type.find(';')) != "application/json") {
                    send_error(res, 415, "DGL requests require application/json");
                    return;
                }
            }
            try {
                handler(req, res);
            } catch (const json::exception&) {
                send_error(res, 400, "Expected valid UTF-8 JSON or script text");
            } catch (const std::exception& error) {
                send_error(res, 500, error.what());
            }
        };
    }

    bool valid_script_name(const std::string& name) {
        return name.size() > std::string_view(".dgl").size() && name.find_first_of("/\\:*?\"<>|") == std::string::npos &&
               std::none_of(name.begin(), name.end(), [](unsigned char c) { return c < 32; }) &&
               fs::u8path(name).extension() == ".dgl";
    }

    bool valid_source(const std::string& source) {
        return source.find('\0') == std::string::npos;
    }

    std::string read_script(const fs::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        const auto length = file.tellg();
        if (length < 0)
            throw std::runtime_error("Could not open script");
        if (static_cast<uintmax_t>(length) > std::string{}.max_size())
            throw std::runtime_error("Script is too large to read");
        std::string source(static_cast<size_t>(length), '\0');
        file.seekg(0);
        if (!file.read(source.data(), length) || file.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("Script changed while reading");
        if (!valid_source(source))
            throw std::runtime_error("Script must contain UTF-8 text");
        return source;
    }

    void save_script(const fs::path& path, const std::string& source) {
        const auto prefix = ".dgl-save-" + std::to_string(unix_timestamp_now_ms()) + "-";
        fs::path staging;
        for (size_t i = 0; staging.empty(); ++i) {
            auto candidate = path.parent_path() / (prefix + std::to_string(i));
            if (fs::create_directory(candidate))
                staging = std::move(candidate);
        }
        const auto temporary = staging / "script";
        try {
            std::ofstream file(temporary, std::ios::binary);
            file.write(source.data(), static_cast<std::streamsize>(source.size()));
            file.close();
            if (!file)
                throw std::runtime_error("Could not write script");
            fs::rename(temporary, path);
        } catch (...) {
            std::error_code ignored;
            fs::remove(temporary, ignored);
            fs::remove(staging, ignored);
            throw;
        }
        std::error_code ignored;
        fs::remove(staging, ignored);
    }
}

void register_dgl_endpoints(httplib::Server& svr, ServerRuntime& runtime) {
    auto workspace = std::make_shared<DglWorkspace>(runtime.svr_params->dgl_dir);
    svr.Get("/sdcpp/v1/dgl/scripts", with_dgl_errors([workspace](const Request&, Response& res) {
                std::lock_guard<std::mutex> lock(workspace->mutex);
                std::vector<std::string> names;
                if (fs::exists(workspace->directory)) {
                    for (const auto& entry : fs::directory_iterator(workspace->directory)) {
                        auto name = entry.path().filename().u8string();
                        if (valid_script_name(name) && fs::is_regular_file(entry.symlink_status()))
                            names.push_back(std::move(name));
                    }
                }
                std::sort(names.begin(), names.end());
                send_json(res, 200, {{"directory", workspace->directory.generic_u8string()}, {"scripts", names}});
            }));

    svr.Get("/sdcpp/v1/dgl/script", with_dgl_errors([workspace](const Request& req, Response& res) {
                const auto name = req.get_param_value("name");
                if (!valid_script_name(name)) {
                    send_error(res, 400, "Expected a .dgl file name");
                    return;
                }
                std::lock_guard<std::mutex> lock(workspace->mutex);
                const auto path = workspace->directory / fs::u8path(name);
                if (!fs::is_regular_file(fs::symlink_status(path))) {
                    send_error(res, 404, "Script not found");
                    return;
                }
                send_json(res, 200, {{"source", read_script(path)}});
            }));

    svr.Post("/sdcpp/v1/dgl/script", with_dgl_errors([workspace](const Request& req, Response& res) {
                 const auto body   = json::parse(req.body);
                 const auto name   = body.at("name").get<std::string>();
                 const auto source = body.at("source").get<std::string>();
                 if (!valid_script_name(name) || !valid_source(source)) {
                     send_error(res, 400, "Expected a .dgl file name and UTF-8 source");
                     return;
                 }
                 const bool updating = body.contains("previous_source");
                 std::lock_guard<std::mutex> lock(workspace->mutex);
                 const auto path   = workspace->directory / fs::u8path(name);
                 const auto status = fs::symlink_status(path);
                 if (updating) {
                     if (!fs::is_regular_file(status) || read_script(path) != body.at("previous_source").get<std::string>()) {
                         send_error(res, 409, "The script changed on disk. Reopen it or save with a new name.");
                         return;
                     }
                 } else if (fs::exists(status)) {
                     send_error(res, 409, "A file with that name already exists");
                     return;
                 }
                 fs::create_directories(workspace->directory);
                 save_script(path, source);
                 send_json(res, updating ? 200 : 201, {{"name", name}});
             }));

    svr.Post("/sdcpp/v1/dgl/jobs", with_dgl_errors([&runtime](const Request& req, Response& res) {
                 auto source = json::parse(req.body).at("source").get<std::string>();
                 if (source.empty() || !valid_source(source)) {
                     send_error(res, 400, "Expected non-empty UTF-8 source");
                     return;
                 }
                 auto job        = std::make_shared<AsyncGenerationJob>();
                 job->kind       = AsyncJobKind::Dgl;
                 job->dgl_source = std::move(source);
                 auto& manager   = *runtime.async_job_manager;
                 {
                     std::lock_guard<std::mutex> lock(manager.mutex);
                     purge_expired_jobs(manager);
                     if (count_pending_jobs(manager) >= manager.max_pending_jobs) {
                         send_error(res, 429, "Job queue is full");
                         return;
                     }
                     job->id = make_async_job_id(manager);
                     manager.jobs.emplace(job->id, job);
                     manager.queue.push_back(job->id);
                     auto response        = make_async_job_json(manager, *job);
                     response["poll_url"] = "/sdcpp/v1/jobs/" + job->id;
                     send_json(res, 202, response);
                 }
                 manager.cv.notify_one();
             }));
}
