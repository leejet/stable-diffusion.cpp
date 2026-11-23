#include "routes.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "async_jobs.h"
#include "common/media_io.h"

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

    bool local_origin(const Request& req) {
        if (req.remote_addr != "127.0.0.1" && req.remote_addr != "::1" && req.remote_addr != "::ffff:127.0.0.1")
            return false;
        const auto host   = req.get_header_value("Host");
        const auto port   = ":" + std::to_string(req.local_port);
        const auto origin = req.get_header_value("Origin");
        const auto site   = req.get_header_value("Sec-Fetch-Site");
        if ((!site.empty() && site != "same-origin" && site != "none") ||
            (!origin.empty() && origin != "http://" + host && origin != "https://" + host))
            return false;
        for (const std::string address : {"localhost", "127.0.0.1", "[::1]"}) {
            if (host == address || host == address + port)
                return true;
        }
        return false;
    }

    httplib::Server::Handler local_dgl_route(httplib::Server::Handler handler) {
        return [handler = std::move(handler)](const Request& req, Response& res) {
            res.set_header("Cache-Control", "no-store");
            res.set_header("Cross-Origin-Resource-Policy", "same-origin");
            if (!local_origin(req)) {
                send_error(res, 403, "DGL is available from the local server's Web UI");
                return;
            }
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

    std::optional<size_t> parse_index(std::string_view value) {
        size_t index      = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), index);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
            return std::nullopt;
        return index;
    }

    std::shared_ptr<AsyncGenerationJob> find_dgl_job(AsyncJobManager& manager, const std::string& id, Response& res) {
        purge_expired_jobs(manager);
        auto it = manager.jobs.find(id);
        if (it != manager.jobs.end() && it->second->kind == AsyncJobKind::Dgl)
            return it->second;
        const bool expired = manager.expired_jobs.count(id) != 0;
        send_error(res, expired ? 410 : 404, expired ? "Job expired" : "Job not found");
        return {};
    }

    void send_output_preview(const Request& req, Response& res, const sd_dgl_result_t* result) {
        const auto output = parse_index(req.matches[2].str());
        if (!output || *output >= sd_dgl_result_count(result)) {
            send_error(res, 404, "Output not found");
            return;
        }
        const auto* tensor = sd_dgl_result_tensor(result, sd_dgl_result_name(result, *output));
        const auto frames  = dgl_image_frame_count(tensor);
        const auto frame   = req.has_param("frame") ? parse_index(req.get_param_value("frame")) : std::optional<size_t>(0);
        if (!frame || *frame >= static_cast<size_t>(frames)) {
            send_error(res, 404, "Frame not found");
            return;
        }
        const size_t stride = sd_tensor_byte_size(tensor) / static_cast<size_t>(frames);
        const auto* pixels  = static_cast<const uint8_t*>(sd_tensor_data(tensor)) + *frame * stride;
        const auto png      = encode_image_to_vector(EncodedImageFormat::PNG, pixels,
                                                     int(sd_tensor_dim(tensor, 1)), int(sd_tensor_dim(tensor, 2)),
                                                     int(sd_tensor_dim(tensor, 0)));
        if (png.empty()) {
            send_error(res, 500, "Could not encode preview");
            return;
        }
        res.set_content(reinterpret_cast<const char*>(png.data()), png.size(), "image/png");
    }
}

void register_dgl_endpoints(httplib::Server& svr, ServerRuntime& runtime) {
    auto workspace = std::make_shared<DglWorkspace>(runtime.svr_params->dgl_dir);
    svr.Get("/sdcpp/v1/dgl/scripts", local_dgl_route([workspace](const Request&, Response& res) {
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

    svr.Get("/sdcpp/v1/dgl/script", local_dgl_route([workspace](const Request& req, Response& res) {
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

    svr.Post("/sdcpp/v1/dgl/script", local_dgl_route([workspace](const Request& req, Response& res) {
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

    svr.Post("/sdcpp/v1/dgl/jobs", local_dgl_route([&runtime](const Request& req, Response& res) {
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
                     send_json(res, 202, make_async_job_json(manager, *job));
                 }
                 manager.cv.notify_one();
             }));

    svr.Get(R"(/sdcpp/v1/dgl/jobs/([A-Za-z0-9_-]+))", local_dgl_route([&runtime](const Request& req, Response& res) {
                auto& manager = *runtime.async_job_manager;
                std::lock_guard<std::mutex> lock(manager.mutex);
                if (auto job = find_dgl_job(manager, req.matches[1].str(), res))
                    send_json(res, 200, make_async_job_json(manager, *job));
            }));

    svr.Get(R"(/sdcpp/v1/dgl/jobs/([A-Za-z0-9_-]+)/outputs/([0-9]+))", local_dgl_route([&runtime](const Request& req, Response& res) {
                auto& manager = *runtime.async_job_manager;
                std::shared_ptr<sd_dgl_result_t> result;
                {
                    std::lock_guard<std::mutex> lock(manager.mutex);
                    const auto job = find_dgl_job(manager, req.matches[1].str(), res);
                    if (!job)
                        return;
                    result = job->dgl_result;
                }
                if (!result) {
                    send_error(res, 404, "Job has no result");
                    return;
                }
                send_output_preview(req, res, result.get());
            }));
}
