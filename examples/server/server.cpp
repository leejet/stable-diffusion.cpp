#include "ggml/ggml.h"
#include "stable-diffusion.h"
#include "util.h"

#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif

#include "httplib.h"
#include "json.hpp"

#ifndef _WIN32
#include <unistd.h>
#endif

// internal website files
#include "index.html.hpp"
#include "main.js.hpp"
#include "preact.js.hpp"
#include "styles.css.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#include <cstddef>
#include <thread>
#include <mutex>
#include <chrono>

using json = nlohmann::json;

static std::string sd_basename(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    pos = path.find_last_of('\\');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

struct server_params
{
    std::string hostname = "127.0.0.1";
    std::string public_path = "public";
    int32_t port = 7860;
    int32_t read_timeout = 600;
    int32_t write_timeout = 600;
};

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string base64_encode(const uint8_t* buf, unsigned int bufLen) {
  std::string base64;
  int i = 0;
  int j = 0;
  uint8_t char_array_3[3];
  uint8_t char_array_4[4];

  while (bufLen--) {
    char_array_3[i++] = *(buf++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for(i = 0; (i <4) ; i++)
        base64 += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i)
  {
    for(j = i; j < 3; j++)
      char_array_3[j] = '\0';

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
    char_array_4[3] = char_array_3[2] & 0x3f;

    for (j = 0; (j < i + 1); j++)
      base64 += base64_chars[char_array_4[j]];

    while((i++ < 3))
      base64 += '=';
  }
  return base64;
}

void wait_(int ms) {
#ifdef _WIN32
Sleep(ms);
#else
sleep(ms);
#endif
}

struct sd_params {
    int n_threads = -1;

    std::string model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    ggml_type wtype = GGML_TYPE_COUNT;
    std::string lora_model_dir;
    std::string output_path = "output.png";
    std::string input_path;

    std::string prompt;
    std::string negative_prompt;
    float cfg_scale = 7.0f;
    int clip_skip   = -1;  // <= 0 represents unspecified
    int width       = 512;
    int height      = 512;
    int batch_count = 1;
    bool stream     = false;

    sample_method_t sample_method = EULER_A;
    schedule_t schedule          = DEFAULT;
    int sample_steps           = 20;
    float strength             = 0.75f;
    rng_type_t rng_type        = CUDA_RNG;
    int64_t seed               = 42;
    bool verbose               = false;
    bool vae_tiling            = false;
};

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
};

// Names of the sampler method, same order as enum sample_method in stable-diffusion.h
const char* sample_method_str[] = {
    "euler_a",
    "euler",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "lcm",
};

template <typename T>
static T json_value(const json &body, const std::string &key, const T &default_value)
{
    // Fallback null to default value
    return body.contains(key) && !body.at(key).is_null()
        ? body.value(key, default_value)
        : default_value;
}

static void server_print_usage(const char *argv0, const sd_params &params,
                               const server_params &sparams)
{
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help                show this help message and exit\n");
    printf("  -v, --verbose             verbose output (default: %s)\n", params.verbose ? "enabled" : "disabled");
    printf("  -t N, --threads N         number of threads to use during computation (default: %d)\n", params.n_threads);
    printf("  -m FNAME, --model FNAME\n");
    printf("                        model path (default: %s)\n", params.model_path.c_str());
    printf("  --type [TYPE]                      weight type (f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0)\n");
    printf("                                     If not specified, the default is the type of the weight file.\n");
    printf("  --lora-model-dir [DIR]             lora model directory\n");
    printf("  --host                ip address to listen (default  (default: %s)\n", sparams.hostname.c_str());
    printf("  --port PORT           port to listen (default  (default: %d)\n", sparams.port);
    printf("  --path PUBLIC_PATH    path from which to serve static files (default %s)\n", sparams.public_path.c_str());
    printf("  -to N, --timeout N    server read/write timeout in seconds (default: %d)\n", sparams.read_timeout);
    printf("  --clip-skip N                      ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)\n");
    printf("  --vae [VAE]                        path to vae\n");
    printf("  --taesd [TAESD_PATH]               path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)\n");
    printf("  --upscale-model [ESRGAN_PATH]      path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now.\n");
    printf("\n");
}

static void server_params_parse(int argc, char **argv, server_params &sparams,
                                sd_params &params)
{
    sd_params default_params;
    server_params default_sparams;
    std::string arg;
    bool invalid_param = false;

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "--port") {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.port = std::stoi(argv[i]);
        } else if (arg == "--host") {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.hostname = argv[i];
        } else if (arg == "--path") {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.public_path = argv[i];
        } else if (arg == "--timeout" || arg == "-to") {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.read_timeout = std::stoi(argv[i]);
            sparams.write_timeout = std::stoi(argv[i]);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.model_path = argv[i];
        } else if (arg == "--type") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::string type = argv[i];
            if (type == "f32") {
                params.wtype = GGML_TYPE_F32;
            } else if (type == "f16") {
                params.wtype = GGML_TYPE_F16;
            } else if (type == "q4_0") {
                params.wtype = GGML_TYPE_Q4_0;
            } else if (type == "q4_1") {
                params.wtype = GGML_TYPE_Q4_1;
            } else if (type == "q5_0") {
                params.wtype = GGML_TYPE_Q5_0;
            } else if (type == "q5_1") {
                params.wtype = GGML_TYPE_Q5_1;
            } else if (type == "q8_0") {
                params.wtype = GGML_TYPE_Q8_0;
            } else {
                fprintf(stderr, "error: invalid weight format %s, must be one of [f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0]\n",
                        type.c_str());
                exit(1);
            }
        } else if (arg == "--clip-skip") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.clip_skip = std::stoi(argv[i]);
        } else if (arg == "--lora-model-dir") {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.lora_model_dir = argv[i];
        } else if (arg == "-h" || arg == "--help") {
            server_print_usage(argv[0], default_params, default_sparams);
            exit(0);
        } else if (arg == "--vae") {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.vae_path = argv[i];
        } else if (arg == "--taesd") {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.taesd_path = argv[i];
        } else if (arg == "--threads" || arg == "-t") {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        }  else if (arg == "--upscale-model") {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.esrgan_path = argv[i];
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            server_print_usage(argv[0], default_params, default_sparams);
            exit(1);
        }
    }

    if (params.n_threads <= 0) {
        params.n_threads = get_num_physical_cores();
    }

    if (invalid_param)
    {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        server_print_usage(argv[0], default_params, default_sparams);
        exit(1);
    }
}

static void parse_options_generation(const json &body, sd_params& params)
{
    sd_params default_params;
    params.prompt = json_value(body, "prompt", default_params.prompt);
    params.negative_prompt = json_value(body, "negative_prompt", default_params.negative_prompt);
    params.seed = json_value(body, "seed", default_params.seed);
    params.sample_steps = json_value(body, "steps", default_params.sample_steps);
    params.sample_method = json_value(body, "sampler", default_params.sample_method);
    params.batch_count = json_value(body, "batch_count", default_params.batch_count);
    params.width = json_value(body, "width", default_params.width);
    params.height = json_value(body, "height", default_params.height);
    params.cfg_scale = json_value(body, "cfg_scale", default_params.cfg_scale);
    params.stream = json_value(body, "stream", default_params.stream);
    params.vae_tiling = json_value(body, "vae_tiling", default_params.vae_tiling);
}

std::string get_image_params(sd_params params, int64_t seed) {
    std::string parameter_string = params.prompt + "\n";
    if (params.negative_prompt.size() != 0) {
        parameter_string += "Negative prompt: " + params.negative_prompt + "\n";
    }
    parameter_string += "Steps: " + std::to_string(params.sample_steps) + ", ";
    parameter_string += "CFG scale: " + std::to_string(params.cfg_scale) + ", ";
    parameter_string += "Seed: " + std::to_string(seed) + ", ";
    parameter_string += "Size: " + std::to_string(params.width) + "x" + std::to_string(params.height) + ", ";
    parameter_string += "Model: " + sd_basename(params.model_path) + ", ";
    parameter_string += "RNG: " + std::string(rng_type_to_str[params.rng_type]) + ", ";
    parameter_string += "Sampler: " + std::string(sample_method_str[params.sample_method]);
    if (params.schedule == KARRAS) {
        parameter_string += " karras";
    }
    parameter_string += ", ";
    parameter_string += "Version: stable-diffusion.cpp";
    return parameter_string;
}

struct shared_data {
    httplib::DataSink* sink;
    sd_ctx_t* sd;
};

void progressFunction(int step, int steps, float time, bool new_image, void* data) {
    if(steps == 0) {
        return;
    }
    shared_data* sh = static_cast<shared_data*>(data);
    if(new_image) {
        json pdata = {{"type", "new_image"}, { "index", step }, {"count", steps }};
        std::string data_str = "data: " +
            pdata.dump(-1, ' ', false, json::error_handler_t::replace) +
            "\n\n";
        if(!sh->sink->write(data_str.c_str(), data_str.size())) {
            sd_request_cancel(sh->sd);
        }
    } else {
        json pdata = {{"type", "status"}, { "progress_current", step }, {"progress_total", steps }};
        std::string data_str = "data: " +
            pdata.dump(-1, ' ', false, json::error_handler_t::replace) +
            "\n\n";
        if(!sh->sink->write(data_str.c_str(), data_str.size())) {
            sd_request_cancel(sh->sd);
        }
    }
}

void vaeStage(void* data) {
    json pdata = {{"type", "status"}, { "decoding", true }};
    std::string data_str = "data: " +
        pdata.dump(-1, ' ', false, json::error_handler_t::replace) +
        "\n\n";
    shared_data* sh = static_cast<shared_data*>(data);
    if(!sh->sink->write(data_str.c_str(), data_str.size())) {
        sd_request_cancel(sh->sd);
    }
}

struct sd_request {
    bool need_attend = false;
    bool images_ready = false;
    bool request_load_model = false;
    bool cancel = false;
    sd_image_t* images = NULL;
};


/* Enables Printing the log level tag in color using ANSI escape codes */
void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    sd_params* params = (sd_params*)data;
    int tag_color;
    const char* level_str;
    FILE* out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

    if (!log || (!params->verbose && level <= SD_LOG_DEBUG)) {
        return;
    }

    switch (level) {
        case SD_LOG_DEBUG:
            tag_color = 37;
            level_str = "DEBUG";
            break;
        case SD_LOG_INFO:
            tag_color = 34;
            level_str = "INFO";
            break;
        case SD_LOG_WARN:
            tag_color = 35;
            level_str = "WARNING";
            break;
        case SD_LOG_ERROR:
            tag_color = 31;
            level_str = "ERROR";
            break;
        default: /* Potential future-proofing */
            tag_color = 33;
            level_str = "?????";
            break;
    }

    fprintf(out_stream, "\033[%d;1m[%-5s]\033[0m ", tag_color, level_str);
    fputs(log, out_stream);
    fflush(out_stream);
}


int main(int argc, char **argv)
{
    // own arguments required by this example
    sd_params params;
    server_params sparams;
    sd_request sdreq;

    server_params_parse(argc, argv, sparams, params);

    sd_set_log_callback(sd_log_cb, (void*)&params);

    struct sd_ctx_t* sd = new_sd_ctx_direct(params.lora_model_dir.c_str(), true, false, params.n_threads, params.rng_type);

    httplib::Server svr;

    svr.set_default_headers({{"Server", "stable-diffusion.cpp"},
                             {"Access-Control-Allow-Origin", "*"},
                             {"Access-Control-Allow-Headers", "content-type"}});

    // this is only called if no index.html is found in the public --path
    svr.Get("/", [](const httplib::Request &, httplib::Response &res)
            {
                res.set_content(reinterpret_cast<const char*>(&index_html), index_html_len, "text/html");
                return false;
            });
    svr.Get("/preact.js", [](const httplib::Request &, httplib::Response &res)
        {
            res.set_content(reinterpret_cast<const char*>(&preact_js), preact_js_len, "application/javascript; charset=utf-8");
            return false;
        });

    svr.Get("/main.js", [](const httplib::Request &, httplib::Response &res)
        {
            res.set_content(reinterpret_cast<const char*>(&main_js), main_js_len, "application/javascript; charset=utf-8");
            return false;
        });

    svr.Get("/styles.css", [](const httplib::Request &, httplib::Response &res)
        {
            res.set_content(reinterpret_cast<const char*>(&styles_css), styles_css_len, "text/plain");
            return false;
        });

    svr.Get("/state", [&sd](const httplib::Request & /*req*/, httplib::Response &res)
            {
                res.set_header("Access-Control-Allow-Origin", "*");
                json data = {
                    { "model_loaded", sd_model_is_loaded(sd) }
                };
                res.set_content(data.dump(), "application/json");
            });

    svr.Post("/txt2img", [&sd, &params, &sdreq](const httplib::Request &req, httplib::Response &res)
            {
                parse_options_generation(json::parse(req.body), params);
                if(params.stream) {
                    const auto chunked_content_provider = [&sd, &params, &sdreq](size_t, httplib::DataSink & sink)
                    {
                        sdreq.images_ready = false;
                        if(!sd_model_is_loaded(sd)) {
                            sdreq.request_load_model = true;
                        } else {
                            json data = {{"type", "status"}, { "loaded", true }};
                                std::string data_str = "data: " +
                                    data.dump(-1, ' ', false, json::error_handler_t::replace) +
                                    "\n\n";
                                sink.write(data_str.c_str(), data_str.size());
                        }
                        shared_data* sh = new shared_data{&sink, sd};
                        // send progressions metrics
                        sdreq.need_attend = true;
                        bool need_model_notification = true;
                        while(true) {
                            if(need_model_notification && !sdreq.request_load_model) {
                                json data = {{"type", "status"}, { "loaded", true }};
                                std::string data_str = "data: " +
                                    data.dump(-1, ' ', false, json::error_handler_t::replace) +
                                    "\n\n";
                                if(!sink.write(data_str.c_str(), data_str.size())) {
                                    return false;
                                }
                                sd_set_progress_callback(progressFunction, sh);
                                sd_set_vae_callback(vaeStage, sh);
                                need_model_notification = false;
                            }
                            if(sdreq.images_ready) {
                                if(!sdreq.images) {
                                    return false;
                                }
                                for(int i = 0; i < params.batch_count; i ++) {
                                    int len;
                                    uint8_t* png = stbi_write_png_to_mem((const unsigned char *) sdreq.images[i].data, 0, params.width, params.height, 3, &len, get_image_params(params, params.seed + i).c_str());
                                    json data = {{ "type", "image" }, {"data", base64_encode(png, len) }, { "seed", sdreq.images[i].seed }, {"stop", i == params.batch_count - 1}};
                                    std::string data_str = "data: " +
                                        data.dump(-1, ' ', false, json::error_handler_t::replace) +
                                        "\n\n";
                                    if(!sink.write(data_str.c_str(), data_str.size())){
                                        return false;
                                    }
                                }
                                break;
                            }
                            wait_(2);
                        }
                        sink.done();
                        return true;
                    };
                    auto on_complete = [&sd, &sdreq] (bool) {
                        // cancel
                        sd_request_cancel(sd);
                    };
                    res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
                } else {
                    // while(true) {

                    // }
                    // json data = {{"images", base64_encode(png, len) }, { "seed", sdreq.images[i].seed }, {"stop", i == params.batch_count - 1}};
                    // res.set_content(data.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
                }
            });

    svr.Options(R"(/.*)", [](const httplib::Request &, httplib::Response &res)
                { return res.set_content("", "application/json"); });

    svr.set_exception_handler([](const httplib::Request &, httplib::Response &res, std::exception_ptr ep)
            {
                const char fmt[] = "500 Internal Server Error\n%s";
                char buf[BUFSIZ];
                try
                {
                    std::rethrow_exception(std::move(ep));
                }
                catch (std::exception &e)
                {
                    snprintf(buf, sizeof(buf), fmt, e.what());
                }
                catch (...)
                {
                    snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
                }
                res.set_content(buf, "text/plain");
                res.status = 500;
            });

    svr.set_error_handler([](const httplib::Request &, httplib::Response &res)
            {
                if (res.status == 400)
                {
                    res.set_content("Invalid request", "text/plain");
                }
                else if (res.status != 500)
                {
                    res.set_content("File Not Found", "text/plain");
                    res.status = 404;
                }
            });

    // set timeouts and change hostname and port
    svr.set_read_timeout (sparams.read_timeout);
    svr.set_write_timeout(sparams.write_timeout);

    if (!svr.bind_to_port(sparams.hostname, sparams.port))
    {
        fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n", sparams.hostname.c_str(), sparams.port);
        return 1;
    }

    // Set the base directory for serving static files
    svr.set_base_dir(sparams.public_path);

    // to make it ctrl+clickable:
    printf("\nstable-diffusion server listening at http://%s:%d\n", sparams.hostname.c_str(), sparams.port);

    std::thread t([&]() {
        if (!svr.listen_after_bind()) {
            return 1;
        }
        return 0;
    });
    while(true) {
        if(sdreq.request_load_model) {
            load_model(sd,
                        params.model_path.c_str(),
                        params.vae_path.c_str(),
                        params.taesd_path.c_str(),
                        "", "", (sd_type_t)params.wtype, params.schedule, false);
            sdreq.request_load_model = false;
        }
        if(sdreq.need_attend) {
            sdreq.images = txt2img(sd,
                            params.prompt.c_str(),
                            params.negative_prompt.c_str(), 0,
                            params.cfg_scale, params.width, params.height,
                            params.sample_method, params.sample_steps, params.seed,
                            params.batch_count, NULL, 0.0f, 0.0f, 0.0f, "", params.vae_tiling);
            sdreq.images_ready = true;
            sdreq.need_attend = false;
        }
        wait_(2);
    }
    return 0;
}
