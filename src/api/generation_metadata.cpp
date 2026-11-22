#include "api/api.h"
#include "api/metadata.h"

#include <map>
#include <set>
#include <tuple>

#include "dgl/syntax/syntax.h"

namespace sd::api {
    using JSON       = nlohmann::json;
    using Expression = std::shared_ptr<dgl::Expr>;
    using Bindings   = std::map<std::string, Expression>;

    static Expression resolve_expression(Expression expr, const Bindings& bindings) {
        std::set<std::string> visited;
        while (expr && expr->form == dgl::Expr::Form::Reference && bindings.count(expr->name)) {
            if (!visited.insert(expr->name).second)
                return {};
            expr = bindings.at(expr->name);
        }
        return expr;
    }

    static JSON constant(Expression expr, const Bindings& bindings, int depth = 0) {
        if (depth > 32)
            return {};
        expr = resolve_expression(std::move(expr), bindings);
        if (!expr)
            return {};
        if (expr->form == dgl::Expr::Form::Literal)
            return expr->literal;
        if (expr->form == dgl::Expr::Form::List) {
            auto values = JSON::array();
            for (const auto& item : expr->arguments) {
                auto value = constant(item.second, bindings, depth + 1);
                if (value.is_null())
                    return {};
                values.push_back(std::move(value));
            }
            return values;
        }
        return {};
    }

    static Expression argument(const dgl::Expr& call, const char* name, size_t position = SIZE_MAX) {
        size_t index = 0;
        for (const auto& [key, expr] : call.arguments) {
            if (key == name || (key.empty() && index == position))
                return expr;
            if (key.empty())
                ++index;
        }
        return {};
    }

    static void collect_calls(const Expression& expr, std::map<std::string, std::vector<Expression>>& calls) {
        if (!expr)
            return;
        if (expr->form == dgl::Expr::Form::Call)
            calls[expr->name].push_back(expr);
        for (const auto& item : expr->arguments)
            collect_calls(item.second, calls);
        for (const auto& statement : expr->body)
            collect_calls(statement.value, calls);
    }

    static std::string filename(const std::string& path) {
        const auto offset = path.find_last_of("/\\");
        return offset == std::string::npos ? path : path.substr(offset + 1);
    }

    nlohmann::json generation_metadata(const std::string& source) noexcept {
        try {
            const auto syntax = dgl::parse_syntax(source, true);
            Bindings bindings;
            std::map<std::string, std::vector<Expression>> calls;
            for (const auto& statement : syntax.entry.body) {
                if (!statement.name.empty())
                    bindings.emplace(statement.name, statement.value);
                collect_calls(statement.value, calls);
            }
            if (calls["sample"].empty() && calls["sefi::sample"].empty())
                return JSON::object();
            for (const auto& shape : calls["latent_shape"])
                if (argument(*shape, "frames"))
                    return JSON::object();
            auto info = JSON::object();
            for (const auto* key : {"prompt", "negative_prompt", "seed", "width", "height", "steps", "cfg_scale", "method", "scheduler", "eta", "strength", "clip_skip", "model_path", "vae_path"}) {
                if (bindings.count(key)) {
                    auto value = constant(bindings.at(key), bindings);
                    if (!value.is_null())
                        info[key] = std::move(value);
                }
            }
            for (const auto* key : {"prompt", "negative_prompt"})
                if (bindings.count(std::string("source_") + key))
                    info[key] = constant(bindings.at(std::string("source_") + key), bindings);
            const auto select = [&](const std::vector<Expression>& candidates, const char* parameter, size_t position, const char* output) {
                JSON selected;
                for (const auto& call : candidates) {
                    const auto value = constant(argument(*call, parameter, position), bindings);
                    if (value.is_null() || (!selected.is_null() && selected != value)) {
                        info.erase(output);
                        return;
                    }
                    selected = value;
                }
                if (!selected.is_null())
                    info[output] = std::move(selected);
            };
            select(calls["load_dit"], "path", 0, "model_path");
            select(calls["load_unet"], "path", 0, "model_path");
            select(calls["load_vae"], "path", 0, "vae_path");
            select(calls["cfg"], "scale", 2, "cfg_scale");
            select(calls["make_sigmas"], "steps", SIZE_MAX, "steps");
            select(calls["make_sigmas"], "scheduler", SIZE_MAX, "scheduler");
            const auto samples = calls["sample"].size() + calls["sefi::sample"].size();
            if (samples == 1) {
                auto& sample = calls["sample"].empty() ? calls["sefi::sample"] : calls["sample"];
                select(sample, "method", SIZE_MAX, "method");
                select(sample, "eta", SIZE_MAX, "eta");
            } else {
                for (const auto* key : {"steps", "method", "eta", "scheduler", "strength"})
                    info.erase(key);
            }
            auto loras = JSON::array();
            std::set<std::string> patches;
            for (const auto& call : calls["apply_lora"]) {
                const auto patch = resolve_expression(argument(*call, "lora", 1), bindings);
                if (!patch || patch->form != dgl::Expr::Form::Call || patch->name != "load_lora")
                    continue;
                const auto path = constant(argument(*patch, "path", 0), bindings);
                auto weight     = constant(argument(*call, "weight"), bindings);
                if (path.is_string() && !weight.is_null()) {
                    auto item = JSON{{"path", path}, {"weight", weight}};
                    if (patches.insert(item.dump()).second)
                        loras.push_back(std::move(item));
                }
            }
            info["loras"]        = std::move(loras);
            std::string infotext = info.value("prompt", "");
            for (const auto& lora : info.value("loras", JSON::array())) {
                auto name            = filename(lora.at("path"));
                const auto extension = name.find_last_of('.');
                if (extension != std::string::npos)
                    name.erase(extension);
                if (!name.empty() && name.find_first_of(":<>\r\n") == std::string::npos)
                    infotext += (infotext.empty() ? "" : " ") + std::string("<lora:") + name + ":" + lora.at("weight").dump() + ">";
            }
            const auto negative = info.value("negative_prompt", "");
            if (!negative.empty())
                infotext += "\nNegative prompt: " + negative;
            infotext += "\n";
            bool first       = true;
            const auto field = [&](const std::string& key, const JSON& value) {
                std::string text = value.is_string() ? value.get<std::string>() : value.dump();
                if (text.find_first_of(",\n\r:\"") != std::string::npos)
                    text = JSON(text).dump();
                infotext += (first ? "" : ", ") + key + ": " + text;
                first = false;
            };
            for (const auto& [key, label] : std::map<std::string, std::string>{{"steps", "Steps"}, {"cfg_scale", "CFG scale"}, {"seed", "Seed"}, {"eta", "Eta"}, {"strength", "Denoising strength"}})
                if (info.contains(key))
                    field(label, info.at(key));
            const std::map<std::string, std::string> samplers{{"euler", "Euler"}, {"euler_a", "Euler a"}, {"heun", "Heun"}, {"dpm2", "DPM2"}, {"dpm++2s_a", "DPM++ 2S a"}, {"dpm++2m", "DPM++ 2M"}, {"dpm++2m_sde", "DPM++ 2M SDE"}, {"lcm", "LCM"}};
            const std::map<std::string, std::string> schedulers{{"discrete", "Normal"}, {"karras", "Karras"}, {"exponential", "Exponential"}, {"sgm_uniform", "SGM Uniform"}, {"simple", "Simple"}, {"ddim_uniform", "DDIM"}, {"beta", "Beta"}};
            for (const auto& [key, label, unknown, names] : {std::tuple{"method", "Sampler", "SDCPP Sampler", &samplers}, {"scheduler", "Schedule type", "SDCPP Schedule", &schedulers}})
                if (info.contains(key)) {
                    const auto name  = info.at(key).get<std::string>();
                    const auto found = names->find(name);
                    field(found == names->end() ? unknown : label, found == names->end() ? name : found->second);
                }
            if (info.contains("width") && info.contains("height"))
                field("Size", info.at("width").dump() + "x" + info.at("height").dump());
            for (const auto& [key, label] : std::map<std::string, std::string>{{"model_path", "Model"}, {"vae_path", "VAE"}})
                if (info.contains(key))
                    field(label, filename(info.at(key)));
            if (info.value("clip_skip", 0) > 0)
                field("Clip skip", info.at("clip_skip"));
            field("Version", "stable-diffusion.cpp");
            info["infotext"] = std::move(infotext);
            return info;
        } catch (...) {
            return JSON::object();
        }
    }
}

static thread_local std::string generation_metadata_json;

const char* sd_get_generation_metadata(const char* source) {
    const char* result = nullptr;
    sd::api::call([&] {
        if (!source)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "source is required");
        generation_metadata_json = sd::api::generation_metadata(source).dump();
        result                   = generation_metadata_json.c_str();
        return SD_STATUS_OK;
    });
    return result;
}
