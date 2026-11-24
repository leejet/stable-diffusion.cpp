#include "dgl/bindings/values.h"

#include "dgl/bindings/conversion.h"

#include <algorithm>
#include <cmath>

namespace sd::dgl {
    static void validate_options(const Values& settings, const std::string& operation) {
        const auto& definition = operations().at(operation);
        for (const auto& setting : settings) {
            auto argument = std::find_if(definition.arguments.begin(), definition.arguments.end(), [&](const Argument& arg) { return arg.name == setting.first; });
            if (argument == definition.arguments.end())
                throw std::runtime_error("unknown " + operation + " option '" + setting.first + "'");
            const auto& value = setting.second;
            auto accepts      = [](Kind expected, Kind actual) { return expected == actual || (expected == Kind::Number && actual == Kind::Integer); };
            if (!accepts(argument->type.kind, value.type().kind))
                throw std::runtime_error("invalid type for " + operation + "." + setting.first);
            if (value.type().kind == Kind::List)
                for (const auto& item : value.items)
                    if (!accepts(argument->type.element, item.type().kind))
                        throw std::runtime_error("invalid list element for " + operation + "." + setting.first);
            if (argument->validate)
                argument->validate(value);
        }
    }

    sd::CacheConfig cache_options(const Values& settings) {
        sd::CacheConfig params;
        validate_options(settings, "cache");
        for (const auto& option : settings) {
            const auto& key   = option.first;
            const auto& value = option.second.scalar;
            if (key == "mode") {
                static const std::map<std::string, sd_cache_mode_t> modes{{"disabled", SD_CACHE_DISABLED}, {"easycache", SD_CACHE_EASYCACHE}, {"ucache", SD_CACHE_UCACHE}, {"dbcache", SD_CACHE_DBCACHE}, {"taylorseer", SD_CACHE_TAYLORSEER}, {"cache_dit", SD_CACHE_CACHE_DIT}, {"spectrum", SD_CACHE_SPECTRUM}};
                auto found = modes.find(value.get<std::string>());
                if (found == modes.end())
                    throw std::runtime_error("unknown cache mode");
                params.mode = found->second;
            } else if (key == "reuse_threshold")
                params.reuse_threshold = value.get<float>();
            else if (key == "start_percent")
                params.start_percent = value.get<float>();
            else if (key == "end_percent")
                params.end_percent = value.get<float>();
            else if (key == "error_decay_rate")
                params.error_decay_rate = value.get<float>();
            else if (key == "use_relative_threshold")
                params.use_relative_threshold = value.get<bool>();
            else if (key == "reset_error_on_compute")
                params.reset_error_on_compute = value.get<bool>();
            else if (key == "residual_diff_threshold")
                params.residual_diff_threshold = value.get<float>();
            else if (key == "max_warmup_steps")
                params.max_warmup_steps = value.get<int>();
            else if (key == "max_cached_steps")
                params.max_cached_steps = value.get<int>();
            else if (key == "max_continuous_cached_steps")
                params.max_continuous_cached_steps = value.get<int>();
            else if (key == "taylorseer_n_derivatives")
                params.taylorseer_n_derivatives = value.get<int>();
            else if (key == "taylorseer_skip_interval")
                params.taylorseer_skip_interval = value.get<int>();
            else if (key == "scm_mask")
                params.scm_mask = value.get_ref<const std::string&>().c_str();
            else if (key == "spectrum_w")
                params.spectrum_w = value.get<float>();
            else if (key == "spectrum_m")
                params.spectrum_m = value.get<int>();
            else if (key == "spectrum_lam")
                params.spectrum_lam = value.get<float>();
            else if (key == "spectrum_window_size")
                params.spectrum_window_size = value.get<int>();
            else if (key == "spectrum_flex_window")
                params.spectrum_flex_window = value.get<float>();
            else if (key == "spectrum_warmup_steps")
                params.spectrum_warmup_steps = value.get<int>();
            else if (key == "spectrum_stop_percent")
                params.spectrum_stop_percent = value.get<float>();
            else
                throw std::runtime_error("unknown cache option '" + key + "'");
        }
        if (params.start_percent > params.end_percent)
            throw std::runtime_error("cache start_percent exceeds end_percent");
        return params;
    }

    sd::TilingConfig tiling_options(const Values& settings) {
        sd::TilingConfig params{};
        params.enabled        = true;
        params.target_overlap = 0.5f;
        validate_options(settings, "tiling");
        for (const auto& option : settings) {
            const auto& key   = option.first;
            const auto& value = option.second.scalar;
            if (key == "enabled")
                params.enabled = value.get<bool>();
            else if (key == "temporal_tiling")
                params.temporal_tiling = value.get<bool>();
            else if (key == "tile_size_w")
                params.tile_size_w = value.get<int>();
            else if (key == "tile_size_h")
                params.tile_size_h = value.get<int>();
            else if (key == "target_overlap")
                params.target_overlap = value.get<float>();
            else if (key == "rel_size_w")
                params.rel_size_w = value.get<float>();
            else if (key == "rel_size_h")
                params.rel_size_h = value.get<float>();
            else if (key == "extra_args")
                params.extra_tiling_args = value.get_ref<const std::string&>().c_str();
        }
        return params;
    }

    Value execute_list_replace(Values& args, RunState&) {
        auto list         = args.at("items");
        const auto index  = uint64_t(get_int(args, "index"));
        const auto& value = args.at("value");
        if (index >= list.items.size())
            throw std::runtime_error("list index out of range");
        if (!accepts_list_element(list.type().element, value.type().kind))
            throw std::runtime_error("replacement must match the list element type");
        list.items[size_t(index)] = value;
        return list;
    }

    Value execute_math_product(Values& args, RunState&) {
        int64_t product = 1;
        for (const auto n : read_int_vector(args.at("items"))) {
            if (n < 0 || (n && product > INT64_MAX / n))
                throw std::runtime_error("product requires nonnegative integers and an int64 result");
            product *= n;
        }
        return Value::literal(product);
    }

    Value execute_math_ceil_div(Values& args, RunState&) {
        const auto a = get_int(args, "numerator"), b = get_int(args, "denominator");
        return Value::literal(a / b + int64_t(a % b != 0));
    }

    Value execute_math_min(Values& args, RunState&) {
        return Value::literal(std::min(get_int(args, "a"), get_int(args, "b")));
    }

    Value execute_math_add(Values& args, RunState&) {
        if (args.at("a").type().kind != Kind::Integer || args.at("b").type().kind != Kind::Integer)
            return Value::literal(get_number(args, "a") + get_number(args, "b"));
        const auto a = get_int(args, "a"), b = get_int(args, "b");
        if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b))
            throw std::runtime_error("addition exceeds int64 range");
        return Value::literal(a + b);
    }

    Value execute_math_sub(Values& args, RunState&) {
        if (args.at("a").type().kind != Kind::Integer || args.at("b").type().kind != Kind::Integer)
            return Value::literal(get_number(args, "a") - get_number(args, "b"));
        const auto a = get_int(args, "a"), b = get_int(args, "b");
        if ((b > 0 && a < INT64_MIN + b) || (b < 0 && a > INT64_MAX + b))
            throw std::runtime_error("subtraction exceeds int64 range");
        return Value::literal(a - b);
    }

    Value execute_math_mul(Values& args, RunState&) {
        if (args.at("a").type().kind != Kind::Integer || args.at("b").type().kind != Kind::Integer)
            return Value::literal(get_number(args, "a") * get_number(args, "b"));
        const auto a = get_int(args, "a"), b = get_int(args, "b");
        if ((a > 0 && ((b > 0 && a > INT64_MAX / b) || (b < 0 && b < INT64_MIN / a))) ||
            (a < 0 && ((b > 0 && a < INT64_MIN / b) || (b < 0 && a < INT64_MAX / b))))
            throw std::runtime_error("multiplication exceeds int64 range");
        return Value::literal(a * b);
    }

    Value execute_math_div(Values& args, RunState&) {
        const auto divisor = get_number(args, "b");
        if (divisor == 0.0)
            throw std::runtime_error("division by zero");
        return Value::literal(get_number(args, "a") / divisor);
    }

    Value execute_configuration(Values& args, RunState&) {
        return Value::dict(args);
    }

    static Value round_math(const Values& args, double (*round)(double)) {
        const auto& value = args.at("value");
        if (value.type().kind == Kind::Integer)
            return value;
        const auto result = round(get_number(args, "value"));
        const auto limit  = std::ldexp(1.0, 63);
        if (!(result >= -limit && result < limit))
            throw std::runtime_error("rounded value exceeds int64 range");
        return Value::literal(int64_t(result));
    }

    Value execute_math_floor(Values& args, RunState&) {
        return round_math(args, std::floor);
    }

    Value execute_math_round(Values& args, RunState&) {
        return round_math(args, std::round);
    }

    Value execute_math_ceil(Values& args, RunState&) {
        return round_math(args, std::ceil);
    }

}  // namespace sd::dgl
