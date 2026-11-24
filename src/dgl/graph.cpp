#include "graph.h"
#include "indexing.h"

#include <algorithm>
#include <set>
#include <stdexcept>

#include "diagnostic.h"

namespace sd::dgl {

    static bool is_literal_argument(const Expr& expr) {
        if (expr.form == Expr::Form::Literal)
            return true;
        if (expr.form != Expr::Form::List && expr.form != Expr::Form::Dict)
            return false;
        return std::all_of(expr.arguments.begin(), expr.arguments.end(), [](const auto& entry) { return is_literal_argument(*entry.second); });
    }

    std::vector<BoundArgument> bind_arguments(const Expr& expr, const std::vector<std::string>& names, bool all_required) {
        std::vector<BoundArgument> args;
        args.reserve(expr.arguments.size());
        std::vector<bool> bound(names.size());
        size_t position = 0;
        for (const auto& [name, expression] : expr.arguments) {
            size_t parameter;
            if (name.empty()) {
                if (position >= names.size())
                    throw std::runtime_error("too many arguments for '" + expr.name + "'");
                parameter = position++;
            } else {
                const auto found = std::find(names.begin(), names.end(), name);
                if (found == names.end())
                    throw std::runtime_error("unknown argument '" + name + "' for '" + expr.name + "'");
                parameter = static_cast<size_t>(found - names.begin());
            }
            if (bound[parameter])
                throw std::runtime_error("duplicate argument '" + names[parameter] + "'");
            bound[parameter] = true;
            args.push_back({parameter, expression});
        }
        if (all_required) {
            const auto missing = std::find(bound.begin(), bound.end(), false);
            if (missing != bound.end())
                throw std::runtime_error("missing function argument '" + names[missing - bound.begin()] + "'");
        }
        return args;
    }

    std::vector<std::string> argument_names(const Operation& operation) {
        std::vector<std::string> names;
        for (const auto& arg : operation.arguments)
            names.push_back(arg.name);
        return names;
    }

    IterationCall bind_iteration(const Expr& expr, const Syntax& syntax) {
        bool fold            = expr.name == "fold";
        size_t process_index = fold ? 2 : 1;
        if (expr.arguments.size() <= process_index)
            throw std::runtime_error(fold ? "fold requires items, initial and process" : "map requires items and process");
        auto named = [&](size_t i, const char* name) { return expr.arguments[i].first.empty() || expr.arguments[i].first == name; };
        if (!named(0, "items") || (fold && !named(1, "initial")) || !named(process_index, "process"))
            throw std::runtime_error(fold ? "fold starts with items, initial and process" : "map starts with items and process");
        const auto& process = expr.arguments[process_index].second;
        auto found          = syntax.functions.find(process->name);
        if (process->form != Expr::Form::Reference || found == syntax.functions.end())
            throw std::runtime_error(expr.name + " process must name a function");
        const auto& function = found->second;
        if (function.parameters.size() < (fold ? 2 : 1))
            throw std::runtime_error(fold ? "fold process requires accumulator and item parameters" : "map process requires an item parameter");
        Expr call = expr;
        call.arguments.erase(call.arguments.begin() + process_index);
        call.arguments[0].first.clear();
        if (fold) {
            call.arguments[1].first.clear();
            std::swap(call.arguments[0], call.arguments[1]);
        }
        auto bound = bind_arguments(call, function.parameters, true);
        return {&function, process->name, expr.arguments[0].second, fold ? expr.arguments[1].second : nullptr, std::move(bound)};
    }

    struct Checked {
        Type type;
        std::shared_ptr<Expr> literal{};
        std::map<std::string, Checked> fields{};
        std::optional<size_t> tensor_rank;
        Kind tensor_scalar = Kind::Any;
        std::optional<size_t> length;
        std::vector<Checked> items;
        bool complete_fields = false;
    };

    using Types = std::map<std::string, Checked>;

    static Checked checked_output(const OutputSignature& signature) {
        Checked result{signature.type};
        for (const auto& [name, field] : signature.fields)
            result.fields.emplace(name, checked_output(field));
        return result;
    }

    static bool same_shape(const Checked& a, const Checked& b) {
        if (!(a.type == b.type) || a.fields.size() != b.fields.size())
            return false;
        for (const auto& field : a.fields) {
            auto other = b.fields.find(field.first);
            if (other == b.fields.end() || !same_shape(field.second, other->second))
                return false;
        }
        return true;
    }

    static Checked shape_only(const Checked& value) {
        Checked result{value.type};
        result.tensor_rank   = value.tensor_rank;
        result.tensor_scalar = value.tensor_scalar;
        for (const auto& field : value.fields)
            result.fields.emplace(field.first, shape_only(field.second));
        return result;
    }

    static void clear_tensor_facts(Checked& value) {
        value.tensor_rank.reset();
        value.tensor_scalar = Kind::Any;
        for (auto& field : value.fields)
            clear_tensor_facts(field.second);
    }

    static void keep_common_fields(Checked& value, const Checked& other) {
        if (value.tensor_rank != other.tensor_rank)
            value.tensor_rank.reset();
        if (value.tensor_scalar != other.tensor_scalar)
            value.tensor_scalar = Kind::Any;
        for (auto field = value.fields.begin(); field != value.fields.end();) {
            const auto match = other.fields.find(field->first);
            if (match == other.fields.end() || !(field->second.type == match->second.type))
                field = value.fields.erase(field);
            else {
                keep_common_fields(field->second, match->second);
                ++field;
            }
        }
    }

    static void summarize_list(Checked& list) {
        if (list.items.empty())
            return;
        auto item = shape_only(list.items.front());
        for (size_t i = 1; i < list.items.size(); ++i)
            keep_common_fields(item, list.items[i]);
        list.fields        = std::move(item.fields);
        list.tensor_rank   = item.tensor_rank;
        list.tensor_scalar = item.tensor_scalar;
    }

    static void infer_tensor_result(const std::string& name, const Types& args, Checked& result) {
        const auto find = [&](const char* key) -> const Checked* {
            const auto found = args.find(key);
            return found == args.end() ? nullptr : &found->second;
        };
        const auto input  = find("input");
        const auto dtype  = find("dtype");
        const auto scalar = [&] {
            if (!dtype)
                return Kind::Number;
            if (!dtype->literal)
                return Kind::Any;
            const auto& text = dtype->literal->literal;
            return text == "u8" || text == "i32" || text == "i64" ? Kind::Integer : Kind::Number;
        };
        if (name == "tensor::shape" || name == "latent_shape" || name == "audio_latent_shape" || name == "tensor::arange") {
            result.tensor_rank   = 1;
            result.tensor_scalar = Kind::Integer;
            if (name == "tensor::shape" && input)
                result.length = input->tensor_rank;
        } else if (name == "tensor::vector") {
            result.tensor_rank   = 1;
            result.tensor_scalar = scalar();
            if (const auto values = find("values"))
                result.length = values->length;
        } else if (name == "tensor::zeros" || name == "tensor::full" || name == "tensor::literal" ||
                   name == "tensor::reshape" || name == "tensor::broadcast_to") {
            if (const auto shape = find("shape"))
                result.tensor_rank = shape->length;
            result.tensor_scalar = input ? input->tensor_scalar : scalar();
        } else if (name == "tensor::unsqueeze") {
            if (input && input->tensor_rank)
                result.tensor_rank = *input->tensor_rank + 1;
            if (input)
                result.tensor_scalar = input->tensor_scalar;
        } else if (name == "tensor::pad" || name == "tensor::crop" || name == "tensor::permute" ||
                   name == "tensor::interpolate" || name == "tensor::translate" || name == "tensor::index_select" ||
                   name == "tensor::index_fill" || name == "tensor::cast") {
            if (input) {
                result.tensor_rank   = input->tensor_rank;
                result.tensor_scalar = name == "tensor::cast" ? scalar() : input->tensor_scalar;
            }
        } else if (name == "tensor::zeros_like" || name == "tensor::randn_like") {
            if (const auto like = find("like")) {
                result.tensor_rank   = like->tensor_rank;
                result.tensor_scalar = like->tensor_scalar;
            }
        } else if (name == "tensor::concat") {
            if (const auto items = find("items")) {
                result.tensor_rank   = items->tensor_rank;
                result.tensor_scalar = items->tensor_scalar;
            }
        } else if (name == "make_sigmas") {
            result.tensor_rank   = 1;
            result.tensor_scalar = Kind::Number;
        } else if (name == "sefi::make_sigmas") {
            for (const char* field : {"sigmas", "semantic_sigmas"}) {
                result.fields.at(field).tensor_rank   = 1;
                result.fields.at(field).tensor_scalar = Kind::Number;
            }
        } else if (name == "image::load") {
            result.tensor_rank   = 3;
            result.tensor_scalar = Kind::Integer;
        } else if (name == "audio::load" || name == "audio_vae_decode") {
            result.fields.at("samples").tensor_rank   = 2;
            result.fields.at("samples").tensor_scalar = Kind::Number;
        } else if (name == "video::load") {
            auto& frames                                                 = result.fields.at("video").fields.at("frames");
            frames.tensor_rank                                           = 4;
            frames.tensor_scalar                                         = Kind::Integer;
            result.fields.at("audio").fields.at("samples").tensor_rank   = 2;
            result.fields.at("audio").fields.at("samples").tensor_scalar = Kind::Number;
        }
    }

    class Checker {
        Graph& graph_;
        const Registry& registry_;
        std::set<std::string> active_;
        size_t visits_           = 0;
        unsigned callback_depth_ = 0;
        std::vector<const Expr*> call_path_;

        const BoundCall& resolve(const Expr& expr) {
            const auto existing = graph_.calls.find(&expr);
            if (existing != graph_.calls.end())
                return existing->second;
            BoundCall call;
            if (expr.name == "map" || expr.name == "fold") {
                call.iteration = bind_iteration(expr, graph_.syntax);
            } else if (const auto user = graph_.syntax.functions.find(expr.name); user != graph_.syntax.functions.end()) {
                call.function  = &user->second;
                call.arguments = bind_arguments(expr, call.function->parameters, true);
            } else {
                const auto found = registry_.find(expr.name);
                if (found == registry_.end())
                    throw std::runtime_error("unknown function '" + expr.name + "'");
                call.operation = &*graph_.operations.try_emplace(found->first, found->second).first;
                call.arguments = bind_arguments(expr, argument_names(call.operation->second));
            }
            return graph_.calls.emplace(&expr, std::move(call)).first->second;
        }

        void constrain(const Checked& value, const Argument& argument) {
            if (!accepts(argument.type, value.type))
                throw Error(SD_STATUS_TYPE_MISMATCH, "argument '" + argument.name + "' expects " + type_name(argument.type) + ", got " + type_name(value.type));
            if (value.literal && argument.validate)
                argument.validate(Value::literal(value.literal->literal));
        }

        Checked check_tensor_to_list(const Types& args) {
            const auto& input = args.at("input");
            if (input.tensor_rank && *input.tensor_rank != 1)
                throw std::runtime_error("tensor::to_list requires a rank-one tensor");
            const auto option   = args.find("element");
            std::string element = "auto";
            if (option != args.end()) {
                if (!option->second.literal)
                    throw std::runtime_error("tensor::to_list element must be known at compile time");
                element = option->second.literal->literal.get<std::string>();
            }
            const auto kind = element == "auto" ? input.tensor_scalar : element == "int" ? Kind::Integer
                                                                                         : Kind::Number;
            if (kind == Kind::Any)
                throw std::runtime_error("tensor::to_list cannot infer dtype; specify element = \"int\" or \"number\"");
            if (input.tensor_scalar != Kind::Any && input.tensor_scalar != kind)
                throw Error(SD_STATUS_TYPE_MISMATCH, "tensor::to_list element does not match input dtype; use tensor::cast to convert it");
            Checked result{{Kind::List, kind}};
            result.length = input.length;
            return result;
        }

        Checked check_tensor_index(const Types& args) {
            Checked result{{Kind::Any}};
            const auto input = args.find("input"), indices = args.find("indices");
            if (input == args.end() || indices == args.end() || !indices->second.length ||
                indices->second.items.size() != *indices->second.length)
                return result;
            const auto& items = indices->second.items;
            if (items.empty())
                throw std::runtime_error("tensor indexing requires at least one axis");
            if (input->second.tensor_rank && items.size() > *input->second.tensor_rank)
                throw std::runtime_error("tensor indices exceed the known tensor rank");
            size_t integers = 0;
            bool slice = false, known = true;
            for (const auto& axis : items) {
                for (const auto& [name, field] : axis.fields) {
                    if (name != "index" && name != "start" && name != "end")
                        throw std::runtime_error("unknown tensor index field '" + name + "'");
                    if (field.type.kind != Kind::Integer && field.type.kind != Kind::Any)
                        throw Error(SD_STATUS_TYPE_MISMATCH, "tensor index field '" + name + "' must be an integer");
                }
                if (axis.fields.count("index")) {
                    if (axis.fields.size() != 1)
                        throw std::runtime_error("a tensor index cannot also have slice endpoints");
                    ++integers;
                } else if (axis.complete_fields) {
                    slice = true;
                } else {
                    known = false;
                }
            }
            if (slice)
                result.type = {Kind::Tensor};
            if (known && input->second.tensor_rank) {
                const auto rank = *input->second.tensor_rank - integers;
                if (rank) {
                    result.type          = {Kind::Tensor};
                    result.tensor_rank   = rank;
                    result.tensor_scalar = input->second.tensor_scalar;
                } else {
                    result.type = {input->second.tensor_scalar == Kind::Any ? Kind::Number : input->second.tensor_scalar};
                }
            }
            return result;
        }

        Checked check_list_index(const Checked& input, const Checked& indices) {
            if (indices.items.size() != 1)
                throw std::runtime_error("list indexing requires exactly one integer index or slice");
            const auto& axis = indices.items.front();
            const bool slice = !axis.fields.count("index");
            if (!slice && axis.fields.size() != 1)
                throw std::runtime_error("a list index cannot also have slice endpoints");
            Values constant;
            bool known = axis.complete_fields;
            for (const auto& [name, field] : axis.fields) {
                if (name != "index" && name != "start" && name != "end")
                    throw std::runtime_error("unknown list index field '" + name + "'");
                if (field.type.kind != Kind::Integer && field.type.kind != Kind::Any)
                    throw Error(SD_STATUS_TYPE_MISMATCH, "list index field '" + name + "' must be an integer");
                if (field.literal)
                    constant.emplace(name, Value::literal(field.literal->literal));
                else
                    known = false;
            }
            if (known && input.length) {
                const auto selected = select_list(*input.length, constant);
                if (input.items.size() == *input.length) {
                    if (!slice)
                        return input.items[selected.begin];
                    auto result = input;
                    result.items.assign(input.items.begin() + selected.begin, input.items.begin() + selected.end);
                    result.length = result.items.size();
                    summarize_list(result);
                    return result;
                }
            }
            auto result = shape_only(input);
            result.type = slice ? input.type : Type{input.type.element};
            return result;
        }

        Checked function_at(const Expr& site, const std::string& name, const Types& params) {
            call_path_.push_back(&site);
            auto result = function(name, params);
            call_path_.pop_back();
            return result;
        }

        Checked function(const std::string& name, const Types& params) {
            for (const auto& [parameter, value] : params)
                if (value.type.kind == Kind::Void)
                    throw std::runtime_error("argument '" + parameter + "' has no value");
            if (!active_.insert(name).second || active_.size() > 32)
                throw std::runtime_error("recursive functions are not supported");
            auto result = body(graph_.syntax.functions.at(name).body, params);
            active_.erase(name);
            return result;
        }

    public:
        Checker(Graph& graph, const Registry& registry)
            : graph_(graph), registry_(registry) {}

        Checked expression(const std::shared_ptr<Expr>& expr, const Types& scope, Kind list_element = Kind::Any, bool discard_result = false) {
            if (++visits_ > 100000)
                throw std::runtime_error("expanded graph limit exceeded");
            try {
                if (expr->form == Expr::Form::Literal)
                    return {Value::literal(expr->literal).type(), expr};
                if (expr->form == Expr::Form::Block)
                    return body(expr->body, scope, list_element);
                if (expr->form == Expr::Form::Conditional) {
                    const auto condition = expression(expr->arguments[0].second, scope);
                    constrain(condition, required("if condition", Kind::Boolean));
                    if (!condition.literal)
                        throw std::runtime_error("if condition must be a boolean known before resource planning");
                    if (expr->arguments.size() == 2) {
                        if (!discard_result)
                            throw std::runtime_error("if used as a value requires else");
                        expression(expr->arguments[1].second, scope);
                        return {{Kind::Void}};
                    }
                    auto selected    = expression(expr->arguments[1].second, scope, list_element);
                    auto alternative = expression(expr->arguments[2].second, scope, list_element);
                    if (!same_shape(selected, alternative))
                        throw Error(SD_STATUS_TYPE_MISMATCH, "if branches must return the same type and dictionary fields, got " +
                                                                 type_name(selected.type) + " and " + type_name(alternative.type));
                    return condition.literal->literal.get<bool>() ? std::move(selected) : std::move(alternative);
                }
                if (expr->form == Expr::Form::Reference) {
                    size_t dot = expr->name.find('.');
                    auto found = scope.find(expr->name.substr(0, dot));
                    if (found == scope.end())
                        throw std::runtime_error("unknown binding '" + expr->name + "'");
                    Checked result = found->second;
                    while (dot != std::string::npos) {
                        auto next  = expr->name.find('.', dot + 1);
                        auto field = expr->name.substr(dot + 1, next == std::string::npos ? next : next - dot - 1);
                        if (result.type.kind != Kind::List && result.fields.count(field)) {
                            auto child = result.fields.at(field);
                            result     = std::move(child);
                        } else
                            throw std::runtime_error("unknown field '" + expr->name + "'");
                        dot = next;
                    }
                    return result;
                }
                if (expr->form == Expr::Form::List) {
                    Checked result{{Kind::List}};
                    result.length = expr->arguments.size();
                    auto& element = result.type.element;
                    for (const auto& item : expr->arguments) {
                        auto checked = expression(item.second, scope);
                        if (list_element != Kind::Any) {
                            constrain(checked, required("list item", list_element));
                            if (checked.type.kind == Kind::Any)
                                checked.type.kind = list_element;
                        }
                        if (checked.type.kind == Kind::Any || checked.type.kind == Kind::List || checked.type.kind == Kind::Void)
                            throw std::runtime_error("list items require a known non-list type");
                        if (element == Kind::Any) {
                            element              = checked.type.kind;
                            result.fields        = shape_only(checked).fields;
                            result.tensor_rank   = checked.tensor_rank;
                            result.tensor_scalar = checked.tensor_scalar;
                        }
                        if ((element == Kind::Integer && checked.type.kind == Kind::Number) || (element == Kind::Number && checked.type.kind == Kind::Integer))
                            element = Kind::Number;
                        else if (!accepts_list_element(element, checked.type.kind))
                            throw std::runtime_error("list items have different types");
                        keep_common_fields(result, checked);
                        result.items.push_back(std::move(checked));
                    }
                    return result;
                }
                if (expr->form == Expr::Form::Dict) {
                    Checked result{{Kind::Dict}};
                    result.complete_fields = true;
                    for (const auto& field : expr->arguments) {
                        auto value = expression(field.second, scope);
                        if (value.type.kind == Kind::Void)
                            throw std::runtime_error("dict field '" + field.first + "' has no value");
                        result.fields.emplace(field.first, std::move(value));
                    }
                    return result;
                }
                if (expr->form == Expr::Form::Index) {
                    auto input   = expression(expr->arguments[0].second, scope);
                    auto indices = expression(expr->arguments[1].second, scope);
                    if (input.type.kind == Kind::List)
                        return check_list_index(input, indices);
                    if (input.type.kind != Kind::Tensor && input.type.kind != Kind::Any)
                        throw Error(SD_STATUS_TYPE_MISMATCH, "indexing requires a list or tensor");
                    return check_tensor_index({{"input", input}, {"indices", indices}});
                }
                const auto& bound = resolve(*expr);
                if (bound.iteration) {
                    const auto& call = *bound.iteration;
                    auto list        = expression(call.items, scope);
                    if (list.type.kind != Kind::List || list.type.element == Kind::Any)
                        throw std::runtime_error(expr->name + " requires a typed list");
                    const auto& parameters = call.function->parameters;
                    size_t item_index      = call.initial ? 1 : 0;
                    auto item              = shape_only(list);
                    item.type              = {list.type.element};
                    Types locals{{parameters[item_index], std::move(item)}};
                    for (const auto& arg : call.arguments)
                        if (arg.parameter != item_index)
                            locals.emplace(parameters[arg.parameter], expression(arg.expression, scope));
                    if (call.initial) {
                        auto shape = shape_only(locals.at(parameters[0]));
                        // Iterations may change tensor rank or dtype while preserving the language type.
                        clear_tensor_facts(shape);
                        if (shape.type.kind == Kind::Any)
                            throw std::runtime_error("fold requires a typed accumulator");
                        locals[parameters[0]] = shape;
                        if (!same_shape(shape, function_at(*expr, call.process, locals)))
                            throw std::runtime_error("fold process must preserve the accumulator type and fields");
                        return shape;
                    }
                    auto result = function_at(*expr, call.process, locals);
                    if (result.type.kind == Kind::List || result.type.kind == Kind::Any || result.type.kind == Kind::Void)
                        throw std::runtime_error("map process must return one typed value");
                    const auto output = std::find_if(graph_.map_outputs.begin(), graph_.map_outputs.end(), [&](const auto& site) {
                        return site.expression == expr.get() && site.call_path == call_path_;
                    });
                    if (output == graph_.map_outputs.end())
                        graph_.map_outputs.push_back({expr.get(), call_path_, result.type.kind});
                    else
                        output->element = result.type.kind;
                    result        = shape_only(result);
                    result.type   = {Kind::List, result.type.kind};
                    result.length = list.length;
                    return result;
                }
                if (bound.function) {
                    const auto& parameters = bound.function->parameters;
                    Types locals;
                    for (const auto& arg : bound.arguments)
                        locals.emplace(parameters[arg.parameter], expression(arg.expression, scope));
                    return function_at(*expr, expr->name, locals);
                }
                const auto& [operation_name, operation] = *bound.operation;
                const auto& args                        = bound.arguments;
                if (callback_depth_ && (operation.resource() || operation.sampling))
                    throw std::runtime_error("prediction transforms cannot load models or start sampling");
                Types checked_args;
                for (const auto& arg : args) {
                    const auto& argument = operation.arguments[arg.parameter];
                    if (argument.callback)
                        continue;
                    const auto element = argument.type.kind == Kind::IntVector ? Kind::Integer
                                         : argument.type.kind == Kind::List    ? argument.type.element
                                                                               : Kind::Any;
                    auto checked       = expression(arg.expression, scope, element);
                    if (operation.resource() && !is_literal_argument(*arg.expression) &&
                        !(active_.empty() && checked.literal))
                        throw std::runtime_error("resource arguments must be literals or literal bindings in main");
                    constrain(checked, argument);
                    checked_args.emplace(argument.name, std::move(checked));
                }
                for (const auto& arg : args) {
                    const auto& argument = operation.arguments[arg.parameter];
                    if (!argument.callback)
                        continue;
                    const auto& expression = arg.expression;
                    const auto fn          = graph_.syntax.functions.find(expression->name);
                    if ((expression->form != Expr::Form::Reference && expression->form != Expr::Form::FunctionReference) ||
                        fn == graph_.syntax.functions.end() || fn->second.parameters.size() != 2)
                        throw std::runtime_error(argument.name + " must name a function with two parameters");
                    expression->form = Expr::Form::FunctionReference;
                    graph_.callbacks.insert_or_assign(expression.get(), &fn->second);
                    const auto& signature = *argument.callback;
                    Checked step{{Kind::Dict}};
                    for (const auto& [name, type] : signature.input_fields)
                        step.fields.emplace(name, Checked{type});
                    const auto callback_args = checked_args.find(signature.args_parameter);
                    auto captures            = callback_args == checked_args.end() ? Checked{{Kind::Dict}} : callback_args->second;
                    ++callback_depth_;
                    auto result = function_at(*expr, fn->first, {{fn->second.parameters[0], std::move(step)}, {fn->second.parameters[1], std::move(captures)}});
                    --callback_depth_;
                    constrain(result, required(argument.name + " return value", Kind::Dict));
                    for (const auto& [name, field] : result.fields) {
                        const auto expected = signature.output_fields.find(name);
                        if (expected == signature.output_fields.end())
                            throw std::runtime_error("unknown " + argument.name + " result field '" + name + "'");
                        if (!accepts(expected->second, field.type))
                            throw Error(SD_STATUS_TYPE_MISMATCH, "invalid type for " + argument.name + " result field '" + name + "'");
                    }
                    checked_args.emplace(argument.name, Checked{{Kind::Function}});
                }
                Checked result{operation.output.type};
                if (operation_name == "math::add" || operation_name == "math::sub" || operation_name == "math::mul") {
                    const auto a = checked_args.find("a"), b = checked_args.find("b");
                    if (a != checked_args.end() && b != checked_args.end()) {
                        const auto left = a->second.type.kind, right = b->second.type.kind;
                        result.type = {left == Kind::Integer && right == Kind::Integer ? Kind::Integer
                                       : left == Kind::Number || right == Kind::Number ? Kind::Number
                                                                                       : Kind::Any};
                    }
                }
                for (const auto& argument : operation.arguments) {
                    const auto found = checked_args.find(argument.name);
                    if (found == checked_args.end()) {
                        if (argument.required)
                            throw std::runtime_error("missing argument '" + argument.name + "'");
                        continue;
                    }
                    const auto& checked = found->second;
                    if (operation.output.same_type_as == argument.name) {
                        result.type   = checked.type;
                        result.fields = shape_only(checked).fields;
                    }
                }
                if (operation_name == "zero_out") {
                    const auto& condition = checked_args.at("condition");
                    result                = shape_only(condition);
                    if (const auto selected = checked_args.find("fields"); selected != checked_args.end())
                        for (const auto& item : selected->second.items) {
                            if (!item.literal)
                                continue;
                            const auto name  = item.literal->literal.get<std::string>();
                            const auto field = condition.fields.find(name);
                            if (field == condition.fields.end()) {
                                if (condition.complete_fields)
                                    throw std::runtime_error("zero_out: condition has no field '" + name + "'");
                            } else if (field->second.type.kind != Kind::Tensor && !(field->second.type == Type{Kind::List, Kind::Tensor})) {
                                throw Error(SD_STATUS_TYPE_MISMATCH, "zero_out: field '" + name + "' must be a tensor or tensor list");
                            }
                        }
                }
                if (operation_name == "list::replace") {
                    result            = checked_args.at("items");
                    const auto& value = checked_args.at("value");
                    if (result.type.element == Kind::Any || !accepts_list_element(result.type.element, value.type.kind))
                        throw Error(SD_STATUS_TYPE_MISMATCH, "replacement must match the list element type: expected " +
                                                                 type_name({result.type.element}) + ", got " + type_name(value.type));
                    const auto& index = checked_args.at("index");
                    if (index.literal && result.length && result.items.size() == *result.length) {
                        const auto selected          = select_list(*result.length, {{"index", Value::literal(index.literal->literal)}});
                        result.items[selected.begin] = value;
                        summarize_list(result);
                    } else {
                        result.items.clear();
                        keep_common_fields(result, value);
                    }
                }
                for (const auto& field : operation.output.fields)
                    result.fields[field.first] = checked_output(field.second);
                infer_tensor_result(operation_name, checked_args, result);
                if (operation_name == "tensor::index")
                    result = check_tensor_index(checked_args);
                else if (operation_name == "tensor::to_list")
                    result = check_tensor_to_list(checked_args);
                return result;
            } catch (const std::bad_alloc&) {
                throw;
            } catch (const std::exception& error) {
                const auto* diagnostic = dynamic_cast<const Error*>(&error);
                throw Error(diagnostic ? diagnostic->status : SD_STATUS_INVALID_ARGUMENT,
                            "line " + std::to_string(expr->line) + ", column " + std::to_string(expr->column) + ": " + error.what());
            }
        }

        Checked body(const std::vector<Statement>& statements, Types scope, Kind list_element = Kind::Any) {
            for (const auto& statement : statements) {
                if (!statement.value)
                    return {{Kind::Void}};
                auto checked = expression(statement.value, scope,
                                          statement.form == Statement::Form::Result ? list_element : Kind::Any,
                                          statement.form == Statement::Form::Call);
                if (statement.form == Statement::Form::Call)
                    continue;
                if (statement.form == Statement::Form::Result)
                    return checked;
                if (checked.type.kind == Kind::Void)
                    throw std::runtime_error("line " + std::to_string(statement.value->line) + ", column " + std::to_string(statement.value->column) + ": function has no return value; call it as a standalone statement");
                if (statement.form == Statement::Form::Return)
                    return checked;
                if (!scope.emplace(statement.name, checked).second)
                    throw std::runtime_error("binding shadows existing name '" + statement.name + "'");
            }
            return {{Kind::Void}};
        }
    };

    static void count_references(const Expr& expr, ReferenceCounts& counts, Graph& graph) {
        if (expr.form == Expr::Form::Block) {
            auto [found, inserted] = graph.blocks.try_emplace(&expr);
            auto& block            = found->second;
            if (inserted) {
                std::set<std::string> locals;
                for (const auto& statement : expr.body) {
                    ReferenceCounts references;
                    if (statement.value)
                        count_references(*statement.value, references, graph);
                    for (const auto& [name, count] : references) {
                        block.references[name] += count;
                        if (!locals.count(name.substr(0, name.find('.'))))
                            block.captures[name] += count;
                    }
                    if (statement.form == Statement::Form::Binding)
                        locals.insert(statement.name);
                }
            }
            for (const auto& [name, count] : block.captures)
                counts[name] += count;
            return;
        }
        if (expr.form == Expr::Form::Reference)
            ++counts[expr.name];
        const bool iteration = expr.form == Expr::Form::Call && (expr.name == "map" || expr.name == "fold");
        const size_t process = expr.name == "fold" ? 2 : 1;
        for (size_t i = 0; i < expr.arguments.size(); ++i) {
            if (!iteration || i != process)
                count_references(*expr.arguments[i].second, counts, graph);
        }
    }

    static ReferenceCounts reference_counts(const Function& function, Graph& graph) {
        ReferenceCounts counts;
        for (const auto& statement : function.body)
            if (statement.value)
                count_references(*statement.value, counts, graph);
        return counts;
    }

    Graph compile(const std::string& source, const Registry& registry, bool host_inputs) {
        Graph graph;
        graph.source = source;
        graph.syntax = parse_syntax(source, host_inputs);
        for (const auto& function : graph.syntax.functions) {
            if (function.first == "map" || function.first == "fold" || registry.count(function.first))
                throw std::runtime_error("function name '" + function.first + "' is reserved for a built-in function");
        }
        Checker checker(graph, registry);
        Types parameters;
        for (const auto& name : graph.syntax.entry.parameters)
            parameters.emplace(name, Checked{{Kind::Tensor}});
        checker.body(graph.syntax.entry.body, std::move(parameters));
        graph.references.emplace("main", reference_counts(graph.syntax.entry, graph));
        for (const auto& [name, function] : graph.syntax.functions)
            graph.references.emplace(name, reference_counts(function, graph));
        analyze_resources(graph);
        return graph;
    }

}  // namespace sd::dgl
