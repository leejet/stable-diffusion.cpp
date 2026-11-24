#include "graph.h"
#include "indexing.h"

#include <set>

namespace sd::dgl {
    struct ResourceValue {
        std::optional<Value> literal;
        std::map<const Expr*, bool> origins;
        std::map<std::string, ResourceValue> fields;
        std::vector<ResourceValue> items;
        bool known_items = false;
    };

    static void merge_origins(std::map<const Expr*, bool>& into, const ResourceValue& value, bool preview = false) {
        for (const auto& [origin, conditional] : value.origins) {
            const bool condition = preview || conditional;
            auto inserted        = into.emplace(origin, condition);
            if (!condition)
                inserted.first->second = false;
        }
        for (const auto& [_, field] : value.fields)
            merge_origins(into, field, preview);
        for (const auto& item : value.items)
            merge_origins(into, item, preview);
    }

    static bool captures_model(Kind kind) {
        return is_model_kind(kind) || kind == Kind::Model || kind == Kind::Lora || kind == Kind::Predictor ||
               kind == Kind::SamplingObserver || kind == Kind::ControlBindings || kind == Kind::IPAdapterBindings || kind == Kind::PuLIDBindings;
    }

    static bool merge_resources(ResourceValue& into, const ResourceValue& from) {
        bool changed = false;
        if (into.literal) {
            into.literal.reset();
            changed = true;
        }
        for (const auto& [origin, preview] : from.origins) {
            auto inserted = into.origins.emplace(origin, preview);
            if (inserted.second)
                changed = true;
            else if (inserted.first->second && !preview) {
                inserted.first->second = false;
                changed                = true;
            }
        }
        for (const auto& [name, field] : from.fields) {
            auto inserted = into.fields.emplace(name, field);
            changed       = inserted.second || changed;
            if (!inserted.second)
                changed = merge_resources(inserted.first->second, field) || changed;
        }
        if (into.known_items && (!from.known_items || into.items.size() != from.items.size())) {
            into.known_items = false;
            changed          = true;
        }
        for (size_t i = 0; i < from.items.size(); ++i) {
            if (i == into.items.size()) {
                into.items.push_back(from.items[i]);
                changed = true;
            } else
                changed = merge_resources(into.items[i], from.items[i]) || changed;
        }
        return changed;
    }

    class ResourceAnalysis {
        Graph& graph_;
        std::set<const Expr*> discovered_;
        std::map<const Expr*, bool> used_;
        size_t iteration_depth_ = 0;
        std::vector<const Expr*> call_path_;
        Graph::Phase pending_{};

        void flush_phase() {
            if (used_.empty())
                return;
            for (const auto& resource : graph_.resources)
                if (auto use = used_.find(resource.expression); use != used_.end())
                    pending_.resources.push_back({use->first, use->second});
            graph_.phases.push_back(pending_);
            pending_.resources.clear();
            used_.clear();
        }

        void boundary(const Expr* expression, bool after = false) {
            if (iteration_depth_)
                return;
            flush_phase();
            pending_ = {expression, expression->name.empty() ? "if" : expression->name, {}, call_path_, after};
        }

        using Scope = std::map<std::string, ResourceValue>;

        ResourceValue body(const std::vector<Statement>& statements, Scope scope, bool phases = false) {
            for (const auto& statement : statements) {
                if (!statement.value)
                    return {};
                if (phases)
                    boundary(statement.value.get());
                auto value = expression(statement.value, scope);
                if (statement.form == Statement::Form::Return || statement.form == Statement::Form::Result)
                    return value;
                if (statement.form == Statement::Form::Binding)
                    scope[statement.name] = std::move(value);
            }
            return {};
        }

        ResourceValue expression(const std::shared_ptr<Expr>& expression, const Scope& scope) {
            const auto& expr = *expression;
            if (expr.form == Expr::Form::Literal)
                return {Value::literal(expr.literal)};
            if (expr.form == Expr::Form::Block) {
                auto result = body(expr.body, scope, true);
                boundary(&expr, true);
                return result;
            }
            if (expr.form == Expr::Form::Conditional) {
                const auto condition = this->expression(expr.arguments[0].second, scope);
                if (!condition.literal || condition.literal->type().kind != Kind::Boolean)
                    throw std::runtime_error("if condition must be a boolean known before resource planning");
                const bool selected = condition.literal->scalar.get<bool>();
                if (expr.arguments.size() == 2) {
                    if (selected)
                        this->expression(expr.arguments[1].second, scope);
                    return {};
                }
                return this->expression(expr.arguments[selected ? 1 : 2].second, scope);
            }
            if (expr.form == Expr::Form::FunctionReference)
                return {};
            if (expr.form == Expr::Form::Reference) {
                auto dot         = expr.name.find('.');
                const auto found = scope.find(expr.name.substr(0, dot));
                if (found == scope.end())
                    return {};
                auto value = found->second;
                while (dot != std::string::npos) {
                    auto next        = expr.name.find('.', dot + 1);
                    const auto field = value.fields.find(expr.name.substr(dot + 1, next == std::string::npos ? next : next - dot - 1));
                    if (field == value.fields.end()) {
                        ResourceValue aggregate;
                        merge_origins(aggregate.origins, value);
                        return aggregate;
                    }
                    auto child = field->second;
                    value      = std::move(child);
                    dot        = next;
                }
                return value;
            }
            if (expr.form == Expr::Form::List || expr.form == Expr::Form::Dict) {
                ResourceValue result;
                result.known_items = expr.form == Expr::Form::List;
                bool constant      = true;
                Values fields;
                std::vector<Value> items;
                for (const auto& [name, item] : expr.arguments) {
                    auto value = this->expression(item, scope);
                    constant   = constant && bool(value.literal);
                    if (expr.form == Expr::Form::List) {
                        if (value.literal)
                            items.push_back(*value.literal);
                        result.items.push_back(std::move(value));
                    } else {
                        if (value.literal)
                            fields.emplace(name, *value.literal);
                        result.fields.emplace(name, std::move(value));
                    }
                }
                if (constant)
                    result.literal = expr.form == Expr::Form::List ? Value::list(std::move(items)) : Value::dict(std::move(fields));
                return result;
            }
            if (expr.form == Expr::Form::Index) {
                auto source        = this->expression(expr.arguments[0].second, scope);
                const auto indices = this->expression(expr.arguments[1].second, scope);
                if (source.known_items && indices.literal && indices.literal->items.size() == 1) {
                    const auto selected = select_list(source.items.size(), indices.literal->items[0].fields);
                    if (!selected.slice)
                        return source.items[selected.begin];
                    if (source.literal)
                        source.literal = index_value(*source.literal, *indices.literal);
                    source.items = std::vector<ResourceValue>(source.items.begin() + selected.begin, source.items.begin() + selected.end);
                    return source;
                }
                ResourceValue result;
                for (const auto& item : source.items)
                    merge_resources(result, item);
                merge_origins(result.origins, source);
                return result;
            }
            const auto& call = graph_.calls.at(&expr);
            if (call.iteration) {
                const auto& iteration = *call.iteration;
                auto collection       = this->expression(iteration.items, scope);
                Scope locals;
                const size_t item_parameter = iteration.initial ? 1 : 0;
                for (const auto& arg : iteration.arguments)
                    if (arg.parameter != item_parameter)
                        locals.emplace(iteration.function->parameters[arg.parameter], this->expression(arg.expression, scope));
                if (collection.literal && collection.items.empty())
                    return iteration.initial ? locals.at(iteration.function->parameters[0]) : ResourceValue{};
                ResourceValue item;
                for (const auto& candidate : collection.items) {
                    merge_origins(item.origins, candidate);
                    for (const auto& [field, child] : candidate.fields)
                        merge_origins(item.fields[field].origins, child);
                }
                merge_origins(item.origins, collection);
                locals[iteration.function->parameters[item_parameter]] = std::move(item);
                ++iteration_depth_;
                if (iteration.initial) {
                    auto accumulator = locals.at(iteration.function->parameters[0]);
                    for (;;) {
                        locals[iteration.function->parameters[0]] = accumulator;
                        auto output                               = body(iteration.function->body, locals);
                        if (!merge_resources(accumulator, output))
                            break;
                    }
                    --iteration_depth_;
                    return accumulator;
                }
                auto output = body(iteration.function->body, std::move(locals));
                --iteration_depth_;
                ResourceValue result;
                result.items.push_back(std::move(output));
                return result;
            }
            if (call.function) {
                Scope locals;
                for (const auto& arg : call.arguments)
                    locals.emplace(call.function->parameters[arg.parameter], this->expression(arg.expression, scope));
                call_path_.push_back(&expr);
                auto result = body(call.function->body, std::move(locals));
                call_path_.pop_back();
                return result;
            }
            const auto& [operation_name, operation] = *call.operation;
            Scope args;
            for (const auto& arg : call.arguments)
                if (!operation.arguments[arg.parameter].callback)
                    args.emplace(operation.arguments[arg.parameter].name, this->expression(arg.expression, scope));
            for (const auto& argument : operation.arguments)
                if (argument.default_value && !args.count(argument.name))
                    args[argument.name].literal = *argument.default_value;
            if (operation.resource()) {
                if (discovered_.insert(&expr).second) {
                    Values arguments;
                    for (const auto& [name, value] : args) {
                        if (!value.literal)
                            throw std::runtime_error("cannot resolve resource argument '" + name + "'");
                        arguments.emplace(name, *value.literal);
                    }
                    graph_.resources.push_back({&expr, call.operation, std::move(arguments), iteration_depth_ != 0});
                }
                ResourceValue value;
                value.origins.emplace(&expr, false);
                return value;
            }
            if (operation_name == "list::replace") {
                auto result       = args.at("items");
                const auto& value = args.at("value");
                const auto& index = args.at("index");
                result.literal.reset();
                if (result.known_items && index.literal) {
                    const auto selected          = select_list(result.items.size(), {{"index", *index.literal}});
                    result.items[selected.begin] = value;
                } else if (result.known_items) {
                    for (auto& item : result.items)
                        merge_resources(item, value);
                } else
                    merge_resources(result, value);
                return result;
            }
            ResourceValue inputs;
            for (const auto& [_, value] : args)
                merge_origins(inputs.origins, value, operation.model_use == Operation::ModelUse::Preview);
            if (operation.model_use == Operation::ModelUse::Capture || operation.model_use == Operation::ModelUse::Preview || captures_model(operation.output.type.kind))
                return inputs;
            if (operation.model_use != Operation::ModelUse::Metadata)
                merge_origins(used_, inputs);
            if (!operation.output.same_type_as.empty())
                return args.at(operation.output.same_type_as);
            return {};
        }

    public:
        explicit ResourceAnalysis(Graph& graph)
            : graph_(graph) {}
        void run() {
            body(graph_.syntax.entry.body, {}, true);
            flush_phase();
        }
    };

    void analyze_resources(Graph& graph) {
        ResourceAnalysis(graph).run();
    }
}
