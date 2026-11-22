#include "executor.h"
#include "context.h"

#include <stdexcept>

#include "core/util.h"
#include "diagnostic.h"
#include "graph.h"
#include "indexing.h"
#include "run_plan.h"

namespace sd::dgl {

    class Executor {
        struct Frame {
            Values values;
            ReferenceCounts remaining;
            std::vector<const Expr*> call_path;

            bool uses(const std::string& name) const {
                if (remaining.count(name))
                    return true;
                auto prefix = name + ".";
                auto field  = remaining.lower_bound(prefix);
                return field != remaining.end() && field->first.compare(0, prefix.size(), prefix) == 0;
            }

            void drop_unused_fields(const std::string& name, Value& value) const {
                if (value.fields.empty() || remaining.count(name))
                    return;
                for (auto field = value.fields.begin(); field != value.fields.end();) {
                    auto path = name + "." + field->first;
                    if (uses(path)) {
                        drop_unused_fields(path, field->second);
                        ++field;
                    } else {
                        field = value.fields.erase(field);
                    }
                }
            }

            Value take(const std::string& reference, const std::string& root) {
                if (--remaining.at(reference) == 0)
                    remaining.erase(reference);
                auto& value = values.at(root);
                if (!uses(root)) {
                    auto result = std::move(value);
                    values.erase(root);
                    return result;
                }
                auto result = value;
                drop_unused_fields(root, value);
                return result;
            }

            void discard_captures(const ReferenceCounts& captures) {
                for (const auto& [reference, count] : captures) {
                    if ((remaining.at(reference) -= count) == 0)
                        remaining.erase(reference);
                    auto root  = reference.substr(0, reference.find('.'));
                    auto value = values.find(root);
                    if (value != values.end()) {
                        if (uses(root))
                            drop_unused_fields(root, value->second);
                        else
                            values.erase(value);
                    }
                }
            }
        };

        const Graph& graph_;
        RunState& state_;
        size_t visits_ = 0;

    public:
        Executor(const Graph& graph, RunState& state)
            : graph_(graph), state_(state) {}

        Value expression(const std::shared_ptr<Expr>& expr, Frame& frame) {
            state_.events.check_cancelled();
            if (++visits_ > 100000)
                throw std::runtime_error("DGL execution exceeded the expression limit");
            auto source_message = [&](const char* message) {
                return "line " + std::to_string(expr->line) + ", column " + std::to_string(expr->column) + ": " + message;
            };
            auto failure = SD_STATUS_GENERATION_FAILED;
            try {
                if (expr->form == Expr::Form::Literal)
                    return Value::literal(expr->literal);
                if (expr->form == Expr::Form::Block) {
                    const auto& block = graph_.blocks.at(expr.get());
                    Values captures;
                    for (const auto& [reference, count] : block.captures) {
                        auto root = reference.substr(0, reference.find('.'));
                        if (!captures.count(root))
                            captures.emplace(root, frame.values.at(root));
                    }
                    frame.discard_captures(block.captures);
                    auto result = body(expr->body, {std::move(captures), block.references, frame.call_path}, true);
                    if (state_.plan)
                        state_.plan->enter(expr.get(), frame.call_path, true);
                    return result;
                }
                if (expr->form == Expr::Form::Conditional) {
                    const auto condition = expression(expr->arguments[0].second, frame);
                    const bool selected  = condition.scalar.get<bool>();
                    if (expr->arguments.size() == 2) {
                        if (selected)
                            expression(expr->arguments[1].second, frame);
                        else
                            frame.discard_captures(graph_.blocks.at(expr->arguments[1].second.get()).captures);
                        return Value::void_value();
                    }
                    frame.discard_captures(graph_.blocks.at(expr->arguments[selected ? 2 : 1].second.get()).captures);
                    return expression(expr->arguments[selected ? 1 : 2].second, frame);
                }
                if (expr->form == Expr::Form::Reference) {
                    auto dot   = expr->name.find('.');
                    auto value = frame.take(expr->name, expr->name.substr(0, dot));
                    while (dot != std::string::npos) {
                        auto next  = expr->name.find('.', dot + 1);
                        auto field = expr->name.substr(dot + 1, next == std::string::npos ? next : next - dot - 1);
                        auto found = value.fields.find(field);
                        if (found == value.fields.end())
                            throw Error(SD_STATUS_INVALID_ARGUMENT, "field '" + expr->name + "' is absent from the runtime value");
                        auto child = std::move(found->second);
                        value      = std::move(child);
                        dot        = next;
                    }
                    return value;
                }
                if (expr->form == Expr::Form::List) {
                    std::vector<Value> items;
                    for (const auto& arg : expr->arguments)
                        items.push_back(expression(arg.second, frame));
                    return Value::list(std::move(items));
                }
                if (expr->form == Expr::Form::Dict) {
                    Values fields;
                    for (const auto& field : expr->arguments)
                        fields[field.first] = expression(field.second, frame);
                    return Value::dict(std::move(fields));
                }
                if (expr->form == Expr::Form::Index) {
                    auto input   = expression(expr->arguments[0].second, frame);
                    auto indices = expression(expr->arguments[1].second, frame);
                    auto result  = index_value(std::move(input), indices);
                    state_.events.check_cancelled();
                    return result;
                }
                const auto& bound = graph_.calls.at(expr.get());
                if (bound.iteration) {
                    const auto& call        = *bound.iteration;
                    auto list               = expression(call.items, frame);
                    const auto& parameters  = call.function->parameters;
                    const size_t item_index = call.initial ? 1 : 0;
                    const auto& item_param  = parameters[item_index];
                    Values locals;
                    for (const auto& arg : call.arguments)
                        if (arg.parameter != item_index)
                            locals.emplace(parameters[arg.parameter], expression(arg.expression, frame));
                    auto call_path = frame.call_path;
                    call_path.push_back(expr.get());
                    if (list.items.size() > 1)
                        LOG_INFO("%s %s: %zu items", expr->name.c_str(), call.process.c_str(), list.items.size());
                    auto run_item = [&](size_t i) {
                        LOG_VERBOSE("%s %s: item %zu/%zu", expr->name.c_str(), call.process.c_str(), i + 1, list.items.size());
                        locals[item_param] = std::move(list.items[i]);
                        return body(call.process, *call.function,
                                    i + 1 == list.items.size() ? std::move(locals) : locals, call_path);
                    };
                    if (call.initial) {
                        const auto& accumulator = parameters[0];
                        for (size_t i = 0; i < list.items.size(); ++i)
                            locals[accumulator] = run_item(i);
                        return std::move(locals.at(accumulator));
                    }
                    std::vector<Value> results;
                    results.reserve(list.items.size());
                    for (size_t i = 0; i < list.items.size(); ++i)
                        results.push_back(run_item(i));
                    const auto output = std::find_if(graph_.map_outputs.begin(), graph_.map_outputs.end(), [&](const auto& site) {
                        return site.expression == expr.get() && site.call_path == frame.call_path;
                    });
                    if (output == graph_.map_outputs.end())
                        throw std::runtime_error("missing map output type for this call");
                    return Value::list(std::move(results), output->element);
                }
                if (bound.function) {
                    const auto& parameters = bound.function->parameters;
                    Values locals;
                    for (const auto& arg : bound.arguments)
                        locals.emplace(parameters[arg.parameter], expression(arg.expression, frame));
                    auto call_path = frame.call_path;
                    call_path.push_back(expr.get());
                    return body(expr->name, *bound.function, std::move(locals), std::move(call_path));
                }
                const auto& operation = bound.operation->second;
                Values args;
                auto set_argument = [&](const Argument& argument, Value value) {
                    if (!accepts(argument.type, value.type()))
                        throw Error(SD_STATUS_TYPE_MISMATCH, "invalid type for '" + argument.name + "'");
                    if (argument.validate)
                        argument.validate(value);
                    args.emplace(argument.name, std::move(value));
                };
                for (const auto& arg : bound.arguments) {
                    const auto& argument = operation.arguments[arg.parameter];
                    if (argument.callback) {
                        const auto* fn     = graph_.callbacks.at(arg.expression.get());
                        auto callback_path = frame.call_path;
                        callback_path.push_back(expr.get());
                        auto callback = std::make_shared<ScriptFunction>([this, fn, name = arg.expression->name, path = std::move(callback_path)](Value input, const Value& captures) {
                            return Executor(graph_, state_).body(name, *fn, {{fn->parameters[0], std::move(input)}, {fn->parameters[1], captures}}, path);
                        });
                        set_argument(argument, Value::object(std::move(callback)));
                    } else {
                        set_argument(argument, expression(arg.expression, frame));
                    }
                }
                for (const auto& argument : operation.arguments)
                    if (argument.default_value && !args.count(argument.name))
                        set_argument(argument, *argument.default_value);
                state_.context.models->clear_host_memory_error();
                if (operation.resource())
                    failure = SD_STATUS_LOAD_FAILED;
                state_.events.check_cancelled();
                Value value = operation.resource() ? (state_.plan ? state_.plan->load_resource(*bound.operation, args) : state_.context.load(*bound.operation, args, true))
                                                   : operation.execute(args, state_);
                if (operation.resource()) {
                    if (state_.plan)
                        state_.plan->resource_loaded(expr.get(), value);
                    state_.context.finish_resource_loading(value);
                }
                state_.events.check_cancelled();
                auto output_type = operation.output.same_type_as.empty()
                                       ? operation.output.type
                                       : args.at(operation.output.same_type_as).type();
                if (!accepts(output_type, value.type()))
                    throw std::runtime_error("built-in function '" + expr->name + "' returned the wrong type");
                return value;
            } catch (const std::bad_alloc&) {
                throw;
            } catch (const Error& error) {
                if (error.status == SD_STATUS_CANCELLED)
                    throw;
                throw Error(error.status, source_message(error.what()));
            } catch (const std::exception& error) {
                throw Error(failure, source_message(error.what()));
            }
        }

        Value body(const std::vector<Statement>& statements, Frame frame, bool entry = false) {
            for (auto value = frame.values.begin(); value != frame.values.end();) {
                if (frame.uses(value->first)) {
                    frame.drop_unused_fields(value->first, value->second);
                    ++value;
                } else {
                    value = frame.values.erase(value);
                }
            }
            for (const auto& statement : statements) {
                if (!statement.value)
                    return Value::void_value();
                if (entry && state_.plan)
                    state_.plan->enter(statement.value.get(), frame.call_path);
                auto value = expression(statement.value, frame);
                if (statement.form == Statement::Form::Return || statement.form == Statement::Form::Result)
                    return value;
                if (statement.form == Statement::Form::Binding && frame.uses(statement.name)) {
                    frame.drop_unused_fields(statement.name, value);
                    frame.values.emplace(statement.name, std::move(value));
                }
            }
            return Value::void_value();
        }

        Value body(const std::string& name, const Function& function, Values scope, std::vector<const Expr*> call_path = {}) {
            return body(function.body, {std::move(scope), graph_.references.at(name), std::move(call_path)}, name == "main");
        }
    };

    Value run(const Graph& graph, RunState& state, Values inputs) {
        state.metadata = state.context.embed_metadata ? graph.source : "";
        state.context.prune_expired_resources();
        sd::ParallelScope tensor_scope(state.context.tensor_executor());
        RunPlan plan(graph, state);
        state.plan = &plan;
        try {
            auto result = Executor(graph, state).body("main", graph.syntax.entry, std::move(inputs));
            state.plan  = nullptr;
            return result;
        } catch (...) {
            state.plan = nullptr;
            throw;
        }
    }

}  // namespace sd::dgl
