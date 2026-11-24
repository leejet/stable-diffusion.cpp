#include "syntax.h"

#include <cctype>
#include <set>
#include <stdexcept>

#include "lexer.h"

namespace sd::dgl {

    class Parser {
        Lexer lexer_;
        Token token_;
        size_t expression_count_ = 0;

        void next() { token_ = lexer_.next(); }

        [[noreturn]] void fail(const std::string& message) const {
            throw std::runtime_error("line " + std::to_string(token_.line) + ", column " + std::to_string(token_.column) + ": " + message);
        }

        bool take(const std::string& text) {
            if (token_.text != text)
                return false;
            next();
            return true;
        }

        void expect(const std::string& text) {
            if (!take(text))
                fail("expected '" + text + "', got '" + token_.text + "'");
        }

        std::string identifier() {
            if (token_.text.empty() || (!std::isalpha(static_cast<unsigned char>(token_.text[0])) && token_.text[0] != '_'))
                fail("expected identifier");
            std::string result = token_.text;
            if (result == "return" || result == "fn" || result == "true" || result == "false" || result == "if" || result == "else")
                fail("reserved identifier '" + result + "'");
            next();
            return result;
        }

        nlohmann::json scalar_literal(const std::string& text) {
            try {
                auto parsed = nlohmann::json::parse(text);
                if ((text[0] == '-' || std::isdigit(static_cast<unsigned char>(text[0]))) && text.find_first_of(".eE") == std::string::npos && !parsed.is_number_integer())
                    fail("integer exceeds int64 range");
                if (parsed.is_number_unsigned() && parsed.get<uint64_t>() > uint64_t(INT64_MAX))
                    fail("integer exceeds int64 range");
                return parsed;
            } catch (const std::exception&) {
                fail("invalid scalar literal");
            }
        }

        static int precedence(const std::string& op) {
            if (op == "+" || op == "-")
                return 1;
            if (op == "*" || op == "/")
                return 2;
            return -1;
        }

        static std::shared_ptr<Expr> arithmetic(const Token& op, std::shared_ptr<Expr> left, std::shared_ptr<Expr> right) {
            auto result       = std::make_shared<Expr>();
            result->form      = Expr::Form::Call;
            result->line      = op.line;
            result->column    = op.column;
            result->name      = op.text == "+" ? "math::add" : op.text == "-" ? "math::sub"
                                                           : op.text == "*"   ? "math::mul"
                                                                              : "math::div";
            result->arguments = {{"", std::move(left)}, {"", std::move(right)}};
            return result;
        }

        std::shared_ptr<Expr> index_suffix(std::shared_ptr<Expr> expr, int& depth, bool literal_only) {
            while (token_.text == "[") {
                if (literal_only)
                    fail("var values must be literals");
                if (++depth > 64 || (expression_count_ += 2) > 1000000)
                    fail("DGL expression limit exceeded");
                auto call    = std::make_shared<Expr>();
                call->form   = Expr::Form::Index;
                call->name   = "index";
                call->line   = token_.line;
                call->column = token_.column;
                auto axes    = std::make_shared<Expr>();
                axes->form   = Expr::Form::List;
                axes->line   = token_.line;
                axes->column = token_.column;
                next();
                if (token_.text == "]")
                    fail("indexing requires at least one axis");
                do {
                    if (++expression_count_ > 1000000)
                        fail("DGL expression limit exceeded");
                    auto axis    = std::make_shared<Expr>();
                    axis->form   = Expr::Form::Dict;
                    axis->line   = token_.line;
                    axis->column = token_.column;
                    if (token_.text == "::")
                        fail("indexing steps are not supported");
                    std::shared_ptr<Expr> first;
                    if (token_.text != ":")
                        first = expression(depth + 1);
                    if (token_.text == "::")
                        fail("indexing steps are not supported");
                    if (take(":")) {
                        if (first)
                            axis->arguments.push_back({"start", std::move(first)});
                        if (token_.text != "," && token_.text != "]")
                            axis->arguments.push_back({"end", expression(depth + 1)});
                        if (token_.text == ":" || token_.text == "::")
                            fail("indexing steps are not supported");
                    } else {
                        axis->arguments.push_back({"index", std::move(first)});
                    }
                    axes->arguments.push_back({"", std::move(axis)});
                } while (take(",") && token_.text != "]");
                expect("]");
                call->arguments = {{"input", std::move(expr)}, {"indices", std::move(axes)}};
                expr            = std::move(call);
            }
            return expr;
        }

        std::shared_ptr<Expr> block(int depth) {
            auto expr    = std::make_shared<Expr>();
            expr->form   = Expr::Form::Block;
            expr->line   = token_.line;
            expr->column = token_.column;
            expect("{");
            std::set<std::string> names;
            while (!take("}")) {
                if (token_.text.empty())
                    fail("unterminated if block");
                if (token_.text == "return")
                    fail("use a final expression for an if block result; return is only allowed in functions");
                auto value = expression(depth + 1);
                if (take("=")) {
                    expr->body.push_back(binding(*value, names, depth + 1));
                    take(";");
                } else {
                    take(";");
                    if (token_.text.empty())
                        fail("unterminated if block");
                    if (token_.text == "}" && !(value->form == Expr::Form::Conditional && value->arguments.size() == 2))
                        expr->body.push_back({{}, std::move(value), Statement::Form::Result});
                    else {
                        if (value->form != Expr::Form::Call && value->form != Expr::Form::Index && value->form != Expr::Form::Conditional)
                            fail("only function calls and conditional expressions can be standalone statements");
                        expr->body.push_back({{}, std::move(value), Statement::Form::Call});
                    }
                }
            }
            return expr;
        }

        std::shared_ptr<Expr> expression(int depth = 0, bool literal_only = false, int minimum_precedence = 0) {
            if (depth > 64 || ++expression_count_ > 1000000)
                fail("DGL expression limit exceeded");
            auto expr    = std::make_shared<Expr>();
            expr->line   = token_.line;
            expr->column = token_.column;
            if (token_.text == "$")
                fail("write input values in the DGL source before compiling");
            if (take("if")) {
                if (literal_only)
                    fail("var values must be literals");
                expr->form      = Expr::Form::Conditional;
                expr->name      = "if";
                auto condition  = expression(depth + 1);
                auto selected   = block(depth);
                expr->arguments = {{"condition", std::move(condition)}, {"then", std::move(selected)}};
                if (take("else"))
                    expr->arguments.push_back({"else", block(depth)});
            } else if (token_.text == "+" || token_.text == "-") {
                const auto sign = token_;
                next();
                if (!token_.text.empty() && std::isdigit(static_cast<unsigned char>(token_.text[0]))) {
                    expr->literal = scalar_literal((sign.text == "-" ? "-" : "") + token_.text);
                    next();
                } else {
                    if (literal_only)
                        fail("var values must be literals");
                    expr->literal = 0;
                    auto operand  = expression(depth + 1, false, 3);
                    expr          = arithmetic(sign, std::move(expr), std::move(operand));
                    if (++expression_count_ > 1000000)
                        fail("DGL expression limit exceeded");
                }
            } else if (take("(")) {
                expr = expression(depth + 1, literal_only);
                expect(")");
            } else if (take("{")) {
                expr->form = Expr::Form::Dict;
                std::set<std::string> names;
                if (!take("}")) {
                    do {
                        auto name = identifier();
                        if (!names.insert(name).second)
                            fail("duplicate dict field");
                        expect("=");
                        expr->arguments.push_back({name, expression(depth + 1, literal_only)});
                    } while (take(",") && token_.text != "}");
                    expect("}");
                }
            } else if (take("[")) {
                expr->form = Expr::Form::List;
                if (!take("]")) {
                    do {
                        expr->arguments.push_back({"", expression(depth + 1, literal_only)});
                    } while (take(",") && token_.text != "]");
                    expect("]");
                }
            } else if (!token_.text.empty() && (token_.text[0] == '"' || std::isdigit(static_cast<unsigned char>(token_.text[0])) || token_.text == "true" || token_.text == "false")) {
                expr->literal = scalar_literal(token_.text);
                next();
            } else {
                if (literal_only)
                    fail("var values must be literals");
                expr->name     = identifier();
                expr->form     = Expr::Form::Reference;
                bool qualified = false;
                while (take("::")) {
                    expr->name += "::" + identifier();
                    qualified = true;
                }
                if (take("(")) {
                    expr->form = Expr::Form::Call;
                    bool named = false;
                    if (!take(")")) {
                        do {
                            auto arg = expression(depth + 1);
                            std::string name;
                            if (take("=")) {
                                if (arg->form != Expr::Form::Reference || arg->name.find('.') != std::string::npos)
                                    fail("expected argument name");
                                name  = arg->name;
                                arg   = expression(depth + 1);
                                named = true;
                            } else if (named)
                                fail("positional arguments must precede named arguments");
                            expr->arguments.push_back({name, arg});
                        } while (take(",") && token_.text != ")");
                        expect(")");
                    }
                } else if (qualified) {
                    fail("expected '(' after qualified function name");
                }
                while (take(".")) {
                    if (expr->form != Expr::Form::Reference)
                        fail("assign a call result before accessing its fields");
                    expr->name += "." + identifier();
                }
            }
            expr = index_suffix(std::move(expr), depth, literal_only);
            while (precedence(token_.text) >= minimum_precedence) {
                if (literal_only)
                    fail("var values must be literals");
                if (++depth > 64 || ++expression_count_ > 1000000)
                    fail("DGL expression limit exceeded");
                const auto op       = token_;
                const auto priority = precedence(op.text);
                next();
                auto right = expression(depth + 1, false, priority + 1);
                expr       = arithmetic(op, std::move(expr), std::move(right));
            }
            return expr;
        }

        Statement binding(const Expr& target, std::set<std::string>& names, int depth = 0) {
            if (target.form != Expr::Form::Reference || target.name.find('.') != std::string::npos)
                fail("expected a binding name before '='");
            if (!names.insert(target.name).second)
                fail("duplicate binding '" + target.name + "'");
            return {target.name, expression(depth)};
        }

        Statement binding(std::set<std::string>& names, bool literal_only = false) {
            auto name = identifier();
            if (!names.insert(name).second)
                fail("duplicate binding '" + name + "'");
            expect("=");
            return {std::move(name), expression(0, literal_only)};
        }

        std::vector<Statement> body(std::vector<Statement> statements = {}, std::set<std::string> names = {}) {
            expect("{");
            bool returned = false;
            while (!take("}")) {
                if (token_.text.empty())
                    fail("unterminated function body");
                if (returned)
                    fail("return must be the last statement");
                if (take("return")) {
                    returned = true;
                    statements.push_back({{}, token_.text == ";" || token_.text == "}" ? nullptr : expression(), Statement::Form::Return});
                } else {
                    auto value = expression();
                    if (take("=")) {
                        statements.push_back(binding(*value, names));
                    } else {
                        if (value->form != Expr::Form::Call && value->form != Expr::Form::Index && value->form != Expr::Form::Conditional)
                            fail("only function calls and conditional expressions can be standalone statements");
                        statements.push_back({{}, std::move(value), Statement::Form::Call});
                    }
                }
                take(";");
            }
            return statements;
        }

    public:
        explicit Parser(const std::string& source)
            : lexer_(source) { next(); }

        Syntax parse(bool host_inputs) {
            Syntax program;
            std::vector<Statement> variables;
            std::set<std::string> variable_names;
            if (take("var")) {
                expect("{");
                while (!take("}")) {
                    if (token_.text.empty())
                        fail("unterminated var block");
                    variables.push_back(binding(variable_names, true));
                    take(";");
                }
            }
            while (take("fn")) {
                std::string name = identifier();
                Function function;
                expect("(");
                std::set<std::string> names;
                if (!take(")")) {
                    if (name == "main" && !host_inputs)
                        fail("fn main must have no parameters");
                    do {
                        auto parameter = identifier();
                        if (!names.insert(parameter).second)
                            fail("duplicate parameter");
                        function.parameters.push_back(parameter);
                    } while (take(","));
                    expect(")");
                }
                function.body = name == "main" ? body(std::move(variables), std::move(variable_names)) : body();
                if (!program.functions.emplace(name, std::move(function)).second)
                    fail("duplicate function '" + name + "'");
            }
            if (token_.text == "var")
                fail("var block may appear once, before the functions");
            if (!token_.text.empty())
                fail("expected a function declaration");
            auto entry = program.functions.extract("main");
            if (entry.empty())
                fail("missing entry point 'fn main'");
            program.entry = std::move(entry.mapped());
            return program;
        }
    };

    Syntax parse_syntax(const std::string& source, bool host_inputs) {
        if (source.size() > 512 * 1024 * 1024)
            throw std::runtime_error("DGL source exceeds 512 MiB");
        return Parser(source).parse(host_inputs);
    }

}  // namespace sd::dgl
