#ifndef __SD_DGL_SYNTAX_SYNTAX_H__
#define __SD_DGL_SYNTAX_SYNTAX_H__

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "json.hpp"

namespace sd::dgl {

    struct Expr;

    struct Statement {
        enum class Form { Binding,
                          Call,
                          Return,
                          Result };
        std::string name;
        std::shared_ptr<Expr> value;
        Form form = Form::Binding;
    };

    struct Expr {
        enum class Form { Literal,
                          Reference,
                          FunctionReference,
                          Call,
                          Index,
                          Conditional,
                          List,
                          Dict,
                          Block };
        Form form = Form::Literal;
        nlohmann::json literal;
        std::string name;
        std::vector<std::pair<std::string, std::shared_ptr<Expr>>> arguments;
        std::vector<Statement> body;
        int line   = 1;
        int column = 1;
    };

    struct Function {
        std::vector<std::string> parameters;
        std::vector<Statement> body;
    };

    struct Syntax {
        std::map<std::string, Function> functions;
        Function entry;
    };

    Syntax parse_syntax(const std::string& source, bool host_inputs);

}  // namespace sd::dgl

#endif  // __SD_DGL_SYNTAX_SYNTAX_H__
