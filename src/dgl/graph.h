#ifndef __SD_DGL_GRAPH_H__
#define __SD_DGL_GRAPH_H__

#include "registry.h"
#include "syntax/syntax.h"

namespace sd::dgl {

    using ReferenceCounts = std::map<std::string, size_t>;

    struct BlockReferences {
        ReferenceCounts references;
        ReferenceCounts captures;
    };

    struct BoundArgument {
        size_t parameter;
        std::shared_ptr<Expr> expression;
    };

    struct IterationCall {
        const Function* function;
        std::string process;
        std::shared_ptr<Expr> items, initial;
        std::vector<BoundArgument> arguments;
    };

    struct BoundCall {
        const Registry::Entry* operation = nullptr;
        const Function* function         = nullptr;
        std::vector<BoundArgument> arguments;
        std::optional<IterationCall> iteration;
    };

    struct Graph {
        struct ResourceCall {
            const Expr* expression;
            const Registry::Entry* operation;
            Values arguments;
            bool deferred = false;
        };
        struct ResourceUse {
            const Expr* resource;
            bool preview = false;
        };
        struct Phase {
            const Expr* statement;
            std::string name;
            std::vector<ResourceUse> resources;
            std::vector<const Expr*> call_path;
            bool after = false;
        };
        struct MapOutput {
            const Expr* expression;
            std::vector<const Expr*> call_path;
            Kind element;
        };
        std::vector<MapOutput> map_outputs;
        std::string source;
        Syntax syntax;
        Registry::Operations operations;
        std::map<const Expr*, BoundCall> calls;
        std::map<const Expr*, const Function*> callbacks;
        std::map<std::string, ReferenceCounts> references;
        std::map<const Expr*, BlockReferences> blocks;
        std::vector<ResourceCall> resources;
        std::vector<Phase> phases;

        Graph()                        = default;
        Graph(Graph&&)                 = default;
        Graph& operator=(Graph&&)      = default;
        Graph(const Graph&)            = delete;
        Graph& operator=(const Graph&) = delete;
    };

    std::vector<BoundArgument> bind_arguments(const Expr& expr, const std::vector<std::string>& names, bool all_required = false);
    std::vector<std::string> argument_names(const Operation& operation);
    IterationCall bind_iteration(const Expr& expr, const Syntax& syntax);
    Graph compile(const std::string& source, const Registry& registry, bool host_inputs = false);
    void analyze_resources(Graph& graph);

}  // namespace sd::dgl

#endif  // __SD_DGL_GRAPH_H__
