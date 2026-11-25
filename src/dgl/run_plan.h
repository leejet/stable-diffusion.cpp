#ifndef __SD_DGL_RUN_PLAN_H__
#define __SD_DGL_RUN_PLAN_H__

#include "graph.h"
#include "model_manager.h"

namespace sd::dgl {
    struct RunState;

    class RunPlan {
        struct ModelEntry {
            std::string key, name;
            Kind kind;
            ModelManager::ResourceMemory memory;
            size_t reserve = 0;
            bool automatic = false;
            bool placed    = false;
        };
        const Graph& graph_;
        RunState& state_;
        bool enabled_;
        bool vae_preview_;
        size_t current_ = 0;
        std::map<const Expr*, Value> prepared_;
        std::map<const Expr*, size_t> origins_;
        std::vector<ModelEntry> models_;
        std::map<std::string, size_t> identities_;
        std::map<ggml_backend_t, size_t> budgets_;
        size_t ram_budget_  = 0;
        size_t host_budget_ = 0;
        std::string default_backend_;

        size_t add_model(const Graph::ResourceCall& call, const Value& value);
        bool active(const Graph::ResourceUse& use) const;
        void place_models();
        void place(size_t index, models::Model& model);
        size_t peak_bytes(ggml_backend_t backend, size_t candidate, bool workspace = true) const;
        void update_priorities();

    public:
        RunPlan(const Graph& graph, RunState& state);
        ~RunPlan();
        Value load_resource(const Registry::Entry& operation, Values& args);
        void enter(const Expr* statement, const std::vector<const Expr*>& call_path = {}, bool after = false);
        void resource_loaded(const Expr* expression, const Value& value);
    };
}

#endif
