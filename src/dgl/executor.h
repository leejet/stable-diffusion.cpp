#ifndef __SD_DGL_EXECUTOR_H__
#define __SD_DGL_EXECUTOR_H__

#include "registry.h"

namespace sd::dgl {

    struct Graph;
    struct Context;
    class RunEvents;
    class RunPlan;
    struct RunState {
        Context& context;
        RunEvents& events;
        RunPlan* plan = nullptr;
        std::string metadata;
    };

    Value run(const Graph& graph, RunState& state, Values inputs);

}  // namespace sd::dgl

#endif  // __SD_DGL_EXECUTOR_H__
