#include "dgl/bindings/seedvr2.h"

#include "dgl/bindings/arguments.h"
#include "dgl/bindings/values.h"
#include "dgl/models/diffusion_model.h"
#include "dgl/models/dit.h"
#include "dgl/ops/seedvr2.h"

namespace sd::dgl {
    Value seedvr2_positive_embedding(Values& args, RunState&) {
        auto& dit = *object<models::DiffusionModel>(args, "dit")->dit;
        Model::ExecutionScope scope(dit);
        scope.require_ready();
        return tensor_value(ops::seedvr2_positive_embedding(dit));
    }
}
