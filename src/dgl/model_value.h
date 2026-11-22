#ifndef __SD_DGL_MODEL_VALUE_H__
#define __SD_DGL_MODEL_VALUE_H__

#include "value.h"

namespace sd::dgl {
    struct Context;

    models::Model& model_from_value(const Value& value);
    void set_conditioning_cache_size(const Value& value, size_t capacity);
    Value build_model_variant(Context& context, const Value& base, const std::shared_ptr<const models::LoraConfig>& loras);
    Value bind_model_variant(const Value& base, Value variant);
    Value with_animatediff(const Value& base, const std::shared_ptr<models::AnimateDiff>& motion);
}

#endif  // __SD_DGL_MODEL_VALUE_H__
