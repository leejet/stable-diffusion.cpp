#ifndef __SD_DGL_INDEXING_H__
#define __SD_DGL_INDEXING_H__

#include "value.h"

namespace sd::dgl {
    struct ListSelection {
        size_t begin, end;
        bool slice;
    };

    ListSelection select_list(size_t count, const Values& axis);
    Value index_value(Value input, const Value& indices);
}

#endif  // __SD_DGL_INDEXING_H__
