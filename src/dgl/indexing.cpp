#include "indexing.h"

#include <algorithm>
#include <type_traits>

#include "dgl/ops/tensor.h"

namespace sd::dgl {
    ListSelection select_list(size_t count, const Values& axis) {
        for (const auto& [name, value] : axis) {
            if (name != "index" && name != "start" && name != "end")
                throw std::runtime_error("unknown list index field '" + name + "'");
            if (value.type().kind != Kind::Integer)
                throw Error(SD_STATUS_TYPE_MISMATCH, "list index field '" + name + "' must be an integer");
        }
        const auto length = int64_t(count);
        if (axis.count("index")) {
            if (axis.size() != 1)
                throw std::runtime_error("a list index cannot also have slice endpoints");
            auto index = get_int(axis, "index");
            if (index < 0)
                index += length;
            if (index < 0 || index >= length)
                throw std::runtime_error("list index out of range");
            return {size_t(index), size_t(index + 1), false};
        }
        auto endpoint = [&](const char* name, int64_t fallback) {
            const auto found = axis.find(name);
            if (found == axis.end())
                return fallback;
            auto value = found->second.scalar.get<int64_t>();
            if (value < 0)
                value += length;
            return std::clamp(value, int64_t(0), length);
        };
        const auto begin = endpoint("start", 0);
        return {size_t(begin), size_t(std::max(begin, endpoint("end", length))), true};
    }

    Value index_value(Value input, const Value& descriptors) {
        if (input.type().kind == Kind::List) {
            if (descriptors.type().kind != Kind::List || descriptors.items.size() != 1 || descriptors.items[0].type().kind != Kind::Dict)
                throw std::runtime_error("list indexing requires exactly one integer index or slice");
            const auto selected = select_list(input.items.size(), descriptors.items[0].fields);
            if (!selected.slice)
                return std::move(input.items[selected.begin]);
            std::vector<Value> items;
            for (size_t i = selected.begin; i < selected.end; ++i)
                items.push_back(std::move(input.items[i]));
            return Value::list(std::move(items), input.type().element);
        }
        if (input.type().kind != Kind::Tensor)
            throw Error(SD_STATUS_TYPE_MISMATCH, "indexing requires a list or tensor");
        std::vector<ops::TensorIndex> indices;
        for (const auto& axis : descriptors.items) {
            if (axis.type().kind != Kind::Dict)
                throw std::runtime_error("tensor indices must be dicts with index or start/end integers");
            for (const auto& [name, value] : axis.fields) {
                if (name != "index" && name != "start" && name != "end")
                    throw std::runtime_error("unknown tensor index field '" + name + "'");
                if (value.type().kind != Kind::Integer)
                    throw Error(SD_STATUS_TYPE_MISMATCH, "tensor index field '" + name + "' must be an integer");
            }
            if (axis.fields.count("index")) {
                if (axis.fields.size() != 1)
                    throw std::runtime_error("a tensor index cannot also have slice endpoints");
                indices.emplace_back(get_int(axis.fields, "index"));
            } else {
                ops::TensorSlice slice;
                if (axis.fields.count("start"))
                    slice.start = get_int(axis.fields, "start");
                if (axis.fields.count("end"))
                    slice.end = get_int(axis.fields, "end");
                indices.emplace_back(slice);
            }
        }
        auto result = ops::index_tensor(input.as<sd_tensor_t>(), indices);
        return std::visit([](auto value) -> Value {
            if constexpr (std::is_same_v<decltype(value), std::shared_ptr<sd_tensor_t>>)
                return Value::object(std::move(value));
            else
                return Value::literal(value);
        },
                          std::move(result));
    }
}
