#include "result.h"

#include <algorithm>

#include "api/api.h"
#include "dgl/tensor.h"

namespace {
    using sd::dgl::Kind;
    using sd::dgl::make_tensor;
    using sd::dgl::Value;

    sd_dgl_value_t scalar(sd_dgl_value_kind_t kind, nlohmann::json value, std::string name = {}) {
        return {kind, std::move(name), std::move(value), {}, {}};
    }

    sd_dgl_value_t tensor(std::shared_ptr<sd_tensor_t> handle, std::string name = {}) {
        return {SD_DGL_VALUE_TENSOR, std::move(name), {}, std::move(handle), {}};
    }

    sd_dgl_value_t export_value(Value value, std::string name = {}) {
        switch (value.type().kind) {
            case Kind::Void:
                return {SD_DGL_VALUE_VOID, std::move(name), {}, {}, {}};
            case Kind::Boolean:
                return scalar(SD_DGL_VALUE_BOOL, std::move(value.scalar), std::move(name));
            case Kind::Integer:
                return scalar(SD_DGL_VALUE_INT, std::move(value.scalar), std::move(name));
            case Kind::Number:
                return scalar(SD_DGL_VALUE_NUMBER, std::move(value.scalar), std::move(name));
            case Kind::String:
                return scalar(SD_DGL_VALUE_STRING, std::move(value.scalar), std::move(name));
            case Kind::Tensor:
                return tensor(value.as<sd_tensor_t>(), std::move(name));
            case Kind::Bytes:
                return {SD_DGL_VALUE_BYTES, std::move(name), {}, {}, {}, value.as<sd::dgl::Bytes>()};
            case Kind::List: {
                sd_dgl_value_t result{SD_DGL_VALUE_LIST, std::move(name), {}, {}, {}};
                result.children.reserve(value.items.size());
                for (auto& item : value.items)
                    result.children.push_back(export_value(std::move(item)));
                return result;
            }
            case Kind::Dict: {
                sd_dgl_value_t result{SD_DGL_VALUE_DICT, std::move(name), {}, {}, {}};
                result.children.reserve(value.fields.size());
                for (auto& [field, item] : value.fields)
                    result.children.push_back(export_value(std::move(item), field));
                return result;
            }
            default:
                return {SD_DGL_VALUE_OPAQUE, std::move(name), {}, {}, {}};
        }
    }

    std::shared_ptr<sd_tensor_t> scalar_tensor(const sd_dgl_value_t& value) {
        switch (value.kind) {
            case SD_DGL_VALUE_BOOL: {
                uint8_t boolean = value.scalar.get<bool>();
                return make_tensor(SD_TENSOR_U8, std::vector<int64_t>{}, &boolean, sizeof(boolean));
            }
            case SD_DGL_VALUE_INT: {
                auto number = value.scalar.get<int64_t>();
                return make_tensor(SD_TENSOR_I64, std::vector<int64_t>{}, &number, sizeof(number));
            }
            case SD_DGL_VALUE_NUMBER: {
                auto number = value.scalar.get<double>();
                return make_tensor(SD_TENSOR_F64, std::vector<int64_t>{}, &number, sizeof(number));
            }
            case SD_DGL_VALUE_STRING: {
                const auto& text = value.scalar.get_ref<const std::string&>();
                return make_tensor(SD_TENSOR_U8, std::vector<int64_t>{int64_t(text.size())}, text.data(), text.size());
            }
            default:
                return {};
        }
    }

    void flatten(const sd_dgl_value_t& value, const std::string& path, sd_dgl_result_t::Tensors& tensors) {
        if (value.kind == SD_DGL_VALUE_DICT || value.kind == SD_DGL_VALUE_LIST) {
            for (size_t i = 0; i < value.children.size(); ++i) {
                const auto& child = value.children[i];
                auto name         = value.kind == SD_DGL_VALUE_LIST ? std::to_string(i) : child.name;
                flatten(child, path.empty() ? name : path + "." + name, tensors);
            }
            return;
        }
        const auto key = path.empty() ? "result" : path;
        switch (value.kind) {
            case SD_DGL_VALUE_TENSOR:
                tensors.emplace(key, value.tensor);
                break;
            default:
                if (auto data = scalar_tensor(value))
                    tensors.emplace(key, std::move(data));
                break;
        }
    }

    const sd_dgl_result_t::Tensors* flat_tensors(const sd_dgl_result_t* result) {
        if (!result)
            return nullptr;
        const sd_dgl_result_t::Tensors* tensors = nullptr;
        sd::api::call([&] {
            tensors = &result->flat_tensors();
            return SD_STATUS_OK;
        });
        return tensors;
    }

    sd_status_t require_kind(const sd_dgl_value_t* value, sd_dgl_value_kind_t kind, const void* out) {
        sd::api::clear_error();
        if (!value || !out)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "value and output are required");
        if (value->kind != kind)
            return sd::api::error(SD_STATUS_TYPE_MISMATCH, nullptr, "DGL result value has a different type");
        return SD_STATUS_OK;
    }
}

sd_dgl_result_t::sd_dgl_result_t(Value value)
    : type(sd::dgl::type_name(value.type())), root(export_value(std::move(value))) {}

const sd_dgl_result_t::Tensors& sd_dgl_result_t::flat_tensors() const {
    std::call_once(flat_init_, [&] {
        Tensors tensors;
        flatten(root, "", tensors);
        flat_ = std::move(tensors);
    });
    return flat_;
}

void sd_dgl_result_free(sd_dgl_result_t* result) {
    delete result;
}

const char* sd_dgl_result_type(const sd_dgl_result_t* result) {
    return result ? result->type.c_str() : nullptr;
}

size_t sd_dgl_result_count(const sd_dgl_result_t* result) {
    auto tensors = flat_tensors(result);
    return tensors ? tensors->size() : 0;
}

const char* sd_dgl_result_name(const sd_dgl_result_t* result, size_t index) {
    auto tensors = flat_tensors(result);
    if (!tensors || index >= tensors->size())
        return nullptr;
    return std::next(tensors->begin(), index)->first.c_str();
}

const sd_tensor_t* sd_dgl_result_tensor(const sd_dgl_result_t* result, const char* name) {
    auto tensors = name ? flat_tensors(result) : nullptr;
    if (!tensors)
        return nullptr;
    auto it = tensors->find(name);
    return it == tensors->end() ? nullptr : it->second.get();
}

const sd_dgl_value_t* sd_dgl_result_root(const sd_dgl_result_t* result) {
    return result ? &result->root : nullptr;
}

sd_dgl_value_kind_t sd_dgl_value_kind(const sd_dgl_value_t* value) {
    return value ? value->kind : SD_DGL_VALUE_INVALID;
}

size_t sd_dgl_value_count(const sd_dgl_value_t* value) {
    return value ? value->children.size() : 0;
}

const sd_dgl_value_t* sd_dgl_value_at(const sd_dgl_value_t* value, size_t index) {
    return value && index < value->children.size() ? &value->children[index] : nullptr;
}

const char* sd_dgl_value_name(const sd_dgl_value_t* value) {
    return value && !value->name.empty() ? value->name.c_str() : nullptr;
}

const sd_dgl_value_t* sd_dgl_value_field(const sd_dgl_value_t* value, const char* name) {
    if (!value || !name || !*name)
        return nullptr;
    const auto match = std::find_if(value->children.begin(), value->children.end(),
                                    [&](const auto& child) { return child.name == name; });
    return match == value->children.end() ? nullptr : &*match;
}

sd_status_t sd_dgl_value_get_bool(const sd_dgl_value_t* value, bool* out) {
    const auto status = require_kind(value, SD_DGL_VALUE_BOOL, out);
    if (status == SD_STATUS_OK)
        *out = value->scalar.get<bool>();
    return status;
}

sd_status_t sd_dgl_value_get_int(const sd_dgl_value_t* value, int64_t* out) {
    const auto status = require_kind(value, SD_DGL_VALUE_INT, out);
    if (status == SD_STATUS_OK)
        *out = value->scalar.get<int64_t>();
    return status;
}

sd_status_t sd_dgl_value_get_number(const sd_dgl_value_t* value, double* out) {
    const auto status = require_kind(value, SD_DGL_VALUE_NUMBER, out);
    if (status == SD_STATUS_OK)
        *out = value->scalar.get<double>();
    return status;
}

sd_status_t sd_dgl_value_get_string(const sd_dgl_value_t* value, const char** out, size_t* size) {
    const auto status = require_kind(value, SD_DGL_VALUE_STRING, size ? out : nullptr);
    if (status == SD_STATUS_OK) {
        const auto& text = value->scalar.get_ref<const std::string&>();
        *out             = text.c_str();
        *size            = text.size();
    }
    return status;
}

const sd_tensor_t* sd_dgl_value_tensor(const sd_dgl_value_t* value) {
    return value && value->kind == SD_DGL_VALUE_TENSOR ? value->tensor.get() : nullptr;
}

sd_status_t sd_dgl_value_get_bytes(const sd_dgl_value_t* value, const void** out, size_t* size) {
    if (out)
        *out = nullptr;
    if (size)
        *size = 0;
    const auto status = require_kind(value, SD_DGL_VALUE_BYTES, size ? out : nullptr);
    if (status == SD_STATUS_OK) {
        *out  = value->bytes->data.data();
        *size = value->bytes->data.size();
    }
    return status;
}
