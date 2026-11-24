#include "dgl/bindings/condition.h"

#include <algorithm>
#include <iterator>

#include "conditioning/conditioner.hpp"
#include "dgl/bindings/arguments.h"
#include "dgl/ops/tensor.h"

namespace sd::dgl {

    static bool tensor_field(const Value& value) {
        if (value.type().kind == Kind::Tensor)
            return true;
        return value.type() == Type{Kind::List, Kind::Tensor};
    }

    static Value zero_field(const Value& value) {
        if (value.type().kind == Kind::Tensor) {
            const auto input = value.as<sd_tensor_t>();
            return Value::object(ops::filled_tensor(input->shape(), input->type, int64_t(0)));
        }
        std::vector<Value> items;
        items.reserve(value.items.size());
        for (const auto& item : value.items)
            items.push_back(zero_field(item));
        return Value::list(std::move(items), Kind::Tensor);
    }

    Value zero_out_condition(Values& args, RunState&) {
        auto fields         = args.at("condition").fields;
        const auto selected = args.find("fields");
        if (selected == args.end()) {
            for (auto& [name, value] : fields)
                if (tensor_field(value))
                    value = zero_field(value);
        } else {
            for (const auto& item : selected->second.items) {
                const auto name  = item.scalar.get<std::string>();
                const auto found = fields.find(name);
                if (found == fields.end())
                    throw std::runtime_error("zero_out: condition has no field '" + name + "'");
                if (!tensor_field(found->second))
                    throw std::runtime_error("zero_out: field '" + name + "' must be a tensor or tensor list");
                found->second = zero_field(found->second);
            }
        }
        return Value::dict(std::move(fields));
    }

    template <auto Member>
    static std::optional<Value> condition_tensor(SDCondition& condition) {
        auto& tensor = condition.*Member;
        if (tensor.empty())
            return std::nullopt;
        return Value::object(make_tensor(std::move(tensor)));
    }

    template <auto Member>
    static std::optional<Value> condition_tensors(SDCondition& condition) {
        auto& tensors = condition.*Member;
        if (tensors.empty())
            return std::nullopt;
        std::vector<Value> items;
        items.reserve(tensors.size());
        for (auto& tensor : tensors) {
            if (tensor.empty())
                throw std::runtime_error("conditioning contains an empty tensor list item");
            items.push_back(Value::object(make_tensor(std::move(tensor))));
        }
        return Value::list(std::move(items), Kind::Tensor);
    }

    static std::optional<Value> image_embeds(SDCondition& condition) {
        if (condition.c_image_embeds.empty())
            return std::nullopt;
        std::vector<Value> items;
        for (auto& [index, tensor] : condition.c_image_embeds) {
            if (tensor.empty())
                throw std::runtime_error("conditioning contains an empty image embedding");
            items.push_back(Value::object(make_tensor(std::move(tensor))));
        }
        return Value::list(std::move(items), Kind::Tensor);
    }

    static std::optional<Value> image_embed_indices(SDCondition& condition) {
        if (condition.c_image_embeds.empty())
            return std::nullopt;
        std::vector<Value> items;
        for (const auto& entry : condition.c_image_embeds)
            items.push_back(Value::object(make_tensor(sd::Tensor<int32_t>::from_vector({entry.first}))));
        return Value::list(std::move(items), Kind::Tensor);
    }

    static const std::string reference_kinds[] = {"image", "video", "audio", "video_audio"};

    static std::optional<Value> reference_blocks(SDCondition& condition) {
        if (condition.c_reference_blocks.empty())
            return std::nullopt;
        std::vector<Value> blocks;
        blocks.reserve(condition.c_reference_blocks.size());
        for (const auto& block : condition.c_reference_blocks) {
            const auto kind = static_cast<size_t>(block.kind);
            if (kind >= std::size(reference_kinds))
                throw std::runtime_error("invalid conditioning reference block kind");
            Values fields{{"kind", Value::literal(reference_kinds[kind])}};
            if (block.kind != MiniMaxH3ReferenceKind::AUDIO)
                fields.emplace("video", Value::literal(block.video_index));
            if (block.kind == MiniMaxH3ReferenceKind::AUDIO || block.kind == MiniMaxH3ReferenceKind::VIDEO_AUDIO)
                fields.emplace("audio", Value::literal(block.audio_index));
            blocks.push_back(Value::dict(std::move(fields)));
        }
        return Value::list(std::move(blocks), Kind::Dict);
    }

    static std::vector<MiniMaxH3ReferenceBlock> read_reference_blocks(const Value& value) {
        if (value.type().kind != Kind::List)
            throw std::runtime_error("reference_blocks must be a list of dicts");
        std::vector<MiniMaxH3ReferenceBlock> blocks;
        blocks.reserve(value.items.size());
        for (size_t i = 0; i < value.items.size(); ++i) {
            const auto& item = value.items[i];
            const auto label = "reference_blocks[" + std::to_string(i) + "]";
            if (item.type().kind != Kind::Dict)
                throw std::runtime_error(label + " must be a dict");
            const auto& fields = item.fields;
            const auto kind    = fields.find("kind");
            if (kind == fields.end() || kind->second.type().kind != Kind::String)
                throw std::runtime_error(label + ".kind must be a string");
            const auto name  = kind->second.scalar.get<std::string>();
            const auto found = std::find(std::begin(reference_kinds), std::end(reference_kinds), name);
            if (found == std::end(reference_kinds))
                throw std::runtime_error(label + ".kind must be image, video, audio or video_audio, got '" + name + "'");
            MiniMaxH3ReferenceBlock block;
            block.kind = MiniMaxH3ReferenceKind(found - std::begin(reference_kinds));
            auto index = [&](const char* field, bool required) {
                const auto entry = fields.find(field);
                if (!required) {
                    if (entry != fields.end())
                        throw std::runtime_error(label + " of kind '" + name + "' does not accept '" + field + "'");
                    return int32_t(-1);
                }
                if (entry == fields.end())
                    throw std::runtime_error(label + " of kind '" + name + "' requires '" + field + "'");
                if (entry->second.type().kind != Kind::Integer)
                    throw std::runtime_error(label + "." + field + " must be an integer");
                const auto number = entry->second.scalar.get<int64_t>();
                if (number < 0 || number > INT32_MAX)
                    throw std::runtime_error(label + "." + field + " must be between 0 and 2147483647");
                return int32_t(number);
            };
            block.video_index = index("video", block.kind != MiniMaxH3ReferenceKind::AUDIO);
            block.audio_index = index("audio", block.kind == MiniMaxH3ReferenceKind::AUDIO || block.kind == MiniMaxH3ReferenceKind::VIDEO_AUDIO);
            for (const auto& [field, entry] : fields)
                if (field != "kind" && field != "video" && field != "audio")
                    throw std::runtime_error(label + " has unknown field '" + field + "'");
            blocks.push_back(block);
        }
        return blocks;
    }

    void validate_reference_blocks(const SDCondition& condition, size_t video_count) {
        for (size_t i = 0; i < condition.c_reference_blocks.size(); ++i) {
            const auto& block = condition.c_reference_blocks[i];
            const auto label  = "reference_blocks[" + std::to_string(i) + "]";
            if (block.video_index >= 0 && size_t(block.video_index) >= video_count)
                throw std::runtime_error(label + ".video index " + std::to_string(block.video_index) + " is out of range for " + std::to_string(video_count) + " reference latents");
            if (block.audio_index >= 0 && size_t(block.audio_index) >= condition.c_ref_audios.size())
                throw std::runtime_error(label + ".audio index " + std::to_string(block.audio_index) + " is out of range for " + std::to_string(condition.c_ref_audios.size()) + " reference audios");
        }
    }

    struct ConditionField {
        const char* name;
        OutputSignature output;
        std::optional<Value> (*extract)(SDCondition&);
        void (*assign)(SDCondition&, const Value&) = nullptr;
    };

    template <typename T>
    static void assign_condition_tensor(sd::Tensor<T>& target, const Value& value) {
        target = read_tensor<T>(value);
    }

    template <auto Member>
    static ConditionField tensor_field(const char* name) {
        return {name, {Kind::Tensor}, condition_tensor<Member>, [](SDCondition& condition, const Value& value) {
                    assign_condition_tensor(condition.*Member, value);
                }};
    }

    template <auto Member>
    static ConditionField tensor_list_field(const char* name) {
        return {name, {Kind::List, Kind::Tensor}, condition_tensors<Member>, [](SDCondition& condition, const Value& value) {
                    condition.*Member = float_tensor_list(value);
                }};
    }

    static const ConditionField condition_fields[] = {
        tensor_field<&SDCondition::c_crossattn>("crossattn"),
        tensor_field<&SDCondition::c_vector>("vector"),
        tensor_field<&SDCondition::c_concat>("concat"),
        tensor_field<&SDCondition::c_t5_ids>("t5_ids"),
        tensor_field<&SDCondition::c_t5_weights>("t5_weights"),
        tensor_field<&SDCondition::c_input_ids>("input_ids"),
        tensor_field<&SDCondition::c_position_ids>("position_ids"),
        tensor_field<&SDCondition::c_token_types>("token_types"),
        tensor_field<&SDCondition::c_vinput_mask>("vinput_mask"),
        tensor_list_field<&SDCondition::extra_c_crossattns>("extra_crossattn"),
        {"image_embeds", {Kind::List, Kind::Tensor}, image_embeds},
        {"image_embed_indices", {Kind::List, Kind::Tensor}, image_embed_indices},
        tensor_list_field<&SDCondition::c_ref_images>("ref_images"),
        tensor_list_field<&SDCondition::c_ref_audios>("ref_audios"),
        {"reference_blocks", {{Kind::List, Kind::Dict}, {{"kind", {Kind::String}}, {"video", {Kind::Integer}}, {"audio", {Kind::Integer}}}}, reference_blocks},
    };

    OutputSignature condition_signature() {
        OutputSignature signature{Kind::Dict};
        for (const auto& field : condition_fields)
            signature.fields.emplace(field.name, field.output);
        return signature;
    }

    Value condition_value(SDCondition condition) {
        Values fields;
        for (const auto& field : condition_fields)
            if (auto value = field.extract(condition))
                fields.emplace(field.name, std::move(*value));
        return Value::dict(std::move(fields));
    }

    SDCondition read_condition(const Values& fields, const Values& overrides) {
        SDCondition c;
        for (const auto& field : condition_fields) {
            auto value = fields.find(field.name);
            if (field.assign && value != fields.end())
                field.assign(c, value->second);
        }
        if (fields.count("image_embeds") && fields.count("image_embed_indices")) {
            const auto embeds   = float_tensor_list(fields.at("image_embeds"));
            const auto& indices = fields.at("image_embed_indices").items;
            if (embeds.size() != indices.size()) {
                throw std::runtime_error("image embeddings and indices differ in length");
            }
            for (size_t i = 0; i < embeds.size(); ++i) {
                const auto& index = read_tensor<int32_t>(indices[i]);
                if (index.numel() != 1 || index[0] < 0) {
                    throw std::runtime_error("invalid image embedding index");
                }
                c.c_image_embeds.emplace_back(index[0], embeds[i]);
            }
        }
        if (overrides.count("ref_audios")) {
            c.c_ref_audios = float_tensor_list(overrides.at("ref_audios"));
        }
        if (overrides.count("keyframe_indices")) {
            std::vector<int32_t> indices;
            for (const auto& index : overrides.at("keyframe_indices").items) {
                auto value = index.scalar.get<int64_t>();
                if (value < 0 || value > INT32_MAX) {
                    throw std::runtime_error("invalid keyframe index");
                }
                indices.push_back(int32_t(value));
            }
            c.c_position_ids = sd::Tensor<int32_t>::from_vector(std::move(indices));
        }
        const auto* block = overrides.count("reference_blocks") ? &overrides.at("reference_blocks")
                            : fields.count("reference_blocks")  ? &fields.at("reference_blocks")
                                                                : nullptr;
        if (block)
            c.c_reference_blocks = read_reference_blocks(*block);
        if (overrides.count("concat")) {
            c.c_concat = float_tensor(overrides, "concat");
        }
        return c;
    }

}  // namespace sd::dgl
