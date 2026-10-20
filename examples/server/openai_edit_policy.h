#ifndef __SD_EXAMPLES_SERVER_OPENAI_EDIT_POLICY_H__
#define __SD_EXAMPLES_SERVER_OPENAI_EDIT_POLICY_H__

#include "common/resource_owners.hpp"

enum class OpenAIEditInitImageAction {
    KeepExisting,
    UseUploaded,
    Clear,
};

inline bool openai_edit_uses_uploaded_image_as_init(bool supports_ref_image_conditioning,
                                                    bool has_ref_images,
                                                    bool has_mask,
                                                    bool has_explicit_strength_override) {
    return !supports_ref_image_conditioning || !has_ref_images || has_mask || has_explicit_strength_override;
}

inline OpenAIEditInitImageAction apply_openai_edit_init_image_policy(
    SDImageOwner& init_image,
    SDImageOwner& uploaded_image,
    bool uploaded_image_was_first_reference,
    bool supports_ref_image_conditioning,
    bool has_ref_images,
    bool has_mask,
    bool has_extra_init_image_override,
    bool has_explicit_strength_override) {
    if (!uploaded_image_was_first_reference || has_extra_init_image_override) {
        return OpenAIEditInitImageAction::KeepExisting;
    }

    if (openai_edit_uses_uploaded_image_as_init(supports_ref_image_conditioning,
                                                has_ref_images,
                                                has_mask,
                                                has_explicit_strength_override)) {
        init_image = std::move(uploaded_image);
        return OpenAIEditInitImageAction::UseUploaded;
    }

    init_image.reset();
    return OpenAIEditInitImageAction::Clear;
}

#endif  // __SD_EXAMPLES_SERVER_OPENAI_EDIT_POLICY_H__
