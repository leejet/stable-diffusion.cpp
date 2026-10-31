#ifndef __SD_CORE_GGML_EXTENSIONS_H__
#define __SD_CORE_GGML_EXTENSIONS_H__

#include "ggml-backend.h"

void sd_ggml_register_extensions(ggml_backend_t backend);

#ifndef SD_USE_UPSTREAM_GGML
const struct ggml_ext_op* sd_ggml_rope_op();
ggml_tensor* sd_ggml_rope_apply(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* x, ggml_tensor* pe, bool interleaved);
#endif

#endif
