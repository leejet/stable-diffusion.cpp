#include "core/ggml_extensions.h"
#include "ggml-ext-metal.h"
#include "int8_source.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <climits>
#include <memory>
#include <vector>

namespace sd_metal {

    struct Kernels {
        id<MTLComputePipelineState> rotate       = nil;
        id<MTLComputePipelineState> quantize     = nil;
        id<MTLComputePipelineState> multiply     = nil;
        id<MTLComputePipelineState> multiply_mpp = nil;
        ggml_ext_metal_buffer_fn buffer          = nullptr;
        ggml_ext_metal_submit_fn submit          = nullptr;

        ~Kernels() {
            [rotate release];
            [quantize release];
            [multiply release];
            [multiply_mpp release];
        }
    };

    static id<MTLComputePipelineState> tensor_multiply(id<MTLDevice> device) {
        if (@available(macOS 26.0, iOS 26.0, *)) {
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            // Keep building with older SDKs; the optional shader is compiled at runtime.
            options.languageVersion = static_cast<MTLLanguageVersion>(4 << 16);
            options.fastMathEnabled = NO;
            NSError* error          = nil;
            id<MTLLibrary> library  = [device newLibraryWithSource:[NSString stringWithUTF8String:sd_metal_i8_mpp_source] options:options error:&error];
            [options release];
            id<MTLFunction> function             = [library newFunctionWithName:@"sd_i8_multiply_mpp"];
            id<MTLComputePipelineState> pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
            [function release];
            [library release];
            if (pipeline && pipeline.threadExecutionWidth == 32 && pipeline.maxTotalThreadsPerThreadgroup >= 128) {
                return pipeline;
            }
            [pipeline release];
            NSLog(@"sd.cpp Metal INT8 TensorOps unavailable; using SIMD-group fallback: %@", error);
        }
        return nil;
    }

    struct QuantParams {
        uint32_t k, rows, group, unused;
        uint64_t scale_offset;
    };

    struct MatParams {
        uint32_t k, n, rows, flags;
    };

    struct Execution {
        const ggml_ext_execution_context* context;
        ggml_tensor* node;
        Kernels* kernels;
        SDI8Operation op = {};
        int group        = 0;
        bool linear      = false;
    };

    static bool dimensions(const ggml_tensor* node) {
        return node->ne[0] <= INT_MAX && ggml_nrows(node) <= INT_MAX;
    }

    static bool supports_i8(ggml_backend_dev_t, const ggml_tensor* node, void*) {
        SDI8Operation op;
        return sd_ggml_i8_describe(node, &op) && dimensions(node) && dimensions(op.weight) &&
               uint64_t(op.weight->ne[0]) * ggml_nrows(node) <= UINT32_MAX;
    }

    static bool supports_convrot(ggml_backend_dev_t, const ggml_tensor* node, void*) {
        int group;
        return sd_ggml_convrot_describe(node, &group) && dimensions(node->src[0]) &&
               uint64_t(ggml_nelements(node->src[0])) <= UINT32_MAX;
    }

    static size_t aligned(size_t bytes) {
        return (bytes + 255) / 256 * 256;
    }

    static size_t packed_size(int64_t k, int64_t rows) {
        return aligned(size_t(k) * ((rows + 3) / 4 * 4) + rows * sizeof(float));
    }

    static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int, void**, size_t* bytes, void*) {
        auto* node = nodes[0];
        SDI8Operation op;
        const bool linear  = sd_ggml_i8_describe(node, &op);
        const auto* input  = linear ? op.input : node->src[0];
        const int64_t rows = linear ? ggml_nrows(node) : ggml_nrows(input);
        int group;
        if (linear) {
            group = op.group_size;
        } else if (!sd_ggml_convrot_describe(node, &group)) {
            return GGML_STATUS_FAILED;
        }
        *bytes = aligned(sd_ggml_modulate_workspace_size(node));
        if (input->type == GGML_TYPE_F32) {
            if (group > 1)
                *bytes += aligned(ggml_nbytes(input));
            if (linear)
                *bytes += packed_size(input->ne[0], rows);
        }
        return GGML_STATUS_SUCCESS;
    }

    static bool bind_tensor(Kernels* kernels, const ggml_tensor* tensor, ggml_ext_metal_buffer& result) {
        auto buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (!buffer || !tensor->data)
            return false;
        const size_t offset = uintptr_t(tensor->data) - uintptr_t(ggml_backend_buffer_get_base(buffer));
        return kernels->buffer(buffer, offset, ggml_nbytes(tensor), &result);
    }

    static void set_buffer(id<MTLComputeCommandEncoder> encoder, const ggml_ext_metal_buffer& buffer, int index) {
        [encoder setBuffer:(id<MTLBuffer>)buffer.buffer offset:buffer.offset atIndex:index];
    }

    static ggml_status encode(void* command_buffer, void* user_data) {
        auto& e                      = *static_cast<Execution*>(user_data);
        auto* kernels                = e.kernels;
        const auto* input            = e.linear ? e.op.input : e.node->src[0];
        const uint32_t k             = uint32_t(input->ne[0]);
        const uint32_t rows          = uint32_t(e.linear ? ggml_nrows(e.node) : ggml_nrows(input));
        id<MTLCommandBuffer> command = (id<MTLCommandBuffer>)command_buffer;
        ggml_ext_metal_buffer x, dst, output, rotated, packed, xs, w, ws, bias;
        if (!bind_tensor(kernels, input, x) || !bind_tensor(kernels, e.node, dst))
            return GGML_STATUS_FAILED;
        output        = dst;
        size_t offset = 0;
        auto scratch  = [&](size_t bytes, ggml_ext_metal_buffer& binding) {
            bool ok = kernels->buffer(e.context->workspace, offset, bytes, &binding);
            offset += aligned(bytes);
            return ok;
        };
        const size_t output_scratch = sd_ggml_modulate_workspace_size(e.node);
        if (output_scratch && !scratch(output_scratch, output))
            return GGML_STATUS_FAILED;

        if (input->type == GGML_TYPE_F32) {
            rotated = x;
            if (e.group > 1 && !scratch(ggml_nbytes(input), rotated))
                return GGML_STATUS_FAILED;
            packed = output;
            if (e.linear && !scratch(packed_size(k, rows), packed))
                return GGML_STATUS_FAILED;
            QuantParams p                   = {k, rows, uint32_t(e.group), 0, uint64_t(k) * ((uint64_t(rows) + 3) / 4 * 4)};
            id<MTLBlitCommandEncoder> clear = [command blitCommandEncoder];
            [clear fillBuffer:(id<MTLBuffer>)packed.buffer range:NSMakeRange(packed.offset, e.linear ? packed_size(k, rows) : ggml_nbytes(e.node)) value:0];
            [clear endEncoding];
            if (e.group > 1) {
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:kernels->rotate];
                set_buffer(encoder, x, 0);
                set_buffer(encoder, rotated, 1);
                [encoder setBytes:&p length:sizeof(p) atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(uint64_t(k) * rows / e.group, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(e.group, 1, 1)];
                [encoder endEncoding];
            }
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:kernels->quantize];
            set_buffer(encoder, rotated, 0);
            set_buffer(encoder, packed, 1);
            [encoder setBytes:&p length:sizeof(p) atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
            x  = packed;
            xs = packed;
            xs.offset += p.scale_offset;
        } else if (e.op.packed) {
            xs = x;
            xs.offset += size_t(k) * ((uint64_t(rows) + 3) / 4 * 4);
        } else if (!bind_tensor(kernels, e.op.input_scale, xs)) {
            return GGML_STATUS_FAILED;
        }

        if (e.linear) {
            if (!bind_tensor(kernels, e.op.weight, w))
                return GGML_STATUS_FAILED;
            ws = bias = xs;
            if (e.op.weight_scale && !bind_tensor(kernels, e.op.weight_scale, ws))
                return GGML_STATUS_FAILED;
            if (e.op.bias && !bind_tensor(kernels, e.op.bias, bias))
                return GGML_STATUS_FAILED;
            MatParams p                          = {k, uint32_t(e.node->ne[0]), rows, uint32_t((e.op.weight_scale ? 1 : 0) | (e.op.bias ? 2 : 0))};
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:kernels->multiply_mpp ? kernels->multiply_mpp : kernels->multiply];
            set_buffer(encoder, w, 0);
            set_buffer(encoder, x, 1);
            set_buffer(encoder, xs, 2);
            set_buffer(encoder, ws, 3);
            set_buffer(encoder, bias, 4);
            set_buffer(encoder, output, 5);
            [encoder setBytes:&p length:sizeof(p) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake((p.n + 31) / 32, (rows + 31) / 32, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
        }
        if (output_scratch) {
            id<MTLBlitCommandEncoder> copy = [command blitCommandEncoder];
            [copy copyFromBuffer:(id<MTLBuffer>)output.buffer
                     sourceOffset:output.offset
                         toBuffer:(id<MTLBuffer>)dst.buffer
                destinationOffset:dst.offset
                             size:ggml_nbytes(e.node)];
            [copy endEncoding];
        }
        return GGML_STATUS_SUCCESS;
    }

    static ggml_status execute(const ggml_ext_execution_context* context, ggml_tensor* const* nodes, int, void*, void* user_data) {
        Execution e = {context, nodes[0], static_cast<Kernels*>(user_data)};
        e.linear    = sd_ggml_i8_describe(e.node, &e.op);
        if (e.linear) {
            e.group = e.op.group_size;
        } else if (!sd_ggml_convrot_describe(e.node, &e.group)) {
            return GGML_STATUS_FAILED;
        }
        return e.kernels->submit(context->backend, encode, &e);
    }

}

void sd_ggml_register_metal_i8(ggml_backend_dev_t device) {
    using namespace sd_metal;
    auto reg        = ggml_backend_dev_backend_reg(device);
    auto get_device = reinterpret_cast<ggml_ext_metal_device_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_metal_device_v1"));
    auto get_buffer = reinterpret_cast<ggml_ext_metal_buffer_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_metal_buffer_v1"));
    auto submit     = reinterpret_cast<ggml_ext_metal_submit_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_metal_submit_v1"));
    if (!get_device || !get_buffer || !submit)
        return;
    @autoreleasepool {
        id<MTLDevice> metal = (id<MTLDevice>)get_device(device);
        if (!metal || ![metal supportsFamily:MTLGPUFamilyApple7])
            return;
        auto kernels               = std::make_unique<Kernels>();
        kernels->buffer            = get_buffer;
        kernels->submit            = submit;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled    = NO;
        NSError* error             = nil;
        id<MTLLibrary> library     = [metal newLibraryWithSource:[NSString stringWithUTF8String:sd_metal_i8_source] options:options error:&error];
        [options release];
        if (!library) {
            NSLog(@"sd.cpp Metal INT8 compilation failed: %@", error);
            return;
        }
        auto pipeline = [&](NSString* name) {
            id<MTLFunction> function           = [library newFunctionWithName:name];
            id<MTLComputePipelineState> result = function ? [metal newComputePipelineStateWithFunction:function error:&error] : nil;
            [function release];
            return result;
        };
        kernels->rotate   = pipeline(@"sd_i8_rotate");
        kernels->quantize = pipeline(@"sd_i8_quantize");
        kernels->multiply = pipeline(@"sd_i8_multiply");
        [library release];
        if (!kernels->rotate || !kernels->quantize || !kernels->multiply ||
            kernels->quantize.threadExecutionWidth != 32 || kernels->multiply.threadExecutionWidth != 32 ||
            kernels->rotate.maxTotalThreadsPerThreadgroup < 256 ||
            kernels->quantize.maxTotalThreadsPerThreadgroup < 256 || kernels->multiply.maxTotalThreadsPerThreadgroup < 128) {
            NSLog(@"sd.cpp Metal INT8 pipelines unavailable: %@", error);
            return;
        }
        kernels->multiply_mpp = tensor_multiply(metal);
        if (kernels->multiply_mpp) {
            NSLog(@"sd.cpp Metal INT8: using TensorOps INT8 x INT8 -> INT32");
        }
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.name                 = "sd.linear_i8.metal.v1";
        desc.op                   = GGML_OP_CUSTOM;
        desc.custom_op            = sd_ggml_i8_op();
        desc.priority             = 100;
        desc.supports             = supports_i8;
        desc.prepare              = prepare;
        desc.destroy              = [](void*, void*) {};
        desc.execute              = execute;
        desc.user_data            = kernels.get();
        GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
        desc.name      = "sd.quantize_i8_convrot.metal.v1";
        desc.custom_op = sd_ggml_convrot_op();
        desc.supports  = supports_convrot;
        GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
        static std::vector<std::unique_ptr<Kernels>> registrations;
        registrations.push_back(std::move(kernels));
    }
}
