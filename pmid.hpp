#ifndef __PMI_HPP__
#define __PMI_HPP__

#include "ggml_extend.hpp"

#include "clip.hpp"


struct FuseBlock {
    // network hparams
    int in_dim;    
    int out_dim;  
    int hidden_dim;
    bool use_residue;

    // network params
    // in_layers

    // layer norm 
    struct ggml_tensor* ln_w;  // [in_dim, ]
    struct ggml_tensor* ln_b;  // [in_dim, ]

    struct ggml_tensor* fc1_w;  // [in_dim, hidden_dim]
    struct ggml_tensor* fc1_b;  // [in_dim, ]
    struct ggml_tensor* fc2_w;  // [hidden_dim, out_dim ]
    struct ggml_tensor* fc2_b;  // [hidden_dim, ]


    FuseBlock(int i_d, int o_d, int h_d, bool use_residue = true)
        : in_dim(i_d), out_dim(o_d), hidden_dim(h_d), 
        use_residue(use_residue){
        
    }
    
    
    size_t calculate_mem_size(ggml_type wtype) {
        size_t mem_size = 0;
        mem_size += 2 * ggml_row_size(GGML_TYPE_F32, channels);                         // in_layer_0_w/b
        mem_size += ggml_row_size(GGML_TYPE_F16, out_channels * channels * 3 * 3);      // in_layer_2_w
        mem_size += 5 * ggml_row_size(GGML_TYPE_F32, out_channels);                     // in_layer_2_b/emb_layer_1_b/out_layer_0_w/out_layer_0_b/out_layer_3_b
        mem_size += ggml_row_size(wtype, out_channels * emb_channels);                  // emb_layer_1_w
        mem_size += ggml_row_size(GGML_TYPE_F16, out_channels * out_channels * 3 * 3);  // out_layer_3_w

        if (out_channels != channels) {
            mem_size += ggml_row_size(GGML_TYPE_F16, out_channels * channels * 1 * 1);  // skip_w
            mem_size += ggml_row_size(GGML_TYPE_F32, out_channels);                     // skip_b
        }
        return mem_size;
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_dim);
        ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_dim);

        fc1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim);
        fc1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, hidden_dim);
        fc2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_dim);
        fc2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, out_dim);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "fc1.weight"] = fc1_w;
        tensors[prefix + "fc1.bias"]   = fc1_b;
        tensors[prefix + "fc2.weight"] = fc2_w;
        tensors[prefix + "fc2.bias"]   = fc2_b;
        tensors[prefix + "layernorm.weight"] = ln_w;
        tensors[prefix + "layernorm.bias"]   = ln_b;        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]

        // in_layers
        auto h = ggml_nn_group_norm(ctx, x, ln_w, ln_b);
        h = ggml_add(ctx, ggml_mul_mat(ctx, fc1_w, h),  fc1_b);
        h = ggml_gelu_inplace(ctx, h);
        h = ggml_add(ctx, ggml_mul_mat(ctx, fc2_w, h),  fc2_b);
        if(use_residue)
            x = ggml_add(ctx, x, h);
        return h;    
    }

};

struct FuseModule{
    // network hparams
    int embed_dim;    
    
    struct FuseBlock mlp1;
    struct FuseBlock mlp2;
    // layer norm 
    struct ggml_tensor* ln_w;  // [in_dim, ]
    struct ggml_tensor* ln_b;  // [in_dim, ]


    FuseModule(int imb_d)
        : embed_dim(imb_d){
        
    }


    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        mlp1 = FuseBlock(embed_dim*2, embed_dim, embed_dim, false);
        mlp2 = FuseBlock(embed_dim*2, embed_dim, embed_dim, true);
        ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, embed_dim);
        ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, embed_dim);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {

        tensors[prefix + "layer_norm.weight"] = ln_w;
        tensors[prefix + "layer_norm.bias"]   = ln_b;      
        mlp1.map_by_name(tensors, prefix + ".mlp1.");
        mlp2.map_by_name(tensors, prefix + ".mlp2.");

    }

    struct ggml_tensor* fuse_fn(struct ggml_context* ctx, 
                                struct ggml_tensor* prompt_embeds, 
                                struct ggml_tensor* id_embeds) {
        // x: [N, channels, h, w]

        // in_layers
        auto stacked_id_embeds = ggml_concat(ctx, prompt_embeds, id_embeds); // check whether concat at dim 2 is right

        stacked_id_embeds = mlp1.forward(ctx, stacked_id_embeds);
        stacked_id_embeds = ggml_add(ctx, stacked_id_embeds, prompt_embeds);
        stacked_id_embeds = mlp2.forward(ctx, stacked_id_embeds);
        stacked_id_embeds = ggml_nn_group_norm(ctx, stacked_id_embeds, ln_w, ln_b);
        return stacked_id_embeds;
        
    }


    struct ggml_tensor* forward(struct ggml_context* ctx, 
                           struct ggml_tensor* prompt_embeds,
                           struct ggml_tensor* id_embeds,
                           struct ggml_tensor* class_tokens_mask) {
        // x: [N, channels, h, w]

        // in_layers
        struct ggml_tensor*  h = NULL; 
        return h;    
    }

    
};


struct PhotoMakerIDEncoder : public GGMLModule {
    SDVersion version = VERSION_XL;    
    CLIPVisionModel vision_model;
    FuseModule fuse_module;
    struct ggml_tensor* visual_projection_2; 
    
    PhotoMakerIDEncoder(SDVersion version = VERSION_XL)
        : version(version){      
        
        
        
    
    
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {

        vision_model = CLIPVisionModel();
        fuse_module =  FuseModule(2048);        
        visual_projection_2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1280, 1024); // python [1024, 1280]
        
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        // vision_model.
        fuse_module.map_by_name(tensors, prefix + ".fuse_module.");
        tensors[prefix + "visual_projection_2.weight"] = visual_projection_2;        
    }


    struct ggml_tensor* forward(struct ggml_context* ctx, 
                           struct ggml_tensor* id_pixel_values,
                           struct ggml_tensor* prompt_embeds,                           
                           struct ggml_tensor* class_tokens_mask) {
        // x: [N, channels, h, w]

        // in_layers

        struct ggml_tensor *shared_id_embeds = vision_model.forward(ctx, id_pixel_values); // [1]
        struct ggml_tensor *id_embeds = vision_model.visual_project(ctx, shared_id_embeds);
        struct ggml_tensor *id_embeds_2 = ggml_mul_mat(ctx, visual_projection_2, shared_id_embeds);
        
        // id_embeds = id_embeds.view(b, num_inputs, 1, -1)
        // id_embeds_2 = id_embeds_2.view(b, num_inputs, 1, -1)    

        // id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1)
        id_embeds = ggml_concat(ctx, id_embeds, id_embeds_2); // check whether concat at dim 2 is right
        struct ggml_tensor * updated_prompt_embeds = fuse_module.forward(ctx, prompt_embeds, id_embeds, class_tokens_mask);

        return updated_prompt_embeds

    }

};



#endif  // __PMI_HPP__

