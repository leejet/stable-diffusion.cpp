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
        mem_size += 2 * ggml_row_size(wtype, in_dim);
        mem_size += ggml_row_size(wtype, in_dim*hidden_dim);
        mem_size += 5 * ggml_row_size(wtype, in_dim);
        mem_size += ggml_row_size(wtype, hidden_dim*out_dim);
        mem_size += ggml_row_size(wtype, hidden_dim);
        
        return mem_size;
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype, ggml_allocr* alloc) {
        ln_w = ggml_new_tensor_1d(ctx, wtype, in_dim);
        ln_b = ggml_new_tensor_1d(ctx, wtype, in_dim);

        fc1_b = ggml_new_tensor_1d(ctx, wtype, hidden_dim);
        fc1_w = ggml_new_tensor_2d(ctx, wtype, in_dim, hidden_dim);
        fc2_b = ggml_new_tensor_1d(ctx, wtype, out_dim);
        fc2_w = ggml_new_tensor_2d(ctx, wtype, hidden_dim, out_dim);
        // alloc all tensors linked to this context
        for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data == NULL) {
                ggml_allocr_alloc(alloc, t);
            }
        }

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


    FuseModule(int imb_d): 
        embed_dim(imb_d), 
        mlp1(imb_d*2, imb_d, imb_d, false),
        mlp2(imb_d, imb_d, imb_d, true) {

        // mlp1 = FuseBlock(embed_dim*2, embed_dim, embed_dim, false);
        // mlp2 = FuseBlock(embed_dim*2, embed_dim, embed_dim, true);
        
        
    }


    void init_params(struct ggml_context* ctx, ggml_type wtype, ggml_allocr* alloc) {                  
        ln_w = ggml_new_tensor_1d(ctx, wtype, embed_dim);
        ln_b = ggml_new_tensor_1d(ctx, wtype, embed_dim);
        // alloc all tensors linked to this context
        mlp1.init_params(ctx, wtype, alloc);
        mlp2.init_params(ctx, wtype, alloc);
        for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data == NULL) {
                ggml_allocr_alloc(alloc, t);
            }
        }
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {

        tensors[prefix + "layer_norm.weight"] = ln_w;
        tensors[prefix + "layer_norm.bias"]   = ln_b;      
        mlp1.map_by_name(tensors, prefix + "mlp1.");
        mlp2.map_by_name(tensors, prefix + "mlp2.");

    }

    size_t calculate_mem_size(ggml_type wtype) {
        size_t mem_size = mlp1.calculate_mem_size(wtype);
        mem_size += mlp2.calculate_mem_size(wtype);
        mem_size += 2 * ggml_row_size(wtype, embed_dim);        
        return mem_size;
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
        : version(version), 
        fuse_module(2048) {      
        vision_model = CLIPVisionModel();
        // fuse_module = FuseModule(2048);

    }

    // void init_params(ggml_context* ctx, ggml_backend_t backend, ggml_type wtype, ggml_allocr* alloc) {   
    void init_params() {       

        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);
        
        vision_model.init_params(params_ctx, backend, wtype, alloc);
        fuse_module.init_params(params_ctx, wtype, alloc);
        // visual_projection_2 = ggml_new_tensor_2d(params_ctx, wtype, 1280, 1024); // python [1024, 1280]
        visual_projection_2 = ggml_new_tensor_2d(params_ctx, wtype, 1024, 1280); // python [1024, 1280]
        ggml_allocr_alloc(alloc, visual_projection_2);
        ggml_allocr_free(alloc); 
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        vision_model.map_by_name(tensors, prefix + "vision_model.", prefix);
        fuse_module.map_by_name(tensors, prefix + "fuse_module.");
        tensors[prefix + "visual_projection_2.weight"] = visual_projection_2;        
    }

    size_t calculate_mem_size() {
        size_t mem_size = vision_model.calculate_mem_size(wtype);

        mem_size += fuse_module.calculate_mem_size(wtype);
        
        mem_size += ggml_row_size(wtype, 1280*1024);
              
        return mem_size;
    }

    size_t get_num_tensors() {
        size_t num_tensors = (3 + 2 + 37);
        
        return num_tensors;
    }


    struct ggml_tensor* forward(struct ggml_context* ctx, 
                           struct ggml_tensor* id_pixel_values,
                           struct ggml_tensor* prompt_embeds,                           
                           struct ggml_tensor* class_tokens_mask) {
        // x: [N, channels, h, w]

        // in_layers

        struct ggml_tensor *shared_id_embeds = vision_model.forward(ctx, id_pixel_values); // [batch_size, seq_length, hidden_size]
        struct ggml_tensor *id_embeds = vision_model.visual_project(ctx, shared_id_embeds); // [batch_size, seq_length, proj_dim(768)]
        struct ggml_tensor *id_embeds_2 = ggml_mul_mat(ctx, visual_projection_2, shared_id_embeds); // [batch_size, seq_length, 1280]
        
        // id_embeds = id_embeds.view(b, num_inputs, 1, -1)
        // id_embeds_2 = id_embeds_2.view(b, num_inputs, 1, -1)    

        // id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1)
        id_embeds = ggml_concat(ctx, id_embeds, id_embeds_2); // [batch_size, seq_length, 1, 2048] check whether concat at dim 2 is right
        struct ggml_tensor * updated_prompt_embeds = fuse_module.forward(ctx, prompt_embeds, id_embeds, class_tokens_mask);

        return updated_prompt_embeds;

    }

};



#endif  // __PMI_HPP__

