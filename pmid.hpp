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
        mem_size += ggml_row_size(wtype, out_dim);
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


    size_t get_num_tensors() {
        return 6;
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
        auto h = ggml_nn_layer_norm(ctx, x, ln_w, ln_b);
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

    size_t get_num_tensors() {
        size_t n = mlp1.get_num_tensors();
        n += mlp2.get_num_tensors();
        n += 2;
        return n;
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

        // stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        // stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        // stacked_id_embeds = self.mlp2(stacked_id_embeds)
        // stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        // return stacked_id_embeds
        // in_layers

        auto prompt_embeds0 = ggml_cont(ctx, ggml_permute(ctx, prompt_embeds, 2, 0, 1, 3));
        auto id_embeds0     = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 2, 0, 1, 3));
        // concat is along dim 2   
        auto stacked_id_embeds = ggml_concat(ctx, prompt_embeds0, id_embeds0); 
        stacked_id_embeds = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 1, 2, 0, 3));

        stacked_id_embeds = mlp1.forward(ctx, stacked_id_embeds);
        stacked_id_embeds = ggml_add(ctx, stacked_id_embeds, prompt_embeds);
        stacked_id_embeds = mlp2.forward(ctx, stacked_id_embeds);
        stacked_id_embeds = ggml_nn_layer_norm(ctx, stacked_id_embeds, ln_w, ln_b);
        return stacked_id_embeds;
        
    }


    struct ggml_tensor* forward(struct ggml_context* ctx, 
                           struct ggml_tensor* prompt_embeds,
                           struct ggml_tensor* id_embeds,
                           struct ggml_tensor* class_tokens_mask,
                           struct ggml_tensor* class_tokens_mask_pos,
                           struct ggml_tensor* left,
                           struct ggml_tensor* right) {
        // x: [N, channels, h, w]

        // in_layers

        //  # id_embeds shape: [b, max_num_inputs, 1, 2048]
        // id_embeds = id_embeds.to(prompt_embeds.dtype)
        // num_inputs = class_tokens_mask.sum().unsqueeze(0) # TODO: check for training case
        // batch_size, max_num_inputs = id_embeds.shape[:2]
        // # seq_length: 77
        // seq_length = prompt_embeds.shape[1]
        // # flat_id_embeds shape: [b*max_num_inputs, 1, 2048]
        // flat_id_embeds = id_embeds.view(
        //     -1, id_embeds.shape[-2], id_embeds.shape[-1]
        // )
        // # valid_id_mask [b*max_num_inputs]
        // valid_id_mask = (
        //     torch.arange(max_num_inputs, device=flat_id_embeds.device)[None, :]
        //     < num_inputs[:, None]
        // )
        // valid_id_embeds = flat_id_embeds[valid_id_mask.flatten()]

        // prompt_embeds = prompt_embeds.view(-1, prompt_embeds.shape[-1])
        // class_tokens_mask = class_tokens_mask.view(-1)
        struct ggml_tensor * valid_id_embeds = id_embeds;
        // # slice out the image token embeddings
        struct ggml_tensor * image_token_embeds = ggml_get_rows(ctx, prompt_embeds, class_tokens_mask_pos);
        // print_ggml_tensor(image_token_embeds, true, "image_token_embeds");
        // print_ggml_tensor(valid_id_embeds, true, "valid_id_embeds");
        struct ggml_tensor *stacked_id_embeds = fuse_fn(ctx, image_token_embeds, valid_id_embeds);
        // print_ggml_tensor(stacked_id_embeds, true, "stacked_id_embeds_before_concat");

        stacked_id_embeds = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 0, 2, 1, 3));
        // print_ggml_tensor(stacked_id_embeds, true, "stacked_id_embeds_after_permute");
        if(left && right){
            // print_ggml_tensor(left, true, "left");
            // print_ggml_tensor(right, true, "right");
            stacked_id_embeds = ggml_concat(ctx, left, stacked_id_embeds);
            stacked_id_embeds = ggml_concat(ctx, stacked_id_embeds, right);
        }else if(left){
            stacked_id_embeds = ggml_concat(ctx, left, stacked_id_embeds);
        }else if(right){
            stacked_id_embeds = ggml_concat(ctx, stacked_id_embeds, right);
        }
        // print_ggml_tensor(stacked_id_embeds, true, "stacked_id_embeds_after_concat");
        stacked_id_embeds = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 0, 2, 1, 3));
        // print_ggml_tensor(stacked_id_embeds, true, "stacked_id_embeds_after_permute_2");
        // print_ggml_tensor(class_tokens_mask, true, "class_tokens_mask");
        // print_ggml_tensor(prompt_embeds, true, "prompt_embeds");
        // assert class_tokens_mask.sum() == stacked_id_embeds.shape[0], f"{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}"
        // prompt_embeds.masked_scatter_(class_tokens_mask[:, None], stacked_id_embeds.to(prompt_embeds.dtype))
        // struct ggml_tensor *prompt_embeds_perm = ggml_cont(ctx, ggml_permute(ctx, prompt_embeds, 1, 0, 2, 3));
        struct ggml_tensor *prompt_embeds_perm = ggml_cont(ctx, ggml_transpose(ctx, prompt_embeds));
        // print_ggml_tensor(prompt_embeds_perm, true, "prompt_embeds_perm");
        class_tokens_mask = ggml_repeat(ctx, class_tokens_mask, prompt_embeds_perm);
        // print_ggml_tensor(class_tokens_mask, true, "class_tokens_mask_repeat");
        class_tokens_mask =  ggml_cont(ctx, ggml_transpose(ctx, class_tokens_mask));
        // print_ggml_tensor(class_tokens_mask, true, "class_tokens_mask_transpose");
        prompt_embeds = ggml_mul(ctx, prompt_embeds, class_tokens_mask);
        // print_ggml_tensor(prompt_embeds, true, "prompt_embeds_after_mul");
        struct ggml_tensor * updated_prompt_embeds = ggml_add(ctx, prompt_embeds, stacked_id_embeds);
        // print_ggml_tensor(updated_prompt_embeds, true, "updated_prompt_embeds");
        // updated_prompt_embeds = prompt_embeds.view(batch_size, seq_length, -1)
        return updated_prompt_embeds;
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
        // wtype = GGML_TYPE_F32;

    }

    // void init_params(ggml_context* ctx, ggml_backend_t backend, ggml_type wtype, ggml_allocr* alloc) {   
    void init_params(ggml_type wtype) {               
        // LOG_INFO(" PMID wtype: %s", ggml_type_name(wtype));
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

        wtype = GGML_TYPE_F32;
        // LOG_INFO(" PMID wtype: %s", ggml_type_name(wtype));

        size_t mem_size = vision_model.calculate_mem_size(wtype);
        mem_size += fuse_module.calculate_mem_size(wtype);
        
        mem_size += ggml_row_size(wtype, 1280*1024);
              
        return mem_size;
    }

    size_t get_num_tensors() {
        size_t num_tensors = (3 + 2 + 37 * vision_model.num_hidden_layers);
        num_tensors += fuse_module.get_num_tensors() + 1;
        return num_tensors;
    }


    struct ggml_tensor* forward(struct ggml_context* ctx, 
                           struct ggml_tensor* id_pixel_values,
                           struct ggml_tensor* prompt_embeds,                           
                           struct ggml_tensor* class_tokens_mask,
                           struct ggml_tensor* class_tokens_mask_pos,
                           struct ggml_tensor* cls,
                           struct ggml_tensor* class_embedding_temp,
                           struct ggml_tensor* positions,
                           struct ggml_tensor* left,
                           struct ggml_tensor* right) {
        // x: [N, channels, h, w]

        // in_layers
        ggml_set_name(id_pixel_values, "id_pixel_values_input");
        ggml_set_name(prompt_embeds, "prompt_embeds_input");
        ggml_set_name(class_tokens_mask, "class_tokens_mask_input");
        ggml_set_name(class_tokens_mask_pos, "class_tokens_mask_pos_input");
        ggml_set_name(cls, "cls_input");
        ggml_set_name(class_embedding_temp, "class_embedding_temp_input");
        ggml_set_name(positions, "positions_input");
        ggml_set_name(left, "left_input");
        ggml_set_name(right, "right_input");
        

        // print_ggml_tensor(prompt_embeds, true, "prompt_embeds");
        // print_ggml_tensor(class_embedding_temp, true, "class_embedding_temp");
        struct ggml_tensor *shared_id_embeds = vision_model.forward(ctx, 
                                                             id_pixel_values,
                                                             cls,
                                                             class_embedding_temp,
                                                             positions
                                                             ); // [batch_size, seq_length, hidden_size]
        // print_ggml_tensor(shared_id_embeds, true, "shared_id_embeds");

        // if(class_tokens_mask->backend == GGML_BACKEND_GPU){            
        //     int *ctm = (int *)malloc(class_tokens_mask->ne[0]);
        //     ggml_backend_tensor_get(class_tokens_mask, ctm, 0, ggml_nbytes(class_tokens_mask));
        //     printf("class_tokens_mask[");
        //     for(int i = 0; i < class_tokens_mask->ne[0]; i++)
        //         printf("%d, ", ctm[i]);
        //     printf("]\n");
        //     free(ctm);
        // }
        struct ggml_tensor *id_embeds = vision_model.visual_project(ctx, shared_id_embeds); // [batch_size, seq_length, proj_dim(768)]
        struct ggml_tensor *id_embeds_2 = ggml_mul_mat(ctx, visual_projection_2, shared_id_embeds); // [batch_size, seq_length, 1280]
        // id_embeds = id_embeds.view(b, num_inputs, 1, -1)
        // id_embeds_2 = id_embeds_2.view(b, num_inputs, 1, -1)    

        // id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1)
        // print_ggml_tensor(id_embeds, true, "id_embeds");
        // print_ggml_tensor(id_embeds_2, true, "id_embeds_2");
       
        id_embeds = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 2, 0, 1, 3));
        id_embeds_2 = ggml_cont(ctx, ggml_permute(ctx, id_embeds_2, 2, 0, 1, 3));
        // print_ggml_tensor(id_embeds, true, "id_embeds_after_perm");
        // print_ggml_tensor(id_embeds_2, true, "id_embeds_2_after_perm");

        id_embeds = ggml_concat(ctx, id_embeds, id_embeds_2); // [batch_size, seq_length, 1, 2048] check whether concat at dim 2 is right
        // print_ggml_tensor(id_embeds, true, "id_embeds_after_cat");
        

        id_embeds = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 1, 2, 0, 3));
        // print_ggml_tensor(id_embeds, true, "id_embeds_after_cat+perm");
        

        struct ggml_tensor * updated_prompt_embeds = fuse_module.forward(ctx,
                                          prompt_embeds, id_embeds,
                                          class_tokens_mask,
                                          class_tokens_mask_pos,
                                          left, right);
        // print_ggml_tensor(updated_prompt_embeds, true, "updated_prompt_embeds_returned");

        return updated_prompt_embeds;

    }

     struct ggml_cgraph* build_graph(struct ggml_allocr* allocr, 
                                     struct ggml_tensor* id_pixel_values,
                                     struct ggml_tensor* prompt_embeds,
                                     std::vector<bool> &class_tokens_mask
                              ) {
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };

        struct ggml_context* ctx0 = ggml_init(params);

        struct ggml_cgraph* gf = ggml_new_graph(ctx0);   

        int64_t hidden_size = prompt_embeds->ne[0];
        int64_t seq_length  = prompt_embeds->ne[1];
        ggml_type type = GGML_TYPE_F32;

        struct ggml_tensor* id_pixel_values_d = ggml_dup_tensor(ctx0, id_pixel_values);
        ggml_allocr_alloc(allocr, id_pixel_values_d);
        struct ggml_tensor* prompt_embeds_d = ggml_dup_tensor(ctx0, prompt_embeds);
        ggml_allocr_alloc(allocr, prompt_embeds_d);
        struct ggml_tensor* class_tokens_mask_d = ggml_new_tensor_1d(ctx0, type, class_tokens_mask.size());
        ggml_allocr_alloc(allocr, class_tokens_mask_d);

        


        std::vector<float> ctm;
        std::vector<ggml_fp16_t> ctmf16;
        std::vector<int> ctmpos;
        struct ggml_tensor*  left  = NULL;
        struct ggml_tensor*  right = NULL;
        for(int i=0; i < class_tokens_mask.size(); i++){
            if(class_tokens_mask[i]){
                ctm.push_back(0.f); // here use 0.f instead of 1.f to make a scale mask
                ctmf16.push_back(ggml_fp32_to_fp16(0.f)); // here use 0.f instead of 1.f to make a scale mask
                ctmpos.push_back(i);
                // printf("push %d, \n", i);
            }else{
                ctm.push_back(1.f); // here use 1.f instead of 0.f to make a scale mask
                ctmf16.push_back(ggml_fp32_to_fp16(1.f)); // here use 0.f instead of 1.f to make a scale mask
            }  
        } 
        if(ctmpos[0] > 0){
            left = ggml_new_tensor_3d(ctx0, type, hidden_size, 1, ctmpos[0]);
            ggml_allocr_alloc(allocr, left);
        }
        if(ctmpos[ctmpos.size()-1] < seq_length - 1){
            right = ggml_new_tensor_3d(ctx0, type,
                   hidden_size, 1, seq_length-ctmpos[ctmpos.size()-1]-1);
            ggml_allocr_alloc(allocr, right);
        }
        struct ggml_tensor*  class_tokens_mask_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ctmpos.size());
        ggml_allocr_alloc(allocr, class_tokens_mask_pos);

        const int image_size = id_pixel_values->ne[0]; 
        int batch_size = id_pixel_values->ne[3];
        const int num_patches = ((image_size / vision_model.patch_size) * (image_size / vision_model.patch_size));
        const int num_positions = num_patches + 1;

        struct ggml_tensor * cls = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch_size);

        struct ggml_tensor * class_embedding_temp = ggml_new_tensor_4d(ctx0, type, 
                                       vision_model.hidden_size, batch_size, 1, 1);
        struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_positions);
        struct ggml_tensor * cast_f32_class = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_model.hidden_size);
        ggml_allocr_alloc(allocr, cls);
        ggml_allocr_alloc(allocr, class_embedding_temp);
        ggml_allocr_alloc(allocr, positions);
        ggml_allocr_alloc(allocr, cast_f32_class);



        if (!ggml_allocr_is_measure(allocr)) {
            ggml_backend_tensor_set(id_pixel_values_d, id_pixel_values->data, 0,  ggml_nbytes(id_pixel_values));
            ggml_backend_tensor_set(prompt_embeds_d, prompt_embeds->data, 0,  ggml_nbytes(prompt_embeds));
            if(type == GGML_TYPE_F16)
                ggml_backend_tensor_set(class_tokens_mask_d, ctmf16.data(), 0,  ggml_nbytes(class_tokens_mask_d));
            else
                ggml_backend_tensor_set(class_tokens_mask_d, ctm.data(), 0,  ggml_nbytes(class_tokens_mask_d));
            std::vector<int> cls_h;
            for (int b = 0; b < batch_size; b++) {
                cls_h.push_back(b * num_positions);
            }
            std::vector<int> pos;
            for (int i = 0; i < num_positions; i++) {
                pos.push_back(i);
            }
            ggml_backend_tensor_set(cls, cls_h.data(), 0, ggml_nbytes(cls));
            ggml_backend_tensor_set(positions, pos.data(), 0, ggml_nbytes(positions));
            ggml_backend_tensor_set(class_tokens_mask_pos, ctmpos.data(), 0, ggml_nbytes(class_tokens_mask_pos));
            std::vector<float> zeros;
            for (int i = 0; i < hidden_size; i++) {
                zeros.push_back(0.f);
            }
            ggml_backend_tensor_set(cast_f32_class, zeros.data(), 0, ggml_nbytes(cast_f32_class));
            if(left){
                if(type == GGML_TYPE_F16){
                    std::vector<ggml_fp16_t> zeros(ggml_nelements(left), ggml_fp32_to_fp16(0.f));
                    ggml_backend_tensor_set(left, zeros.data(), 0, ggml_nbytes(left));
                }else{
                    std::vector<float> zeros(ggml_nelements(left), 0.f);
                    ggml_backend_tensor_set(left, zeros.data(), 0, ggml_nbytes(left));
                }
            }
            if(right){
                if(type == GGML_TYPE_F16){
                    std::vector<ggml_fp16_t> zeros(ggml_nelements(right), ggml_fp32_to_fp16(0.f));
                    ggml_backend_tensor_set(right, zeros.data(), 0, ggml_nbytes(right));
                }else{
                    std::vector<float> zeros(ggml_nelements(right), 0.f);
                    ggml_backend_tensor_set(right, zeros.data(), 0, ggml_nbytes(right));
                }
            }
        }     
        struct ggml_tensor*  updated_prompt_embeds = forward(ctx0, 
                                                            id_pixel_values_d, 
                                                            prompt_embeds_d,
                                                            class_tokens_mask_d,
                                                            class_tokens_mask_pos,
                                                            cls,
                                                            class_embedding_temp,
                                                            positions,
                                                            left, right
                                                            );
        // print_ggml_tensor(updated_prompt_embeds, true, "updated_prompt_embeds_returned_forward");
        ggml_build_forward_expand(gf, updated_prompt_embeds);
        // ggml_graph_dump_dot(gf, NULL, "id_encoder.dot");
        ggml_free(ctx0);

        return gf;
    }

    void alloc_compute_buffer(ggml_context* work_ctx, 
                              struct ggml_tensor* id_pixel_values,
                              struct ggml_tensor* prompt_embeds,
                              std::vector<bool> &class_tokens_mask) {
        auto get_graph = [&]() -> struct ggml_cgraph* {            
            
            return build_graph(compute_allocr, id_pixel_values, prompt_embeds, class_tokens_mask);
        };
        GGMLModule::alloc_compute_buffer(get_graph);
    }

    void compute(const int n_threads,
                 struct ggml_tensor* id_pixel_values,
                 struct ggml_tensor* prompt_embeds,
                 std::vector<bool> &class_tokens_mask,                 
                 struct ggml_tensor* updated_prompt_embeds) {
        
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(compute_allocr, id_pixel_values, prompt_embeds, class_tokens_mask);
        };

        GGMLModule::compute(get_graph, n_threads, updated_prompt_embeds);
        
    }

};



#endif  // __PMI_HPP__

