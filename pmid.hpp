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
        print_ggml_tensor(image_token_embeds, true, "image_token_embeds");
        print_ggml_tensor(valid_id_embeds, true, "valid_id_embeds");
        struct ggml_tensor *stacked_id_embeds = fuse_fn(ctx, image_token_embeds, valid_id_embeds);
        print_ggml_tensor(stacked_id_embeds, true, "stacked_id_embeds_before_concat");

        stacked_id_embeds = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 0, 2, 1, 3));
        print_ggml_tensor(stacked_id_embeds, true, "stacked_id_embeds_after_permute");
        if(left && right){
            print_ggml_tensor(left, true, "left");
            print_ggml_tensor(right, true, "right");
            stacked_id_embeds = ggml_concat(ctx, left, stacked_id_embeds);
            stacked_id_embeds = ggml_concat(ctx, stacked_id_embeds, right);
        }else if(left){
            stacked_id_embeds = ggml_concat(ctx, left, stacked_id_embeds);
        }else if(right){
            stacked_id_embeds = ggml_concat(ctx, stacked_id_embeds, right);
        }
        print_ggml_tensor(stacked_id_embeds, true, "stacked_id_embeds_after_concat");
        stacked_id_embeds = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 0, 2, 1, 3));
        print_ggml_tensor(stacked_id_embeds, true, "stacked_id_embeds_after_permute_2");
        // print_ggml_tensor(class_tokens_mask, true, "class_tokens_mask");
        // print_ggml_tensor(prompt_embeds, true, "prompt_embeds");
        // assert class_tokens_mask.sum() == stacked_id_embeds.shape[0], f"{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}"
        // prompt_embeds.masked_scatter_(class_tokens_mask[:, None], stacked_id_embeds.to(prompt_embeds.dtype))
        // struct ggml_tensor *prompt_embeds_perm = ggml_cont(ctx, ggml_permute(ctx, prompt_embeds, 1, 0, 2, 3));
        // struct ggml_tensor *prompt_embeds_perm = ggml_cont(ctx, ggml_transpose(ctx, prompt_embeds));
        print_ggml_tensor(prompt_embeds, true, "prompt_embeds_perm");
        // class_tokens_mask = ggml_repeat(ctx, class_tokens_mask, prompt_embeds_perm);
        class_tokens_mask = ggml_cont(ctx, ggml_transpose(ctx, class_tokens_mask));
        print_ggml_tensor(class_tokens_mask, true, "class_tokens_mask_repeat");
        // class_tokens_mask =  ggml_cont(ctx, ggml_transpose(ctx, class_tokens_mask));
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
        print_ggml_tensor(id_embeds, true, "id_embeds_after_cat");
        

        id_embeds = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 1, 2, 0, 3));
        print_ggml_tensor(id_embeds, true, "id_embeds_after_cat+perm");
        

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
        ggml_allocr_alloc(allocr, cls);
        ggml_allocr_alloc(allocr, class_embedding_temp);
        ggml_allocr_alloc(allocr, positions);



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



#define PM_LORA_GRAPH_SIZE 10240

struct PhotoMakerLoraModel : public GGMLModule {
    float multiplier = 1.0f;
    std::map<std::string, struct ggml_tensor*> lora_tensors;
    std::string file_path;
     int in_channels                        = 4;
    // ModelLoader model_loader;
    bool load_failed = false;
    int model_channels                     = 320;  // only for SDXL
    int num_heads                          = -1;   // only for SDXL
    int num_head_channels                  = 64;   // only for SDXL
    int context_dim                        = 2048; // only for SDXL

    std::vector<int>  transformer_depth     = {1, 2, 10}; // only for SDXL
    std::vector<int>  channel_mult          = {1, 2, 4}; // only for SDXL

    struct SpatialTransformerLora{
        int in_channels;        // mult * model_channels
        int n_head;             // num_heads
        int d_head;             // in_channels // n_heads
        int depth       = 1;    // 1
        int context_dim = 768;  // hidden_size, 1024 for VERSION_2_x
          

        struct TransformerLora {
        
            // attn1
            struct ggml_tensor* attn1_q_w_up;  // [in_channels, in_channels]
            struct ggml_tensor* attn1_q_w_dn;  // [in_channels, in_channels]
            struct ggml_tensor* attn1_k_w_up;  // [in_channels, in_channels]
            struct ggml_tensor* attn1_k_w_dn;  // [in_channels, in_channels]
            struct ggml_tensor* attn1_v_w_up;  // [in_channels, in_channels]
            struct ggml_tensor* attn1_v_w_dn;  // [in_channels, in_channels]

            struct ggml_tensor* attn1_out_w_up;  // [in_channels, in_channels]
            struct ggml_tensor* attn1_out_w_dn;  // [in_channels, in_channels]

            // attn2
            struct ggml_tensor* attn2_q_w_up;  // [in_channels, in_channels]
            struct ggml_tensor* attn2_q_w_dn;  // [in_channels, in_channels]
            struct ggml_tensor* attn2_k_w_up;  // [in_channels, context_dim]
            struct ggml_tensor* attn2_k_w_dn;  // [in_channels, context_dim]
            struct ggml_tensor* attn2_v_w_up;  // [in_channels, context_dim]
            struct ggml_tensor* attn2_v_w_dn;  // [in_channels, context_dim]

            struct ggml_tensor* attn2_out_w_up;  // [in_channels, in_channels]
            struct ggml_tensor* attn2_out_w_dn;  // [in_channels, in_channels]



        };

        std::vector<TransformerLora> transformers;

        SpatialTransformerLora(int depth = -1)
        : depth(depth) {
            if(depth > 0)
                transformers.resize(depth);
        }

        int get_num_tensors() {
            return depth * 16;
        }

        size_t calculate_mem_size(ggml_type wtype) {
            size_t mem_size = 0;            
            // transformer
            for (auto& transformer : transformers) {
                mem_size += 12 * ggml_row_size(wtype, in_channels * in_channels);      // attn1_q/k/v/out_w attn2_q/out_w
                mem_size +=  4 * ggml_row_size(wtype, in_channels * context_dim);      // attn2_k/v_w
            }
            return mem_size;
        }

       void init_params(struct ggml_context* ctx, ggml_type wtype) {
            // transformer
            for (auto& transformer : transformers) {

                transformer.attn1_q_w_up = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
                transformer.attn1_q_w_dn = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
                transformer.attn1_k_w_up = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
                transformer.attn1_k_w_dn = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
                transformer.attn1_v_w_up = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
                transformer.attn1_v_w_dn = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);

                transformer.attn1_out_w_up = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
                transformer.attn1_out_w_dn = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);

            

                transformer.attn2_q_w_up = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
                transformer.attn2_q_w_dn = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
                transformer.attn2_k_w_up = ggml_new_tensor_2d(ctx, wtype, context_dim, in_channels);
                transformer.attn2_k_w_dn = ggml_new_tensor_2d(ctx, wtype, context_dim, in_channels);
                transformer.attn2_v_w_up = ggml_new_tensor_2d(ctx, wtype, context_dim, in_channels);
                transformer.attn2_v_w_dn = ggml_new_tensor_2d(ctx, wtype, context_dim, in_channels);

                transformer.attn2_out_w_up = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
                transformer.attn2_out_w_dn = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
            
            }
        }


    };


    SpatialTransformerLora down_blocks[3][2];
    SpatialTransformerLora mid_block;
    SpatialTransformerLora up_blocks[3][3];



    PhotoMakerLoraModel(const std::string file_path = "")
        : file_path(file_path) {
        name = "photomaker lora";
        // if (!model_loader.init_from_file(file_path)) {
        //     load_failed = true;
        // }
        int ch = model_channels;
        for(int i = 0; i < 3; i++){
            int mult = channel_mult[i];
            if(i == 0) continue;
            for(int j = 0; j < 2; j++){
                ch = mult * model_channels;
                int n_head = num_heads;
                int d_head = ch / num_heads;
                if (num_head_channels != -1) {
                    d_head = num_head_channels;
                    n_head = ch / d_head;
                }
                down_blocks[i][j] = SpatialTransformerLora(transformer_depth[i]);
                down_blocks[i][j].in_channels = ch;
                down_blocks[i][j].n_head      = n_head;
                down_blocks[i][j].d_head      = d_head;
                down_blocks[i][j].context_dim = context_dim;
            }
        }
        int n_head = num_heads;
        int d_head = ch / num_heads;
        if (num_head_channels != -1) {
            d_head = num_head_channels;
            n_head = ch / d_head;
        }
        mid_block = SpatialTransformerLora(transformer_depth[transformer_depth.size()-1]);
        mid_block.in_channels = ch;
        mid_block.n_head      = n_head;
        mid_block.d_head      = d_head;
        mid_block.context_dim = context_dim;


        for(int i = 2; i >= 0 ; i--){
            int mult = channel_mult[i];
            if(i == 0) continue;
            for(int j = 0; j < 2; j++){
                ch = mult * model_channels;
                int n_head = num_heads;
                int d_head = ch / num_heads;
                if (num_head_channels != -1) {
                    d_head = num_head_channels;
                    n_head = ch / d_head;
                }
                up_blocks[i][j] = SpatialTransformerLora(transformer_depth[i]);
                up_blocks[i][j].in_channels = ch;
                up_blocks[i][j].n_head      = n_head;
                up_blocks[i][j].d_head      = d_head;
                up_blocks[i][j].context_dim = context_dim;
            }
        }
      
        
    }

    size_t get_num_tensors() {
        size_t num_tensors = 0;         
        // return model_loader.cal_mem_size(NULL);
        for(int i = 0; i < 3; i++){
            if(i == 0) continue;
            for(int j = 0; j < 2; j++)               
                num_tensors += down_blocks[i][j].get_num_tensors();
        }
        for(int i = 2; i >= 0 ; i--){
            if(i == 0) continue;
            for(int j = 0; j < 2; j++)
                num_tensors += up_blocks[i][j].get_num_tensors();
        }
        num_tensors += mid_block.get_num_tensors();
        return num_tensors;     
               
    }

    size_t calculate_mem_size(ggml_type wtype) {
        size_t mem_size = 0;         
        // return model_loader.cal_mem_size(NULL);
        for(int i = 0; i < 3; i++){
            if(i == 0) continue;
            for(int j = 0; j < 2; j++)               
                mem_size += down_blocks[i][j].calculate_mem_size(wtype);
        }
        for(int i = 2; i >= 0 ; i--){
            if(i == 0) continue;
            for(int j = 0; j < 2; j++)
                mem_size += up_blocks[i][j].calculate_mem_size(wtype);
               
        }
        mem_size += mid_block.calculate_mem_size(wtype);
        return mem_size;       
    }


    void init_params(ggml_type wtype){
        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);

        for(int i = 0; i < 3; i++){
            if(i == 0) continue;
            for(int j = 0; j < 2; j++)               
                down_blocks[i][j].init_params(params_ctx, wtype);
        }
        for(int i = 2; i >= 0 ; i--){
            if(i == 0) continue;
            for(int j = 0; j < 2; j++)
                up_blocks[i][j].init_params(params_ctx, wtype);
        }

        mid_block.init_params(params_ctx, wtype);

        // alloc all tensors linked to this context
        for (struct ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != NULL; t = ggml_get_next_tensor(params_ctx, t)) {
            if (t->data == NULL) {
                ggml_allocr_alloc(alloc, t);
            }
        }
        
        ggml_allocr_free(alloc); 

    }

    bool load_from_file(ggml_backend_t backend) {
        if (!alloc_params_buffer(backend)) {
            return false;
        }
        LOG_INFO("loading LoRA from '%s'", file_path.c_str());

        if (load_failed) {
            LOG_ERROR("init lora model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);

        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            const std::string& name = tensor_storage.name;

            struct ggml_tensor* real = ggml_new_tensor(params_ctx, tensor_storage.type, tensor_storage.n_dims, tensor_storage.ne);
            ggml_allocr_alloc(alloc, real);

            *dst_tensor = real;

            lora_tensors[name] = real;
            return true;
        };

        // model_loader.load_tensors(on_new_tensor_cb, backend);

        LOG_DEBUG("finished loaded lora");
        ggml_allocr_free(alloc);
        return true;
    }

    struct ggml_cgraph* build_graph(std::map<std::string, struct ggml_tensor*> model_tensors) {
        // make a graph to compute all lora, expected lora and models tensors are in the same backend
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * PM_LORA_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        struct ggml_context* ctx0 = ggml_init(params);
        struct ggml_cgraph* gf    = ggml_new_graph_custom(ctx0, PM_LORA_GRAPH_SIZE, false);

        std::set<std::string> applied_lora_tensors;
        for (auto it : model_tensors) {
            std::string k_tensor       = it.first;
            struct ggml_tensor* weight = model_tensors[it.first];

            size_t k_pos = k_tensor.find(".weight");
            if (k_pos == std::string::npos) {
                continue;
            }
            k_tensor = k_tensor.substr(0, k_pos);
            replace_all_chars(k_tensor, '.', '_');
            std::string lora_up_name   = "lora." + k_tensor + ".lora_up.weight";
            std::string lora_down_name = "lora." + k_tensor + ".lora_down.weight";
            std::string alpha_name     = "lora." + k_tensor + ".alpha";
            std::string scale_name     = "lora." + k_tensor + ".scale";

            ggml_tensor* lora_up   = NULL;
            ggml_tensor* lora_down = NULL;

            if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                lora_up = lora_tensors[lora_up_name];
            }

            if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                lora_down = lora_tensors[lora_down_name];
            }

            if (lora_up == NULL || lora_down == NULL) {
                continue;
            }

            applied_lora_tensors.insert(lora_up_name);
            applied_lora_tensors.insert(lora_down_name);
            applied_lora_tensors.insert(alpha_name);
            applied_lora_tensors.insert(scale_name);

            // calc_cale
            int64_t dim       = lora_down->ne[ggml_n_dims(lora_down) - 1];
            float scale_value = 1.0f;
            if (lora_tensors.find(scale_name) != lora_tensors.end()) {
                scale_value = ggml_backend_tensor_get_f32(lora_tensors[scale_name]);
            } else if (lora_tensors.find(alpha_name) != lora_tensors.end()) {
                float alpha = ggml_backend_tensor_get_f32(lora_tensors[alpha_name]);
                scale_value = alpha / dim;
            }
            scale_value *= multiplier;

            // flat lora tensors to multiply it
            int64_t lora_up_rows   = lora_up->ne[ggml_n_dims(lora_up) - 1];
            lora_up                = ggml_reshape_2d(ctx0, lora_up, ggml_nelements(lora_up) / lora_up_rows, lora_up_rows);
            int64_t lora_down_rows = lora_down->ne[ggml_n_dims(lora_down) - 1];
            lora_down              = ggml_reshape_2d(ctx0, lora_down, ggml_nelements(lora_down) / lora_down_rows, lora_down_rows);

            // ggml_mul_mat requires tensor b transposed
            lora_down                  = ggml_cont(ctx0, ggml_transpose(ctx0, lora_down));
            struct ggml_tensor* updown = ggml_mul_mat(ctx0, lora_up, lora_down);
            updown                     = ggml_cont(ctx0, ggml_transpose(ctx0, updown));
            updown                     = ggml_reshape(ctx0, updown, weight);
            GGML_ASSERT(ggml_nelements(updown) == ggml_nelements(weight));
            updown = ggml_scale_inplace(ctx0, updown, scale_value);
            ggml_tensor* final_weight;
            // if (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) {
            //     final_weight = ggml_new_tensor(ctx0, GGML_TYPE_F32, weight->n_dims, weight->ne);
            //     final_weight = ggml_cpy_inplace(ctx0, weight, final_weight);
            //     final_weight = ggml_add_inplace(ctx0, final_weight, updown);
            //     final_weight = ggml_cpy_inplace(ctx0, final_weight, weight);
            // } else {
            //     final_weight = ggml_add_inplace(ctx0, weight, updown);
            // }
            final_weight = ggml_add_inplace(ctx0, weight, updown);  // apply directly
            ggml_build_forward_expand(gf, final_weight);
        }

        for (auto& kv : lora_tensors) {
            if (applied_lora_tensors.find(kv.first) == applied_lora_tensors.end()) {
                LOG_WARN("unused lora tensor %s", kv.first.c_str());
            }
        }

        return gf;
    }

    void alloc_compute_buffer(std::map<std::string, struct ggml_tensor*> model_tensors) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(model_tensors);
        };
        GGMLModule::alloc_compute_buffer(get_graph);
    }

    void apply(std::map<std::string, struct ggml_tensor*> model_tensors, int n_threads) {
        alloc_compute_buffer(model_tensors);

        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(model_tensors);
        };
        GGMLModule::compute(get_graph, n_threads);
    }
};


#endif  // __PMI_HPP__

