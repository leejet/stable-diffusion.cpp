#ifndef __PMI_HPP__
#define __PMI_HPP__

#include "ggml_extend.hpp"

#include "clip.hpp"


struct FuseBlock : public GGMLBlock {
    // network hparams
    int in_dim;    
    int out_dim;  
    int hidden_dim;
    bool use_residue;

    // network params
    // in_layers

    // layer norm 
    // struct ggml_tensor* ln_w;  // [in_dim, ]
    // struct ggml_tensor* ln_b;  // [in_dim, ]

    // struct ggml_tensor* fc1_w;  // [in_dim, hidden_dim]
    // struct ggml_tensor* fc1_b;  // [in_dim, ]
    // struct ggml_tensor* fc2_w;  // [hidden_dim, out_dim ]
    // struct ggml_tensor* fc2_b;  // [hidden_dim, ]

public:

    FuseBlock(int i_d, int o_d, int h_d, bool use_residue = true)
        : in_dim(i_d), out_dim(o_d), hidden_dim(h_d), 
        use_residue(use_residue){
        // blocks["fc1"] = std::shared_ptr<GGMLBlock>(new Linear(d_model, intermediate_size));
        // blocks["fc2"] = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, d_model));
        blocks["fc1"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim,  true));
        blocks["fc2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, out_dim, true));
        blocks["layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(in_dim));
    }
    
    
    // size_t calculate_mem_size(ggml_type wtype) {
    //     size_t mem_size = 0;
    //     mem_size += 2 * ggml_row_size(wtype, in_dim);
    //     mem_size += ggml_row_size(wtype, in_dim*hidden_dim);
    //     mem_size += ggml_row_size(wtype, out_dim);
    //     mem_size += ggml_row_size(wtype, hidden_dim*out_dim);
    //     mem_size += ggml_row_size(wtype, hidden_dim);
        
    //     return mem_size;
    // }

    // void init_params(struct ggml_context* ctx, ggml_type wtype, ggml_allocr* alloc) {
    //     ln_w = ggml_new_tensor_1d(ctx, wtype, in_dim);
    //     ln_b = ggml_new_tensor_1d(ctx, wtype, in_dim);

    //     fc1_b = ggml_new_tensor_1d(ctx, wtype, hidden_dim);
    //     fc1_w = ggml_new_tensor_2d(ctx, wtype, in_dim, hidden_dim);
    //     fc2_b = ggml_new_tensor_1d(ctx, wtype, out_dim);
    //     fc2_w = ggml_new_tensor_2d(ctx, wtype, hidden_dim, out_dim);
    //     // alloc all tensors linked to this context
    //     for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
    //         if (t->data == NULL) {
    //             ggml_allocr_alloc(alloc, t);
    //         }
    //     }

    // }s

    // size_t get_num_tensors() {
    //     return 6;
    // }

    // void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
    //     tensors[prefix + "fc1.weight"] = fc1_w;
    //     tensors[prefix + "fc1.bias"]   = fc1_b;
    //     tensors[prefix + "fc2.weight"] = fc2_w;
    //     tensors[prefix + "fc2.bias"]   = fc2_b;
    //     tensors[prefix + "layernorm.weight"] = ln_w;
    //     tensors[prefix + "layernorm.bias"]   = ln_b;        
    // }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        // auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
        // auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);

        // x = fc1->forward(ctx, x);
        // if (use_gelu) {
        //     x = ggml_gelu_inplace(ctx, x);
        // } else {
        //     x = ggml_gelu_quick_inplace(ctx, x);
        // }
        // x = fc2->forward(ctx, x);
        // return x;

        // in_layers

        auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
        auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);
        auto layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);

        struct ggml_tensor* r = x;
        // x = ggml_nn_layer_norm(ctx, x, ln_w, ln_b);
        x = layer_norm->forward(ctx, x);
        // x = ggml_add(ctx, ggml_mul_mat(ctx, fc1_w, x),  fc1_b);
        x = fc1->forward(ctx, x);
        x = ggml_gelu_inplace(ctx, x);
        x = fc2->forward(ctx, x);
        // x = ggml_add(ctx, ggml_mul_mat(ctx, fc2_w, x),  fc2_b);
        if(use_residue)
            x = ggml_add(ctx, x, r);
        return x;
    }

};

struct FuseModule : public GGMLBlock{
    // network hparams
    int embed_dim;    
    
    // struct FuseBlock mlp1;
    // struct FuseBlock mlp2;
    // layer norm 
    // struct ggml_tensor* ln_w;  // [in_dim, ]
    // struct ggml_tensor* ln_b;  // [in_dim, ]


    // FuseModule(int imb_d): 
    //     embed_dim(imb_d), 
    //     mlp1(imb_d*2, imb_d, imb_d, false),
    //     mlp2(imb_d, imb_d, imb_d, true) {
    // }

public:

    FuseModule(int imb_d): 
        embed_dim(imb_d){
        blocks["mlp1"]   = std::shared_ptr<GGMLBlock>(new FuseBlock(imb_d*2, imb_d, imb_d, false));
        blocks["mlp2"]   = std::shared_ptr<GGMLBlock>(new FuseBlock(imb_d, imb_d, imb_d, true));
        blocks["layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(embed_dim));
    }


    // void init_params(struct ggml_context* ctx, ggml_type wtype, ggml_allocr* alloc) {                  
    //     ln_w = ggml_new_tensor_1d(ctx, wtype, embed_dim);
    //     ln_b = ggml_new_tensor_1d(ctx, wtype, embed_dim);
    //     // alloc all tensors linked to this context
    //     mlp1.init_params(ctx, wtype, alloc);
    //     mlp2.init_params(ctx, wtype, alloc);
    //     for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
    //         if (t->data == NULL) {
    //             ggml_allocr_alloc(alloc, t);
    //         }
    //     }
    // }

    // void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {

    //     tensors[prefix + "layer_norm.weight"] = ln_w;
    //     tensors[prefix + "layer_norm.bias"]   = ln_b;      
    //     mlp1.map_by_name(tensors, prefix + "mlp1.");
    //     mlp2.map_by_name(tensors, prefix + "mlp2.");

    // }

    // size_t get_num_tensors() {
    //     size_t n = mlp1.get_num_tensors();
    //     n += mlp2.get_num_tensors();
    //     n += 2;
    //     return n;
    // }

    // size_t calculate_mem_size(ggml_type wtype) {
    //     size_t mem_size = mlp1.calculate_mem_size(wtype);
    //     mem_size += mlp2.calculate_mem_size(wtype);
    //     mem_size += 2 * ggml_row_size(wtype, embed_dim);        
    //     return mem_size;
    // }

    struct ggml_tensor* fuse_fn(struct ggml_context* ctx, 
                                struct ggml_tensor* prompt_embeds, 
                                struct ggml_tensor* id_embeds) {


        auto mlp1 = std::dynamic_pointer_cast<FuseBlock>(blocks["mlp1"]);
        auto mlp2 = std::dynamic_pointer_cast<FuseBlock>(blocks["mlp2"]);    
        auto layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);                        

        auto prompt_embeds0 = ggml_cont(ctx, ggml_permute(ctx, prompt_embeds, 2, 0, 1, 3));
        auto id_embeds0     = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 2, 0, 1, 3));
        // concat is along dim 2   
        auto stacked_id_embeds = ggml_concat(ctx, prompt_embeds0, id_embeds0); 
        stacked_id_embeds = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 1, 2, 0, 3));

        // stacked_id_embeds = mlp1.forward(ctx, stacked_id_embeds);
        // stacked_id_embeds = ggml_add(ctx, stacked_id_embeds, prompt_embeds);
        // stacked_id_embeds = mlp2.forward(ctx, stacked_id_embeds);
        // stacked_id_embeds = ggml_nn_layer_norm(ctx, stacked_id_embeds, ln_w, ln_b);

        stacked_id_embeds = mlp1->forward(ctx, stacked_id_embeds);
        stacked_id_embeds = ggml_add(ctx, stacked_id_embeds, prompt_embeds);
        stacked_id_embeds = mlp2->forward(ctx, stacked_id_embeds);
        stacked_id_embeds = layer_norm->forward(ctx, stacked_id_embeds);

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

        struct ggml_tensor * valid_id_embeds = id_embeds;
        // # slice out the image token embeddings
        struct ggml_tensor * image_token_embeds = ggml_get_rows(ctx, prompt_embeds, class_tokens_mask_pos);

        struct ggml_tensor *stacked_id_embeds = fuse_fn(ctx, image_token_embeds, valid_id_embeds);

        stacked_id_embeds = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 0, 2, 1, 3));
        if(left && right){
            stacked_id_embeds = ggml_concat(ctx, left, stacked_id_embeds);
            stacked_id_embeds = ggml_concat(ctx, stacked_id_embeds, right);
        }else if(left){
            stacked_id_embeds = ggml_concat(ctx, left, stacked_id_embeds);
        }else if(right){
            stacked_id_embeds = ggml_concat(ctx, stacked_id_embeds, right);
        }
        stacked_id_embeds = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 0, 2, 1, 3));
        class_tokens_mask = ggml_cont(ctx, ggml_transpose(ctx, class_tokens_mask));
        class_tokens_mask = ggml_repeat(ctx, class_tokens_mask, prompt_embeds);
        prompt_embeds = ggml_mul(ctx, prompt_embeds, class_tokens_mask);
        struct ggml_tensor * updated_prompt_embeds = ggml_add(ctx, prompt_embeds, stacked_id_embeds);
        return updated_prompt_embeds;

    }
    
};


struct VisualProjection : public GGMLBlock{

public:
    int32_t hidden_size    = 1024;
    int32_t projection_dim = 1280;

    VisualProjection(int32_t h, int32_t p) :
        hidden_size(h), projection_dim(p){      
        blocks["visual_projection_2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, projection_dim, false));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, 
                           struct ggml_tensor* pixel_values){

        auto visual_projection_2 = std::dynamic_pointer_cast<Linear>(blocks["visual_projection_2"]);
        auto x      = visual_projection_2->forward(ctx, pixel_values);        // [N, projection_dim]
        return x;  // [N, projection_dim]
    }


};


struct PhotoMakerIDEncoder : public GGMLModule {

public:    
    SDVersion version = VERSION_XL;    
    CLIPVisionModel vision_model;
    FuseModule fuse_module;
    // struct ggml_tensor* visual_projection_2; 
    VisualProjection visual_projection_2;
    float style_strength;

public:    
    PhotoMakerIDEncoder(ggml_backend_t backend, ggml_type wtype, 
                      SDVersion version = VERSION_XL, float sty = 20.f)
        : GGMLModule(backend, wtype), 
          version(version), 
        fuse_module(2048),
        visual_projection_2(1024, 1280),
        style_strength(sty){
        // vision_model = CLIPVisionModel();
        vision_model.init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "pmid";
    }

    // void init_params(ggml_type wtype) {               
    //     ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);
    //     vision_model.init_params(params_ctx, backend, wtype, alloc);
    //     fuse_module.init_params(params_ctx, wtype, alloc);
    //     visual_projection_2 = ggml_new_tensor_2d(params_ctx, wtype, 1024, 1280); // python [1024, 1280]        
    //     ggml_allocr_alloc(alloc, visual_projection_2);
    //     ggml_allocr_free(alloc); 
    // }

    // void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
    //     vision_model.map_by_name(tensors, prefix + "vision_model.", prefix);
    //     fuse_module.map_by_name(tensors, prefix + "fuse_module.");
    //     tensors[prefix + "visual_projection_2.weight"] = visual_projection_2;        
    // }

    // size_t calculate_mem_size() {        

    //     wtype = GGML_TYPE_F32;

    //     size_t mem_size = vision_model.calculate_mem_size(wtype);
    //     mem_size += fuse_module.calculate_mem_size(wtype);
        
    //     mem_size += ggml_row_size(wtype, 1280*1024);
              
    //     return mem_size;
    // }

    // size_t get_num_tensors() {
    //     size_t num_tensors = (3 + 2 + 37 * vision_model.num_hidden_layers);
    //     num_tensors += fuse_module.get_num_tensors() + 1;
    //     return num_tensors;
    // }

    size_t get_params_mem_size() {
        size_t params_mem_size = vision_model.get_params_mem_size();        
        params_mem_size += fuse_module.get_params_mem_size();
        params_mem_size += ggml_row_size(wtype, 1280*1024);
        return params_mem_size;
    }

    size_t get_params_num() {
        size_t params_num = vision_model.get_params_num();
        
        params_num += fuse_module.get_params_num() + 1;
        
        return params_num;
    }


    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        vision_model.get_param_tensors(tensors, prefix);
        fuse_module.get_param_tensors(tensors, prefix);
        visual_projection_2.get_param_tensors(tensors, prefix);
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

        // struct ggml_tensor *shared_id_embeds = vision_model.forward(ctx, 
        //                                                      id_pixel_values,
        //                                                      cls,
        //                                                      class_embedding_temp,
        //                                                      positions
        //                                                      ); // [batch_size, seq_length, hidden_size]
        struct ggml_tensor *shared_id_embeds = vision_model.forward(ctx, 
                                        id_pixel_values, false);     // [batch_size, seq_length, hidden_size]


        struct ggml_tensor *id_embeds = vision_model.visual_project(ctx, shared_id_embeds); // [batch_size, seq_length, proj_dim(768)]
        
        // struct ggml_tensor *id_embeds_2 = ggml_mul_mat(ctx, visual_projection_2, shared_id_embeds); // [batch_size, seq_length, 1280]
        struct ggml_tensor *id_embeds_2 = visual_projection_2.forward(ctx, shared_id_embeds); // [batch_size, seq_length, 1280]


       
        id_embeds = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 2, 0, 1, 3));
        id_embeds_2 = ggml_cont(ctx, ggml_permute(ctx, id_embeds_2, 2, 0, 1, 3));

        id_embeds = ggml_concat(ctx, id_embeds, id_embeds_2); // [batch_size, seq_length, 1, 2048] check whether concat at dim 2 is right
        id_embeds = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 1, 2, 0, 3));

        struct ggml_tensor * updated_prompt_embeds = fuse_module.forward(ctx,
                                          prompt_embeds, id_embeds,
                                          class_tokens_mask,
                                          class_tokens_mask_pos,
                                          left, right);

        return updated_prompt_embeds;

    }

     struct ggml_cgraph* build_graph(struct ggml_allocr* allocr, 
                                     struct ggml_tensor* id_pixel_values,
                                     struct ggml_tensor* prompt_embeds,
                                     std::vector<bool> &class_tokens_mask
                              ) {
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        // static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        // static std::vector<uint8_t> buf(buf_size);

        // struct ggml_init_params params = {
        //     /*.mem_size   =*/buf_size,
        //     /*.mem_buffer =*/buf.data(),
        //     /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        // };

        // struct ggml_context* ctx0 = ggml_init(params);

        // struct ggml_cgraph* gf = ggml_new_graph(ctx0);   

        ggml_context *ctx0 = compute_ctx;

        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        int64_t hidden_size = prompt_embeds->ne[0];
        int64_t seq_length  = prompt_embeds->ne[1];
        ggml_type type = GGML_TYPE_F32;

        // struct ggml_tensor* id_pixel_values_d = ggml_dup_tensor(ctx0, id_pixel_values);
        // ggml_allocr_alloc(allocr, id_pixel_values_d);
        // struct ggml_tensor* prompt_embeds_d = ggml_dup_tensor(ctx0, prompt_embeds);
        // ggml_allocr_alloc(allocr, prompt_embeds_d);        
        struct ggml_tensor* class_tokens_mask_d = ggml_new_tensor_1d(ctx0, type, class_tokens_mask.size());
        ggml_allocr_alloc(allocr, class_tokens_mask_d);


        struct ggml_tensor* id_pixel_values_d = to_backend(id_pixel_values);
        struct ggml_tensor* prompt_embeds_d   = to_backend(prompt_embeds);

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
            // ggml_backend_tensor_set(id_pixel_values_d, id_pixel_values->data, 0,  ggml_nbytes(id_pixel_values));
            // ggml_backend_tensor_set(prompt_embeds_d, prompt_embeds->data, 0,  ggml_nbytes(prompt_embeds));
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
        ggml_build_forward_expand(gf, updated_prompt_embeds);
        // ggml_free(ctx0);

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
                 struct ggml_tensor** updated_prompt_embeds,
                 ggml_context* output_ctx) {
        
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(compute_allocr, id_pixel_values, prompt_embeds, class_tokens_mask);
        };

        // GGMLModule::compute(get_graph, n_threads, updated_prompt_embeds);
        GGMLModule::compute(get_graph, n_threads, true, updated_prompt_embeds, output_ctx);
        
    }

};



#define PM_LORA_GRAPH_SIZE 10240

struct PhotoMakerLoraModel : public GGMLModule {
    float multiplier = 1.f;
    SDVersion version;
    std::map<std::string, struct ggml_tensor*> lora_tensors;
    std::vector<std::string> lora_tensors_to_be_ignored;
    std::string file_path;   
    ModelLoader model_loader;
    bool load_failed = false;
   
    // PhotoMakerLoraModel()       
    // {}

public:

    PhotoMakerLoraModel(ggml_backend_t backend, ggml_type wtype,
                       SDVersion version = VERSION_XL, 
                       const std::string file_path = "")
        : GGMLModule(backend, wtype), version(version), file_path(file_path) {
        // name = "photomaker lora";
        if (!model_loader.init_from_file(file_path, "pmid.")) {
            load_failed = true;
        }
    }

    std::string get_desc() {
        return "pmid lora";
    }

    size_t get_params_num() {
        return PM_LORA_GRAPH_SIZE;
    }

    size_t get_params_mem_size() {
        return model_loader.get_params_mem_size(NULL);
    }


    // size_t get_num_tensors() {
    //     return PM_LORA_GRAPH_SIZE;
    // }

    // size_t calculate_mem_size() {
    //     return model_loader.cal_mem_size(NULL);
    // }
   

    bool load_from_file(ggml_backend_t backend) {
        // if (!alloc_params_buffer(backend)) {
        //     return false;
        // }
        LOG_INFO("loading LoRA from '%s'", file_path.c_str());

        if (load_failed) {
            LOG_ERROR("init lora model loader from file failed: '%s'", file_path.c_str());
            return false;
        }
        alloc_params_buffer();

        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);

        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            std::string name = tensor_storage.name;
            // LOG_INFO("loading LoRA tesnor '%s'", name.c_str());
            if (!starts_with(name, "pmid.unet")){
                // LOG_INFO("skipping LoRA tesnor '%s'", name.c_str());
                return true;
            }

               
            // LOG_INFO("loading LoRA tesnor '%s'", name.c_str());
            struct ggml_tensor* real = ggml_new_tensor(params_ctx, tensor_storage.type, tensor_storage.n_dims, tensor_storage.ne);
            ggml_allocr_alloc(alloc, real);

            *dst_tensor = real;
            lora_tensors_to_be_ignored.push_back(name);
            size_t k_pos = name.find(".processor");
            if(k_pos != std::string::npos)
               name.replace(k_pos, strlen(".processor"), "");
            // if(starts_with(name, "pmid.unet.down_blocks.2.attentions.1.transformer_blocks.9.attn2"))
            //     print_ggml_tensor(real, true, name.c_str());
            lora_tensors[name] = real;
            return true;
        };

        model_loader.load_tensors(on_new_tensor_cb, backend);

        LOG_DEBUG("finished loaded lora");
        ggml_allocr_free(alloc);
        return true;
    }

    std::pair<int,int> find_ij0(int n){
        int i, j;
        for(i = 0; i < 3; i++){
            for(j = 0; j < 2; j++){
                if((i*3+j+1) == n)
                    return {i,j};
            }
        }
        return {-1, -1};
    }

    std::pair<int,int> find_ij(int n){
        int i, j;
        for(i = 0; i < 2; i++){
            for(j = 0; j < 3; j++){
                if((i*3+j) == n)
                    return {i,j};
            }
        }
        return {-1, -1};
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
            std::string full_name = k_tensor;
            // size_t k_pos = k_tensor.find(".weight");
            size_t k_pos = k_tensor.find(".attn1");
            if (k_pos == std::string::npos) {
                k_pos = k_tensor.find(".attn2");
                if (k_pos == std::string::npos) {
                    continue;
                }
            }
            if(ends_with(k_tensor, "bias"))
               continue;
            int block_kind = -1;
            int block_id = -1;
            if ((k_pos = k_tensor.find("input_blocks")) != std::string::npos) {
                block_id = atoi(k_tensor.substr(k_pos+strlen("input_blocks")+1).c_str());
                block_kind = 0;  // input ->  down block

            }else if ((k_pos = k_tensor.find("output_blocks")) != std::string::npos) {
                block_id = atoi(k_tensor.substr(k_pos+strlen("output_blocks")+1).c_str());
                block_kind = 1;  // output ->  up block
            }else{
                k_pos = k_tensor.find("transformer_blocks");
                block_id = atoi(k_tensor.substr(k_pos-4,1).c_str());
                block_kind = 2;  // middle block
            }


            std::string lora_up_name;
            std::string lora_down_name;
            std::string prefix = "pmid.unet";
            if (block_kind == 0){
                prefix = prefix + ".down_blocks";
                k_pos = k_tensor.find(".weight");
                k_tensor = k_tensor.substr(0, k_pos);
                k_pos = k_tensor.find("transformer_blocks");
                k_tensor = k_tensor.substr(k_pos);
                if(ends_with(k_tensor, "0")){
                    k_tensor = k_tensor.substr(0, k_tensor.length()-2);
                }
                std::pair<int, int> ij = find_ij0(block_id);
                if(ij.first == -1)
                   continue;
                prefix = prefix + "."+std::to_string(ij.first)+".attentions."+std::to_string(ij.second)+".";
                lora_up_name     = prefix + k_tensor + "_lora.up.weight";
                lora_down_name   = prefix + k_tensor + "_lora.down.weight";
            }else if (block_kind == 1){
                prefix = prefix + ".up_blocks";
                k_pos = k_tensor.find(".weight");
                k_tensor = k_tensor.substr(0, k_pos);
                k_pos = k_tensor.find("transformer_blocks");
                k_tensor = k_tensor.substr(k_pos);
                if(ends_with(k_tensor, "0")){
                    k_tensor = k_tensor.substr(0, k_tensor.length()-2);
                }
                std::pair<int, int> ij = find_ij(block_id);
                if(ij.first == -1)
                   continue;
                prefix = prefix + "."+std::to_string(ij.first)+".attentions."+std::to_string(ij.second)+".";
                lora_up_name     = prefix + k_tensor + "_lora.up.weight";
                lora_down_name   = prefix + k_tensor + "_lora.down.weight";
            }else{
                prefix = prefix + ".mid_block" + ".attentions.0.";
                k_pos = k_tensor.find(".weight");
                k_tensor = k_tensor.substr(0, k_pos); 
                k_pos = k_tensor.find("transformer_blocks");
                k_tensor = k_tensor.substr(k_pos);
                if(ends_with(k_tensor, "0")){
                    k_tensor = k_tensor.substr(0, k_tensor.length()-2);
                }
                lora_up_name     = prefix + k_tensor + "_lora.up.weight";
                lora_down_name   = prefix + k_tensor + "_lora.down.weight";
            }
            // LOG_INFO("unet transformer tensor: %s ", full_name.c_str());
            // LOG_INFO("corresponding up tensor: %s ", lora_up_name.c_str());
            // LOG_INFO("corresponding dn tensor: %s ", lora_down_name.c_str());

            ggml_tensor* lora_up   = NULL;
            ggml_tensor* lora_down = NULL;

            if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                lora_up = lora_tensors[lora_up_name];
            }

            if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                lora_down = lora_tensors[lora_down_name];
            }

            if (lora_up == NULL || lora_down == NULL) {
                LOG_WARN("can not find: %s and %s  k,id = (%d, %d)", lora_down_name.c_str(), lora_up_name.c_str(), block_kind, block_id);
                continue;
            }

            // print_ggml_tensor(lora_down, true, lora_down_name.c_str());
            // print_ggml_tensor(lora_up, true, lora_up_name.c_str());

            // ggml_tensor* lora_up_orig = lora_up;

            applied_lora_tensors.insert(lora_up_name);
            applied_lora_tensors.insert(lora_down_name);
            
            // ggml_mul_mat requires tensor b transposed
            // lora_down                  = ggml_cont(ctx0, ggml_transpose(ctx0, lora_down));
            // struct ggml_tensor* updown = ggml_mul_mat(ctx0, lora_down, lora_up);
            // updown                     = ggml_cont(ctx0, updown);
            // same as in lora.hpp
            lora_down                  = ggml_cont(ctx0, ggml_transpose(ctx0, lora_down));
            struct ggml_tensor* updown = ggml_mul_mat(ctx0, lora_up, lora_down);
            updown                     = ggml_cont(ctx0, ggml_transpose(ctx0, updown));
            updown                     = ggml_reshape(ctx0, updown, weight);
            GGML_ASSERT(ggml_nelements(updown) == ggml_nelements(weight));
            updown = ggml_scale_inplace(ctx0, updown, multiplier);
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
        // GGMLModule::compute(get_graph, n_threads);
        GGMLModule::compute(get_graph, n_threads, true);
    }
};


#endif  // __PMI_HPP__
