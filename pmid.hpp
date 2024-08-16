#ifndef __PMI_HPP__
#define __PMI_HPP__

#include "ggml_extend.hpp"

#include "clip.hpp"
#include "lora.hpp"


enum PMVersion {
    VERSION_1,
    VERSION_2,
};
struct FuseBlock : public GGMLBlock {
    // network hparams
    int in_dim;
    int out_dim;
    int hidden_dim;
    bool use_residue;

public:
    FuseBlock(int i_d, int o_d, int h_d, bool use_residue = true)
        : in_dim(i_d), out_dim(o_d), hidden_dim(h_d), use_residue(use_residue) {
        blocks["fc1"]       = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, true));
        blocks["fc2"]       = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, out_dim, true));
        blocks["layernorm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(in_dim));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]

        auto fc1        = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
        auto fc2        = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);
        auto layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["layernorm"]);

        struct ggml_tensor* r = x;
        // x = ggml_nn_layer_norm(ctx, x, ln_w, ln_b);
        x = layer_norm->forward(ctx, x);
        // x = ggml_add(ctx, ggml_mul_mat(ctx, fc1_w, x),  fc1_b);
        x = fc1->forward(ctx, x);
        x = ggml_gelu_inplace(ctx, x);
        x = fc2->forward(ctx, x);
        // x = ggml_add(ctx, ggml_mul_mat(ctx, fc2_w, x),  fc2_b);
        if (use_residue)
            x = ggml_add(ctx, x, r);
        return x;
    }
};

/*
class QFormerPerceiver(nn.Module):
    def __init__(self, id_embeddings_dim, cross_attention_dim, num_tokens, embedding_dim=1024, use_residual=True, ratio=4):
        super().__init__()

        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.use_residual = use_residual
        print(cross_attention_dim*num_tokens)
        self.token_proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim*ratio),
            nn.GELU(),
            nn.Linear(id_embeddings_dim*ratio, cross_attention_dim*num_tokens),
        )
        self.token_norm = nn.LayerNorm(cross_attention_dim)
        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=128,
            heads=cross_attention_dim // 128,
            embedding_dim=embedding_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def forward(self, x, last_hidden_state):
        x = self.token_proj(x)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.token_norm(x) # cls token
        out = self.perceiver_resampler(x, last_hidden_state) # retrieve from patch tokens
        if self.use_residual: # TODO: if use_residual is not true
            out = x + 1.0 * out 
        return out
*/

struct QFormerPerceiver : public GGMLBlock {
    // network hparams
    int num_tokens;
    int cross_attention_dim;
    bool use_residul;


public:
    QFormerPerceiver(int id_embeddings_dim, int cross_attention_d, int num_t, int embedding_dim=1024, 
                     bool use_r=true, int ratio=4)
        : cross_attention_dim(cross_attention_d),  num_tokens(num_t), use_residul(use_r) {
        // blocks["mlp1"]       = std::shared_ptr<GGMLBlock>(new FuseBlock(imb_d * 2, imb_d, imb_d, false));
        // blocks["mlp2"]       = std::shared_ptr<GGMLBlock>(new FuseBlock(imb_d, imb_d, imb_d, true));
        // blocks["layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(embed_dim));
        blocks["token_proj"] = std::shared_ptr<GGMLBlock>(new Mlp(id_embeddings_dim, 
                                                                  id_embeddings_dim*ratio, 
                                                                  cross_attention_dim*num_tokens, 
                                                                  true));
        blocks["token_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(cross_attention_d));
    }

    

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* last_hidden_state){
        // x: [N, channels, h, w]
        auto mlp1       = std::dynamic_pointer_cast<FuseBlock>(blocks["mlp1"]);
        auto mlp2       = std::dynamic_pointer_cast<FuseBlock>(blocks["mlp2"]);
        auto token_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["token_norm"]);

       
    }
};

/*
class FacePerceiverResampler(torch.nn.Module):
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()
        
        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)
*/

struct FacePerceiverResampler : public GGMLBlock {    
    // network hparams  
public:
    FacePerceiverResampler( int dim=768,
                            int depth=4,
                            int dim_head=64,
                            int heads=16,
                            int embedding_dim=1280,
                            int output_dim=768,
                            int ff_mult=4) {
        blocks["proj_in"] = std::shared_ptr<GGMLBlock>(new Linear(embedding_dim, dim, true));
        blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(dim, output_dim, true));
        blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new LayerNorm(output_dim));
    }

    

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* last_hidden_state){
        // x: [N, channels, h, w]
        auto mlp1       = std::dynamic_pointer_cast<FuseBlock>(blocks["mlp1"]);
        auto mlp2       = std::dynamic_pointer_cast<FuseBlock>(blocks["mlp2"]);
        auto token_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["token_norm"]);

       
    }
};

/*

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

*/

struct FeedForward : public GGMLBlock {
    // network hparams
    int dim;
    
public:
    FeedForward(int d, int multi=4)
    : dim(d) {
        int inner_dim = dim * multi;
        blocks["norm"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));        
        blocks["ff"]   = std::shared_ptr<GGMLBlock>(new Mlp(dim, inner_dim, dim, false));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x){

        auto norm = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
        auto ff   = std::dynamic_pointer_cast<Mlp>(blocks["ff"]);

        x = norm->forward(ctx, x);
        x = ff->forward(ctx, x);                            
        return x;
    }

};
                                



struct PerceiverAttention : public GGMLBlock {
    // network hparams
    float scale; // = dim_head**-0.5
    int dim_head; // = dim_head
    int heads; // = heads
public:
    PerceiverAttention(int dim, int dim_h=64, int h=8)
    : scale(powf(dim_h, -0.5)), dim_head(dim_h), heads(h) {

        int inner_dim = dim_head * heads;
        blocks["norm1"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
        blocks["norm2"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
        blocks["to_q"]   = std::shared_ptr<GGMLBlock>(new Linear(dim, inner_dim, false));
        blocks["to_kv"]  = std::shared_ptr<GGMLBlock>(new Linear(dim, inner_dim*2, false));
        blocks["to_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, dim, false));
    }

    struct ggml_tensor* reshape_tensor(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                int heads) {
        int64_t ne[4];
        for(int i = 0; i < 4; ++i)
           ne[i] = x->ne[i];
        x = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], heads, x->ne[2]/heads, 
                                 x->nb[1], x->nb[2], x->nb[3], 0);
        // x = ggml_transpose(ctx, x); 
        x  = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        x  = ggml_reshape_4d(ctx, x, ne[0], heads, ne[1], ne[2]/heads);
        return x;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* latents){

        // x (torch.Tensor): image features
        //     shape (b, n1, D)
        // latent (torch.Tensor): latent features
        //     shape (b, n2, D)
        int64_t ne[4];
        for(int i = 0; i < 4; ++i)
           ne[i] = latents->ne[i];

        auto norm1      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto norm2      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        x = norm1->forward(ctx, x);
        latents = norm2->forward(ctx, latents);
        auto to_q = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
        auto q = to_q->forward(ctx, latents);

        auto kv_input = ggml_concat(ctx, x, latents, 1);
        auto to_kv    = std::dynamic_pointer_cast<Linear>(blocks["to_kv"]);
        auto kv = to_kv->forward(ctx, kv_input);
        int64_t n = kv->ne[2] / 2;
        int64_t offset = kv->nb[2] * n;
        auto k = ggml_view_3d(ctx, kv, kv->ne[0], kv->ne[1], n, kv->nb[1], kv->nb[2], offset*0);
        auto v = ggml_view_3d(ctx, kv, kv->ne[0], kv->ne[1], n, kv->nb[1], kv->nb[2], offset*1);
        q = reshape_tensor(ctx, q, heads);
        k = reshape_tensor(ctx, k, heads);
        v = reshape_tensor(ctx, v, heads);

        scale = 1.f / sqrt(sqrt((float)dim_head));
        k = ggml_scale_inplace(ctx, k, scale);
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 1, 3, 2)); 
        q = ggml_scale_inplace(ctx, q, scale);
        auto weight = ggml_mul_mat(ctx, q, k);
        weight = ggml_soft_max(ctx, weight);
        auto out = ggml_mul_mat(ctx, weight, v);

        out  = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
        out  = ggml_reshape_3d(ctx, out, ne[0], ne[1], ggml_nelements(out)/(ne[0]*ne[1]));
        auto to_out    = std::dynamic_pointer_cast<Linear>(blocks["to_out"]);
        out  = to_out->forward(ctx, out);
        return out;
    }
};


struct FuseModule : public GGMLBlock {
    // network hparams
    int embed_dim;

public:
    FuseModule(int imb_d)
        : embed_dim(imb_d) {
        blocks["mlp1"]       = std::shared_ptr<GGMLBlock>(new FuseBlock(imb_d * 2, imb_d, imb_d, false));
        blocks["mlp2"]       = std::shared_ptr<GGMLBlock>(new FuseBlock(imb_d, imb_d, imb_d, true));
        blocks["layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(embed_dim));
    }

    struct ggml_tensor* fuse_fn(struct ggml_context* ctx,
                                struct ggml_tensor* prompt_embeds,
                                struct ggml_tensor* id_embeds) {
        auto mlp1       = std::dynamic_pointer_cast<FuseBlock>(blocks["mlp1"]);
        auto mlp2       = std::dynamic_pointer_cast<FuseBlock>(blocks["mlp2"]);
        auto layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);

        auto prompt_embeds0 = ggml_cont(ctx, ggml_permute(ctx, prompt_embeds, 2, 0, 1, 3));
        auto id_embeds0     = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 2, 0, 1, 3));
        // concat is along dim 2
        auto stacked_id_embeds = ggml_concat(ctx, prompt_embeds0, id_embeds0, 2);
        stacked_id_embeds      = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 1, 2, 0, 3));

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

        struct ggml_tensor* valid_id_embeds = id_embeds;
        // # slice out the image token embeddings
        // print_ggml_tensor(class_tokens_mask_pos, false);
        ggml_set_name(class_tokens_mask_pos, "class_tokens_mask_pos");
        ggml_set_name(prompt_embeds, "prompt_embeds");
        // print_ggml_tensor(valid_id_embeds, true, "valid_id_embeds");
        // print_ggml_tensor(class_tokens_mask_pos, true, "class_tokens_mask_pos");
        struct ggml_tensor* image_token_embeds = ggml_get_rows(ctx, prompt_embeds, class_tokens_mask_pos);
        ggml_set_name(image_token_embeds, "image_token_embeds");
        struct ggml_tensor* stacked_id_embeds = fuse_fn(ctx, image_token_embeds, valid_id_embeds);

        stacked_id_embeds = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 0, 2, 1, 3));
        if (left && right) {
            stacked_id_embeds = ggml_concat(ctx, left, stacked_id_embeds, 2);
            stacked_id_embeds = ggml_concat(ctx, stacked_id_embeds, right, 2);
        } else if (left) {
            stacked_id_embeds = ggml_concat(ctx, left, stacked_id_embeds, 2);
        } else if (right) {
            stacked_id_embeds = ggml_concat(ctx, stacked_id_embeds, right, 2);
        }
        stacked_id_embeds                         = ggml_cont(ctx, ggml_permute(ctx, stacked_id_embeds, 0, 2, 1, 3));
        class_tokens_mask                         = ggml_cont(ctx, ggml_transpose(ctx, class_tokens_mask));
        class_tokens_mask                         = ggml_repeat(ctx, class_tokens_mask, prompt_embeds);
        prompt_embeds                             = ggml_mul(ctx, prompt_embeds, class_tokens_mask);
        struct ggml_tensor* updated_prompt_embeds = ggml_add(ctx, prompt_embeds, stacked_id_embeds);
        ggml_set_name(updated_prompt_embeds, "updated_prompt_embeds");
        return updated_prompt_embeds;
    }
};

struct PhotoMakerIDEncoderBlock : public CLIPVisionModelProjection {
    PhotoMakerIDEncoderBlock()
        : CLIPVisionModelProjection(OPENAI_CLIP_VIT_L_14) {
        blocks["visual_projection_2"] = std::shared_ptr<GGMLBlock>(new Linear(1024, 1280, false));
        blocks["fuse_module"]         = std::shared_ptr<GGMLBlock>(new FuseModule(2048));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* id_pixel_values,
                                struct ggml_tensor* prompt_embeds,
                                struct ggml_tensor* class_tokens_mask,
                                struct ggml_tensor* class_tokens_mask_pos,
                                struct ggml_tensor* left,
                                struct ggml_tensor* right) {
        // x: [N, channels, h, w]
        auto vision_model        = std::dynamic_pointer_cast<CLIPVisionModel>(blocks["vision_model"]);
        auto visual_projection   = std::dynamic_pointer_cast<CLIPProjection>(blocks["visual_projection"]);
        auto visual_projection_2 = std::dynamic_pointer_cast<Linear>(blocks["visual_projection_2"]);
        auto fuse_module         = std::dynamic_pointer_cast<FuseModule>(blocks["fuse_module"]);

        struct ggml_tensor* shared_id_embeds = vision_model->forward(ctx, id_pixel_values);          // [N, hidden_size]
        struct ggml_tensor* id_embeds        = visual_projection->forward(ctx, shared_id_embeds);    // [N, proj_dim(768)]
        struct ggml_tensor* id_embeds_2      = visual_projection_2->forward(ctx, shared_id_embeds);  // [N, 1280]

        id_embeds   = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 2, 0, 1, 3));
        id_embeds_2 = ggml_cont(ctx, ggml_permute(ctx, id_embeds_2, 2, 0, 1, 3));

        id_embeds = ggml_concat(ctx, id_embeds, id_embeds_2, 2);  // [batch_size, seq_length, 1, 2048] check whether concat at dim 2 is right
        id_embeds = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 1, 2, 0, 3));

        struct ggml_tensor* updated_prompt_embeds = fuse_module->forward(ctx,
                                                                         prompt_embeds,
                                                                         id_embeds,
                                                                         class_tokens_mask,
                                                                         class_tokens_mask_pos,
                                                                         left, right);
        return updated_prompt_embeds;
    }
};

struct PhotoMakerIDEncoder_CLIPInsightfaceExtendtokenBlock : public CLIPVisionModelProjection {
    PhotoMakerIDEncoder_CLIPInsightfaceExtendtokenBlock()
        : CLIPVisionModelProjection(OPENAI_CLIP_VIT_L_14) {
        blocks["visual_projection_2"] = std::shared_ptr<GGMLBlock>(new Linear(1024, 1280, false));
        blocks["fuse_module"]         = std::shared_ptr<GGMLBlock>(new FuseModule(2048));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* id_pixel_values,
                                struct ggml_tensor* prompt_embeds,
                                struct ggml_tensor* class_tokens_mask,
                                struct ggml_tensor* class_tokens_mask_pos,
                                struct ggml_tensor* left,
                                struct ggml_tensor* right) {
        // x: [N, channels, h, w]
        auto vision_model        = std::dynamic_pointer_cast<CLIPVisionModel>(blocks["vision_model"]);
        auto visual_projection   = std::dynamic_pointer_cast<CLIPProjection>(blocks["visual_projection"]);
        auto visual_projection_2 = std::dynamic_pointer_cast<Linear>(blocks["visual_projection_2"]);
        auto fuse_module         = std::dynamic_pointer_cast<FuseModule>(blocks["fuse_module"]);

        struct ggml_tensor* shared_id_embeds = vision_model->forward(ctx, id_pixel_values);          // [N, hidden_size]
        struct ggml_tensor* id_embeds        = visual_projection->forward(ctx, shared_id_embeds);    // [N, proj_dim(768)]
        struct ggml_tensor* id_embeds_2      = visual_projection_2->forward(ctx, shared_id_embeds);  // [N, 1280]

        id_embeds   = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 2, 0, 1, 3));
        id_embeds_2 = ggml_cont(ctx, ggml_permute(ctx, id_embeds_2, 2, 0, 1, 3));

        id_embeds = ggml_concat(ctx, id_embeds, id_embeds_2, 2);  // [batch_size, seq_length, 1, 2048] check whether concat at dim 2 is right
        id_embeds = ggml_cont(ctx, ggml_permute(ctx, id_embeds, 1, 2, 0, 3));

        struct ggml_tensor* updated_prompt_embeds = fuse_module->forward(ctx,
                                                                         prompt_embeds,
                                                                         id_embeds,
                                                                         class_tokens_mask,
                                                                         class_tokens_mask_pos,
                                                                         left, right);
        return updated_prompt_embeds;
    }
};

struct PhotoMakerIDEncoder : public GGMLRunner {
public:
    SDVersion version = VERSION_XL;
    PMVersion pm_version = VERSION_1;
    PhotoMakerIDEncoderBlock id_encoder;
    PhotoMakerIDEncoder_CLIPInsightfaceExtendtokenBlock id_encoder2;
    float style_strength;

    std::vector<float> ctm;
    std::vector<ggml_fp16_t> ctmf16;
    std::vector<int> ctmpos;

    std::vector<ggml_fp16_t> zeros_left_16;
    std::vector<float> zeros_left;
    std::vector<ggml_fp16_t> zeros_right_16;
    std::vector<float> zeros_right;

public:
    PhotoMakerIDEncoder(ggml_backend_t backend, ggml_type wtype, SDVersion version = VERSION_XL, 
                        PMVersion pm_v = VERSION_1, float sty = 20.f)
        : GGMLRunner(backend, wtype),
          version(version),
          pm_version(pm_v),
          style_strength(sty) {
        if(pm_version == VERSION_1){     
            id_encoder.init(params_ctx, wtype);
        }else if(pm_version == VERSION_2){
            id_encoder2.init(params_ctx, wtype);
        }
    }

    std::string get_desc() {
        return "pmid";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        if(pm_version == VERSION_1)
            id_encoder.get_param_tensors(tensors, prefix);
        else if(pm_version == VERSION_2)
            id_encoder2.get_param_tensors(tensors, prefix);

    }

    struct ggml_cgraph* build_graph(  // struct ggml_allocr* allocr,
        struct ggml_tensor* id_pixel_values,
        struct ggml_tensor* prompt_embeds,
        std::vector<bool>& class_tokens_mask) {
        ctm.clear();
        ctmf16.clear();
        ctmpos.clear();
        zeros_left.clear();
        zeros_left_16.clear();
        zeros_right.clear();
        zeros_right_16.clear();

        ggml_context* ctx0 = compute_ctx;

        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        int64_t hidden_size = prompt_embeds->ne[0];
        int64_t seq_length  = prompt_embeds->ne[1];
        ggml_type type      = GGML_TYPE_F32;

        struct ggml_tensor* class_tokens_mask_d = ggml_new_tensor_1d(ctx0, type, class_tokens_mask.size());

        struct ggml_tensor* id_pixel_values_d = to_backend(id_pixel_values);
        struct ggml_tensor* prompt_embeds_d   = to_backend(prompt_embeds);

        struct ggml_tensor* left  = NULL;
        struct ggml_tensor* right = NULL;
        for (int i = 0; i < class_tokens_mask.size(); i++) {
            if (class_tokens_mask[i]) {
                ctm.push_back(0.f);                        // here use 0.f instead of 1.f to make a scale mask
                ctmf16.push_back(ggml_fp32_to_fp16(0.f));  // here use 0.f instead of 1.f to make a scale mask
                ctmpos.push_back(i);
            } else {
                ctm.push_back(1.f);                        // here use 1.f instead of 0.f to make a scale mask
                ctmf16.push_back(ggml_fp32_to_fp16(1.f));  // here use 0.f instead of 1.f to make a scale mask
            }
        }
        if (ctmpos[0] > 0) {
            left = ggml_new_tensor_3d(ctx0, type, hidden_size, 1, ctmpos[0]);
        }
        if (ctmpos[ctmpos.size() - 1] < seq_length - 1) {
            right = ggml_new_tensor_3d(ctx0, type,
                                       hidden_size, 1, seq_length - ctmpos[ctmpos.size() - 1] - 1);
        }
        struct ggml_tensor* class_tokens_mask_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ctmpos.size());

        {
            if (type == GGML_TYPE_F16)
                set_backend_tensor_data(class_tokens_mask_d, ctmf16.data());
            else
                set_backend_tensor_data(class_tokens_mask_d, ctm.data());
            set_backend_tensor_data(class_tokens_mask_pos, ctmpos.data());
            if (left) {
                if (type == GGML_TYPE_F16) {
                    for (int i = 0; i < ggml_nelements(left); ++i)
                        zeros_left_16.push_back(ggml_fp32_to_fp16(0.f));
                    set_backend_tensor_data(left, zeros_left_16.data());
                } else {
                    for (int i = 0; i < ggml_nelements(left); ++i)
                        zeros_left.push_back(0.f);
                    set_backend_tensor_data(left, zeros_left.data());
                }
            }
            if (right) {
                if (type == GGML_TYPE_F16) {
                    for (int i = 0; i < ggml_nelements(right); ++i)
                        zeros_right_16.push_back(ggml_fp32_to_fp16(0.f));
                    set_backend_tensor_data(right, zeros_right_16.data());
                } else {
                    for (int i = 0; i < ggml_nelements(right); ++i)
                        zeros_right.push_back(0.f);
                    set_backend_tensor_data(right, zeros_right.data());
                }
            }
        }
        struct ggml_tensor* updated_prompt_embeds = NULL;
        if(pm_version == VERSION_1)
            updated_prompt_embeds = id_encoder.forward(ctx0,
                                                        id_pixel_values_d,
                                                        prompt_embeds_d,
                                                        class_tokens_mask_d,
                                                        class_tokens_mask_pos,
                                                        left, right);
        else if(pm_version == VERSION_2)
            updated_prompt_embeds = id_encoder2.forward(ctx0,
                                                        id_pixel_values_d,
                                                        prompt_embeds_d,
                                                        class_tokens_mask_d,
                                                        class_tokens_mask_pos,
                                                        left, right);

        ggml_build_forward_expand(gf, updated_prompt_embeds);

        return gf;
    }

    void compute(const int n_threads,
                 struct ggml_tensor* id_pixel_values,
                 struct ggml_tensor* prompt_embeds,
                 std::vector<bool>& class_tokens_mask,
                 struct ggml_tensor** updated_prompt_embeds,
                 ggml_context* output_ctx) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            // return build_graph(compute_allocr, id_pixel_values, prompt_embeds, class_tokens_mask);
            return build_graph(id_pixel_values, prompt_embeds, class_tokens_mask);
        };

        // GGMLRunner::compute(get_graph, n_threads, updated_prompt_embeds);
        GGMLRunner::compute(get_graph, n_threads, true, updated_prompt_embeds, output_ctx);
    }
};

#endif  // __PMI_HPP__
