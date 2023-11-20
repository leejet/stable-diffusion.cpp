import struct
import json
import os

import numpy as np
import torch
import re
import safetensors.torch

this_file_dir = os.path.dirname(__file__)
vocab_dir = this_file_dir

SD1 = 0
SD2 = 1

ggml_ftype_str_to_int = {
    "f32": 0,
    "f16": 1,
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 8,
    "q5_1": 9,
    "q8_0": 7
}

ggml_ttype_str_to_int = {
    "f32": 0,
    "f16": 1,
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 6,
    "q5_1": 7,
    "q8_0": 8
}

QK4_0 = 32
def quantize_q4_0(x):
    assert x.shape[-1] % QK4_0 == 0 and x.shape[-1] > QK4_0
    x = x.reshape(-1, QK4_0)
    max = np.take_along_axis(x, np.argmax(np.abs(x), axis=-1)[:, np.newaxis], axis=-1)
    d = max / -8
    qs = ((x / d) + 8).round().clip(min=0, max=15).astype(np.int8)
    half = QK4_0 // 2
    qs = qs[:, :half] | (qs[:, half:] << 4)
    d = d.astype(np.float16).view(np.int8)
    y = np.concatenate((d, qs), axis=-1)
    return y

QK4_1 = 32
def quantize_q4_1(x):
    assert x.shape[-1] % QK4_1 == 0 and x.shape[-1] > QK4_1
    x = x.reshape(-1, QK4_1)
    min = np.min(x, axis=-1, keepdims=True)
    max = np.max(x, axis=-1, keepdims=True)
    d = (max - min) / ((1 << 4) - 1)
    qs = ((x - min) / d).round().clip(min=0, max=15).astype(np.int8)
    half = QK4_1 // 2
    qs = qs[:, :half] | (qs[:, half:] << 4)
    d = d.astype(np.float16).view(np.int8)
    m = min.astype(np.float16).view(np.int8)
    y = np.concatenate((d, m, qs), axis=-1)
    return y

QK5_0 = 32
def quantize_q5_0(x):
    assert x.shape[-1] % QK5_0 == 0 and x.shape[-1] > QK5_0
    x = x.reshape(-1, QK5_0)
    max = np.take_along_axis(x, np.argmax(np.abs(x), axis=-1)[:, np.newaxis], axis=-1)
    d = max / -16
    xi = ((x / d) + 16).round().clip(min=0, max=31).astype(np.int8)
    half = QK5_0 // 2
    qs = (xi[:, :half] & 0x0F) | (xi[:, half:] << 4)
    qh = np.zeros(qs.shape[:-1], dtype=np.int32)
    for i in range(QK5_0):
        qh |= ((xi[:, i] & 0x10) >> 4).astype(np.int32) << i
    d = d.astype(np.float16).view(np.int8)
    qh = qh[..., np.newaxis].view(np.int8)
    y = np.concatenate((d, qh, qs), axis=-1)
    return y

QK5_1 = 32
def quantize_q5_1(x):
    assert x.shape[-1] % QK5_1 == 0 and x.shape[-1] > QK5_1
    x = x.reshape(-1, QK5_1)
    min = np.min(x, axis=-1, keepdims=True)
    max = np.max(x, axis=-1, keepdims=True)
    d = (max - min) / ((1 << 5) - 1)
    xi = ((x - min) / d).round().clip(min=0, max=31).astype(np.int8)
    half = QK5_1//2
    qs = (xi[:, :half] & 0x0F) | (xi[:, half:] << 4)
    qh = np.zeros(xi.shape[:-1], dtype=np.int32)
    for i in range(QK5_1):
        qh |= ((xi[:, i] & 0x10) >> 4).astype(np.int32) << i
    d = d.astype(np.float16).view(np.int8)
    m = min.astype(np.float16).view(np.int8)
    qh = qh[..., np.newaxis].view(np.int8)
    ndarray = np.concatenate((d, m, qh, qs), axis=-1)
    return ndarray

QK8_0 = 32
def quantize_q8_0(x):
    assert x.shape[-1] % QK8_0 == 0 and x.shape[-1] > QK8_0
    x = x.reshape(-1, QK8_0)
    amax = np.max(np.abs(x), axis=-1, keepdims=True)
    d = amax / ((1 << 7) - 1)
    qs = (x / d).round().clip(min=-128, max=127).astype(np.int8)
    d = d.astype(np.float16).view(np.int8)
    y = np.concatenate((d, qs), axis=-1)
    return y

# copy from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L16
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def load_model_from_file(model_path):
    print("loading model from {}".format(model_path))
    if model_path.lower().endswith(".safetensors"):
        pl_sd = safetensors.torch.load_file(model_path, device="cpu")
    else:
        pl_sd = torch.load(model_path, map_location="cpu")
    state_dict = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    print("loading model from {} completed".format(model_path))
    return state_dict

def get_alpha_comprod(linear_start=0.00085, linear_end=0.0120, timesteps=1000):
    betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float32) ** 2
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas.numpy(), axis=0)
    return torch.tensor(alphas_cumprod)

unused_tensors = [
    "betas",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "log_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod",
    "posterior_variance",
    "posterior_log_variance_clipped",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
    "cond_stage_model.transformer.text_model.embeddings.position_ids",
    "cond_stage_model.model.logit_scale",
    "cond_stage_model.model.text_projection",
    "model_ema.decay",
    "model_ema.num_updates",
    "control_model",
    "lora_te_text_model",
    "embedding_manager"
]


def preprocess(state_dict):
    alphas_cumprod = state_dict.get("alphas_cumprod")
    if alphas_cumprod != None:
        # print((np.abs(get_alpha_comprod().numpy() - alphas_cumprod.numpy()) < 0.000001).all())
        pass
    else:
        print("no alphas_cumprod in file, generate new one")
        alphas_cumprod = get_alpha_comprod()
        state_dict["alphas_cumprod"] = alphas_cumprod

    new_state_dict = {}
    for name, w in state_dict.items():
        # ignore unused tensors
        if not isinstance(w, torch.Tensor):
            continue
        skip = False
        for unused_tensor in unused_tensors:
            if name.startswith(unused_tensor):
                skip = True
                break
        if skip:
            continue

        # convert BF16 to FP16
        if w.dtype == torch.bfloat16:
            w = w.to(torch.float16)

        # convert open_clip to hf CLIPTextModel (for SD2.x)
        open_clip_to_hf_clip_model = {
            "cond_stage_model.model.ln_final.bias": "cond_stage_model.transformer.text_model.final_layer_norm.bias",
            "cond_stage_model.model.ln_final.weight": "cond_stage_model.transformer.text_model.final_layer_norm.weight",
            "cond_stage_model.model.positional_embedding": "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
            "cond_stage_model.model.token_embedding.weight": "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
            "first_stage_model.decoder.mid.attn_1.to_k.bias": "first_stage_model.decoder.mid.attn_1.k.bias",
            "first_stage_model.decoder.mid.attn_1.to_k.weight": "first_stage_model.decoder.mid.attn_1.k.weight",
            "first_stage_model.decoder.mid.attn_1.to_out.0.bias": "first_stage_model.decoder.mid.attn_1.proj_out.bias",
            "first_stage_model.decoder.mid.attn_1.to_out.0.weight": "first_stage_model.decoder.mid.attn_1.proj_out.weight",
            "first_stage_model.decoder.mid.attn_1.to_q.bias": "first_stage_model.decoder.mid.attn_1.q.bias",
            "first_stage_model.decoder.mid.attn_1.to_q.weight": "first_stage_model.decoder.mid.attn_1.q.weight",
            "first_stage_model.decoder.mid.attn_1.to_v.bias": "first_stage_model.decoder.mid.attn_1.v.bias",
            "first_stage_model.decoder.mid.attn_1.to_v.weight": "first_stage_model.decoder.mid.attn_1.v.weight",
        }
        open_clip_to_hk_clip_resblock = {
            "attn.out_proj.bias": "self_attn.out_proj.bias",
            "attn.out_proj.weight": "self_attn.out_proj.weight",
            "ln_1.bias": "layer_norm1.bias",
            "ln_1.weight": "layer_norm1.weight",
            "ln_2.bias": "layer_norm2.bias",
            "ln_2.weight": "layer_norm2.weight",
            "mlp.c_fc.bias": "mlp.fc1.bias",
            "mlp.c_fc.weight": "mlp.fc1.weight",
            "mlp.c_proj.bias": "mlp.fc2.bias",
            "mlp.c_proj.weight": "mlp.fc2.weight",
        }
        open_clip_resblock_prefix = "cond_stage_model.model.transformer.resblocks."
        hf_clip_resblock_prefix = "cond_stage_model.transformer.text_model.encoder.layers."
        if name in open_clip_to_hf_clip_model:
            new_name = open_clip_to_hf_clip_model[name]
            print(f"preprocess {name} => {new_name}")
            name = new_name
        if name.startswith(open_clip_resblock_prefix):
            remain = name[len(open_clip_resblock_prefix):]
            idx = remain.split(".")[0]
            suffix = remain[len(idx)+1:]
            if suffix == "attn.in_proj_weight":
                w_q, w_k, w_v = w.chunk(3)
                for new_suffix, new_w in zip(["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"], [w_q, w_k, w_v]):
                    new_name = hf_clip_resblock_prefix + idx + "." + new_suffix
                    new_state_dict[new_name] = new_w
                    print(f"preprocess {name}{w.size()} => {new_name}{new_w.size()}")
            elif suffix == "attn.in_proj_bias":
                w_q, w_k, w_v = w.chunk(3)
                for new_suffix, new_w in zip(["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"], [w_q, w_k, w_v]):
                    new_name = hf_clip_resblock_prefix + idx + "." + new_suffix
                    new_state_dict[new_name] = new_w
                    print(f"preprocess {name}{w.size()} => {new_name}{new_w.size()}")
            else:
                new_suffix = open_clip_to_hk_clip_resblock[suffix]
                new_name = hf_clip_resblock_prefix + idx + "." + new_suffix
                new_state_dict[new_name] = w
                print(f"preprocess {name} => {new_name}")
            continue

        # convert unet transformer linear to conv2d 1x1
        if name.startswith("model.diffusion_model.") and (name.endswith("proj_in.weight") or name.endswith("proj_out.weight")):
            if len(w.shape) == 2:
                new_w = w.unsqueeze(2).unsqueeze(3)
                new_state_dict[name] = new_w
                print(f"preprocess {name} {w.size()} => {name} {new_w.size()}")
                continue

        # convert vae attn block linear to conv2d 1x1
        if name.startswith("first_stage_model.") and "attn_1" in name:
            if len(w.shape) == 2:
                new_w = w.unsqueeze(2).unsqueeze(3)
                new_state_dict[name] = new_w
                print(f"preprocess {name} {w.size()} => {name} {new_w.size()}")
                continue

        new_state_dict[name] = w
    return new_state_dict

re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}


def convert_diffusers_name_to_compvis(key):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f'model_diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, r"lora_unet_conv_out(.*)"):
        return f'model_diffusion_model_out_2{m[0]}'

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"model_diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"model_diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"model_diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"model_diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"model_diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"model_diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        return f"cond_stage_model_transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return None

def preprocess_lora(state_dict):
    new_state_dict = {}
    for name, w in state_dict.items():
        if not isinstance(w, torch.Tensor):
            continue

        # convert BF16 to FP16
        if w.dtype == torch.bfloat16:
            w = w.to(torch.float16)

        name_without_network_parts, network_part = name.split(".", 1)
        new_name_without_network_parts = convert_diffusers_name_to_compvis(name_without_network_parts)
        if new_name_without_network_parts == None:
            raise Exception(f"unknown lora tensor: {name}")
        new_name = new_name_without_network_parts + "." + network_part
        print(f"preprocess {name} => {new_name}")
        new_state_dict[new_name] = w
    return new_state_dict

def convert(model_path, out_type = None, out_file=None, lora=False):
    # load model
    if not lora:
        with open(os.path.join(vocab_dir, "vocab.json"), encoding="utf-8") as f:
            clip_vocab = json.load(f)

    state_dict = load_model_from_file(model_path)
    model_type = SD1 # lora only for SD1 now
    if not lora and "cond_stage_model.model.token_embedding.weight" in state_dict.keys():
        model_type = SD2
        print("Stable diffuison 2.x")
    else:
        print("Stable diffuison 1.x")
    if lora:
        state_dict = preprocess_lora(state_dict)
    else:
        state_dict = preprocess(state_dict)

    # output option
    if lora:
        out_type = "f16" # only f16 for now
    if out_type == None:
        weight = state_dict["model.diffusion_model.input_blocks.0.0.weight"].numpy()
        if weight.dtype == np.float32:
            out_type = "f32"
        elif weight.dtype == np.float16:
            out_type = "f16"
        elif weight.dtype == np.float64:
            out_type = "f32"
        else:
            raise Exception("unsupported weight type %s" % weight.dtype)
    if out_file == None:
        if lora:
            out_file = os.path.splitext(os.path.basename(model_path))[0] + f"-ggml-lora.bin"
        else:
            out_file = os.path.splitext(os.path.basename(model_path))[0] + f"-ggml-model-{out_type}.bin"
        out_file = os.path.join(os.getcwd(), out_file)
    print(f"Saving GGML compatible file to {out_file}")

    # convert and save
    with open(out_file, "wb") as file:
        # magic: ggml in hex
        file.write(struct.pack("i", 0x67676D6C))
        # model & file type
        ftype = (model_type << 16) | ggml_ftype_str_to_int[out_type]
        file.write(struct.pack("i", ftype))

        # vocab
        if not lora:
            byte_encoder = bytes_to_unicode()
            byte_decoder = {v: k for k, v in byte_encoder.items()}
            file.write(struct.pack("i", len(clip_vocab)))
            for key in clip_vocab:
                text = bytearray([byte_decoder[c] for c in key])
                file.write(struct.pack("i", len(text)))
                file.write(text)

        # weights
        for name in state_dict.keys():
            if not isinstance(state_dict[name], torch.Tensor):
                continue
            skip = False
            for unused_tensor in unused_tensors:
                if name.startswith(unused_tensor):
                    skip = True
                    break
            if skip:
                continue
            if name in unused_tensors:
                continue

            data = state_dict[name].numpy()

            n_dims = len(data.shape)
            shape = data.shape
            old_type = data.dtype

            ttype = "f32"
            if n_dims == 4 and not lora:
                data = data.astype(np.float16)
                ttype = "f16"
            elif n_dims == 2 and name[-7:] == ".weight":
                if out_type == "f32":
                    data = data.astype(np.float32)
                elif out_type == "f16":
                    data = data.astype(np.float16)
                elif out_type == "q4_0":
                    data = quantize_q4_0(data)
                elif out_type == "q4_1":
                    data = quantize_q4_1(data)
                elif out_type == "q5_0":
                    data = quantize_q5_0(data)
                elif out_type == "q5_1":
                    data = quantize_q5_1(data)
                elif out_type == "q8_0":
                    data = quantize_q8_0(data)
                else:
                    raise Exception("invalid out_type {}".format(out_type))
                ttype = out_type
            else:
                data = data.astype(np.float32)
                ttype = "f32"

            print("Processing tensor: {} with shape {}, {} -> {}".format(name, data.shape, old_type, ttype))

            # header
            name_bytes = name.encode("utf-8")
            file.write(struct.pack("iii", n_dims, len(name_bytes), ggml_ttype_str_to_int[ttype]))
            for i in range(n_dims):
                file.write(struct.pack("i", shape[n_dims - 1 - i]))
            file.write(name_bytes)
            # data
            data.tofile(file)
        print("Convert done")
        print(f"Saved GGML compatible file to {out_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Stable Diffuison model to GGML compatible file format")
    parser.add_argument("--out_type", choices=["f32", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"], help="output format (default: based on input)")
    parser.add_argument("--out_file", help="path to write to; default: based on input and current working directory")
    parser.add_argument("--lora", action='store_true', default = False, help="convert lora weight; default: false")
    parser.add_argument("model_path", help="model file path (*.pth, *.pt, *.ckpt, *.safetensors)")
    args = parser.parse_args()
    convert(args.model_path, args.out_type, args.out_file, args.lora)
