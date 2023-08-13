import struct
import json
import os

import numpy as np
import torch
import safetensors.torch

this_file_dir = os.path.dirname(__file__)
vocab_dir = this_file_dir

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
    assert x.shape[-1] % QK4_0 == 0
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
    assert x.shape[-1] % QK4_1 == 0
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
    assert x.shape[1] % QK5_0 == 0
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
    assert x.shape[-1] % QK5_1 == 0
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
    assert x.shape[-1] % QK8_0 == 0
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
    This is a signficant percentage of your normal, say, 32K bpe vocab.
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
    "model_ema.decay",
    "model_ema.num_updates"
]

def convert(model_path, out_type = None, out_file=None):
    # load model
    with open(os.path.join(vocab_dir, "vocab.json"), encoding="utf-8") as f:
        clip_vocab = json.load(f)
    
    state_dict = load_model_from_file(model_path)
    alphas_cumprod = state_dict.get("alphas_cumprod")
    if alphas_cumprod != None:
        # print((np.abs(get_alpha_comprod().numpy() - alphas_cumprod.numpy()) < 0.000001).all())
        pass
    else:
        print("no alphas_cumprod in file, generate new one")
        alphas_cumprod = get_alpha_comprod()
        state_dict["alphas_cumprod"] = alphas_cumprod


    # output option
    if out_type == None:
        weight = state_dict["cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight"].numpy()
        if weight.dtype == np.float32:
            out_type = "f32"
        elif weight.dtype == np.float16:
            out_type = "f16"
    if out_file == None:
        out_file = os.path.splitext(os.path.basename(model_path))[0] + f"-ggml-model-{out_type}.bin"
        out_file = os.path.join(os.getcwd(), out_file)
    print(f"Saving GGML compatible file to {out_file}")

    # convert and save
    with open(out_file, "wb") as file:
        # magic: ggml in hex
        file.write(struct.pack("i", 0x67676D6C))
        # out type
        file.write(struct.pack("i", ggml_ftype_str_to_int[out_type]))

        # vocab
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
            if name in unused_tensors:
                continue
            data = state_dict[name].numpy()

            n_dims = len(data.shape)
            shape = data.shape
            old_type = data.dtype

            ttype = "f32"
            if n_dims == 4:
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
    parser.add_argument("model_path", help="model file path (*.pth, *.pt, *.ckpt, *.safetensors)")
    args = parser.parse_args()
    convert(args.model_path, args.out_type, args.out_file)
