#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
import sys
import urllib.request
import urllib.error

def parse_arguments():
    ap = argparse.ArgumentParser(
        description="Client for stable-diffusion.cpp sd-server",
        allow_abbrev=False
    )

    ap.add_argument("--server-url", default=os.environ.get("SD_SERVER_URL"),
        help="URL of the sd-server OpenAI-compatible endpoint. Defaults to SD_SERVER_URL env var.")

    ap.add_argument("-o", "--output", default="./output.png",
        help="path to write result image to. You can use printf-style %%d format specifiers for image sequences (default: ./output.png) (e.g., output_%%03d.png).")
    ap.add_argument("--output-begin-idx", type=int, default=None,
        help="starting index for output image sequence, must be non-negative (default 0 if specified %%d in output path, 1 otherwise).")
    ap.add_argument("-v", "--verbose", action="store_true",
        help="print extra info.")

    ap.add_argument("-p", "--prompt", default="",
        help="the prompt to render")
    ap.add_argument("-n", "--negative-prompt", dest="negative_prompt", default=None,
        help="the negative prompt (default: \"\")")
    ap.add_argument("-H", "--height", type=int,
        help="image height, in pixel space (default: 512)")
    ap.add_argument("-W", "--width", type=int,
        help="image width, in pixel space (default: 512)")
    ap.add_argument("--steps", type=int,
        help="number of sample steps (default: 20)")
    ap.add_argument("--high-noise-steps", type=int, dest="high_noise_steps",
        help="(high noise) number of sample steps (default: -1 = auto)")
    ap.add_argument("--clip-skip", type=int, dest="clip_skip",
        help="ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1). <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x")
    ap.add_argument("-b", "--batch-count", type=int, dest="batch_count",
        help="batch count")
    ap.add_argument("--video-frames", type=int, dest="video_frames",
        help="video frames (default: 1)")
    ap.add_argument("--fps", type=int,
        help="fps (default: 24)")
    ap.add_argument("--upscale-repeats", type=int, dest="upscale_repeats",
        help="Run the ESRGAN upscaler this many times (default: 1)")
    ap.add_argument("--cfg-scale", type=float, dest="cfg_scale",
        help="unconditional guidance scale (default: 7.0)")
    ap.add_argument("--img-cfg-scale", type=float, dest="img_cfg_scale",
        help="image guidance scale for inpaint or instruct-pix2pix models (default: same as --cfg-scale)")
    ap.add_argument("--guidance", type=float,
        help="distilled guidance scale for models with guidance input (default: 3.5)")
    ap.add_argument("--strength", type=float,
        help="strength for noising/unnoising (default: 0.75)")
    ap.add_argument("--pm-style-strength", type=float, dest="pm_style_strength",
        help="PhotoMaker style strength")
    ap.add_argument("--control-strength", type=float, dest="control_strength",
        help="strength to apply Control Net (default: 0.9). 1.0 corresponds to full destruction of information in init image")
    ap.add_argument("--moe-boundary", type=float, dest="moe_boundary",
        help="timestep boundary for Wan2.2 MoE model (default: 0.875). Only enabled if --high-noise-steps is set to -1")
    ap.add_argument("--vace-strength", type=float, dest="vace_strength",
        help="wan vace strength")
    ap.add_argument("--increase-ref-index", action="store_true", dest="increase_ref_index", default=None,
        help="automatically increase the indices of references images based on the order they are listed (starting with 1)")
    ap.add_argument("--disable-auto-resize-ref-image", action="store_false", dest="auto_resize_ref_image", default=None,
        help="disable auto resize of ref images")
    ap.add_argument("-s", "--seed", type=int,
        help="RNG seed (default: 42, use random seed for < 0)")
    ap.add_argument("--sampling-method", dest="sample_method", default=None,
        help="sampling method, one of [euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd] (default: euler for Flux/SD3/Wan, euler_a otherwise)")
    ap.add_argument("--high-noise-sampling-method", dest="high_noise_sample_method", default=None,
        help="(high noise) sampling method, one of [euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd] (default: euler for Flux/SD3/Wan, euler_a otherwise)")
    ap.add_argument("--scheduler", default=None,
        help="denoiser sigma scheduler, one of [discrete, karras, exponential, ays, gits, smoothstep, sgm_uniform, simple, kl_optimal, lcm], default: discrete")
    ap.add_argument("--skip-layers", dest="skip_layers", default=None,
        help="layers to skip for SLG steps (default: [7,8,9]).")
    ap.add_argument("--high-noise-skip-layers", dest="high_noise_skip_layers", default=None,
        help="(high noise) layers to skip for SLG steps (default: [7,8,9])")
    ap.add_argument("--cache-mode", dest="cache_mode",
        help="caching method: 'easycache' (DiT), 'ucache' (UNET), 'dbcache'/'taylorseer'/'cache-dit' (DiT block-level).")
    ap.add_argument("--cache-option", dest="cache_option",
        help="named cache params (key=value format, comma-separated). easycache/ucache: threshold=,start=,end=,decay=,relative=,reset=; dbcache/taylorseer/cache-dit: Fn=,Bn=,threshold=,warmup=. Examples: \"threshold=0.25\" or \"threshold=1.5,reset=0\"")
    ap.add_argument("--cache-preset", dest="cache_preset",
        help="cache-dit preset: 'slow'/'s', 'medium'/'m', 'fast'/'f', 'ultra'/'u'")
    ap.add_argument("--scm-mask", dest="scm_mask",
        help="SCM steps mask for cache-dit: comma-separated 0/1 (e.g., \"1,1,1,0,0,1,0,0,1,0\") - 1=compute, 0=can cache")

    args, unknown = ap.parse_known_args()

    for u_arg in unknown:
        print(f"Warning: Unsupported argument '{u_arg}' will be ignored.")

    args_dict = vars(args)

    for arg in ["skip_layers", "high_noise_skip_layers"]:
        if args_dict.get(arg) is not None:
            args_dict[arg] = [int(x) for x in args_dict[arg].split(',')]

    if args_dict.get("output"):
        output_format = 'png'
        output_ext = os.path.splitext(args_dict['output'])[-1].lower()
        if output_ext in ['.jpg', '.jpeg', '.jpe']:
            output_format = 'jpeg'
        args_dict["output_format"] = output_format

    util_keys = {'verbose', 'server_url', 'output', 'output_begin_idx'}

    util_opts = {k: v for k, v in args_dict.items() if k in util_keys and v is not None}
    gen_opts = {k: v for k, v in args_dict.items() if k not in util_keys and v is not None}

    return util_opts, gen_opts


def build_openai_payload(gen_opts, util_opts):
    extension_data = {}
    api_data = {}

    extension_keys = [
        "negative_prompt", "seed", "video_frames", "fps",
        "cfg_scale", "img_cfg_scale", "guidance", "strength",
        "steps", "sample_method", "scheduler",
        "high_noise_steps", "high_noise_sample_method",
        "clip_skip", "upscale_repeats", "moe_boundary",
        "control_strength", "pm_style_strength", "vace_strength",
        "cache_mode", "cache_option", "cache_preset", "scm_mask",
        "increase_ref_index", "auto_resize_ref_image",
        "skip_layers", "high_noise_skip_layers"
    ]

    for key in extension_keys:
        if gen_opts.get(key) is not None:
            extension_data[key] = gen_opts[key]

    width = gen_opts.get("width")
    height = gen_opts.get("height")
    if width and height:
        api_data["size"] = f"{width}x{height}"
    elif width:
        extension_data["width"] = width
    elif height:
        extension_data["height"] = height

    if gen_opts.get("output_format"):
        api_data["output_format"] = gen_opts["output_format"]

    if gen_opts.get("batch_count"):
        api_data["n"] = gen_opts["batch_count"]

    prompt = gen_opts.get('prompt', '')
    json_payload = json.dumps(extension_data)
    api_data["prompt"] = f"{prompt}<sd_cpp_extra_args>{json_payload}</sd_cpp_extra_args>"

    return api_data


def decode_openai_response(response_body):
    try:
        data = json.loads(response_body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")

    if 'data' not in data:
        raise ValueError(f"Unexpected response format (no 'data' key)")

    images = data['data']
    decoded_images = []

    for i, img_data in enumerate(images):
        b64_data = img_data.get("b64_json")
        if not b64_data:
            raise ValueError(f"No image data found for item {i}")
        try:
            decoded_images.append(base64.b64decode(b64_data))
        except base64.binascii.Error as e:
            raise ValueError(f"Failed to decode base64 data for item {i}: {e}")

    return decoded_images


def save_images(image_list, util_opts):
    verbose = util_opts.get("verbose", False)
    output = util_opts.get("output", "./output.png")
    output_begin_idx = util_opts.get("output_begin_idx")

    dirname, filename = os.path.split(output)

    format_specifier = re.search(r'%\d*d', filename)

    if format_specifier:
        start_idx = 0
        fmt_pref = filename[:format_specifier.start()]
        fmt_spec = format_specifier.group()
        fmt_suf = filename[format_specifier.end():]
    else:
        start_idx = 1
        stem, ext = os.path.splitext(filename)
        fmt_pref = stem
        fmt_spec = '_%d'
        fmt_suf = ext

    if output_begin_idx is not None:
        start_idx = output_begin_idx

    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    for i, img_bytes in enumerate(image_list):
        if not format_specifier and i == 0:
            filepath = output
        else:
            fmt_file = ''.join([fmt_pref, fmt_spec % (i + start_idx), fmt_suf])
            filepath = os.path.join(dirname, fmt_file)

        with open(filepath, "wb") as f:
            f.write(img_bytes)

        if verbose:
            print(f"Saved image to {filepath}")


def main():
    util_opts, gen_opts = parse_arguments()

    verbose = bool(util_opts.get("verbose"))

    server_url = util_opts.get("server_url")
    if not server_url:
        print("Error: --server-url not provided and SD_SERVER_URL env var not found.", file=sys.stderr)
        sys.exit(1)

    if not server_url.endswith('/'):
        server_url += '/'
    endpoint = server_url + "v1/images/generations"

    api_payload = build_openai_payload(gen_opts, util_opts)

    if verbose:
        print(f"Sending request to: {endpoint}")
        print(f"Payload: {json.dumps(api_payload, indent=2)}")

    req_data = json.dumps(api_payload).encode('utf-8')
    req = urllib.request.Request(endpoint, data=req_data, headers={'Content-Type': 'application/json'})

    response_body = None
    try:
        with urllib.request.urlopen(req) as response:
            response_body = response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        sys.exit(1)
    except Exception as e:
        print(f"Request Error: {e}")
        sys.exit(1)

    try:
        images = decode_openai_response(response_body)
    except ValueError as e:
        print(f"Error decoding response: {e}")
        sys.exit(1)

    save_images(images, util_opts)


if __name__ == "__main__":
    main()

