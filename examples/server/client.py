#!/usr/bin/env python3

# Example client program for the stable-diffusion.cpp server
# Supports openai and sdapi endpoints
# Should only require Python 3.6; optionally uses the OpenAI Python module

import argparse
import base64
import json
import os
import re
import sys
import urllib.request
import urllib.error
import urllib.parse

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

apis = ['sdapi', 'openai', 'openai-module']

def extract_lora_tags(prompt):

    pattern = r'<lora:([^:>]+):([^>]+)>'
    lora_data = []

    matches = list(re.finditer(pattern, prompt))

    for match in matches:
        raw_path = match.group(1)
        raw_mul = match.group(2)
        try:
            mul = float(raw_mul)
        except ValueError:
            continue

        is_high_noise = False
        prefix = "|high_noise|"
        if raw_path.startswith(prefix):
            raw_path = raw_path[len(prefix):]
            is_high_noise = True

        lora_data.append({
            'name': raw_path,
            'multiplier': mul,
            'is_high_noise': is_high_noise,
            })

        prompt = prompt.replace(match.group(0), "", 1)

    return prompt, lora_data


def parse_arguments():
    ap = argparse.ArgumentParser(
        description="Client for stable-diffusion.cpp sd-server",
        allow_abbrev=False
    )

    ap.add_argument("--server-url", default=os.environ.get("SD_SERVER_URL"),
        help="URL of the sd-server OpenAI-compatible endpoint. Defaults to SD_SERVER_URL env var.")
    ap.add_argument("--api", choices=apis, default=apis[0],
        help=f"Select API backend, one of {apis} (default: {apis[0]})")

    # replicate sd-cli parameters

    ap.add_argument("-o", "--output", default="./output.png",
        help="path to write result image to. You can use printf-style %%d format specifiers for image sequences (default: ./output.png) (e.g., output_%%03d.png).")
    ap.add_argument("--output-begin-idx", type=int, default=None,
        help="starting index for output image sequence, must be non-negative (default 0 if specified %%d in output path, 1 otherwise).")
    ap.add_argument("-v", "--verbose", action="store_true",
        help="print extra info.")

    ap.add_argument("-p", "--prompt", required=True,
        help="the prompt to render")

    ap.add_argument("-i", "--init-img", dest="init_img", default=None,
        help="path to the init image")
    ap.add_argument("--mask", default=None,
        help="path to the mask image")
    ap.add_argument("-r", "--ref-image", dest="ref_image",
        action="append", default=[],
        help="reference image for Flux Kontext models (can be used multiple times)")

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
    samplers = 'euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd, res_multistep, res_2s'
    ap.add_argument("--sampling-method", dest="sample_method", default=None,
        help="sampling method, one of [" + samplers + "] (default: euler for Flux/SD3/Wan, euler_a otherwise)")
    ap.add_argument("--high-noise-sampling-method", dest="high_noise_sample_method", default=None,
        help="(high noise) sampling method, one of [" + samplers + "] (default: euler for Flux/SD3/Wan, euler_a otherwise)")
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
        output_ext = os.path.splitext(args_dict['output'])[-1].lower()
        if output_ext == 'png':
            args_dict["output_format"] = 'png'
        elif output_ext in ['.jpg', '.jpeg', '.jpe']:
            args_dict["output_format"] = 'jpeg'

    if not args_dict.get("server_url", "").strip():
        ap.error("--server-url not provided and SD_SERVER_URL env var not found")

    prompt = args_dict.get("prompt", "").strip()
    prompt, lora = extract_lora_tags(prompt)
    if not prompt:
        ap.error("argument -p/--prompt must be non‑empty")
    args_dict["prompt"] = prompt
    if lora:
        args_dict["lora"] = lora

    util_keys = {'verbose', 'server_url', 'output', 'output_begin_idx', 'api', 'init_img', 'mask', 'ref_image', 'output_format'}

    util_opts = {k: v for k, v in args_dict.items() if k in util_keys and v is not None}
    gen_opts = {k: v for k, v in args_dict.items() if k not in util_keys and v is not None}

    return util_opts, gen_opts


def build_openai_parameters(gen_opts, util_opts):
    extension_data = {}
    api_data = {}

    api_keys = {'width', 'height', 'batch_count', 'prompt', 'output_format'}

    for key, value in gen_opts.items():
        if key not in api_keys:
            extension_data[key] = value

    width = gen_opts.get("width")
    height = gen_opts.get("height")
    # the openai api has no way to specify these separately,
    # so use the extension if we only got one
    if width and height:
        api_data["size"] = f"{width}x{height}"
    elif width:
        extension_data["width"] = width
    elif height:
        extension_data["height"] = height

    if util_opts.get('output_format'):
        api_data['output_format'] = util_opts.get('output_format')

    if gen_opts.get("batch_count"):
        api_data["n"] = gen_opts["batch_count"]

    prompt = gen_opts['prompt']
    json_payload = json.dumps(extension_data)
    api_data["prompt"] = f"{prompt}<sd_cpp_extra_args>{json_payload}</sd_cpp_extra_args>"

    api_data["model"] = "sd.cpp"

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


def encode_image_to_base64(image_path):
    """Encode an image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_images_sdapi(util_opts):
    image_opts = {}
    init_img_path = util_opts.get("init_img")
    if init_img_path:
        try:
            image_opts["init_img"] = encode_image_to_base64(init_img_path)
        except FileNotFoundError:
            raise ValueError(f"Init image file not found: {init_img_path}")
        except Exception as e:
            raise ValueError(f"Failed to encode init image: {e}")

    mask_path = util_opts.get("mask")
    if mask_path:
        try:
            image_opts["mask"] = encode_image_to_base64(mask_path)
        except FileNotFoundError:
            raise ValueError(f"Mask image file not found: {mask_path}")
        except Exception as e:
            raise ValueError(f"Failed to encode mask image: {e}")

    ref_images = []
    for ref_image_path in util_opts.get("ref_image", []):
        try:
            ref_images.append(encode_image_to_base64(ref_image_path))
        except FileNotFoundError:
            raise ValueError(f"Reference image file not found: {ref_image_path}")
        except Exception as e:
            raise ValueError(f"Failed to encode reference image: {e}")
    if ref_images:
        image_opts["ref_image"] = ref_images

    return image_opts


def load_images_openai(util_opts, verbose=False):
    image_opts = {}

    image_paths = []

    init_img_path = util_opts.get("init_img")
    if init_img_path:
        if verbose:
            print('warning: the openai api does not support a separate initial image; adding it as the first reference image')
        image_paths.append(init_img_path)

    image_paths.extend(util_opts.get("ref_image", []))

    if image_paths:
        images = []
        for image_path in image_paths:
            try:
                images.append(open(image_path, "rb"))
            except FileNotFoundError:
                raise ValueError(f"Reference image file not found: {image_path}")
        image_opts["image"] = images

    mask_path = util_opts.get("mask")
    if mask_path:
        try:
            image_opts["mask"] = open(mask_path, "rb")
        except FileNotFoundError:
            raise ValueError(f"Mask image file not found: {mask_path}")

    return image_opts


def build_sdapi_parameters(gen_opts, util_opts, image_opts={}):
    same_keys = ["prompt", "negative_prompt", "width", "height",
        "steps", "cfg_scale", "seed", "scheduler", "clip_skip"]
    translated_keys = [
        ("batch_size", "batch_count"),
        ("sampler_name", "sample_method"),
        ("denoising_strength", "strength"),
    ]
    params = {}
    for key in same_keys:
        params[key] = gen_opts.get(key)
    for dkey, okey in translated_keys:
        params[dkey] = gen_opts.get(okey)

    payload = {k: v for k, v in params.items() if v is not None}

    init_img = image_opts.get('init_img')
    if init_img:
        payload["init_images"] = [init_img]

        mask = image_opts.get("mask")
        if mask:
            payload["mask"] = mask

    ref_image = image_opts.get("ref_image")
    if ref_image:
        payload["extra_images"] = ref_image

    # use server defaults
    if 'seed' in payload and payload["seed"] < 0:
        del payload["seed"]

    return payload


def decode_sdapi_response(response_body):
    try:
        data = json.loads(response_body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")

    if 'images' not in data:
        raise ValueError(f"Unexpected response format (no 'images' key)")

    images = data['images']
    decoded_images = []

    for i, b64_data in enumerate(images):
        if not b64_data:
            raise ValueError(f"No image data found for item {i}")
        try:
            decoded_images.append(base64.b64decode(b64_data))
        except base64.binascii.Error as e:
            raise ValueError(f"Failed to decode base64 data for item {i}: {e}")

    return decoded_images


def truncate_for_debug(obj, max_length=512):
    if isinstance(obj, dict):
        return {k: truncate_for_debug(v, max_length) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [truncate_for_debug(item, max_length) for item in obj]
    elif isinstance(obj, str) and len(obj) > max_length:
        return f"{obj[:max_length]}... ({len(obj)} chars total)"
    else:
        return obj


def do_request(url, data, headers):

    if HAS_REQUESTS:
        response = requests.post(url, headers=headers, data=data, timeout=600)
        if response.status_code != 200:
            print(f"HTTP {response.status_code}: {response.reason}")
            print(f"     {response.text}")
        response.raise_for_status()
        return response.text

    else:
        req_data = data.encode('utf-8') if isinstance(data, str) else data
        req = urllib.request.Request(url, data=req_data, headers=headers)
        try:
            with urllib.request.urlopen(req) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.reason}")
            body = e.read().decode("utf-8", errors='replace')
            if body:
                print(f"Server error details: {body}")
            else:
                print(f"No server error details")
            raise
        except urllib.error.URLError as e:
            print(f"URL error: {e.reason}")
            raise

def do_request_form(url, form_data, form_files, headers):

    if HAS_REQUESTS:
        response = requests.post(url, headers=headers, data=form_data, files=form_files, timeout=30)
        response.raise_for_status()
        return response.text

    else:
        raise NotImplementedError("missing urllib implementation for form data")

def main_sdapi(util_opts, gen_opts, verbose=False):

    server_url = util_opts.get("server_url")

    image_opts = load_images_sdapi(util_opts)

    if image_opts.get('init_img'):
        endpoint = urllib.parse.urljoin(server_url, "sdapi/v1/img2img")
    else:
        endpoint = urllib.parse.urljoin(server_url, "sdapi/v1/txt2img")

    api_parameters = build_sdapi_parameters(gen_opts, util_opts, image_opts)
    lora = gen_opts.get('lora')
    if lora:
        # TODO: refactor, error handling, is_high_noise
        lora_list = json.loads(requests.get(urllib.parse.urljoin(server_url, "sdapi/v1/loras")).text)
        #print(f"remote lora list: {json.dumps(lora_list, indent=2)}")
        lora_map = {l.get("name"): l.get("path") for l in lora_list}
        req_lora = []
        print(f"requesting LoRAs")
        for llora in lora:
            name = llora['name']
            rlora = lora_map.get(name)
            if rlora:
                entry = {'path': rlora, 'multiplier': llora["multiplier"]}
                if verbose:
                    print(f"  lora '{name}' mapped to '{rlora}'")
                req_lora.append(entry)
            else:
                if verbose:
                    print(f"  warning: lora '{name}' not found on remote")
        if req_lora:
            api_parameters["lora"] = req_lora

    if verbose:
        print(f"Using sdapi")
        print(f"Sending request to: {endpoint}")
        print(f"Payload: {json.dumps(truncate_for_debug(api_parameters), indent=2)}")

    req_data = json.dumps(api_parameters)
    headers = {'Content-Type': 'application/json'}
    response_body = do_request(endpoint, req_data, headers)

    try:
        images = decode_sdapi_response(response_body)
    except ValueError as e:
        print(f"Error decoding response: {e}")
        raise

    return images


def main_openai(util_opts, gen_opts, verbose=False):

    server_url = util_opts.get("server_url")
    api_parameters = build_openai_parameters(gen_opts, util_opts)
    image_opts = load_images_openai(util_opts, verbose)

    headers = {"Authorization": "Bearer local-api-key"}

    files = []
    for i, image in enumerate(image_opts.get('image', [])):
        files.append(('image[]', (f'image_{i}.png', image)))

    if files:
        endpoint = urllib.parse.urljoin(server_url, "v1/images/edits")

        mask = image_opts.get('mask')
        if mask:
            files.append(('mask', ('mask.png', mask)))

        if verbose:
            print(f"Using OpenAI API (form)")
            print(f"Sending request to: {endpoint}")
            print(f"Payload: {json.dumps(api_parameters, indent=2)}")
            print(f"         {len(files)} files")

        response_body = do_request_form(endpoint, api_parameters, files, headers)

    else:
        endpoint = urllib.parse.urljoin(server_url, "v1/images/generations")
        headers['Content-Type'] = 'application/json'
        req_data = json.dumps(api_parameters)

        if verbose:
            print(f"Using OpenAI API")
            print(f"Sending request to: {endpoint}")
            print(f"Payload: {json.dumps(api_parameters, indent=2)}")

        req_data = json.dumps(api_parameters)
        response_body = do_request(endpoint, req_data, headers)

    try:
        images = decode_openai_response(response_body)
    except ValueError as e:
        print(f"Error decoding response: {e}")
        raise

    return images


def main_openai_module(util_opts, gen_opts, verbose=False):

    import openai

    server_url = util_opts.get("server_url")
    base_url = urllib.parse.urljoin(server_url, "v1")

    openai_version = openai.version.VERSION
    if verbose:
        print(f"Using OpenAI module {openai_version}")

    api_parameters = build_openai_parameters(gen_opts, util_opts)

    output_format = api_parameters.get('output_format')
    if output_format:
        # the output_format parameter was added to the openai module only in version
        # 1.76; for simplicity, always request it through the extra_body parameter
        api_parameters['extra_body'] = {"output_format": output_format}
        del api_parameters['output_format']

    image_opts = load_images_openai(util_opts, verbose)
    image = image_opts.get('image')
    if image:
        # XXX 'old' openai module versions only accept a single image
        api_parameters['image'] = image[0]
        mask = image_opts.get('mask')
        if mask:
            api_parameters['mask'] = mask

    if verbose:
        print(f"Base URL: {base_url}")
        print(f"Parameters: {api_parameters}")

    client = openai.OpenAI(api_key="local-api-key", base_url=base_url)
    if image:
        result = client.images.edit(**api_parameters)
    else:
        result = client.images.generate(**api_parameters)
    images = [base64.b64decode(img.b64_json) for img in result.data]

    return images


def save_images(image_list, util_opts):
    verbose = util_opts.get("verbose", False)
    output = util_opts.get("output", "./output.png")
    output_begin_idx = util_opts.get("output_begin_idx")

    dirname, filename = os.path.split(output)

    format_specifier = re.search(r'%\d*d', filename)

    # same logic as sd-cli
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
    api = util_opts.get("api") or apis[0]

    if api == 'sdapi':
        images = main_sdapi(util_opts, gen_opts, verbose)

    if api == 'openai':
        images = main_openai(util_opts, gen_opts, verbose)

    if api == 'openai-module':
        images = main_openai_module(util_opts, gen_opts, verbose)

    save_images(images, util_opts)


if __name__ == "__main__":
    main()

