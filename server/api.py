"""
OpenAI-compatible API endpoints
/v1/models, /v1/images/generations
"""
import os
import base64
import random
import asyncio
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel
from models import (
    get_all_models,
    DEFAULT_DIFFUSION_MODEL,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_STEPS,
    DEFAULT_CFG,
    DEFAULT_SAMPLER,
    DEFAULT_LLM,
    DEFAULT_VAE,
    DEFAULT_OFFLOAD,
    DEFAULT_VAE_TILING,
    DEFAULT_FLASH_ATTN,
)
from handlers import image_generator, image_editor

router = APIRouter()

# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================
class OpenAIImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: Optional[int] = 1
    size: Optional[str] = "768x768"
    response_format: Optional[str] = "url"
    negative_prompt: Optional[str] = ""
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    seed: Optional[int] = None
    sampling_method: Optional[str] = None
    llm: Optional[str] = None
    vae: Optional[str] = None
    clip_l: Optional[str] = None
    t5: Optional[str] = None
    offload: Optional[bool] = None
    tiling: Optional[bool] = None
    flash_attn: Optional[bool] = None

class OpenAIImageResponse(BaseModel):
    created: int
    data: List[dict]

# ============================================================
# MODEL LIST ENDPOINT
# ============================================================
@router.get("/v1/models")
async def list_models_openai():
    """List available models (OpenAI API compatible)"""
    return {"object": "list", "data": get_model_list_for_openai()}

def get_model_list_for_openai():
    """Format model list for OpenAI API compatibility"""
    models = get_all_models()
    return [
        {
            "id": os.path.basename(model),
            "object": "model",
            "created": 1686935002,
            "owned_by": "stable-diffusion",
        }
        for model in models
        if model != "NO_MODEL_FOUND"
    ]

# ============================================================
# UNIFIED IMAGE GENERATION ENDPOINT (handles both txt2img and img2img)
# ============================================================
@router.post("/v1/images/generations")
async def create_image_openai(
    request: Request,
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    size: Optional[str] = Form("768x768"),
    response_format: Optional[str] = Form("url"),
    negative_prompt: Optional[str] = Form(""),
    steps: Optional[int] = Form(None),
    cfg_scale: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    sampling_method: Optional[str] = Form(None),
    llm: Optional[str] = Form(None),
    vae: Optional[str] = Form(None),
    clip_l: Optional[str] = Form(None),
    t5: Optional[str] = Form(None),
    offload: Optional[bool] = Form(None),
    tiling: Optional[bool] = Form(None),
    flash_attn: Optional[bool] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """Generate image from text prompt or edit existing image (OpenAI API compatible)"""
    # ==============================
    # Parse size (OpenAI style "512x512")
    # ==============================
    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    if size:
        try:
            w, h = size.lower().split("x")
            width, height = int(w), int(h)
        except Exception:
            pass

    # ==============================
    # Select diffusion model
    # ==============================
    models = get_all_models()
    diffusion_model = DEFAULT_DIFFUSION_MODEL
    if model:
        diffusion_model = next(
            (m for m in models if model == os.path.basename(m) or model in m),
            DEFAULT_DIFFUSION_MODEL,
        )
    if not diffusion_model or diffusion_model == "NO_MODEL_FOUND":
        raise HTTPException(status_code=400, detail="No models available")

    # ==============================
    # Apply defaults
    # ==============================
    steps = steps if steps is not None else DEFAULT_STEPS
    cfg_scale = cfg_scale if cfg_scale is not None else DEFAULT_CFG
    sampling_method = sampling_method or DEFAULT_SAMPLER
    offload = offload if offload is not None else DEFAULT_OFFLOAD
    tiling = tiling if tiling is not None else DEFAULT_VAE_TILING
    flash_attn = flash_attn if flash_attn is not None else DEFAULT_FLASH_ATTN

    # Deterministic or random seed
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    # ==============================
    # Handle image upload for img2img
    # ==============================
    init_img_path = None
    if image:
        try:
            import tempfile
            import uuid
            from PIL import Image

            img_ext = os.path.splitext(image.filename)[1] or '.png'
            temp_img_path = os.path.join("/tmp", f"upload_{uuid.uuid4().hex}{img_ext}")

            with open(temp_img_path, "wb") as f:
                f.write(await image.read())

            with Image.open(temp_img_path) as img:
                img.load()
                print(f"✅ Image validated: {img.size}")

            init_img_path = temp_img_path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)[:100]}")

    # ==============================
    # Generate image(s)
    # ==============================
    output_paths = []
    try:
        for _ in range(n or 1):
            output_path = await image_generator.generate(
                prompt=prompt,
                diffusion_model=diffusion_model,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                sampling_method=sampling_method,
                llm=llm or DEFAULT_LLM,
                vae=vae or DEFAULT_VAE,
                clip_l=clip_l,
                t5=t5,
                offload=offload,
                tiling=tiling,
                flash_attn=flash_attn,
                mode="img",
                init_img=init_img_path
            )
            output_paths.append(output_path)
    except Exception as e:
        if init_img_path and os.path.exists(init_img_path):
            os.remove(init_img_path)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        if init_img_path and os.path.exists(init_img_path):
            os.remove(init_img_path)

    # ==============================
    # Format Response
    # ==============================
    base_url = str(request.base_url).rstrip("/")
    images_data = []
    for output_path in output_paths:
        if response_format == "b64_json":
            with open(output_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            images_data.append({"b64_json": img_b64})
        else:
            filename = os.path.basename(output_path)
            image_url = f"{base_url}/images/{filename}"
            images_data.append({"url": image_url})

    return {"created": int(asyncio.get_event_loop().time()), "data": images_data}
