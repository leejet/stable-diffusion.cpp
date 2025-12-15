"""
Image upload and editing endpoints with proper asyncio.time() fix
"""
import os
import base64
import random
import time  # Use standard time module
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageFile
from handlers import image_editor
from models import (
    get_all_models, DEFAULT_DIFFUSION_MODEL, DEFAULT_WIDTH, DEFAULT_HEIGHT,
    DEFAULT_STEPS, DEFAULT_CFG, DEFAULT_SAMPLER, DEFAULT_OFFLOAD,
    DEFAULT_VAE_TILING, DEFAULT_FLASH_ATTN, TMP_IMG_PATH, cleanup_tmp_img
)

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

class OpenAIImageResponse(BaseModel):
    created: int
    data: list[dict]

@router.post("/v1/images/edits")
async def edit_image_openai(
    image_file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    img: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    response_format: Optional[str] = Form("url"),
    negative_prompt: Optional[str] = Form(""),
    steps: Optional[int] = Form(None),
    cfg_scale: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    sampling_method: Optional[str] = Form(None),
    strength: Optional[float] = Form(0.75),
    llm: Optional[str] = Form(None),
    vae: Optional[str] = Form(None),
    clip_l: Optional[str] = Form(None),
    t5: Optional[str] = Form(None),
    offload: Optional[bool] = Form(None),
    tiling: Optional[bool] = Form(None),
    flash_attn: Optional[bool] = Form(None),
):
    """Edit image with prompt (OpenAI API compatible)"""

    # Find uploaded image
    image_data = None
    for potential_file in [image_file, image, img, file]:
        if potential_file and potential_file.filename:
            try:
                image_data = await potential_file.read()
                print(f"✅ Image loaded: {potential_file.filename} ({len(image_data)} bytes)")
                break
            except Exception as e:
                print(f"⚠ Error reading file {potential_file.filename}: {e}")
                continue

    if not image_data:
        raise HTTPException(400, detail="No image file provided")

    # Save & validate image
    try:
        with open(TMP_IMG_PATH, "wb") as f:
            f.write(image_data)

        # Verify image
        try:
            with Image.open(TMP_IMG_PATH) as pil_img:
                pil_img.load()
                print(f"✅ Image validated: {pil_img.size}")
        except Exception as img_err:
            print(f"⚠ Image warning: {img_err}")

    except Exception as e:
        cleanup_tmp_img()
        raise HTTPException(400, detail=f"Cannot process image: {str(e)[:80]}")

    try:
        # Setup parameters
        models = get_all_models()
        diffusion_model = next(
            (m for m in models if model == os.path.basename(m) or model in m),
            DEFAULT_DIFFUSION_MODEL,
        ) if model else DEFAULT_DIFFUSION_MODEL

        if not diffusion_model or diffusion_model == "NO_MODEL_FOUND":
            raise HTTPException(400, detail="No valid model available")

        width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
        if size:
            try:
                w, h = size.lower().split("x")
                width, height = int(w), int(h)
            except:
                pass

        # Generate with working parameters
        output_path = await image_editor.edit_with_prompt(
            prompt=prompt,
            init_img=TMP_IMG_PATH,
            diffusion_model=diffusion_model,
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            steps=steps or 20,
            cfg_scale=cfg_scale or 7.0,
            seed=seed or random.randint(0, 2**32 - 1),
            sampling_method=sampling_method or "euler",
            strength=strength,
            vae=vae,
            llm=llm,
            clip_l=clip_l,
            t5=t5,
            offload=offload or False,
            tiling=tiling or False,
            flash_attn=flash_attn or True,
        )

        # FIXED: Use time.time() instead of asyncio.time()
        if response_format == "b64_json":
            with open(output_path, "rb") as f:
                data = [{"b64_json": base64.b64encode(f.read()).decode()}]
        else:
            filename = os.path.basename(output_path)
            data = [{"url": f"http://127.0.0.1:8000/images/{filename}"}]

        return OpenAIImageResponse(created=int(time.time()), data=data)  # FIXED

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        raise HTTPException(500, detail=f"Generation failed: {str(e)[:200]}")
    finally:
        cleanup_tmp_img()
