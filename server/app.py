"""
Main FastAPI application with Gradio UI
SD.CPP Server - Fully Modularized Edition
"""
import os
import glob
import asyncio
import base64
import random
import json
import logging
from typing import Optional, List
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse
import gradio as gr
from PIL import Image
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Import routers and shared components
from gallery import router as gallery_router
from api import router as api_router
from upload import router as upload_router
from models import (
    OUTPUT_DIR, TMP_IMG_PATH, get_base_models, get_text_encoder_models,
    get_vae_models, load_settings, save_settings, get_setting, cleanup_tmp_img,
    DEFAULT_STEPS, DEFAULT_CFG, DEFAULT_SAMPLER, DEFAULT_OFFLOAD,
    DEFAULT_VAE_TILING, DEFAULT_FLASH_ATTN
)
from handlers import image_generator, image_editor
# ======================
# FASTAPI APP SETUP
# ======================
app = FastAPI(title="SD.CPP Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")
# Include all routers
app.include_router(gallery_router)
app.include_router(api_router)
app.include_router(upload_router)
# ======================
# FIXED EXCEPTION HANDLERS
# ======================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom handler to prevent bytes-to-JSON encoding crash"""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request format. Use multipart/form-data with 'prompt' and image file."}
    )
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP error: {exc}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
@app.get("/health")
async def health():
    """Health check endpoint"""
    from models import get_all_models
    return {"status": "ok", "models": len(get_all_models())}
# ======================
# FIXED process_ui()
# ======================
def process_ui(
    diff, vae, t5, clip, llm, vis, w, h, steps, cfg, seed, samp, off, tile, fa, temp,
    mode_key, p, n, img, str_val, frames, flow, reps
):
    """Process Gradio UI requests - ALWAYS returns 3 outputs"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        saved_path = None
        if img is not None:
            img.save(TMP_IMG_PATH)
            saved_path = TMP_IMG_PATH

        # Set output directory based on temp checkbox
        if temp:
            output_dir = "/tmp"
        else:
            output_dir = OUTPUT_DIR

        seed_int = int(seed) if seed and str(seed).strip() else None
        if seed_int is None:
            seed_int = random.randint(0, 2**32 - 1)

        if mode_key == "Upscale":
            res = loop.run_until_complete(
                image_editor.upscale(
                    diffusion_model=diff,
                    upscale_repeats=reps,
                    offload=off,
                    tiling=tile,
                    flash_attn=fa,
                    output_dir=output_dir
                )
            )
            return (
                gr.update(value=Image.open(res), visible=True),
                gr.update(value=None, visible=False),
                "✅ Upscaling complete!"
            )
        elif mode_key == "Video (Wan)":
            res = loop.run_until_complete(
                image_generator.generate(
                    prompt=p,
                    diffusion_model=diff,
                    negative_prompt=n,
                    init_img=saved_path,
                    width=w,
                    height=h,
                    steps=steps,
                    cfg_scale=cfg,
                    seed=seed_int,
                    sampling_method=samp,
                    vae=vae if vae != "None" else None,
                    t5=t5 if t5 != "None" else None,
                    clip_l=clip if clip != "None" else None,
                    llm=llm if llm != "None" else None,
                    offload=off,
                    tiling=tile,
                    flash_attn=fa,
                    mode="vid",
                    video_frames=frames,
                    flow_shift=flow,
                    output_dir=output_dir
                )
            )
            return (
                gr.update(value=None, visible=False),
                gr.update(value=res, visible=True),
                " Video generation complete!"
            )
        else:  # Image Gen
            res = loop.run_until_complete(
                image_generator.generate(
                    prompt=p,
                    diffusion_model=diff,
                    negative_prompt=n,
                    init_img=saved_path,
                    strength=str_val,
                    width=w,
                    height=h,
                    steps=steps,
                    cfg_scale=cfg,
                    seed=seed_int,
                    sampling_method=samp,
                    vae=vae if vae != "None" else None,
                    t5=t5 if t5 != "None" else None,
                    clip_l=clip if clip != "None" else None,
                    llm=llm if llm != "None" else None,
                    offload=off,
                    tiling=tile,
                    flash_attn=fa,
                    mode="img",
                    output_dir=output_dir
                )
            )
            return (
                gr.update(value=Image.open(res), visible=True),
                gr.update(value=None, visible=False),
                " Image generated successfully!"
            )
    except Exception as e:
        logger.error(f"UI processing error: {str(e)}", exc_info=True)
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            f"❌ Error: {str(e)}"
        )
    finally:
        loop.close()
        cleanup_tmp_img()
# ======================
# GRADIO UI BUILD
# ======================
saved_settings = load_settings()
base_models = get_base_models()
text_encoder_models = get_text_encoder_models()
vae_models = get_vae_models()
text_encoder_extras = ["None"] + text_encoder_models
vae_extras = ["None"] + vae_models
default_main_model = get_setting("main_model", None)
if not default_main_model or default_main_model not in base_models:
    default_main_model = base_models[0] if base_models and base_models[0] != "NO_MODEL_FOUND" else None
with gr.Blocks(title="SD.CPP Workstation") as demo:
    gr.Markdown("# SD.CPP Workstation")
    gr.Markdown("[Open Gallery](/gallery) | [OpenAI API Models](/v1/models)")
    with gr.Group():
        with gr.Row():
            diff = gr.Dropdown(base_models, value=default_main_model, label="Main Model", interactive=True)
            vae = gr.Dropdown(vae_extras, value=get_setting("vae", "None"), label="VAE", interactive=True)
        with gr.Accordion("External Components (Flux/Wan/Z-Image)", open=False):
            with gr.Row():
                llm = gr.Dropdown(text_encoder_extras, value=get_setting("llm", "None"), label="LLM (Z-Image)", interactive=True)
                t5 = gr.Dropdown(text_encoder_extras, value=get_setting("t5", "None"), label="T5 (Flux/Wan)", interactive=True)
                clip = gr.Dropdown(text_encoder_extras, value=get_setting("clip", "None"), label="CLIP (Flux)", interactive=True)
                vis = gr.Dropdown(text_encoder_extras, value=get_setting("vision", "None"), label="Vision (Wan)", interactive=True)
        with gr.Row():
            refresh = gr.Button("Refresh Files")
            save_defaults = gr.Button("Save as Defaults", variant="primary")
        def on_refresh():
            m = get_base_models()
            te = ["None"] + get_text_encoder_models()
            v = ["None"] + get_vae_models()
            return (
                gr.update(choices=m, value=m[0] if m and m[0] != "NO_MODEL_FOUND" else None),
                gr.update(choices=v),
                gr.update(choices=te), gr.update(choices=te),
                gr.update(choices=te), gr.update(choices=te)
            )
        refresh.click(on_refresh, outputs=[diff, vae, llm, t5, clip, vis])
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Image Gen"):
                    prompt = gr.Textbox(label="Prompt", lines=3, value=get_setting("prompt", ""))
                    neg = gr.Textbox(label="Negative", value=get_setting("negative_prompt", ""))
                    with gr.Row():
                        w = gr.Slider(256, 1536, step=64, value=get_setting("width", 512), label="Width")
                        h = gr.Slider(256, 1536, step=64, value=get_setting("height", 512), label="Height")
                    with gr.Row():
                        steps = gr.Slider(1, 60, value=get_setting("steps", 20), label="Steps")
                        cfg = gr.Slider(0, 15, value=get_setting("cfg", 7.0), step=0.1, label="CFG")
                    img_in = gr.Image(label="Input (img2img optional)", type="pil", height=150)
                    strength = gr.Slider(0.0, 1.0, value=get_setting("strength", 0.75), label="Strength (img2img)")
                    b_img = gr.Button("Generate Image", variant="primary")
                with gr.Tab("Video (Wan)"):
                    vid_p = gr.Textbox(label="Prompt", lines=3)
                    vid_n = gr.Textbox(label="Negative")
                    vid_in = gr.Image(label="Input (optional)", type="pil", height=150)
                    with gr.Row():
                        v_frames = gr.Slider(1, 120, value=get_setting("video_frames", 33), label="Frames")
                        v_flow = gr.Slider(0, 10, value=get_setting("flow_shift", 3.0), label="Flow")
                    b_vid = gr.Button("Generate Video", variant="primary")
                with gr.Tab("Upscale"):
                    up_in = gr.Image(label="Input", type="pil")
                    up_reps = gr.Slider(1, 4, value=get_setting("upscale_repeats", 1), label="Repeats")
                    b_up = gr.Button("Upscale", variant="primary")
            with gr.Accordion("Hardware / Seed", open=False):
                seed = gr.Textbox(label="Seed", value=get_setting("seed", ""))
                samp = gr.Dropdown(["euler", "euler_a", "dpm++2m", "lcm"], value=get_setting("sampler", "euler"), label="Sampler")
                off = gr.Checkbox(get_setting("offload", True), label="Offload to CPU")
                tile = gr.Checkbox(get_setting("vae_tiling", True), label="VAE Tiling")
                fa = gr.Checkbox(get_setting("flash_attn", True), label="Flash Attention")
                temp = gr.Checkbox(False, label="Temporary (don't save to gallery)")
        with gr.Column(scale=1):
            out_img = gr.Image(label="Output Image", type="pil")
            out_vid = gr.Video(label="Output Video", visible=False)
            save_status = gr.Textbox(label="Status", visible=True, interactive=False)
    common_settings = [diff, vae, t5, clip, llm, vis, w, h, steps, cfg, seed, samp, off, tile, fa, temp]
    def wrap_img(*args):
        common = args[:16]
        p, n, i, s = args[16:20]
        return process_ui(*common, "Image Gen", p, n, i, s, 0, 0, 1)
    def wrap_vid(*args):
        common = args[:16]
        p, n, i, f, fl = args[16:21]
        return process_ui(*common, "Video (Wan)", p, n, i, 0.75, f, fl, 1)
    def wrap_up(*args):
        common = args[:16]
        i, r = args[16:18]
        return process_ui(*common, "Upscale", "", "", i, 0, 0, 0, r)
    def save_current_settings(
        main_model, vae_model, t5_model, clip_model, llm_model, vis_model,
        width, height, steps_val, cfg_val, seed_val, sampler,
        offload, vae_tiling, flash_attn, temp_val, prompt_text, neg_text, strength_val,
        v_frames_val, v_flow_val, up_reps_val
    ):
        settings = {
            "main_model": main_model, "vae": vae_model, "t5": t5_model, "clip": clip_model,
            "llm": llm_model, "vision": vis_model, "width": width, "height": height,
            "steps": steps_val, "cfg": cfg_val, "seed": seed_val, "sampler": sampler,
            "offload": offload, "vae_tiling": vae_tiling, "flash_attn": flash_attn,
            "temp": temp_val, "prompt": prompt_text, "negative_prompt": neg_text, "strength": strength_val,
            "video_frames": v_frames_val, "flow_shift": v_flow_val, "upscale_repeats": up_reps_val
        }
        if save_settings(settings):
            logger.info(f"Settings saved: {settings}")
            return "Settings saved successfully!"
        return "Failed to save settings"
    # Wire up the buttons
    b_img.click(
        wrap_img,
        inputs=[*common_settings, prompt, neg, img_in, strength, out_img, out_vid],
        outputs=[out_img, out_vid, save_status]
    )
    b_vid.click(
        wrap_vid,
        inputs=[*common_settings, vid_p, vid_n, vid_in, v_frames, v_flow, out_img, out_vid],
        outputs=[out_img, out_vid, save_status]
    )
    b_up.click(
        wrap_up,
        inputs=[*common_settings, up_in, up_reps, out_img, out_vid],
        outputs=[out_img, out_vid, save_status]
    )
    save_defaults.click(
        save_current_settings,
        inputs=[diff, vae, t5, clip, llm, vis, w, h, steps, cfg, seed, samp, off, tile, fa, temp,
                prompt, neg, strength, v_frames, v_flow, up_reps],
        outputs=save_status
    )
# Mount Gradio UI at root
app = gr.mount_gradio_app(app, demo, path="/")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
