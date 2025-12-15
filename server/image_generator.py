"""
Image Generation Handler for SD.CPP
Handles text-to-image and image-to-image generation
"""
import os
import asyncio
import uuid
import random
from typing import Optional
from fastapi import HTTPException
class ImageGenerator:
    def __init__(self, sd_binary: str, output_dir: str, gpu_semaphore: asyncio.Semaphore):
        self.sd_binary = sd_binary
        self.output_dir = output_dir
        self.gpu_semaphore = gpu_semaphore
    async def generate(
        self,
        prompt: str,
        diffusion_model: str,
        negative_prompt: Optional[str] = None,
        width: int = 768,
        height: int = 768,
        steps: int = 30,
        cfg_scale: float = 2.5,
        seed: Optional[int] = None,
        sampling_method: str = "dpm++2m",
        init_img: Optional[str] = None,
        strength: Optional[float] = None,
        vae: Optional[str] = None,
        t5: Optional[str] = None,
        clip_l: Optional[str] = None,
        llm: Optional[str] = None,
        offload: bool = False,
        tiling: bool = False,
        flash_attn: bool = True,
        mode: str = "img",
        video_frames: int = 33,
        flow_shift: float = 3.0,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Generate an image using SD.CPP
        Args:
            prompt: Text prompt for generation
            diffusion_model: Path to the main diffusion model
            negative_prompt: Negative prompt (optional)
            width: Image width
            height: Image height
            steps: Number of sampling steps
            cfg_scale: CFG scale
            seed: Random seed (None for random)
            sampling_method: Sampling method to use
            init_img: Path to input image for img2img (optional)
            strength: Strength for img2img (0.0-1.0)
            vae: Path to VAE model (optional)
            t5: Path to T5 text encoder (optional)
            clip_l: Path to CLIP text encoder (optional)
            llm: Path to LLM model (optional)
            offload: Enable CPU offloading
            tiling: Enable VAE tiling
            flash_attn: Enable flash attention
            mode: Generation mode ('img' or 'vid')
            video_frames: Number of frames for video generation
            flow_shift: Flow shift parameter for video
            output_dir: Optional custom output directory
        Returns:
            Path to generated image/video file
        """
        async with self.gpu_semaphore:
            # Use custom output_dir if provided, otherwise use default
            actual_output_dir = output_dir if output_dir else self.output_dir

            cmd = [self.sd_binary]
            # Mode selection
            if mode == "vid":
                cmd += ["-M", "vid_gen"]
            else:
                cmd += ["-M", "img_gen"]
            # Model configuration
            cmd += ["--diffusion-model", diffusion_model]
            # Optional external components
            for key, path in {
                "vae": vae,
                "t5xxl": t5,
                "clip_l": clip_l,
                "llm": llm,
            }.items():
                if path and path != "None" and os.path.exists(path):
                    cmd += [f"--{key}", path]
            # Prompt
            cmd += ["-p", prompt]
            if negative_prompt:
                cmd += ["-n", negative_prompt]
            # Image dimensions
            cmd += ["-H", str(height), "-W", str(width)]
            # Sampling parameters
            cmd += ["--steps", str(steps)]
            cmd += ["--cfg-scale", str(cfg_scale)]
            cmd += ["--sampling-method", sampling_method]
            # Seed
            if seed is not None:
                cmd += ["--seed", str(seed)]
            else:
                cmd += ["--seed", str(random.randint(0, 2**32 - 1))]
            # Input image (img2img)
            if init_img and os.path.exists(init_img):
                cmd += ["-i", init_img]
                if strength is not None and mode == "img":
                    cmd += ["--strength", str(strength)]
            # Video-specific parameters
            if mode == "vid":
                cmd += ["--video-frames", str(video_frames)]
                cmd += ["--flow-shift", str(flow_shift)]
            # Hardware options
            if offload:
                cmd += ["--offload-to-cpu"]
            if tiling:
                cmd += ["--vae-tiling"]
            if flash_attn:
                cmd += ["--diffusion-fa"]
            # Output path
            ext = ".mp4" if mode == "vid" else ".png"
            output_path = os.path.join(
                actual_output_dir, f"output_{uuid.uuid4().hex[:8]}{ext}"
            )
            cmd += ["-o", output_path]
            print("->", " ".join(cmd))
            # Execute command
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0 or not os.path.exists(output_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"SD generation failed: {stderr.decode(errors='ignore')[-500:]}",
                )
            return output_path
