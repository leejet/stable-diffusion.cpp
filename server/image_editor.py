"""
Image Editing Handler for SD.CPP
Handles image editing operations including upscaling
"""
import os
import asyncio
import uuid
from typing import Optional
from fastapi import HTTPException
class ImageEditor:
    def __init__(self, sd_binary: str, output_dir: str, gpu_semaphore: asyncio.Semaphore):
        self.sd_binary = sd_binary
        self.output_dir = output_dir
        self.gpu_semaphore = gpu_semaphore
    async def upscale(
        self,
        diffusion_model: str,
        upscale_repeats: int = 1,
        offload: bool = False,
        tiling: bool = False,
        flash_attn: bool = True,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Upscale an image using SD.CPP upscale model
        Args:
            diffusion_model: Path to the upscale model
            upscale_repeats: Number of upscale iterations
            offload: Enable CPU offloading
            tiling: Enable VAE tiling
            flash_attn: Enable flash attention
            output_dir: Optional custom output directory
        Returns:
            Path to upscaled image file
        """
        async with self.gpu_semaphore:
            # Use custom output_dir if provided, otherwise use default
            actual_output_dir = output_dir if output_dir else self.output_dir

            cmd = [self.sd_binary]
            # Upscale mode
            cmd += ["-M", "upscale"]
            cmd += ["--upscale-model", diffusion_model]
            cmd += ["--upscale-repeats", str(upscale_repeats)]
            # Hardware options
            if offload:
                cmd += ["--offload-to-cpu"]
            if tiling:
                cmd += ["--vae-tiling"]
            if flash_attn:
                cmd += ["--diffusion-fa"]
            # Output path
            output_path = os.path.join(
                actual_output_dir, f"output_{uuid.uuid4().hex[:8]}.png"
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
                    detail=f"SD upscale failed: {stderr.decode(errors='ignore')[-500:]}",
                )
            return output_path
    async def edit_with_prompt(
        self,
        prompt: str,
        init_img: str,
        diffusion_model: str,
        negative_prompt: Optional[str] = None,
        width: int = 768,
        height: int = 768,
        steps: int = 30,
        cfg_scale: float = 2.5,
        seed: Optional[int] = None,
        sampling_method: str = "dpm++2m",
        strength: float = 0.75,
        vae: Optional[str] = None,
        t5: Optional[str] = None,
        clip_l: Optional[str] = None,
        llm: Optional[str] = None,
        offload: bool = False,
        tiling: bool = False,
        flash_attn: bool = True,
    ) -> str:
        """
        Edit an image using a text prompt (img2img with prompt)
        Args:
            prompt: Text prompt for editing
            init_img: Path to input image
            diffusion_model: Path to the main diffusion model
            negative_prompt: Negative prompt (optional)
            width: Output image width
            height: Output image height
            steps: Number of sampling steps
            cfg_scale: CFG scale
            seed: Random seed (None for random)
            sampling_method: Sampling method to use
            strength: How much to transform the input (0.0-1.0)
            vae: Path to VAE model (optional)
            t5: Path to T5 text encoder (optional)
            clip_l: Path to CLIP text encoder (optional)
            llm: Path to LLM model (optional)
            offload: Enable CPU offloading
            tiling: Enable VAE tiling
            flash_attn: Enable flash attention
        Returns:
            Path to edited image file
        """
        # Import here to avoid circular dependency
        from image_generator import ImageGenerator
        generator = ImageGenerator(self.sd_binary, self.output_dir, self.gpu_semaphore)
        return await generator.generate(
            prompt=prompt,
            diffusion_model=diffusion_model,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            sampling_method=sampling_method,
            init_img=init_img,
            strength=strength,
            vae=vae,
            t5=t5,
            clip_l=clip_l,
            llm=llm,
            offload=offload,
            tiling=tiling,
            flash_attn=flash_attn,
            mode="img"
        )
