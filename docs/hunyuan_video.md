# HunyuanVideo 1.5

HunyuanVideo 1.5 uses a HunyuanVideo diffusion transformer, a causal video VAE, Qwen2.5-VL 7B for the main text conditioning, 
and ByT5 Small GlyphXL for glyph-aware text conditioning.

## Download weights

- Download HunyuanVideo 1.5
    - safetensors: https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main/split_files/diffusion_models
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main/split_files/vae
- Download qwen_2.5_vl 7b
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders
    - gguf: https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF/tree/main
- Download byt5 small glyphxl
    - safetensros: https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main/split_files/text_encoders

## Text-to-video example

```shell
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model  ..\models\diffusion_models\hunyuanvideo1.5_720p_t2v_fp16.safetensors --vae ..\models\vae\hunyuanvideo15_vae_fp16.safetensors --llm ..\models\text_encoders\qwen_2.5_vl_7b.safetensors --t5xxl ..\models\text_encoders\byt5_small_glyphxl_fp16.safetensors  -p "a lovely cat" --cfg-scale 6.0 --sampling-method euler -v -W 1280 -H 720 --offload-to-cpu --diffusion-fa --video-frames 33 --vae-tiling
```

<video src=../assets/hunyuan_video/hy1.5_t2v.mp4 controls="controls" muted="muted" type="video/mp4"></video>