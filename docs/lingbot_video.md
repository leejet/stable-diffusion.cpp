# How to Use

Lingbot Video uses a Lingbot diffusion transformer, the Wan2.1 VAE, and Qwen3-VL 4B as the LLM text encoder.

## Download weights

- Download lingbot-video-dense-1.3b
    - safetensors: https://huggingface.co/robbyant/lingbot-video-dense-1.3b/tree/main/transformer
- Download lingbot-video-moe-30b-a3b
    - safetensors: https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b/tree/main/transformer
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors
- Download Qwen3-VL 4B
    - safetensors: https://huggingface.co/Comfy-Org/Krea-2/tree/main/text_encoders
    - gguf: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/tree/main

## Notes

- Use `-M vid_gen`.
- T2V uses the text prompt only.
- I2V uses `-i` as the first video frame. The same image is also passed to the
  Qwen3-VL prompt enhancer when vision weights are available.
- Video frames are aligned to Wan-style temporal compression, so use frame counts
  like 33, 49, or 81.

## Examples

### LingBot-Video T2V

```sh
.\bin\Release\sd-cli.exe -M vid_gen --diffusion-model ..\models\diffusion_models\lingbot-video-dens-1.3b.safetensors --llm ..\models\text_encoders\Qwen3-VL-4B-Instruct-Q4_K_M.gguf --vae ..\models\vae\wan_2.1_vae.safetensors -p '{"caption":{"comprehensive_description":"A lovely cat sits comfortably on a soft cushion near a sunlit window, looking calm, gentle, and adorable. The cat has soft fluffy fur, bright expressive eyes, small rounded ears, delicate whiskers, and a relaxed posture. Warm daylight falls across the cat from one side, creating soft highlights on the fur and a cozy glow around the scene. The background is softly blurred, showing hints of a peaceful indoor home environment with warm tones and gentle natural light. The overall atmosphere is cute, tender, serene, and photorealistic, emphasizing the cat''s charming appearance, soft texture, and affectionate presence.","camera_info":{"color":"Warm","frame_size":"Close Up","shot_type_angle":"Eye level","lens_size":"Medium Lens","composition":"Centered balanced","lighting":"Soft light","lighting_type":"Daylight"},"world_knowledge":[],"prominent_elements":[{"name":"lovely cat","description":"A cute and gentle domestic cat with soft fluffy fur, expressive eyes, and a calm relaxed presence.","location":"center of the frame","relative_size":"large","shape_and_color":"Small animal body with rounded face, triangular ears, bright eyes, and soft fur in warm natural tones","texture":"soft, fluffy, silky","appearance_details":"The cat has clean well-groomed fur, delicate whiskers, small ears, a cute nose, and bright attentive eyes. Its expression appears calm, affectionate, and slightly curious.","relationship":"Acts as the main subject and emotional focal point of the scene.","orientation":"facing the camera","pose":"sitting comfortably with a relaxed posture","expression":"gentle, adorable, calm, slightly curious","clothing":"","gender":"","skin_tone_and_texture":""},{"name":"cat eyes","description":"Bright expressive eyes that give the cat a sweet and affectionate appearance.","location":"upper center of the cat face","relative_size":"small","shape_and_color":"Round almond-like eyes with glossy reflections","texture":"clear, glossy, reflective","appearance_details":"The eyes catch the soft daylight, creating small natural highlights that make the cat look vivid and alive.","relationship":"Enhance the emotional charm and cuteness of the cat.","orientation":"looking toward the camera","pose":"","expression":"soft and attentive","clothing":"","gender":"","skin_tone_and_texture":""},{"name":"soft cushion","description":"A comfortable cushion or blanket where the cat is resting.","location":"bottom portion of the frame","relative_size":"medium","shape_and_color":"Soft rounded fabric surface in light warm neutral tones","texture":"plush, fabric, cozy","appearance_details":"The cushion gently supports the cat and adds a comfortable home-like feeling to the scene.","relationship":"Provides a cozy resting place for the cat.","orientation":"horizontal","pose":"","expression":"","clothing":"","gender":"","skin_tone_and_texture":""},{"name":"sunlit indoor background","description":"A softly blurred indoor background with warm daylight and peaceful home atmosphere.","location":"behind the cat, filling the upper and side areas of the frame","relative_size":"large","shape_and_color":"Soft abstract shapes in warm beige, cream, and pale golden tones","texture":"soft, blurry, bokeh-like","appearance_details":"The background is intentionally out of focus, keeping attention on the cat while creating a cozy and serene mood.","relationship":"Provides a warm and gentle environment that supports the cute domestic scene.","orientation":"upright","pose":"","expression":"","clothing":"","gender":"","skin_tone_and_texture":"","is_cluster":true,"number_of_objects":"numerous"}]}}' -n '{"universal_negative":{"visual_quality":["low quality","worst quality","blurry","pixelated","jpeg artifacts","low resolution","unstable color","color flicker","underexposed","overexposed","invisible subject","subject hidden in darkness"],"artistic_style":["painting","illustration","drawing","cartoon","3d render","cgi","sketch","digital art"],"composition_and_content":["text","watermark","signature","logo","subtitles","pillarboxed","side bars","portrait image in landscape frame"],"temporal_and_motion_stability":["flickering","jittery","motion blur","temporal inconsistency","warping","morphing","incoherent motion","unnatural movement","static object with sudden jump","frame-to-frame inconsistency"],"material_and_structure":["plastic-like glass","unrealistic texture","deformed bottle","liquid freezing improperly","distorted reflections"]}}' --diffusion-fa --offload-to-cpu --cfg-scale 3 --video-frames 33 -v
```
