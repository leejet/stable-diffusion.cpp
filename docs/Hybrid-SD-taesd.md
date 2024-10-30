## Using TAESD & TAESDXL to faster decoding

We'd like to introduce our recent work [Hybrid-SD](https://github.com/bytedance/Hybrid-SD.git). It provides a [Hybrid-sd-tinyvae](https://huggingface.co/cqyan/hybrid-sd-tinyvae) TinyAutoencoder and a [Hybrid-sd-tinyvae-xl](https://huggingface.co/cqyan/hybrid-sd-tinyvae-xl) TinyAutoencoder, which are finetuned version of [TAESD](https://huggingface.co/madebyollin/taesd) and [TAESDXL](https://huggingface.co/madebyollin/taesdxl) respectively. We aim to fix the low-saturation and low-resolution problem in SD1.x and SDXL, while narrowing the loss from using smaller decoders. You can use our models to accelerate the decoding of latent images in base model SD1.x or SDXL by following these steps:


### Hybrid-sd-tinyvae for SD1.x
Download the model weight [Hybrid-sd-tinyvae](https://huggingface.co/cqyan/hybrid-sd-tinyvae) 
```bash
curl -L -O https://huggingface.co/cqyan/hybrid-sd-tinyvae/blob/main/diffusion_pytorch_model.safetensors
```
T2I Comparison using one A100 GPU, The image order from left to right : SD1.4-VAE -> TAESD -> Hybrid-sd-tinyvae
![alt text](image.png)
![alt text](image-1.png)
Specify the model path using the `--taesd PATH` parameter. example:
```bash
sd -m ../models/v1-5-pruned-emaonly.safetensors -p "a lovely cat" --taesd ../models/diffusion_pytorch_model.safetensors
```
<br />
<br />




### Hybrid-sd-tinyvae for SDXL
Download the model weight [Hybrid-sd-tinyvae-xl](https://huggingface.co/cqyan/hybrid-sd-tinyvae-xl) 
```bash
curl -L -O https://huggingface.co/cqyan/hybrid-sd-tinyvae-xl/blob/main/diffusion_pytorch_model.safetensors
```
T2I Comparison using one A100 GPU, The image order from left to right : SD1.4-VAE -> TAESD -> Hybrid-sd-tinyvae
![alt text](image-2.png)
![alt text](image-3.png)
Specify the model path using the `--taesd PATH` parameter. example:
```bash
sd -m ../models/sd_xl_base_1.0.safetensors -p "a lovely cat" --taesd ../models/diffusion_pytorch_model.safetensors
```
