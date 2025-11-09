# Running distilled models: SSD1B and SDx.x with tiny U-Nets

## Preface 

These models feature a reduced U-Net architecture. Unlike standard SDXL models, the SSD-1B U-Net contains only one middle block and fewer attention layers in its up- and down-blocks, resulting in significantly smaller file sizes. Using these models can reduce inference time by more than 33%. For more details, refer to Segmind's paper: https://arxiv.org/abs/2401.02677v1.
Similarly, SD1.x- and SD2.x-style models with a tiny U-Net consist of only 6 U-Net blocks, leading to very small files and time savings of up to 50%. For more information, see the paper: https://arxiv.org/pdf/2305.15798.pdf.

## SSD1B

Note that not all of these models follow the standard parameter naming conventions. However, several useful SSD-1B models are available online, such as:

 * https://huggingface.co/segmind/SSD-1B/resolve/main/SSD-1B-A1111.safetensors
 * https://huggingface.co/hassenhamdi/SSD-1B-fp8_e4m3fn/resolve/main/SSD-1B_fp8_e4m3fn.safetensors

Useful LoRAs are also available:

 * https://huggingface.co/seungminh/lora-swarovski-SSD-1B/resolve/main/pytorch_lora_weights.safetensors
 * https://huggingface.co/kylielee505/mylcmlorassd/resolve/main/pytorch_lora_weights.safetensors

These files can be used out-of-the-box, unlike the models described in the next section.


## SD1.x, SD2.x with tiny U-Nets

These models require conversion before use. You will need a Python script provided by the diffusers team, available on GitHub:

 * https://raw.githubusercontent.com/huggingface/diffusers/refs/heads/main/scripts/convert_diffusers_to_original_stable_diffusion.py

### SD2.x

NotaAI provides the following model online:

* https://huggingface.co/nota-ai/bk-sdm-v2-tiny

Creating a .safetensors file involves two steps. First, run this short Python script to download the model from Hugging Face:

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("nota-ai/bk-sdm-v2-tiny",cache_dir="./")
```

Second, create the .safetensors file by running:

```bash
python convert_diffusers_to_original_stable_diffusion.py \
      --model_path  models--nota-ai--bk-sdm-v2-tiny/snapshots/68277af553777858cd47e133f92e4db47321bc74 \
      --checkpoint_path bk-sdm-v2-tiny.safetensors --half --use_safetensors
```

This will generate the **file bk-sdm-v2-tiny.safetensors**, which is now ready for use with sd.cpp.

### SD1.x

Several Tiny SD 1.x models are available online, such as:

 * https://huggingface.co/segmind/tiny-sd
 * https://huggingface.co/segmind/portrait-finetuned
 * https://huggingface.co/nota-ai/bk-sdm-tiny

These models also require conversion, partly because some tensors are stored in a non-contiguous manner. To create a usable checkpoint file, follow these simple steps:
Download and prepare the model using Python: 

##### Download the model using Python on your computer, for example this way:

```python
import torch
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("segmind/tiny-sd")
unet=pipe.unet
for param in unet.parameters():
    param.data = param.data.contiguous()     # <- important here
pipe.save_pretrained("segmindtiny-sd", safe_serialization=True)
```

##### Run the conversion script:

```bash
python convert_diffusers_to_original_stable_diffusion.py \
      --model_path  ./segmindtiny-sd \
      --checkpoint_path ./segmind_tiny-sd.ckpt --half
```

The file segmind_tiny-sd.ckpt will be generated and is now ready for use with sd.cpp. You can follow a similar process for the other models mentioned above.


### Another available .ckpt file:

 * https://huggingface.co/ClashSAN/small-sd/resolve/main/tinySDdistilled.ckpt

To use this file, you must first adjust its non-contiguous tensors:

```python
import torch
ckpt = torch.load("tinySDdistilled.ckpt", map_location=torch.device('cpu'))
for key, value in ckpt['state_dict'].items():
    if isinstance(value, torch.Tensor):
        ckpt['state_dict'][key] = value.contiguous()
torch.save(ckpt, "tinySDdistilled_fixed.ckpt")
```
