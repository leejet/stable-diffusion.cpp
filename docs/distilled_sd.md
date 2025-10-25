# Running distilled models: SSD1B and SD1.x with tiny U-Nets

## Preface

This kind of models have a reduced U-Net part. 
Unlike other SDXL models the U-Net of SSD1B has only one middle block and lesser attention layers in up and down blocks, resulting in relatively smaller files. Running these models saves more than 33% of the time. For more details, refer to Segmind's paper on https://arxiv.org/abs/2401.02677v1 .
Unlike other SD 1.x models Tiny-UNet models consist of only 6 U-Net blocks, resulting in relatively smaller files (approximately 1 GB). Running these models saves almost 50% of the time. For more details, refer to the paper: https://arxiv.org/pdf/2305.15798.pdf .

## SSD1B

Unfortunately not all of this models follow the standard model parameter naming mapping. 
Anyway there are some very useful SSD1B models available online, such as:

 * https://huggingface.co/segmind/SSD-1B/resolve/main/SSD-1B-A1111.safetensors
 * https://huggingface.co/hassenhamdi/SSD-1B-fp8_e4m3fn/resolve/main/SSD-1B_fp8_e4m3fn.safetensors 

Also there are useful LORAs available:

 * https://huggingface.co/seungminh/lora-swarovski-SSD-1B/resolve/main/pytorch_lora_weights.safetensors
 * https://huggingface.co/kylielee505/mylcmlorassd/resolve/main/pytorch_lora_weights.safetensors   

You can use this files **out-of-the-box** - unlike models in next section.


## SD1.x with tiny U-Nets

There are some Tiny SD 1.x models available online, such as:

 * https://huggingface.co/segmind/tiny-sd
 * https://huggingface.co/segmind/portrait-finetuned
 * https://huggingface.co/nota-ai/bk-sdm-tiny

These models need some conversion, for example because partially tensors are **non contiguous** stored. To create a usable checkpoint file, follow these **easy** steps:

### Download model from Hugging Face

Download the model using Python on your computer, for example this way:

```python
import torch
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("segmind/tiny-sd")
unet=pipe.unet
for param in unet.parameters():
    param.data = param.data.contiguous()     # <- important here
pipe.save_pretrained("segmindtiny-sd", safe_serialization=True)
```

### Convert that to a ckpt file 

To convert the downloaded model to a checkpoint file, you need another Python script. Download the conversion script from here:

 * https://raw.githubusercontent.com/huggingface/diffusers/refs/heads/main/scripts/convert_diffusers_to_original_stable_diffusion.py


### Run convert script

Now, run that conversion script:

```bash
python convert_diffusers_to_original_stable_diffusion.py \
	--model_path  ./segmindtiny-sd \
	--checkpoint_path ./segmind_tiny-sd.ckpt --half
```

The file **segmind_tiny-sd.ckpt**  will be generated and is now ready to use with sd.cpp

You can follow a similar process for other models mentioned above from Hugging Face. 


### Another ckpt file on the net

There is another model file available online: 

 * https://huggingface.co/ClashSAN/small-sd/resolve/main/tinySDdistilled.ckpt
 
If you want to use that, you have to adjust some **non-contiguous tensors** first:

```python
import torch
ckpt = torch.load("tinySDdistilled.ckpt", map_location=torch.device('cpu'))
for key, value in ckpt['state_dict'].items():
    if isinstance(value, torch.Tensor):
        ckpt['state_dict'][key] = value.contiguous()
torch.save(ckpt, "tinySDdistilled_fixed.ckpt")
```
