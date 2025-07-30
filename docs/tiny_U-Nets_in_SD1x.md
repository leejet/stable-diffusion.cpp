# Running SD1.x models with tiny U-Nets

### Preface

Tiny SD 1.x models have a very small U-Net part.  Unlike other 1.x models they consist of only 6 U-Net blocks, resulting in relatively small checkpoint files (approximately 1 GB). Running these models saves almost 50% of the time. For more details, refer to the paper: https://arxiv.org/pdf/2305.15798.pdf .

There are only a few Tiny SD 1.x models available online, such as:

 * https://huggingface.co/segmind/tiny-sd
 * https://huggingface.co/segmind/portrait-finetuned
 * https://huggingface.co/nota-ai/bk-sdm-tiny

To create a checkpoint file, follow these steps:

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

