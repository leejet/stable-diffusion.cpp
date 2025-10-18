# Running distilled SDXL models: SSD1B

### Preface

This kind of models has a reduced U-Net part. Unlike other SDXL models the U-Net has only one middle block and lesser attention layers in up and down blocks, resulting in relatively smaller files. Running these models saves more than 33% of the time. For more details, refer to Segmind's paper on https://arxiv.org/abs/2401.02677v1 .

### How to Use 

Unfortunately not all of this models follow the standard model parameter naming mapping. 
Anyway there are some useful SSD1B models available online, such as:

 * https://huggingface.co/segmind/SSD-1B/resolve/main/SSD-1B-A1111.safetensors
 * https://huggingface.co/hassenhamdi/SSD-1B-fp8_e4m3fn/resolve/main/SSD-1B_fp8_e4m3fn.safetensors 

Also there are useful LORAs available:

 * https://huggingface.co/seungminh/lora-swarovski-SSD-1B/resolve/main/pytorch_lora_weights.safetensors
 * https://huggingface.co/kylielee505/mylcmlorassd/resolve/main/pytorch_lora_weights.safetensors   
