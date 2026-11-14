## Using PhotoMaker to personalize image generation

You can use [PhotoMaker](https://github.com/TencentARC/PhotoMaker) to personalize generated images with your own ID.

**NOTE**, currently PhotoMaker **ONLY** works with **SDXL** (any SDXL model files will work).

Download PhotoMaker model file (in safetensor format) [here](https://huggingface.co/bssrdf/PhotoMaker). The official release of the model file (in .bin format) does not work with ```stablediffusion.cpp```.

- Specify the PhotoMaker model path using the `--photo-maker PATH` parameter.
- Specify the input images path using the `--pm-id-images-dir PATH` parameter.

In prompt, make sure you have a class word followed by the trigger word ```"img"``` (hard-coded for now). The class word could be one of ```"man, woman, girl, boy"```. If input ID images contain asian faces, add ```Asian``` before the class
word.

Another PhotoMaker specific parameter:

- ```--pm-style-strength  (0-100)%```: default is 20 and 10-20 typically gets good results. Lower ratio means more faithfully following input ID (not necessarily better quality).

Other parameters recommended for running Photomaker:

- ```--cfg-scale 5.0```
- ```-H 1024```
- ```-W 1024```

If on low memory GPUs (<= 8GB), recommend running with ```--vae-on-cpu``` option to get artifact free images.

Example:

```bash
bin/sd-cli -m ../models/sdxlUnstableDiffusers_v11.safetensors  --vae ../models/sdxl_vae.safetensors --photo-maker ../models/photomaker-v1.safetensors --pm-id-images-dir ../assets/photomaker_examples/scarletthead_woman -p "a girl img, retro futurism, retro game art style but extremely beautiful, intricate details, masterpiece, best quality, space-themed, cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed" -n "realistic, photo-realistic, worst quality, greyscale, bad anatomy, bad hands, error, text" --cfg-scale 5.0  --sampling-method euler -H 1024 -W 1024 --pm-style-strength 10 --vae-on-cpu --steps 50
```

## PhotoMaker Version 2

[PhotoMaker Version 2 (PMV2)](https://github.com/TencentARC/PhotoMaker/blob/main/README_pmv2.md) has some key improvements. Unfortunately it has a very heavy dependency which makes running it a bit involved in ```SD.cpp```. 

Running PMV2 is now a two-step process:

- Run a python script ```face_detect.py``` to obtain **id_embeds** for the given input images
```
python face_detect.py input_image_dir
```
An ```id_embeds.bin``` file will be generated in ```input_images_dir```

**Note: this step is only needed to run once; the same ```id_embeds``` can be reused**

- Run the same command as in version 1 but replacing ```photomaker-v1.safetensors``` with ```photomaker-v2.safetensors```.

  You can download ```photomaker-v2.safetensors``` from [here](https://huggingface.co/bssrdf/PhotoMakerV2)

- All the command line parameters from Version 1 remain the same for Version 2 plus one extra pointing to a valid ```id_embeds``` file:  --pm-id-embed-path [path_to__id_embeds.bin] 


