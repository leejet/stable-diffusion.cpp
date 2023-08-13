# Model Convert Script

## Requirements

- vocab.json, from https://huggingface.co/openai/clip-vit-large-patch14/raw/main/vocab.json


```shell
pip install -r requirements.txt
```

## Usage
```
usage: convert.py [-h] [--out_type {f32,f16,q4_0,q4_1,q5_0,q5_1,q8_0}] [--out_file OUT_FILE] model_path

Convert Stable Diffuison model to GGML compatible file format

positional arguments:
  model_path            model file path (*.pth, *.pt, *.ckpt, *.safetensors)

options:
  -h, --help            show this help message and exit
  --out_type {f32,f16,q4_0,q4_1,q5_0,q5_1,q8_0}
                        output format (default: based on input)
  --out_file OUT_FILE   path to write to; default: based on input and current working directory
```
