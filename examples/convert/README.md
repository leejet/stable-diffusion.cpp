# Model Convert

## Usage
```
usage: convert.exe [MODEL_PATH] -tp [OUT_TYPE] [arguments]
Model supported for conversion: .safetensors models or .ckpt checkpoints models

arguments:
  -h, --help                         show this help message and exit
  -o, --out [FILENAME]               path or name to converted model
  -v, --vocab [FILENAME]             path to custom vocab.json (usually unnecessary)
  -tp, --type [OUT_TYPE]             output format (f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0)
```
