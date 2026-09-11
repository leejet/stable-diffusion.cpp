# Usage

For detailed command-line arguments, run:

```bash
./bin/sd-cli -h
```

Logging defaults to `info`. Use `--log-level <level>` to select `debug`, `verbose`,
`info`, `warn`, or `error` (from most to least detailed). Each level includes
messages at that level and all less detailed levels. `-v` and `--verbose` are
equivalent to `--log-level verbose`. If repeated, the last logging option wins.

For direct image repair or automatic post-generation YOLOv8 detection followed by cropped inpainting, see
[ADetailer](../../docs/adetailer.md).

Metadata mode inspects PNG/JPEG container metadata without loading any model:

```bash
./bin/sd-cli -M metadata --image ./output.png
./bin/sd-cli -M metadata --image ./output.jpg --metadata-format json
./bin/sd-cli -M metadata --image ./output.png --metadata-raw
./bin/sd-cli -M metadata --image ./output.png --metadata-all
```
