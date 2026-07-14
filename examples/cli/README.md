# Usage

For detailed command-line arguments, run:

```bash
./bin/sd-cli -h
```

For direct image repair or automatic post-generation YOLOv8 detection followed by cropped inpainting, see
[ADetailer](../../docs/adetailer.md).

Metadata mode inspects PNG/JPEG container metadata without loading any model:

```bash
./bin/sd-cli -M metadata --image ./output.png
./bin/sd-cli -M metadata --image ./output.jpg --metadata-format json
./bin/sd-cli -M metadata --image ./output.png --metadata-raw
./bin/sd-cli -M metadata --image ./output.png --metadata-all
```
