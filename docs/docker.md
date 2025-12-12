## Docker

### Building using Docker

```shell
docker build -t sd .
```

### Run

```shell
docker run -v /path/to/models:/models -v /path/to/output/:/output sd-cli [args...]
# For example
# docker run -v ./models:/models -v ./build:/output sd-cli -m /models/sd-v1-4.ckpt -p "a lovely cat" -v -o /output/output.png
```