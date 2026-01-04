# Docker

## Building using Docker

```shell
docker build -t sd .
```

## Run CLI

```shell
docker run --rm -v /path/to/models:/models -v /path/to/output/:/output sd [args...]
# For example
# docker run --rm -v ./models:/models -v ./build:/output sd -m /models/sd-v1-4.ckpt -p "a lovely cat" -v -o /output/output.png
```

## Run server

```shell
docker run --rm --init -v /path/to/models:/models -v /path/to/output/:/output -p "1234:1234" --entrypoint "/sd-server" sd [args...]
# For example
# docker run --rm --init -v ./models:/models -v ./build:/output -p "1234:1234" --entrypoint "/sd-server" sd -m /models/sd-v1-4.ckpt -p "a lovely cat" -v -o /output/output.png
```
