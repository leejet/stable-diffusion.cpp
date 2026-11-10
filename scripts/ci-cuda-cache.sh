#!/usr/bin/env bash

set -euo pipefail

mode="${1:-}"
if [[ "$mode" != restore && "$mode" != save ]]; then
    echo "Usage: $0 restore|save" >&2
    exit 1
fi

cache_dir="${RUNNER_TEMP:?}/cuda-ccache"
builder="${CUDA_CACHE_BUILDER:?}"
work_dir="$(mktemp -d "${RUNNER_TEMP}/cuda-cache.XXXXXX")"
trap 'rm -rf -- "$work_dir"' EXIT
mkdir -p "$cache_dir"

# Cache mounts are outside image layers; always execute the transfer RUN.
if [[ "$mode" == restore ]]; then
    cat > "$work_dir/Dockerfile" <<'DOCKERFILE'
FROM ubuntu:24.04
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=bind,source=.,target=/cache-input \
    cp -a /cache-input/. /root/.cache/ccache/
DOCKERFILE
    docker buildx build --builder "$builder" --no-cache \
        --file "$work_dir/Dockerfile" --output type=cacheonly "$cache_dir"
else
    cat > "$work_dir/Dockerfile" <<'DOCKERFILE'
FROM ubuntu:24.04 AS transfer
RUN --mount=type=cache,target=/root/.cache/ccache \
    mkdir /cache-output && cp -a /root/.cache/ccache/. /cache-output/
FROM scratch
COPY --from=transfer /cache-output/ /
DOCKERFILE
    docker buildx build --builder "$builder" --no-cache \
        --file "$work_dir/Dockerfile" \
        --output "type=local,dest=$work_dir/export" "$work_dir"
    # Replace only after export succeeds, including removal of evicted entries.
    mv "$cache_dir" "$work_dir/previous-cache"
    mv "$work_dir/export" "$cache_dir"
fi
