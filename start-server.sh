#!/bin/sh
set -eu

SD_SERVER_PATH="${SD_SERVER_PATH:-}"
MODEL_DIR="./models"
COMMANDLINE_ARGS=""

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -f "$script_dir/start-server.local.sh" ]; then
    . "$script_dir/start-server.local.sh"
fi

if [ "${1:-}" != "--version" ]; then
    # Split configured arguments on whitespace without expanding wildcards.
    set -f
    set -- --model-dir "$MODEL_DIR" $COMMANDLINE_ARGS "$@"
fi

run_server() {
    printf 'Running:'
    for arg do
        printf " '%s'" "$(printf '%s' "$arg" | sed "s/'/'\\\\''/g")"
    done
    printf '\n'
    exec "$@"
}

if [ -n "$SD_SERVER_PATH" ]; then
    case "$SD_SERVER_PATH" in
        */*) ;;
        *) SD_SERVER_PATH="./$SD_SERVER_PATH" ;;
    esac
    if [ ! -f "$SD_SERVER_PATH" ] || [ ! -x "$SD_SERVER_PATH" ]; then
        printf 'SD_SERVER_PATH is not an executable file: %s\n' "$SD_SERVER_PATH" >&2
        exit 1
    fi
    run_server "$SD_SERVER_PATH" "$@"
fi

for server in \
    "$script_dir/sd-server" \
    "$script_dir/bin/Release/sd-server" \
    "$script_dir/bin/sd-server" \
    "$script_dir/build/bin/Release/sd-server" \
    "$script_dir/build/bin/sd-server"
do
    if [ -f "$server" ] && [ -x "$server" ]; then
        run_server "$server" "$@"
    fi
done

printf '%s\n' 'sd-server was not found. Build it first or set SD_SERVER_PATH to its executable path.' >&2
exit 1
