#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --tdf-root DIR --config FILE --mesh FILE --texture FILE --texture-name NAME --cache-key KEY [--checkpoint FILE] [--python PYTHON]" >&2
}

PYTHON_BIN="python"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tdf-root) TDF_ROOT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --mesh) MESH="$2"; shift 2 ;;
        --texture) TEXTURE="$2"; shift 2 ;;
        --texture-name) TEXTURE_NAME="$2"; shift 2 ;;
        --cache-key) CACHE_KEY="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

for variable in TDF_ROOT CONFIG MESH TEXTURE TEXTURE_NAME CACHE_KEY; do
    if [[ -z "${!variable:-}" ]]; then
        echo "Missing required option for $variable" >&2
        usage
        exit 2
    fi
done

mkdir -p "$TDF_ROOT/data/tactile_textures"
cp "$TEXTURE" "$TDF_ROOT/data/tactile_textures/${TEXTURE_NAME}_tactile_texture_map_2_normal.png"

cd "$TDF_ROOT"
COMMAND=(
    "$PYTHON_BIN" main.py
    --config "$CONFIG"
    "save_path=$CACHE_KEY"
    "mesh=$MESH"
    "tactile_texture_object=$TEXTURE_NAME"
)
if [[ -n "${CHECKPOINT:-}" ]]; then
    COMMAND+=("load=$CHECKPOINT")
fi
exec "${COMMAND[@]}"
