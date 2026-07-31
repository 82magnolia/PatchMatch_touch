#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARF_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMG2TOUCH_DIR="${TARF_DIR}/img2touch"
SOURCE_CKPT="${IMG2TOUCH_DIR}/pretrained_models/img2touch.ckpt"
FIRST_STAGE_CKPT="${IMG2TOUCH_DIR}/pretrained_models/img2touch_first_stage.ckpt"

if [[ ! -f "${FIRST_STAGE_CKPT}" ]]; then
  if [[ ! -f "${SOURCE_CKPT}" ]]; then
    echo "Missing released checkpoint: ${SOURCE_CKPT}" >&2
    exit 2
  fi
  conda run --no-capture-output -n TaRF python \
    "${SCRIPT_DIR}/extract_img2touch_first_stage.py" \
    --source "${SOURCE_CKPT}" \
    --output "${FIRST_STAGE_CKPT}"
fi

export TARF_TRAIN_CONFIG="configs/patchmatch_sim_tactile_normal_train.yaml"
export TARF_RUN_NAME="patchmatch_sim_tactile_normal"
exec bash "${SCRIPT_DIR}/train_img2touch_sim.sh" "$@"
