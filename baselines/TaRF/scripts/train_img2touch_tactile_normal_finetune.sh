#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARF_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_CKPT="${TARF_DIR}/img2touch/pretrained_models/img2touch.ckpt"

if [[ ! -f "${SOURCE_CKPT}" ]]; then
  echo "Missing released checkpoint: ${SOURCE_CKPT}" >&2
  exit 2
fi

export TARF_TRAIN_CONFIG="configs/patchmatch_sim_tactile_normal_finetune.yaml"
export TARF_RUN_NAME="patchmatch_sim_tactile_normal_finetune_ref_even_query_odd"
exec bash "${SCRIPT_DIR}/train_img2touch_sim.sh" "$@"
