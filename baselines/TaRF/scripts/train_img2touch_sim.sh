#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARF_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${TARF_DIR}/../.." && pwd)"
IMG2TOUCH_DIR="${TARF_DIR}/img2touch"
RUN_ROOT="${PROJECT_DIR}/log/baselines/tarf_training/runs"
NUM_GPUS="${TARF_NUM_GPUS:-4}"
TRAIN_CONFIG="${TARF_TRAIN_CONFIG:-configs/patchmatch_sim_train.yaml}"
RUN_NAME="${TARF_RUN_NAME:-patchmatch_sim_ref_even_query_odd}"

echo "Current GPU occupancy:"
nvidia-smi

if [[ -z "${TARF_GPUS:-}" ]]; then
  TARF_GPUS="$(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits |
      sort -t, -k2,2n -k3,3n |
      head -n "${NUM_GPUS}" |
      cut -d, -f1 |
      tr -d ' ' |
      paste -sd, -
  )"
fi

IFS=',' read -r -a PHYSICAL_GPUS <<< "${TARF_GPUS}"
if (( ${#PHYSICAL_GPUS[@]} < 2 )); then
  echo "TaRF training requires at least two selected GPUs; got ${TARF_GPUS}." >&2
  exit 2
fi

LOGICAL_GPUS=""
for index in "${!PHYSICAL_GPUS[@]}"; do
  LOGICAL_GPUS+="${index},"
done

mkdir -p "${RUN_ROOT}"
echo "Using physical GPUs ${TARF_GPUS} (DDP ranks ${LOGICAL_GPUS})."

RUN_MODE=(--name "${RUN_NAME}")
if [[ -n "${TARF_RESUME:-}" ]]; then
  RUN_MODE=(--resume "${TARF_RESUME}")
  echo "Resuming from ${TARF_RESUME}."
fi

cd "${IMG2TOUCH_DIR}"
export CUDA_VISIBLE_DEVICES="${TARF_GPUS}"
exec conda run --no-capture-output -n TaRF python main.py \
  --base configs/tarf.yaml "${TRAIN_CONFIG}" \
  "${RUN_MODE[@]}" \
  --logdir "${RUN_ROOT}" \
  -t --gpus "${LOGICAL_GPUS}" --no-test true \
  "$@"
