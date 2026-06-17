#!/usr/bin/env bash
# Train ReBotNet XS variant on 240×320 transferred data in residual mode.
# Usage: bash train_refine_scripts/train_rebot_240x320_residual/train_rebot_XS.sh <gpu_id>
#   from the PatchMatch_touch project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash train_rebot_XS.sh <gpu_id>}
MODEL_SIZE=rebot_XS

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/train.py" \
    --transfer_dir  "$PROJECT_ROOT/log/transfer_240x320" \
    --save_dir      "$PROJECT_ROOT/log/rebot_checkpoints_XS_240x320_residual" \
    --model_size    $MODEL_SIZE \
    --epochs        100 \
    --batch_size    8 \
    --lr            2e-4 \
    --num_workers   4 \
    --residual \
    --wandb_project tactile_enhance \
    --wandb_run_name "${MODEL_SIZE}_240x320_residual_bs8_lr2e-4"
