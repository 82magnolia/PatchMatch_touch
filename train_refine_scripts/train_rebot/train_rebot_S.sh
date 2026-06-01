#!/usr/bin/env bash
# Train ReBotNet S variant.
# Usage: bash train_refine_scripts/train_rebot/train_rebot_S.sh
#   from the PatchMatch_touch project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash train_rebot_S.sh <gpu_id>}
MODEL_SIZE=rebot_S

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/train.py" \
    --transfer_dir  "$PROJECT_ROOT/log/transfer" \
    --save_dir      "$PROJECT_ROOT/log/rebot_checkpoints_S" \
    --model_size    $MODEL_SIZE \
    --epochs        100 \
    --batch_size    8 \
    --lr            2e-4 \
    --num_workers   4 \
    --wandb_project tactile_enhance \
    --wandb_run_name "${MODEL_SIZE}_bs8_lr2e-4"
