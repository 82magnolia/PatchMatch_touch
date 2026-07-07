#!/usr/bin/env bash
# Train ReBotNet L variant on gelsight_pseudo_mini transferred data.
# Usage: bash train_refine_scripts/train_rebot_pseudo_mini/train_rebot_L.sh <gpu_id>
#   from the PatchMatch_touch project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash train_rebot_L.sh <gpu_id>}
MODEL_SIZE=rebot_L

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/train.py" \
    --transfer_dir  "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini" \
    --save_dir      "$PROJECT_ROOT/log/rebot_checkpoints_L_pseudo_mini" \
    --model_size    $MODEL_SIZE \
    --epochs        100 \
    --batch_size    4 \
    --lr            2e-4 \
    --num_workers   4 \
    --wandb_project tactile_enhance \
    --wandb_run_name "${MODEL_SIZE}_pseudo_mini_bs4_lr2e-4"
