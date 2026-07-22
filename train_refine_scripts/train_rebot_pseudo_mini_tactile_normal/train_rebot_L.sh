#!/usr/bin/env bash
# Train ReBotNet L variant on gelsight_pseudo_mini tactile_normal-domain
# transferred data: PatchMatch-transferred surface-normal-encoded videos
# (--video_type tactile_normal, see
# train_refine_scripts/transfer_all_multi_pseudo_mini_tactile_normal/) instead
# of the shadow/appearance domain used by train_rebot_pseudo_mini/. GT is the
# query's own tactile_normal video ({pair}_query_tactile_normal.mp4).
# Usage: bash train_refine_scripts/train_rebot_pseudo_mini_tactile_normal/train_rebot_L.sh <gpu_id>
#   from the PatchMatch_touch project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash train_rebot_L.sh <gpu_id>}
MODEL_SIZE=rebot_L

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/train.py" \
    --transfer_dir  "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal" \
    --save_dir      "$PROJECT_ROOT/log/rebot_checkpoints_L_pseudo_mini_tactile_normal" \
    --model_size    $MODEL_SIZE \
    --video_type    tactile_normal \
    --epochs        100 \
    --batch_size    8 \
    --lr            2e-4 \
    --num_workers   4 \
    --wandb_project tactile_enhance \
    --wandb_run_name "${MODEL_SIZE}_pseudo_mini_tactile_normal_bs8_lr2e-4"
