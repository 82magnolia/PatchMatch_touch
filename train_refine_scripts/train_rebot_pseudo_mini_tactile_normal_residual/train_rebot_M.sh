#!/usr/bin/env bash
# Train ReBotNet M variant on gelsight_pseudo_mini tactile_normal-domain
# transferred data in residual mode. Residual blank is the fixed
# flat-surface-normal (0,0,1) encoding (--normal_blank), not frame 0 of the
# transferred video: the true no-contact tactile_normal reading is a
# universal constant, unlike the shadow/appearance domain's per-video blank
# (see rebot_net/dataset.py's _FLAT_NORMAL_RGB).
# Usage: bash train_refine_scripts/train_rebot_pseudo_mini_tactile_normal_residual/train_rebot_M.sh <gpu_id>
#   from the PatchMatch_touch project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash train_rebot_M.sh <gpu_id>}
MODEL_SIZE=rebot_M

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/train.py" \
    --transfer_dir  "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal" \
    --save_dir      "$PROJECT_ROOT/log/rebot_checkpoints_M_pseudo_mini_tactile_normal_residual" \
    --model_size    $MODEL_SIZE \
    --video_type    tactile_normal \
    --epochs        100 \
    --batch_size    8 \
    --lr            2e-4 \
    --num_workers   4 \
    --residual \
    --normal_blank \
    --wandb_project tactile_enhance \
    --wandb_run_name "${MODEL_SIZE}_pseudo_mini_tactile_normal_residual_bs8_lr2e-4"
