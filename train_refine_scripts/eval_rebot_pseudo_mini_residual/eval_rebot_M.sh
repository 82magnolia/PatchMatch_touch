#!/usr/bin/env bash
# Evaluate ReBotNet M variant trained in residual mode on gelsight_pseudo_mini data.
# Usage: bash train_refine_scripts/eval_rebot_pseudo_mini_residual/eval_rebot_M.sh <gpu_id>
#   from the PatchMatch_touch project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash eval_rebot_M.sh <gpu_id>}
MODEL_SIZE=rebot_M

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/eval.py" \
    --transfer_dir "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini" \
    --checkpoint   "$PROJECT_ROOT/log/rebot_checkpoints_M_pseudo_mini_residual/best.pth" \
    --model_size   $MODEL_SIZE \
    --save_dir     "$PROJECT_ROOT/log/rebot_eval_M_pseudo_mini_residual" \
    --video_save \
    --save_gt \
    --residual
