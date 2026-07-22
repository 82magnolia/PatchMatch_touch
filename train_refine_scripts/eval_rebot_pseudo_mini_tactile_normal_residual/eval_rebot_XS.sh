#!/usr/bin/env bash
# Evaluate ReBotNet XS variant trained in residual mode on
# gelsight_pseudo_mini tactile_normal-domain transferred data. --normal_blank
# matches the fixed flat-normal blank used at training time (see
# train_refine_scripts/train_rebot_pseudo_mini_tactile_normal_residual/).
# Usage: bash train_refine_scripts/eval_rebot_pseudo_mini_tactile_normal_residual/eval_rebot_XS.sh <gpu_id>
#   from the PatchMatch_touch project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash eval_rebot_XS.sh <gpu_id>}
MODEL_SIZE=rebot_XS

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/eval.py" \
    --transfer_dir "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal" \
    --checkpoint   "$PROJECT_ROOT/log/rebot_checkpoints_XS_pseudo_mini_tactile_normal_residual/best.pth" \
    --model_size   $MODEL_SIZE \
    --video_type   tactile_normal \
    --save_dir     "$PROJECT_ROOT/log/rebot_eval_XS_pseudo_mini_tactile_normal_residual" \
    --video_save \
    --save_gt \
    --residual \
    --normal_blank
