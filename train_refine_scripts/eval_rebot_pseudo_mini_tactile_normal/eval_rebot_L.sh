#!/usr/bin/env bash
# Evaluate ReBotNet L variant trained on gelsight_pseudo_mini
# tactile_normal-domain transferred data.
# Usage: bash train_refine_scripts/eval_rebot_pseudo_mini_tactile_normal/eval_rebot_L.sh <gpu_id>
#   from the PatchMatch_touch project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash eval_rebot_L.sh <gpu_id>}
MODEL_SIZE=rebot_L

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/eval.py" \
    --transfer_dir "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal" \
    --checkpoint   "$PROJECT_ROOT/log/rebot_checkpoints_L_pseudo_mini_tactile_normal/best.pth" \
    --model_size   $MODEL_SIZE \
    --video_type   tactile_normal \
    --save_dir     "$PROJECT_ROOT/log/rebot_eval_L_pseudo_mini_tactile_normal" \
    --video_save \
    --save_gt
