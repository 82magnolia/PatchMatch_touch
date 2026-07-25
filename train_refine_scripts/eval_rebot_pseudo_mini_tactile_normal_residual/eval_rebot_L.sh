#!/usr/bin/env bash
# Evaluate ReBotNet L variant trained in residual mode on
# gelsight_pseudo_mini tactile_normal-domain transferred data. --normal_blank
# matches the fixed flat-normal blank used at training time.
# Usage: bash train_refine_scripts/eval_rebot_pseudo_mini_tactile_normal_residual/eval_rebot_L.sh <gpu_id> [matcher]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash eval_rebot_L.sh <gpu_id> [matcher]}
MATCHER="${2:-disk_lightglue}"
MODEL_SIZE=rebot_L

case "$MATCHER" in
    loftr|disk_lightglue|sift_lightglue|superpoint_lightglue|superpoint_superglue) ;;
    *)
        echo "Unknown matcher '$MATCHER'. Expected one of: loftr, disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue" >&2
        exit 1
        ;;
esac

if [ "$MATCHER" = "disk_lightglue" ]; then
    SUFFIX=""
else
    SUFFIX="_${MATCHER}"
fi

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/eval.py" \
    --transfer_dir "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal${SUFFIX}" \
    --checkpoint   "$PROJECT_ROOT/log/rebot_checkpoints_L_pseudo_mini_tactile_normal_residual${SUFFIX}/best.pth" \
    --model_size   $MODEL_SIZE \
    --video_type   tactile_normal \
    --save_dir     "$PROJECT_ROOT/log/rebot_eval_L_pseudo_mini_tactile_normal_residual${SUFFIX}" \
    --video_save \
    --save_gt \
    --residual \
    --normal_blank
