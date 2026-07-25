#!/usr/bin/env bash
# Train ReBotNet S variant on gelsight_pseudo_mini tactile_normal-domain
# transferred data: PatchMatch-transferred surface-normal-encoded videos
# (--video_type tactile_normal, see
# train_refine_scripts/transfer_all_multi_pseudo_mini_tactile_normal/) instead
# of the shadow/appearance domain used by train_rebot_pseudo_mini/. GT is the
# query's own tactile_normal video ({pair}_query_tactile_normal.mp4).
# Usage: bash train_refine_scripts/train_rebot_pseudo_mini_tactile_normal/train_rebot_S.sh <gpu_id> [matcher]
#   matcher: disk_lightglue (default), loftr, sift_lightglue, superpoint_lightglue, superpoint_superglue
#            -- must match a completed transfer_all_multi_pseudo_mini_tactile_normal run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash train_rebot_S.sh <gpu_id> [matcher]}
MATCHER="${2:-disk_lightglue}"
MODEL_SIZE=rebot_S

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

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/train.py" \
    --transfer_dir  "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal${SUFFIX}" \
    --save_dir      "$PROJECT_ROOT/log/rebot_checkpoints_S_pseudo_mini_tactile_normal${SUFFIX}" \
    --model_size    $MODEL_SIZE \
    --video_type    tactile_normal \
    --epochs        100 \
    --batch_size    8 \
    --lr            2e-4 \
    --num_workers   4 \
    --wandb_project tactile_enhance \
    --wandb_run_name "${MODEL_SIZE}_pseudo_mini_tactile_normal${SUFFIX}_bs8_lr2e-4"
