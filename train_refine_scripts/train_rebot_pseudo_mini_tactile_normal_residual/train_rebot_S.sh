#!/usr/bin/env bash
# Train ReBotNet S variant on gelsight_pseudo_mini tactile_normal-domain
# transferred data in residual mode. Residual blank is the fixed
# flat-surface-normal (0,0,1) encoding (--normal_blank), not frame 0 of the
# transferred video: the true no-contact tactile_normal reading is a
# universal constant, unlike the shadow/appearance domain's per-video blank
# (see rebot_net/dataset.py's _FLAT_NORMAL_RGB).
# Usage: bash train_refine_scripts/train_rebot_pseudo_mini_tactile_normal_residual/train_rebot_S.sh <gpu_id> [matcher]
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
    --save_dir      "$PROJECT_ROOT/log/rebot_checkpoints_S_pseudo_mini_tactile_normal_residual${SUFFIX}" \
    --model_size    $MODEL_SIZE \
    --video_type    tactile_normal \
    --epochs        100 \
    --batch_size    8 \
    --lr            2e-4 \
    --num_workers   4 \
    --residual \
    --normal_blank \
    --wandb_project tactile_enhance \
    --wandb_run_name "${MODEL_SIZE}_pseudo_mini_tactile_normal_residual${SUFFIX}_bs8_lr2e-4"
