#!/usr/bin/env bash
# Train ReBotNet M variant on gelsight_pseudo_mini transferred data in residual mode.
# Usage: bash train_refine_scripts/train_rebot_pseudo_mini_residual/train_rebot_M.sh <gpu_id> [matcher] [masked]
#   from the PatchMatch_touch project root.
#   matcher: one of loftr (default), disk_lightglue, sift_lightglue,
#            superpoint_lightglue, superpoint_superglue -- matches the
#            --matcher backends swept in transfer_all_multi_pseudo_mini/,
#            each of which writes to its own log/transfer_feat_match_pseudo_mini* dir.
#   masked: pass the literal 'masked' to train on the render-mask-blended
#           log/transfer_feat_match_pseudo_mini*_masked dir (from
#           postprocess_mask_transfer.py) instead of the unmasked one.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash train_rebot_M.sh <gpu_id> [matcher] [masked]}
MATCHER="${2:-loftr}"
USE_MASK="${3:-}"
MODEL_SIZE=rebot_M

case "$MATCHER" in
    loftr|disk_lightglue|sift_lightglue|superpoint_lightglue|superpoint_superglue) ;;
    *)
        echo "Unknown matcher '$MATCHER'. Expected one of: loftr, disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue" >&2
        exit 1
        ;;
esac

case "$USE_MASK" in
    ""|masked) ;;
    *)
        echo "Unknown mask option '$USE_MASK'. Expected empty or 'masked'." >&2
        exit 1
        ;;
esac

# loftr keeps the original (suffix-less) transfer dir naming for backward
# compatibility; every other matcher gets a _<matcher> suffix, mirroring
# transfer_all_multi_pseudo_mini/'s OUT_BASE convention. save_dir/wandb
# always keep _residual, with the matcher suffix appended after it.
if [ "$MATCHER" = "loftr" ]; then
    TRANSFER_SUFFIX=""
    NAME_SUFFIX="_residual"
else
    TRANSFER_SUFFIX="_${MATCHER}"
    NAME_SUFFIX="_residual_${MATCHER}"
fi

# masked runs against the postprocess_mask_transfer.py output dir, which
# appends _masked after the matcher suffix.
if [ "$USE_MASK" = "masked" ]; then
    TRANSFER_SUFFIX="${TRANSFER_SUFFIX}_masked"
    NAME_SUFFIX="${NAME_SUFFIX}_masked"
fi

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/train.py" \
    --transfer_dir  "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini${TRANSFER_SUFFIX}" \
    --save_dir      "$PROJECT_ROOT/log/rebot_checkpoints_M_pseudo_mini${NAME_SUFFIX}" \
    --model_size    $MODEL_SIZE \
    --epochs        100 \
    --batch_size    8 \
    --lr            2e-4 \
    --num_workers   4 \
    --residual \
    --wandb_project tactile_enhance \
    --wandb_run_name "${MODEL_SIZE}_pseudo_mini${NAME_SUFFIX}_bs8_lr2e-4"
