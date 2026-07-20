#!/usr/bin/env bash
# Evaluate ReBotNet L variant trained on gelsight_pseudo_mini data.
# Usage: bash train_refine_scripts/eval_rebot_pseudo_mini/eval_rebot_L.sh <gpu_id> [matcher] [masked]
#   from the PatchMatch_touch project root.
#   matcher: one of loftr (default), disk_lightglue, sift_lightglue,
#            superpoint_lightglue, superpoint_superglue -- must match the
#            matcher the checkpoint was trained with (train_rebot_pseudo_mini/).
#   masked: pass the literal 'masked' to evaluate the checkpoint trained on
#           the render-mask-blended log/transfer_feat_match_pseudo_mini*_masked
#           data (from postprocess_mask_transfer.py) instead of the unmasked one.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash eval_rebot_L.sh <gpu_id> [matcher] [masked]}
MATCHER="${2:-loftr}"
USE_MASK="${3:-}"
MODEL_SIZE=rebot_L

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

# loftr keeps the original (suffix-less) transfer/checkpoint/save naming for
# backward compatibility; every other matcher gets a _<matcher> suffix,
# mirroring train_rebot_pseudo_mini/'s convention.
if [ "$MATCHER" = "loftr" ]; then
    SUFFIX=""
else
    SUFFIX="_${MATCHER}"
fi

# masked evaluates the checkpoint trained on the postprocess_mask_transfer.py
# output dir, which appends _masked after the matcher suffix.
if [ "$USE_MASK" = "masked" ]; then
    SUFFIX="${SUFFIX}_masked"
fi

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/eval.py" \
    --transfer_dir "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini${SUFFIX}" \
    --checkpoint   "$PROJECT_ROOT/log/rebot_checkpoints_L_pseudo_mini${SUFFIX}/best.pth" \
    --model_size   $MODEL_SIZE \
    --save_dir     "$PROJECT_ROOT/log/rebot_eval_L_pseudo_mini${SUFFIX}" \
    --video_save \
    --save_gt
