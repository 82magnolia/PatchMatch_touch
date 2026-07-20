#!/usr/bin/env bash
# Build the render-mask-blended copy of the real-data transfer tree, for
# fine-tuning the *_cond scripts with data_mode=masked.
#
# Composites, per frame, exactly as main_retrieval_transfer_*.py --use_mask:
#     output = mask * transferred + (1 - mask) * base_frame
# with mask from the session's {idx}_render_mask.mp4 under
# log/real_data_gt_retrieval and base_frame = frame 0 of {idx}_ref_shadow.mp4.
#
# Output: log/transfer_pipeline_real_data_gt_retrieval[_<matcher>]_masked
# postprocess_mask_transfer.py refuses to overwrite an existing --out_dir.
#
# Usage: bash train_refine_scripts/prepare_masked_transfer/make_masked_real.sh [matcher]
#   matcher: loftr (default), disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MATCHER="${1:-loftr}"

case "$MATCHER" in
    loftr|disk_lightglue|sift_lightglue|superpoint_lightglue|superpoint_superglue) ;;
    *)
        echo "Unknown matcher '$MATCHER'. Expected one of: loftr, disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue" >&2
        exit 1
        ;;
esac

if [ "$MATCHER" = "loftr" ]; then
    TRANSFER_SUFFIX=""
else
    TRANSFER_SUFFIX="_${MATCHER}"
fi

SRC="$PROJECT_ROOT/log/transfer_pipeline_real_data_gt_retrieval${TRANSFER_SUFFIX}"
OUT="${SRC}_masked"

# Nested layout (<session>/transfer/{idx}_transferred*.mp4); the session name
# under --src_dir matches the one under --query_dir_root.
python "$PROJECT_ROOT/postprocess_mask_transfer.py" \
    --src_dir        "$SRC" \
    --query_dir_root "$PROJECT_ROOT/log/real_data_gt_retrieval" \
    --out_dir        "$OUT" \
    --video_type     shadow
