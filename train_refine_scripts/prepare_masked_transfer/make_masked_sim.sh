#!/usr/bin/env bash
# Build the render-mask-blended copy of a sim (pseudo_mini) transfer tree, for
# training the *_cond scripts with data_mode=masked.
#
# Composites, per frame, exactly as main_retrieval_transfer_*.py --use_mask:
#     output = mask * transferred + (1 - mask) * base_frame
# with mask from the query's {idx}_render_mask.mp4 and base_frame = frame 0 of
# {idx}_ref_shadow.mp4. This puts the query contact footprint directly into the
# input pixels, instead of leaving the network to infer it from the reference
# warp alone.
#
# Output: log/transfer_feat_match_pseudo_mini[_<matcher>]_masked
# postprocess_mask_transfer.py refuses to overwrite an existing --out_dir.
#
# Usage: bash train_refine_scripts/prepare_masked_transfer/make_masked_sim.sh [matcher]
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

SRC="$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini${TRANSFER_SUFFIX}"
OUT="${SRC}_masked"

# Per-object layout (<obj_id>/{idx}_transferred*.mp4), so --query_dir_root maps
# each object subfolder onto the matching one under the query root.
python "$PROJECT_ROOT/postprocess_mask_transfer.py" \
    --src_dir        "$SRC" \
    --query_dir_root "$PROJECT_ROOT/Taxim/results/gen_contact_full_query_pseudo_mini" \
    --out_dir        "$OUT" \
    --video_type     shadow
