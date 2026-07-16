#!/usr/bin/env bash
# Full-pipeline evaluation (retrieve -> feature-match transfer -> ReBotNet S
# refinement) on the gelsight_pseudo_mini test split, via transfer_pipeline.py end to end.
# Uses the pre-trained checkpoint produced by
# train_refine_scripts/train_rebot_pseudo_mini/train_rebot_S.sh.
#
# Usage: bash pipeline_scrips/eval_pipeline_pseudo_mini/eval_pipeline_S.sh <gpu_id> [matcher] [masked]
#   from the PatchMatch_touch project root.
#   matcher: one of loftr (default), disk_lightglue, sift_lightglue,
#            superpoint_lightglue, superpoint_superglue -- must match the
#            matcher the checkpoint was trained with (see
#            train_refine_scripts/transfer_all_multi_pseudo_mini/).
#   masked: pass the literal 'masked' to composite the transfer with the query's
#           render mask (transfer_pipeline.py --use_mask), matching a checkpoint
#           trained on postprocess_mask_transfer.py's masked output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash eval_pipeline_S.sh <gpu_id> [matcher] [masked]}
MATCHER="${2:-loftr}"
USE_MASK="${3:-}"
MODEL_SIZE=rebot_S

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

# loftr keeps the original (suffix-less) checkpoint naming for backward
# compatibility; every other matcher gets a _<matcher> suffix, mirroring
# train_rebot_pseudo_mini/train_rebot_S.sh's convention.
if [ "$MATCHER" = "loftr" ]; then
    SUFFIX=""
else
    SUFFIX="_${MATCHER}"
fi

# masked runs against a checkpoint trained on the render-mask-blended transfer
# output, mirrored here by passing --use_mask through transfer_pipeline.py's
# Stage 2 (see postprocess_mask_transfer.py's docstring: identical compositing).
MASK_FLAG=()
if [ "$USE_MASK" = "masked" ]; then
    SUFFIX="${SUFFIX}_masked"
    MASK_FLAG=(--use_mask)
fi

CHECKPOINT="$PROJECT_ROOT/log/rebot_checkpoints_S_pseudo_mini${SUFFIX}/best.pth"
REF_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_pseudo_mini"
QUERY_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_query_pseudo_mini"
OUT_BASE="$PROJECT_ROOT/log/pipeline_eval_S_pseudo_mini${SUFFIX}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    echo "Train it first with train_refine_scripts/train_rebot_pseudo_mini/train_rebot_S.sh $MATCHER $USE_MASK" >&2
    exit 1
fi

# Same held-out test split as rebot_net/train.py (sorted ids, last 50):
# objects present in both REF_BASE and QUERY_BASE, in the same numeric order
# transfer_all_multi_pseudo_mini/run.sh would have produced.
mapfile -t ALL_IDS < <(
    for d in "$REF_BASE"/*/; do
        idx=$(basename "$d")
        [ -d "$QUERY_BASE/$idx" ] && echo "$idx"
    done | sort -n
)
N=${#ALL_IDS[@]}
TEST_START=950
if [ "$N" -le "$TEST_START" ]; then
    echo "Only $N objects found under $REF_BASE (need > $TEST_START for the test split)." >&2
    exit 1
fi
TEST_IDS=("${ALL_IDS[@]:TEST_START}")
echo "Test objects: ${#TEST_IDS[@]}  |  matcher=$MATCHER  masked=${USE_MASK:-no}"

CUDA_VISIBLE_DEVICES=$GPU
export CUDA_VISIBLE_DEVICES

for idx in "${TEST_IDS[@]}"; do
    echo ""
    echo "=== Object $idx ==="
    python "$PROJECT_ROOT/transfer_pipeline.py" \
        --ref_dir            "$REF_BASE/$idx" \
        --query_dir           "$QUERY_BASE/$idx" \
        --scale               25 \
        --retrieval_mode      sim_gt_retrieval \
        --retrieval_modality  curvature \
        --transfer_modality   curvature \
        --video_type          shadow \
        --transfer_backend    dinov3_feat_match \
        --transfer_matcher    "$MATCHER" \
        "${MASK_FLAG[@]}" \
        --checkpoint          "$CHECKPOINT" \
        --model_size          $MODEL_SIZE \
        --save_dir            "$OUT_BASE/$idx"
done

echo ""
echo "Done. Per-object pipeline outputs under: $OUT_BASE/<idx>/{retrieval,transfer,enhanced,viz}"
