#!/usr/bin/env bash
# Full-pipeline evaluation (retrieve -> feature-match transfer -> ReBotNet M
# refinement) over every session in log/real_data_gt_retrieval, via
# transfer_pipeline.py end to end. Uses the pre-trained checkpoint produced by
# train_refine_scripts/train_rebot_pseudo_mini/train_rebot_M.sh
# (trained on synthetic Taxim gelsight_pseudo_mini data; this evaluates how well
# it generalizes to real GelSight captures).
#
# Follows train_refine_scripts/transfer_all_real_data_gt_retrieval/run.sh's
# retrieval/transfer conventions: --retrieval_mode real_gt_retrieval (odd-indexed
# touches = query, matched to the even-indexed touch directly below, e.g. 1->0,
# 3->2 -- --ref_dir and --query_dir are the same session directory), scale=8,
# modality=curvature, video_type=shadow.
#
# Usage: bash pipeline_scrips/eval_pipeline_real_data_gt_retrieval/eval_pipeline_M.sh <gpu_id> [matcher] [masked]
#   from the PatchMatch_touch project root.
#   matcher: one of loftr (default), disk_lightglue, sift_lightglue,
#            superpoint_lightglue, superpoint_superglue -- must match the
#            matcher the checkpoint was trained with (see
#            train_refine_scripts/transfer_all_multi_pseudo_mini/).
#   masked: pass the literal 'masked' to composite the transfer with the query's
#           render mask (transfer_pipeline.py --use_mask), matching a checkpoint
#           trained on postprocess_mask_transfer.py's masked output.

set -euo pipefail

# torch/OpenCV/numpy(MKL,OpenBLAS) all default to using every CPU core. That's fine
# for a single run, but running several copies of this script side by side makes them
# oversubscribe the same cores and slow each other down. Cap per-process thread count;
# override e.g. `NUM_THREADS=7 bash eval_pipeline_M.sh <gpu_id>` when running
# 4-way in parallel on 28 cores.
export NUM_THREADS="${NUM_THREADS:-4}"
export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"
export NUMEXPR_NUM_THREADS="$NUM_THREADS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash eval_pipeline_M.sh <gpu_id> [matcher] [masked]}
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

# loftr keeps the original (suffix-less) checkpoint naming for backward
# compatibility; every other matcher gets a _<matcher> suffix, mirroring
# train_rebot_pseudo_mini/train_rebot_M.sh's convention.
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

CHECKPOINT="$PROJECT_ROOT/log/rebot_checkpoints_M_pseudo_mini${SUFFIX}/best.pth"
SESSIONS_BASE="$PROJECT_ROOT/log/real_data_gt_retrieval"
OUT_BASE="$PROJECT_ROOT/log/pipeline_eval_M_real_data_gt_retrieval${SUFFIX}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    echo "Train it first with train_refine_scripts/train_rebot_pseudo_mini/train_rebot_M.sh $MATCHER $USE_MASK" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU
export CUDA_VISIBLE_DEVICES

# Count eligible sessions
total=0
for session_dir in "$SESSIONS_BASE"/*/; do
    total=$((total + 1))
done
echo "Sessions to process: $total  |  matcher=$MATCHER  masked=${USE_MASK:-no}"

done_count=0
for session_dir in "$SESSIONS_BASE"/*/; do
    idx=$(basename "$session_dir")
    save_dir="$OUT_BASE/$idx"

    done_count=$((done_count + 1))
    echo ""
    echo "=== Session $idx ($done_count/$total) ==="

    python "$PROJECT_ROOT/transfer_pipeline.py" \
        --ref_dir            "$session_dir" \
        --query_dir          "$session_dir" \
        --scale               8 \
        --retrieval_mode      real_gt_retrieval \
        --transfer_backend    dinov3_feat_match \
        --transfer_modality   curvature \
        --transfer_matcher    "$MATCHER" \
        --video_type          shadow \
        "${MASK_FLAG[@]}" \
        --checkpoint          "$CHECKPOINT" \
        --model_size          $MODEL_SIZE \
        --save_dir            "$save_dir"

    echo "  → done  ($done_count/$total sessions)"
done

echo ""
echo "Done. Per-session pipeline outputs under: $OUT_BASE/<session>/{retrieval,transfer,enhanced,viz}"
