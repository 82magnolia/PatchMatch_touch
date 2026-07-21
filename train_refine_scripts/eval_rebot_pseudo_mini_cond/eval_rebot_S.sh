#!/usr/bin/env bash
# Evaluate a CONDITIONED ReBotNet S trained on gelsight_pseudo_mini data
# (see train_refine_scripts/train_rebot_pseudo_mini_cond/train_rebot_S.sh).
#
# Usage: bash train_refine_scripts/eval_rebot_pseudo_mini_cond/eval_rebot_S.sh <gpu_id> [matcher] [cond_mode]
#   matcher:   loftr (default), disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue
#   cond_mode: both-normal (default), mask, film-{normal,curvature,height}, both-{normal,curvature,height}
#   -- must match what the checkpoint was trained with.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash eval_rebot_S.sh <gpu_id> [matcher] [cond_mode]}
MATCHER="${2:-loftr}"
COND_MODE="${3:-both-normal}"
MODEL_SIZE=rebot_S

case "$MATCHER" in
    loftr|disk_lightglue|sift_lightglue|superpoint_lightglue|superpoint_superglue) ;;
    *)
        echo "Unknown matcher '$MATCHER'. Expected one of: loftr, disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue" >&2
        exit 1
        ;;
esac
# Conditioning config (must match the sim pretrain run so the checkpoint loads
# without weight surgery). film-<mod> picks the geometry render.
case "$COND_MODE" in
    mask)           COND_FLAGS=(--mask_cond) ;;
    film-normal)    COND_FLAGS=(--film_modality normal) ;;
    film-curvature) COND_FLAGS=(--film_modality curvature) ;;
    film-height)    COND_FLAGS=(--film_modality height) ;;
    both-normal)    COND_FLAGS=(--mask_cond --film_modality normal) ;;
    both-curvature) COND_FLAGS=(--mask_cond --film_modality curvature) ;;
    both-height)    COND_FLAGS=(--mask_cond --film_modality height) ;;
    *)
        echo "Unknown cond_mode '$COND_MODE'. Expected: mask, film-{normal,curvature,height}, both-{normal,curvature,height}" >&2
        exit 1
        ;;
esac

if [ "$MATCHER" = "loftr" ]; then
    TRANSFER_SUFFIX=""
    CKPT_NAME=""
else
    TRANSFER_SUFFIX="_${MATCHER}"
    CKPT_NAME="_${MATCHER}"
fi

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/eval.py" \
    --transfer_dir "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini${TRANSFER_SUFFIX}" \
    --checkpoint   "$PROJECT_ROOT/log/rebot_checkpoints_S_pseudo_mini${CKPT_NAME}_cond-${COND_MODE}/best.pth" \
    --model_size   $MODEL_SIZE \
    --save_dir     "$PROJECT_ROOT/log/rebot_eval_S_pseudo_mini${CKPT_NAME}_cond-${COND_MODE}" \
    --cond_dir     "$PROJECT_ROOT/Taxim/results/gen_contact_full_query_pseudo_mini" \
    --film_scale   100 \
    "${COND_FLAGS[@]}" \
    --video_save \
    --save_gt
