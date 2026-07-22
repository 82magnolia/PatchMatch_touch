#!/usr/bin/env bash
# Evaluate a CONDITIONED ReBotNet M trained on gelsight_pseudo_mini
# tactile_normal-domain transferred data (see
# train_refine_scripts/train_rebot_pseudo_mini_tactile_normal_cond/train_rebot_M.sh).
#
# Usage: bash train_refine_scripts/eval_rebot_pseudo_mini_tactile_normal_cond/eval_rebot_M.sh <gpu_id> [matcher] [cond_mode]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash eval_rebot_M.sh <gpu_id> [matcher] [cond_mode]}
MATCHER="${2:-disk_lightglue}"
COND_MODE="${3:-both-normal}"
MODEL_SIZE=rebot_M

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

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/eval.py" \
    --transfer_dir "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal${SUFFIX}" \
    --checkpoint   "$PROJECT_ROOT/log/rebot_checkpoints_M_pseudo_mini_tactile_normal${SUFFIX}_cond-${COND_MODE}/best.pth" \
    --model_size   $MODEL_SIZE \
    --video_type   tactile_normal \
    --save_dir     "$PROJECT_ROOT/log/rebot_eval_M_pseudo_mini_tactile_normal${SUFFIX}_cond-${COND_MODE}" \
    --cond_dir     "$PROJECT_ROOT/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini" \
    --film_scale   100 \
    "${COND_FLAGS[@]}" \
    --video_save \
    --save_gt
