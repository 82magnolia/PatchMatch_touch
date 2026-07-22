#!/usr/bin/env bash
# Evaluate a CONDITIONED ReBotNet XS trained in residual mode on
# gelsight_pseudo_mini tactile_normal-domain transferred data (see
# train_refine_scripts/train_rebot_pseudo_mini_tactile_normal_residual_cond/train_rebot_XS.sh).
# --normal_blank matches the fixed flat-normal blank used at training time.
#
# Usage: bash train_refine_scripts/eval_rebot_pseudo_mini_tactile_normal_residual_cond/eval_rebot_XS.sh <gpu_id> [cond_mode]
#   cond_mode: both-normal (default), mask, film-{normal,curvature,height}, both-{normal,curvature,height}
#   -- must match what the checkpoint was trained with.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash eval_rebot_XS.sh <gpu_id> [cond_mode]}
COND_MODE="${2:-both-normal}"
MODEL_SIZE=rebot_XS

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
    --transfer_dir "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal" \
    --checkpoint   "$PROJECT_ROOT/log/rebot_checkpoints_XS_pseudo_mini_tactile_normal_residual_cond-${COND_MODE}/best.pth" \
    --model_size   $MODEL_SIZE \
    --video_type   tactile_normal \
    --save_dir     "$PROJECT_ROOT/log/rebot_eval_XS_pseudo_mini_tactile_normal_residual_cond-${COND_MODE}" \
    --cond_dir     "$PROJECT_ROOT/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini" \
    --film_scale   100 \
    "${COND_FLAGS[@]}" \
    --video_save \
    --save_gt \
    --residual \
    --normal_blank
