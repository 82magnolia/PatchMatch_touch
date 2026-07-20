#!/usr/bin/env bash
# Pretrain a CONDITIONED ReBotNet S on gelsight_pseudo_mini in sim.
# Query conditioning is baked into the architecture so the real-data fine-tune
# loads it with matching shapes (no weight surgery). --mask_cond concatenates the
# aligned per-frame render_mask; --film_modality injects a static query geometry
# render (normal/curvature/height) via FiLM. See rebot_net/cond_utils.py.
#
# Usage: bash train_refine_scripts/train_rebot_pseudo_mini_cond/train_rebot_S.sh <gpu_id> [matcher] [cond_mode]
#   matcher:   loftr (default), disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue
#   cond_mode: both-normal (default), mask, film-{normal,curvature,height}, both-{normal,curvature,height}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash train_rebot_S.sh <gpu_id> [matcher] [cond_mode]}
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
# Conditioning config (must match between sim pretrain and real finetune so the
# checkpoint loads without weight surgery). film-<mod> picks the geometry render.
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

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/train.py" \
    --transfer_dir  "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini${TRANSFER_SUFFIX}" \
    --save_dir      "$PROJECT_ROOT/log/rebot_checkpoints_S_pseudo_mini${CKPT_NAME}_cond-${COND_MODE}" \
    --model_size    $MODEL_SIZE \
    --epochs        100 \
    --batch_size    8 \
    --lr            2e-4 \
    --num_workers   4 \
    --cond_dir      "$PROJECT_ROOT/Taxim/results/gen_contact_full_query_pseudo_mini" \
    --film_scale    100 \
    "${COND_FLAGS[@]}" \
    --wandb_project tactile_enhance \
    --wandb_run_name "${MODEL_SIZE}_pseudo_mini${CKPT_NAME}_cond-${COND_MODE}_bs8_lr2e-4"
