#!/usr/bin/env bash
# Pretrain a CONDITIONED ReBotNet L (residual) on gelsight_pseudo_mini in sim.
# Query conditioning is baked into the architecture so the real-data fine-tune
# loads it with matching shapes (no weight surgery). --mask_cond concatenates the
# aligned per-frame render_mask; --film_modality injects a static query geometry
# render (normal/curvature/height) via FiLM. See rebot_net/cond_utils.py.
#
# Usage: bash train_refine_scripts/train_rebot_pseudo_mini_residual_cond/train_rebot_L.sh <gpu_id> [matcher] [cond_mode] [data_mode]
#   matcher:   loftr (default), disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue
#   cond_mode: both-normal (default), mask, film-{normal,curvature,height}, both-{normal,curvature,height}
#   data_mode: raw (default) or masked (render-mask-blended transfer tree; build it
#              first with train_refine_scripts/prepare_masked_transfer/make_masked_sim.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash train_rebot_L.sh <gpu_id> [matcher] [cond_mode] [data_mode]}
MATCHER="${2:-loftr}"
COND_MODE="${3:-both-normal}"
DATA_MODE="${4:-raw}"
MODEL_SIZE=rebot_L

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
# Training data variant. 'masked' uses the render-mask-blended transfer tree
# produced by postprocess_mask_transfer.py (see prepare_masked_transfer/), i.e.
#   input = mask * transferred + (1 - mask) * base_frame
# so the network already receives the query contact footprint in the pixels.
case "$DATA_MODE" in
    raw)    DATA_SUFFIX="" ;;
    masked) DATA_SUFFIX="_masked" ;;
    *)
        echo "Unknown data_mode '$DATA_MODE'. Expected one of: raw, masked" >&2
        exit 1
        ;;
esac

if [ "$MATCHER" = "loftr" ]; then
    TRANSFER_SUFFIX=""
    CKPT_NAME="_residual"
else
    TRANSFER_SUFFIX="_${MATCHER}"
    CKPT_NAME="_residual_${MATCHER}"
fi

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/train.py" \
    --transfer_dir  "$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini${TRANSFER_SUFFIX}${DATA_SUFFIX}" \
    --save_dir      "$PROJECT_ROOT/log/rebot_checkpoints_L_pseudo_mini${CKPT_NAME}_cond-${COND_MODE}${DATA_SUFFIX}" \
    --model_size    $MODEL_SIZE \
    --epochs        100 \
    --batch_size    4 \
    --lr            2e-4 \
    --num_workers   4 \
    --residual \
    --cond_dir      "$PROJECT_ROOT/Taxim/results/gen_contact_full_query_pseudo_mini" \
    --film_scale    100 \
    "${COND_FLAGS[@]}" \
    --wandb_project tactile_enhance \
    --wandb_run_name "${MODEL_SIZE}_pseudo_mini${CKPT_NAME}_cond-${COND_MODE}${DATA_SUFFIX}_bs4_lr2e-4"
