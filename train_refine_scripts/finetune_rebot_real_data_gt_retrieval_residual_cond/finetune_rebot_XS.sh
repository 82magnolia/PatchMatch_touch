#!/usr/bin/env bash
# Fine-tune the CONDITIONED pseudo_mini-pretrained ReBotNet XS (residual) on
# real-data (transferred -> query) pairs. Loads the checkpoint from
# train_refine_scripts/train_rebot_pseudo_mini_residual_cond/train_rebot_XS.sh (same cond_mode ->
# matching shapes, no weight surgery). finetune_mode defaults to 'full' because
# the render_mask input weights live in the encoder stem.
#
# Usage: bash train_refine_scripts/finetune_rebot_real_data_gt_retrieval_residual_cond/finetune_rebot_XS.sh <gpu_id> [matcher] [finetune_mode] [cond_mode] [data_mode]
#   matcher:       loftr (default), disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue
#   finetune_mode: full (default), decoder_bottleneck, decoder, last
#   cond_mode:     both-normal (default), mask, film-{normal,curvature,height}, both-{normal,curvature,height}  (must match pretrain)
#   data_mode:     raw (default) or masked. 'masked' fine-tunes on the render-mask-blended
#                  transfer tree (build it with prepare_masked_transfer/make_masked_real.sh)
#                  AND loads the correspondingly masked-pretrained sim checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash finetune_rebot_XS.sh <gpu_id> [matcher] [finetune_mode] [cond_mode] [data_mode]}
MATCHER="${2:-loftr}"
FINETUNE_MODE="${3:-full}"
COND_MODE="${4:-both-normal}"
DATA_MODE="${5:-raw}"
MODEL_SIZE=rebot_XS

case "$MATCHER" in
    loftr|disk_lightglue|sift_lightglue|superpoint_lightglue|superpoint_superglue) ;;
    *)
        echo "Unknown matcher '$MATCHER'. Expected one of: loftr, disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue" >&2
        exit 1
        ;;
esac
case "$FINETUNE_MODE" in
    full|decoder_bottleneck|decoder|last) ;;
    *)
        echo "Unknown finetune_mode '$FINETUNE_MODE'. Expected one of: full, decoder_bottleneck, decoder, last" >&2
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

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/finetune.py" \
    --pretrained    "$PROJECT_ROOT/log/rebot_checkpoints_XS_pseudo_mini${CKPT_NAME}_cond-${COND_MODE}${DATA_SUFFIX}" \
    --transfer_dir  "$PROJECT_ROOT/log/transfer_pipeline_real_data_gt_retrieval${TRANSFER_SUFFIX}${DATA_SUFFIX}" \
    --save_dir      "$PROJECT_ROOT/log/rebot_finetune_XS_real_data_gt_retrieval_residual${TRANSFER_SUFFIX}_${FINETUNE_MODE}_cond-${COND_MODE}${DATA_SUFFIX}" \
    --model_size    $MODEL_SIZE \
    --finetune_mode $FINETUNE_MODE \
    --num_eval      20 \
    --epochs        3 \
    --batch_size    8 \
    --lr            5e-5 \
    --num_workers   4 \
    --residual \
    --cond_dir      "$PROJECT_ROOT/log/real_data_gt_retrieval" \
    --film_scale    4 \
    "${COND_FLAGS[@]}" \
    --wandb_project tactile_enhance \
    --wandb_run_name "finetune_XS_real_residual${TRANSFER_SUFFIX}_${FINETUNE_MODE}_cond-${COND_MODE}${DATA_SUFFIX}_bs8_lr5e-5"
