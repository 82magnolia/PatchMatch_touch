#!/usr/bin/env bash
# Fine-tune the CONDITIONED pseudo_mini-pretrained ReBotNet M (residual) on
# real-data (transferred -> query) pairs in the TACTILE_NORMAL video domain --
# i.e. the surface-normal-encoded videos ({idx}_tactile_normal.mp4), not the
# shadow/appearance domain used by
# finetune_rebot_real_data_gt_retrieval_residual_cond/.
#
# Loads the checkpoint from
# train_refine_scripts/train_rebot_pseudo_mini_tactile_normal_residual_cond/train_rebot_M.sh
# (same cond_mode -> matching shapes, no weight surgery) and fine-tunes on the
# transfer tree built by
# train_refine_scripts/transfer_all_real_data_gt_retrieval_tactile_normal/run_superpoint_superglue.sh.
# finetune_mode defaults to 'full' because the render_mask input weights live in
# the encoder stem.
# --normal_blank matches the pretrain: in the tactile_normal domain the
# no-contact reading is the fixed flat normal (0,0,1) rather than frame 0 of
# the transferred video, so the residual is taken against that constant.
#
# Defaults differ from the shadow-domain scripts: matcher is superpoint_superglue
# and cond_mode is film-normal, matching the only tactile_normal sim checkpoints
# that exist (log/rebot_checkpoints_S_pseudo_mini_tactile_normal_residual_superpoint_superglue_cond-film-normal).
#
# Usage: bash train_refine_scripts/finetune_rebot_real_data_gt_retrieval_tactile_normal_residual_cond/finetune_rebot_M.sh <gpu_id> [matcher] [finetune_mode] [cond_mode] [data_mode]
#   matcher:       superpoint_superglue (default), loftr, disk_lightglue, sift_lightglue, superpoint_lightglue
#   finetune_mode: full (default), decoder_bottleneck, decoder, last
#   cond_mode:     film-normal (default), mask, film-{normal,curvature,height}, both-{normal,curvature,height}  (must match pretrain)
#   data_mode:     raw (default) or masked. 'masked' fine-tunes on the render-mask-blended
#                  transfer tree (build it with prepare_masked_transfer/make_masked_real.sh)
#                  AND loads the correspondingly masked-pretrained sim checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash finetune_rebot_M.sh <gpu_id> [matcher] [finetune_mode] [cond_mode] [data_mode]}
MATCHER="${2:-superpoint_superglue}"
FINETUNE_MODE="${3:-full}"
COND_MODE="${4:-film-normal}"
DATA_MODE="${5:-raw}"
MODEL_SIZE=rebot_M

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
# Note this conditions on the query *geometry* render, never on the
# tactile_normal video itself (that is derived from the GT -- see cond_utils.py).
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

# Two different "default matcher = no suffix" conventions collide here, so the
# suffixes are computed separately:
#   sim pretrain tree (train_rebot_pseudo_mini_tactile_normal*) omits the suffix
#     for disk_lightglue;
#   real transfer tree (transfer_all_real_data_gt_retrieval*) omits it for loftr.
if [ "$MATCHER" = "disk_lightglue" ]; then
    CKPT_SUFFIX=""
else
    CKPT_SUFFIX="_${MATCHER}"
fi
if [ "$MATCHER" = "loftr" ]; then
    TRANSFER_SUFFIX=""
else
    TRANSFER_SUFFIX="_${MATCHER}"
fi

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/finetune.py" \
    --pretrained    "$PROJECT_ROOT/log/rebot_checkpoints_M_pseudo_mini_tactile_normal_residual${CKPT_SUFFIX}_cond-${COND_MODE}${DATA_SUFFIX}" \
    --transfer_dir  "$PROJECT_ROOT/log/transfer_pipeline_real_data_gt_retrieval_tactile_normal${TRANSFER_SUFFIX}${DATA_SUFFIX}" \
    --save_dir      "$PROJECT_ROOT/log/rebot_finetune_M_real_data_gt_retrieval_tactile_normal_residual${TRANSFER_SUFFIX}_${FINETUNE_MODE}_cond-${COND_MODE}${DATA_SUFFIX}" \
    --model_size    $MODEL_SIZE \
    --finetune_mode $FINETUNE_MODE \
    --video_type    tactile_normal \
    --num_eval      20 \
    --epochs        8 \
    --batch_size    8 \
    --lr            5e-5 \
    --num_workers   4 \
    --residual \
    --normal_blank \
    --cond_dir      "$PROJECT_ROOT/log/real_data_gt_retrieval" \
    --film_scale    8 \
    "${COND_FLAGS[@]}" \
    --wandb_project tactile_enhance \
    --wandb_run_name "finetune_M_real_tactile_normal_residual${TRANSFER_SUFFIX}_${FINETUNE_MODE}_cond-${COND_MODE}${DATA_SUFFIX}_bs8_lr5e-5"
