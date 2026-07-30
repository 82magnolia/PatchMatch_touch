#!/usr/bin/env bash
# Train a CONDITIONED ReBotNet S FROM SCRATCH (random init) on real-data
# (transferred -> query) pairs. This is the from-scratch counterpart of
# train_refine_scripts/finetune_rebot_real_data_gt_retrieval_cond/finetune_rebot_S.sh:
# it uses the *same* real transfer tree, the same cond_dir/film_scale, and the
# same first-num_eval / rest split (via train.py --real_data), differing only in
# that no sim-pretrained checkpoint is loaded. Use it as a control to test whether
# the network can overfit the real data at all (as it does in sim) and thereby
# isolate whether the sim->real fine-tune init/freezing is the bottleneck.
#
# Usage: bash train_refine_scripts/train_rebot_real_data_gt_retrieval_cond/train_rebot_S.sh <gpu_id> [matcher] [cond_mode] [data_mode]
#   matcher:   loftr (default), disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue
#   cond_mode: both-normal (default), mask, film-{normal,curvature,height}, both-{normal,curvature,height}
#   data_mode: raw (default) or masked (render-mask-blended transfer tree; build it
#              first with train_refine_scripts/prepare_masked_transfer/make_masked_real.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash train_rebot_S.sh <gpu_id> [matcher] [cond_mode] [data_mode]}
MATCHER="${2:-loftr}"
COND_MODE="${3:-both-normal}"
DATA_MODE="${4:-raw}"
MODEL_SIZE=rebot_S

case "$MATCHER" in
    loftr|disk_lightglue|sift_lightglue|superpoint_lightglue|superpoint_superglue) ;;
    *)
        echo "Unknown matcher '$MATCHER'. Expected one of: loftr, disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue" >&2
        exit 1
        ;;
esac
# Conditioning config (same flags as pretrain/finetune so the architecture matches).
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
# Training data variant. 'masked' uses the render-mask-blended transfer tree.
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
else
    TRANSFER_SUFFIX="_${MATCHER}"
fi

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/train.py" \
    --real_data \
    --transfer_dir  "$PROJECT_ROOT/log/transfer_pipeline_real_data_gt_retrieval${TRANSFER_SUFFIX}${DATA_SUFFIX}" \
    --save_dir      "$PROJECT_ROOT/log/rebot_scratch_S_real_data_gt_retrieval${TRANSFER_SUFFIX}_cond-${COND_MODE}${DATA_SUFFIX}" \
    --model_size    $MODEL_SIZE \
    --num_eval      20 \
    --epochs        100 \
    --batch_size    8 \
    --lr            2e-4 \
    --num_workers   4 \
    --cond_dir      "$PROJECT_ROOT/log/real_data_gt_retrieval" \
    --film_scale    4 \
    "${COND_FLAGS[@]}" \
    --wandb_project tactile_enhance \
    --wandb_run_name "scratch_S_real${TRANSFER_SUFFIX}_cond-${COND_MODE}${DATA_SUFFIX}_bs8_lr2e-4"
