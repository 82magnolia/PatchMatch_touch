#!/usr/bin/env bash
# Fine-tune a pseudo_mini-pretrained ReBotNet XS variant on real-data
# (transferred -> query) pairs from the GT-retrieval pipeline in residual mode.
#
# Loads the synthetic-pretrained checkpoint produced by
# train_refine_scripts/train_rebot_pseudo_mini_residual/train_rebot_XS.sh and adapts it to the
# real GelSight domain. Split: the first 20 of the 100 objects are held out for
# eval, the remaining 80 are used for fine-tuning (see rebot_net/finetune.py).
#
# Usage: bash train_refine_scripts/finetune_rebot_real_data_gt_retrieval_residual/finetune_rebot_XS.sh <gpu_id> [matcher] [refined] [finetune_mode]
#   from the PatchMatch_touch project root.
#   matcher: loftr (default), disk_lightglue, sift_lightglue,
#            superpoint_lightglue, superpoint_superglue -- selects BOTH the
#            pretrained pseudo_mini checkpoint and the real-data transfer dir.
#   refined: pass the literal 'refined' to fine-tune on the --photometric_refine
#            real-data transfer dir (transfer_pipeline_real_data_gt_retrieval_refined_*)
#            instead of the base one; pass nothing to skip.
#   finetune_mode: which layers to unfreeze -- decoder_bottleneck (default),
#            full, decoder, or last (see rebot_net/finetune.py).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash finetune_rebot_XS.sh <gpu_id> [matcher] [refined] [finetune_mode]}
MATCHER="${2:-loftr}"
REFINED="${3:-}"
FINETUNE_MODE="${4:-decoder_bottleneck}"
MODEL_SIZE=rebot_XS

case "$MATCHER" in
    loftr|disk_lightglue|sift_lightglue|superpoint_lightglue|superpoint_superglue) ;;
    *)
        echo "Unknown matcher '$MATCHER'. Expected one of: loftr, disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue" >&2
        exit 1
        ;;
esac

case "$REFINED" in
    ""|refined) ;;
    *)
        echo "Unknown refine option '$REFINED'. Expected empty or 'refined'." >&2
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

# Pretrained checkpoint suffix: loftr is suffix-less, every other matcher gets a
# _<matcher> suffix (mirrors train_rebot_pseudo_mini_residual/'s save_dir naming).
if [ "$MATCHER" = "loftr" ]; then
    CKPT_SUFFIX=""
else
    CKPT_SUFFIX="_${MATCHER}"
fi

# Real-data transfer dir: the base (non-refined) loftr run is suffix-less and
# every other matcher gets _<matcher>; the refined (--photometric_refine)
# variant is always _refined_<matcher>, loftr included (mirrors
# transfer_all_real_data_gt_retrieval/'s OUT_BASE convention).
if [ "$REFINED" = "refined" ]; then
    TRANSFER_SUFFIX="_refined_${MATCHER}"
else
    TRANSFER_SUFFIX="$CKPT_SUFFIX"
fi

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/finetune.py" \
    --pretrained    "$PROJECT_ROOT/log/rebot_checkpoints_XS_pseudo_mini_residual${CKPT_SUFFIX}" \
    --transfer_dir  "$PROJECT_ROOT/log/transfer_pipeline_real_data_gt_retrieval${TRANSFER_SUFFIX}" \
    --save_dir      "$PROJECT_ROOT/log/rebot_finetune_XS_real_data_gt_retrieval_residual${TRANSFER_SUFFIX}_${FINETUNE_MODE}" \
    --model_size    $MODEL_SIZE \
    --finetune_mode $FINETUNE_MODE \
    --num_eval      20 \
    --epochs        8 \
    --batch_size    8 \
    --lr            5e-5 \
    --num_workers   4 \
    --residual \
    --wandb_project tactile_enhance \
    --wandb_run_name "finetune_XS_real_residual${TRANSFER_SUFFIX}_${FINETUNE_MODE}_bs8_lr5e-5"
