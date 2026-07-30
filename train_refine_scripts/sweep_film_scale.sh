#!/usr/bin/env bash
# Sweep the conditioning-normal render scale (--film_scale) for the real-data
# tactile_normal fine-tune.
#
# Motivation: the sim pretrain conditions on {pair}_scale100_normal.jpg (Taxim's
# scale tag is the object's max-axis length in mm), while the real fine-tune
# conditions on {pair}_scale8_normal.jpg (the real tag is a render-scale
# multiplier of the 18.6 x 14.3 mm sensor field of view, so scale8 = 148.8 mm).
# The two domains therefore show the network geometry at different zoom levels.
# This sweeps the real side over every scale that exists on disk.
#
# Everything except --film_scale, --save_dir, --epochs and the wandb run name is
# identical to finetune_rebot_real_data_gt_retrieval_tactile_normal{,_residual}_cond
# /finetune_rebot_S.sh.
#
# Epochs are 5 rather than 8: both 8-epoch runs peaked at epoch 2 and drifted
# down afterwards, so 5 covers the peak with margin at ~60% of the cost.
#
# Usage: bash train_refine_scripts/sweep_film_scale.sh <gpu_id> <variant> <scale>
#   variant: nonresidual | residual
#   scale:   1 | 2 | 4 | 8
set -euo pipefail

GPU=${1:?usage: sweep_film_scale.sh <gpu_id> <nonresidual|residual> <scale>}
VARIANT=${2:?}
SCALE=${3:?}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

case "$VARIANT" in
    nonresidual) CKPT_INFIX=""; RES_FLAGS=() ;;
    residual)    CKPT_INFIX="_residual"; RES_FLAGS=(--residual) ;;
    *) echo "variant must be nonresidual|residual" >&2; exit 1 ;;
esac

PRETRAINED="$PROJECT_ROOT/log/rebot_checkpoints_S_pseudo_mini_tactile_normal${CKPT_INFIX}_superpoint_superglue_cond-film-normal"
TRANSFER="$PROJECT_ROOT/log/transfer_pipeline_real_data_gt_retrieval_tactile_normal_superpoint_superglue"
SAVE="$PROJECT_ROOT/log/sweep_film_scale/${VARIANT}_scale${SCALE}"

mkdir -p "$SAVE"
CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/finetune.py" \
    --pretrained    "$PRETRAINED" \
    --transfer_dir  "$TRANSFER" \
    --save_dir      "$SAVE" \
    --model_size    rebot_S \
    --finetune_mode full \
    --video_type    tactile_normal \
    --num_eval      20 \
    --epochs        5 \
    --batch_size    8 \
    --lr            5e-5 \
    --num_workers   4 \
    --cond_dir      "$PROJECT_ROOT/log/real_data_gt_retrieval" \
    --film_modality normal \
    --film_scale    "$SCALE" \
    "${RES_FLAGS[@]}" \
    --wandb_project tactile_enhance \
    --wandb_run_name "sweep_filmscale_${VARIANT}_s${SCALE}"
