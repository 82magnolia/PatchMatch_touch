#!/usr/bin/env bash
# Evaluate ReBotNet L variant trained in residual mode on 240×320 data.
# Usage: bash train_refine_scripts/eval_rebot_240x320_residual/eval_rebot_L.sh <gpu_id>
#   from the PatchMatch_touch project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU=${1:?Usage: bash eval_rebot_L.sh <gpu_id>}
MODEL_SIZE=rebot_L

CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/rebot_net/eval.py" \
    --transfer_dir "$PROJECT_ROOT/log/transfer_240x320" \
    --checkpoint   "$PROJECT_ROOT/log/rebot_checkpoints_L_240x320_residual/best.pth" \
    --model_size   $MODEL_SIZE \
    --save_dir     "$PROJECT_ROOT/log/rebot_eval_L_240x320_residual" \
    --video_save \
    --save_gt \
    --residual
