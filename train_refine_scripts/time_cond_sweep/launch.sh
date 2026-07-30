#!/usr/bin/env bash
# Temporal-conditioning sweep. Holds the agreed base recipe fixed --
#   charbonnier-only loss, zero_init_final, lambda_delta 0.1, bottleneck_hw 24,
#   FiLM-normal query conditioning, tactile_normal / superpoint_superglue transfer
# -- and varies ONLY --time_cond across 5 arms (control + 4 mechanisms), each on
# its own GPU. Conda is activated by the CALLING shell (not here) to avoid the
# set -u / conda-hook clash that killed an earlier orchestrator.
set -o pipefail

ROOT=/data1/junhokim/Projects/PatchMatch_touch
TRANSFER=$ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue
COND=$ROOT/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini
LOGDIR=$ROOT/log/rebot_train_logs_time_cond
mkdir -p "$LOGDIR"

EPOCHS=20

# arm_name  gpu  time_cond
ARMS=(
  "none        0 none"
  "film        1 film"
  "token       2 token"
  "filmtoken   3 film_token"
  "concat      4 concat"
)

for spec in "${ARMS[@]}"; do
  read -r NAME GPU MODE <<< "$spec"
  SAVE=$ROOT/log/rebot_checkpoints_S_time_${NAME}
  echo "launching arm=$NAME gpu=$GPU time_cond=$MODE -> $SAVE"
  CUDA_VISIBLE_DEVICES=$GPU nohup python "$ROOT/rebot_net/train.py" \
    --transfer_dir "$TRANSFER" \
    --save_dir "$SAVE" \
    --model_size rebot_S \
    --video_type tactile_normal \
    --epochs $EPOCHS \
    --batch_size 8 \
    --lr 2e-4 \
    --num_workers 8 \
    --cond_dir "$COND" \
    --film_modality normal \
    --film_scale 100 \
    --zero_init_final \
    --lambda_delta 0.1 \
    --bottleneck_hw 24 \
    --time_cond "$MODE" \
    --wandb_project tactile_enhance \
    --wandb_run_name "time_${NAME}_20ep" \
    > "$LOGDIR/${NAME}.log" 2>&1 &
  echo "  pid=$!"
  sleep 8   # stagger starts so wandb init / dataset scan don't collide
done

echo "all arms launched."
