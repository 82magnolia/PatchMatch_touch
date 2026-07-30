#!/usr/bin/env bash
# Second sweep: inject the query NORMAL render by CONCATENATION (3 aligned input
# channels) instead of via FiLM (--geom_concat, FiLM off), combined with each of
# {no time, sinusoidal FiLM time, sinusoidal token time}. 3 arms on GPUs 5/6/7.
# Base recipe otherwise identical to the first sweep (charbonnier + zero_init +
# lambda_delta 0.1, bottleneck_hw 24). Conda activated by the CALLING shell.
set -o pipefail

ROOT=/data1/junhokim/Projects/PatchMatch_touch
TRANSFER=$ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue
COND=$ROOT/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini
LOGDIR=$ROOT/log/rebot_train_logs_geomcat
mkdir -p "$LOGDIR"

EPOCHS=20

# arm_name  gpu  time_cond
ARMS=(
  "none   5 none"
  "film   6 film"
  "token  7 token"
)

for spec in "${ARMS[@]}"; do
  read -r NAME GPU MODE <<< "$spec"
  SAVE=$ROOT/log/rebot_checkpoints_S_geomcat_${NAME}
  echo "launching geomcat arm=$NAME gpu=$GPU time_cond=$MODE -> $SAVE"
  CUDA_VISIBLE_DEVICES=$GPU nohup python "$ROOT/rebot_net/train.py" \
    --transfer_dir "$TRANSFER" \
    --save_dir "$SAVE" \
    --model_size rebot_S \
    --video_type tactile_normal \
    --epochs $EPOCHS \
    --batch_size 8 \
    --lr 2e-4 \
    --num_workers 6 \
    --cond_dir "$COND" \
    --film_modality normal \
    --film_scale 100 \
    --geom_concat \
    --zero_init_final \
    --lambda_delta 0.1 \
    --bottleneck_hw 24 \
    --time_cond "$MODE" \
    --wandb_project tactile_enhance \
    --wandb_run_name "geomcat_${NAME}_20ep" \
    > "$LOGDIR/${NAME}.log" 2>&1 &
  echo "  pid=$!"
  sleep 8
done

echo "all geomcat arms launched."
