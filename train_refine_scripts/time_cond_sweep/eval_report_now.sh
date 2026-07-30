#!/usr/bin/env bash
# One-off: snapshot the CURRENT latest.pth of each arm, eval on the 50-object
# test set (GPUs 5/6/7 in waves), and build an interim report. Independent of
# the running orchestrator (different tag; finishes long before epoch 14).
set -o pipefail

ROOT=/data1/junhokim/Projects/PatchMatch_touch
PY=/home/junhokim/miniconda3/envs/pm_touch/bin/python
TRANSFER=$ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue
COND=$ROOT/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini
ELOG=$ROOT/log/rebot_eval_logs_time_cond
GEN=$ROOT/train_refine_scripts/time_cond_sweep/gen_time_report.py
TAG=interim
mkdir -p "$ELOG"

NAMES=(none film token filmtoken concat)
declare -A MODE=( [none]=none [film]=film [token]=token [filmtoken]=film_token [concat]=concat )
GPUS=(5 6 7)

ckdir() { echo "$ROOT/log/rebot_checkpoints_S_time_$1"; }

# snapshot current latest.pth
for n in "${NAMES[@]}"; do
  cp "$(ckdir "$n")/latest.pth" "$(ckdir "$n")/${TAG}_snapshot.pth" || { echo "no latest.pth for $n yet"; exit 1; }
done

eval_arm() {
  local n=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/rebot_net/eval.py" \
    --transfer_dir "$TRANSFER" \
    --checkpoint "$(ckdir "$n")/${TAG}_snapshot.pth" \
    --model_size rebot_S \
    --video_type tactile_normal \
    --cond_dir "$COND" --film_modality normal --film_scale 100 \
    --bottleneck_hw 24 \
    --time_cond "${MODE[$n]}" \
    --save_dir "$ROOT/log/rebot_eval_S_time_${n}_${TAG}" \
    --video_save --save_gt \
    > "$ELOG/${n}_${TAG}.log" 2>&1
}

i=0
while [ $i -lt ${#NAMES[@]} ]; do
  pids=()
  for ((g=0; g<${#GPUS[@]} && i<${#NAMES[@]}; g++, i++)); do
    n=${NAMES[$i]}
    echo "[now] $(date) eval $n on gpu ${GPUS[$g]}"
    eval_arm "$n" "${GPUS[$g]}" & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
done

"$PY" "$GEN" "$TAG" "~9 (interim)"
echo "[now] $(date) INTERIM REPORT DONE -> log/tactile_normal_time_cond_${TAG}_report.html"
