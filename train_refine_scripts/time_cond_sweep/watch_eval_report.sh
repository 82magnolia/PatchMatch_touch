#!/usr/bin/env bash
# Orchestrate the temporal-conditioning sweep's evaluation + reporting.
# For each milestone epoch: wait until every arm has completed it, snapshot each
# arm's latest.pth, eval on the 50-object test set, then build one HTML report.
# Uses the env python by absolute path (no conda activate -> no set-u/hook clash).
set -o pipefail

ROOT=/data1/junhokim/Projects/PatchMatch_touch
PY=/home/junhokim/miniconda3/envs/pm_touch/bin/python
TRANSFER=$ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue
COND=$ROOT/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini
TLOG=$ROOT/log/rebot_train_logs_time_cond
ELOG=$ROOT/log/rebot_eval_logs_time_cond
GEN=$ROOT/train_refine_scripts/time_cond_sweep/gen_time_report.py
mkdir -p "$ELOG"

NAMES=(none film token filmtoken concat)
declare -A MODE=( [none]=none [film]=film [token]=token [filmtoken]=film_token [concat]=concat )

ckdir()  { echo "$ROOT/log/rebot_checkpoints_S_time_$1"; }

# wait_marker <grep-pattern>: block until every arm's training log contains it
wait_all_marker() {
  local pat="$1"
  while true; do
    local ready=1
    for n in "${NAMES[@]}"; do
      grep -q "$pat" "$TLOG/$n.log" 2>/dev/null || { ready=0; break; }
    done
    [ "$ready" -eq 1 ] && return 0
    sleep 120
  done
}

# eval_arm <name> <gpu> <snapshot> <evaldir>
eval_arm() {
  local n=$1 gpu=$2 snap=$3 edir=$4
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/rebot_net/eval.py" \
    --transfer_dir "$TRANSFER" \
    --checkpoint "$snap" \
    --model_size rebot_S \
    --video_type tactile_normal \
    --cond_dir "$COND" --film_modality normal --film_scale 100 \
    --bottleneck_hw 24 \
    --time_cond "${MODE[$n]}" \
    --save_dir "$edir" \
    --video_save --save_gt \
    > "$ELOG/${n}_$5.log" 2>&1
}

# do_milestone <epoch> <tag> <gpu-list...>
do_milestone() {
  local epoch=$1 tag=$2; shift 2
  local gpus=("$@")
  echo "[watch] $(date) milestone epoch=$epoch tag=$tag: snapshotting + evaluating"
  # snapshot each arm's latest.pth
  for n in "${NAMES[@]}"; do
    cp "$(ckdir "$n")/latest.pth" "$(ckdir "$n")/${tag}_snapshot.pth"
  done
  # eval, filling the gpu pool in waves
  local ng=${#gpus[@]} i=0
  while [ $i -lt ${#NAMES[@]} ]; do
    local pids=()
    for ((g=0; g<ng && i<${#NAMES[@]}; g++, i++)); do
      local n=${NAMES[$i]}
      local edir="$ROOT/log/rebot_eval_S_time_${n}_${tag}"
      eval_arm "$n" "${gpus[$g]}" "$(ckdir "$n")/${tag}_snapshot.pth" "$edir" "$tag" &
      pids+=($!)
      echo "[watch] $(date)   eval $n on gpu ${gpus[$g]} -> $edir"
    done
    for p in "${pids[@]}"; do wait "$p"; done
  done
  "$PY" "$GEN" "$tag" "$epoch"
  echo "[watch] $(date) milestone $tag DONE -> log/tactile_normal_time_cond_${tag}_report.html"
}

echo "[watch] $(date) waiting for all arms to finish epoch 14 (marker: === Epoch 15/20 ===)"
wait_all_marker "=== Epoch 15/20 ==="
do_milestone 14 mid 5 6 7

echo "[watch] $(date) waiting for all arms to finish training (marker: Training complete)"
wait_all_marker "Training complete"
do_milestone 20 final 0 1 2 3 4

echo "[watch] $(date) ALL DONE (mid + final reports written)"
