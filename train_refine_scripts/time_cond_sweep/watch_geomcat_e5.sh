#!/usr/bin/env bash
# Wait for all 3 geom-concat arms to finish epoch 5, snapshot, eval on the
# 50-object test set (each arm on its OWN training GPU -- 24GB cards, ~20GB free,
# so eval shares safely), then build the epoch-5 report. Absolute-path python
# (no conda activate). Training continues to epoch 20 afterwards untouched.
set -o pipefail

ROOT=/data1/junhokim/Projects/PatchMatch_touch
PY=/home/junhokim/miniconda3/envs/pm_touch/bin/python
TRANSFER=$ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue
COND=$ROOT/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini
TLOG=$ROOT/log/rebot_train_logs_geomcat
ELOG=$ROOT/log/rebot_eval_logs_geomcat
GEN=$ROOT/train_refine_scripts/time_cond_sweep/gen_geom_report.py
mkdir -p "$ELOG"
TAG=e5

NAMES=(none film token)
declare -A MODE=( [none]=none [film]=film [token]=token )
declare -A GPU=(  [none]=5    [film]=6    [token]=7 )

ckdir() { echo "$ROOT/log/rebot_checkpoints_S_geomcat_$1"; }

echo "[geomcat-e5] $(date) waiting for all 3 arms to finish epoch 5 (marker: === Epoch 6/20 ===)"
while true; do
  ready=1
  for n in "${NAMES[@]}"; do
    grep -q "=== Epoch 6/20 ===" "$TLOG/$n.log" 2>/dev/null || { ready=0; break; }
  done
  [ "$ready" -eq 1 ] && break
  sleep 120
done

echo "[geomcat-e5] $(date) epoch 5 reached; snapshotting + evaluating"
for n in "${NAMES[@]}"; do
  cp "$(ckdir "$n")/latest.pth" "$(ckdir "$n")/${TAG}_snapshot.pth"
done

pids=()
for n in "${NAMES[@]}"; do
  edir="$ROOT/log/rebot_eval_S_geomcat_${n}_${TAG}"
  CUDA_VISIBLE_DEVICES=${GPU[$n]} "$PY" "$ROOT/rebot_net/eval.py" \
    --transfer_dir "$TRANSFER" \
    --checkpoint "$(ckdir "$n")/${TAG}_snapshot.pth" \
    --model_size rebot_S \
    --video_type tactile_normal \
    --cond_dir "$COND" --film_modality normal --film_scale 100 --geom_concat \
    --bottleneck_hw 24 \
    --time_cond "${MODE[$n]}" \
    --save_dir "$edir" \
    --video_save --save_gt \
    > "$ELOG/${n}_${TAG}.log" 2>&1 &
  pids+=($!)
  echo "[geomcat-e5] $(date)   eval $n on gpu ${GPU[$n]} -> $edir"
done
for p in "${pids[@]}"; do wait "$p"; done

"$PY" "$GEN" "$TAG" 5
echo "[geomcat-e5] $(date) DONE -> log/tactile_normal_geomcat_${TAG}_report.html"
