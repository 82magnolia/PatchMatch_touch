#!/usr/bin/env bash
# One-off: snapshot the CURRENT latest.pth of each geom-concat arm, eval on the
# 50-object test set, build a visual report NOW (tag 'now'). Each arm evals on
# its own training GPU (24GB cards, ~20GB free -> shares safely). Independent of
# the epoch-5 watcher (different tag).
set -o pipefail

ROOT=/data1/junhokim/Projects/PatchMatch_touch
PY=/home/junhokim/miniconda3/envs/pm_touch/bin/python
TRANSFER=$ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue
COND=$ROOT/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini
ELOG=$ROOT/log/rebot_eval_logs_geomcat
GEN=$ROOT/train_refine_scripts/time_cond_sweep/gen_geom_report.py
mkdir -p "$ELOG"
TAG=now

NAMES=(none film token)
declare -A MODE=( [none]=none [film]=film [token]=token )
declare -A GPU=(  [none]=5    [film]=6    [token]=7 )
ckdir() { echo "$ROOT/log/rebot_checkpoints_S_geomcat_$1"; }

for n in "${NAMES[@]}"; do
  cp "$(ckdir "$n")/latest.pth" "$(ckdir "$n")/${TAG}_snapshot.pth" || { echo "no latest.pth for $n"; exit 1; }
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
  echo "[geomcat-now] $(date) eval $n on gpu ${GPU[$n]} -> $edir"
done
for p in "${pids[@]}"; do wait "$p"; done

"$PY" "$GEN" "$TAG" "~1 (very early)"
echo "[geomcat-now] $(date) DONE -> log/tactile_normal_geomcat_${TAG}_report.html"
