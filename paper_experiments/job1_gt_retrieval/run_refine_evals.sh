#!/usr/bin/env bash
# Job 1 (+ two ablation arms): ReBotNet refinement on the ground-truth-retrieval
# benchmark, 50 eval objects (951-1000).
#
#   ours          OURS               (geom_concat normal + temporal FiLM)
#   wo_film       w/o Temporal FiLM  (geom_concat normal, no time conditioning)
#   wo_normalcat  w/o Normal concat  (FiLM-injected normal, no concatenation)
#
# MODALITY selects which coarse transfer to refine:
#   normal    (default) log/paper_job1_transfer_normal -- surface normals at 4x,
#             the alignment the paper describes, rebuilt by run_transfer_normal.sh
#   curvature           the original transfer dir, which is what the refinement
#             checkpoints were trained on; keep for the mismatch comparison
#
# Uses paper_experiments/eval_refine_generic.py rather than rebot_net/eval.py:
# the latter hard-codes its test split as all_ids[950:], which is EMPTY for a
# directory that contains only the 50 eval objects.
set -o pipefail

ROOT=/data1/junhokim/Projects/PatchMatch_touch
PY=/home/junhokim/miniconda3/envs/pm_touch/bin/python
COND=$ROOT/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini
ELOG=$ROOT/paper_experiments/job1_gt_retrieval/logs
mkdir -p "$ELOG"

MODALITY="${MODALITY:-normal}"
if [ "$MODALITY" = "normal" ]; then
  TRANSFER=$ROOT/log/paper_job1_transfer_normal
  SUF=_normal
else
  TRANSFER=$ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue
  SUF=
fi

run_arm () {
  local name=$1 gpu=$2 ckpt=$3
  shift 3
  local edir=$ROOT/log/paper_job1_refine_${name}${SUF}
  echo "[job1] $(date) eval ${name} on gpu ${gpu} -> ${edir}"
  CUDA_VISIBLE_DEVICES=$gpu NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  "$PY" "$ROOT/paper_experiments/eval_refine_generic.py" \
    --transfer_dir "$TRANSFER" \
    --layout       flat \
    --object_ids   $(seq 951 1000) \
    --num_pairs    8 \
    --checkpoint   "$ckpt" \
    --model_size   rebot_S \
    --video_type   tactile_normal \
    --cond_dir     "$COND" \
    --film_scale   100 \
    --bottleneck_hw 24 \
    --save_dir     "$edir" \
    --video_save --save_gt --max_videos 8 \
    "$@" \
    > "$ELOG/refine_${name}${SUF}.log" 2>&1 &
}

run_arm ours 5 \
  "$ROOT/log/rebot_checkpoints_S_geomcat_film/best.pth" \
  --film_modality normal --geom_concat --time_cond film

run_arm wo_film 6 \
  "$ROOT/log/rebot_checkpoints_S_geomcat_none/best.pth" \
  --film_modality normal --geom_concat --time_cond none

run_arm wo_normalcat 7 \
  "$ROOT/log/rebot_checkpoints_S_pseudo_mini_tactile_normal_superpoint_superglue_cond-film-normal/best.pth" \
  --film_modality normal --time_cond none

wait
echo "[job1] $(date) all refinement evals done."
