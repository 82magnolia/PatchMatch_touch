#!/usr/bin/env bash
# Job 2 stage 3: refine the full-pipeline coarse transfers with the paper's
# network (normal concatenation + temporal FiLM).
#
# Usage: bash run_refine.sh <gpu_id>
set -o pipefail

GPU_ID="${1:?Usage: $0 <gpu_id> [modality]}"
# Which coarse transfer to refine. "normal" is the method default.
MODALITY="${2:-normal}"

ROOT=/data1/junhokim/Projects/PatchMatch_touch
PY=/home/junhokim/miniconda3/envs/pm_touch/bin/python
if [ "$MODALITY" = "normal" ]; then
    TRANSFER=$ROOT/log/paper_job2_pipeline_normal
    OUT=$ROOT/log/paper_job2_refine_ours_normal
else
    TRANSFER=$ROOT/log/paper_job2_pipeline
    OUT=$ROOT/log/paper_job2_refine_ours
fi
COND=$ROOT/Taxim/results/gen_contact_raw_eval_tactile_normal_pseudo_mini
LOGDIR=$ROOT/paper_experiments/job2_full_pipeline/logs
mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NUM_THREADS="${NUM_THREADS:-8}"
export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"

# --num_pairs 32 covers the largest touch index in the benchmark (objects have
# up to 31 touches); indices with no transferred video are skipped.
"$PY" "$ROOT/paper_experiments/eval_refine_generic.py" \
    --transfer_dir "$TRANSFER" \
    --layout nested \
    --num_pairs 32 \
    --checkpoint "$ROOT/log/rebot_checkpoints_S_geomcat_film/best.pth" \
    --model_size rebot_S \
    --video_type tactile_normal \
    --cond_dir "$COND" \
    --film_scale 100 \
    --bottleneck_hw 24 \
    --film_modality normal --geom_concat --time_cond film \
    --save_dir "$OUT" \
    --video_save --save_gt --max_videos 5 \
    > "$LOGDIR/refine_ours_${MODALITY}.log" 2>&1

echo "[job2-refine] done -> $OUT/metrics.json"
