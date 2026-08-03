#!/usr/bin/env bash
# Job 3, part C/D/E: network-refinement ablations on the 20-object subset of the
# full-pipeline benchmark.
#
#   ours    geom_concat normal + temporal FiLM   (log/rebot_checkpoints_S_geomcat_film)
#   wo_film w/o temporal FiLM                    (log/rebot_checkpoints_S_geomcat_none)
#   wo_cat  w/o normal concatenation, FiLM'd     (..._cond-film-normal)
#
# "w/o neural-network refinement" needs no run: it is the coarse-transfer metric
# already produced by run_coarse_ablation.sh's scale_4x/mod_curvature arm.
#
# All three read the SAME coarse transfer (the default arm) so only the network
# differs. Usage: bash run_refine_arms.sh <arm> <gpu_id> [coarse_arm]
set -o pipefail

ARM="${1:?Usage: $0 <ours|wo_film|wo_cat> <gpu_id> [coarse_arm]}"
GPU_ID="${2:?gpu id}"
# Default coarse alignment is surface normals at 4x, so the network arms must
# refine THAT transfer -- otherwise the "w/o refinement" row (which reports the
# default coarse arm) is not comparable to the refined rows.
COARSE_ARM="${3:-mod_normal}"

ROOT=/data1/junhokim/Projects/PatchMatch_touch
PY=/home/junhokim/miniconda3/envs/pm_touch/bin/python
TRANSFER=$ROOT/log/paper_job3_ablation/$COARSE_ARM
COND=$ROOT/Taxim/results/gen_contact_raw_eval_tactile_normal_pseudo_mini
SUBSET=$ROOT/paper_experiments/job3_ablation/subset_objects.txt
OUT=$ROOT/log/paper_job3_refine_${ARM}_${COARSE_ARM}
LOGDIR=$ROOT/paper_experiments/job3_ablation/logs
mkdir -p "$LOGDIR"

case "$ARM" in
    ours)    CKPT=$ROOT/log/rebot_checkpoints_S_geomcat_film/best.pth
             FLAGS=(--film_modality normal --geom_concat --time_cond film) ;;
    wo_film) CKPT=$ROOT/log/rebot_checkpoints_S_geomcat_none/best.pth
             FLAGS=(--film_modality normal --geom_concat --time_cond none) ;;
    wo_cat)  CKPT=$ROOT/log/rebot_checkpoints_S_pseudo_mini_tactile_normal_superpoint_superglue_cond-film-normal/best.pth
             FLAGS=(--film_modality normal --time_cond none) ;;
    *) echo "Unknown arm '$ARM'" >&2; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NUM_THREADS="${NUM_THREADS:-7}"
export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"

echo "[job3-refine $ARM] gpu=$GPU_ID coarse=$COARSE_ARM -> $OUT"
"$PY" "$ROOT/paper_experiments/eval_refine_generic.py" \
    --transfer_dir "$TRANSFER" \
    --layout nested \
    --object_ids_file "$SUBSET" \
    --num_pairs 32 \
    --checkpoint "$CKPT" \
    --model_size rebot_S \
    --video_type tactile_normal \
    --cond_dir "$COND" \
    --film_scale 100 \
    --bottleneck_hw 24 \
    "${FLAGS[@]}" \
    --save_dir "$OUT" \
    --video_save --save_gt --max_videos 3 \
    > "$LOGDIR/refine_${ARM}.log" 2>&1

echo "[job3-refine $ARM] done -> $OUT/metrics.json"
