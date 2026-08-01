#!/usr/bin/env bash
# Job 2 baselines on the full-pipeline benchmark: same query/reference split and
# same DINOv3 retrieval as our method, so only the prediction stage differs.
#
# The full-pipeline benchmark stores every touch of an object in ONE contact-point
# file (objfolder_pts_real_eval/{obj}/picked_points.ply), indexed by the original
# touch index -- so unlike Job 1, the INR baseline can use the same file for its
# training touches and its query touches.
#
# tarf uses the img2touch latent diffusion checkpoint with best-of-8 DDIM
# sampling reranked by the two encoder checkpoints; it predicts ONE image per
# query, tiled to the reference video length.
#
# Usage: bash run_baselines.sh <quilting|inr> <worker_id> <gpu_id> <num_workers>
set -o pipefail

METHOD="${1:?Usage: $0 <quilting|inr|tarf> <worker_id> <gpu_id> <num_workers>}"
WORKER_ID="${2:?worker id}"
GPU_ID="${3:?gpu id}"
NUM_WORKERS="${4:-5}"

ROOT=/data1/junhokim/Projects/PatchMatch_touch
PY=/home/junhokim/miniconda3/envs/pm_touch/bin/python
export RQT_PYTHON="$PY"
export OBJECTFOLDER_PYTHON="$PY"
export TARF_PYTHON=/home/junhokim/miniconda3/envs/TaRF/bin/python
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NUM_THREADS="${NUM_THREADS:-6}"
export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"

BENCH=$ROOT/log/paper_job2_bench
PTS=$ROOT/Taxim/results/objfolder_pts_real_eval
DINO=$ROOT/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
OUT_BASE=$ROOT/log/paper_job2_baselines/$METHOD
mkdir -p "$OUT_BASE"

pos=0
for obj_dir in "$BENCH"/*/; do
    OBJ=$(basename "$obj_dir")
    if [ $((pos % NUM_WORKERS)) -ne "$WORKER_ID" ]; then pos=$((pos + 1)); continue; fi
    pos=$((pos + 1))

    SAVE="$OUT_BASE/$OBJ"
    if [ -f "$SAVE/transfer/metrics.pkl" ]; then
        echo "[$METHOD w$WORKER_ID] obj $OBJ done, skipping"; continue
    fi
    mkdir -p "$SAVE"
    echo "[$METHOD w$WORKER_ID gpu$GPU_ID] $(date +%H:%M:%S) obj $OBJ"

    COMMON=(--ref_dir "$obj_dir/ref" --query_dir "$obj_dir/query" --save_dir "$SAVE"
            --scale 100 --video_type tactile_normal
            --retrieval_mode dinov3 --retrieval_modality normal --dino_weights "$DINO")

    if [ "$METHOD" = "tarf" ]; then
        "$TARF_PYTHON" "$ROOT/baselines/TaRF/run_baseline.py" \
            "${COMMON[@]}" \
            --diffusion_ckpt       "$ROOT/log/tarf_pretrained.ckpt" \
            --ranking_rgb_enc_ckpt "$ROOT/log/reranking_rgb_enc.ckpt" \
            --ranking_tac_enc_ckpt "$ROOT/log/reranking_tac_enc.ckpt" \
            > "$SAVE/run.log" 2>&1 \
            || echo "[$METHOD w$WORKER_ID] obj $OBJ FAILED (see $SAVE/run.log)"
    elif [ "$METHOD" = "quilting" ]; then
        "$PY" "$ROOT/baselines/RandomQuiltingTactile/run_baseline.py" \
            "${COMMON[@]}" --object_id "$OBJ" \
            > "$SAVE/run.log" 2>&1 \
            || echo "[$METHOD w$WORKER_ID] obj $OBJ FAILED (see $SAVE/run.log)"
    else
        "$PY" "$ROOT/baselines/objectfolder_inr/run_baseline.py" \
            "${COMMON[@]}" --device cuda \
            --checkpoint "$SAVE/touchnet.pth" \
            --contact_points "$PTS/$OBJ/picked_points.npy" \
            --train_only --train_if_missing \
            > "$SAVE/train.log" 2>&1 \
            || { echo "[$METHOD w$WORKER_ID] obj $OBJ TRAIN FAILED"; continue; }
        "$PY" "$ROOT/baselines/objectfolder_inr/run_baseline.py" \
            "${COMMON[@]}" --device cuda \
            --checkpoint "$SAVE/touchnet.pth" \
            --contact_points "$PTS/$OBJ/picked_points.npy" \
            > "$SAVE/run.log" 2>&1 \
            || echo "[$METHOD w$WORKER_ID] obj $OBJ INFER FAILED (see $SAVE/run.log)"
    fi
done

echo "[$METHOD w$WORKER_ID] done."
