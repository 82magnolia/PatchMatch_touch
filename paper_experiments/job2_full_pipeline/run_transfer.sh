#!/usr/bin/env bash
# Job 2: full-pipeline benchmark. For each object, run the complete pipeline
# (DINOv3 retrieval over the reference touches -> SuperPoint+SuperGlue coarse
# alignment) on the held-out query touches built by build_splits.py.
#
# Stage 3 (network refinement) is deliberately skipped here: transfer_pipeline.py
# has no flags for the query-conditioning the paper's model uses (--geom_concat /
# --film_modality / --time_cond), so refinement runs afterwards through
# paper_experiments/eval_refine_generic.py against {save_dir}/{obj}/transfer.
#
# Usage: bash run_transfer.sh <worker_id> <gpu_id> <num_workers>
set -o pipefail

WORKER_ID="${1:?Usage: $0 <worker_id> <gpu_id> <num_workers>}"
GPU_ID="${2:?gpu id}"
NUM_WORKERS="${3:-5}"
# Alignment modality for retrieval + feature matching. The method's default is
# surface normals; pass "curvature" to reproduce the earlier curvature run.
MODALITY="${MODALITY:-normal}"

ROOT=/data1/junhokim/Projects/PatchMatch_touch
PY=/home/junhokim/miniconda3/envs/pm_touch/bin/python
BENCH=$ROOT/log/paper_job2_bench
if [ "$MODALITY" = "normal" ]; then OUT=$ROOT/log/paper_job2_pipeline_normal
else OUT=$ROOT/log/paper_job2_pipeline; fi
DINO=$ROOT/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NUM_THREADS="${NUM_THREADS:-6}"
export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"
export NUMEXPR_NUM_THREADS="$NUM_THREADS"

mkdir -p "$OUT"
pos=0
for obj_dir in "$BENCH"/*/; do
    OBJ=$(basename "$obj_dir")
    if [ $((pos % NUM_WORKERS)) -ne "$WORKER_ID" ]; then pos=$((pos + 1)); continue; fi
    pos=$((pos + 1))

    SAVE="$OUT/$OBJ"
    if [ -f "$SAVE/transfer/metrics.pkl" ]; then
        echo "[job2 w$WORKER_ID] obj $OBJ done, skipping"
        continue
    fi
    mkdir -p "$SAVE"
    echo "[job2 w$WORKER_ID gpu$GPU_ID] $(date +%H:%M:%S) obj $OBJ"

    "$PY" "$ROOT/transfer_pipeline.py" \
        --ref_dir   "$obj_dir/ref" \
        --query_dir "$obj_dir/query" \
        --save_dir  "$SAVE" \
        --scale 100 \
        --match_scale 25 --match_scale_convention obj_scale_factor \
        --retrieval_mode dinov3 --retrieval_modality "$MODALITY" \
        --dino_weights "$DINO" \
        --transfer_backend dinov3_feat_match \
        --transfer_modality "$MODALITY" \
        --transfer_matcher superpoint_superglue \
        --transfer_offset_matcher superpoint_superglue \
        --transfer_offset_method median \
        --video_type tactile_normal \
        --skip_refine --skip_viz \
        > "$SAVE/pipeline.log" 2>&1 \
        || echo "[job2 w$WORKER_ID] obj $OBJ FAILED (see $SAVE/pipeline.log)"
done

echo "[job2 w$WORKER_ID] done."
