#!/usr/bin/env bash
# Job 1 coarse transfer, rebuilt with the method's DEFAULT alignment modality.
#
# The Job 1 table originally reused the pre-existing transfer dir
# log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue, which
# train_refine_scripts/transfer_all_multi_pseudo_mini_tactile_normal/_run.sh
# built with `--modality curvature`. The paper's method is surface normals at 4x
# the sensor footprint, so this rebuilds the same transfer with `--modality
# normal` over just the 50 eval objects (951-1000).
#
# Retrieval is unchanged and remains ground-truth: log/touch_retrieval/{obj}/
# results.pkl comes from retrieve_touch_all.sh in tsv mode against
# log/identity_mapping.tsv, i.e. query index i is paired with reference index i.
#
# Usage: bash run_transfer_normal.sh <worker_id> <gpu_id> <num_workers>
set -o pipefail

WORKER_ID="${1:?Usage: $0 <worker_id> <gpu_id> <num_workers>}"
GPU_ID="${2:?gpu id}"
NUM_WORKERS="${3:-5}"

ROOT=/data1/junhokim/Projects/PatchMatch_touch
PY=/home/junhokim/miniconda3/envs/pm_touch/bin/python
REF_BASE=$ROOT/Taxim/results/gen_contact_full_tactile_normal_pseudo_mini
QUERY_BASE=$ROOT/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini
RETRIEVAL_BASE=$ROOT/log/touch_retrieval
OUT=$ROOT/log/paper_job1_transfer_normal

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NUM_THREADS="${NUM_THREADS:-6}"
export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"

mkdir -p "$OUT"
pos=0
for OBJ in $(seq 951 1000); do
    if [ $((pos % NUM_WORKERS)) -ne "$WORKER_ID" ]; then pos=$((pos + 1)); continue; fi
    pos=$((pos + 1))

    SAVE="$OUT/$OBJ"
    if [ -f "$SAVE/metrics.pkl" ]; then
        echo "[j1n w$WORKER_ID] obj $OBJ done, skipping"; continue
    fi
    mkdir -p "$SAVE"
    echo "[j1n w$WORKER_ID gpu$GPU_ID] $(date +%H:%M:%S) obj $OBJ"

    "$PY" "$ROOT/main_retrieval_transfer_feat_match.py" \
        --query_dir     "$QUERY_BASE/$OBJ" \
        --ref_dir       "$REF_BASE/$OBJ" \
        --retrieval_pkl "$RETRIEVAL_BASE/$OBJ/results.pkl" \
        --modality      normal \
        --video_type    tactile_normal \
        --video_scale   100. \
        --match_scale   25. \
        --match_scale_convention obj_scale_factor \
        --matcher        superpoint_superglue \
        --offset_matcher superpoint_superglue \
        --offset_method  median \
        --save_dir      "$SAVE" \
        --no_nnf_figures \
        --eval \
        > "$SAVE/transfer.log" 2>&1 \
        || echo "[j1n w$WORKER_ID] obj $OBJ FAILED (see $SAVE/transfer.log)"
done

echo "[j1n w$WORKER_ID] done."
