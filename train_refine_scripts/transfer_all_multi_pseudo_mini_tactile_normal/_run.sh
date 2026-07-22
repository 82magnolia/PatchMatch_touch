#!/usr/bin/env bash
# Worker: runs main_retrieval_transfer_feat_match.py (gelsight_pseudo_mini
# 240x320 data) for 1/NUM_WORKERS of all objects -- tactile_normal variant of
# transfer_all_multi_pseudo_mini: transfers the surface-normal-encoded video
# domain (--video_type tactile_normal) instead of the shadow/appearance
# domain. Reuses the SAME retrieval results (query/ref indices are
# geometry-derived and shared across modalities -- both video domains are
# rendered from the same picked_points_query.ply/picked_points_fps.ply
# contact locations, see gen_contact_query_tactile_normal_pseudo_mini/_run.sh
# and gen_contact_ref_tactile_normal_pseudo_mini/_run.sh).
# Usage: bash _run.sh <WORKER_ID> <GPU_ID> [MATCHER] [NUM_WORKERS]
#   WORKER_ID fixes which 1/NUM_WORKERS of objects this worker handles (always
#   matching run_gpu0.sh..run_gpu5.sh when NUM_WORKERS=6, the default); GPU_ID
#   is the physical CUDA device to run on and can be any value, independent of
#   WORKER_ID. MATCHER (default disk_lightglue) selects the feature-matcher
#   backend for both the linear and offset stages (mirrors
#   transfer_all_multi_pseudo_mini/run_<matcher>.sh); output goes to
#   log/transfer_feat_match_pseudo_mini_tactile_normal[_<matcher>] (no suffix
#   for disk_lightglue, the default/canonical matcher for this pipeline).

set -euo pipefail

WORKER_ID="${1:?Usage: $0 <WORKER_ID> <GPU_ID> [MATCHER] [NUM_WORKERS]}"
GPU_ID="${2:?Usage: $0 <WORKER_ID> <GPU_ID> [MATCHER] [NUM_WORKERS]}"
MATCHER="${3:-disk_lightglue}"
NUM_WORKERS="${4:-6}"

case "$MATCHER" in
    loftr|disk_lightglue|sift_lightglue|superpoint_lightglue|superpoint_superglue) ;;
    *)
        echo "Unknown matcher '$MATCHER'. Expected one of: loftr, disk_lightglue, sift_lightglue, superpoint_lightglue, superpoint_superglue" >&2
        exit 1
        ;;
esac

if [ "$WORKER_ID" -lt 0 ] || [ "$WORKER_ID" -ge "$NUM_WORKERS" ]; then
    echo "Error: WORKER_ID must be in [0, $((NUM_WORKERS - 1))], got $WORKER_ID" >&2
    exit 1
fi

if [ "$MATCHER" = "disk_lightglue" ]; then
    OUT_SUFFIX=""
else
    OUT_SUFFIX="_${MATCHER}"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID
# main_retrieval_transfer_feat_match.py has no PatchMatch/CUDA (pycuda) dependency at all
# (the selected --matcher backend is the entire correspondence mechanism), so unlike
# transfer_all_multi_240x320's _run.sh, no CUDA-11.8 nvcc PATH override is needed here.

# torch/OpenCV/numpy(MKL,OpenBLAS) all default to using every CPU core; cap per-process
# thread count since NUM_WORKERS copies run side by side.
export NUM_THREADS="${NUM_THREADS:-8}"
export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"
export NUMEXPR_NUM_THREADS="$NUM_THREADS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REF_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_tactile_normal_pseudo_mini"
QUERY_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini"
RETRIEVAL_BASE="$PROJECT_ROOT/log/touch_retrieval"
OUT_BASE="$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal${OUT_SUFFIX}"
TRANSFER_SCRIPT="$PROJECT_ROOT/main_retrieval_transfer_feat_match.py"
# Same decomposed-transfer config as transfer_all_multi_pseudo_mini/run.sh
# (video_scale/match_scale/matcher/offset_method); only --video_type and the
# ref/query/out directories change for the tactile_normal domain.

TOUCHES_PER_OBJ=8

# Count eligible objects assigned to this worker
total=0
pos=0
for ref_dir in "$REF_BASE"/*/; do
    idx=$(basename "$ref_dir")
    if [ ! -d "$QUERY_BASE/$idx" ] || [ ! -f "$RETRIEVAL_BASE/$idx/results.pkl" ]; then
        continue
    fi
    if [ $((pos % NUM_WORKERS)) -eq "$WORKER_ID" ]; then
        total=$((total + 1))
    fi
    pos=$((pos + 1))
done
total_touch=$((total * TOUCHES_PER_OBJ))
echo "[worker $WORKER_ID | GPU $GPU_ID] Objects to process: $total  |  Touch locations: $total_touch"

done_count=0
done_touch=0
pos=0
for ref_dir in "$REF_BASE"/*/; do
    idx=$(basename "$ref_dir")
    query_dir="$QUERY_BASE/$idx"
    retrieval_pkl="$RETRIEVAL_BASE/$idx/results.pkl"

    if [ ! -d "$query_dir" ] || [ ! -f "$retrieval_pkl" ]; then
        continue
    fi

    if [ $((pos % NUM_WORKERS)) -ne "$WORKER_ID" ]; then
        pos=$((pos + 1))
        continue
    fi
    pos=$((pos + 1))

    save_dir="$OUT_BASE/$idx"
    mkdir -p "$save_dir"

    done_count=$((done_count + 1))
    echo "[worker $WORKER_ID | GPU $GPU_ID] [obj ${idx}] ($done_count/$total) | touch locations: $done_touch/$total_touch"

    python "$TRANSFER_SCRIPT" \
        --query_dir      "$query_dir" \
        --ref_dir        "$ref_dir" \
        --retrieval_pkl  "$retrieval_pkl" \
        --modality       curvature \
        --video_type     tactile_normal \
        --video_scale          100. \
        --match_scale          25. \
        --match_scale_convention obj_scale_factor \
        --matcher        "$MATCHER" \
        --offset_matcher "$MATCHER" \
        --offset_method         median \
        --save_dir       "$save_dir" \
        --no_nnf_figures \
        --eval

    done_touch=$((done_touch + TOUCHES_PER_OBJ))
done

echo "[worker $WORKER_ID | GPU $GPU_ID] Done. $done_count objects, $done_touch touch locations transferred."
