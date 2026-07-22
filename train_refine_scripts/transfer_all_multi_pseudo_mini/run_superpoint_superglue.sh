#!/usr/bin/env bash
# Runs main_retrieval_transfer_feat_match.py (gelsight_pseudo_mini 240x320 data)
# sequentially over all objects, on this machine's single GPU.
# Usage: bash run_superpoint_superglue.sh

set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# main_retrieval_transfer_feat_match.py has no PatchMatch/CUDA (pycuda) dependency at all
# (the selected --matcher backend is the entire correspondence mechanism), so unlike
# transfer_all_multi_240x320's _run.sh, no CUDA-11.8 nvcc PATH override is needed here.

# torch/OpenCV/numpy(MKL,OpenBLAS) all default to using every CPU core. That's fine
# for a single run, but running several copies of this script side by side makes them
# oversubscribe the same cores and slow each other down. Cap per-process thread count;
# override e.g. `NUM_THREADS=7 bash run.sh` when running 4-way in parallel on 28 cores.
export NUM_THREADS="${NUM_THREADS:-4}"
export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"
export NUMEXPR_NUM_THREADS="$NUM_THREADS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REF_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_pseudo_mini"
QUERY_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_query_pseudo_mini"
RETRIEVAL_BASE="$PROJECT_ROOT/log/touch_retrieval"
OUT_BASE="$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini_superpoint_superglue"
TRANSFER_SCRIPT="$PROJECT_ROOT/main_retrieval_transfer_feat_match.py"
# Same modality/scale settings as run.sh (loftr sweep), but using the
# superpoint_superglue feature matcher backend (image-matching-webui) instead of loftr.

TOUCHES_PER_OBJ=8

# Count eligible objects
total=0
for ref_dir in "$REF_BASE"/*/; do
    idx=$(basename "$ref_dir")
    if [ ! -d "$QUERY_BASE/$idx" ] || [ ! -f "$RETRIEVAL_BASE/$idx/results.pkl" ]; then
        continue
    fi
    total=$((total + 1))
done
total_touch=$((total * TOUCHES_PER_OBJ))
echo "Objects to process: $total  |  Touch locations: $total_touch"

done_count=0
done_touch=0
for ref_dir in "$REF_BASE"/*/; do
    idx=$(basename "$ref_dir")
    query_dir="$QUERY_BASE/$idx"
    retrieval_pkl="$RETRIEVAL_BASE/$idx/results.pkl"

    if [ ! -d "$query_dir" ] || [ ! -f "$retrieval_pkl" ]; then
        continue
    fi

    save_dir="$OUT_BASE/$idx"
    mkdir -p "$save_dir"

    done_count=$((done_count + 1))
    echo "[obj ${idx}] ($done_count/$total) | touch locations: $done_touch/$total_touch"

    python "$TRANSFER_SCRIPT" \
        --query_dir      "$query_dir" \
        --ref_dir        "$ref_dir" \
        --retrieval_pkl  "$retrieval_pkl" \
        --modality       curvature \
        --video_type     shadow \
        --video_scale          100. \
        --match_scale          25. \
        --match_scale_convention obj_scale_factor \
        --matcher        superpoint_superglue \
        --offset_matcher       disk_lightglue \
        --offset_method         median \
        --save_dir       "$save_dir" \
        --eval

    done_touch=$((done_touch + TOUCHES_PER_OBJ))
    echo "  → done  ($done_touch/$total_touch touch locations transferred)"
done

echo "Done. $done_count objects, $done_touch touch locations transferred."
