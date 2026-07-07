#!/usr/bin/env bash
# Runs main_retrieval_transfer_feat_match.py (gelsight_pseudo_mini 240x320 data)
# sequentially over all objects, on this machine's single GPU.
# Usage: bash run.sh

set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# main_retrieval_transfer_feat_match.py has no PatchMatch/CUDA (pycuda) dependency at all
# (DINOv3 is the entire correspondence mechanism), so unlike transfer_all_multi_240x320's
# _run.sh, no CUDA-11.8 nvcc PATH override is needed here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REF_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_pseudo_mini"
QUERY_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_query_pseudo_mini"
RETRIEVAL_BASE="$PROJECT_ROOT/log/touch_retrieval"
OUT_BASE="$PROJECT_ROOT/log/transfer_feat_match_pseudo_mini"
TRANSFER_SCRIPT="$PROJECT_ROOT/main_retrieval_transfer_feat_match.py"
DINOV3_WEIGHTS="$PROJECT_ROOT/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

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
        --modality       raw_normal \
        --video_type     shadow \
        --scale          100. \
        --dinov3_match_scale 100. \
        --dinov3_match_scale_convention obj_scale_factor \
        --dinov3_weights "$DINOV3_WEIGHTS" \
        --save_dir       "$save_dir" \
        --eval

    done_touch=$((done_touch + TOUCHES_PER_OBJ))
    echo "  → done  ($done_touch/$total_touch touch locations transferred)"
done

echo "Done. $done_count objects, $done_touch touch locations transferred."
