#!/usr/bin/env bash
# Worker: runs main_retrieval_transfer_accel.py for 1/8 of all objects.
# Usage: bash _run.sh <GPU_ID (0-7)>

set -euo pipefail

GPU_ID="${1:?Usage: $0 <GPU_ID (0-7)>}"
NUM_GPUS=8
WORKER_ID=$GPU_ID   # GPU index directly maps to worker slot (0-7)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID
# NVCC 12.4 (conda) generates cubins that require a CUDA 12.x driver, but the
# installed driver only supports CUDA 11.4.  Use CUDA 11.8 nvcc instead — within
# CUDA 11.x all minor versions are ABI-compatible with each other.
export PATH="/usr/local/cuda-11.8/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REF_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full"
QUERY_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_query"
RETRIEVAL_BASE="$PROJECT_ROOT/log/touch_retrieval"
OUT_BASE="$PROJECT_ROOT/log/transfer"
TRANSFER_SCRIPT="$PROJECT_ROOT/main_retrieval_transfer_accel.py"

TOUCHES_PER_OBJ=8

# Count eligible objects assigned to this GPU
total=0
pos=0
for ref_dir in "$REF_BASE"/*/; do
    idx=$(basename "$ref_dir")
    if [ ! -d "$QUERY_BASE/$idx" ] || [ ! -f "$RETRIEVAL_BASE/$idx/results.pkl" ]; then
        continue
    fi
    if [ $((pos % NUM_GPUS)) -eq "$WORKER_ID" ]; then
        total=$((total + 1))
    fi
    pos=$((pos + 1))
done
total_touch=$((total * TOUCHES_PER_OBJ))
echo "[GPU $GPU_ID] Objects to process: $total  |  Touch locations: $total_touch"

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

    if [ $((pos % NUM_GPUS)) -ne "$WORKER_ID" ]; then
        pos=$((pos + 1))
        continue
    fi
    pos=$((pos + 1))

    save_dir="$OUT_BASE/$idx"
    mkdir -p "$save_dir"

    done_count=$((done_count + 1))
    echo "[obj ${idx}] ($done_count/$total) [GPU $GPU_ID] | touch locations: $done_touch/$total_touch"

    python "$TRANSFER_SCRIPT" \
        --query_dir      "$query_dir" \
        --ref_dir        "$ref_dir" \
        --retrieval_pkl  "$retrieval_pkl" \
        --modality       raw_normal \
        --scale          100 \
        --video_type     shadow \
        --save_dir       "$save_dir" \
        --em \
        --use_ref_contact_mask \
        --use_ref_static_mask \
        --em_iters       10 \
        --use_mask \
        --eval \
        --use_accel \
        --em_iters_subseq 1 \
        --downsample_res 4 \
        --use_downsample_em \
        --use_keyframe \
        --no_nnf_figures

    done_touch=$((done_touch + TOUCHES_PER_OBJ))
    echo "  → done  ($done_touch/$total_touch touch locations transferred) [GPU $GPU_ID]"
done

echo "[GPU $GPU_ID] Done. $done_count objects, $done_touch touch locations transferred."
