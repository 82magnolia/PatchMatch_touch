#!/usr/bin/env bash
# Worker: runs transfer_pipeline.py (dinov3_feat_match backend, gelsight_pseudo_mini
# 240x320 data) for 1/8 of all objects.
# Usage: bash _run.sh <GPU_ID (0-7)>

set -euo pipefail

GPU_ID="${1:?Usage: $0 <GPU_ID (0-7)>}"
NUM_GPUS=8
WORKER_ID=$GPU_ID   # GPU index directly maps to worker slot (0-7)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID
# transfer_backend=dinov3_feat_match does not use the PatchMatch CUDA kernel (pycuda/nvcc),
# so unlike transfer_all_multi_240x320's _run.sh, no CUDA-11.8 nvcc PATH override is needed here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REF_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_pseudo_mini"
QUERY_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_query_pseudo_mini"
OUT_BASE="$PROJECT_ROOT/log/pipeline_sim_dino_pseudo_mini"
TRANSFER_SCRIPT="$PROJECT_ROOT/transfer_pipeline.py"
DINOV3_WEIGHTS="$PROJECT_ROOT/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
REBOT_CHECKPOINT="$PROJECT_ROOT/log/rebot_checkpoints_S_240x320_residual/best.pth"

cd "$PROJECT_ROOT"

# Count eligible objects assigned to this GPU
total=0
pos=0
for ref_dir in "$REF_BASE"/*/; do
    idx=$(basename "$ref_dir")
    if [ ! -d "$QUERY_BASE/$idx" ]; then
        continue
    fi
    if [ $((pos % NUM_GPUS)) -eq "$WORKER_ID" ]; then
        total=$((total + 1))
    fi
    pos=$((pos + 1))
done
echo "[GPU $GPU_ID] Objects to process: $total"

done_count=0
pos=0
for ref_dir in "$REF_BASE"/*/; do
    idx=$(basename "$ref_dir")
    query_dir="$QUERY_BASE/$idx"

    if [ ! -d "$query_dir" ]; then
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
    echo "[obj ${idx}] ($done_count/$total) [GPU $GPU_ID]"

    python "$TRANSFER_SCRIPT" \
        --ref_dir "$ref_dir" \
        --query_dir "$query_dir" \
        --scale 1 --retrieval_mode sim_gt_retrieval \
        --transfer_backend dinov3_feat_match \
        --dinov3_weights "$DINOV3_WEIGHTS" \
        --save_dir "$save_dir" \
        --checkpoint "$REBOT_CHECKPOINT" --save_nnf_figures --scale 100. --dinov3_match_scale 100. \
        --dinov3_match_scale_convention obj_scale_factor --residual

    echo "  → done  ($done_count/$total objects) [GPU $GPU_ID]"
done

echo "[GPU $GPU_ID] Done. $done_count objects transferred."
