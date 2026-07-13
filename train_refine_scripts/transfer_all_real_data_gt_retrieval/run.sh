#!/usr/bin/env bash
# Runs transfer_pipeline.py (retrieval + DINOv3 feature-match transfer) over
# every session in log/real_data_gt_retrieval, on this machine's single GPU.
#
# Uses --retrieval_mode real_gt_retrieval (odd-indexed touches = query,
# matched to the even-indexed touch directly below, e.g. 1->0, 3->2 --
# --ref_dir and --query_dir are the same session directory) and
# --transfer_backend dinov3_feat_match (main_retrieval_transfer_feat_match.py,
# no PatchMatch/CUDA dependency).
#
# modality=curvature + dinov3_vith16plus + scale=8 (direct, no
# --dinov3_match_scale): best config found by the combined synthetic/real-data
# tuning sweep (see report) -- the only one of the 4 backfilled real-data
# modalities/scales that's good on BOTH datasets, not just real data alone.
#
# Usage: bash run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SESSIONS_BASE="$PROJECT_ROOT/log/real_data_gt_retrieval"
OUT_BASE="$PROJECT_ROOT/log/transfer_pipeline_real_data_gt_retrieval"
PIPELINE_SCRIPT="$PROJECT_ROOT/transfer_pipeline.py"
DINOV3_WEIGHTS="$PROJECT_ROOT/dinov3/pretrained/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"

# Count eligible sessions
total=0
for session_dir in "$SESSIONS_BASE"/*/; do
    idx=$(basename "$session_dir")
    total=$((total + 1))
done
echo "Sessions to process: $total"

done_count=0
for session_dir in "$SESSIONS_BASE"/*/; do
    idx=$(basename "$session_dir")
    save_dir="$OUT_BASE/$idx"

    done_count=$((done_count + 1))
    echo "[session ${idx}] ($done_count/$total)"

    python "$PIPELINE_SCRIPT" \
        --ref_dir            "$session_dir" \
        --query_dir          "$session_dir" \
        --scale              8 \
        --retrieval_mode     real_gt_retrieval \
        --transfer_backend   dinov3_feat_match \
        --transfer_modality  curvature \
        --video_type         shadow \
        --dinov3_model       dinov3_vith16plus \
        --dinov3_weights     "$DINOV3_WEIGHTS" \
        --skip_refine \
        --skip_viz \
        --save_dir           "$save_dir"

    echo "  → done  ($done_count/$total sessions transferred)"
done

echo "Done. $done_count sessions processed."
