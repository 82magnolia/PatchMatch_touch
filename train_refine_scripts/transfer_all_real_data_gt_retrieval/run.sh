#!/usr/bin/env bash
# Runs transfer_pipeline.py (retrieval + feature-match transfer) over
# every session in log/real_data_gt_retrieval, on this machine's single GPU.
#
# Uses --retrieval_mode real_gt_retrieval (odd-indexed touches = query,
# matched to the even-indexed touch directly below, e.g. 1->0, 3->2 --
# --ref_dir and --query_dir are the same session directory) and
# --transfer_backend dinov3_feat_match (main_retrieval_transfer_feat_match.py,
# no PatchMatch/CUDA dependency -- despite the backend's name, the actual
# correspondence matcher is chosen via --transfer_matcher, see below).
#
# modality=curvature + matcher=loftr + scale=8 (direct, no --dinov3_match_scale):
# a follow-up sweep that scores the warped static image directly (instead of
# via reconstructed video, which masked this) found DINOv3 mostly outputs a
# near-identity transform on real data, while loftr reaches 0/30 fallback and
# far higher warp accuracy (SSIM 0.985 vs. DINOv3's 0.902) at this dataset's
# max render scale -- see report.
#
# Usage: bash run.sh

set -euo pipefail

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

SESSIONS_BASE="$PROJECT_ROOT/log/real_data_gt_retrieval"
OUT_BASE="$PROJECT_ROOT/log/transfer_pipeline_real_data_gt_retrieval"
PIPELINE_SCRIPT="$PROJECT_ROOT/transfer_pipeline.py"

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
        --transfer_matcher   loftr \
        --video_type         shadow \
        --skip_refine \
        --skip_viz \
        --save_match_figures \
        --save_dir           "$save_dir"

    echo "  → done  ($done_count/$total sessions transferred)"
done

echo "Done. $done_count sessions processed."
