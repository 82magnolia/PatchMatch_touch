#!/usr/bin/env bash
# Same as run_disk_lightglue.sh (disk_lightglue, modality=curvature, scale=8),
# but with --photometric_refine enabled (--photometric_refine_loss gradient).
#
# A comparison sweep (test_scripts/compare_photometric_refine.py, see
# log/output_photometric_sweep/) found scale=8/modality=curvature to be the
# best-performing (scale, modality) setting overall, and rbf_homography +
# gradient-loss refinement to be the winning combo specifically there --
# even though gradient underperforms the RANSAC baseline on average across
# the full sweep (higher failure rate, worse mean rank). That's the
# motivation for this specific loss choice; it is not a universally safe
# default -- re-check test_scripts/compare_photometric_refine.py before
# reusing --photometric_refine_loss gradient on a different scale/modality.
#
# Usage: bash run_refined_disk_lightglue.sh

set -euo pipefail

# torch/OpenCV/numpy(MKL,OpenBLAS) all default to using every CPU core. That's fine
# for a single run, but running several copies of this script side by side makes them
# oversubscribe the same cores and slow each other down. Cap per-process thread count;
# override e.g. `NUM_THREADS=7 bash run_refined_disk_lightglue.sh` when running 4-way in parallel on 28 cores.
export NUM_THREADS="${NUM_THREADS:-4}"
export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"
export NUMEXPR_NUM_THREADS="$NUM_THREADS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SESSIONS_BASE="$PROJECT_ROOT/log/real_data_gt_retrieval"
OUT_BASE="$PROJECT_ROOT/log/transfer_pipeline_real_data_gt_retrieval_refined_disk_lightglue"
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
        --transfer_matcher   disk_lightglue \
        --video_type         shadow \
        --photometric_refine \
        --photometric_refine_loss gradient \
        --skip_refine \
        --skip_viz \
        --save_match_figures \
        --save_dir           "$save_dir"

    echo "  → done  ($done_count/$total sessions transferred)"
done

echo "Done. $done_count sessions processed."
