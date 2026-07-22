#!/usr/bin/env bash
# Runs transfer_pipeline.py (retrieval + feature-match transfer) over
# every session in log/real_data_gt_retrieval, on this machine's single GPU.
#
# Same modality/scale settings as run.sh (loftr sweep), but using the
# disk_lightglue feature matcher backend (image-matching-webui) instead of loftr.
#
# Usage: bash run_disk_lightglue.sh

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
OUT_BASE="$PROJECT_ROOT/log/transfer_pipeline_real_data_gt_retrieval_disk_lightglue"
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
        --scale              1 \
        --match_scale              8 \
        --match_scale_convention render_scale \
        --retrieval_mode     real_gt_retrieval \
        --transfer_backend   dinov3_feat_match \
        --transfer_modality  curvature \
        --transfer_matcher   disk_lightglue \
        --transfer_offset_matcher  disk_lightglue \
        --transfer_offset_method  median \
        --video_type         shadow \
        --skip_refine \
        --skip_viz \
        --save_match_figures \
        --save_dir           "$save_dir"

    echo "  → done  ($done_count/$total sessions transferred)"
done

echo "Done. $done_count sessions processed."
