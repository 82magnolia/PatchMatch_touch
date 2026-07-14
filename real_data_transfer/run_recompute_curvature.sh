#!/usr/bin/env bash
# Recomputes curvature (and every other per-touch output) for every session in
# log/real_data_gt_retrieval, via process_single_shot.py, now that
# height2laplacian's boundary-artifact fix (mask-aware, separately-normalized
# composite of interior + boundary-ring curvature -- see _gelsight_processing.py)
# has landed. process_single_shot.py has no "curvature-only" mode: it re-derives
# color/normal/height/curvature/videos together from each session's cached
# raw inputs (object_cache_N.npz, session_poses.npz, ...), so this reprocesses
# everything, not just the curvature files.
#
# Writes to a NEW output tree (OUT_BASE below) rather than overwriting
# log/real_data_gt_retrieval in place -- process_single_shot.py reads all its
# inputs from --session_dir and writes only to --output_dir, so a fresh
# directory works with no dependency on pre-existing content there.
#
# Same render_scale coverage (1/2/4/8) and default args as the original
# run_backfill_scales.sh, so segmentation/touch detection reproduces
# identically -- only the curvature encoding itself changes.
#
# Usage: bash run_recompute_curvature.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSIONS_BASE="$PROJECT_ROOT/log/real_data_gt_retrieval"
OUT_BASE="$PROJECT_ROOT/log/real_data_gt_retrieval_curvature_fix"

cd "$SCRIPT_DIR"

total=0
for session_dir in "$SESSIONS_BASE"/*/; do
    total=$((total + 1))
done
echo "Sessions to process: $total"
echo "Output base: $OUT_BASE"

done_count=0
for session_dir in "$SESSIONS_BASE"/*/; do
    idx=$(basename "$session_dir")
    out_dir="$OUT_BASE/$idx"
    done_count=$((done_count + 1))
    echo "[session ${idx}] ($done_count/$total)"
    python3 process_single_shot.py \
        --session_dir "$session_dir" \
        --output_dir "$out_dir" \
        --render_scale 1 2 4 8
done

echo "Done. $done_count sessions processed. Output under: $OUT_BASE"
