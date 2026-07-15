#!/usr/bin/env bash
# Backfills unmasked static outputs (color/normals/height/curvature, no SAM
# clipping) for every session in log/real_data_gt_retrieval, via
# process_single_shot.py --no_mask. Outputs are written to a sibling folder
# so the original (masked) session data is left untouched.
# Usage: bash run_backfill_unmasked.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSIONS_BASE="$PROJECT_ROOT/log/real_data_gt_retrieval"
OUTPUT_BASE="$PROJECT_ROOT/log/real_data_gt_retrieval_unmasked"

cd "$SCRIPT_DIR"

total=0
for session_dir in "$SESSIONS_BASE"/*/; do
    idx=$(basename "$session_dir")
    total=$((total + 1))
done
echo "Sessions to process: $total"

done_count=0
for session_dir in "$SESSIONS_BASE"/*/; do
    idx=$(basename "$session_dir")
    done_count=$((done_count + 1))
    echo "[session ${idx}] ($done_count/$total)"
    python3 process_single_shot.py \
        --session_dir "$session_dir" \
        --output_dir "$OUTPUT_BASE/$idx" \
        --render_scale 1 2 4 8 \
        --no_mask
done

echo "Done. $done_count sessions processed. Output: $OUTPUT_BASE"
