#!/usr/bin/env bash
# Backfills height/color/normal/curvature static outputs at scales 1,2,4,8
# for every session in log/real_data_gt_retrieval, via process_single_shot.py.
# Usage: bash run_backfill_scales.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSIONS_BASE="$PROJECT_ROOT/log/real_data_gt_retrieval"

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
        --render_scale 1 2 4 8
done

echo "Done. $done_count sessions processed."
