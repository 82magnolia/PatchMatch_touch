#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${RQT_PYTHON:-}" ]]; then
    PYTHON_COMMAND=("$RQT_PYTHON")
elif command -v conda >/dev/null 2>&1; then
    PYTHON_COMMAND=(conda run -n RandomQuiltingTactile python)
else
    PYTHON_COMMAND=(python)
fi

exec "${PYTHON_COMMAND[@]}" "$BASELINE_ROOT/run_baseline.py" \
    --scale 100 \
    --video_type shadow \
    --retrieval_mode sim_gt_retrieval \
    "$@"
