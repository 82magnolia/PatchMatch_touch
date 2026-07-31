#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OBJECT_ID="${TACTILE_NORMAL_OBJECT_ID:-1}"
REAL_ROOT="${TACTILE_NORMAL_REAL_ROOT:-$REPO_DIR/log/real_data_gt_retrieval}"
BACKGROUND="${TARF_TACTILE_NORMAL_BACKGROUND:-$REPO_DIR/baselines/TaRF/img2touch/touch_bg/gelsight_pseudo_background.jpg}"

exec "$SCRIPT_DIR/run_real.sh" \
    --ref_dir "$REAL_ROOT/$OBJECT_ID" \
    --query_dir "$REAL_ROOT/$OBJECT_ID" \
    --save_dir "$REPO_DIR/log/baselines/tactile_normal/real/object_$OBJECT_ID/tarf" \
    --background "$BACKGROUND" \
    --video_type tactile_normal \
    "$@"
