#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OBJECT_ID="${TACTILE_NORMAL_OBJECT_ID:-1}"
SIM_REF_ROOT="${TACTILE_NORMAL_SIM_REF_ROOT:-$REPO_DIR/Taxim/results/gen_contact_full_tactile_normal_pseudo_mini}"
SIM_QUERY_ROOT="${TACTILE_NORMAL_SIM_QUERY_ROOT:-$REPO_DIR/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini}"

exec "$SCRIPT_DIR/run_sim.sh" \
    --ref_dir "$SIM_REF_ROOT/$OBJECT_ID" \
    --query_dir "$SIM_QUERY_ROOT/$OBJECT_ID" \
    --save_dir "$REPO_DIR/log/baselines/tactile_normal/sim/object_$OBJECT_ID/taxim" \
    --video_type tactile_normal \
    "$@"
