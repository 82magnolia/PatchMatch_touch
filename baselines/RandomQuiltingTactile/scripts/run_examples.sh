#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$BASELINE_ROOT/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/log/baselines/random_quilting/examples}"
SIM_REF_ROOT="${RQT_SIM_REF_ROOT:-$PROJECT_ROOT/Taxim/results/gen_contact_full_pseudo_mini}"
SIM_QUERY_ROOT="${RQT_SIM_QUERY_ROOT:-$PROJECT_ROOT/Taxim/results/gen_contact_full_query_pseudo_mini}"

# Override with, for example:
#   RQT_EXAMPLES="2:1 25:4" bash scripts/run_examples.sh
EXAMPLES="${RQT_EXAMPLES:-1:0 1:3 10:0 100:7}"

for example in $EXAMPLES; do
    object_id="${example%%:*}"
    query_idx="${example##*:}"
    ref_dir="$SIM_REF_ROOT/$object_id"
    query_dir="$SIM_QUERY_ROOT/$object_id"
    save_dir="$OUTPUT_ROOT/object_${object_id}_query_${query_idx}"

    echo "[RandomQuiltingTactile] object=$object_id query=$query_idx"
    bash "$SCRIPT_DIR/run_sim.sh" \
        --ref_dir "$ref_dir" \
        --query_dir "$query_dir" \
        --save_dir "$save_dir" \
        --query_indices "$query_idx" \
        --pipeline_mode fallback \
        --quilt_max_candidates 512 \
        --debug_images \
        --skip_eval
done

echo "Examples written under: $OUTPUT_ROOT"
