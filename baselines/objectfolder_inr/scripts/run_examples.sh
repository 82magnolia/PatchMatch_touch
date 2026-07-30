#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$BASELINE_ROOT/../.." && pwd)"
OUTPUT_ROOT="${OBJECTFOLDER_EXAMPLE_ROOT:-$PROJECT_ROOT/log/baselines/objectfolder_inr/examples}"
SIM_REF_ROOT="${OBJECTFOLDER_SIM_REF_ROOT:-$PROJECT_ROOT/Taxim/results/gen_contact_full_pseudo_mini}"
SIM_QUERY_ROOT="${OBJECTFOLDER_SIM_QUERY_ROOT:-$PROJECT_ROOT/Taxim/results/gen_contact_full_query_pseudo_mini}"
EXAMPLES="${OBJECTFOLDER_EXAMPLES:-1:0 10:3 100:7}"

mkdir -p "$OUTPUT_ROOT/checkpoints"

for spec in $EXAMPLES; do
    object_id="${spec%%:*}"
    query_idx="${spec##*:}"
    ref_dir="$SIM_REF_ROOT/$object_id"
    query_dir="$SIM_QUERY_ROOT/$object_id"
    save_dir="$OUTPUT_ROOT/object_${object_id}_query_${query_idx}"
    checkpoint="$OUTPUT_ROOT/checkpoints/object_${object_id}_quality.pth"

    if [[ ! -d "$ref_dir" || ! -d "$query_dir" ]]; then
        echo "Missing Dataset directories for object $object_id" >&2
        exit 1
    fi

    echo "[ObjectFolder example] object=$object_id query=$query_idx"
    bash "$SCRIPT_DIR/run_sim.sh" \
        --ref_dir "$ref_dir" \
        --query_dir "$query_dir" \
        --save_dir "$save_dir" \
        --query_indices "$query_idx" \
        --checkpoint "$checkpoint" \
        --train_if_missing \
        --allow_index_coordinate_fallback \
        --levels "${OBJECTFOLDER_EXAMPLE_LEVELS:-6}" \
        --network_depth "${OBJECTFOLDER_EXAMPLE_DEPTH:-6}" \
        --network_width "${OBJECTFOLDER_EXAMPLE_WIDTH:-128}" \
        --epochs "${OBJECTFOLDER_EXAMPLE_EPOCHS:-20}" \
        --samples_per_touch "${OBJECTFOLDER_EXAMPLE_SAMPLES:-2048}" \
        --batch_size "${OBJECTFOLDER_EXAMPLE_BATCH:-4096}" \
        --inr_height "${OBJECTFOLDER_EXAMPLE_INR_HEIGHT:-60}" \
        --inr_width "${OBJECTFOLDER_EXAMPLE_INR_WIDTH:-80}" \
        --debug_images \
        --skip_eval
done

echo "Examples saved under: $OUTPUT_ROOT"
