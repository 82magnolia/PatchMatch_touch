#!/usr/bin/env bash
# Generate query contact videos for every object in Taxim/data/ObjectFolder.
# Inputs:  Taxim/results/object_folder_touch_query/{idx}/picked_points_query.ply
# Outputs: Taxim/results/gen_contact_full_query/{idx}/
#
# Usage: bash train_refine_scripts/gen_contact_query.sh
#   from the PatchMatch_gpu project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OBJ_DIR="$PROJECT_ROOT/Taxim/data/ObjectFolder"
QUERY_BASE="$PROJECT_ROOT/Taxim/results/object_folder_touch_query"
OUT_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_query"
GEN_SCRIPT="$PROJECT_ROOT/Taxim/OpticalSimulation/gen_contact_video.py"

cd "$PROJECT_ROOT/Taxim/OpticalSimulation"

total=0
for obj_dir in "$OBJ_DIR"/*/; do
    idx=$(basename "$obj_dir")
    if [ -f "$obj_dir/model.obj" ] && [ -f "$QUERY_BASE/$idx/picked_points_query.ply" ]; then
        total=$((total + 1))
    fi
done
echo "Total objects to process: $total"

done_count=0
for obj_dir in "$OBJ_DIR"/*/; do
    idx=$(basename "$obj_dir")
    obj_path="$obj_dir/model.obj"
    contact_ply="$QUERY_BASE/$idx/picked_points_query.ply"
    save_dir="$OUT_BASE/$idx"

    if [ ! -f "$obj_path" ]; then
        echo "[SKIP] $idx — model.obj not found at $obj_path"
        continue
    fi

    if [ ! -f "$contact_ply" ]; then
        echo "[SKIP] $idx — query PLY not found at $contact_ply"
        continue
    fi

    mkdir -p "$save_dir"

    done_count=$((done_count + 1))
    echo "[${idx}] ($done_count/$total) Generating query contact video → $save_dir"
    python "$GEN_SCRIPT" \
        --obj_path "$obj_path" \
        --contact_ply "$contact_ply" \
        --mode back_forth_press \
        --depth_range_info 0. 10. 50 \
        --rand_contact_theta \
        --save_dir "$save_dir" \
        --obj_scale_factor 100.0
done

echo "Done."
