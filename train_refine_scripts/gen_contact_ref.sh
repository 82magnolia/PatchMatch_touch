#!/usr/bin/env bash
# Generate reference contact videos for every object in Taxim/data/ObjectFolder.
# Inputs:  Taxim/results/object_folder_touch/{idx}/picked_points_fps.ply
# Outputs: Taxim/results/gen_contact_full/{idx}/
#
# Usage: bash train_refine_scripts/gen_contact_ref.sh
#   from the PatchMatch_gpu project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OBJ_DIR="$PROJECT_ROOT/Taxim/data/ObjectFolder"
REF_BASE="$PROJECT_ROOT/Taxim/results/object_folder_touch"
OUT_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full"
GEN_SCRIPT="$PROJECT_ROOT/Taxim/OpticalSimulation/gen_contact_video.py"

cd "$PROJECT_ROOT/Taxim/OpticalSimulation"

total=0
for obj_dir in "$OBJ_DIR"/*/; do
    idx=$(basename "$obj_dir")
    if [ -f "$obj_dir/model.obj" ] && [ -f "$REF_BASE/$idx/picked_points_fps.ply" ]; then
        total=$((total + 1))
    fi
done
echo "Total objects to process: $total"

done_count=0
for obj_dir in "$OBJ_DIR"/*/; do
    idx=$(basename "$obj_dir")
    obj_path="$obj_dir/model.obj"
    contact_ply="$REF_BASE/$idx/picked_points_fps.ply"
    save_dir="$OUT_BASE/$idx"

    if [ ! -f "$obj_path" ]; then
        echo "[SKIP] $idx — model.obj not found at $obj_path"
        continue
    fi

    if [ ! -f "$contact_ply" ]; then
        echo "[SKIP] $idx — reference PLY not found at $contact_ply"
        continue
    fi

    mkdir -p "$save_dir"

    done_count=$((done_count + 1))
    echo "[${idx}] ($done_count/$total) Generating reference contact video → $save_dir"
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
