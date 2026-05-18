#!/usr/bin/env bash
# Sample query contact points near the FPS reference points for every object.
# Inputs:  Taxim/results/object_folder_touch/{idx}/picked_points_fps.ply
# Outputs: Taxim/results/object_folder_touch_query/{idx}/picked_points_query.ply
#
# Usage: bash train_refine_scripts/sample_query_contact_points.sh
#   from the PatchMatch_gpu project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OBJ_DIR="$PROJECT_ROOT/Taxim/data/ObjectFolder"
REF_BASE="$PROJECT_ROOT/Taxim/results/object_folder_touch"
OUT_BASE="$PROJECT_ROOT/Taxim/results/object_folder_touch_query"
PICKER="$PROJECT_ROOT/Taxim/OpticalSimulation/pick_contact_points_from_ref.py"

for obj_dir in "$OBJ_DIR"/*/; do
    idx=$(basename "$obj_dir")
    obj_path="$obj_dir/model.obj"
    ref_ply="$REF_BASE/$idx/picked_points_fps.ply"
    out_ply="$OUT_BASE/$idx/picked_points_query.ply"

    if [ ! -f "$obj_path" ]; then
        echo "[SKIP] $idx — model.obj not found at $obj_path"
        continue
    fi

    if [ ! -f "$ref_ply" ]; then
        echo "[SKIP] $idx — reference PLY not found at $ref_ply"
        continue
    fi

    mkdir -p "$(dirname "$out_ply")"

    echo "[${idx}] Sampling query points → $out_ply"
    python "$PICKER" \
        --obj_path "$obj_path" \
        --prepicked_ply "$ref_ply" \
        --output_ply "$out_ply"
done

echo "Done."
