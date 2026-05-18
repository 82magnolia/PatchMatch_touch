#!/usr/bin/env bash
# Sample FPS contact points for every object in data/ObjectFolder_touch.
# Outputs: Taxim/results/object_folder_touch/{idx}/picked_points_fps.ply
#
# Usage: bash train_refine_scripts/sample_contact_points.sh
#   from the PatchMatch_gpu project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OBJ_DIR="$PROJECT_ROOT/Taxim/data/ObjectFolder"
OUT_BASE="$PROJECT_ROOT/Taxim/results/object_folder_touch"
PICKER="$PROJECT_ROOT/Taxim/OpticalSimulation/pick_contact_points.py"
NUM_FPS_POINTS=8

for obj_dir in "$OBJ_DIR"/*/; do
    idx=$(basename "$obj_dir")
    obj_path="$obj_dir/model.obj"
    out_ply="$OUT_BASE/$idx/picked_points_fps.ply"

    if [ ! -f "$obj_path" ]; then
        echo "[SKIP] $idx — model.obj not found at $obj_path"
        continue
    fi

    mkdir -p "$(dirname "$out_ply")"

    echo "[${idx}] Sampling $NUM_FPS_POINTS FPS points → $out_ply"
    python "$PICKER" \
        --obj_path "$obj_path" \
        --output_ply "$out_ply" \
        --num_fps_points "$NUM_FPS_POINTS"
done

echo "Done."
