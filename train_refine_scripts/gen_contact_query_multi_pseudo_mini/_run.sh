#!/usr/bin/env bash
# Worker: generates query contact videos (240×320, gelsight_pseudo_mini calibration)
# for 1/6 of all objects.
# Usage: bash _run.sh <GPU_ID (0-5)>

set -euo pipefail

GPU_ID="${1:?Usage: $0 <GPU_ID (0-5)>}"
NUM_GPUS=6

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYOPENGL_PLATFORM=egl
# EGL device ordering differs from CUDA PCI bus ordering on this server.
# Mapping discovered empirically: EGL {0..7} → physical GPU {3,2,1,0,7,6,5,4}.
# Reverse map so run_gpuN.sh uses physical GPU N. Only GPU_ID 0-5 are used here.
_EGL_MAP=(3 2 1 0 7 6 5 4)
export EGL_DEVICE_ID=${_EGL_MAP[$GPU_ID]}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OBJ_DIR="$PROJECT_ROOT/Taxim/data/ObjectFolder"
QUERY_BASE="$PROJECT_ROOT/Taxim/results/object_folder_touch_query"
OUT_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_query_pseudo_mini"
GEN_SCRIPT="$PROJECT_ROOT/Taxim/OpticalSimulation/gen_contact_video.py"
CALIB_DIR="$PROJECT_ROOT/Taxim/calibs/gelsight_pseudo_mini"

cd "$PROJECT_ROOT/Taxim/OpticalSimulation"

# Count eligible objects assigned to this GPU
total=0
pos=0
for obj_dir in "$OBJ_DIR"/*/; do
    idx=$(basename "$obj_dir")
    if [ ! -f "$obj_dir/model.obj" ] || [ ! -f "$QUERY_BASE/$idx/picked_points_query.ply" ]; then
        continue
    fi
    if [ $((pos % NUM_GPUS)) -eq "$GPU_ID" ]; then
        total=$((total + 1))
    fi
    pos=$((pos + 1))
done
echo "[GPU $GPU_ID] Objects to process: $total"

done_count=0
pos=0
for obj_dir in "$OBJ_DIR"/*/; do
    idx=$(basename "$obj_dir")
    obj_path="$obj_dir/model.obj"
    contact_ply="$QUERY_BASE/$idx/picked_points_query.ply"
    save_dir="$OUT_BASE/$idx"

    if [ ! -f "$obj_path" ] || [ ! -f "$contact_ply" ]; then
        continue
    fi

    if [ $((pos % NUM_GPUS)) -ne "$GPU_ID" ]; then
        pos=$((pos + 1))
        continue
    fi
    pos=$((pos + 1))

    mkdir -p "$save_dir"
    done_count=$((done_count + 1))
    echo "[${idx}] ($done_count/$total) [GPU $GPU_ID] Generating query contact video → $save_dir"
    python "$GEN_SCRIPT" \
        --obj_path "$obj_path" \
        --contact_ply "$contact_ply" \
        --mode back_forth_press \
        --depth_range_info 0. 10. 50 \
        --rand_contact_theta \
        --rand_contact_theta_mag 0.26179938779 \
        --save_dir "$save_dir" \
        --obj_scale_factor 100.0 \
        --override_hw 240 320 \
        --data_folder "$CALIB_DIR"
done

echo "[GPU $GPU_ID] Done."
