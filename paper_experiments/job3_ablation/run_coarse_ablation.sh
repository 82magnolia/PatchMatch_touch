#!/usr/bin/env bash
# Job 3, part A/B: coarse-alignment ablations on a 20-object subset of the
# full-pipeline benchmark.
#
#   A. modality used for retrieval AND feature matching, scale fixed at 4x
#      the tactile sensor (obj_scale_factor 25):  normal | color | curvature | height
#   B. scale used for retrieval AND feature matching, modality fixed:
#      1x (scale 100) | 2x (scale 50) | 4x (scale 25)
#
# Taxim renders name scales by object-scale factor, so a SMALLER factor covers a
# LARGER physical footprint: 100 -> 1x sensor, 50 -> 2x, 25 -> 4x.
#
# Usage: bash run_coarse_ablation.sh <arm> <gpu_id>
#   arm = mod_normal | mod_color | mod_curvature | mod_height | scale_1x | scale_2x
#   (curvature@4x is the default configuration and doubles as scale_4x)
set -o pipefail

ARM="${1:?Usage: $0 <arm> <gpu_id>}"
GPU_ID="${2:?gpu id}"

ROOT=/data1/junhokim/Projects/PatchMatch_touch
PY=/home/junhokim/miniconda3/envs/pm_touch/bin/python
BENCH=$ROOT/log/paper_job2_bench
OUT=$ROOT/log/paper_job3_ablation/$ARM
DINO=$ROOT/dinov3/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
SUBSET=$ROOT/paper_experiments/job3_ablation/subset_objects.txt

# The method's default coarse alignment is surface normals at 4x the sensor
# footprint, so the scale sweep holds modality at normal and only varies scale.
# scale_4x is then the same configuration as mod_normal by construction.
DEFAULT_MODALITY=normal

case "$ARM" in
    mod_normal)     MODALITY=normal;    MATCH_SCALE=25 ;;
    mod_color)      MODALITY=color;     MATCH_SCALE=25 ;;
    mod_curvature)  MODALITY=curvature; MATCH_SCALE=25 ;;
    mod_height)     MODALITY=height;    MATCH_SCALE=25 ;;
    scale_1x)       MODALITY=$DEFAULT_MODALITY; MATCH_SCALE=100 ;;
    scale_2x)       MODALITY=$DEFAULT_MODALITY; MATCH_SCALE=50  ;;
    scale_4x)       MODALITY=$DEFAULT_MODALITY; MATCH_SCALE=25  ;;
    *) echo "Unknown arm '$ARM'" >&2; exit 1 ;;
esac

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NUM_THREADS="${NUM_THREADS:-7}"
export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"
export NUMEXPR_NUM_THREADS="$NUM_THREADS"

mkdir -p "$OUT"
echo "[abl $ARM] modality=$MODALITY match_scale=$MATCH_SCALE gpu=$GPU_ID"

while read -r OBJ; do
    [ -z "$OBJ" ] && continue
    SAVE="$OUT/$OBJ"
    if [ -f "$SAVE/transfer/metrics.pkl" ]; then
        echo "[abl $ARM] obj $OBJ done, skipping"; continue
    fi
    mkdir -p "$SAVE"
    echo "[abl $ARM] $(date +%H:%M:%S) obj $OBJ"

    "$PY" "$ROOT/transfer_pipeline.py" \
        --ref_dir   "$BENCH/$OBJ/ref" \
        --query_dir "$BENCH/$OBJ/query" \
        --save_dir  "$SAVE" \
        --scale 100 \
        --match_scale "$MATCH_SCALE" --match_scale_convention obj_scale_factor \
        --retrieval_mode dinov3 --retrieval_modality "$MODALITY" \
        --dino_weights "$DINO" \
        --transfer_backend dinov3_feat_match \
        --transfer_modality "$MODALITY" \
        --transfer_matcher superpoint_superglue \
        --transfer_offset_matcher superpoint_superglue \
        --transfer_offset_method median \
        --video_type tactile_normal \
        --skip_refine --skip_viz \
        > "$SAVE/pipeline.log" 2>&1 \
        || echo "[abl $ARM] obj $OBJ FAILED (see $SAVE/pipeline.log)"
done < "$SUBSET"

echo "[abl $ARM] done."
