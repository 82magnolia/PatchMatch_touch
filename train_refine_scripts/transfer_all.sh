#!/usr/bin/env bash
# Run main_retrieval_transfer_accel.py for every object in gen_contact_full /
# gen_contact_full_query that has a retrieval PKL in log/touch_retrieval/{idx}/.
# Per-object transferred videos go to log/transfer/{idx}/.
#
# Usage: bash train_refine_scripts/transfer_all.sh
#   from the PatchMatch_touch project root.

set -euo pipefail

# NVCC 12.4 (conda) generates cubins that require a CUDA 12.x driver, but the
# installed driver only supports CUDA 11.4.  Use CUDA 11.8 nvcc instead — within
# CUDA 11.x all minor versions are ABI-compatible with each other.
export PATH="/usr/local/cuda-11.8/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REF_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full"
QUERY_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_query"
RETRIEVAL_BASE="$PROJECT_ROOT/log/touch_retrieval"
OUT_BASE="$PROJECT_ROOT/log/transfer"
TRANSFER_SCRIPT="$PROJECT_ROOT/main_retrieval_transfer_accel.py"

TOUCHES_PER_OBJ=8

# Count eligible objects
total_obj=0
for ref_dir in "$REF_BASE"/*/; do
    idx=$(basename "$ref_dir")
    if [ -d "$QUERY_BASE/$idx" ] && [ -f "$RETRIEVAL_BASE/$idx/results.pkl" ]; then
        total_obj=$((total_obj + 1))
    fi
done
total_touch=$((total_obj * TOUCHES_PER_OBJ))
echo "Objects to process: $total_obj  |  Total touch locations: $total_touch"
echo ""

done_obj=0
done_touch=0
for ref_dir in "$REF_BASE"/*/; do
    idx=$(basename "$ref_dir")
    query_dir="$QUERY_BASE/$idx"
    retrieval_pkl="$RETRIEVAL_BASE/$idx/results.pkl"

    if [ ! -d "$query_dir" ]; then
        echo "[SKIP] $idx — missing from query dir"
        continue
    fi
    if [ ! -f "$retrieval_pkl" ]; then
        echo "[SKIP] $idx — retrieval PKL not found at $retrieval_pkl"
        continue
    fi

    save_dir="$OUT_BASE/$idx"
    mkdir -p "$save_dir"

    done_obj=$((done_obj + 1))
    echo "[obj ${idx}] ($done_obj/$total_obj) | touch locations: $done_touch/$total_touch"

    python "$TRANSFER_SCRIPT" \
        --query_dir      "$query_dir" \
        --ref_dir        "$ref_dir" \
        --retrieval_pkl  "$retrieval_pkl" \
        --modality       raw_normal \
        --scale          100 \
        --video_type     shadow \
        --save_dir       "$save_dir" \
        --em \
        --use_ref_contact_mask \
        --use_ref_static_mask \
        --em_iters       10 \
        --use_mask \
        --eval \
        --use_accel \
        --em_iters_subseq 1 \
        --downsample_res 4 \
        --use_downsample_em \
        --use_keyframe \
        --no_nnf_figures

    done_touch=$((done_touch + TOUCHES_PER_OBJ))
    echo "  → done  ($done_touch/$total_touch touch locations transferred)"
done

echo ""
echo "Done. $done_obj objects processed, $done_touch touch locations transferred."
echo "Results in: $OUT_BASE"
