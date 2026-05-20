#!/usr/bin/env bash
# Run retrieve_touch.py (tsv mode) for every object in gen_contact_full /
# gen_contact_full_query.  Requires log/identity_mapping.tsv to exist; run
# python gen_identity_tsv.py first if it does not.
#
# Per-object outputs (results.pkl) go to log/touch_retrieval/{idx}/
#
# Usage: bash train_refine_scripts/retrieve_touch_all.sh
#   from the PatchMatch_touch project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REF_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full"
QUERY_BASE="$PROJECT_ROOT/Taxim/results/gen_contact_full_query"
TSV="$PROJECT_ROOT/log/identity_mapping.tsv"
OUT_BASE="$PROJECT_ROOT/log/touch_retrieval"
RETRIEVE_SCRIPT="$PROJECT_ROOT/retrieve_touch.py"

SCALE=100
MODALITY=normal
TOUCHES_PER_OBJ=8

if [ ! -f "$TSV" ]; then
    echo "ERROR: identity-mapping TSV not found at $TSV"
    echo "       Run:  python gen_identity_tsv.py"
    exit 1
fi

# Count objects present in both ref and query
total_obj=0
for ref_dir in "$REF_BASE"/*/; do
    idx=$(basename "$ref_dir")
    [ -d "$QUERY_BASE/$idx" ] && total_obj=$((total_obj + 1))
done
total_touch=$((total_obj * TOUCHES_PER_OBJ))
echo "Objects to process: $total_obj  |  Total touch locations: $total_touch"
echo ""

done_obj=0
done_touch=0
for ref_dir in "$REF_BASE"/*/; do
    idx=$(basename "$ref_dir")
    query_dir="$QUERY_BASE/$idx"

    if [ ! -d "$query_dir" ]; then
        echo "[SKIP] $idx — missing from query dir"
        continue
    fi

    save_dir="$OUT_BASE/$idx"
    mkdir -p "$save_dir"

    done_obj=$((done_obj + 1))
    echo "[obj ${idx}] ($done_obj/$total_obj) | touch locations: $done_touch/$total_touch"

    python "$RETRIEVE_SCRIPT" \
        --ref_dir   "$ref_dir" \
        --query_dir "$query_dir" \
        --modality  "$MODALITY" \
        --scale     "$SCALE" \
        --retrieval_mode tsv \
        --tsv       "$TSV" \
        --save_dir  "$save_dir" \
        --no_figures

    done_touch=$((done_touch + TOUCHES_PER_OBJ))
    echo "  → done  ($done_touch/$total_touch touch locations retrieved)"
done

echo ""
echo "Done. $done_obj objects processed, $done_touch touch locations retrieved."
echo "Results in: $OUT_BASE"
