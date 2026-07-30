#!/usr/bin/env bash
# Drive the remaining film_scale sweep runs two at a time (one per GPU).
# Round 1 (nonresidual scales 1 and 2) is launched separately; this covers the
# rest. Each pair runs concurrently and the script blocks until both finish
# before starting the next pair, so the two GPUs are never oversubscribed.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_pair() {
    local a_variant=$1 a_scale=$2 b_variant=$3 b_scale=$4
    echo "=== launching ${a_variant}/scale${a_scale} on GPU0 and ${b_variant}/scale${b_scale} on GPU1"
    bash "$SCRIPT_DIR/sweep_film_scale.sh" 0 "$a_variant" "$a_scale" > /tmp/sweep_${a_variant}_${a_scale}.log 2>&1 &
    local pa=$!
    bash "$SCRIPT_DIR/sweep_film_scale.sh" 1 "$b_variant" "$b_scale" > /tmp/sweep_${b_variant}_${b_scale}.log 2>&1 &
    local pb=$!
    wait $pa; local ra=$?
    wait $pb; local rb=$?
    echo "=== done ${a_variant}/scale${a_scale} rc=$ra   ${b_variant}/scale${b_scale} rc=$rb"
}

run_pair nonresidual 4 nonresidual 8
run_pair residual    1 residual    2
run_pair residual    4 residual    8
echo "ALL SWEEP RUNS COMPLETE"
