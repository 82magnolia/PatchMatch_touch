#!/usr/bin/env bash
# Run every trained TaRF diffusion checkpoint over both benchmarks.
#
# Methods (each writes to its own log/paper_job{1,2}_baselines/<method>/ tree):
#   tarf     log/tarf_pretrained.ckpt     epoch 5, finetuned from released TaRF
#   tarf_v2  log/tarf_pretrained_v2.ckpt  epoch 29, released weights NOT imported
#   tarf_v3  log/tarf_pretrained_v3.ckpt  epoch 29 of the same run as `tarf`
#
# All three are run with the float32 conditioning-encoder fix in
# baselines/TaRF/patchmatch_tarf/generator.py, so the rows are comparable.
#
# One worker per GPU, 8 GPUs; the benchmarks run one after another so each
# sweep gets the whole machine.
set -o pipefail
ROOT=/data1/junhokim/Projects/PatchMatch_touch
NGPU=8

for JOB in 1 2; do
    case "$JOB" in
        1) DIR=$ROOT/paper_experiments/job1_gt_retrieval ;;
        2) DIR=$ROOT/paper_experiments/job2_full_pipeline ;;
    esac
    for METHOD in tarf tarf_v2 tarf_v3; do
        echo "=== job$JOB / $METHOD  $(date +%H:%M:%S) ==="
        for W in $(seq 0 $((NGPU - 1))); do
            bash "$DIR/run_baselines.sh" "$METHOD" "$W" "$W" "$NGPU" \
                > "$DIR/logs/${METHOD}_w${W}.log" 2>&1 &
        done
        wait
        echo "=== job$JOB / $METHOD done $(date +%H:%M:%S) ==="
    done
done
echo "ALL TARF SWEEPS DONE $(date +%H:%M:%S)"
