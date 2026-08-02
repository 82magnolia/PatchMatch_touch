#!/usr/bin/env bash
# Job 1 baselines on the ground-truth-retrieval benchmark, 50 eval objects
# (951-1000). Identity pairing (sim_gt_retrieval) matches the benchmark's
# query/reference construction.
#
#   quilting  Tactile Normal Quilting  (Tactile DreamFusion-style, fallback mode)
#   inr       ObjectFolder TouchNet implicit neural representation, trained
#             per-object on the reference touches and queried at query locations
#   tarf      Tactile-Augmented Radiance Fields img2touch latent diffusion:
#             best-of-8 DDIM samples reranked by the two encoder checkpoints.
#             Predicts ONE image per query, tiled to the reference video length.
#             Three trained diffusion checkpoints are available; each writes to
#             its own output directory so the earlier runs stay intact:
#               tarf     log/tarf_pretrained.ckpt     epoch 5 of the run that
#                        finetunes from the released TaRF weights
#               tarf_v2  log/tarf_pretrained_v2.ckpt  epoch 29, trained without
#                        importing the released diffusion/conditioning weights
#               tarf_v3  log/tarf_pretrained_v3.ckpt  epoch 29 of the same
#                        finetune-from-released run that produced `tarf`
#
# Usage: bash run_baselines.sh <method> <worker_id> <gpu_id> <num_workers>
set -o pipefail

METHOD="${1:?Usage: $0 <quilting|inr|tarf|tarf_v2|tarf_v3> <worker_id> <gpu_id> <num_workers>}"
WORKER_ID="${2:?worker id}"
GPU_ID="${3:?gpu id}"
NUM_WORKERS="${4:-5}"

ROOT=/data1/junhokim/Projects/PatchMatch_touch
PY=/home/junhokim/miniconda3/envs/pm_touch/bin/python
export RQT_PYTHON="$PY"
export OBJECTFOLDER_PYTHON="$PY"
export TARF_PYTHON=/home/junhokim/miniconda3/envs/TaRF/bin/python
case "$METHOD" in
    tarf)    TARF_DIFF=$ROOT/log/tarf_pretrained.ckpt ;;
    tarf_v2) TARF_DIFF=$ROOT/log/tarf_pretrained_v2.ckpt ;;
    tarf_v3) TARF_DIFF=$ROOT/log/tarf_pretrained_v3.ckpt ;;
    *)       TARF_DIFF= ;;
esac
TARF_RGB=$ROOT/log/reranking_rgb_enc.ckpt
TARF_TAC=$ROOT/log/reranking_tac_enc.ckpt
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Cap per-process threads: NUM_WORKERS copies share the same cores.
export NUM_THREADS="${NUM_THREADS:-6}"
export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"
export NUMEXPR_NUM_THREADS="$NUM_THREADS"

OUT_BASE="$ROOT/log/paper_job1_baselines/$METHOD"
REF_PTS="$ROOT/Taxim/results/object_folder_touch"
QUERY_PTS="$ROOT/Taxim/results/object_folder_touch_query"
mkdir -p "$OUT_BASE"

pos=0
for OBJ in $(seq 951 1000); do
    if [ $((pos % NUM_WORKERS)) -ne "$WORKER_ID" ]; then pos=$((pos + 1)); continue; fi
    pos=$((pos + 1))

    SAVE="$OUT_BASE/$OBJ"
    if [ -f "$SAVE/transfer/metrics.pkl" ]; then
        echo "[$METHOD w$WORKER_ID] obj $OBJ already done, skipping"
        continue
    fi
    mkdir -p "$SAVE"
    echo "[$METHOD w$WORKER_ID gpu$GPU_ID] $(date +%H:%M:%S) obj $OBJ"

    if [ -n "$TARF_DIFF" ]; then
        TACTILE_NORMAL_OBJECT_ID=$OBJ bash "$ROOT/baselines/TaRF/scripts/run_sim_tactile_normal.sh" \
            --save_dir "$SAVE" \
            --diffusion_ckpt "$TARF_DIFF" \
            --ranking_rgb_enc_ckpt "$TARF_RGB" \
            --ranking_tac_enc_ckpt "$TARF_TAC" \
            > "$SAVE/run.log" 2>&1 \
            || echo "[$METHOD w$WORKER_ID] obj $OBJ FAILED (see $SAVE/run.log)"
    elif [ "$METHOD" = "quilting" ]; then
        TACTILE_NORMAL_OBJECT_ID=$OBJ bash "$ROOT/baselines/RandomQuiltingTactile/scripts/run_sim_tactile_normal.sh" \
            --save_dir "$SAVE" --object_id "$OBJ" \
            > "$SAVE/run.log" 2>&1 \
            || echo "[$METHOD w$WORKER_ID] obj $OBJ FAILED (see $SAVE/run.log)"
    else
        # Phase 1: fit the INR on the REFERENCE touches (fps contact points).
        TACTILE_NORMAL_OBJECT_ID=$OBJ bash "$ROOT/baselines/objectfolder_inr/scripts/run_sim_tactile_normal.sh" \
            --save_dir "$SAVE" --checkpoint "$SAVE/touchnet.pth" \
            --contact_points "$REF_PTS/$OBJ/picked_points_fps.npy" \
            --train_only --train_if_missing \
            > "$SAVE/train.log" 2>&1 \
            || { echo "[$METHOD w$WORKER_ID] obj $OBJ TRAIN FAILED"; continue; }
        # Phase 2: evaluate at the QUERY contact points (checkpoint already exists,
        # so the runner skips straight to inference).
        TACTILE_NORMAL_OBJECT_ID=$OBJ bash "$ROOT/baselines/objectfolder_inr/scripts/run_sim_tactile_normal.sh" \
            --save_dir "$SAVE" --checkpoint "$SAVE/touchnet.pth" \
            --contact_points "$QUERY_PTS/$OBJ/picked_points_query.npy" \
            > "$SAVE/run.log" 2>&1 \
            || echo "[$METHOD w$WORKER_ID] obj $OBJ INFER FAILED (see $SAVE/run.log)"
    fi
done

echo "[$METHOD w$WORKER_ID] done."
