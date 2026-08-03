#!/usr/bin/env bash
# Coarse transfer (superpoint+superglue, curvature modality, 4x match scale) over
# the 50 ground-truth-retrieval eval objects (951-1000), tactile_normal domain.
set -uo pipefail
ROOT=/home/junhokim/Projects/PatchMatch_gpu
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
OUT=$ROOT/log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue
cd "$ROOT"
for idx in $(seq 951 1000); do
  q=$ROOT/Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini/$idx
  r=$ROOT/Taxim/results/gen_contact_full_tactile_normal_pseudo_mini/$idx
  p=$ROOT/log/touch_retrieval/$idx/results.pkl
  [ -d "$q" ] && [ -d "$r" ] && [ -f "$p" ] || { echo "SKIP $idx"; continue; }
  if [ -f "$OUT/$idx/metrics.pkl" ]; then echo "DONE-ALREADY $idx"; continue; fi
  mkdir -p "$OUT/$idx"
  echo "=== [$(date +%T)] object $idx ==="
  python main_retrieval_transfer_feat_match.py \
    --query_dir "$q" --ref_dir "$r" --retrieval_pkl "$p" \
    --modality curvature --video_type tactile_normal \
    --video_scale 100. --match_scale 25. --match_scale_convention obj_scale_factor \
    --matcher superpoint_superglue --offset_matcher superpoint_superglue \
    --offset_method median --save_dir "$OUT/$idx" --no_nnf_figures --eval \
    2>&1 | tail -4
done
echo "ALL DONE $(date)"
