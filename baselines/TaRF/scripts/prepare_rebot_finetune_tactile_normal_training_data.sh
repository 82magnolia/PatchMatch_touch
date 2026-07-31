#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARF_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${TARF_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"
exec conda run --no-capture-output -n TaRF python \
  baselines/TaRF/scripts/prepare_sim_training_data.py \
  --roots \
    Taxim/results/gen_contact_full_tactile_normal_pseudo_mini \
    Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini \
  --output \
    log/baselines/tarf_training/patchmatch_sim_tactile_normal_rebot_finetune \
  --target-video-type tactile_normal \
  --split-mode rebot_finetune \
  --workers "${TARF_DATA_WORKERS:-12}" \
  "$@"
