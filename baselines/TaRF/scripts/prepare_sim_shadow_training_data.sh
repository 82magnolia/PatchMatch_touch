#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARF_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${TARF_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"
exec conda run --no-capture-output -n TaRF python \
  baselines/TaRF/scripts/prepare_sim_training_data.py \
  --roots \
    Taxim/results/gen_contact_full_pseudo_mini \
    Taxim/results/gen_contact_full_query_pseudo_mini \
  --output log/baselines/tarf_training/patchmatch_sim_ref_even_query_odd \
  --target-video-type shadow \
  --workers "${TARF_DATA_WORKERS:-12}" \
  "$@"
