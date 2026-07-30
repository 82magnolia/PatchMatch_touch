#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${BASELINE_DIR}/../.." && pwd)"
TAXIM_MPL_CACHE="${TMPDIR:-/tmp}/taxim-matplotlib-${USER:-user}"
mkdir -p "${TAXIM_MPL_CACHE}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TAXIM_MPL_CACHE}}"

if [[ -n "${TAXIM_PYTHON:-}" ]]; then
  PYTHON_COMMAND=("${TAXIM_PYTHON}")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_COMMAND=(conda run --no-capture-output -n Taxim python)
else
  echo "Conda was not found. Activate Taxim or set TAXIM_PYTHON." >&2
  exit 1
fi

exec "${PYTHON_COMMAND[@]}" "${BASELINE_DIR}/run_baseline.py" \
  --ref_dir "${TAXIM_SIM_REF_ROOT:-${REPO_DIR}/Taxim/results/gen_contact_full_pseudo_mini}/1" \
  --query_dir "${TAXIM_SIM_QUERY_ROOT:-${REPO_DIR}/Taxim/results/gen_contact_full_query_pseudo_mini}/1" \
  --save_dir "${REPO_DIR}/log/baselines/taxim/sim_object1" \
  --scale 100 \
  --video_type shadow \
  --retrieval_mode sim_gt_retrieval \
  --data_mode sim \
  "$@"
