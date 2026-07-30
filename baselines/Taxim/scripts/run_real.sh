#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${BASELINE_DIR}/../.." && pwd)"
REAL_DATA_ROOT="${TAXIM_REAL_DATA_ROOT:-${REPO_DIR}/log/real_data_gt_retrieval}"
REAL_OBJECT_ID="${TAXIM_REAL_OBJECT_ID:-1}"
REAL_DIR="${REAL_DATA_ROOT}/${REAL_OBJECT_ID}"
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
  --ref_dir "${REAL_DIR}" \
  --query_dir "${REAL_DIR}" \
  --save_dir "${REPO_DIR}/log/baselines/taxim/real_object${REAL_OBJECT_ID}" \
  --scale 1 \
  --video_type shadow \
  --retrieval_mode real_gt_retrieval \
  --data_mode real \
  --real_geometry_mode full_pose \
  --sensor_offset_file "${REPO_DIR}/log/gelsight_sensor_offset.json" \
  "$@"
