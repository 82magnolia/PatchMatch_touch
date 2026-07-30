#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${OBJECTFOLDER_PYTHON:-}" ]]; then
    exec "$OBJECTFOLDER_PYTHON" "$BASELINE_ROOT/train.py" \
        --train_only --train_if_missing --device cuda "$@"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found. Activate ObjectFolder or set OBJECTFOLDER_PYTHON." >&2
    exit 1
fi
exec conda run --no-capture-output -n ObjectFolder \
    python "$BASELINE_ROOT/train.py" --train_only --train_if_missing \
    --device cuda "$@"
