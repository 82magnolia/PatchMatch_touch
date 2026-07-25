#!/usr/bin/env bash
bash "$(dirname "${BASH_SOURCE[0]}")/_run.sh" 0 "${1:?Usage: run_gpu0.sh <GPU_ID> [MATCHER] [NUM_WORKERS]}" "${2:-disk_lightglue}" "${3:-6}"
