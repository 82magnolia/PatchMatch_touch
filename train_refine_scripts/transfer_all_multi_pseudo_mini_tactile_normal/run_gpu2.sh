#!/usr/bin/env bash
bash "$(dirname "${BASH_SOURCE[0]}")/_run.sh" 2 "${1:?Usage: run_gpu2.sh <GPU_ID> [MATCHER] [NUM_WORKERS]}" "${2:-disk_lightglue}" "${3:-6}"
