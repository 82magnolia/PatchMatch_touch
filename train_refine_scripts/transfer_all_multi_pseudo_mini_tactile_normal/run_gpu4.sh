#!/usr/bin/env bash
bash "$(dirname "${BASH_SOURCE[0]}")/_run.sh" 4 "${1:?Usage: run_gpu4.sh <GPU_ID>}"
