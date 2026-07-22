#!/usr/bin/env bash
bash "$(dirname "${BASH_SOURCE[0]}")/_run.sh" 5 "${1:?Usage: run_gpu5.sh <GPU_ID>}"
