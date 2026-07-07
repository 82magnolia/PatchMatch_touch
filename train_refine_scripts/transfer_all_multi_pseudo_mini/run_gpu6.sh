#!/usr/bin/env bash
bash "$(dirname "${BASH_SOURCE[0]}")/_run.sh" "${1:?Usage: run_gpu6.sh <GPU_ID>}"
